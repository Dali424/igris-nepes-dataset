"""Parallel collector with synthetic children: concurrency bound, order independence,
quota/budget/checkpoint parity with the serial engine, resource gate, ownership,
STOP/DRAIN/signals, attempt-local and global errors, crash and resume.
Review fixes (exit records, hang-up, cleanup, DRAIN race, gate) are in test_parallel_recovery.py.

No Isaac is launched; the global lock lives in each fixture's own TMPDIR.
"""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from parallel_process_fixture import DEFAULT_CONTROL, ParallelFixture, read, running, sha

BUNDLE = Path(__file__).resolve().parents[1]
SEEDED = BUNDLE / 'scripts/seeded'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


parallel = load('nepes_test_collect_parallel', SEEDED / 'collect_parallel.py')
serial = parallel.helpers  # the unchanged serial engine, as the parallel engine loads it
campaign_sources = load('nepes_test_campaign_sources', BUNDLE / 'scripts/campaign_sources.py')
PROFILES = ('left2', 'right4')


def max_overlap(events):
    points = [(e['time'], 1) for e in events if e['kind'] == 'native_started']
    points += [(e['time'], -1) for e in events if e['kind'] in ('native_finished', 'native_interrupted')]
    current = best = 0
    for _, delta in sorted(points):
        current += delta; best = max(best, current)
    return best


def gate_kind(fixture):
    return fixture.process()['gate'].get('kind')


# A 48 GB synthetic GPU admits three warming Kits; DEFAULT_CONTROL mirrors the measured 16 GB RTX 5080.
LARGE_GPU = [dict(DEFAULT_CONTROL['gpus'][0], total=49152, used=4186, free=44966)]


class ParallelCollectorTests(unittest.TestCase):
    def fixture(self, name, control=None, **options):
        root = tempfile.TemporaryDirectory(prefix='nepes-parallel-'); self.addCleanup(root.cleanup)
        fixture = ParallelFixture(Path(root.name), name, control, **options)
        self.addCleanup(fixture.close)
        return fixture

    def sentinel(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
        self.addCleanup(lambda: process.poll() is None and (process.kill(), process.wait(timeout=5)))
        return process

    # ------------------------------------------------------------------ dispatch rule and plan parity
    def test_next_dispatch_equals_serial_next_profile_and_counts_in_flight(self):
        rng = random.Random(20261030)
        for _ in range(3000):
            quotas = {name: rng.randint(1, 4) for name in PROFILES}
            budgets = {name: quotas[name] + rng.randint(0, 4) for name in PROFILES}
            plan = ({'quota_each': quotas['left2'], 'max_attempts_each': budgets['left2']} if rng.random() < .3
                    else {'quota_by_profile': quotas, 'max_attempts_by_profile': budgets})
            attempts = [{'profile': rng.choice(PROFILES), 'status': rng.choice(('accepted', 'failed', 'interrupted', 'running', 'surplus'))}
                        for _ in range(rng.randint(0, 10))]
            state = {'plan': plan, 'attempts': attempts}
            self.assertEqual(parallel.next_dispatch(state), serial.next_profile(state))
            self.assertEqual(parallel.next_dispatch(state, {'left2': 0, 'right4': 0}, None), serial.next_profile(state))
            in_flight = {name: rng.randint(0, 2) for name in PROFILES}
            checkpoint = rng.choice((None, rng.randint(1, 6)))
            selected = parallel.next_dispatch(state, in_flight, checkpoint)
            totals = serial.counts(state)
            eligible = [name for _, name in serial.PROFILES
                        if totals[name] + in_flight[name] < serial.quota_for(plan, name)
                        and sum(a['profile'] == name for a in attempts) < serial.attempts_for(plan, name)]
            if checkpoint is not None and sum(totals.values()) + sum(in_flight.values()) >= checkpoint:
                self.assertIsNone(selected)
            elif not eligible:
                self.assertIsNone(selected)
            else:
                first = 'right4' if attempts and attempts[-1]['profile'] == 'left2' else 'left2'
                self.assertEqual(selected[1], first if first in eligible else eligible[0])

    def capture(self, function, argv):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(function(argv), 0)
        return json.loads(stream.getvalue())

    def test_options_and_dry_run_plan_match_serial_engine(self):
        base = ['--bundle', str(BUNDLE), '--output', '/nonexistent/batch', '--isaac-python', sys.executable,
                '--dataset-python', sys.executable]
        for extra in ([], ['--max-successes-per-pair', '7', '--first-seed', '9'],
                      ['--left-successes', '3', '--right-successes', '2', '--max-attempts-per-pair', '9',
                       '--stop-after-total-successes', '2', '--max-wall-seconds', '60', '--resume', '--dry-run']):
            old = vars(serial.arguments(base + extra))
            new = vars(parallel.arguments(base + extra + ['--parallel', '2']))
            self.assertEqual({key: new[key] for key in old}, old)
        defaults = vars(parallel.arguments(base))
        self.assertEqual({key: defaults[key] for key in ('min_free_vram_gib', 'kit_vram_gib', 'min_available_ram_gib', 'kit_ram_gib',
                                                         'disk_reserve_gib_per_attempt', 'launch_spacing_seconds', 'warmup_seconds',
                                                         'gpu_index', 'max_idle_wait_seconds', 'foreign_kit', 'parallel')},
                         {'min_free_vram_gib': 4.5, 'kit_vram_gib': 4.5, 'min_available_ram_gib': 10., 'kit_ram_gib': 6.5,
                          'disk_reserve_gib_per_attempt': 10., 'launch_spacing_seconds': 90., 'warmup_seconds': 300.,
                          'gpu_index': 0, 'max_idle_wait_seconds': None, 'foreign_kit': 'wait', 'parallel': None})
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(serial, 'verify_bundle', return_value='fixture_manifest'), \
                mock.patch.object(serial, 'upper_body_ik_plan', return_value={'schema': 'fixture_upper_body_ik'}), \
                mock.patch.object(parallel, 'query_gpu', side_effect=RuntimeError('no GPU in unit test')), \
                mock.patch.object(parallel, 'native_pids', return_value=[]):
            argv = ['--bundle', str(BUNDLE), '--output', str(Path(directory) / 'out'), '--isaac-python', sys.executable,
                    '--dataset-python', sys.executable, '--first-seed', '10000', '--left-successes', '3',
                    '--right-successes', '2', '--dry-run']
            old = self.capture(serial.main, argv)
            new = self.capture(parallel.main, argv + ['--parallel', '3'])
            for slots in (1, 4):
                with self.assertRaisesRegex(ValueError, '--parallel must be 2..3'):
                    parallel.main(argv + ['--parallel', str(slots)])
            with self.assertRaisesRegex(ValueError, 'finite and nonnegative'):
                parallel.main(argv + ['--parallel', '2', '--min-free-vram-gib', '-1'])
            self.assertFalse((Path(directory) / 'out').exists())
        self.assertEqual(new['examples'], old['examples'])  # identical native commands
        expected = dict(old['plan'], ordering='round_robin_parallel_v1', parallel=new['plan']['parallel'],
                        tools_sha256=dict(old['plan']['tools_sha256'], **{'collect_parallel.py': sha(SEEDED / 'collect_parallel.py')}))
        self.assertEqual(new['plan'], expected)
        self.assertEqual(list(new['plan']['tools_sha256']), ['collect_seeded.py', 'merge_seeded.py', 'seeded_config.py', 'collect_parallel.py'])
        self.assertEqual((new['plan']['parallel']['slots'], new['plan']['parallel']['engine']), (3, 'collect_parallel.py'))
        self.assertEqual(new['resource_policy']['native_floor_gib'], 20.)
        self.assertIn('gpu_error', new['resource_preview'])

    # ------------------------------------------------------------------ concurrency and determinism
    def test_runs_at_most_n_natives_and_honours_launch_spacing(self):
        for slots, spacing, quotas in ((2, 0., (3, 3)), (3, 0., (3, 3)), (2, .6, (1, 1))):
            with self.subTest(slots=slots, spacing=spacing):
                fixture = self.fixture(f'bound_{slots}_{spacing}', {'delay': {'*': 1. if spacing else .5}, 'gpus': LARGE_GPU},
                                       slots=slots, quotas=quotas, spacing=spacing)
                state = fixture.wait(fixture.start(), 0)
                self.assertEqual(state['status'], 'complete')
                starts = sorted(event['time'] for event in fixture.events('native_started'))
                self.assertEqual(len(starts), sum(quotas))
                self.assertEqual(max_overlap(fixture.events()), slots)  # own running Kits never block each other
                if spacing:
                    self.assertGreaterEqual(starts[1] - starts[0], spacing - .15)
                self.assertFalse([a for a in state['attempts'] if a['status'] not in ('accepted',)])

    def test_results_do_not_depend_on_completion_order_and_equal_serial(self):
        outcomes = {'2:0': 'fail', '2:2': 'fail', '4:1': 'fail'}
        early = {f'{pair}:{seed}': .1 + .2 * seed for pair in (2, 4) for seed in range(6)}
        late = {f'{pair}:{seed}': .1 + .2 * (5 - seed) + (.15 if pair == 4 else 0) for pair in (2, 4) for seed in range(6)}
        runs = {}
        for name, engine, slots, delays in (('serial', 'serial', 1, {}), ('early_first', 'parallel', 2, early),
                                            ('late_first', 'parallel', 3, late)):
            fixture = self.fixture(name, {'outcome': outcomes, 'delay': delays, 'gpus': LARGE_GPU}, engine=engine,
                                   slots=slots, quotas=(2, 2), budget=6)
            runs[name] = (fixture, fixture.wait(fixture.start(), 0))
        expected = {'left2': [(0, 0, 'failed'), (1, 1, 'accepted'), (2, 2, 'failed'), (3, 3, 'accepted')],
                    'right4': [(0, 0, 'accepted'), (1, 1, 'failed'), (2, 2, 'accepted')]}
        finished = {}
        for name, (fixture, state) in runs.items():
            with self.subTest(run=name):
                self.assertEqual(state['status'], 'complete')
                self.assertEqual({profile: [record[1:] for record in fixture.records(state, profile)] for profile in PROFILES}, expected)
                self.assertEqual(read(fixture.output / 'dataset_4/meta/info.json')['episodes'],
                                 [['left2', 1], ['left2', 3], ['right4', 0], ['right4', 2]])
                _, rows, quotas = campaign_sources.new_rows(fixture.output, fixture.bundle.resolve())
                self.assertEqual([(row['profile'], row['seed']) for row in rows],
                                 [('left2', 1), ('left2', 3), ('right4', 0), ('right4', 2)])
                finished[name] = [(event['pair'], event['seed']) for event in fixture.events('native_finished')]
        self.assertNotEqual(finished['early_first'], finished['late_first'])  # completion order really differed

    def test_measured_16_gb_gpu_admits_only_two_warming_kits_for_parallel_three(self):
        fixture = self.fixture('rtx5080', {'delay': {'*': .8}}, slots=3, quotas=(3, 3))
        state = fixture.wait(fixture.start(), 0)
        self.assertEqual((state['status'], max_overlap(fixture.events())), ('complete', 2))
        self.assertIn('Waiting before the next attempt: GPU memory', fixture.log())

    def test_in_flight_attempts_count_against_quota_without_surplus(self):
        fixture = self.fixture('quota_one', {'delay': {'*': .4}, 'gpus': LARGE_GPU}, slots=3, quotas=(1, 1))
        state = fixture.wait(fixture.start(), 0)
        self.assertEqual(fixture.records(state), [('left2', 0, 0, 'accepted'), ('right4', 0, 0, 'accepted')])
        self.assertEqual(len(fixture.events('native_started')), 2)
        rng = random.Random(7)
        delays = {f'{pair}:{seed}': rng.uniform(.05, .5) for pair in (2, 4) for seed in range(8)}
        fixture = self.fixture('quota_two', {'outcome': {'2:0': 'fail', '2:2': 'fail'}, 'delay': delays, 'gpus': LARGE_GPU},
                               slots=3, quotas=(2, 1), budget=8)
        state = fixture.wait(fixture.start(), 0)
        self.assertEqual([record[1:] for record in fixture.records(state, 'left2')],
                         [(0, 0, 'failed'), (1, 1, 'accepted'), (2, 2, 'failed'), (3, 3, 'accepted')])
        self.assertEqual(sorted((event['pair'], event['seed']) for event in fixture.events('native_started')),
                         [(2, 0), (2, 1), (2, 2), (2, 3), (4, 0)])
        self.assertEqual(state['counts'], {'left2': 2, 'right4': 1})

    def test_defensive_surplus_success_is_kept_for_audit_but_never_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / 'left2/attempt_000001_seed_000001'
            (attempt / 'profile').mkdir(parents=True)
            (attempt / 'profile/sample.json').write_text('{}')
            (attempt / 'verification.json').write_text(json.dumps({
                'profile': 'left2', 'pair': 2, 'seed': 1, 'xy_half_range_m': .003, 'config_sha256': 'c', 'config_file_sha256': 'f',
                'sampled_offset_xy_m': [0, 0], 'startup_reset': {'measured_pose_w': [0] * 7}, 'initial_measurement': {'poses': {}},
                'motion_exact_sha256': 'm', 'motion_quantized_sha256': 'q'}))
            plan = {'quota_each': 1, 'max_attempts_each': 3, 'xy_half_range_by_profile_m': {'left2': .003, 'right4': .005}}
            other = {'profile': 'left2', 'status': 'accepted'}
            record = {'profile': 'left2', 'pair': 2, 'seed': 1, 'index': 1, 'directory': str(attempt), 'status': 'running'}
            state = {'plan': plan, 'attempts': [other, record]}
            ctx = SimpleNamespace(dataset=sys.executable, bundle=Path(directory), plan=plan, state=state,
                                  accepted={'left2': [], 'right4': []},
                                  sampler=SimpleNamespace(effective_xy_range=lambda pair, seed, requested=None: .003),
                                  merger=SimpleNamespace(validate_record_revision=lambda *args: {},
                                                         reject_duplicates=lambda previous, current: None))

            def drive(revalidate=False):
                task = parallel.verify_task(ctx, record, revalidate)
                self.assertEqual(next(task).label, 'verify')
                with self.assertRaises(StopIteration):
                    task.send(0)

            drive()
            self.assertEqual(record['status'], 'surplus')
            self.assertIn('excluded from counts and merge', record['surplus_reason'])
            self.assertEqual(serial.counts(state), {'left2': 1, 'right4': 0})
            self.assertEqual(ctx.accepted['left2'], [])
            drive(revalidate=True)  # resume keeps surplus as surplus
            self.assertEqual(record['status'], 'surplus')
            self.assertEqual(serial.next_profile(state), (4, 'right4'))
            other['status'], record['status'] = 'failed', 'running'
            drive()  # below quota: an ordinary acceptance
            self.assertEqual((record['status'], serial.counts(state)['left2'], len(ctx.accepted['left2'])), ('accepted', 1, 1))

    def test_budget_exhaustion_matches_serial_semantics(self):
        fixture = self.fixture('budget', {'outcome': {f'2:{seed}': 'fail' for seed in range(10)}, 'delay': {'*': .05}})
        state = fixture.wait(fixture.start(), 1)
        self.assertEqual(state['status'], 'needs_review')
        self.assertIn('exhausted attempt budget', state['error'])
        self.assertEqual(state['counts'], {'left2': 0, 'right4': 1})
        self.assertEqual([record[3] for record in fixture.records(state, 'left2')], ['failed'] * 5)

    def test_checkpoint_counts_in_flight_and_resume_completes(self):
        fixture = self.fixture('checkpoint', {'delay': {'*': .3}}, quotas=(2, 2), extra=['--stop-after-total-successes', '2'])
        state = fixture.wait(fixture.start(), 0)
        self.assertEqual(state['status'], 'checkpoint')
        self.assertEqual(sum(state['counts'].values()), 2)
        self.assertEqual([record[3] for record in fixture.records(state)], ['accepted', 'accepted'])
        fixture.command = fixture.command[:-2]
        state = fixture.wait(fixture.start(resume=True), 0)
        self.assertEqual((state['status'], state['counts']), ('complete', {'left2': 2, 'right4': 2}))
        self.assertEqual(len(fixture.events('native_started')), 4)

    # ------------------------------------------------------------------ stop, drain, errors, crash
    def test_stop_interrupts_every_in_flight_attempt_and_resume_continues(self):
        foreign = self.sentinel()
        for mode, sig in (('file', None), ('sigint', signal.SIGINT), ('sigterm', signal.SIGTERM)):  # SIGHUP drains
            with self.subTest(mode=mode):
                fixture = self.fixture('stop_' + mode, {'block': ['2:0', '4:0'], 'grandchild': {'*': True}})
                child = fixture.start()
                fixture.wait_events('grandchild_started', 2)
                if sig is None:
                    (fixture.output / 'STOP').touch()
                else:
                    child.send_signal(sig)
                state = fixture.wait(child, 130)
                self.assertEqual((state['status'], state['stop_mode']), ('stopped', 'stop'))
                self.assertEqual(fixture.records(state), [('left2', 0, 0, 'interrupted'), ('right4', 0, 0, 'interrupted')])
                for event in fixture.events('native_started'):
                    self.assertFalse(running(event['pid']))
                for event in fixture.events('grandchild_started'):
                    self.assertFalse(running(event['child_pid']))
                self.assertEqual(fixture.synthetic_pids(), [])
                self.assertIsNone(foreign.poll())
                fixture.update_control(block=[])
                state = fixture.wait(fixture.start(resume=True), 0)
                self.assertEqual(fixture.records(state)[2:], [('left2', 1, 1, 'accepted'), ('right4', 1, 1, 'accepted')])
                self.assertEqual(len(list(fixture.output.glob('STOP_resumed_*'))), int(sig is None))

    def test_stop_while_one_slot_verifies_keeps_that_record_for_resume(self):
        fixture = self.fixture('stop_verify', {'verify_block': ['2:0'], 'block': ['4:0']})
        child = fixture.start()
        fixture.wait_for(lambda: [e for e in fixture.events('verify_started') if e['pair'] == 2], 'left verification')
        fixture.wait_for(lambda: [e for e in fixture.events('native_started') if e['pair'] == 4], 'right native')
        (fixture.output / 'STOP').touch()
        state = fixture.wait(child, 130)
        left, right = state['attempts']
        self.assertEqual((left['status'], left['returncode']), ('running', 0))
        self.assertEqual(right['status'], 'interrupted')
        fixture.update_control(verify_block=[], block=[])
        state = fixture.wait(fixture.start(resume=True), 0)
        self.assertEqual(fixture.records(state), [('left2', 0, 0, 'accepted'), ('right4', 0, 0, 'interrupted'),
                                                  ('right4', 1, 1, 'accepted')])
        self.assertEqual([(e['pair'], e['seed']) for e in fixture.events('native_started')], [(2, 0), (4, 0), (4, 1)])

    def test_drain_finishes_in_flight_attempts_and_starts_nothing_new(self):
        fixture = self.fixture('drain', {'block': ['2:0', '4:0']}, quotas=(2, 2))
        child = fixture.start()
        fixture.wait_events('native_started', 2)
        (fixture.output / 'DRAIN').touch()
        fixture.wait_for(lambda: fixture.state().get('draining'), 'drain recorded')
        fixture.update_control(block=[])
        state = fixture.wait(child, 130)
        self.assertEqual((state['status'], state['stop_mode']), ('stopped', 'drain'))
        self.assertEqual(fixture.records(state), [('left2', 0, 0, 'accepted'), ('right4', 0, 0, 'accepted')])
        self.assertEqual(len(fixture.events('native_started')), 2)
        state = fixture.wait(fixture.start(resume=True), 0)
        self.assertEqual((state['status'], state['counts']), ('complete', {'left2': 2, 'right4': 2}))
        self.assertNotIn('draining', state)
        self.assertEqual(len(list(fixture.output.glob('DRAIN_resumed_*'))), 1)

    def test_attempt_local_error_lets_other_slot_finish_then_blocks_resume(self):
        fixture = self.fixture('local_error', {'outcome': {'2:0': 'fail'}, 'classification_error': {'2:0': True},
                                               'block': ['4:0']}, quotas=(2, 2))
        child = fixture.start()
        fixture.wait_for(lambda: fixture.state().get('draining'), 'drain after the attempt-local error')
        self.assertEqual(fixture.state()['draining']['mode'], 'local_error')
        fixture.update_control(block=[])
        state = fixture.wait(child, 1)
        self.assertEqual(state['status'], 'needs_review')
        self.assertIn('Unclassified failure/crash/storage error', state['error'])
        left, right = state['attempts']
        self.assertEqual((left['status'], left['returncode'], right['status']), ('running', 2, 'accepted'))
        self.assertEqual(len(fixture.events('native_started')), 2)
        self.assertEqual(fixture.start(resume=True).wait(timeout=30), 1)
        self.assertIn('automatic resume is blocked', fixture.log())

    def test_global_error_interrupts_every_slot(self):
        fixture = self.fixture('global_error', {'block': ['2:0', '4:0'], 'grandchild': {'4:0': True}})
        child = fixture.start()
        fixture.wait_events('grandchild_started', 1)
        fixture.wait_events('native_started', 2)
        with (fixture.tools / 'seeded_config.py').open('a') as stream:
            stream.write('\n# changed while collecting\n')
        fixture.update_control(block=['4:0'])  # left finishes; its verifier launch finds the changed tool
        state = fixture.wait(child, 1)
        self.assertEqual(state['status'], 'needs_review')
        self.assertIn('Frozen batch tool changed', state['error'])
        left, right = state['attempts']
        self.assertEqual((left['status'], left['returncode']), ('running', 0))
        self.assertEqual(right['status'], 'interrupted')
        self.assertTrue(right['reason'].startswith('Orchestrator cleanup after ValueError: Frozen batch tool changed'))
        self.assertEqual(fixture.synthetic_pids(), [])

    def test_surviving_descendant_is_escalated_to_sigkill_before_the_slot_is_freed(self):
        fixture = self.fixture('stubborn', {'stubborn': {'2:0': True}})
        state = fixture.wait(fixture.start(), 0)
        self.assertEqual((state['status'], state['counts']), ('complete', {'left2': 1, 'right4': 1}))
        stubborn = fixture.events('stubborn_started')[0]
        self.assertFalse(running(stubborn['child_pid']))
        native_end = [e['time'] for e in fixture.events('native_finished') if e['pair'] == 2][0]
        verify_start = [e['time'] for e in fixture.events('verify_started') if e['pair'] == 2][0]
        self.assertGreaterEqual(verify_start - native_end, DEFAULT_CONTROL['timing']['KILL_AFTER'] - .5)  # SIGINT, SIGTERM ignored

    def test_orchestrator_crash_blocks_resume_until_every_owned_process_ended(self):
        fixture = self.fixture('crash', {'block': ['2:0', '4:0'], 'daemon': {'4:0': 120}})
        child = fixture.start()
        fixture.wait_events('native_started', 2)
        daemon = fixture.wait_events('daemon_started', 1)[0]['child_pid']
        fixture.wait_for(lambda: str(daemon) in fixture.process()['owned_processes'], 'heartbeat with the daemon')
        os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=5)
        natives = [event['pid'] for event in fixture.events('native_started')]
        self.assertTrue(all(running(pid) for pid in natives))
        self.assertEqual(fixture.start(resume=True).wait(timeout=30), 1)
        self.assertIn('Previous owned processes remain alive', fixture.log())
        fixture.update_control(block=[])
        fixture.wait_for(lambda: not any(running(pid) for pid in natives), 'natives to finish', watch=False)
        process = fixture.process(); process['owned_processes'] = {}  # an owned pid that process.json missed
        (fixture.output / 'process.json').write_text(json.dumps(process))
        self.assertEqual(fixture.start(resume=True).wait(timeout=30), 1)
        self.assertIn(f'Previous owned processes remain alive: [\'{daemon}\']', fixture.log())
        os.kill(daemon, signal.SIGKILL)
        fixture.wait_for(lambda: not running(daemon), 'daemon to end', watch=False)
        state = fixture.wait(fixture.start(resume=True), 0)
        self.assertEqual(fixture.records(state), [('left2', 0, 0, 'accepted'), ('right4', 0, 0, 'accepted')])
        self.assertEqual(len(fixture.events('native_started')), 2)
        self.assertEqual(len(state['sessions']), 2)  # refused resumes write nothing

    # ------------------------------------------------------------------ resource gate and foreign Kits
    def test_foreign_kit_wait_and_fail_never_create_a_record(self):
        foreign = self.sentinel()
        fixture = self.fixture('foreign_wait', {'foreign_pids': [foreign.pid]})
        child = fixture.start()
        fixture.wait_for(lambda: gate_kind(fixture) == 'foreign_kit', 'foreign Kit wait')
        time.sleep(.5)
        self.assertEqual((fixture.events('native_started'), fixture.state()['attempts']), ([], []))
        self.assertEqual(fixture.process()['gate']['foreign_kits'], [foreign.pid])
        fixture.update_control(foreign_pids=[])
        state = fixture.wait(child, 0)
        self.assertEqual(state['status'], 'complete')
        self.assertIsNone(foreign.poll())
        fixture = self.fixture('foreign_fail', {'foreign_pids': [foreign.pid]}, extra=['--foreign-kit', 'fail'])
        state = fixture.wait(fixture.start(), 130)
        self.assertEqual((state['status'], state['stop_mode'], state['attempts']), ('stopped', 'foreign_kit', []))
        self.assertIn(f'Another native Kit process is active: [{foreign.pid}]', state['error'])
        self.assertFalse((fixture.output / 'left2').exists() or (fixture.output / 'right4').exists())
        fixture.update_control(foreign_pids=[])
        self.assertEqual(fixture.wait(fixture.start(resume=True), 0)['status'], 'complete')
        self.assertIsNone(foreign.poll())

    def test_resource_gate_waits_for_disk_gpu_query_vram_and_ram(self):
        gpus = DEFAULT_CONTROL['gpus']
        fixture = self.fixture('gate', {'disk_free_gib': 25})  # native floor 20 + 10 per attempt
        child = fixture.start()
        for kind, change in (('disk', None),
                             ('gpu_query', {'disk_free_gib': 1000, 'gpus': [dict(gpus[0], index=1)]}),
                             ('vram', {'gpus': [dict(gpus[0], used=12303, free=4000)]}),
                             ('ram', {'gpus': gpus, 'ram_available_mib': 9000})):
            if change:
                fixture.update_control(**change)
            fixture.wait_for(lambda: gate_kind(fixture) == kind, f'{kind} gate')
            time.sleep(.3)
            self.assertEqual((fixture.events('native_started'), fixture.state()['attempts']), ([], []))
            self.assertEqual(fixture.state()['status'], 'running')
            self.assertFalse(fixture.process()['gate']['satisfied'])
        fixture.update_control(ram_available_mib=65536)
        self.assertEqual(fixture.wait(child, 0)['status'], 'complete')

    def test_warm_up_reservation_holds_second_kit_until_memory_is_reported(self):
        gpu = [dict(DEFAULT_CONTROL['gpus'][0], used=8303, free=8000)]
        fixture = self.fixture('warmup', {'gpus': gpu, 'block': ['2:0', '4:0']}, extra=['--warmup-seconds', '60'])
        child = fixture.start()
        first = fixture.wait_events('native_started', 1)[0]
        fixture.wait_for(lambda: gate_kind(fixture) == 'vram', 'reservation hold')
        time.sleep(.5)
        self.assertEqual(len(fixture.events('native_started')), 1)
        self.assertEqual(fixture.process()['gate']['vram_reserved_mib'], 4608.)
        fixture.update_control(apps=[{'uuid': gpu[0]['uuid'], 'pid': first['pid'], 'used': 4608}])
        fixture.wait_events('native_started', 2)
        fixture.update_control(block=[])
        self.assertEqual(fixture.wait(child, 0)['status'], 'complete')
        fixture = self.fixture('warmup_expiry', {'gpus': gpu, 'delay': {'*': 2.}}, extra=['--warmup-seconds', '1'])
        state = fixture.wait(fixture.start(), 0)
        starts = sorted(event['time'] for event in fixture.events('native_started'))
        self.assertGreaterEqual(starts[1] - starts[0], .8)
        self.assertEqual(max_overlap(fixture.events()), 2)

    def test_idle_gate_timeout_stops_resumably(self):
        fixture = self.fixture('idle_timeout', {'disk_free_gib': 25}, extra=['--max-idle-wait-seconds', '.5'])
        state = fixture.wait(fixture.start(), 130)
        self.assertEqual((state['status'], state['stop_mode'], state['attempts']), ('stopped', 'gate_timeout', []))
        self.assertIn('Disk', state['error'])
        fixture.update_control(disk_free_gib=1000)
        self.assertEqual(fixture.wait(fixture.start(resume=True), 0)['status'], 'complete')

    def test_nvidia_smi_parsing_and_preflight(self):
        rows = parallel.parse_gpu_rows('0, GPU-a, 16303, 4186, 12117\n1, GPU-b, 24576, [N/A], [N/A]\n')
        self.assertEqual((rows[0]['uuid'], rows[0]['free_mib'], rows[1]['free_mib']), ('GPU-a', 12117., None))
        apps = parallel.parse_compute_apps('GPU-a, 10, 3988\nGPU-b, 11, 100\nGPU-a, 12, [N/A]\nGPU-a, 10, 12\n', 'GPU-a')
        self.assertEqual(apps, {10: 4000., 12: 0.})
        with self.assertRaisesRegex(ValueError, 'Unexpected nvidia-smi GPU row'):
            parallel.parse_gpu_rows('No devices were found')
        with mock.patch.object(parallel, 'nvidia_smi', return_value='0, GPU-a, 1, 1, 1\n'), \
                self.assertRaisesRegex(ValueError, 'no GPU index 1'):
            parallel.query_gpu(1)
        fixture = self.fixture('no_gpu', {'nvidia_smi_error': True})
        self.assertEqual(fixture.start().wait(timeout=30), 1)
        self.assertIn('needs a working nvidia-smi', fixture.log())
        self.assertFalse(fixture.output.exists())

    # ------------------------------------------------------------------ isolation and exclusion
    def test_children_get_private_tmpdir_and_session_token(self):
        fixture = self.fixture('environment', {'delay': {'*': .3}}, quotas=(2, 2))
        state = fixture.wait(fixture.start(), 0)
        session = state['sessions'][-1]
        self.assertEqual((session['slots'], session['resource_policy']['min_free_vram_gib']), (2, 4.5))
        self.assertEqual(fixture.process()['owner_token'], session['owner_token'])
        by_seed = {(record['pair'], record['seed']): record for record in state['attempts']}
        for event in fixture.events('native_started'):
            token, slot = event['owner'].split(':')
            self.assertEqual(token, session['owner_token'])
            self.assertEqual(int(slot), by_seed[(event['pair'], event['seed'])]['slot'])
            tmpdir = Path(event['tmpdir'])
            self.assertEqual(tmpdir.parent, fixture.tmp / f'igris-nepes-{os.getuid()}-slots')
            self.assertTrue(tmpdir.name.endswith('-' + slot) and len(str(tmpdir.name)) <= 12)
            runtime = fixture.bundle.resolve() / 'runtime' / ('left2' if event['pair'] == 2 else 'right4') / 'source'
            self.assertEqual(event['pythonpath'].split(os.pathsep), [str(runtime / 'robros_motion_planning'), str(runtime / 'robros_lab')])
        first = [event for event in fixture.events('native_started') if event['seed'] == 0]
        self.assertNotEqual(first[0]['tmpdir'], first[1]['tmpdir'])  # concurrent attempts never share TMPDIR

    def test_global_lock_still_excludes_other_simulator_tools(self):
        fixture = self.fixture('exclusive', {'block': ['2:0', '4:0']})
        child = fixture.start()
        fixture.wait_events('native_started', 2)
        for launcher, extra in (('serial_launch.py', []), ('launch.py', ['--parallel', '2'])):
            command = [sys.executable, '-B', str(fixture.tools / launcher), '--bundle', str(fixture.bundle),
                       '--output', str(fixture.root / ('other_' + launcher)), '--isaac-python', sys.executable,
                       '--dataset-python', sys.executable, *extra]
            result = subprocess.run(command, env=fixture.environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn('Resource temporarily unavailable', result.stderr)
        fixture.update_control(block=[])
        self.assertEqual(fixture.wait(child, 0)['status'], 'complete')


if __name__ == '__main__':
    unittest.main(verbosity=2)
