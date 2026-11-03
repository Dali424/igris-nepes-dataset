"""Recovery and robustness of the parallel collector (review fixes), with synthetic children.

- Native exit records: an orchestrator crash with a physical failure in flight resumes; the observed
  return code is saved before verification; a native that never reported an exit (PC restart) needs
  --interrupt-unfinished; a stop forwarded by the wrapper means interrupted; the wrapper exits like the native.
- A hang-up (SIGHUP) drains; under nohup it is ignored.
- A global error keeps a native that had already exited pending for resume (never relabelled interrupted).
- A DRAIN created while a resume starts applies to it; a failed fresh start leaves no output behind.
- Launch spacing has a floor; a slow nvidia-smi never delays STOP; changed Kit settings hold new launches.

No Isaac is launched; every fixture has its own TMPDIR, so the real global lock is never touched.
"""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from parallel_process_fixture import DEFAULT_CONTROL, ParallelFixture, read, running

SEEDED = Path(__file__).resolve().parents[1] / 'scripts/seeded'
# What nohup does before exec: start the collector with SIGHUP ignored.
NOHUP = [sys.executable, '-c', 'import os,signal,sys; signal.signal(signal.SIGHUP, signal.SIG_IGN); os.execv(sys.argv[1], sys.argv[1:])']


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


parallel = load('nepes_test_parallel_recovery', SEEDED / 'collect_parallel.py')


def children(pid):
    try:
        return [int(value) for value in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]
    except OSError:
        return []


class ParallelRecoveryTests(unittest.TestCase):
    def fixture(self, name, control=None, **options):
        root = tempfile.TemporaryDirectory(prefix='nepes-recovery-'); self.addCleanup(root.cleanup)
        fixture = ParallelFixture(Path(root.name), name, control, **options)
        self.addCleanup(fixture.close)
        return fixture

    @staticmethod
    def crash(child):
        """SIGKILL the orchestrator only (OOM killer, kill -9); its wrappers and natives run in their own sessions."""
        os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)

    # ------------------------------------------------------------------ native exit records
    def test_crash_with_inflight_physical_failure_resumes_from_native_exit_records(self):
        fixture = self.fixture('crash_failure', {'block': ['2:0', '4:0'], 'outcome': {'2:0': 'fail'}}, quotas=(2, 2))
        child = fixture.start()
        natives = [event['pid'] for event in fixture.wait_events('native_started', 2)]
        self.crash(child)
        self.assertTrue(all(running(pid) for pid in natives))
        fixture.update_control(block=[])  # the orphaned natives finish on their own: one fails, one succeeds
        fixture.wait_for(lambda: not any(running(pid) for pid in natives), 'orphaned natives to finish', watch=False)
        before = fixture.state()
        self.assertEqual([(a['status'], 'returncode' in a) for a in before['attempts']], [('running', False)] * 2)
        exits = {a['profile']: read(Path(a['directory']) / 'native_exit.json') for a in before['attempts']}
        self.assertEqual({name: (value['returncode'], value['forwarded_signals']) for name, value in exits.items()},
                         {'left2': (2, []), 'right4': (0, [])})
        state = fixture.wait(fixture.start(resume=True), 0)
        self.assertEqual(state['status'], 'complete')
        left, right = state['attempts'][:2]
        self.assertEqual((left['status'], left['returncode'], left['returncode_source']), ('failed', 2, 'native_exit.json'))
        self.assertEqual((right['status'], right['returncode'], right['returncode_source']), ('accepted', 0, 'native_exit.json'))
        self.assertEqual([record[1:] for record in fixture.records(state, 'left2')], [(0, 0, 'failed'), (1, 1, 'accepted'), (2, 2, 'accepted')])
        self.assertEqual(read(fixture.output / 'dataset_4/meta/info.json')['episodes'],
                         [['left2', 1], ['left2', 2], ['right4', 0], ['right4', 1]])

    def test_return_code_is_saved_before_failure_verification(self):
        fixture = self.fixture('crash_verify_failure', {'outcome': {'2:0': 'fail'}, 'verify_failure_block': ['2:0'], 'block': ['4:0']})
        child = fixture.start()
        fixture.wait_for(lambda: fixture.events('verify_failure_started'), 'failure verification')
        self.crash(child)
        record = fixture.state()['attempts'][0]
        self.assertEqual((record['profile'], record['status'], record['returncode']), ('left2', 'running', 2))
        self.assertNotIn('returncode_source', record)  # observed by the orchestrator itself, saved immediately
        fixture.update_control(verify_failure_block=[], block=[])
        orphans = [event['pid'] for kind in ('native_started', 'verify_failure_started') for event in fixture.events(kind)]
        fixture.wait_for(lambda: not any(running(pid) for pid in orphans), 'orphans to finish', watch=False)
        state = fixture.wait(fixture.start(resume=True), 0)
        self.assertEqual(fixture.records(state), [('left2', 0, 0, 'failed'), ('right4', 0, 0, 'accepted'), ('left2', 1, 1, 'accepted')])

    def test_native_without_any_recorded_exit_needs_interrupt_unfinished(self):
        fixture = self.fixture('restart', {'block': ['2:0', '4:0']})
        child = fixture.start()
        natives = [event['pid'] for event in fixture.wait_events('native_started', 2)]
        wrappers = [fixture.wrapper_pid(pid) for pid in natives]
        self.crash(child)
        for wrapper in wrappers:  # a PC restart: each wrapper dies together with its native and records nothing
            os.killpg(wrapper, signal.SIGKILL)
        fixture.wait_for(lambda: not any(running(pid) for pid in natives + wrappers), 'natives to die', watch=False)
        before = (fixture.output / 'batch.json').read_bytes()
        self.assertEqual(fixture.start(resume=True).wait(timeout=60), 1)
        self.assertIn('never reported a native exit', fixture.log())
        self.assertIn('--interrupt-unfinished', fixture.log())
        self.assertEqual((fixture.output / 'batch.json').read_bytes(), before)  # refused before any write
        self.assertEqual(fixture.start(extra=['--interrupt-unfinished']).wait(timeout=60), 1)  # only with --resume
        fixture.update_control(block=[])
        state = fixture.wait(fixture.start(resume=True, extra=['--interrupt-unfinished']), 0)
        self.assertEqual(fixture.records(state), [('left2', 0, 0, 'interrupted'), ('right4', 0, 0, 'interrupted'),
                                                  ('left2', 1, 1, 'accepted'), ('right4', 1, 1, 'accepted')])
        self.assertIn('--interrupt-unfinished', state['attempts'][0]['reason'])
        self.assertTrue(state['sessions'][-1]['interrupt_unfinished'])

    def test_pending_outcome_uses_exit_records_and_rejects_foreign_ones(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = {'bundle': str(root / 'bundle'), 'isaac_python': '/sim/python', 'dataset_python': '/data/python',
                    'max_wall_seconds': 60}
            record = {'profile': 'left2', 'seed': 7, 'directory': str(root / 'left2/attempt_000007_seed_000007'), 'status': 'running'}
            attempt = Path(record['directory']); (attempt / 'collection').mkdir(parents=True)
            command = parallel.helpers.command_for(plan, 'left2', 7, attempt)

            def exit_record(**change):
                document = {'schema': parallel.NATIVE_EXIT_SCHEMA, 'returncode': 2, 'forwarded_signals': [],
                            'command_sha256': parallel.command_digest(command), 'wall_seconds': 1., 'finished_utc': 'x', **change}
                (attempt / 'native_exit.json').write_text(json.dumps(document))

            self.assertEqual(parallel.pending_outcome(plan, record), ('unfinished', None))
            exit_record()
            self.assertEqual(parallel.pending_outcome(plan, record)[0], 'unclassified')  # exited, outcome missing
            (attempt / 'collection/collection.json').write_text(json.dumps({'attempts': [{'accepted': False}]}))
            self.assertEqual(parallel.pending_outcome(plan, record)[0], 'failure')
            exit_record(forwarded_signals=['SIGINT'], returncode=130)
            self.assertEqual(parallel.pending_outcome(plan, record)[0], 'interrupted')
            exit_record(command_sha256='0' * 64)
            with self.assertRaisesRegex(ValueError, 'does not belong to this attempt'):
                parallel.pending_outcome(plan, record)
            (attempt / 'native_exit.json').write_text('{"schema": ')
            with self.assertRaisesRegex(ValueError, 'Unreadable native exit record'):
                parallel.pending_outcome(plan, record)
            (attempt / 'native_exit.json').unlink()
            self.assertEqual(parallel.pending_outcome(plan, dict(record, returncode=2))[0], 'failure')  # serial rule

    def test_wrapper_forwards_stops_records_the_exit_and_exits_like_the_native(self):
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        environment.pop('PYTHONPATH', None)
        with tempfile.TemporaryDirectory() as directory:
            def run(name, code=0, stop=None, kill_native=False):
                target = Path(directory) / f'{name}.json'
                native = [sys.executable, '-c', 'import sys,time; time.sleep(float(sys.argv[1])); raise SystemExit(int(sys.argv[2]))',
                          '60' if stop or kill_native else '0', str(code)]
                wrapper = subprocess.Popen(parallel.wrapped_command(native, target), env=environment, start_new_session=True,
                                           stderr=subprocess.DEVNULL)  # the interrupted native prints a traceback
                if stop or kill_native:
                    end = time.monotonic() + 20
                    while not children(wrapper.pid) and time.monotonic() < end:
                        time.sleep(.02)
                    time.sleep(.3)  # the native interpreter installed its KeyboardInterrupt handling
                    if stop:
                        wrapper.send_signal(stop)
                    else:
                        os.kill(children(wrapper.pid)[0], signal.SIGKILL)
                return wrapper.wait(timeout=30), read(target)

            for name, options, status, recorded, forwarded in (
                    ('success', {}, 0, 0, []), ('failure', {'code': 2}, 2, 2, []),
                    ('sigint', {'stop': signal.SIGINT}, -signal.SIGINT, -signal.SIGINT, ['SIGINT']),
                    ('sigterm', {'stop': signal.SIGTERM}, -signal.SIGTERM, -signal.SIGTERM, ['SIGTERM']),
                    ('killed', {'kill_native': True}, -signal.SIGKILL, -signal.SIGKILL, [])):
                with self.subTest(case=name):
                    code, document = run(name, **options)
                    self.assertEqual(code, status)
                    self.assertEqual((document['schema'], document['returncode'], document['forwarded_signals']),
                                     (parallel.NATIVE_EXIT_SCHEMA, recorded, forwarded))
                    self.assertEqual(document['command_sha256'], parallel.command_digest(
                        [sys.executable, '-c', 'import sys,time; time.sleep(float(sys.argv[1])); raise SystemExit(int(sys.argv[2]))',
                         '60' if 'stop' in options or 'kill_native' in options else '0', str(options.get('code', 0))]))

    # ------------------------------------------------------------------ hang-up
    def test_hangup_drains_in_flight_attempts_and_nohup_ignores_it(self):
        fixture = self.fixture('hangup', {'block': ['2:0', '4:0']}, quotas=(2, 2))
        child = fixture.start()
        fixture.wait_events('native_started', 2)
        child.send_signal(signal.SIGHUP)  # the terminal or SSH session closed
        fixture.wait_for(lambda: (fixture.state().get('draining') or {}).get('mode') == 'hangup', 'hang-up drain')
        self.assertEqual(fixture.events('native_interrupted'), [])
        fixture.update_control(block=[])
        state = fixture.wait(child, 130)
        self.assertEqual((state['status'], state['stop_mode'], state['sessions'][-1]['sighup']), ('stopped', 'hangup', 'drain'))
        self.assertEqual(fixture.records(state), [('left2', 0, 0, 'accepted'), ('right4', 0, 0, 'accepted')])
        state = fixture.wait(fixture.start(resume=True), 0)
        self.assertEqual((state['status'], state['counts']), ('complete', {'left2': 2, 'right4': 2}))
        ignored = self.fixture('hangup_nohup', {'block': ['2:0', '4:0']})
        child = ignored.start(command=NOHUP + ignored.command)
        ignored.wait_events('native_started', 2)
        child.send_signal(signal.SIGHUP)
        time.sleep(1.)
        self.assertIsNone(child.poll())
        self.assertNotIn('draining', ignored.state())
        ignored.update_control(block=[])
        state = ignored.wait(child, 0)
        self.assertEqual((state['status'], state['sessions'][-1]['sighup']), ('complete', 'ignored'))

    # ------------------------------------------------------------------ global error with a finished native
    def test_global_error_keeps_an_exited_native_pending_for_resume(self):
        fixture = self.fixture('exited_native', {'block': ['2:0', '4:0'], 'stubborn': {'4:0': True}})
        child = fixture.start()
        fixture.wait_events('native_started', 2)
        fixture.wait_events('stubborn_started', 1)
        fixture.update_control(block=['2:0'])  # right4 seed 0 succeeds; its slot then waits for a stubborn helper
        fixture.wait_for(lambda: [e for e in fixture.events('native_finished') if e['pair'] == 4], 'right native finished')
        config = fixture.tools / 'seeded_config.py'
        original = config.read_text()
        config.write_text(original + '\n# frozen tool changed while collecting\n')
        fixture.update_control(block=[])  # left2 finishes; launching its verifier finds the changed tool (global error)
        state = fixture.wait(child, 1)
        left, right = state['attempts']
        self.assertEqual(state['status'], 'needs_review')
        self.assertIn('Frozen batch tool changed', state['error'])
        self.assertEqual((left['status'], left['returncode']), ('running', 0))
        self.assertEqual((right['status'], right['returncode']), ('running', 0))  # finished success, not interrupted
        self.assertFalse(running(fixture.events('stubborn_started')[0]['child_pid']))
        config.write_text(original)
        document = fixture.state(); document['status'] = 'stopped'  # the operator reviewed and unblocked the batch
        (fixture.output / 'batch.json').write_text(json.dumps(document, indent=2))
        state = fixture.wait(fixture.start(resume=True), 0)
        self.assertEqual(fixture.records(state), [('left2', 0, 0, 'accepted'), ('right4', 0, 0, 'accepted')])

    # ------------------------------------------------------------------ requests while a resume starts, fresh-start failure
    def test_drain_requested_while_a_resume_starts_applies_to_it(self):
        fixture = self.fixture('drain_race', {'block': ['2:0', '4:0']}, quotas=(3, 3))
        child = fixture.start()
        fixture.wait_events('native_started', 2)
        (fixture.output / 'STOP').touch()  # ends the first session; stale for the next one
        self.assertEqual(fixture.wait(child, 130)['status'], 'stopped')
        calls = fixture.root / 'nvidia_smi_calls.log'
        before = len(calls.read_text().splitlines())
        fixture.update_control(block=[], nvidia_smi_delay=1.5)  # a slow preflight query widens the start-up window
        child = fixture.start(resume=True)
        fixture.wait_for(lambda: len(calls.read_text().splitlines()) > before, 'resume preflight')
        (fixture.output / 'DRAIN').touch()  # e.g. campaign.py forwarding a campaign DRAIN
        state = fixture.wait(child, 130, timeout=60)
        self.assertEqual((state['status'], state['stop_mode']), ('stopped', 'drain'))
        self.assertEqual(len(fixture.events('native_started')), 2)  # nothing new started
        self.assertTrue((fixture.output / 'DRAIN').exists())
        self.assertEqual(len(list(fixture.output.glob('STOP_resumed_*'))), 1)
        self.assertIn('DRAIN was requested while this resume started', fixture.log())

    def test_failed_fresh_start_leaves_no_output(self):
        fixture = self.fixture('unsafe_slots')
        base = fixture.tmp / f'igris-nepes-{os.getuid()}-slots'
        base.write_text('not a directory')
        self.assertEqual(fixture.start().wait(timeout=60), 1)
        self.assertIn('Unsafe per-slot temporary directory', fixture.log())
        self.assertFalse(fixture.output.exists())  # a fresh run is still possible once the cause is fixed
        base.unlink()
        self.assertEqual(fixture.wait(fixture.start(), 0)['status'], 'complete')

    # ------------------------------------------------------------------ gate robustness
    def test_launch_spacing_has_a_floor(self):
        base = ['--bundle', '/b', '--output', '/o', '--isaac-python', '/i', '--dataset-python', '/d', '--parallel', '2']
        self.assertEqual(parallel.MIN_LAUNCH_SPACING, 60.)
        self.assertEqual(parallel.ResourcePolicy(parallel.arguments(base + ['--launch-spacing-seconds', '60'])).launch_spacing_seconds, 60.)
        for value in ('59.9', '0'):
            with self.assertRaisesRegex(ValueError, 'at least 60'):
                parallel.ResourcePolicy(parallel.arguments(base + ['--launch-spacing-seconds', value]))

    def test_slow_nvidia_smi_never_delays_stop_or_supervision(self):
        fixture = self.fixture('slow_smi', {'block': ['2:0'], 'gpus': [dict(DEFAULT_CONTROL['gpus'][0], free=6000)]}, quotas=(2, 2))
        child = fixture.start()
        fixture.wait_events('native_started', 1)  # the second slot waits on VRAM (the warming Kit stays reserved)
        fixture.wait_for(lambda: fixture.process()['gate'].get('kind') == 'vram', 'VRAM gate closed')
        fixture.update_control(nvidia_smi_delay=10.)  # a wedged driver: samples stop arriving
        fixture.wait_for(lambda: 'has not answered' in (fixture.process()['gate'].get('reason') or ''), 'stall report', timeout=30)
        heartbeat = fixture.process()['updated_utc']
        fixture.wait_for(lambda: fixture.process()['updated_utc'] != heartbeat, 'heartbeat during the stall')
        stop = time.time(); (fixture.output / 'STOP').touch()
        interrupted = fixture.wait_events('native_interrupted', 1, timeout=30)[0]['time']
        self.assertLess(interrupted - stop, 1.5)
        self.assertEqual(child.wait(timeout=60), 130)

    def test_changed_or_torn_kit_settings_hold_new_launches(self):
        fixture = self.fixture('kit_settings', {'block': ['2:0', '4:0']}, isaac_prefix=True, spacing=2.)
        child = fixture.start()
        fixture.wait_events('native_started', 1)
        original = fixture.kit_settings.read_text()
        fixture.kit_settings.write_text(original.replace('"numThreads": 8', '"numThreads": 4'))
        fixture.wait_for(lambda: fixture.process()['gate'].get('kind') == 'kit_settings', 'Kit settings hold')
        self.assertIn('numThreads', fixture.process()['gate']['reason'])
        fixture.kit_settings.write_text('{"persistent": {"phys')  # torn while a Kit rewrites it
        fixture.wait_for(lambda: 'unreadable' in (fixture.process()['gate'].get('reason') or ''), 'torn settings hold')
        self.assertEqual(len(fixture.events('native_started')), 1)
        fixture.kit_settings.write_text(original)
        fixture.wait_events('native_started', 2)
        fixture.update_control(block=[])
        state = fixture.wait(child, 0)
        self.assertEqual(state['sessions'][-1]['kit_settings'],
                         {str(fixture.kit_settings): {'/persistent/physics/numThreads': 8, '/persistent/physics/useLocalMeshCache': True,
                                                      '/persistent/physics/localMeshCacheSizeMB': 1024}})


if __name__ == '__main__':
    unittest.main(verbosity=2)
