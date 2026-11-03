"""Entry points: N=1 keeps the previous commands and plans; --parallel 2..3 routes to
scripts/seeded/collect_parallel.py and is frozen, inherited on resume and forwarded.
Also checks the release inventory and manifest."""
import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

BUNDLE = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, BUNDLE / relative)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


collect = load('nepes_test_public_collect', 'scripts/collect.py')
campaign = load('nepes_test_public_campaign', 'scripts/campaign.py')
pipeline = load('nepes_test_public_pipeline', 'scripts/pipeline.py')
engine = load('nepes_test_parallel_engine', 'scripts/seeded/collect_parallel.py')
build_release = load('nepes_test_build_release', 'scripts/build_release.py')
ENGINE_OPTIONS = {flag: kind for flag, kind, *_ in engine.RESOURCE_OPTIONS} | {'--foreign-kit': str}


class CollectEntrypointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='nepes-entry-'); self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / 'batch'

    def command(self, *argv):
        with mock.patch.object(collect.os, 'execve') as execute:
            collect.main([*argv, '--output', str(self.output), '--isaac-python', 'I', '--dataset-python', 'D'])
        self.assertEqual(execute.call_args.args[0], sys.executable)
        return execute.call_args.args[1]

    def legacy(self, *tail):
        """The exact command of the previous release (scripts/collect.py before --parallel)."""
        return [sys.executable, '-B', str(BUNDLE / 'scripts/seeded/collect_seeded.py'), '--bundle', str(BUNDLE),
                '--output', str(self.output), '--isaac-python', 'I', '--dataset-python', 'D', '--first-seed', '5',
                '--max-wall-seconds', '5400', *tail]

    def batch(self, plan):
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / 'batch.json').write_text(json.dumps({'plan': plan, 'status': 'stopped', 'attempts': []}))

    def test_serial_command_is_unchanged_for_omitted_or_explicit_one(self):
        for extra in ([], ['--parallel', '1']):
            self.assertEqual(self.command('--per-pair', '2', '--first-seed', '5', *extra), self.legacy('--max-successes-per-pair', '2'))
            self.assertEqual(self.command('--left-successes', '86', '--right-successes', '79', '--first-seed', '5',
                                          '--max-attempts-per-pair', '395', '--stop-after-total-successes', '4', '--dry-run', *extra),
                             self.legacy('--left-successes', '86', '--right-successes', '79', '--max-attempts-per-pair', '395',
                                         '--stop-after-total-successes', '4', '--dry-run'))

    def test_parallel_routes_to_engine_with_gate_options(self):
        command = self.command('--per-pair', '2', '--first-seed', '5', '--parallel', '2', '--min-free-vram-gib', '9',
                               '--foreign-kit', 'fail', '--gpu-index', '1')
        expected = self.legacy('--max-successes-per-pair', '2', '--parallel', '2', '--min-free-vram-gib', '9.0',
                               '--gpu-index', '1', '--foreign-kit', 'fail')
        expected[2] = str(BUNDLE / 'scripts/seeded/collect_parallel.py')
        self.assertEqual(command, expected)

    def test_invalid_combinations_are_rejected(self):
        for argv in (['--parallel', '4'], ['--parallel', '0'], ['--min-free-vram-gib', '9'],
                     ['--parallel', '1', '--foreign-kit', 'fail'], ['--foreign-kit', 'later', '--parallel', '2']):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.command('--per-pair', '2', *argv)

    def test_resume_inherits_frozen_concurrency_and_refuses_a_different_one(self):
        self.batch({'ordering': 'round_robin_parallel_v1', 'parallel': {'slots': 3}})
        command = self.command('--per-pair', '2', '--first-seed', '5', '--resume', '--min-free-vram-gib', '9')
        self.assertEqual(command[2], str(BUNDLE / 'scripts/seeded/collect_parallel.py'))
        self.assertEqual(command[-5:], ['--resume', '--parallel', '3', '--min-free-vram-gib', '9.0'])
        for value in ('1', '2'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.command('--per-pair', '2', '--resume', '--parallel', value)
        self.batch({'ordering': 'round_robin_serial'})
        self.assertEqual(self.command('--per-pair', '2', '--first-seed', '5', '--resume'),
                         self.legacy('--max-successes-per-pair', '2', '--resume'))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.command('--per-pair', '2', '--resume', '--parallel', '2')
        (self.output / 'batch.json').write_text('not json')  # the engine reports it, as before
        self.assertEqual(self.command('--per-pair', '2', '--first-seed', '5', '--resume'),
                         self.legacy('--max-successes-per-pair', '2', '--resume'))

    def test_interrupt_unfinished_is_forwarded_only_when_resuming_a_parallel_batch(self):
        self.batch({'ordering': 'round_robin_parallel_v1', 'parallel': {'slots': 2}})
        command = self.command('--per-pair', '2', '--first-seed', '5', '--resume', '--interrupt-unfinished')
        self.assertEqual(command[-4:], ['--resume', '--parallel', '2', '--interrupt-unfinished'])
        self.assertNotIn('--interrupt-unfinished', self.command('--per-pair', '2', '--first-seed', '5', '--resume'))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.command('--per-pair', '2', '--parallel', '2', '--interrupt-unfinished')  # fresh batch
        self.batch({'ordering': 'round_robin_serial'})
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.command('--per-pair', '2', '--resume', '--interrupt-unfinished')  # serial batch: never changed

    def status(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(collect.main(['--output', str(self.output), '--status']), 0)
        return json.loads(stream.getvalue())

    def test_status_is_unchanged_for_serial_and_extended_for_parallel(self):
        self.batch({'ordering': 'round_robin_serial'})
        self.assertEqual(set(self.status()), {'status', 'counts', 'active', 'dataset', 'error', 'updated_utc'})
        self.output.joinpath('batch.json').write_text(json.dumps({
            'plan': {'parallel': {'slots': 2}}, 'status': 'running', 'active': ['a', 'b'], 'draining': None,
            'attempts': [{'profile': 'left2', 'status': 'accepted'}, {'profile': 'left2', 'status': 'running'},
                         {'profile': 'right4', 'status': 'failed'}]}))
        self.output.joinpath('process.json').write_text(json.dumps({'slots': [{'slot': 0, 'phase': 'native'}],
                                                                    'gate': {'satisfied': False, 'kind': 'vram'}}))
        status = self.status()
        self.assertEqual((status['parallel'], status['active'], status['gate']['kind']), (2, ['a', 'b'], 'vram'))
        self.assertEqual(status['records'], {'left2': {'accepted': 1, 'running': 1}, 'right4': {'failed': 1}})
        self.assertEqual(status['slots'], [{'slot': 0, 'phase': 'native'}])


class CampaignEntrypointTests(unittest.TestCase):
    def make_args(self, root, **extra):
        ledger = root / 'imports.json'
        ledger.write_text(json.dumps({'imports': [{'profile': name, 'attempt': str(root / name / str(index))}
                                                  for name, count in (('left2', 14), ('right4', 21)) for index in range(count)]}))
        return argparse.Namespace(imports=ledger, output=root / 'run', base_per_pair=100, augmentation_episodes=100,
                                  first_seed=10000, max_attempts_per_pair=None, max_attempts_per_parent=5, workers=2,
                                  video_preset='darker', isaac_python=sys.executable, dataset_python=sys.executable, **extra)

    def test_serial_plan_and_commands_are_unchanged(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(campaign.helpers, 'verify_bundle', return_value='frozen'):
            root = Path(directory)
            legacy = campaign.build_plan(self.make_args(root))  # namespace without the new attribute
            for value in (None, 1):
                self.assertEqual(campaign.build_plan(self.make_args(root, parallel=value)), legacy)
            self.assertNotIn('parallel', legacy)
            stages = dict(campaign.commands(legacy))
            self.assertNotIn('--parallel', stages['collect_base'])
            self.assertEqual(stages['collect_base'][-4:], ['--isaac-python', sys.executable, '--dataset-python', sys.executable])
            parallel = campaign.build_plan(self.make_args(root, parallel=2))
            self.assertEqual(parallel, dict(legacy, parallel=2))
            changed = dict(campaign.commands(parallel))
            self.assertEqual(changed['collect_base'], stages['collect_base'] + ['--parallel', '2'])
            self.assertEqual({k: v for k, v in changed.items() if k != 'collect_base'},
                             {k: v for k, v in stages.items() if k != 'collect_base'})
            with self.assertRaisesRegex(ValueError, 'Use --parallel 1, 2 or 3'):
                campaign.build_plan(self.make_args(root, parallel=4))

    def test_new_base_concurrency_is_bound_only_for_parallel_campaigns(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(campaign.helpers, 'verify_bundle', return_value='frozen'):
            root = Path(directory)
            plan = campaign.build_plan(self.make_args(root, parallel=2))
            source = {'quota_by_profile': plan['new_base_by_profile'],
                      'max_attempts_by_profile': {name: plan['max_attempts_per_pair'] for name in campaign.PROFILES},
                      'first_seed': plan['first_seed'], 'bundle': plan['bundle'], 'output': str(Path(plan['output']) / 'new_base'),
                      'bundle_manifest_sha256': plan['bundle_manifest_sha256'], 'isaac_python': plan['isaac_python'],
                      'dataset_python': plan['dataset_python']}
            serial_plan = dict(plan); del serial_plan['parallel']
            campaign.validate_new_batch_plan(serial_plan, {'plan': source})  # serial campaigns: unchanged checks
            with self.assertRaisesRegex(ValueError, 'concurrency differs'):
                campaign.validate_new_batch_plan(plan, {'plan': source})
            for slots, error in ((2, None), (3, 'concurrency differs')):
                batch = {'plan': dict(source, ordering='round_robin_parallel_v1', parallel={'slots': slots})}
                if error:
                    with self.assertRaisesRegex(ValueError, error):
                        campaign.validate_new_batch_plan(plan, batch)
                else:
                    campaign.validate_new_batch_plan(plan, batch)

    def test_gate_options_require_parallel(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
            args = self.make_args(Path(directory))
            for extra in (['--min-free-vram-gib', '9'], ['--parallel', '4']):
                with self.assertRaises(SystemExit):
                    campaign.main(['--imports', str(args.imports), '--output', str(args.output), *extra])

    def test_drain_is_forwarded_to_parallel_collection_and_resume_inherits_concurrency(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(campaign.helpers, 'verify_bundle', return_value='frozen'):
            root = Path(directory); args = self.make_args(root)
            child = root / 'collect_base_stage.py'
            child.write_text('import json,sys,time\nfrom pathlib import Path\n'
                             'base,record=Path(sys.argv[1]),Path(sys.argv[2]);base.mkdir(parents=True,exist_ok=True)\n'
                             'record.write_text(json.dumps(sys.argv[3:]))\nend=time.monotonic()+20\n'
                             "while not (base/'DRAIN').exists() and time.monotonic()<end:time.sleep(.05)\n"
                             "raise SystemExit(130 if (base/'DRAIN').exists() else 3)\n")
            recorded = root / 'argv.json'
            base = ['--imports', str(args.imports), '--output', str(args.output), '--isaac-python', sys.executable,
                    '--dataset-python', sys.executable]

            real_commands = campaign.commands

            def stages(plan):
                self.assertEqual(dict(real_commands(plan))['collect_base'][-2:], ['--parallel', '2'])
                return [('collect_base', [sys.executable, '-B', str(child), str(args.output / 'new_base'), str(recorded)])]

            def drain():
                end = time.monotonic() + 15
                while not recorded.exists() and time.monotonic() < end:
                    time.sleep(.02)
                (args.output / 'DRAIN').touch()

            watcher = threading.Thread(target=drain, daemon=True); watcher.start()
            with mock.patch.object(campaign, 'commands', side_effect=stages):
                self.assertEqual(campaign.main(base + ['--parallel', '2', '--min-free-vram-gib', '9']), 130)
            watcher.join(timeout=5)
            self.assertTrue((args.output / 'new_base/DRAIN').exists())
            self.assertEqual(json.loads(recorded.read_text()), ['--min-free-vram-gib', '9.0'])  # runtime-only option
            state = campaign.helpers.read(args.output / 'campaign.json')
            self.assertEqual((state['status'], state['plan']['parallel']), ('stopped', 2))
            self.assertNotIn('min_free_vram_gib', json.dumps(state['plan']))
            with self.assertRaisesRegex(ValueError, 'plan/source/code changed'):
                campaign.main(base + ['--resume', '--parallel', '3'])
            quick = [('verify_imports', [sys.executable, '-c', 'pass'])]
            with mock.patch.object(campaign, 'commands', return_value=quick), \
                    mock.patch.object(campaign, 'validate_completion', side_effect=ValueError('synthetic incomplete')), \
                    self.assertRaisesRegex(ValueError, 'synthetic incomplete'):
                campaign.main(base + ['--resume'])  # concurrency inherited from the frozen plan
            self.assertFalse((args.output / 'DRAIN').exists())
            self.assertEqual(len(list(args.output.glob('DRAIN_resumed_*'))), 1)

    def test_parallel_campaign_rejects_new_seeds_that_repeat_imported_ones(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(campaign.helpers, 'verify_bundle', return_value='frozen'):
            root = Path(directory); args = self.make_args(root)
            rows = [{'profile': 'left2', 'attempt': str(root / 'old/left2' / str(seed)), 'seed': seed} for seed in (101, 10000, 10023)]
            rows += [{'profile': 'right4', 'attempt': str(root / 'old/right4' / str(seed)), 'seed': seed} for seed in range(10000, 10053)]
            args.imports.write_text(json.dumps({'imports': rows}))
            args.augmentation_episodes = 40
            serial = campaign.build_plan(args)  # serial campaigns: exactly the previous behaviour
            self.assertEqual((serial['first_seed'], 'parallel' in serial), (10000, False))
            with self.assertRaisesRegex(ValueError, r'New seeds 10000\.\.10484 repeat imported left2 seeds \[10000, 10023\]'):
                campaign.build_plan(argparse.Namespace(**vars(args), parallel=2))
            plan = campaign.build_plan(argparse.Namespace(**{**vars(args), 'first_seed': 10430}, parallel=2))
            self.assertEqual((plan['first_seed'], plan['parallel'], plan['max_attempts_per_pair']), (10430, 2, 485))

    def test_campaign_drain_is_recreated_after_a_resuming_collector_renames_it(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(campaign.helpers, 'verify_bundle', return_value='frozen'):
            root = Path(directory); args = self.make_args(root)
            child = root / 'collect_base_stage.py'
            child.write_text('import sys,time\nfrom pathlib import Path\n'
                             'base=Path(sys.argv[1]);base.mkdir(parents=True,exist_ok=True);drain=base/"DRAIN";end=time.monotonic()+20\n'
                             'while not drain.exists() and time.monotonic()<end:time.sleep(.02)\n'
                             'drain.rename(base/"DRAIN_resumed_1")  # a resuming collector renames a DRAIN older than itself\n'
                             'while not drain.exists() and time.monotonic()<end:time.sleep(.02)\n'
                             'raise SystemExit(130 if drain.exists() else 3)\n')
            base = ['--imports', str(args.imports), '--output', str(args.output), '--isaac-python', sys.executable,
                    '--dataset-python', sys.executable, '--parallel', '2']

            def drain():
                end = time.monotonic() + 15
                while not (args.output / 'new_base').exists() and time.monotonic() < end:
                    time.sleep(.02)
                (args.output / 'DRAIN').touch()

            watcher = threading.Thread(target=drain, daemon=True); watcher.start()
            stages = [('collect_base', [sys.executable, '-B', str(child), str(args.output / 'new_base')])]
            with mock.patch.object(campaign, 'commands', return_value=stages):
                self.assertEqual(campaign.main(base), 130)  # stopped at the boundary, not failed after the 20 s wait
            watcher.join(timeout=5)
            self.assertTrue((args.output / 'new_base/DRAIN').exists() and (args.output / 'new_base/DRAIN_resumed_1').exists())

    def test_interrupt_unfinished_is_forwarded_to_a_resumed_parallel_collect_base(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(campaign.helpers, 'verify_bundle', return_value='frozen'), \
                contextlib.redirect_stderr(io.StringIO()):
            root = Path(directory); args = self.make_args(root)
            recorded = root / 'argv.json'
            child = root / 'stage.py'
            child.write_text('import json,sys\nfrom pathlib import Path\nPath(sys.argv[1]).write_text(json.dumps(sys.argv[2:]))\nraise SystemExit(130)\n')
            base = ['--imports', str(args.imports), '--output', str(args.output), '--isaac-python', sys.executable,
                    '--dataset-python', sys.executable]
            for extra in (['--interrupt-unfinished'], ['--parallel', '2', '--interrupt-unfinished']):
                with self.assertRaises(SystemExit):  # fresh campaign
                    campaign.main(base + extra)
            with mock.patch.object(campaign, 'commands', side_effect=lambda plan: [('collect_base', [sys.executable, '-B', str(child), str(recorded)])]), \
                    mock.patch.object(campaign, 'validate_new_batch_plan'):
                self.assertEqual(campaign.main(base + ['--parallel', '2']), 130)
                self.assertEqual(json.loads(recorded.read_text()), [])
                (args.output / 'new_base').mkdir(); (args.output / 'new_base/batch.json').write_text(json.dumps({'status': 'stopped'}))
                self.assertEqual(campaign.main(base + ['--resume', '--interrupt-unfinished']), 130)  # N inherited from the plan
                self.assertEqual(json.loads(recorded.read_text()), ['--resume', '--interrupt-unfinished'])
                self.assertEqual(campaign.main(base + ['--resume']), 130)
                self.assertEqual(json.loads(recorded.read_text()), ['--resume'])  # runtime-only, never frozen

    def test_serial_campaign_runs_stages_with_drain_present(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(campaign.helpers, 'verify_bundle', return_value='frozen'):
            root = Path(directory); args = self.make_args(root)
            touched = root / 'ran'
            quick = [('verify_imports', [sys.executable, '-c', 'import sys;open(sys.argv[1],"w").close()', str(touched)])]
            base = ['--imports', str(args.imports), '--output', str(args.output), '--isaac-python', sys.executable,
                    '--dataset-python', sys.executable]
            with mock.patch.object(campaign, 'commands', return_value=quick), \
                    mock.patch.object(campaign, 'validate_completion', side_effect=ValueError('synthetic incomplete')):
                with self.assertRaisesRegex(ValueError, 'synthetic incomplete'):
                    campaign.main(base)
                touched.unlink()
                (args.output / 'DRAIN').touch()
                with self.assertRaisesRegex(ValueError, 'synthetic incomplete'):
                    campaign.main(base + ['--resume'])
            self.assertTrue(touched.exists())  # DRAIN is a parallel-campaign feature only
            self.assertTrue((args.output / 'DRAIN').exists())


class PipelineEntrypointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='nepes-pipeline-'); self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name).resolve() / 'run'

    def dry(self, *extra):
        result = subprocess.run([sys.executable, '-B', str(BUNDLE / 'scripts/pipeline.py'), '--per-pair', '3',
                                 '--output', str(self.output), '--isaac-python', 'I', '--dataset-python', 'D', '--dry-run', *extra],
                                capture_output=True, text=True)
        self.assertFalse(self.output.exists())
        commands = json.loads(result.stdout)['commands'] if result.returncode == 0 else None
        return result, commands, self.output

    def test_serial_commands_unchanged_and_parallel_forwarded(self):
        _, serial, output = self.dry()
        self.assertEqual(shlex.split(serial['collect']),
                         [sys.executable, '-B', str(BUNDLE / 'scripts/collect.py'), '--per-pair', '3', '--first-seed', '1000',
                          '--output', str(output / 'collection'), '--isaac-python', 'I', '--dataset-python', 'D'])
        _, parallel, _ = self.dry('--parallel', '2', '--min-free-vram-gib', '9', '--foreign-kit', 'fail')
        self.assertEqual(shlex.split(parallel['collect']),
                         shlex.split(serial['collect'])[:9] + ['--parallel', '2', '--min-free-vram-gib', '9.0', '--foreign-kit', 'fail',
                                                              '--isaac-python', 'I', '--dataset-python', 'D'])
        self.assertEqual(parallel['augment'], serial['augment'])
        result, _, _ = self.dry('--parallel', '4')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--parallel must be 1, 2 or 3', result.stderr)
        result, _, _ = self.dry('--parallel', '2', '--interrupt-unfinished')
        self.assertIn('--interrupt-unfinished requires --resume', result.stderr)

    def test_interrupt_unfinished_is_forwarded_with_a_collection_resume(self):
        (self.output / 'collection').mkdir(parents=True)
        (self.output / 'collection/batch.json').write_text('{}')
        result = subprocess.run([sys.executable, '-B', str(BUNDLE / 'scripts/pipeline.py'), '--per-pair', '3', '--output', str(self.output),
                                 '--isaac-python', 'I', '--dataset-python', 'D', '--dry-run', '--resume', '--interrupt-unfinished'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(shlex.split(json.loads(result.stdout)['commands']['collect'])[-2:], ['--resume', '--interrupt-unfinished'])


class RelativeIKPythonTests(unittest.TestCase):
    """The batch preflight resolves NEPES_IK_PYTHON against the caller's folder, but Isaac attempts start in the
    release folder: every entry point refuses a relative value before anything runs (env.sh exports absolute ones)."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='nepes-relative-ik-'); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def refused(self, call):
        with contextlib.redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit) as raised:
            call()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn('NEPES_IK_PYTHON must be an absolute path', error.getvalue())

    def test_collect_campaign_and_pipeline_refuse_a_relative_ik_python(self):
        ledger = self.root / 'imports.json'
        ledger.write_text(json.dumps({'imports': []}))
        for value in ('.venv/ik/bin/python', 'ik/bin/python'):
            with self.subTest(value=value), mock.patch.dict(os.environ, {'NEPES_IK_PYTHON': value}):
                with mock.patch.object(collect.os, 'execve') as execute:
                    self.refused(lambda: collect.main(['--per-pair', '1', '--output', str(self.root / 'batch'), '--dry-run']))
                    self.refused(lambda: collect.main(['--per-pair', '1', '--parallel', '2', '--output', str(self.root / 'batch')]))
                execute.assert_not_called()
                self.refused(lambda: campaign.main(['--imports', str(ledger), '--output', str(self.root / 'run'), '--dry-run']))
                result = subprocess.run([sys.executable, '-B', str(BUNDLE / 'scripts/pipeline.py'), '--per-pair', '1',
                                         '--output', str(self.root / 'pipeline'), '--dry-run'],
                                        env=dict(os.environ, NEPES_IK_PYTHON=value), capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('NEPES_IK_PYTHON must be an absolute path', result.stderr)
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ['imports.json'])
        for value in ('/opt/ik/bin/python', '~/ik/bin/python'):  # absolute (also after ~) values pass through unchanged
            with self.subTest(value=value), mock.patch.dict(os.environ, {'NEPES_IK_PYTHON': value}), \
                    mock.patch.object(collect.os, 'execve') as execute:
                collect.main(['--per-pair', '1', '--output', str(self.root / 'batch'), '--isaac-python', 'I', '--dataset-python', 'D'])
                self.assertEqual(execute.call_args.args[2]['NEPES_IK_PYTHON'], value)


class ReleaseTests(unittest.TestCase):
    def test_forwarded_gate_options_match_engine(self):
        for module in (collect, campaign, pipeline):
            self.assertEqual(dict(module.GATE_OPTIONS), ENGINE_OPTIONS, module.__name__)

    def test_release_inventory_and_manifest(self):
        files = dict(build_release.release_files())
        for name in ('scripts/seeded/collect_parallel.py', 'scripts/import_candidates.py', 'scripts/compare_attempts.py',
                     'tests/parallel_process_fixture.py', 'tests/test_parallel_collect.py', 'tests/test_parallel_recovery.py',
                     'tests/test_parallel_tools.py', 'tests/test_parallel_entrypoints.py', 'tests/test_parallel_identity.py',
                     'tests/test_parallel_pink.py'):
            self.assertIn(name, files)
        for name in ('PARALLEL_DESIGN.md', 'PARALLEL_NOTES.md', 'PINK_INTEGRATION_NOTES.md', 'RC2_NOTES.md'):
            self.assertNotIn(name, files)  # development and operator notes for this PC, not shipped
        self.assertFalse([name for name in files if '__pycache__' in name or name.endswith(('.pyc', '.pyo'))])
        # Like build_release.release_files(): a local .venv (setup.sh may create <release>/.venv, whose interpreters are
        # symlinks), caches and .git are never shipped, so only the rest of the tree must be free of symlinks.
        unshipped = {'__pycache__', '.pytest_cache', '.git', '.venv'}
        self.assertFalse([path for path in BUNDLE.rglob('*')
                          if path.is_symlink() and not unshipped.intersection(path.relative_to(BUNDLE).parts)])
        manifest = json.loads((BUNDLE / 'manifest.json').read_text())
        self.assertIn('parallel_collection', manifest['capabilities'])
        self.assertEqual(set(manifest['files']), set(files))
        for name, path in files.items():
            self.assertEqual(manifest['files'][name], {'bytes': path.stat().st_size, 'sha256': build_release.checksum(path)}, name)
        result = subprocess.run([sys.executable, '-B', str(BUNDLE / 'scripts/check_release.py')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
