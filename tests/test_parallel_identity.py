"""N=1 identity proof against the release whose serial engine this one ships, read-only on that release.

Skipped unless NEPES_ORIGINAL_BUNDLE names that release. For this Pink IK + parallel release it is the
Pink IK release (igris_nepes_dataset_200_pink): its serial engine (collect_seeded.py, merge_seeded.py,
check_release.py, setup.sh, ...) is shipped byte-identical, and only CHANGED differ and ADDED are new.
Both releases run the Pink batch preflight and freeze the IK Python path in their plans, so give one IK
Python outside both releases (the reference has no .venv; a path inside one release is normalised differently):
  NEPES_ORIGINAL_BUNDLE=/path/to/igris_nepes_dataset_200_pink NEPES_IK_PYTHON=/path/to/env-root/ik/bin/python \
    "$NEPES_DATASET_PYTHON" -B -m unittest discover -s tests -p test_parallel_identity.py
Against the earlier native-solver release (igris_nepes_dataset_200) it fails by design on the files the
Pink lane changed or added (serial engine hashes, upper_body_ik in the plans).
Every original file is only read; its scripts run with -B and PYTHONDONTWRITEBYTECODE.
Outputs of both releases are compared after replacing each release's own path
and manifest digest, the only inputs that differ by construction.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

from seeded_process_fixture import Fixture

NEW = Path(__file__).resolve().parents[1]
ORIGINAL = Path(os.environ['NEPES_ORIGINAL_BUNDLE']).resolve() if os.environ.get('NEPES_ORIGINAL_BUNDLE') else None
CHANGED = {'README.md', 'scripts/collect.py', 'scripts/campaign.py', 'scripts/pipeline.py', 'scripts/build_release.py'}
ADDED = {'scripts/seeded/collect_parallel.py', 'scripts/import_candidates.py', 'scripts/compare_attempts.py',
         'tests/parallel_process_fixture.py', 'tests/test_parallel_collect.py', 'tests/test_parallel_recovery.py',
         'tests/test_parallel_tools.py', 'tests/test_parallel_entrypoints.py', 'tests/test_parallel_identity.py',
         'tests/test_parallel_pink.py'}
SERIAL_ENGINE = ('scripts/seeded/collect_seeded.py', 'scripts/seeded/merge_seeded.py', 'scripts/seeded/seeded_config.py',
                 'scripts/campaign_sources.py', 'scripts/check_release.py', 'scripts/augment.py', 'scripts/replay.py',
                 'scripts/setup.sh')
ENVIRONMENT = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
for _key in ('PYTHONPATH', 'NEPES_SIM_PYTHON', 'NEPES_DATASET_PYTHON', 'NEPES_PARALLEL_OWNER'):
    ENVIRONMENT.pop(_key, None)
VOLATILE = {'created_utc', 'updated_utc', 'started_utc', 'finished_utc', 'pid', 'wall_seconds'}

ARGV_DRIVER = r'''
import importlib.util,json,os,sys
from pathlib import Path
from unittest import mock
spec=importlib.util.spec_from_file_location('public_collect_under_test',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
with mock.patch.object(module.os,'execve') as execute:
    module.main(sys.argv[2:])
executable,argv,env=execute.call_args.args
print(json.dumps({'executable':executable,'argv':argv,'environment_changes':{k:v for k,v in env.items() if os.environ.get(k)!=v}}))
'''

CAMPAIGN_DRIVER = r'''
import importlib.util,json,sys
from pathlib import Path
from unittest import mock
bundle,run,ledger,stage,mode=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],sys.argv[4],sys.argv[5]
spec=importlib.util.spec_from_file_location('campaign_under_test',bundle/'scripts/campaign.py')
campaign=importlib.util.module_from_spec(spec);spec.loader.exec_module(campaign)
def stages(plan):
    return [(label,[sys.executable,'-B',stage,label,str(run)]) for label in ('verify_imports','collect_base','merge_base','augment')]
argv=['--imports',ledger,'--output',str(run),'--isaac-python',sys.executable,'--dataset-python',sys.executable]
if mode!='start':argv.append('--resume')
if mode=='start':argv+=['--stop-after-new-successes','2']
with mock.patch.object(campaign.helpers,'verify_bundle',return_value='frozen_manifest'),\
     mock.patch.object(campaign,'commands',side_effect=stages),\
     mock.patch.object(campaign,'validate_completion',side_effect=ValueError('synthetic completion check')):
    try:code=campaign.main(argv)
    except ValueError as exc:code='ValueError: '+str(exc)
print(json.dumps({'code':code}))
'''

STAGE = r'''
import json,signal,sys,time
from pathlib import Path
label,run=sys.argv[1],Path(sys.argv[2])
if label=='collect_base':
    plan=json.loads((run/'campaign.json').read_text())['plan'];batch=run/'new_base/batch.json'
    status='complete' if batch.exists() else 'checkpoint'
    batch.parent.mkdir(exist_ok=True)
    batch.write_text(json.dumps({'schema':'nepes_seeded_batch_v1','status':status,'attempts':[{'profile':'left2','status':'accepted'}],
        'counts':{'left2':1,'right4':0},'plan':{'quota_by_profile':plan['new_base_by_profile'],
        'max_attempts_by_profile':{p:plan['max_attempts_per_pair'] for p in ('left2','right4')},'first_seed':plan['first_seed'],
        'bundle':plan['bundle'],'output':str(run/'new_base'),'bundle_manifest_sha256':plan['bundle_manifest_sha256'],
        'isaac_python':plan['isaac_python'],'dataset_python':plan['dataset_python']}}))
if label=='merge_base' and not (run.parent/'merge_stopped_once').exists():
    (run.parent/'merge_stopped_once').touch()
    signal.signal(signal.SIGINT,lambda *_:sys.exit(130))
    (run/'STOP').touch()
    while True:time.sleep(.05)
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ledger_document():
    """Fixed, path-independent import ledger so both releases hash identical bytes."""
    return {'imports': [{'profile': profile, 'attempt': f'/nonexistent/nepes-identity/old/{profile}/{index}'}
                        for profile, count in (('left2', 14), ('right4', 21)) for index in range(count)]}


def sampled_config(bundle, pair, seed):
    """(config_sha256, config bytes with the release path normalised) from the release's own sampler."""
    spec = importlib.util.spec_from_file_location(f'identity_sampler_{abs(hash(str(bundle)))}', bundle / 'scripts/seeded/seeded_config.py')
    sampler = importlib.util.module_from_spec(spec); spec.loader.exec_module(sampler)
    config, sample = sampler.build_config(bundle, pair, seed)
    payload = sampler.json_bytes(config)
    assert hashlib.sha256(payload).hexdigest() == sample['config_sha256']
    return sample['config_sha256'], payload.replace(str(bundle).encode(), b'<BUNDLE>')


def normalise(value, replacements, drop=frozenset()):
    if isinstance(value, dict):
        return {normalise(key, replacements): normalise(item, replacements, drop) for key, item in value.items() if key not in drop}
    if isinstance(value, list):
        return [normalise(item, replacements, drop) for item in value]
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
    return value


def text(value):
    """Order-sensitive serialisation, as the collectors write their JSON."""
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)


def run(command, **options):
    return subprocess.run(command, capture_output=True, text=True, env=options.pop('env', ENVIRONMENT), timeout=120, **options)


@unittest.skipUnless(ORIGINAL, 'Set NEPES_ORIGINAL_BUNDLE to the Pink IK release (see module docstring) to run the N=1 identity proof')
class SerialIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bytecode = cls.check_original()

    @classmethod
    def tearDownClass(cls):
        assert cls.check_original() == cls.bytecode, 'The original release gained bytecode'

    @staticmethod
    def check_original():
        """The original release verifies against its own manifest; returns its existing bytecode files."""
        result = run([sys.executable, '-B', str(ORIGINAL / 'scripts/check_release.py')])
        assert result.returncode == 0, result.stderr
        return sorted((str(path), path.stat().st_mtime_ns) for path in ORIGINAL.rglob('*.py[co]'))

    def replacements(self, bundle):
        return [(str(bundle), '<BUNDLE>'), (sha(bundle / 'manifest.json'), '<MANIFEST>')]

    def both(self, function):
        return {name: function(bundle) for name, bundle in (('original', ORIGINAL), ('new', NEW))}

    def assert_same(self, outputs, drop=frozenset()):
        original = normalise(outputs['original'], self.replacements(ORIGINAL), drop)
        new = normalise(outputs['new'], self.replacements(NEW), drop)
        self.assertEqual(text(new), text(original))  # values and key order
        return new

    def test_serial_engine_and_dependencies_are_byte_identical(self):
        original = json.loads((ORIGINAL / 'manifest.json').read_text())['files']
        new = json.loads((NEW / 'manifest.json').read_text())['files']
        self.assertEqual(set(new) - set(original), ADDED)
        self.assertEqual(set(original) - set(new), set())
        self.assertEqual({name for name in original if original[name] != new[name]}, CHANGED)
        for name in sorted(set(original) - CHANGED):  # on-disk bytes, independent of either manifest
            self.assertEqual(sha(NEW / name), sha(ORIGINAL / name), name)
        for name in SERIAL_ENGINE:
            self.assertEqual(sha(NEW / name), sha(ORIGINAL / name), name)

    def test_collect_entrypoint_emits_the_previous_command(self):
        with tempfile.TemporaryDirectory(prefix='nepes-identity-') as directory:
            output = Path(directory) / 'batch'
            driver = Path(directory) / 'driver.py'; driver.write_text(ARGV_DRIVER)
            interpreters = ['--isaac-python', '/opt/sim/bin/python', '--dataset-python', '/opt/dataset/bin/python']
            cases = [['--per-pair', '3', '--first-seed', '10', *interpreters],
                     ['--left-successes', '86', '--right-successes', '79', '--max-attempts-per-pair', '395', '--first-seed', '10000', *interpreters],
                     ['--per-pair', '2', '--stop-after-total-successes', '4', '--max-wall-seconds', '600', '--dry-run', *interpreters],
                     ['--per-pair', '25'],  # interpreter defaults inside each release
                     ['--per-pair', '2', '--resume', *interpreters]]
            for argv in cases:
                with self.subTest(argv=argv):
                    if '--resume' in argv:
                        output.mkdir(exist_ok=True)
                        (output / 'batch.json').write_text(json.dumps({'plan': {'ordering': 'round_robin_serial'}}))

                    def emit(bundle):
                        result = run([sys.executable, '-B', str(driver), str(bundle / 'scripts/collect.py'), *argv, '--output', str(output)])
                        self.assertEqual(result.returncode, 0, result.stderr)
                        return json.loads(result.stdout)

                    self.assertIn('collect_seeded.py', self.assert_same(self.both(emit))['argv'][2])

    def test_collect_dry_run_plans_are_identical(self):
        with tempfile.TemporaryDirectory(prefix='nepes-identity-') as directory:
            for argv, seed in ((['--per-pair', '25', '--first-seed', '1000'], 1000),
                               (['--left-successes', '86', '--right-successes', '79', '--first-seed', '10000', '--max-attempts-per-pair', '395'], 10000)):
                with self.subTest(argv=argv):
                    def emit(bundle):
                        result = run([sys.executable, '-B', str(bundle / 'scripts/collect.py'), *argv, '--output', str(Path(directory) / 'batch'),
                                      '--isaac-python', sys.executable, '--dataset-python', sys.executable, '--dry-run'])
                        self.assertEqual(result.returncode, 0, result.stderr)
                        document = json.loads(result.stdout)
                        for example in document['examples']:  # its config embeds the release path; compare the bytes instead
                            digest, payload = sampled_config(bundle, example['sample']['pair'], seed)
                            self.assertEqual(example['sample']['config_sha256'], digest)
                            example['sample']['config_sha256'] = payload.decode()
                        return document

                    plan = self.assert_same(self.both(emit))['plan']
                    self.assertEqual((plan['ordering'], 'parallel' in plan), ('round_robin_serial', False))
            self.assertFalse((Path(directory) / 'batch').exists())

    def test_campaign_and_pipeline_dry_runs_are_identical(self):
        with tempfile.TemporaryDirectory(prefix='nepes-identity-') as directory:
            root = Path(directory)
            ledger = root / 'imports.json'
            ledger.write_text(json.dumps(ledger_document()))

            def campaign(bundle):
                result = run([sys.executable, '-B', str(bundle / 'scripts/campaign.py'), '--imports', str(ledger), '--output', str(root / 'run'),
                              '--isaac-python', sys.executable, '--dataset-python', sys.executable, '--dry-run'])
                self.assertEqual(result.returncode, 0, result.stderr)
                return json.loads(result.stdout)

            self.assertNotIn('parallel', self.assert_same(self.both(campaign))['plan'])
            for argv in (['--per-pair', '42', '--augmentation-mode', 'both', '--augmentation-episodes', '42'],
                         ['--per-pair', '2', '--max-attempts-per-pair', '9', '--video-presets', 'darker', 'brighter', '--resume']):
                with self.subTest(argv=argv):
                    def pipeline(bundle):
                        result = run([sys.executable, '-B', str(bundle / 'scripts/pipeline.py'), *argv, '--output', str(root / 'pipeline'),
                                      '--isaac-python', '/opt/sim/bin/python', '--dataset-python', '/opt/dataset/bin/python', '--dry-run'])
                        self.assertEqual(result.returncode, 0, result.stderr)
                        return json.loads(result.stdout)

                    self.assertNotIn('--parallel', self.assert_same(self.both(pipeline))['commands']['collect'])
            self.assertFalse((root / 'run').exists() or (root / 'pipeline').exists())

    def test_serial_collector_state_files_are_identical(self):
        for scenario, control in (('complete', {}), ('budget', {'collect_failure': 'left'})):
            with self.subTest(scenario=scenario):
                outputs = {}
                for name, bundle in (('original', ORIGINAL), ('new', NEW)):
                    with tempfile.TemporaryDirectory(prefix='nepes-identity-') as directory:
                        fixture = Fixture(Path(directory), bundle / 'scripts/seeded/collect_seeded.py', name, control)
                        try:
                            code = fixture.start().wait(timeout=60)
                            replacements = [(str(fixture.root), '<ROOT>')]
                            outputs[name] = {'code': code,
                                             'batch': normalise(json.loads((fixture.output / 'batch.json').read_text()), replacements, VOLATILE),
                                             'process': normalise(json.loads((fixture.output / 'process.json').read_text()), replacements, VOLATILE),
                                             'files': sorted(str(p.relative_to(fixture.output)) for p in fixture.output.rglob('*'))}
                        finally:
                            fixture.close()
                self.assertEqual(text(outputs['new']), text(outputs['original']))
                self.assertEqual(outputs['new']['code'], 0 if scenario == 'complete' else 1)

    def test_serial_campaign_state_is_identical_including_ignored_drain(self):
        steps = {}
        for name, bundle in (('original', ORIGINAL), ('new', NEW)):
            with tempfile.TemporaryDirectory(prefix='nepes-identity-') as directory:
                root = Path(directory)
                ledger = root / 'imports.json'
                ledger.write_text(json.dumps(ledger_document()))
                (root / 'driver.py').write_text(CAMPAIGN_DRIVER); (root / 'stage.py').write_text(STAGE)
                result = []
                for mode in ('start', 'resume', 'resume'):
                    if mode == 'resume' and not (root / 'run/DRAIN').exists():
                        (root / 'run/DRAIN').touch()  # a serial campaign ignores DRAIN in both releases
                    completed = run([sys.executable, '-B', str(root / 'driver.py'), str(bundle), str(root / 'run'), str(ledger),
                                     str(root / 'stage.py'), mode])
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    state = json.loads((root / 'run/campaign.json').read_text())
                    files = sorted(re.sub(r'_resumed_\d+', '_resumed_N', str(p.relative_to(root / 'run'))) for p in (root / 'run').rglob('*'))
                    result.append(normalise({'code': json.loads(completed.stdout)['code'], 'state': state, 'files': files},
                                            [(str(root), '<ROOT>'), (str(bundle), '<BUNDLE>')], VOLATILE | {'owned_processes'}))
                steps[name] = result
        self.assertEqual(text(steps['new']), text(steps['original']))
        self.assertEqual([step['code'] for step in steps['new']], [0, 130, 'ValueError: synthetic completion check'])
        self.assertEqual([step['state']['status'] for step in steps['new']], ['checkpoint', 'stopped', 'needs_review'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
