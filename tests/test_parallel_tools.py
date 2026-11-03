"""Tools for continuing a paused campaign and for the live determinism test.

scripts/import_candidates.py: ledger rows from accepted batch records pass the campaign's real import
verifier (campaign_sources.verify_one with a synthetic pinned verifier), the advice gives a safe first seed,
locked/changed/duplicate sources are refused and a ledger is never overwritten.
scripts/compare_attempts.py: identical attempts compare equal; any physics difference is located; the Pink IK
worker's process id (different in every run) is never a difference, its request counter is.
"""
import argparse
import contextlib
import fcntl
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

BUNDLE = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, BUNDLE / relative)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


tool = load('nepes_test_import_candidates', 'scripts/import_candidates.py')
compare = load('nepes_test_compare_attempts', 'scripts/compare_attempts.py')
campaign = load('nepes_test_campaign_for_tools', 'scripts/campaign.py')
sources = tool.sources

# A pinned "original" verifier: reproduces the saved evidence and states the lineage policy.
VERIFIER = r'''
import json,shutil,sys
from pathlib import Path
def validate_record_revision(plan,record,sample):
    return {'sample_schema':sample['schema'],'recipe_id':'synthetic'}
if __name__=='__main__':
    args=sys.argv[1:];attempt=Path(args[args.index('--attempt')+1]);output=Path(args[args.index('--output')+1])
    output.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(attempt/'verification.json',output)
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))
    return path


class SyntheticBatch:
    """A seeded batch made by a pinned bundle, with accepted and failed records."""

    def __init__(self, root, name='batch', first_seed=500, records=(('left2', 'accepted'), ('left2', 'failed'), ('right4', 'accepted'))):
        self.bundle = root / f'{name}_bundle'
        tools = {}
        for filename, content in (('merge_seeded.py', VERIFIER), ('collect_seeded.py', '# serial engine\n'), ('seeded_config.py', '# sampler\n')):
            path = self.bundle / 'scripts/seeded' / filename; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content)
            tools[filename] = sha(path)
        manifest = write(self.bundle / 'manifest.json', {'files': {f'scripts/seeded/{n}': {'sha256': c} for n, c in tools.items()}})
        self.path = root / name
        self.path.mkdir(); (self.path / '.batch.lock').touch()
        plan = {'bundle': str(self.bundle), 'first_seed': first_seed, 'max_attempts_by_profile': {'left2': 10, 'right4': 10},
                'quota_by_profile': {'left2': 3, 'right4': 3}, 'tools_sha256': tools, 'bundle_manifest_sha256': sha(manifest)}
        attempts, indices = [], {'left2': 0, 'right4': 0}
        for profile, status in records:
            index = indices[profile]; indices[profile] += 1
            seed, pair = first_seed + index, 2 if profile == 'left2' else 4
            attempt = self.path / profile / f'attempt_{index:06d}_seed_{seed:06d}'
            archive = attempt / 'collection/archives/attempt_000000'
            record = {'profile': profile, 'pair': pair, 'seed': seed, 'index': index, 'directory': str(attempt), 'status': status}
            write(attempt / 'profile/sample.json', {'schema': 'nepes_seeded_robot_initial_xy_v3', 'pair': pair, 'profile': profile, 'seed': seed})
            if status == 'accepted':
                write(archive / 'meta/info.json', {'features': {'action': {'shape': [34]}}})
                write(archive / 'meta/nepes_collection.json', {'episodes': [{'provenance': {'kind': 'original'}}]})
                receipt = write(archive / 'meta/nepes_storage_verification.json', {'verified': True, 'seed': seed})
                evidence = {'schema': 'nepes_seeded_attempt_verification_v3', 'profile': profile, 'pair': pair, 'seed': seed,
                            'attempt': str(attempt), 'archive': str(archive), 'archive_receipt_sha256': sha(receipt),
                            'capture_schema': {'fps': 30}, 'config_sha256': f'config-{profile}-{seed}',
                            'motion_exact_sha256': f'motion-{profile}-{seed}', 'motion_quantized_sha256': f'q-{profile}-{seed}',
                            'frames': 100 + seed, 'source_revision': {'sample_schema': 'nepes_seeded_robot_initial_xy_v3', 'recipe_id': 'synthetic'}}
                record['verification_sha256'] = sha(write(attempt / 'verification.json', evidence))
            attempts.append(record)
        write(self.path / 'batch.json', {'schema': 'nepes_seeded_batch_v1', 'plan': plan, 'attempts': attempts, 'status': 'stopped'})


class ImportCandidatesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='nepes-import-'); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.batch = SyntheticBatch(self.root)
        self.ledger = write(self.root / 'old_ledger.json', {
            'schema': tool.SCHEMA, 'expected_counts': {'left2': 1, 'right4': 0},
            'imports': [{'source_id': 'old-1', 'profile': 'left2', 'pair': 2, 'seed': 101, 'attempt': str(self.root / 'old/left2/a101')}]})

    def run_tool(self, *argv):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(tool.main(list(map(str, argv))), 0)
        return json.loads(stream.getvalue())

    def test_rows_of_accepted_records_pass_the_campaign_import_verifier(self):
        output = self.root / 'new/import_candidates.json'
        advice = self.run_tool('--ledger', self.ledger, '--batch', self.batch.path, '--output', output)
        document = sources.load_imports(output)
        self.assertEqual(document['expected_counts'], {'left2': 2, 'right4': 1})
        self.assertEqual([(row['profile'], row['seed']) for row in document['imports']], [('left2', 101), ('left2', 500), ('right4', 500)])
        self.assertEqual(document['imports'][0], json.loads(self.ledger.read_text())['imports'][0])  # copied verbatim
        self.assertEqual(advice['sha256'], sha(output))
        self.assertEqual((advice['suggested_first_seed'], advice['new_base_by_profile'], advice['max_augmentation_episodes']),
                         (510, {'left2': 98, 'right4': 99}, 196))  # the batch reserved seeds 500..509
        cache = set()
        for index, row in enumerate(document['imports'][1:]):  # the campaign's own verifier code, pinned verifier subprocess
            entry = sources.verify_one(row, self.root / 'evidence' / f'{index}.json', cache)
            self.assertEqual((entry['evidence']['seed'], entry['source_id']), (row['seed'], row['source_id']))
        with mock.patch.object(campaign.helpers, 'verify_bundle', return_value='frozen'):
            args = argparse.Namespace(imports=output, output=self.root / 'run', base_per_pair=100, augmentation_episodes=100,
                                      first_seed=500, max_attempts_per_pair=None, max_attempts_per_parent=5, workers=2,
                                      video_preset='darker', isaac_python=sys.executable, dataset_python=sys.executable, parallel=2)
            with self.assertRaisesRegex(ValueError, r'repeat imported left2 seeds \[500\]'):
                campaign.build_plan(args)
            args.first_seed = advice['suggested_first_seed']
            self.assertEqual(campaign.build_plan(args)['imported_by_profile'], {'left2': 2, 'right4': 1})

    def test_profile_filter_and_refusals(self):
        output = self.root / 'left_only.json'
        self.run_tool('--batch', self.batch.path, '--batch-profiles', 'left2', '--output', output)
        self.assertEqual(sources.load_imports(output)['expected_counts'], {'left2': 1, 'right4': 0})
        with self.assertRaisesRegex(ValueError, 'exists'):
            tool.main(['--batch', str(self.batch.path), '--output', str(output)])
        excluded = self.root / 'excluded.json'  # e.g. a source under /tmp that a reboot would delete
        advice = self.run_tool('--ledger', self.ledger, '--batch', self.batch.path, '--output', excluded,
                               '--exclude-attempt', self.root / 'old/left2/a101', '--exclude-attempt', self.batch.path / 'right4/attempt_000000_seed_000500')
        self.assertEqual((advice['imported'], [row['seed'] for row in sources.load_imports(excluded)['imports']]),
                         ({'left2': 1, 'right4': 0}, [500]))
        self.assertEqual(json.loads(excluded.read_text())['inputs']['ledgers'][0]['excluded'], 1)
        with self.assertRaisesRegex(ValueError, 'appears twice'):
            tool.main(['--batch', str(self.batch.path), '--batch', str(self.batch.path), '--output', str(self.root / 'twice.json')])
        with self.assertRaisesRegex(ValueError, 'outside the imported batches'):
            tool.main(['--batch', str(self.batch.path), '--output', str(self.batch.path / 'ledger.json')])
        with (self.batch.path / '.batch.lock').open('a') as holder:  # a running collector holds it exclusively
            fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, 'in use by a running collector'):
                tool.main(['--batch', str(self.batch.path), '--output', str(self.root / 'locked.json')])
        (self.batch.bundle / 'scripts/seeded/seeded_config.py').write_text('# changed\n')
        with self.assertRaisesRegex(ValueError, 'frozen tools'):
            tool.main(['--batch', str(self.batch.path), '--output', str(self.root / 'changed.json')])
        self.assertFalse(any((self.root / name).exists() for name in ('twice.json', 'locked.json', 'changed.json')))


class CompareAttemptsTests(unittest.TestCase):
    def attempt(self, root, seed=7, trace_value=1.0, config='c1', outcome='success', ik=None):
        attempt = root / (f'attempt_{seed}_{trace_value}_{config}_{outcome}' + (f'_ik{ik[0]}_{ik[1]}' if ik else ''))
        run = attempt / 'collection/runs/attempt_000000'
        write(attempt / 'profile/sample.json', {'profile': 'right4', 'pair': 4, 'seed': seed, 'applied_offset_xy_m': [0.001, -0.002]})
        if outcome == 'success':
            write(attempt / 'verification.json', {'motion_exact_sha256': 'm', 'motion_quantized_sha256': 'q', 'frames': 3,
                                                  'columns': {}, 'startup_reset': {'measured_pose_w': [0, 0, 1]},
                                                  'initial_measurement': {'poses': {'ring': [1, 2, 3]}}})
        else:
            write(attempt / 'failure_verification.json', {'reason': 'jig_displaced', 'stage': 'place'})
        write(run / 'cycle_0000_crs_pickup_load_jig/result.json', {'success': outcome == 'success', 'last_stage': 'settle', 'reason': ''})
        (run / 'startup_trace.jsonl').write_text('{"t": 0.0, "q": [0.1]}\n{"t": 0.004, "q": [0.2]}\n')
        lines = [json.dumps({'command': {'config_sha256': config, 'step': step, **({'upper_body_control': {'ik_backend': {
                                 'backend': 'pink', 'worker_pid': ik[0], 'request_id': ik[1] + step}}} if ik else {})},
                             'observation': {'time_s': step * .02, 'x': trace_value if step == 2 else 0.5, 'source': str(attempt / 'x')}})
                 for step in range(4)]
        with gzip.open(run / 'cycle_0000_crs_pickup_load_jig/trace.jsonl.gz', 'wt') as stream:
            stream.write('\n'.join(lines) + '\n')
        return attempt

    def run_compare(self, *argv):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = compare.main(list(map(str, argv)))
        return code, json.loads(stream.getvalue())

    def test_identical_and_differing_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self.attempt(root)
            copy = root / 'copy'; shutil.copytree(base, copy)
            code, report = self.run_compare(base, copy)
            self.assertEqual((code, report['bit_identical']), (0, True))  # attempt paths are normalised
            code, report = self.run_compare(base, self.attempt(root, trace_value=1.0000001))
            self.assertEqual(code, 1)
            trace = report['traces']['cycle_0000_crs_pickup_load_jig/trace.jsonl']
            self.assertEqual((trace['first_difference_line'], trace['differences']), (2, [['.observation.x', 1.0, 1.0000001]]))
            other_package = self.attempt(root, config='c2')
            self.assertEqual(self.run_compare(base, other_package)[0], 1)
            code, report = self.run_compare(base, other_package, '--ignore', 'config_sha256')
            self.assertEqual((code, report['ignored_keys']), (0, ['config_sha256', 'worker_pid']))
            code, report = self.run_compare(base, self.attempt(root, outcome='failure'))
            self.assertEqual((code, report['checks']['outcome']), (1, {'identical': False, 'values': ['success', 'failure']}))
            code, report = self.run_compare(base, self.attempt(root, seed=8))
            self.assertEqual((code, report['comparable']), (2, False))

    def test_pink_worker_pid_is_never_a_difference_but_its_requests_are(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            solo = self.attempt(root, ik=(41001, 1))
            code, report = self.run_compare(solo, self.attempt(root, ik=(52002, 1)))  # another run: another worker process
            self.assertEqual((code, report['bit_identical'], report['ignored_keys']), (0, True, ['worker_pid']))
            code, report = self.run_compare(solo, self.attempt(root, ik=(52002, 2)))  # one more IK request: a real difference
            trace = report['traces']['cycle_0000_crs_pickup_load_jig/trace.jsonl']
            self.assertEqual((code, trace['first_difference_line'], trace['differences']),
                             (1, 0, [['.command.upper_body_control.ik_backend.request_id', 1, 2]]))
            code, report = self.run_compare(self.attempt(root), solo)  # native-solver attempt vs Pink attempt
            self.assertEqual((code, report['traces']['cycle_0000_crs_pickup_load_jig/trace.jsonl']['differences'][0][:2]),
                             (1, ['.command.upper_body_control', '<missing>']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
