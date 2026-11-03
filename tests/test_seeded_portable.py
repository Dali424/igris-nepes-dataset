"""CPU tests on temporary inputs; no collected dataset and no native simulation."""
from copy import deepcopy
import contextlib
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

BUNDLE = Path(__file__).resolve().parents[1]
TOOLS = BUNDLE / 'scripts/seeded'
sys.path.insert(0, str(TOOLS))
import seeded_config as sampler
import merge_seeded as verifier
import collect_seeded as collector

# Stand-in for scripts/check_release.py batch_preflight (exercised in test_ik_pink_bridge.py).
IK_PLAN_FIXTURE = {'schema': 'nepes_batch_upper_body_ik_v1', 'backend': 'pink', 'python': '/fixture/ik/bin/python'}


def ik_record(**changes):
    """run.json upper_body_ik_backend of a cleanly closed Pink run of this bundle (see upper_body_ik_evidence)."""
    sources = {'ik/pink/' + name: verifier.sha(BUNDLE / 'ik/pink' / name) for name in verifier.IK_WORKER_SOURCES}
    record = {'schema': 'nepes_upper_body_ik_backend_v1', 'backend': 'pink', 'status': 'closed', 'error': None,
              'worker': {'python': '/fixture/ik/bin/python', 'pid': 4242, 'packages': {}, 'sources_sha256': sources,
                         'returncode': 0},
              'requirements': {'path': 'requirements/ik_pink.txt', 'sha256': verifier.sha(BUNDLE / 'requirements/ik_pink.txt')}}
    record.update(changes)
    return record


class PortableTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='portable-seeded-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_every_seed_same_profile_recipe_and_jitter_bounds(self):
        for pair in (2, 4):
            configs = []
            for seed in (0, 1, 100, 101, 118, 121, 122, 100000):
                config, sample = sampler.build_config(BUNDLE, pair, seed)
                configs.append(config)
                self.assertEqual(sample['xy_half_range_m'], .003 if pair == 2 else .005)
                self.assertTrue(all(abs(v) <= sample['xy_half_range_m'] for v in sample['applied_offset_xy_m']))
                self.assertEqual((config, sample), sampler.build_config(BUNDLE, pair, seed))
                for path in sampler.POSE_PATHS:
                    before = sampler.get(json.loads((BUNDLE / 'configs' / f'{sample["profile"]}.json').read_text()), path)
                    after = sampler.get(config, path)
                    self.assertEqual(before[2:], after[2:])
                    for axis in (0, 1):
                        self.assertAlmostEqual(after[axis] - before[axis], sample['applied_offset_xy_m'][axis])
                if pair == 2:
                    self.assertEqual(config['tasks']['crs_load_jig']['guarded_approach']['completion_position_m'], .0008)
                    self.assertEqual(config['tasks']['crs_load_jig']['object_symmetry'], sampler.LEFT_SYMMETRY)
            self.assertNotEqual(configs[0], configs[-1])

    def test_arbitrary_first_seed_lineage_and_legacy_rejected(self):
        _, sample = sampler.build_config(BUNDLE, 2, 10005)
        plan = {'first_seed': 10000, 'source_revisions': sampler.revision_plan(), 'xy_half_range_by_profile_m': {'left2': .003, 'right4': .005}}
        record = {'pair': 2, 'profile': 'left2', 'seed': 10005, 'index': 5, 'source_revision': sampler.lineage(sample)}
        verifier.validate_record_revision(plan, record, sample)
        record['index'] = 4
        with self.assertRaises(ValueError):
            verifier.validate_record_revision(plan, record, sample)
        config, sample = sampler.build_config(BUNDLE, 2, 0)
        sampler.materialize(self.root / 'profile', config, sample)
        verifier.validate_sample(BUNDLE, self.root)
        sample['schema'] = 'nepes_seeded_robot_initial_xy_v1'
        (self.root / 'profile/sample.json').write_text(json.dumps(sample))
        with self.assertRaisesRegex(ValueError, 'schema'):
            verifier.validate_sample(BUNDLE, self.root)

    def test_directed_and_wrong_range_rejected(self):
        with self.assertRaises(ValueError):
            sampler.build_config(BUNDLE, 2, 0, .005)
        config, sample = sampler.build_config(BUNDLE, 2, 0, offset_xy=(0, 0))
        sampler.materialize(self.root / 'profile', config, sample)
        with self.assertRaisesRegex(ValueError, 'seed-derived'):
            verifier.validate_sample(BUNDLE, self.root)

    def test_expected_physical_reasons_and_runtime_errors(self):
        config, _ = sampler.build_config(BUNDLE, 2, 0)
        for reason in verifier.PHYSICAL_REASONS:
            self.assertTrue(verifier.classify_physical_reason(reason, 'transfer', config), reason)
        for reason, stage in [('RuntimeError: progress_stalled:transfer', 'transfer'), ('stage_timeout:release', 'release'), ('posture_tracking_error:rest_posture', 'rest_posture'), ('stationary_feet_hold:left:ground_normal_force', 'lift'), ('unsafe_payload_transport:carry_posture:orientation', 'carry_posture')]:
            self.assertTrue(verifier.classify_physical_reason(reason, stage, config), reason)
        for reason in ('RuntimeError: CUDA out of memory', 'invalid_observation', 'observation_gap', 'contact_sensor_unavailable', 'pregrasp:invalid_contact_force', 'runtime_fault:reset', 'upright_command_changed', 'stationary_feet_hold:stationary_feet_anchor_missing', 'unsafe_payload_transport:invalid_measured_payload_pose', 'unsafe_payload_transport:invented:orientation'):
            self.assertFalse(verifier.classify_physical_reason(reason, 'transfer', config), reason)
        self.assertFalse(verifier.classify_physical_reason('jig_displaced', 'invented'))

    def test_exhausted_profile_does_not_prevent_other_quota(self):
        state = {'plan': {'quota_each': 1, 'max_attempts_each': 4}, 'attempts': [{'profile': 'left2', 'status': 'failed'} for _ in range(4)]}
        self.assertEqual(collector.next_profile(state), (4, 'right4'))
        state['attempts'].append({'profile': 'right4', 'status': 'accepted'})
        self.assertIsNone(collector.next_profile(state))

    def test_dryrun_arbitrary_quota_and_seed_no_outputs(self):
        # Manifest hash tested separately; this isolates plan construction from package build edits.
        for quota, first in ((1, 0), (25, 5000), (250, 10000)):
            output = self.root / f'batch{quota}'
            args = ['--bundle', str(BUNDLE), '--output', str(output), '--isaac-python', sys.executable, '--dataset-python', sys.executable, '--max-successes-per-pair', str(quota), '--first-seed', str(first), '--dry-run']
            capture = io.StringIO()
            with patch.object(collector, 'verify_bundle', return_value='fixture_manifest'), \
                    patch.object(collector, 'upper_body_ik_plan', return_value=IK_PLAN_FIXTURE) as preflight, \
                    contextlib.redirect_stdout(capture):
                self.assertEqual(collector.main(args), 0)
            result = json.loads(capture.getvalue())
            preflight.assert_called_once()
            self.assertEqual(result['plan']['upper_body_ik'], IK_PLAN_FIXTURE)  # frozen; resume compares it
            self.assertEqual(result['plan']['quota_each'], quota)
            self.assertEqual(result['plan']['first_seed'], first)
            self.assertEqual(result['plan']['max_attempts_each'], 3 * quota)
            self.assertEqual(result['plan']['xy_half_range_by_profile_m'], {'left2': .003, 'right4': .005})
            self.assertFalse(output.exists())

    def test_no_bundle_overlap(self):
        for target in (BUNDLE, BUNDLE.parent):
            with self.assertRaisesRegex(ValueError, 'overlap'):
                collector.main(['--bundle', str(BUNDLE), '--output', str(target), '--isaac-python', sys.executable, '--dataset-python', sys.executable, '--dry-run'])

    def failed_fixture(self, reason='jig_displaced'):
        config, sample = sampler.build_config(BUNDLE, 2, 0)
        sampler.materialize(self.root / 'profile', config, sample)
        verifier.compact_module(BUNDLE)
        from robros_motion_planning.stationary_catalog import config_digest, load_stationary_config
        digest = config_digest(load_stationary_config(self.root / 'profile/config.json'))
        collection = self.root / 'collection'; run = collection / 'runs/attempt_000000'
        ep = 'cycle_0000_crs_pickup_load_jig'; episode_dir = run / ep; episode_dir.mkdir(parents=True)
        result = {'episode_id': ep, 'task_id': 'crs_pickup_load_jig', 'seed': 0, 'success': False, 'full_task_completed': False, 'reason': reason, 'last_stage': 'transfer'}
        (episode_dir / 'result.json').write_text(json.dumps(result))
        report = {'schema': 'nepes_task_run_v1', 'mode': 'execute', 'complete': True, 'seed': 0, 'config_sha256': digest, 'all_episodes_succeeded': False, 'all_tasks_succeeded': False, 'episodes': [result], 'upper_body_ik_backend': ik_record()}
        (run / 'run.json').write_text(json.dumps(report))
        with gzip.open(episode_dir / 'trace.jsonl.gz', 'wb') as stream:
            stream.write(b'{"synthetic_test_fixture":true}\n')
        trace_path = episode_dir / 'trace.jsonl.gz'
        trace = {'state': 'gzipped', 'path': ep + '/trace.jsonl', 'compressed_path': ep + '/trace.jsonl.gz', 'compressed_size_bytes': trace_path.stat().st_size, 'compressed_sha256': verifier.sha(trace_path)}
        failure = 'native_process_failed_or_exit_unrecorded:2'
        receipt = {'schema': 'nepes_attempt_storage_v1', 'kind': 'failed_attempt', 'cleanup_status': 'complete', 'collection_root': str(collection), 'run_dir': str(run), 'attempt_index': 0, 'admitted_evidence': None, 'failure_reason': failure, 'trace_policy': 'gzip', 'traces': [trace]}
        (run / 'storage_receipt.json').write_text(json.dumps(receipt))
        algorithm = 'source/robros_motion_planning/robros_motion_planning/stationary_pick_load.py'
        plan = {'config_source_sha256': sample['config_sha256'], 'config_sha256': digest, 'seed': 0, 'config': str(self.root / 'profile/config.json'), 'algorithm_sha256': {algorithm: verifier.sha(BUNDLE / 'runtime/left2' / algorithm)}}
        trial = {'accepted': False, 'status': 'complete', 'seed': 0, 'returncode': 2, 'timed_out': False, 'forced_termination': False, 'stop_requested': False, 'storage_complete': True, 'failure_reason': failure, 'run_dir': str(run), 'failure_diagnostics': {'reason': reason, 'stage': 'transfer', 'episode_id': ep, 'source': str(run / 'run.json'), 'report_errors': []}, 'storage': {'kind': 'failed_attempt', 'cleanup_status': 'complete', 'receipt_path': str(run / 'storage_receipt.json'), 'receipt_sha256': verifier.sha(run / 'storage_receipt.json'), 'traces': [trace]}}
        document = {'complete': True, 'operator_stopped': False, 'plan': plan, 'plan_sha256': config_digest(plan), 'attempts': [trial]}
        (collection / 'collection.json').write_text(json.dumps(document))
        return sample

    def test_full_jig_displaced_report_passes_but_corruption_stops(self):
        sample = self.failed_fixture()
        evidence = verifier.verify_failed_attempt(BUNDLE, self.root, 2)
        self.assertEqual(evidence['reason'], 'jig_displaced')
        self.assertTrue(evidence['retryable_physical_failure'])
        self.assertEqual((evidence['upper_body_ik_backend']['backend'], evidence['upper_body_ik_backend']['status']), ('pink', 'closed'))
        for key, value in [('timed_out', True), ('returncode', -11), ('storage_complete', False)]:
            path = self.root / 'collection/collection.json'; original = path.read_bytes(); document = json.loads(original)
            document['attempts'][0][key] = value; path.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                verifier.verify_reported_physical_failure(BUNDLE, self.root, 2, sample)
            path.write_bytes(original)
        trace = self.root / 'collection/runs/attempt_000000/cycle_0000_crs_pickup_load_jig/trace.jsonl.gz'
        trace.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'trace'):
            verifier.verify_reported_physical_failure(BUNDLE, self.root, 2, sample)

    def test_failure_under_another_or_failed_ik_backend_is_not_retryable(self):
        sample = self.failed_fixture()
        run_json = self.root / 'collection/runs/attempt_000000/run.json'
        report = json.loads(run_json.read_text())
        for record, pattern in ((ik_record(backend='native'), 'not this release'),
                                (ik_record(status='failed', error='IKBackendError: worker exited'), 'did not close cleanly'),
                                (ik_record(status='ready'), 'did not close cleanly'),
                                (ik_record(requirements={'sha256': 'other'}), 'sources/requirements'),
                                (None, 'lacks the upper-body IK')):
            with self.subTest(record=record):
                run_json.write_text(json.dumps(dict(report, upper_body_ik_backend=record) if record else
                                               {key: value for key, value in report.items() if key != 'upper_body_ik_backend'}))
                with self.assertRaisesRegex(ValueError, pattern):
                    verifier.verify_reported_physical_failure(BUNDLE, self.root, 2, sample)

    def test_full_unknown_reason_cannot_be_retried(self):
        self.failed_fixture('RuntimeError: CUDA out of memory')
        with self.assertRaisesRegex(ValueError, 'Unknown or nonphysical'):
            verifier.verify_failed_attempt(BUNDLE, self.root, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
