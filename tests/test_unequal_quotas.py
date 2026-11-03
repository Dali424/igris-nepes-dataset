"""CPU-only unequal-quota orchestration and source-ledger tests."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / 'scripts/seeded'))
sys.path.insert(0, str(BUNDLE / 'scripts/augmentation'))
import collect_seeded as collector
import merge_seeded as merger
import source_selection as selection
from seeded_process_fixture import Fixture


class UnequalQuotaTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='nepes-unequal-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def arguments(self):
        return ['--bundle', str(BUNDLE), '--output', str(self.root / 'output'),
                '--isaac-python', sys.executable, '--dataset-python', sys.executable,
                '--first-seed', '10000', '--dry-run']

    def dry_plan(self, *extra):
        capture = io.StringIO()
        with patch.object(collector, 'verify_bundle', return_value='fixture_manifest'), \
                patch.object(collector, 'upper_body_ik_plan', return_value=None), contextlib.redirect_stdout(capture):
            self.assertEqual(collector.main(self.arguments() + list(extra)), 0)
        self.assertFalse((self.root / 'output').exists())
        return json.loads(capture.getvalue())['plan']

    def test_unequal_default_budget_and_common_explicit_budget(self):
        plan = self.dry_plan('--left-successes', '86', '--right-successes', '79')
        self.assertEqual(plan['quota_by_profile'], {'left2': 86, 'right4': 79})
        self.assertEqual(plan['max_attempts_by_profile'], {'left2': 258, 'right4': 237})
        self.assertNotIn('quota_each', plan)
        self.assertNotIn('max_attempts_each', plan)
        explicit = self.dry_plan('--left-successes', '86', '--right-successes', '79', '--max-attempts-per-pair', '300')
        self.assertEqual(explicit['max_attempts_by_profile'], {'left2': 300, 'right4': 300})

    def test_equal_plan_retains_legacy_fields(self):
        default = self.dry_plan()
        explicit = self.dry_plan('--max-successes-per-pair', '25')
        paired = self.dry_plan('--left-successes', '25', '--right-successes', '25')
        self.assertEqual(default, explicit)
        self.assertEqual(default, paired)
        self.assertEqual(default['quota_each'], 25)
        self.assertNotIn('quota_by_profile', default)

    def test_invalid_quota_combinations_rejected(self):
        for flags in [('--left-successes', '2'), ('--right-successes', '1'),
                      ('--left-successes', '2', '--right-successes', '1', '--max-successes-per-pair', '3'),
                      ('--left-successes', '0', '--right-successes', '1'),
                      ('--left-successes', '86', '--right-successes', '79', '--max-attempts-per-pair', '80')]:
            with self.subTest(flags=flags), self.assertRaises((ValueError, SystemExit)):
                self.dry_plan(*flags)

    def test_scheduler_honors_each_budget_and_quota(self):
        plan = {'quota_by_profile': {'left2': 2, 'right4': 1},
                'max_attempts_by_profile': {'left2': 4, 'right4': 3}}
        state = {'plan': plan, 'attempts': [{'profile': 'left2', 'status': 'accepted'},
                                         {'profile': 'right4', 'status': 'accepted'}]}
        self.assertEqual(collector.next_profile(state), (2, 'left2'))
        state['attempts'].append({'profile': 'left2', 'status': 'accepted'})
        self.assertIsNone(collector.next_profile(state))
        state['attempts'] = [{'profile': 'left2', 'status': 'failed'} for _ in range(4)]
        self.assertEqual(collector.next_profile(state), (4, 'right4'))
        state['attempts'].append({'profile': 'right4', 'status': 'accepted'})
        self.assertIsNone(collector.next_profile(state))

    def test_subprocess_checkpoint_resume_and_quota_binding(self):
        fixture = Fixture(self.root, Path(collector.__file__), 'process')
        self.addCleanup(fixture.close)
        index = fixture.command.index('--max-successes-per-pair')
        del fixture.command[index:index + 2]
        fixture.command += ['--left-successes', '2', '--right-successes', '1',
                            '--stop-after-total-successes', '2']
        state = fixture.wait(fixture.start(), 0)
        self.assertEqual(state['status'], 'checkpoint')
        self.assertEqual(state['counts'], {'left2': 1, 'right4': 1})
        del fixture.command[-2:]
        index = fixture.command.index('--left-successes') + 1
        fixture.command[index] = '3'
        fixture.wait(fixture.start(resume=True), 1)
        self.assertEqual(len([event for event in fixture.events() if event['kind'] == 'native_started']), 2)
        fixture.command[index] = '2'
        state = fixture.wait(fixture.start(resume=True), 0)
        self.assertEqual(state['status'], 'complete')
        self.assertEqual(state['counts'], {'left2': 2, 'right4': 1})
        self.assertEqual(Path(state['dataset']).name, 'dataset_3')
        self.assertEqual([(event['pair'], event['seed']) for event in fixture.events() if event['kind'] == 'native_started'],
                         [(2, 0), (4, 0), (2, 1)])

    def test_merge_expectations_accept_only_exact_quota_schema(self):
        plan = {'quota_by_profile': {'left2': 86, 'right4': 79}}
        args = SimpleNamespace(expected_per_pair=None, expected_left=86, expected_right=79)
        self.assertEqual(merger.requested_quotas(args), merger.plan_quotas(plan))
        self.assertEqual(merger.quota_arguments(plan), ['--expected-left', '86', '--expected-right', '79'])
        self.assertEqual(selection.source_quota_arguments(plan), merger.quota_arguments(plan))
        self.assertEqual(merger.quota_arguments({'quota_each': 25}), ['--expected-per-pair', '25'])
        for bad in ({}, {'quota_each': 25, **plan}, {'quota_by_profile': {'left2': 86}},
                    {'quota_by_profile': {'left2': True, 'right4': 79}}):
            with self.subTest(plan=bad), self.assertRaises(ValueError):
                merger.plan_quotas(bad)
        for bad in [SimpleNamespace(expected_per_pair=None, expected_left=86, expected_right=None),
                    SimpleNamespace(expected_per_pair=25, expected_left=86, expected_right=79),
                    SimpleNamespace(expected_per_pair=None, expected_left=None, expected_right=None)]:
            with self.assertRaises(ValueError):
                merger.requested_quotas(bad)

    def test_public_entrypoint_forwards_unequal_flags(self):
        spec = importlib.util.spec_from_file_location('quota_public_collect', BUNDLE / 'scripts/collect.py')
        public = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(public)
        with patch.object(public.os, 'execve') as execute:
            public.main(['--left-successes', '86', '--right-successes', '79', '--dry-run'])
        command = execute.call_args.args[1]
        self.assertEqual(command[command.index('--left-successes') + 1], '86')
        self.assertEqual(command[command.index('--right-successes') + 1], '79')
        self.assertNotIn('--max-successes-per-pair', command)
        self.assertIn('--dry-run', command)

    @staticmethod
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def completed_source(self, quotas):
        bundle, batch = self.root / 'bundle', self.root / 'batch'
        self.save(bundle / 'manifest.json', {'synthetic_fixture_only': True})
        dataset = batch / f'dataset_{sum(quotas.values())}'
        self.save(dataset / selection.RECEIPT,
                  {'verified': True, 'kind': 'merged_dataset', 'total_episodes': sum(quotas.values())})
        records, mapping = [], []
        for profile, quota in quotas.items():
            for seed in range(quota):
                attempt = batch / profile / f'attempt_{seed}'
                self.save(attempt / 'profile/config.json', {'synthetic': True, 'profile': profile, 'seed': seed})
                archive = attempt / 'archive'
                self.save(archive / selection.RECEIPT,
                          {'synthetic_fixture_only': True, 'episode_payload_sha256': {'0': f'{profile}_{seed}'}})
                self.save(attempt / 'verification.json',
                          {'profile': profile, 'seed': seed, 'archive': str(archive),
                           'archive_receipt_sha256': selection.sha(archive / selection.RECEIPT)})
                digest = selection.sha(attempt / 'verification.json')
                records.append({'profile': profile, 'seed': seed, 'directory': str(attempt),
                                'status': 'accepted', 'verification_sha256': digest})
                mapping.append({'episode_index': len(mapping), 'profile': profile, 'seed': seed,
                                'verification_sha256': digest, 'motion_exact_sha256': f'{profile}_{seed}'})
        self.save(dataset / 'meta/nepes_stationary_collection.json',
                  {'schema': 'nepes_seeded_collection_aggregate_v1', 'total_episodes': len(records),
                   'expected_by_profile': quotas, 'episodes': mapping})
        state = {'schema': 'nepes_seeded_batch_v1', 'status': 'complete',
                 'plan': {'bundle': str(bundle), 'output': str(batch), 'quota_by_profile': quotas,
                          'bundle_manifest_sha256': selection.sha(bundle / 'manifest.json')},
                 'counts': quotas, 'attempts': records, 'dataset': str(dataset)}
        self.save(batch / 'batch.json', state)
        return bundle, batch, state

    def test_source_165_selects_balanced_100_and_preserves_remainder(self):
        quotas = {'left2': 86, 'right4': 79}
        bundle, batch, state = self.completed_source(quotas)
        selected = selection.complete_source(batch, bundle, 100)
        self.assertEqual(selected['source_quota_by_profile'], quotas)
        self.assertEqual(selected['source_total_episodes'], 165)
        self.assertEqual(selected['expected_profile_counts'], {'left2': 50, 'right4': 50})
        self.assertEqual([row['profile'] for row in selected['parents']], ['left2', 'right4'] * 50)
        complete = selection.complete_source(batch, bundle)
        self.assertEqual(len(complete['parents']), 165)
        self.assertEqual(complete['expected_profile_counts'], quotas)
        self.assertEqual([row['profile'] for row in complete['parents'][-7:]], ['left2'] * 7)
        self.assertEqual(len({row['parent_verification_sha256'] for row in complete['parents']}), 165)
        state['counts'] = {'left2': 85, 'right4': 80}
        self.save(batch / 'batch.json', state)
        with self.assertRaisesRegex(ValueError, 'count ledger'):
            selection.complete_source(batch, bundle, 100)

    def test_full_unequal_video_source_is_declared_selection(self):
        bundle, batch, _ = self.completed_source({'left2': 2, 'right4': 1})
        source = selection.complete_source(batch, bundle)
        receipt = {'verified': True, 'total_episodes': 3,
                   'episode_payload_sha256': {str(i): row['source_episode']['motion_exact_sha256']
                                              for i, row in enumerate(source['parents'])}}
        compact = SimpleNamespace(RECEIPT_PATH=selection.RECEIPT,
                                  merge_verified_shards=Mock(return_value=receipt))
        # This fixture checks the subset/full-unequal manifest branch, not MP4 integrity.
        with patch.dict(sys.modules, {'robros_motion_planning': SimpleNamespace(compact_dataset=compact)}):
            target = selection.selected_video_source(bundle, source, self.root / 'selection')
        self.assertNotEqual(target, Path(source['parent_merged_dataset']))
        compact.merge_verified_shards.assert_called_once()
        manifest = json.loads((self.root / 'selection.sources.json').read_text())
        self.assertEqual(manifest['schema'], 'nepes_augmentation_parent_selection_v1')
        self.assertEqual(manifest['profile_counts'], {'left2': 2, 'right4': 1})
        self.assertEqual(manifest['total_episodes'], 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
