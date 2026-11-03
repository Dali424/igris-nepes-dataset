"""CPU regression checks for mixed original campaign admission (no simulator)."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('campaign_sources_tested', ROOT / 'scripts/campaign_sources.py')
C = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(C)


def entry(tag, profile='left2', schema='nepes_seeded_attempt_verification_v1'):
    e = {'verified': True, 'schema': schema, 'profile': profile, 'pair': C.PROFILES[profile],
         'attempt': tag, 'archive': tag + '/archive', 'capture_schema': {'fps': 30},
         'motion_exact_sha256': tag + '-motion', 'motion_quantized_sha256': tag + '-qmotion',
         'columns': {c: {'exact_sha256': tag + c, 'quantized_sha256': tag + c + 'q'} for c in ('observation.state', 'action')},
         'startup_reset': {'xy_signature_sha256': tag + '-reset'}, 'initial_measurement': {'xy_signature_sha256': tag + '-initial'}}
    return {'evidence': e, 'features': {'observation.state': {'shape': [33]}}, 'verification_sha256': tag + '-evidence'}


class CampaignSourcesTests(unittest.TestCase):
    def test_mixed_legacy_and_current_evidence_preserved(self):
        rows = [entry('old'), entry('new', 'right4', 'nepes_seeded_attempt_verification_v3')]; before = deepcopy(rows)
        self.assertEqual(C.unique_compatible(rows, {'left2': 1, 'right4': 1}), {'left2': 1, 'right4': 1})
        self.assertEqual(rows, before)

    def test_quotas_must_be_exact(self):
        with self.assertRaisesRegex(ValueError, 'Exact source quota'):
            C.unique_compatible([entry('a')], {'left2': 1, 'right4': 1})

    def test_state_and_action_independent_duplicate_guards(self):
        for column in ('observation.state', 'action'):
            for key in ('exact_sha256', 'quantized_sha256'):
                a, b = entry('a'), entry('b'); b['evidence']['columns'][column][key] = a['evidence']['columns'][column][key]
                with self.subTest(column=column, key=key), self.assertRaisesRegex(ValueError, 'Duplicate'):
                    C.unique_compatible([a, b])

    def test_combined_motion_and_measured_xy_duplicate_guards(self):
        for key in ('motion_exact_sha256', 'motion_quantized_sha256', 'startup_reset', 'initial_measurement'):
            a, b = entry('a'), entry('b'); b['evidence'][key] = a['evidence'][key]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Duplicate'):
                C.unique_compatible([a, b])

    def test_incompatible_capture_or_features_rejected(self):
        for key in ('capture', 'features'):
            a, b = entry('a'), entry('b')
            if key == 'capture': b['evidence']['capture_schema']['fps'] = 15
            else: b['features']['observation.state']['shape'] = [34]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Incompatible'):
                C.unique_compatible([a, b])

    def test_duplicate_archive_and_evidence_rejected(self):
        for key in ('archive', 'verification_sha256'):
            a, b = entry('a'), entry('b')
            if key == 'archive': b['evidence'][key] = a['evidence'][key]
            else: b[key] = a[key]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Duplicate source'):
                C.unique_compatible([a, b])

    def test_output_cannot_overlap_source_or_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = root / 'source'; source.mkdir()
            for output in (root, source, source / 'audit.json'):
                with self.assertRaisesRegex(ValueError, 'overlaps'):
                    C.no_overlap(output, [source])
            C.no_overlap(root / 'out', [source])

    def test_atomic_resume_audit_refuses_changed_content(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'audit.json'; C.stable_write(p, {'verified': True}); before = p.read_bytes()
            C.stable_write(p, {'verified': True}); self.assertEqual(p.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, 'differs'): C.stable_write(p, {'verified': False})
            self.assertEqual(p.read_bytes(), before)

    def test_bundle_changed_byte_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); f = root / 'code.py'; f.write_text('print(1)\n')
            m = root / 'manifest.json'; m.write_text(json.dumps({'files': {'code.py': {'sha256': C.sha(f)}}}))
            C.bundle_integrity(root, C.sha(m)); f.write_text('print(2)\n')
            with self.assertRaisesRegex(ValueError, 'file changed'): C.bundle_integrity(root, C.sha(m))

    def test_round_robin_new_batch_matches_grouped_merged_order(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); bundle = root / 'bundle'; bundle.mkdir(); batch = root / 'batch'; batch.mkdir()
            records = []
            for profile, index in [('left2', 0), ('right4', 0), ('left2', 1), ('right4', 1)]:
                a = batch / profile / str(index); (a / 'profile').mkdir(parents=True)
                (a / 'profile/sample.json').write_text(json.dumps({'schema': 'nepes_seeded_robot_initial_xy_v3'}))
                (a / 'verification.json').write_text(json.dumps({'archive': str(a / 'archive'), 'archive_receipt_sha256': 'receipt'}))
                records.append({'directory': str(a), 'status': 'accepted', 'profile': profile, 'pair': C.PROFILES[profile], 'seed': 501 + index, 'verification_sha256': 'evidence'})
            state = {'status': 'complete', 'plan': {'bundle': str(bundle), 'quota_by_profile': {'left2': 2, 'right4': 2}},
                     'counts': {'left2': 2, 'right4': 2}, 'attempts': records}
            (batch / 'batch.json').write_text(json.dumps(state))
            _, rows, _ = C.new_rows(batch, bundle)
            self.assertEqual([(r['profile'], r['seed']) for r in rows], [('left2', 501), ('left2', 502), ('right4', 501), ('right4', 502)])
            state['status'] = 'collecting'; (batch / 'batch.json').write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, 'must be complete'): C.new_rows(batch, bundle)

    def test_legacy_baseline_record_annotation_uses_source_policy_without_upgrading_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            verifier = Path(td) / 'merge_seeded.py'
            verifier.write_text('''def validate_record_revision(plan,record,sample):
    expected={"sample_schema":sample["schema"],"recipe_id":"baseline","source_config_sha256":sample["source_config_sha256"]}
    if plan.get("policy")!="legacy" or record.get("source_revision")!=expected:
        raise ValueError("Original record policy rejected")
    return expected
''')
            sample = {'schema': 'nepes_seeded_robot_initial_xy_v1', 'source_config_sha256': 'original-config'}
            lineage = {'sample_schema': sample['schema'], 'recipe_id': 'baseline', 'source_config_sha256': 'original-config'}
            record = {'source_revision': lineage}; evidence = {'schema': 'nepes_seeded_attempt_verification_v1'}
            before = deepcopy(evidence)
            self.assertEqual(C.validate_source_revision(verifier, {'policy': 'legacy'}, record, sample, evidence), lineage)
            self.assertEqual(evidence, before)
            bad = deepcopy(record); bad['source_revision']['source_config_sha256'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'Original source revision policy rejected'):
                C.validate_source_revision(verifier, {'policy': 'legacy'}, bad, sample, evidence)

    def test_current_evidence_revision_must_match_original_policy(self):
        with tempfile.TemporaryDirectory() as td:
            verifier = Path(td) / 'merge_seeded.py'
            verifier.write_text('def validate_record_revision(plan,record,sample):\n    return record["source_revision"]\n')
            sample = {'schema': 'nepes_seeded_robot_initial_xy_v3'}
            lineage = {'sample_schema': sample['schema'], 'recipe_id': 'current'}
            evidence = {'schema': 'nepes_seeded_attempt_verification_v3', 'source_revision': lineage}
            self.assertEqual(C.validate_source_revision(verifier, {}, {'source_revision': lineage}, sample, evidence), lineage)
            for revision in (None, {'sample_schema': sample['schema'], 'recipe_id': 'changed'}):
                with self.subTest(revision=revision), self.assertRaisesRegex(ValueError, 'revision differs'):
                    C.validate_source_revision(verifier, {}, {'source_revision': lineage}, sample, {**evidence, 'source_revision': revision})


if __name__ == '__main__': unittest.main()
