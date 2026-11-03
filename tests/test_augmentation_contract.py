"""CPU-only temporary schema fixtures, never eligible as physical datasets."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import sys

AUGMENTATION = Path(__file__).resolve().parents[1] / 'scripts/augmentation'
sys.path.insert(0, str(AUGMENTATION))
import source_selection as select
import video_common
import render_collect
import render_verify
import render_merge


class PortableTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='nepes_portable_contract_')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.batch = self.root/'batch'; self.batch.mkdir()
        self.bundle = self.root/'bundle'; self.bundle.mkdir(); self.save(self.bundle/'manifest.json', {'synthetic_fixture_only':True})
        self.dataset = self.batch/'dataset_6'; self.dataset.mkdir()
        self.save(self.dataset/select.RECEIPT, {'verified':True,'kind':'merged_dataset','total_episodes':6,
                                             'schema':'synthetic_fixture_not_native_verification'})
        records=[]; mapping=[]
        for profile in ('left2','right4'):
            for seed in range(3):
                attempt=self.batch/profile/f'attempt_{seed}';(attempt/'profile').mkdir(parents=True)
                self.save(attempt/'profile/config.json',{'synthetic_fixture_only':True,'profile':profile,'seed':seed})
                archive=attempt/'archive';archive.mkdir();self.save(archive/select.RECEIPT,{'synthetic_fixture_only':True})
                self.save(attempt/'verification.json',{'profile':profile,'seed':seed,'archive':str(archive),'archive_receipt_sha256':select.sha(archive/select.RECEIPT)})
                digest=select.sha(attempt/'verification.json')
                records.append({'profile':profile,'seed':seed,'directory':str(attempt),'status':'accepted','verification_sha256':digest})
                mapping.append({'episode_index':len(mapping),'profile':profile,'seed':seed,'verification_sha256':digest,'motion_exact_sha256':f'{profile}_{seed}'})
        self.save(self.dataset/'meta/nepes_stationary_collection.json',{'schema':'nepes_seeded_collection_aggregate_v1','total_episodes':6,'expected_per_pair':3,'episodes':mapping})
        self.state={'schema':'nepes_seeded_batch_v1','status':'complete','plan':{'bundle':str(self.bundle),'output':str(self.batch),'quota_each':3,'bundle_manifest_sha256':select.sha(self.bundle/'manifest.json')},'counts':{'left2':3,'right4':3},'attempts':records,'dataset':str(self.dataset)}
        self.save(self.batch/'batch.json',self.state)

    def save(self,path,document):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(document))

    def test_selection_count_and_round_robin_arbitrary_small_seeds(self):
        for count in (1,3,5,6):
            source=select.complete_source(self.batch,self.bundle,count)
            self.assertEqual(source['selected_episodes'],count)
            self.assertEqual([r['profile'] for r in source['parents']],(['left2','right4']*3)[:count])
            self.assertEqual(source['expected_profile_counts'],{'left2':(count+1)//2,'right4':count//2})
            self.assertEqual(source['parents'][0]['seed'],0)
        for count in (0,7,-1):
            with self.assertRaises(ValueError):select.complete_source(self.batch,self.bundle,count)

    def test_incomplete_source_and_changed_evidence_rejected(self):
        bad=deepcopy(self.state);bad['status']='collecting';self.save(self.batch/'batch.json',bad)
        with self.assertRaisesRegex(ValueError,'Finish'):select.complete_source(self.batch,self.bundle)
        self.save(self.batch/'batch.json',self.state)
        first=Path(self.state['attempts'][0]['directory'])/'verification.json';first.write_text('{}')
        with self.assertRaisesRegex(ValueError,'verification changed'):select.complete_source(self.batch,self.bundle)

    def test_one_episode_selection_declaration_accepts_zero_other_profile(self):
        info={'total_episodes':1,'total_frames':12}
        receipt={'schema':'nepes_verified_lerobot_storage_v1','verified':True,'kind':'merged_dataset','total_episodes':1,'total_frames':12,'checks':{key:True for key in ('all_mp4_fully_decoded','parquet_metadata_consistent','camera_schema_consistent','source_binding_consistent','video_bytes_preserved_without_reencoding','episode_numeric_payloads_preserved')},'camera_keys':list(video_common.CAMERAS),'fps':30}
        manifest={'schema':'nepes_augmentation_parent_selection_v1','total_episodes':1,'profile_counts':{'left2':1,'right4':0},'selection_from_verified_complete_batch':True,'parent_merge_receipt_sha256':'synthetic','episodes':[{'episode_index':0,'profile':'left2','seed':0,'config_sha256':'config','motion_quantized_sha256':'motion'}]}
        extension={'episodes':[{'episode_index':0,'success':True,'provenance':{'kind':'original','seed':0,'config_sha256':'config'}}]}
        video_common.check_source_declarations(info,receipt,extension,manifest,1)
        manifest['profile_counts']['right4']=1
        with self.assertRaisesRegex(ValueError,'selection lineage'):video_common.check_source_declarations(info,receipt,extension,manifest,1)

    def test_native_single_attempt_and_no_consecutive_failure_branch(self):
        plan={'bundle':str(self.bundle),'isaac_python':'python','dataset_python':'python'}
        argv=render_collect.command(plan,{'profile':'left2','seed':0},self.root/'child')
        self.assertEqual(argv[argv.index('--max-attempts')+1],'1')
        self.assertNotIn('--max-consecutive-failures',argv)
        text=Path(render_collect.__file__).read_text()
        self.assertNotIn('failure_streak',text)
        self.assertIn("len(job['attempts']) < plan['max_attempts_per_parent']",text)
        self.assertIn('classify_failure(job, trial)',text)

    def test_portable_scripts_no_old_machine_paths_or_fixed_quota(self):
        for path in Path(render_collect.__file__).parent.glob('*.py'):
            content=path.read_text()
            for forbidden in (str(Path.home()) + '/', 'dataset_50', 'tools_v4', 'SOURCE_FREEZE_SHA256'):
                self.assertNotIn(forbidden,content,path.name)

    def test_external_bundle_must_be_exact_source_and_owns_verifier(self):
        wrong=self.root/'wrong_bundle';wrong.mkdir()
        with self.assertRaisesRegex(ValueError,'location/bundle'):
            select.complete_source(self.batch,wrong,1)
        module=self.bundle/'scripts/seeded/test_marker.py';module.parent.mkdir(parents=True)
        module.write_text("OWNER = 'source_bundle_not_worker_directory'\n")
        loaded=select.seeded_module('test_marker',self.bundle)
        self.assertEqual(loaded.OWNER,'source_bundle_not_worker_directory')


if __name__=='__main__':unittest.main(verbosity=2)
