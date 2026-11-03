#!/usr/bin/env python3
"""Small CPU-only synthetic tests; fixtures cannot pass production source gates."""
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
import av
import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
AUGMENTATION = Path(__file__).resolve().parents[1] / 'scripts/augmentation'
sys.path.insert(0, str(AUGMENTATION))
import video_common as common
import video_collect as worker
from video_common import engine

class VideoWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='seeded_video_worker_synthetic_');cls.root=Path(cls.tmp.name)
        cls.camera_keys=sorted(common.CAMERAS)
        features={'observation.state':{'dtype':'float32','shape':(2,),'names':['q0','q1']},'action':{'dtype':'float32','shape':(2,),'names':['a0','a1']},'observation.sim_time':{'dtype':'float64','shape':(1,),'names':['seconds']},common.FLAG:{'dtype':'int64','shape':(1,),'names':['is_augmented']}}
        for key in cls.camera_keys:features[key]={'dtype':'video','shape':(48,64,3),'names':['height','width','channels']}
        roots=[]
        for episode in range(2):
            source=cls.root/f'source_{episode}';roots.append(source)
            dataset=LeRobotDataset.create(repo_id=f'local/synthetic_video_{episode}',root=source,fps=30,features=features,robot_type='synthetic_test_fixture',vcodec='h264',streaming_encoding=True,encoder_threads=1,encoder_queue_maxsize=30)
            for index in range(12):
                yy,xx=np.mgrid[:48,:64];base=np.stack([(35+xx*2+index*2)%200+20,(50+yy*2+episode*7)%200+20,np.full((48,64),100+index)],axis=2).astype(np.uint8)
                dataset.add_frame({'task':f'synthetic integrity task {episode}; not robot evidence','observation.state':np.array([-0.0,index/7+episode],dtype=np.float32),'action':np.array([index/3,-index-episode],dtype=np.float32),'observation.sim_time':np.array([3.7+index/30],dtype=np.float64),common.FLAG:np.array([0],dtype=np.int64),**{key:base.copy() for key in cls.camera_keys}})
            dataset.save_episode();dataset.finalize();del dataset
        cls.source=cls.root/'merged_synthetic';aggregate_datasets(repo_ids=[f'local/synthetic_video_{i}' for i in range(2)],aggr_repo_id='local/synthetic_merged',roots=roots,aggr_root=cls.source,video_files_size_in_mb=1e-6)
        # Deliberately NOT a valid physical verification receipt: CLI must reject it.
        common.atomic_json(cls.source/engine.RECEIPT,{'schema':'synthetic_test_marker_not_production_receipt','verified':False})
        cls.layout=common.dataset_layout(cls.source,2)
        for job in cls.layout['jobs']:job['source_sha256']=common.sha(cls.source/job['path'])
        cls.plan={'schema':'synthetic_test_only','source':str(cls.source),'source_receipt_sha256':common.sha(cls.source/engine.RECEIPT),'source_files':engine.inventory(cls.source,exclude=(engine.RECEIPT,)),'source_inventory_sha256':common.digest(engine.inventory(cls.source)),'source_seeded_manifest_sha256':'synthetic_only','expected_episodes':2,'total_frames':24,'fps':30,'camera_keys':cls.camera_keys,'presets':{name:deepcopy(engine.PRESETS[name]) for name in common.PRESETS},'minimum_absolute_mean_luma_delta':.025,'tool_sha256':common.tool_hashes(),'parents':[{'episode_index':i,'motion_family':f'synthetic_parent_{i}','split_group':f'synthetic_parent_{i}'} for i in range(2)],'split_policy':'Synthetic fixture: parent and variants share group.'}
        cls.output=cls.root/'derived';cls.output.mkdir();cls.before=engine.inventory(cls.source)
        cls.events=[]
        av.logging.set_level(av.logging.ERROR)
        for preset in common.PRESETS:
            worker.process_preset(cls.source,cls.output,cls.plan,cls.layout,preset,2,[False],lambda *items:cls.events.append(items))
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_two_datasets_official_getitem_numeric_tasks_and_four_images(self):
        original=LeRobotDataset(repo_id='local/synthetic_parent',root=self.source,video_backend='pyav')
        for preset in common.PRESETS:
            derived=LeRobotDataset(repo_id='local/synthetic_derived',root=self.output/preset,video_backend='pyav')
            self.assertEqual(derived.num_episodes,2)
            for index in (0,11,12,23):
                a,b=original[index],derived[index];self.assertEqual(b[common.FLAG].item(),1)
                for key in ('observation.state','action','timestamp','observation.sim_time','task_index'):
                    self.assertEqual(a[key].numpy().tobytes(),b[key].numpy().tobytes(),key)
                for key in self.camera_keys:self.assertFalse(np.array_equal(a[key].numpy(),b[key].numpy()))
            self.assertEqual(common.sha(self.source/'meta/tasks.parquet'),common.sha(self.output/preset/'meta/tasks.parquet'))
    def test_four_cameras_visible_change_and_true_decoded_stats(self):
        for preset in common.PRESETS:
            target=self.output/preset;audit=common.read(target/common.AUDIT);stats=common.read(target/'meta/stats.json')
            for key in self.camera_keys:
                pixels=[]
                for record in audit['video_records']:
                    if record['job']['camera_key']!=key:continue
                    self.assertEqual(record['audit']['decoded_frames'],12)
                    direction=1 if 'brighter' in preset else -1
                    self.assertGreaterEqual(direction*record['audit']['mean_luma_delta'],.025)
                    with av.open(str(target/record['job']['path'])) as container:
                        pixels.extend(frame.to_ndarray(format='rgb24').reshape(-1,3) for frame in container.decode(video=0))
                values=np.concatenate(pixels)/255
                np.testing.assert_allclose(np.asarray(stats[key]['mean']).reshape(3),values.mean(axis=0),atol=1e-12)
                np.testing.assert_allclose(np.asarray(stats[key]['std']).reshape(3),values.std(axis=0),atol=1e-12)
    def test_source_unchanged_and_no_copied_physics_receipts(self):
        self.assertEqual(engine.inventory(self.source),self.before)
        for preset in common.PRESETS:
            self.assertFalse((self.output/preset/engine.RECEIPT).exists())
            self.assertFalse((self.output/preset/'meta/nepes_storage_sources.json').exists())
            self.assertEqual(common.read(self.output/preset/common.PROVENANCE)['parents'],self.plan['parents'])
    def test_final_resume_revalidates_and_preserves_output_bytes(self):
        for preset in common.PRESETS:
            before=engine.inventory(self.output/preset)
            worker.process_preset(self.source,self.output,self.plan,self.layout,preset,1,[False],lambda *args:None)
            self.assertEqual(engine.inventory(self.output/preset),before)
    def test_corrupted_output_rejected_even_with_complete_audit(self):
        target=self.output/common.PRESETS[0];path=target/self.layout['jobs'][0]['path'];original=path.read_bytes()
        try:
            path.write_bytes(original+b'corrupt')
            with self.assertRaisesRegex(ValueError,'changed after verification'):common.verify_final(self.source,target,self.plan,common.PRESETS[0],self.layout)
        finally:path.write_bytes(original)
    def test_source_inventory_corruption_rejected(self):
        expected=engine.inventory(self.source,exclude=(engine.RECEIPT,));path=self.source/'meta/tasks.parquet';original=path.read_bytes()
        try:
            path.write_bytes(original+b'corrupt')
            with self.assertRaisesRegex(ValueError,'verified inventory'):common.check_inventory(self.source,expected)
        finally:path.write_bytes(original)
    def test_cli_rejects_synthetic_source_without_output(self):
        output=self.root/'must_not_be_created'
        result=subprocess.run([sys.executable,'-B',str(Path(worker.__file__)),'--source',str(self.source),'--output',str(output),'--expected-episodes','2','--preflight'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertFalse(output.exists())
    def test_incomplete_and_wrong_pair_source_declarations_rejected(self):
        info={'total_episodes':2,'total_frames':24};receipt={'schema':'nepes_verified_lerobot_storage_v1','verified':True,'kind':'merged_dataset','total_episodes':2,'total_frames':24,'checks':{key:True for key in ['all_mp4_fully_decoded','parquet_metadata_consistent','camera_schema_consistent','source_binding_consistent','video_bytes_preserved_without_reencoding','episode_numeric_payloads_preserved']},'camera_keys':self.camera_keys,'fps':30}
        manifest={'schema':'nepes_seeded_collection_aggregate_v1','total_episodes':2,'expected_per_pair':1,'episodes':[{'episode_index':i,'profile':profile,'seed':101,'config_sha256':f'config{i}','motion_quantized_sha256':f'motion{i}'} for i,profile in enumerate(('left2','right4'))]};extension={'episodes':[{'episode_index':i,'success':True,'provenance':{'kind':'original','seed':101,'config_sha256':f'config{i}'}} for i in range(2)]}
        # Pure in-memory contract test; no valid production receipt is ever issued to the synthetic fixture.
        common.check_source_declarations(info,receipt,extension,manifest,2)
        partial=deepcopy(receipt);partial['total_episodes']=1
        with self.assertRaisesRegex(ValueError,'Incomplete'):common.check_source_declarations(info,partial,extension,manifest,2)
        wrong=deepcopy(manifest);wrong['episodes'][1]['profile']='left2'
        with self.assertRaisesRegex(ValueError,'balanced'):common.check_source_declarations(info,receipt,extension,wrong,2)
    def test_interrupted_partial_video_resume(self):
        output=self.root/'partial_resume';output.mkdir();stop=[False]
        def checkpoint(preset,status,count,*rest):
            if status=='encoding' and count>=1:stop[0]=True
        with self.assertRaises(InterruptedError):worker.process_preset(self.source,output,self.plan,self.layout,common.PRESETS[0],1,stop,checkpoint)
        orphan=output/('.'+common.PRESETS[0]+'.work')/'videos'/'.file-000.encoding-killed.mp4'
        orphan.parent.mkdir(exist_ok=True);orphan.write_bytes(b'incomplete private encode')
        checkpoint_path=output/'checkpoints'/common.PRESETS[0]/'video_000000.json';first=common.read(checkpoint_path);first_hash=first['audit']['output_sha256']
        worker.process_preset(self.source,output,self.plan,self.layout,common.PRESETS[0],1,[False],lambda *args:None)
        self.assertEqual(common.sha(output/common.PRESETS[0]/first['job']['path']),first_hash)
        self.assertFalse(orphan.exists())
        common.verify_final(self.source,output/common.PRESETS[0],self.plan,common.PRESETS[0],self.layout)
    def test_wrong_video_frame_count_and_noop_brightness_fail(self):
        job=deepcopy(self.layout['jobs'][0]);job['frames']+=1
        with self.assertRaises(ValueError):worker.process_video(self.source,self.root/'bad_count',job,self.plan['presets'][common.PRESETS[0]],.025)
        recipe=deepcopy(self.plan['presets'][common.PRESETS[0]]);recipe['brightness_odds_gain']=1;recipe['logit_contrast']=1
        with self.assertRaisesRegex(ValueError,'brightness'):worker.process_video(self.source,self.root/'no_change',self.layout['jobs'][0],recipe,.025)

if __name__=='__main__':unittest.main(verbosity=2)
