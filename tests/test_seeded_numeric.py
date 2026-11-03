"""CPU-only tests; temporary numerical fixtures are NOT collected episodes."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BUNDLE=Path(__file__).resolve().parents[1]
TOOLS=BUNDLE/'scripts/seeded'
sys.path.insert(0,str(TOOLS))
import collect_seeded as collector
import merge_seeded as verifier
import seeded_config as sampler

class SeededToolsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='nepes_seeded_unit_');self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name)
    def shard(self,name,states=None,actions=None,frame=None,seed=999):
        path=self.base/name;(path/'data').mkdir(parents=True)
        states=states if states is not None else [[.10001,.20001],[.30001,.40001]]
        actions=actions if actions is not None else [[.50001,.60001],[.70001,.80001]]
        table=pa.table({'observation.state':states,'action':actions,'frame_index':frame if frame is not None else [0,1], 'episode_index':[0,0],'seed':[seed,seed]})
        pq.write_table(table,path/'data/file.parquet');return path
    def evidence(self,path,seed=101,pose='pose1'):
        return {**verifier.numeric_signature(path),'profile':'left2','seed':seed,'startup_reset':{'xy_signature_sha256':pose},'initial_measurement':{'xy_signature_sha256':pose}}
    def test_numeric_ignores_seed_and_metadata(self):
        a=self.evidence(self.shard('a'));b=self.evidence(self.shard('b',seed=1000),102,'pose2')
        self.assertEqual(a['motion_exact_sha256'],b['motion_exact_sha256'])
        with self.assertRaisesRegex(ValueError,'Duplicate'):verifier.reject_duplicates([a],b)
    def test_same_action_is_rejected_even_if_state_changes(self):
        a=self.evidence(self.shard('a'));b=self.evidence(self.shard('b',states=[[.2,.3],[.4,.5]]),102,'pose2')
        with self.assertRaisesRegex(ValueError,'Duplicate action'):verifier.reject_duplicates([a],b)
    def test_same_state_is_rejected_even_if_action_changes(self):
        a=self.evidence(self.shard('a'));b=self.evidence(self.shard('b',actions=[[.6,.7],[.8,.9]]),102,'pose2')
        with self.assertRaisesRegex(ValueError,'Duplicate observation.state'):verifier.reject_duplicates([a],b)
    def test_quantized_microchanges_rejected(self):
        a=self.evidence(self.shard('a'));b=self.evidence(self.shard('b',states=[[.100011,.200011],[.300011,.400011]],actions=[[.500011,.600011],[.700011,.800011]]),102,'pose2')
        self.assertNotEqual(a['motion_exact_sha256'],b['motion_exact_sha256'])
        with self.assertRaisesRegex(ValueError,'quantized'):verifier.reject_duplicates([a],b)
    def test_different_numeric_motion_and_pose_accepts(self):
        a=self.evidence(self.shard('a'));b=self.evidence(self.shard('b',states=[[.2,.3],[.4,.5]],actions=[[.6,.7],[.8,.9]]),102,'pose2')
        verifier.reject_duplicates([a],b)
    def test_identical_measured_pose_rejected(self):
        a=self.evidence(self.shard('a'));b=self.evidence(self.shard('b',states=[[.2,.3],[.4,.5]],actions=[[.6,.7],[.8,.9]]),102)
        with self.assertRaisesRegex(ValueError,'measured'):verifier.reject_duplicates([a],b)
    def test_frame_order_is_canonical(self):
        a=verifier.numeric_signature(self.shard('a'));b=verifier.numeric_signature(self.shard('b',states=[[.30001,.40001],[.10001,.20001]],actions=[[.70001,.80001],[.50001,.60001]],frame=[1,0]))
        self.assertEqual(a,b)
    def test_bad_frame_indices_rejected(self):
        with self.assertRaisesRegex(ValueError,'frame indices'):verifier.numeric_signature(self.shard('a',frame=[0,0]))
    def test_nonfinite_vector_rejected(self):
        with self.assertRaisesRegex(ValueError,'Invalid numeric'):verifier.numeric_signature(self.shard('a',states=[[np.nan,0],[0,0]]))
    def test_sample_rebuild_and_tamper(self):
        config,sample=sampler.build_config(BUNDLE,2,0);sampler.materialize(self.base/'profile',config,sample)
        verifier.validate_sample(BUNDLE,self.base)
        sample['applied_offset_xy_m'][0]+=.001;(self.base/'profile/sample.json').write_text(json.dumps(sample))
        with self.assertRaisesRegex(ValueError,'deterministic sampler'):verifier.validate_sample(BUNDLE,self.base)
    def test_round_robin_and_quota(self):
        state={'plan':{'quota_each':2},'attempts':[]}
        self.assertEqual(collector.next_profile(state),(2,'left2'))
        state['attempts'].append({'profile':'left2','status':'accepted'})
        self.assertEqual(collector.next_profile(state),(4,'right4'))
        state['attempts'] += [{'profile':'right4','status':'failed'},{'profile':'left2','status':'accepted'}]
        self.assertEqual(collector.next_profile(state),(4,'right4'))
    def test_active_collector_has_no_streak_policy_symbols_or_branches(self):
        text=(TOOLS/'collect_seeded.py').read_text()
        for forbidden in ('failure_streak','CONSECUTIVE_FAILURE_STOP','consecutive_failure_stop','max-consecutive-failures'):
            self.assertNotIn(forbidden,text)
        self.assertTrue(collector.PHYSICAL_FAILURE_CONTINUATION['enabled'])
        self.assertEqual(collector.PHYSICAL_FAILURE_CONTINUATION['bound'],'per_profile_attempt_budget')
    def test_atomic_exclusive_verification_publish(self):
        path=self.base/'verification.json'
        verifier.stable_write(path,{'verified':True})
        verifier.stable_write(path,{'verified':True})
        with self.assertRaisesRegex(ValueError,'differs'):
            verifier.stable_write(path,{'verified':False})
        self.assertEqual(verifier.read(path),{'verified':True})
        self.assertFalse(list(self.base.glob('.*.tmp')))
    def test_native_command_is_one_attempt_without_local_failure_flag(self):
        plan={'bundle':str(BUNDLE),'isaac_python':'python','dataset_python':'python','max_wall_seconds':5400}
        command=collector.command_for(plan,'left2',122,self.base)
        self.assertEqual(command[command.index('--max-attempts')+1],'1')
        self.assertEqual(command[command.index('--successes')+1],'1')
        self.assertNotIn('--max-consecutive-failures',command)

if __name__=='__main__':unittest.main(verbosity=2)
