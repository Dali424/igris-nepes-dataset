#!/usr/bin/env python3
"""Verify seeded physical episodes and merge exactly the requested distinct quotas.

Run with the dataset interpreter. This tool never starts Isaac or changes a
source dataset. Per-episode config hashes may differ; source bindings may not.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile

PROFILES={2:'left2',4:'right4'}
LAYOUTS={'left2':{'source_slot_number':4,'target_slot_number':2,'source_object_prim_path':'/World/Scene/Ring_04','manipulation_arm':'left','target_slot_index':1,'handover_enabled':False},'right4':{'source_slot_number':2,'target_slot_number':4,'source_object_prim_path':'/World/Scene/Ring_02','manipulation_arm':'right','target_slot_index':3,'handover_enabled':False}}
CAMERAS={'observation.images.head':(1280,720),'observation.images.third_person':(1280,720),'observation.images.wrist_left':(640,480),'observation.images.wrist_right':(640,480)}
def require(value,message):
 if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while chunk:=f.read(1024*1024):h.update(chunk)
 return h.hexdigest()
def stable_write(path,data):
 path=Path(path);payload=(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
 if path.exists():require(path.read_bytes()==payload,f'Existing verification differs: {path}');return
 path.parent.mkdir(parents=True,exist_ok=True)
 temporary=None
 try:
  with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.'+path.name+'.',suffix='.tmp',delete=False) as stream:
   temporary=Path(stream.name);stream.write(payload);stream.flush();os.fsync(stream.fileno())
  try:os.link(temporary,path)  # Atomic, exclusive publish: interrupted writes never expose partial JSON.
  except FileExistsError:require(path.read_bytes()==payload,f'Existing verification differs: {path}')
  descriptor=os.open(path.parent,os.O_DIRECTORY)
  try:os.fsync(descriptor)
  finally:os.close(descriptor)
 finally:
  if temporary is not None:temporary.unlink(missing_ok=True)
def load_sampler():
 path=Path(__file__).with_name('seeded_config.py');spec=importlib.util.spec_from_file_location('seeded_sampler',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def compact_module(bundle,profile='left2'):
 package=Path(bundle)/'runtime'/profile/'source/robros_motion_planning';sys.path.insert(0,str(package))
 from robros_motion_planning import compact_dataset
 compact_dataset._official_version();return compact_dataset

def numeric_signature(root):
 """Hash numeric vectors only: excludes image bytes, seed, IDs and Parquet metadata."""
 import numpy as np
 import pyarrow as pa
 import pyarrow.parquet as pq
 paths=sorted((Path(root)/'data').rglob('*.parquet'));require(bool(paths),'No episode Parquet data')
 table=pa.concat_tables([pq.read_table(p,columns=['observation.state','action','episode_index','frame_index']) for p in paths])
 frame=np.asarray(table['frame_index'].to_pylist(),dtype=np.int64);order=np.argsort(frame)
 require(np.array_equal(frame[order],np.arange(len(frame))),'Noncontiguous/duplicate frame indices')
 require(set(table['episode_index'].to_pylist())=={0},'Expected a single episode')
 values={k:np.asarray(table[k].to_pylist(),dtype='<f8')[order] for k in ('observation.state','action')}
 exact=hashlib.sha256();quantized=hashlib.sha256();per_column={}
 for key,value in values.items():
  require(value.ndim==2 and len(value)>0 and np.isfinite(value).all(),f'Invalid numeric vector {key}')
  header=key.encode()+struct.pack('<QQ',*value.shape);raw=np.ascontiguousarray(value).tobytes();q=np.ascontiguousarray(np.rint(value/1e-4),dtype='<i8').tobytes()
  exact.update(header+raw);quantized.update(header+q);per_column[key]={'shape':list(value.shape),'exact_sha256':hashlib.sha256(header+raw).hexdigest(),'quantized_sha256':hashlib.sha256(header+q).hexdigest()}
 return {'motion_exact_sha256':exact.hexdigest(),'motion_quantized_sha256':quantized.hexdigest(),'motion_quantization_step':1e-4,'columns':per_column,'frames':len(frame),'scope':'Ordered state and action vectors only; equality detection, not a general diversity or VLA quality metric.'}

def canonical_config_sha256(config):
 return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def pose_signature(poses,quantum):
 values=[float(v) for pose in poses for v in pose]
 require(values and all(math.isfinite(v) for v in values),'Nonfinite measured initial pose')
 return hashlib.sha256(b''.join(struct.pack('<q',round(v/quantum)) for v in values)).hexdigest()

def validate_sample(bundle,attempt):
 sample_path=attempt/'profile/sample.json';config_path=attempt/'profile/config.json';sample=read(sample_path)
 require(sample.get('schema') == 'nepes_seeded_robot_initial_xy_v3','Wrong sampler schema')
 require(sample.get('sampling_mode')=='seeded_uniform_square' and sample.get('directed_offset_override') is False,'Production collection requires seed-derived offsets, not directed overrides')
 require(sample['pair'] in PROFILES and sample['profile']==PROFILES[sample['pair']],'Pair/profile mismatch')
 require(type(sample['seed']) is int and 0<=sample['seed']<=2_147_483_647 and sample['xy_half_range_m']>0,'Invalid production seed or XY range')
 requested_range=sample['xy_half_range_m']
 sampler=load_sampler();config,recomputed=sampler.build_config(bundle,sample['pair'],sample['seed'],requested_range)
 require(sample==recomputed,'Sample does not match deterministic sampler/current immutable source')
 require(config_path.read_bytes()==sampler.json_bytes(config),'Materialized config differs from sampled configuration')
 require(sha(config_path)==sample['config_sha256'],'Sample/config checksum mismatch')
 return sample,config

IK_WORKER_SOURCES=('worker.py','servo_factory.py','pin_backend.py','pink_qp.py')
def upper_body_ik_evidence(bundle,report,provenance=None):
 """Pink release: the run used this bundle's declared upper-body IK backend and closed it cleanly.

 None when the bundle declares no IK backend (older releases). A native, failed or
 unclosed backend record is neither accepted data nor a retryable physical failure.
 """
 bundle=Path(bundle);declared=read(bundle/'manifest.json').get('ik_backend')
 if declared is None:return None
 block=report.get('upper_body_ik_backend');require(isinstance(block,dict),'Run report lacks the upper-body IK backend record')
 require(block.get('backend')==declared['default'],f"Upper-body IK backend {block.get('backend')!r} is not this release's {declared['default']!r}; batches never mix solvers")
 require(block.get('status')=='closed' and block.get('error') is None,f"Upper-body IK backend did not close cleanly: {block.get('status')}: {block.get('error')}")
 worker=block.get('worker');require(isinstance(worker,dict),'Upper-body IK worker identity missing')
 sources={'ik/pink/'+name:sha(bundle/'ik/pink'/name) for name in IK_WORKER_SOURCES}
 require(worker.get('sources_sha256')==sources and (block.get('requirements') or {}).get('sha256')==sha(bundle/declared['requirements']),'IK worker sources/requirements differ from this release')
 if provenance is not None:
  recorded=provenance.get('upper_body_ik_backend')
  require(isinstance(recorded,dict) and recorded.get('backend')==block['backend'] and recorded.get('status')=='ready' and recorded.get('error') is None and (recorded.get('worker') or {}).get('pid')==worker.get('pid'),'Episode provenance IK backend differs from the run report')
 return {'backend':block['backend'],'status':block['status'],'python':worker.get('python'),'python_version':worker.get('python_version'),'packages':worker.get('packages'),'requirements_sha256':block['requirements']['sha256'],'worker_sources_sha256':sources,'native_prefix_sha256':(block.get('native_solver') or {}).get('native_prefix_sha256'),'urdf_sha256':block.get('urdf_sha256'),'worker_returncode':worker.get('returncode')}

def verify_attempt(bundle,attempt):
 bundle=Path(bundle).resolve(strict=True);attempt=Path(attempt).resolve(strict=True);sample,config=validate_sample(bundle,attempt);profile=sample['profile'];compact=compact_module(bundle,profile)
 from robros_motion_planning.stationary_catalog import config_digest,load_stationary_config
 from robros_motion_planning.compact_storage import verify_archived_success
 normalized_config=load_stationary_config(attempt/'profile/config.json')
 native_config_sha256=config_digest(normalized_config)
 collection=attempt/'collection';document=read(collection/'collection.json');trials=document.get('attempts',[])
 require(len(trials)==1,'Seeded attempt must contain exactly one native attempt')
 require(document['plan']['config_source_sha256']==sample['config_sha256'] and document['plan']['config_sha256']==native_config_sha256,'Collection source-file/canonical config binding mismatch')
 trial=trials[0];require(trial.get('accepted') is True and trial.get('storage_complete') is True,'Native/storage attempt was not accepted')
 require(trial.get('seed')==sample['seed'] and trial.get('returncode')==0,'Native seed/return code mismatch')
 require(trial.get('evidence',{}).get('config_sha256')==native_config_sha256,'Admission config differs from sampled config')
 run=Path(trial['run_dir']).resolve(strict=True);require(run.is_relative_to(collection),'Native run escaped attempt collection')
 shard=Path(trial['storage']['archive']['path']).resolve(strict=True);require(shard.is_relative_to(collection),'Archive escaped attempt collection')
 require(Path(verify_archived_success(collection,trial,document['plan'])).resolve()==shard,'Archive durable admission lineage differs')
 receipt_path=shard/compact.RECEIPT_PATH;receipt=read(receipt_path)
 require(sha(receipt_path)==trial['storage']['archive']['verification_receipt_sha256'],'Archive verification receipt changed')
 require(receipt.get('verified') is True and receipt.get('kind')=='single_episode_shard' and receipt.get('total_episodes')==1,'Invalid archive verification')
 require(all(receipt.get('checks',{}).get(k) is True for k in ('all_mp4_fully_decoded','parquet_metadata_consistent','camera_schema_consistent','source_binding_consistent')),'Missing archive storage checks')
 require(compact._inventory(shard)==receipt['files'],'Archive file hashes changed after verification')
 require(receipt.get('fps')==30 and set(receipt.get('camera_keys',[]))==set(CAMERAS),'Expected four cameras at30fps')
 for v in receipt['videos']:require(tuple(v['resolution'])==CAMERAS[v['camera_key']] and v['fps']==30,'Camera dimensions/rate changed')
 extension=read(shard/'meta/nepes_collection.json');require(len(extension['episodes'])==1,'Archive episode count differs');episode=extension['episodes'][0];provenance=episode['provenance']
 ik=upper_body_ik_evidence(bundle,read(run/'run.json'),provenance)
 require(episode.get('success') is True and provenance.get('seed')==sample['seed'] and provenance.get('config_sha256')==native_config_sha256,'Episode success/seed/config binding mismatch')
 require(all(provenance['single_arm_layout'].get(k)==v for k,v in LAYOUTS[profile].items()),'Wrong arm/source/JIG slot')
 import pyarrow.parquet as pq
 require(pq.read_table(shard/'meta/tasks.parquet').to_pandas().index.tolist()==[config['tasks']['crs_load_jig']['title']],'Task caption differs from source config')
 dataset=attempt/'dataset';info=read(dataset/'meta/info.json');export_receipt=read(dataset/compact.RECEIPT_PATH);export_sources=read(dataset/compact.SOURCES_PATH)
 require(info.get('total_episodes')==1 and info.get('codebase_version')=='v3.0','Missing one-episode LeRobotv3 export')
 require(export_receipt.get('verified') is True and export_receipt.get('kind')=='merged_dataset','Export was not verified')
 require(compact._inventory(dataset)==export_receipt['files'],'Export file hashes changed')
 require(export_sources.get('source_shards')==[{'path':str(shard),'receipt_sha256':sha(receipt_path)}],'Export does not bind exactly this archive')
 require(export_receipt['episode_payload_sha256']['0']==receipt['episode_payload_sha256']['0'],'Export numerical payload differs from archive')
 # The first startup sample is measured before the first post-reset physics step.
 startup=run/'startup_trace.jsonl'
 with startup.open() as f:first=json.loads(f.readline())
 expected=sample['effective_requested_reset_pose_w'];measured=first['base_pose_w']
 require(abs(float(first['time_s']))<1e-9 and len(measured)==len(expected)==7,'Missing t=0 measured startup reset')
 require(all(math.isfinite(float(v)) for v in measured+expected),'Nonfinite startup pose')
 reset_error=max(abs(float(a)-float(b)) for a,b in zip(measured,expected));require(reset_error<=1e-6,f'Measured reset differs from sampled pose: {reset_error}')
 initialization=provenance['initialization'];require(initialization.get('robot_reset_pose_w')==expected,'Declared episode reset is inconsistent (not used as measured proof)')
 measurements=list(run.glob('*/initial_measurement.json'));require(len(measurements)==1,'Expected exactly one actual initial measurement');measurement_path=measurements[0];m=read(measurement_path)
 foot=m['foot_observation'];require(foot.get('available') is True and foot.get('source')=='native_articulation_body_link_state','Need native measured foot-link states')
 poses={'base_pose_w':m['base_pose_w'],'pelvis_pose_w':m['pelvis_pose_w']}
 for side in ('left','right'):
  require(foot['feet'][side].get('available') is True,'Missing measured foot state');poses[side+'_foot_pose_w']=foot['feet'][side]['pose_w']
 require(all(len(v)==7 and all(math.isfinite(float(x)) for x in v) for v in poses.values()),'Invalid measured initial world poses')
 signatures=numeric_signature(shard)
 evidence={'schema':'nepes_seeded_attempt_verification_v1','verified':True,'profile':profile,'pair':sample['pair'],'seed':sample['seed'],'xy_half_range_m':sample['xy_half_range_m'],'sampled_offset_xy_m':sample['applied_offset_xy_m'],'attempt':str(attempt),'config_sha256':native_config_sha256,'config_file_sha256':sample['config_sha256'],'sample_sha256':sha(attempt/'profile/sample.json'),'source_config_sha256':sample['source_config_sha256'],'sampler_sha256':sample['sampler_sha256'],'run':str(run),'archive':str(shard),'archive_receipt_sha256':sha(receipt_path),'export_receipt_sha256':sha(dataset/compact.RECEIPT_PATH),'capture_schema':extension['capture_schema'],'startup_reset':{'source':'startup_trace first t=0 native base_pose_w','requested_pose_w':expected,'measured_pose_w':measured,'maximum_component_error':reset_error,'startup_trace_sha256':sha(startup),'xy_quantization_m':1e-6,'xy_signature_sha256':pose_signature([measured[:2]],1e-6)},'initial_measurement':{'source':str(measurement_path),'sha256':sha(measurement_path),'measurement_phase':m.get('measurement_phase'),'time_s':m['time_s'],'poses':poses,'foot_pose_source':foot['source'],'xy_quantization_m':1e-5,'xy_signature_sha256':pose_signature([v[:2] for v in poses.values()],1e-5)},**signatures,'limitations':['Initial settled poses are measured after HugWBC settling; they are not expected to equal the requested pre-settle reset.','Quantized sequence equality rejects duplicate trajectories but does not prove useful task diversity or VLA generalization.']}

 evidence.update(schema='nepes_seeded_attempt_verification_v3',source_revision=load_sampler().lineage(sample))
 if ik is not None:evidence['upper_body_ik_backend']=ik
 return evidence

def reject_duplicates(previous,current):
 for old in previous:
  require(old['profile']==current['profile'],'Duplicate comparison mixed profiles')
  for key in ('motion_exact_sha256','motion_quantized_sha256'):
   require(old[key]!=current[key],f'Duplicate {key} for {current["profile"]}: seeds{old["seed"]},{current["seed"]}')
  for column in ('observation.state','action'):
   for key in ('exact_sha256','quantized_sha256'):
    require(old['columns'][column][key]!=current['columns'][column][key],f'Duplicate {column} {key} for {current["profile"]}')
  for section in ('startup_reset','initial_measurement'):
   require(old[section]['xy_signature_sha256']!=current[section]['xy_signature_sha256'],f'Duplicate measured {section} XY for {current["profile"]}')

def validate_record_revision(plan,record,sample):
 sampler=load_sampler()
 require(plan.get('source_revisions')==sampler.revision_plan(),'Source profile policy differs')
 require(type(plan.get('first_seed')) is int and plan['first_seed']>=0 and record['index']>=0 and record['index']==record['seed']-plan['first_seed'],'Attempt index/seed differs from batch plan')
 require(sample['pair']==record['pair'] and sample['profile']==record['profile'] and sample['seed']==record['seed'],'Sample/record identity differs')
 expected_range=sampler.effective_xy_range(record['pair'],record['seed'],plan['xy_half_range_by_profile_m'][record['profile']])
 require(sample['xy_half_range_m']==expected_range and sample['schema']=='nepes_seeded_robot_initial_xy_v3','Sample schema or effective XY range differs')
 lineage=sampler.lineage(sample)
 require(record.get('source_revision')==lineage,'Batch record profile lineage differs')
 return lineage


PHYSICAL_STAGES=frozenset(('prerequisites','navigate_source','reach_posture','pregrasp','approach','close','lift','carry_posture','navigate_target','place_posture','transfer','place','release','retreat','rest_posture','settle'))
PHYSICAL_REASONS=frozenset(('target_unreachable','task_timeout','jig_displaced','source_object_displaced','grasp_lost','release_lost','placement_displaced_after_release','placement_displaced_during_release','stationary_pelvis_drift','no_walk_stance_outside_source_station','transfer_transit_object_deviation'))

def classify_physical_reason(reason,stage,config=None):
 """Only measured task-limit failures, never missing sensors or arbitrary exceptions.

 Inventory: stationary_pick_load.py, pregrasp.py, hinge_controller.py,
 motion_watchdog.py, stationary_feet.py and payload_transport.py in the bundled
 runtime. All success thresholds remain unchanged. Unknown reasons stop collection.
 """
 if stage not in PHYSICAL_STAGES or not isinstance(reason,str):return False
 if reason in PHYSICAL_REASONS:return True
 if reason in (f'stage_timeout:{stage}',f'progress_stalled:{stage}',f'RuntimeError: progress_stalled:{stage}'):return True
 if stage in ('reach_posture','carry_posture','place_posture','rest_posture') and reason==f'posture_tracking_error:{stage}':return True
 if reason in ('pregrasp:unexpected_object_contact','RuntimeError: pregrasp:unexpected_object_contact','RuntimeError: guarded_approach:unexpected_approach_contact'):return True
 if reason in ('reach_posture:unexpected_object_contact:left','reach_posture:unexpected_object_contact:right'):return True
 if reason in {f'stationary_feet_hold:{side}:{gate}' for side in ('left','right') for gate in ('sole_xy_stationary','sole_z_stationary','ground_normal_force')}:return True
 if reason.startswith('unsafe_payload_transport:') and config is not None:
  task=config['tasks']['crs_load_jig'];settings=task.get('payload_transport',task.get('transport_constraints',{}))
  # Rule names and obstacle identifiers must come from this exact configuration.
  allowed=set()
  def visit(node):
   if isinstance(node,dict):
    for name,rule in node.items():
     if isinstance(rule,dict):
      for field,gate in (('minimum_world_z_m','minimum_world_z'),('maximum_world_z_m','maximum_world_z'),('orientation','orientation')):
       if field in rule:allowed.add(name+':'+gate)
      for obstacle in rule.get('forbidden_aabbs',[]):allowed.add(name+':forbidden_aabb:'+obstacle['id'])
      visit(rule)
  visit(settings)
  violations=reason.removeprefix('unsafe_payload_transport:').split(';')
  return bool(allowed) and all(item in allowed for item in violations)
 return False

def verify_failed_attempt(bundle,attempt,returncode):
 """Allow retry only for an explicit, fully reported physical task failure.

 A false accepted flag by itself is insufficient: crashes, missing reports,
 timeouts, export/storage failures and unknown RuntimeErrors stop the batch.
 This is failure classification, never relaxed success admission.
 """
 bundle=Path(bundle).resolve(strict=True);attempt=Path(attempt).resolve(strict=True)
 sample,_=validate_sample(bundle,attempt)
 return verify_reported_physical_failure(bundle,attempt,returncode,sample)

def verify_reported_physical_failure(bundle,attempt,returncode,sample):
 """Strict failure evidence for an independently validated original/render config.

 Caller must establish deterministic sample or visual-parent lineage first.
 This function rechecks native source/config, execution and durable trace bindings.
 """
 bundle=Path(bundle).resolve(strict=True);attempt=Path(attempt).resolve(strict=True)
 require(sample['pair'] in PROFILES and sample['profile']==PROFILES[sample['pair']], 'Invalid failed pair/profile')
 profile=sample['profile']
 require(sha(attempt/'profile/config.json')==sample['config_sha256'],'Failed config source hash differs')
 package=bundle/'runtime'/profile/'source/robros_motion_planning'
 sys.path.insert(0,str(package))
 from robros_motion_planning.stationary_catalog import config_digest,load_stationary_config
 normalized_config=load_stationary_config(attempt/'profile/config.json');native_digest=config_digest(normalized_config)
 collection=attempt/'collection';document=read(collection/'collection.json');plan=document['plan'];trials=document.get('attempts',[])
 require(returncode==2,'Outer collector did not exit with the documented incomplete-quota code 2')
 require(document.get('complete') is True and document.get('operator_stopped') is False and len(trials)==1,'Incomplete/stopped/ambiguous native collection')
 require(plan['config_source_sha256']==sample['config_sha256'] and plan['config_sha256']==native_digest and plan['seed']==sample['seed'],'Failure source/config/seed binding mismatch')
 require(Path(plan['config']).resolve()==attempt/'profile/config.json','Failure config path differs')
 require(document.get('plan_sha256')==config_digest(plan),'Native collection plan digest differs')
 runtime=bundle/'runtime'/profile
 require(bool(plan.get('algorithm_sha256')),'Missing native algorithm identity')
 for relative,expected in plan['algorithm_sha256'].items():
  path=(runtime/relative).resolve(strict=True);require(path.is_relative_to(runtime) and sha(path)==expected,'Native failure algorithm source changed')
 trial=trials[0]
 require(trial.get('accepted') is False and trial.get('status')=='complete' and trial.get('seed')==sample['seed'] and trial.get('returncode')==2,'Native trial did not finish a reported task failure')
 require(all(trial.get(key) is False for key in ('timed_out','forced_termination','stop_requested')),'Timed out, killed or interrupted native failure is not retryable')
 require(trial.get('storage_complete') is True,'Failure trace storage is incomplete')
 require(trial.get('failure_reason')=='native_process_failed_or_exit_unrecorded:2','Different admission/storage/export error requires review')
 run=Path(trial['run_dir']).resolve(strict=True)
 require(run==collection/'runs/attempt_000000' and not Path(trial['run_dir']).is_symlink(),'Unexpected native failure run path')
 report=read(run/'run.json');episodes=report.get('episodes',[])
 require(report.get('schema')=='nepes_task_run_v1' and report.get('mode')=='execute' and report.get('complete') is True and len(episodes)==1,'Missing normally completed native run report')
 require(report.get('seed')==sample['seed'] and report.get('config_sha256')==native_digest and report.get('all_episodes_succeeded') is False and report.get('all_tasks_succeeded') is False,'Failure run identity/outcome differs')
 episode=episodes[0];episode_id='cycle_0000_crs_pickup_load_jig'
 require(episode.get('episode_id')==episode_id and episode.get('task_id')=='crs_pickup_load_jig' and episode.get('seed')==sample['seed'] and episode.get('success') is False and episode.get('full_task_completed') is False,'Failure episode identity/outcome differs')
 result_path=run/episode_id/'result.json';require(read(result_path)==episode,'Native failed result and run report differ')
 ik=upper_body_ik_evidence(bundle,report)
 stage=episode.get('last_stage');reason=episode.get('reason')
 require(classify_physical_reason(reason,stage,normalized_config),'Unknown or nonphysical failure; explicit physical allowlist only')
 diagnostics=trial.get('failure_diagnostics',{})
 require(diagnostics.get('reason')==reason and diagnostics.get('stage')==stage and diagnostics.get('episode_id')==episode_id and Path(diagnostics.get('source','')).resolve()==run/'run.json' and diagnostics.get('report_errors')==[],'Failure diagnostics/report binding differs')
 storage=trial['storage'];receipt_path=run/'storage_receipt.json'
 require(Path(storage['receipt_path']).resolve()==receipt_path and sha(receipt_path)==storage['receipt_sha256'],'Failed storage receipt hash/path changed')
 receipt=read(receipt_path)
 require(receipt.get('schema')=='nepes_attempt_storage_v1' and receipt.get('kind')=='failed_attempt' and receipt.get('cleanup_status')=='complete' and storage.get('kind')=='failed_attempt' and storage.get('cleanup_status')=='complete','Failure storage receipt is not complete')
 require(receipt.get('collection_root')==str(collection) and receipt.get('run_dir')==str(run) and receipt.get('attempt_index')==0 and receipt.get('admitted_evidence') is None and receipt.get('failure_reason')==trial['failure_reason'],'Failed storage receipt lineage differs')
 require(receipt.get('trace_policy')=='gzip' and receipt.get('traces')==storage.get('traces') and len(receipt.get('traces',[]))==1,'Failed trace retention differs')
 trace=receipt['traces'][0];relative=Path(trace.get('compressed_path',''))
 require(trace.get('state')=='gzipped' and trace.get('path')==episode_id+'/trace.jsonl' and relative==Path(episode_id)/'trace.jsonl.gz','Failed trace shape differs')
 trace_path=(run/relative).resolve(strict=True)
 require(trace_path.is_relative_to(run) and not (run/relative).is_symlink() and trace_path.stat().st_size==trace['compressed_size_bytes'] and sha(trace_path)==trace['compressed_sha256'],'Failed compressed trace is missing or corrupt')
 require(not (attempt/'dataset').exists() and not list((collection/'archives').glob('attempt_*')),'Unexpected export/archive on a failed attempt')
 return {'schema':'nepes_known_physical_failure_v1','retryable_physical_failure':True,'success':False,'profile':profile,'pair':sample['pair'],'seed':sample['seed'],'reason':reason,'stage':stage,'config_sha256':native_digest,'config_file_sha256':sample['config_sha256'],'collection_sha256':sha(collection/'collection.json'),'run_sha256':sha(run/'run.json'),'result_sha256':sha(result_path),'storage_receipt_sha256':sha(receipt_path),'compressed_trace_sha256':trace['compressed_sha256'],'native_returncode':2,'collector_returncode':2,**({'upper_body_ik_backend':ik} if ik is not None else {})}

def plan_quotas(plan):
 """Read legacy equal quotas or explicit unequal quotas; never silently combine them."""
 require(('quota_each' in plan) != ('quota_by_profile' in plan),'Ambiguous/missing profile quotas')
 quotas=({name:plan['quota_each'] for name in PROFILES.values()} if 'quota_each' in plan else dict(plan['quota_by_profile']))
 require(set(quotas)==set(PROFILES.values()) and all(type(v) is int and v>0 for v in quotas.values()),'Invalid per-profile quotas')
 return quotas

def quota_arguments(plan):
 quotas=plan_quotas(plan)
 if 'quota_each' in plan:return ['--expected-per-pair',str(plan['quota_each'])]
 return ['--expected-left',str(quotas['left2']),'--expected-right',str(quotas['right4'])]

def requested_quotas(args):
 each=getattr(args,'expected_per_pair',None);left=getattr(args,'expected_left',None);right=getattr(args,'expected_right',None)
 require((left is None)==(right is None),'Provide both --expected-left and --expected-right')
 require((each is None)!=(left is None),'Use --expected-per-pair OR both per-profile expectations')
 return plan_quotas({'quota_each':each} if each is not None else {'quota_by_profile':{'left2':left,'right4':right}})

def merge_batch(args):
 root=args.batch.resolve(strict=True)
 output=args.output.resolve();require(output.is_dir() if args.verify_existing else not output.exists(),'Use --verify-existing only for a published dataset; never overwrite')
 with ExitStack() as locks:
  if args.batch_lock_fd is None:
   stream=locks.enter_context((root/'.batch.lock').open('r'));fcntl.flock(stream.fileno(),fcntl.LOCK_SH|fcntl.LOCK_NB)
  else:
   a=os.fstat(args.batch_lock_fd);b=(root/'.batch.lock').stat();require((a.st_dev,a.st_ino)==(b.st_dev,b.st_ino),'Inherited batch lock is not this batch');fcntl.flock(args.batch_lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  state=read(root/'batch.json');plan=state['plan'];bundle=Path(plan['bundle']).resolve(strict=True)
  sampler=load_sampler();require(plan.get('source_revisions')==sampler.revision_plan(),'Source revision plan differs')
  require(not output.is_relative_to(bundle) and not bundle.is_relative_to(output),'Merge output cannot overlap immutable bundle')
  quotas=plan_quotas(plan);require(requested_quotas(args)==quotas,'Expected quota differs from batch plan')
  require(state.get('status') in ('collected','merging','complete'),'Batch is incomplete; exact quotas required')
  require(sha(bundle/'manifest.json')==plan['bundle_manifest_sha256'],'Bundle manifest changed')
  for relative,item in read(bundle/'manifest.json')['files'].items():
   file=bundle/relative;require(file.resolve().is_relative_to(bundle) and not file.is_symlink() and sha(file)==item['sha256'],'Bundle source changed: '+relative)
  require(all(sha(Path(__file__).with_name(name))==digest for name,digest in plan['tools_sha256'].items()),'Frozen batch tools changed')
  before_sha=sha(root/'batch.json');groups={profile:[] for profile in PROFILES.values()};mapping=[];shards=[]
  for profile in groups:
   accepted=[a for a in state['attempts'] if a['profile']==profile and a.get('status')=='accepted'];require(len(accepted)==quotas[profile],f'{profile}: quota not exact')
   for record in accepted:
    attempt=Path(record['directory']).resolve(strict=True);require(attempt.is_relative_to(root),'Attempt outside batch')
    lock_path=attempt/'collection/.compact_collection.lock';require(lock_path.is_file(),'Missing collector lock');stream=locks.enter_context(lock_path.open('r'));fcntl.flock(stream.fileno(),fcntl.LOCK_SH|fcntl.LOCK_NB)
    evidence=verify_attempt(bundle,attempt);require(evidence==read(attempt/'verification.json'),'Saved verification changed');require(sha(attempt/'verification.json')==record['verification_sha256'],'Batch verification binding differs')
    require(evidence['profile']==profile and evidence['pair']==record['pair'] and evidence['seed']==record['seed'] and evidence['xy_half_range_m']==sampler.effective_xy_range(record['pair'],record['seed'],plan['xy_half_range_by_profile_m'][profile]),'Accepted evidence profile/pair/seed/effective range differs from plan record')
    sample=read(attempt/'profile/sample.json');revision=validate_record_revision(plan,record,sample)
    reject_duplicates(groups[profile],evidence);groups[profile].append(evidence);shards.append(Path(evidence['archive']));mapping.append({'episode_index':len(mapping),'profile':profile,'seed':evidence['seed'],'config_sha256':evidence['config_sha256'],'config_file_sha256':evidence['config_file_sha256'],'sample_sha256':evidence['sample_sha256'],'xy_half_range_m':evidence['xy_half_range_m'],'offset_xy_m':evidence['sampled_offset_xy_m'],'motion_exact_sha256':evidence['motion_exact_sha256'],'split_group':'seeded_motion_'+evidence['motion_exact_sha256'],'motion_family':'seeded_motion_'+evidence['motion_exact_sha256'],'motion_quantized_sha256':evidence['motion_quantized_sha256'],'measured_startup_pose_w':evidence['startup_reset']['measured_pose_w'],'measured_initial_world_poses':evidence['initial_measurement']['poses'],'source_archive':evidence['archive'],'verification_sha256':record['verification_sha256'],'source_revision':revision})
  require(len(set(shards))==len(shards),'Duplicate archive paths');capture=next(iter(groups.values()))[0]['capture_schema'];require(all(e['capture_schema']==capture for g in groups.values() for e in g),'Capture schemas differ')
  manifest=output.with_name(output.name+'.seeded_sources.json');document={'schema':'nepes_seeded_collection_aggregate_v1',**({'expected_per_pair':plan['quota_each']} if 'quota_each' in plan else {'expected_by_profile':quotas}),'total_episodes':len(shards),'different_configs_per_episode':True,'source_revisions':plan['source_revisions'],'xy_half_range_by_profile_m':plan['xy_half_range_by_profile_m'],'xy_half_range_scope':'Fixed per-profile range for every seed; each episode records its applied offset.','episodes':mapping,'source_selection_sha256':canonical_config_sha256({'plan':plan,'episodes':mapping})};stable_write(manifest,document)
  compact=compact_module(bundle)
  if args.verify_existing:
   saved=read(output/compact.RECEIPT_PATH);source=read(output/compact.SOURCES_PATH)
   require(saved.get('verified') is True and saved.get('kind')=='merged_dataset','Published merge was not verified')
   require(all(saved.get('checks',{}).get(key) is True for key in ('video_bytes_preserved_without_reencoding','episode_numeric_payloads_preserved')),'Missing published aggregation checks')
   require(source.get('source_shards')==[{'path':str(shard),'receipt_sha256':sha(shard/compact.RECEIPT_PATH)} for shard in shards],'Published merge source archives differ')
   require(source.get('collection_manifest')=={'path':str(manifest),'sha256':sha(manifest)} and read(output/'meta/nepes_stationary_collection.json')==document,'Published seeded merge manifest differs')
   receipt=compact.verify_dataset(output,expected_episodes=len(shards))
   require(receipt['files']==saved['files'] and receipt['episode_payload_sha256']==saved['episode_payload_sha256'],'Published merge files/payload changed')
   for index,shard in enumerate(shards):
    require(receipt['episode_payload_sha256'][str(index)]==read(shard/compact.RECEIPT_PATH)['episode_payload_sha256']['0'],'Merged episode numeric payload differs from source')
  else:
   receipt=compact.merge_verified_shards(shards,output,repo_id=args.repo_id,collection_manifest=manifest)
  require(receipt['total_episodes']==sum(quotas.values()) and receipt['verified'] is True,'Merged quota/verification differs');require(sha(root/'batch.json')==before_sha,'Batch changed during merge')
  return {'dataset':str(output),'episodes':receipt['total_episodes'],'frames':receipt['total_frames'],'verified':True,'sources_manifest':str(manifest)}

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 v=sub.add_parser('verify-attempt');v.add_argument('--bundle',type=Path,required=True);v.add_argument('--attempt',type=Path,required=True);v.add_argument('--output',type=Path,required=True)
 f=sub.add_parser('verify-failure');f.add_argument('--bundle',type=Path,required=True);f.add_argument('--attempt',type=Path,required=True);f.add_argument('--returncode',type=int,required=True);f.add_argument('--output',type=Path,required=True)
 m=sub.add_parser('merge');m.add_argument('--batch',type=Path,required=True);m.add_argument('--output',type=Path,required=True);m.add_argument('--expected-per-pair',type=int);m.add_argument('--expected-left',type=int);m.add_argument('--expected-right',type=int);m.add_argument('--repo-id',default='local/igris-c-seeded-crs-pairs');m.add_argument('--batch-lock-fd',type=int);m.add_argument('--verify-existing',action='store_true')
 args=p.parse_args()
 if args.command=='verify-failure':
  require(not args.output.resolve().is_relative_to(args.bundle.resolve()),'Verification output cannot be inside immutable bundle')
  evidence=verify_failed_attempt(args.bundle,args.attempt,args.returncode);stable_write(args.output,evidence);print(json.dumps(evidence));return 0
 if args.command=='verify-attempt':
  require(not args.output.resolve().is_relative_to(args.bundle.resolve()),'Verification output cannot be inside immutable bundle')
  evidence=verify_attempt(args.bundle,args.attempt);stable_write(args.output,evidence);print(json.dumps({k:evidence[k] for k in ('verified','profile','seed','frames','motion_exact_sha256','motion_quantized_sha256')}));return 0
 result=merge_batch(args);print(json.dumps(result,indent=2));return 0
if __name__=='__main__':
 try:raise SystemExit(main())
 except (OSError,ValueError,KeyError,RuntimeError) as exc:raise SystemExit(f'Seeded verification stopped: {exc}') from exc
