"""Shared CPU-only video augmentation contracts. No simulator or source writes."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import augment_video_v2 as engine

FLAG=engine.FLAG
PRESETS=('visible_darker_v2','visible_brighter_v2')
CAMERAS={'observation.images.head','observation.images.wrist_left','observation.images.wrist_right','observation.images.third_person'}
AUDIT='meta/video_augmentation_audit.json'
PROVENANCE='meta/video_augmentation.json'
TOOLS=('video_collect.py','video_common.py','augment_video_v2.py')
require=engine.require
read=engine.read_json
sha=engine.sha

def digest(value):
    return hashlib.sha256(engine.canonical(value)).hexdigest()

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.'+path.name+'.',delete=False) as stream:
            temporary=Path(stream.name);stream.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False).encode()+b'\n');stream.flush();os.fsync(stream.fileno())
        temporary.replace(path)
        descriptor=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)

def tool_hashes():
    return {name:sha(Path(__file__).with_name(name)) for name in TOOLS}

def check_inventory(source,expected):
    require(engine.inventory(source,exclude=(engine.RECEIPT,))==expected,'Source files no longer match their verified inventory')

def dataset_layout(source,expected_episodes=None):
    info=read(source/'meta/info.json')
    require(info.get('codebase_version')=='v3.0' and info.get('fps')==30,'Expected LeRobotv3 at30fps')
    count=info['total_episodes'];require(count>0 and (expected_episodes is None or count==expected_episodes),'Incomplete/unexpected episode quota')
    camera_keys={k for k,f in info['features'].items() if f['dtype']=='video'}
    require(camera_keys==CAMERAS,'Exactly the four head/wrist-left/wrist-right/third-person cameras are required')
    require(info['features'].get(FLAG,{}).get('dtype')=='int64' and info['features'][FLAG]['shape']==[1],'Missing augmentation flag feature')
    metadata_paths=sorted((source/'meta/episodes').rglob('*.parquet'))
    tables=[pq.read_table(p) for p in metadata_paths];rows=[row for table in tables for row in table.to_pylist()]
    rows.sort(key=lambda row:row['episode_index'])
    require([row['episode_index'] for row in rows]==list(range(count)),'Episode metadata indices are incomplete/duplicated')
    jobs=[];data_files=set();video_files=set();offset=0
    for row in rows:
        length=row['length'];require(length>1 and row['dataset_from_index']==offset and row['dataset_to_index']==offset+length,'Episode offsets do not match frame counts');offset+=length
        name=info['data_path'].format(chunk_index=row['data/chunk_index'],file_index=row['data/file_index']);engine.relative_file(source,name);data_files.add(name)
        for key in sorted(camera_keys):
            prefix=f'videos/{key}/';name=info['video_path'].format(video_key=key,chunk_index=row[prefix+'chunk_index'],file_index=row[prefix+'file_index'])
            engine.relative_file(source,name)
            require(row[prefix+'from_timestamp']==0 and abs(row[prefix+'to_timestamp']-length/30)<1e-6,'Source must have one complete unshared video file per episode and camera')
            require(name not in video_files,'Shared video files are unsupported; use the verified seeded merger output')
            video_files.add(name);jobs.append({'index':len(jobs),'episode_index':row['episode_index'],'camera_key':key,'path':name,'frames':length,'shape':info['features'][key]['shape']})
    require(offset==info['total_frames'],'Total frame count differs')
    require({p.relative_to(source).as_posix() for p in (source/'videos').rglob('*.mp4')}==video_files,'Unreferenced/missing video files')
    require({p.relative_to(source).as_posix() for p in (source/'data').rglob('*.parquet')}==data_files,'Unreferenced/missing data Parquet files')
    return {'info':info,'episodes':rows,'jobs':jobs,'data_files':sorted(data_files),'episode_metadata_files':[p.relative_to(source).as_posix() for p in metadata_paths]}

def check_source_declarations(info,receipt,extension,manifest,expected):
    require(expected>0,'Expected source quota must be positive')
    require(receipt.get('schema')=='nepes_verified_lerobot_storage_v1' and receipt.get('verified') is True and receipt.get('kind')=='merged_dataset','Source is not a published verified final merged dataset')
    require(receipt.get('total_episodes')==expected==info['total_episodes'] and receipt.get('total_frames')==info['total_frames'],'Incomplete source quota/frame count')
    checks=('all_mp4_fully_decoded','parquet_metadata_consistent','camera_schema_consistent','source_binding_consistent','video_bytes_preserved_without_reencoding','episode_numeric_payloads_preserved')
    require(all(receipt.get('checks',{}).get(key) is True for key in checks),'Final source lacks required storage/merge verification')
    require(manifest.get('schema') in ('nepes_seeded_collection_aggregate_v1','nepes_augmentation_parent_selection_v1') and manifest.get('total_episodes')==expected,'Source is not a verified complete collection or explicit parent selection')
    mapping=manifest.get('episodes',[]);episodes=extension.get('episodes',[])
    require(len(mapping)==len(episodes)==expected,'Source seeded lineage is incomplete')
    counts={'left2':0,'right4':0};seen={k:set() for k in counts}
    for index,(entry,parent) in enumerate(zip(mapping,episodes)):
        profile=entry.get('profile');require(profile in counts and entry.get('episode_index')==parent.get('episode_index')==index,'Parent episode mapping/order differs')
        counts[profile]+=1;require(parent.get('success') is True and parent['provenance'].get('kind')=='original','Parent must be an original physically successful rollout')
        require(entry['seed']>=0 and parent['provenance'].get('seed')==entry['seed'] and parent['provenance'].get('config_sha256')==entry['config_sha256'],'Seed/config lineage differs')
        signature=entry['motion_quantized_sha256'];require(signature not in seen[profile],'Repeated source motion within a pair');seen[profile].add(signature)
    if manifest['schema']=='nepes_seeded_collection_aggregate_v1':
        require(expected%2==0 and manifest.get('expected_per_pair')==expected//2 and set(counts.values())=={expected//2},'Source pair quota is not balanced/exact')
    else:
        require(manifest.get('profile_counts')==counts and manifest.get('selection_from_verified_complete_batch') is True and manifest.get('parent_merge_receipt_sha256'),'Explicit source selection lineage differs')
    require(set(receipt.get('camera_keys',[]))==CAMERAS and receipt.get('fps')==30,'Source camera/fps verification differs')

def source_plan(source,expected=None,presets=PRESETS):
    require(engine.version('lerobot')=='0.4.4','Use pinned lerobot==0.4.4 dataset Python')
    receipt=read(source/engine.RECEIPT);layout=dataset_layout(source,expected)
    expected=layout['info']['total_episodes']
    require(presets and len(set(presets))==len(presets) and all(name in PRESETS for name in presets),'Unknown/duplicate video preset')
    extension=read(source/'meta/nepes_collection.json');manifest=read(source/'meta/nepes_stationary_collection.json')
    check_source_declarations(layout['info'],receipt,extension,manifest,expected)
    check_inventory(source,receipt['files'])
    for relative in layout['data_files']:
        table=pq.read_table(source/relative,columns=[FLAG]);require(all(value in (0,[0]) for value in table[FLAG].to_pylist()),'Source is already augmented')
    index={item['path']:item for item in receipt['files']}
    for job in layout['jobs']:job['source_sha256']=index[job['path']]['sha256']
    # Every sibling variant of this parent episode stays in the SAME data split.
    parents=[]
    for item in manifest['episodes']:
        family='seeded_motion_'+item['motion_exact_sha256']
        parents.append({'episode_index':item['episode_index'],'profile':item['profile'],'seed':item['seed'],'config_sha256':item['config_sha256'],'motion_exact_sha256':item['motion_exact_sha256'],'motion_quantized_sha256':item['motion_quantized_sha256'],'parent_numeric_payload_sha256':receipt['episode_payload_sha256'][str(item['episode_index'])],'motion_family':family,'split_group':family})
    import av
    versions={name:engine.version(name) for name in ('lerobot','av','numpy','pyarrow')}
    versions['ffmpeg_libraries']={name:list(value) for name,value in av.library_versions.items()}
    plan={'versions':versions,'source_files':receipt['files'],'schema':'nepes_seeded_video_batch_plan_v1','source':str(source),'source_receipt_sha256':sha(source/engine.RECEIPT),'source_inventory_sha256':digest(receipt['files']),'source_seeded_manifest_sha256':sha(source/'meta/nepes_stationary_collection.json'),'expected_episodes':expected,'total_frames':layout['info']['total_frames'],'fps':30,'camera_keys':sorted(CAMERAS),'presets':{name:deepcopy(engine.PRESETS[name]) for name in presets},'minimum_absolute_mean_luma_delta':.025,'tool_sha256':tool_hashes(),'parents':parents,'split_policy':'Parent and all photometric siblings share split_group. No independent physical rollouts or independent train/test samples are claimed.'}
    return plan,layout,receipt

def validate_job(source,work,job,recipe,record,minimum_delta):
    require(record.get('job')==job and record.get('recipe_sha256')==digest(recipe),'Video checkpoint job/preset changed')
    audit=record['audit'];target=engine.relative_file(work,job['path'])
    require(sha(target)==audit['output_sha256'] and sha(source/job['path'])==audit['source_sha256']==job['source_sha256'],'Video checkpoint/source file changed')
    require(audit['decoded_frames']==job['frames'] and audit['full_source_and_output_decode'] is True and audit['fps']==30 and audit['resolution']==[job['shape'][1],job['shape'][0]],'Incomplete video decode/shape/count verification')
    delta=audit['mean_luma_delta'];direction=1 if recipe['name']=='visible_brighter_v2' else -1
    require(direction*delta>=minimum_delta,f'Preset brightness change is too small or wrong direction: {delta:+.6f}; required {minimum_delta}')
    histogram=np.asarray(audit['output_histogram_rgb'],dtype=np.int64)
    require(histogram.shape==(3,256) and (histogram>=0).all(),'Invalid output histogram')
    require(engine.histogram_stats(histogram)==record['statistics'],'Checkpoint RGB statistics differ from decoded histogram')
    return histogram

def write_metadata(source,work,layout,plan,preset,records):
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.video_utils import get_video_info
    info=deepcopy(layout['info']);stats=read(source/'meta/stats.json');episode_updates={row['episode_index']:{} for row in layout['episodes']};hist={key:np.zeros((3,256),dtype=np.int64) for key in CAMERAS}
    for record in records:
        job=record['job'];key=job['camera_key'];hist[key]+=np.asarray(record['audit']['output_histogram_rgb'],dtype=np.int64)
        episode_updates[job['episode_index']][key]=record['statistics'];info['features'][key]['info']=get_video_info(work/job['path'])
    for key in CAMERAS:stats[key]=engine.histogram_stats(hist[key])
    def flag_stats(template,count):return {key:[0.0 if key=='std' else count if key=='count' else 1.0] for key in template}
    stats[FLAG]=flag_stats(stats[FLAG],info['total_frames'])
    for row in layout['episodes']:episode_updates[row['episode_index']][FLAG]=flag_stats(stats[FLAG],row['length'])
    preserved={}
    for relative in layout['data_files']:
        original=pq.read_table(source/relative);field=original.schema.field(FLAG);values=original[FLAG].to_pylist()
        require(all(value in (0,[0]) for value in values),'Original augmentation flag changed')
        updated=original.set_column(original.schema.get_field_index(FLAG),field,pa.array([1 if value==0 else [1] for value in values],type=field.type))
        path=work/relative;path.parent.mkdir(parents=True,exist_ok=True);pq.write_table(updated,path,compression='zstd')
        reread=pq.read_table(path);columns=[key for key in original.column_names if key!=FLAG]
        require(original.schema==reread.schema and original.select(columns).equals(reread.select(columns)),'A state/action/time/task or other preserved Arrow column changed')
        digests={key:engine.column_digest(original,key,info['features'][key]) for key in columns}
        require(all(engine.column_digest(reread,key,info['features'][key])==value for key,value in digests.items()),'Preserved numeric values are not bit-identical')
        preserved[relative]={'unchanged_numeric_column_sha256':digests,'rows':original.num_rows,'source_sha256':sha(source/relative),'output_sha256':sha(path),'unchanged_columns':columns,'all_preserved_arrow_values_identical':True}
    for relative in layout['episode_metadata_files']:
        table=pq.read_table(source/relative);rows=table.to_pylist()
        for row in rows:
            for key,values in episode_updates[row['episode_index']].items():
                for statistic,value in values.items():
                    name=f'stats/{key}/{statistic}';require(name in row,'Missing episode statistics column '+name);row[name]=value
        path=work/relative;path.parent.mkdir(parents=True,exist_ok=True);pq.write_table(pa.Table.from_pylist(rows,schema=table.schema),path,compression='zstd')
    (work/'meta').mkdir(exist_ok=True)
    import shutil
    shutil.copyfile(source/'meta/tasks.parquet',work/'meta/tasks.parquet')
    atomic_json(work/'meta/info.json',info);atomic_json(work/'meta/stats.json',stats)
    provenance={'schema':'nepes_seeded_offline_video_augmentation_v1','kind':'video_augmentation','augmentation_id':'videoaug_'+digest({'parent':plan['source_receipt_sha256'],'preset':preset}),'total_episodes':plan['expected_episodes'],'total_frames':plan['total_frames'],'camera_keys':plan['camera_keys'],'versions':plan.get('versions'),'preset':preset,'recipe':plan['presets'][preset],'source':plan['source'],'source_receipt_sha256':plan['source_receipt_sha256'],'source_inventory_sha256':plan['source_inventory_sha256'],'source_seeded_manifest_sha256':plan['source_seeded_manifest_sha256'],'parents':plan['parents'],'split_policy':plan['split_policy'],'original_rollout_success_inherited':True,'independent_physics_execution':False,'physical_success_revalidated':False,'source_physical_receipt_applies_to_parent_only':True,'not_scene_rerendering':True,'modified_numeric_columns':[FLAG],'tool_sha256':plan['tool_sha256']}
    atomic_json(work/PROVENANCE,provenance)
    official=LeRobotDatasetMetadata(repo_id='local/igris-c-seeded-'+preset,root=work)
    require(official.total_episodes==plan['expected_episodes'] and official.total_frames==plan['total_frames'],'Official LeRobot reload quota differs')
    for key in CAMERAS:require(official.stats[key]['mean'].tolist()==stats[key]['mean'],'Official RGB statistics reload differs')
    return preserved,provenance

def verify_final(source,target,plan,preset,layout):
    require(target.is_dir(),'Final augmented dataset is absent')
    audit=read(target/AUDIT);provenance=read(target/PROVENANCE)
    require(audit.get('schema')=='nepes_seeded_video_dataset_audit_v1' and audit.get('verified') is True and audit.get('preset')==preset and audit.get('plan_sha256')==digest(plan),'Final dataset audit does not match frozen plan')
    require(engine.inventory(target,exclude=(AUDIT,))==audit['files'],'Augmented output files changed after verification')
    require(not (target/engine.RECEIPT).exists() and not (target/'meta/nepes_storage_sources.json').exists(),'Derivative must not impersonate a verified physical rollout')
    require(provenance['parents']==plan['parents'] and provenance['recipe']==plan['presets'][preset] and provenance['source_receipt_sha256']==plan['source_receipt_sha256'],'Derived parent/preset provenance changed')
    target_layout=dataset_layout(target,plan['expected_episodes']);require(target_layout['info']['total_frames']==plan['total_frames'],'Final frame count differs')
    require(sha(source/'meta/tasks.parquet')==sha(target/'meta/tasks.parquet'),'Task text/index mapping changed')
    for relative in layout['data_files']:
        original=pq.read_table(source/relative);derived=pq.read_table(target/relative);columns=[key for key in original.column_names if key!=FLAG]
        require(original.schema==derived.schema and original.select(columns).equals(derived.select(columns)),'Augmentation changed state/action/timestamps/tasks or other Arrow columns')
        require(all(value in (1,[1]) for value in derived[FLAG].to_pylist()),'Output augmentation flag missing')
        require(all(engine.column_digest(original,key,layout['info']['features'][key])==engine.column_digest(derived,key,layout['info']['features'][key]) for key in columns),'Preserved numeric column bytes changed')
    require(len(audit['video_records'])==len(layout['jobs']),'Missing per-video verification')
    for job,record in zip(layout['jobs'],audit['video_records']):validate_job(source,target,job,plan['presets'][preset],record,plan['minimum_absolute_mean_luma_delta'])
    # Check exact decoded histogram statistics globally and per episode again, not only an audit status flag.
    stats=read(target/'meta/stats.json');rows={row['episode_index']:row for row in target_layout['episodes']};hist={key:np.zeros((3,256),dtype=np.int64) for key in CAMERAS}
    for record in audit['video_records']:
        job=record['job'];key=job['camera_key'];hist[key]+=np.asarray(record['audit']['output_histogram_rgb'],dtype=np.int64)
        for statistic,value in record['statistics'].items():require(rows[job['episode_index']][f'stats/{key}/{statistic}']==value,'Per-episode RGB statistics changed')
    for key in CAMERAS:require(engine.histogram_stats(hist[key])==stats[key],'Global RGB histogram statistics changed')
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    metadata=LeRobotDatasetMetadata(repo_id='local/igris-c-seeded-'+preset,root=target)
    require(metadata.total_episodes==plan['expected_episodes'] and metadata.total_frames==plan['total_frames'],'Official final metadata reload differs')
    return {'preset':preset,'dataset':str(target),'episodes':plan['expected_episodes'],'frames':plan['total_frames'],'audit_sha256':sha(target/AUDIT),'verified':True}
