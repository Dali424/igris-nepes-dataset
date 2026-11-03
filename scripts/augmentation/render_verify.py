#!/usr/bin/env python3
"""Verify a NEW native render episode, then label an immutable LeRobot derivative.

Native actions/states belong to the child, never to its linked parent condition.
Only augmentation provenance changes in the derivative; MP4 bytes are preserved.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

from render_scene import CASES, read, require, sha, write

FLAG = 'provenance.is_augmented'
CAMERAS = {'head': (1280, 720), 'wrist_left': (640, 480), 'wrist_right': (640, 480), 'third_person': (1280, 720)}

def compact_module(bundle):
    sys.path.insert(0, str(Path(bundle) / 'runtime/left2/source/robros_motion_planning'))
    from robros_motion_planning import compact_dataset
    compact_dataset._official_version()
    return compact_dataset

def digest_column(table, name, features):
    import numpy as np
    value = np.ascontiguousarray(np.asarray(table[name].to_pylist(), dtype=features[name]['dtype']))
    return hashlib.sha256(value.tobytes()).hexdigest()

def video_paths(root):
    import pyarrow.parquet as pq
    info = read(root / 'meta/info.json')
    tables = sorted((root / 'meta/episodes').rglob('*.parquet'))
    require(len(tables) == 1, 'Expected a one-episode metadata file')
    rows = pq.read_table(tables[0]).to_pylist()
    require(len(rows) == 1 and rows[0]['episode_index'] == 0, 'Expected one episode')
    row = rows[0]
    videos = {name: root / info['video_path'].format(video_key='observation.images.' + name,
              chunk_index=row['videos/observation.images.' + name + '/chunk_index'],
              file_index=row['videos/observation.images.' + name + '/file_index']) for name in CAMERAS}
    require(all(path.resolve().is_relative_to(root.resolve()) and path.is_file() for path in videos.values()), 'Escaped/missing video')
    return info, tables[0], videos

def histogram_stats(hist):
    import numpy as np
    counts = hist.sum(axis=1); require(counts[0] > 0 and np.all(counts == counts[0]), 'Invalid RGB histogram')
    x = np.arange(256, dtype=float) / 255
    mean = (hist * x).sum(axis=1) / counts
    values = {'min': np.array([x[np.flatnonzero(h)[0]] for h in hist]),
              'max': np.array([x[np.flatnonzero(h)[-1]] for h in hist]),
              'mean': mean, 'std': np.sqrt(np.maximum(0, (hist * x**2).sum(axis=1) / counts - mean**2))}
    for q in (1, 10, 50, 90, 99):
        values[f'q{q:02d}'] = np.array([x[np.searchsorted(np.cumsum(h), n * q / 100)] for h, n in zip(hist, counts)])
    return {**{key: value.reshape(3, 1, 1).tolist() for key, value in values.items()}, 'count': [int(counts[0])]}

def decoded_video(path, size, frame_count):
    import av
    import numpy as np
    histogram = np.zeros((3, 256), dtype=np.int64)
    selected = {0, round((frame_count - 1) * .15), round((frame_count - 1) * .5), round((frame_count - 1) * .85), frame_count - 1}
    sample, count, time_hash = {}, 0, hashlib.sha256()
    with av.open(str(path)) as container:
        require(len(container.streams.video) == 1 and not container.streams.audio, 'Expected a silent one-stream video')
        stream = container.streams.video[0]
        require((stream.width, stream.height) == size and stream.average_rate == 30, 'Camera size/fps mismatch')
        time_base = stream.time_base
        for index, frame in enumerate(container.decode(stream)):
            require(frame.pts is not None and abs(float(frame.pts * frame.time_base) - index / 30) < 1e-6, 'Nonuniform encoded frame time')
            rgb = frame.to_ndarray(format='rgb24')
            require(rgb.shape == (size[1], size[0], 3), 'Decoded RGB dimensions differ')
            for channel in range(3):
                histogram[channel] += np.bincount(rgb[:, :, channel].ravel(), minlength=256)
            if index in selected:
                sample[index] = rgb
            time_hash.update(f'{frame.pts}:{frame.time_base}\n'.encode()); count += 1
    require(count == frame_count, 'Wrong fully decoded video frame count')
    stats = histogram_stats(histogram)
    return stats, sample, {'decoded_frames': count, 'time_base': str(time_base), 'pts_sha256': time_hash.hexdigest(),
                           'encoded_rgb_mean': np.asarray(stats['mean']).reshape(-1).tolist(),
                           'endpoint_fraction': float((histogram[:, 0].sum() + histogram[:, 255].sum()) / histogram.sum())}

def first_video_frame(path):
    import av
    with av.open(str(path)) as container:
        return next(container.decode(video=0)).to_ndarray(format='rgb24')


def verify_visual_binding(bundle, child, parent_verification):
    """Check a prepared child against its parent before success/failure admission."""
    compact_module(bundle)
    from robros_motion_planning.stationary_catalog import load_stationary_config
    child = Path(child).resolve(strict=True); parent = read(parent_verification)
    parent_path = Path(parent['attempt']) / 'profile/config.json'
    config = load_stationary_config(child / 'profile/config.json')
    original = load_stationary_config(parent_path)
    restored = deepcopy(config)
    for key in ('scene_usd', 'scene_sha256'):
        restored['single_arm_layout'][key] = original['single_arm_layout'][key]
    require(restored == original, 'Child changed nonvisual settings')
    visual = read(child / 'profile/visual.json')
    require(visual['case'] in CASES and visual['recipe'] == CASES[visual['case']], 'Visual recipe differs')
    require(Path(visual['parent_config']).resolve() == parent_path.resolve()
            and sha(parent_path) == visual['parent_config_sha256'], 'Visual parent binding changed')
    require(sha(child / 'profile/config.json') == visual['config_sha256'], 'Visual child config changed')
    overlay = Path(visual['overlay']).resolve(strict=True)
    require(overlay.is_relative_to(child / 'profile') and sha(overlay) == visual['overlay_sha256'], 'Visual overlay changed/escaped')
    require(config['single_arm_layout']['scene_usd'] == str(overlay)
            and config['single_arm_layout']['scene_sha256'] == sha(overlay), 'Native scene differs from prepared overlay')
    require(visual['all_physics_geometry_poses_and_cameras_unchanged'] is True
            and visual['only_visual_allowlisted_properties_authored'] is True
            and visual['stage_metadata_unchanged'] is True, 'Incomplete visual-isolation evidence')
    return parent, config, visual


def verify_failure(bundle, child, parent_verification, returncode):
    from source_selection import seeded_module
    parent, _, visual = verify_visual_binding(bundle, child, parent_verification)
    sample = {'profile': parent['profile'], 'pair': parent['pair'], 'seed': parent['seed'],
              'config_sha256': sha(Path(child) / 'profile/config.json')}
    evidence = seeded_module('merge_seeded', bundle).verify_reported_physical_failure(bundle, child, returncode, sample)
    evidence.update(parent_verification_sha256=sha(parent_verification), case=visual['case'],
                    visual_overlay_sha256=visual['overlay_sha256'])
    return evidence

def verify_native(bundle, child, parent_verification):
    verify_visual_binding(bundle, child, parent_verification)
    compact = compact_module(bundle)
    from robros_motion_planning.compact_storage import verify_archived_success
    from robros_motion_planning.stationary_catalog import load_stationary_config, config_digest
    child = Path(child).resolve(strict=True); parent = read(parent_verification)
    document = read(child / 'collection/collection.json'); trials = document['attempts']
    require(len(trials) == 1, 'Each render attempt must contain exactly one native attempt')
    trial = trials[0]
    require(trial.get('accepted') is True and trial.get('storage_complete') is True and trial.get('returncode') == 0, 'Native/storage success required')
    require(trial['seed'] == parent['seed'], 'Child must reuse parent condition seed')
    archive = Path(verify_archived_success(child / 'collection', trial, document['plan'])).resolve(strict=True)
    require(archive.is_relative_to(child), 'Native archive escaped child output')
    receipt = read(archive / compact.RECEIPT_PATH)
    require(receipt['files'] == compact._inventory(archive), 'Native archive changed after verification')
    run = Path(trial['run_dir']).resolve(strict=True); require(run.is_relative_to(child), 'Native run escaped child')
    report = read(run / 'run.json'); parent_report = read(Path(parent['run']) / 'run.json')
    from source_selection import seeded_module
    seeded_module('merge_seeded', bundle).upper_body_ik_evidence(bundle, report)  # Pink release: declared backend, closed cleanly
    config = load_stationary_config(child / 'profile/config.json')
    parent_config = load_stationary_config(Path(parent['attempt']) / 'profile/config.json')
    restored = deepcopy(config)
    for key in ('scene_usd', 'scene_sha256'):
        restored['single_arm_layout'][key] = parent_config['single_arm_layout'][key]
    require(restored == parent_config, 'Child changed nonvisual task/control/camera settings')
    config_hash = config_digest(config)
    require(report['config_sha256'] == document['plan']['config_sha256'] == config_hash, 'Native configuration binding mismatch')
    require(report['controller_contract'] == parent_report['controller_contract'], 'HugWBC/free-pelvis/no-attachment physical contract changed')
    require(report['camera_configuration'] == parent_report['camera_configuration'], 'Actual camera configuration changed')
    require(report['robot_gripper'] == parent_report['robot_gripper'], 'Robot/gripper physics contract changed')
    require(report['physics_scene_ccd'] == parent_report['physics_scene_ccd'], 'Physics timing/CCD changed')
    require(report['hugwbc_policy']['checkpoint_sha256'] == parent_report['hugwbc_policy']['checkpoint_sha256'], 'HugWBC checkpoint changed')
    require(len(report['episodes']) == 1 and report['episodes'][0]['success'] is True and report['episodes'][0]['full_task_completed'] is True, 'Full physical task success required')
    visual = read(child / 'profile/visual.json')
    require(sha(child / 'profile/config.json') == visual['config_sha256'] and sha(visual['overlay']) == visual['overlay_sha256'], 'Visual preparation changed')
    require(visual['all_physics_geometry_poses_and_cameras_unchanged'] and visual['only_visual_allowlisted_properties_authored'], 'Visual isolation receipt missing')
    require(sha(visual['parent_config']) == visual['parent_config_sha256'], 'Parent config changed')
    return compact, archive, report, visual, parent

def finalize(bundle, child, parent_verification, output):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    compact, archive, report, visual, parent = verify_native(bundle, child, parent_verification)
    output = Path(output).resolve(); require(not output.exists(), 'Refuse to overwrite a published render derivative')
    native_receipt_sha = sha(archive / compact.RECEIPT_PATH)
    parent_archive = Path(parent['archive']).resolve(strict=True)
    for protected in (Path(bundle).resolve(), archive, parent_archive, Path(parent['attempt']).resolve(), Path(child).resolve() / 'collection'):
        require(not output.is_relative_to(protected) and not protected.is_relative_to(output), 'Derivative output overlaps source/collection/bundle')
    require(sha(parent_archive / compact.RECEIPT_PATH) == parent['archive_receipt_sha256'], 'Linked parent receipt changed')
    info, ep_path, videos = video_paths(archive)
    _, _, parent_videos = video_paths(parent_archive)
    require(set(info['features'][k]['dtype'] for k in info['features'] if k.startswith('observation.images.')) == {'video'}, 'Wrong camera schema')
    updates, video_audits, comparison = {}, {}, {}
    for camera, path in videos.items():
        print('Verifying encoded render camera:', camera, flush=True)
        values, samples, audit = decoded_video(path, CAMERAS[camera], info['total_frames'])
        baseline, rendered = first_video_frame(parent_videos[camera]), samples[0]
        difference = rendered.astype(float) - baseline.astype(float)
        parent_clip = np.mean((baseline == 0) | (baseline == 255))
        child_clip = np.mean((rendered == 0) | (rendered == 255))
        comparison[camera] = {'first_frame_mean_absolute_difference_u8': float(np.abs(difference).mean()),
                              'first_frame_signed_mean_difference_u8': float(difference.mean()),
                              'first_frame_added_endpoint_fraction': float(child_clip - parent_clip),
                              'first_frame_std_u8': float(rendered.std()),
                              'first_frame_parent_mean_u8': float(baseline.mean()), 'first_frame_child_mean_u8': float(rendered.mean())}
        require(all(float(image.std()) > 2.0 for image in samples.values()), 'Blank/degenerate sampled view: ' + camera)
        require(child_clip - parent_clip < .20, 'Excessive additional clipping: ' + camera)
        updates['observation.images.' + camera] = values; video_audits[camera] = audit
    changed = sum(value['first_frame_mean_absolute_difference_u8'] >= 2.0 for value in comparison.values())
    require(changed >= 2, 'Render appearance effect below pilot/episode visibility gate in most views')
    sign = -1 if visual['case'] == 'dim_matte' else 1
    directional = sum(sign * value['first_frame_signed_mean_difference_u8'] >= .5 for value in comparison.values())
    require(directional >= 2, 'Encoded images do not show expected light direction in at least two views')
    data_paths = sorted((archive / 'data').rglob('*.parquet')); require(len(data_paths) == 1, 'Expected one native data file')
    original = pq.read_table(data_paths[0]); require(original.num_rows == info['total_frames'], 'Data/video frame count mismatch')
    require(all(value in (0, [0]) for value in original[FLAG].to_pylist()), 'Native original augmentation flag was not zero')
    table = original.set_column(original.schema.get_field_index(FLAG), original.schema.field(FLAG),
                               pa.array([1 if value == 0 else [1] for value in original[FLAG].to_pylist()], type=original[FLAG].type))
    preserved = [name for name in original.column_names if name != FLAG]
    digests = {name: digest_column(original, name, info['features']) for name in preserved}
    target, temporary = compact._destination(output)
    try:
        shutil.copytree(archive, temporary, ignore=lambda path, names: [compact.RECEIPT_PATH.name] if Path(path).name == 'meta' else [])
        pq.write_table(table, temporary / data_paths[0].relative_to(archive), compression='zstd')
        reread = pq.read_table(temporary / data_paths[0].relative_to(archive))
        require(original.select(preserved).equals(reread.select(preserved)) and original.schema == reread.schema, 'Native numeric columns changed')
        stats = read(temporary / 'meta/stats.json')
        updates[FLAG] = {key: [0.0 if key == 'std' else info['total_frames'] if key == 'count' else 1.0] for key in stats[FLAG]}
        episodes = pq.read_table(ep_path)
        for key, values in updates.items():
            stats[key] = values
            for statistic, value in values.items():
                column = f'stats/{key}/{statistic}'; field = episodes.schema.field(column)
                episodes = episodes.set_column(episodes.schema.get_field_index(column), field, pa.array([value], type=field.type))
        pq.write_table(episodes, temporary / ep_path.relative_to(archive), compression='zstd')
        write(temporary / 'meta/stats.json', stats)
        extension = read(temporary / 'meta/nepes_collection.json')
        linkage = {'parent_condition_archive': str(parent_archive), 'parent_condition_receipt_sha256': parent['archive_receipt_sha256'],
                   'parent_verification_sha256': sha(parent_verification), 'parent_profile': parent['profile'], 'parent_seed': parent['seed'],
                   'native_child_archive': str(archive), 'native_child_receipt_sha256': native_receipt_sha,
                   'child_config_sha256': report['config_sha256'], 'case': visual['case'], 'recipe': visual['recipe'],
                   'visual_overlay_sha256': visual['overlay_sha256'], 'new_physics_execution': True,
                   'parent_motion_exact_sha256': parent['motion_exact_sha256'], 'split_group': 'seeded_motion_' + parent['motion_exact_sha256'],
                   'parent_actions_states_timestamps_reused': False, 'numeric_payload_origin': 'new_successful_native_child'}
        extension['episodes'][0]['provenance'].update(kind='augmentation', parent_episode_id=parent['archive_receipt_sha256'],
                                                     augmentations=[{'kind': 'native_lighting_material_rerun', **linkage}])
        write(temporary / 'meta/nepes_collection.json', extension)
        audit = {'schema': 'nepes_native_render_augmentation_v1', 'verified': True, **linkage,
                 'physical_success_revalidated': True, 'physics_controller_cameras_preserved': True,
                 'encoded_video_bytes_preserved_from_native_child': True, 'native_child_numeric_column_sha256': digests,
                 'changed_numeric_columns': [FLAG], 'videos': video_audits, 'appearance_qa': comparison,
                 'appearance_qa_thresholds': {'minimum_changed_views': 2, 'mean_absolute_difference_u8': 2.0,
                                            'minimum_light_direction_views': 2, 'signed_mean_difference_u8': .5,
                                            'maximum_added_endpoint_fraction': .20},
                 'limitations': ['First-frame appearance comparison is a coarse visibility gate, not a photometric calibration or pixel-aligned causality proof.',
                                'NuRec captured background may remain fixed while CAD is relit.',
                                'The parent identifies a starting condition; this child has independently measured actions/states/timestamps.']}
        write(temporary / 'meta/render_augmentation.json', audit)
        receipt = compact.verify_dataset(temporary, expected_episodes=1, video_workers=1)
        require(sha(archive / compact.RECEIPT_PATH) == native_receipt_sha and compact._inventory(archive) == read(archive / compact.RECEIPT_PATH)['files'], 'Native child source changed')
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        loaded = LeRobotDataset(repo_id='local/native-render-check', root=temporary, video_backend='pyav')
        for index in (0, info['total_frames'] // 2, info['total_frames'] - 1):
            item = loaded[index]
            require(all(tuple(item['observation.images.' + name].shape) == (3, size[1], size[0]) for name, size in CAMERAS.items()), 'Official loader four-camera retrieval failed')
        receipt['checks'].update(native_child_numeric_values_preserved=True, native_child_video_bytes_preserved=True,
                                 augmentation_provenance_explicit=True, official_first_middle_last_loaded=True)
        compact._publish(temporary, target, receipt)
        return {'output': str(target), 'receipt_sha256': sha(target / compact.RECEIPT_PATH), 'audit': audit,
                'frames': info['total_frames'], 'duration_s': report['episodes'][0]['duration_s']}
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'child', 'parent-verification', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--failure-returncode', type=int)
    args = parser.parse_args()
    if args.failure_returncode is not None:
        result = verify_failure(args.bundle, args.child, args.parent_verification, args.failure_returncode)
        if args.output.exists():
            require(read(args.output) == result, 'Saved failure verification changed')
        else:
            write(args.output, result)
    else:
        result = finalize(args.bundle, args.child, args.parent_verification, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != 'audit'}, indent=2))

if __name__ == '__main__':
    main()
