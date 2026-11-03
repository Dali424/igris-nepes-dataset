"""Stream retained control traces; never interpret commands as measurements."""
from __future__ import annotations

import bisect
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .fk import URDF, pose_to_T, T_err


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def bound(root, path):
    path = path.resolve(strict=True)
    require(path.is_relative_to(root), 'Input path escapes the supplied attempt: ' + str(path))
    return path


def pose(value, name):
    array = np.asarray(value, dtype=float)
    require(array.shape == (7,) and np.isfinite(array).all(), 'Invalid xyz+wxyz pose: ' + name)
    require(abs(np.linalg.norm(array[3:]) - 1) <= .001, 'Non-unit quaternion: ' + name)
    return value


def discover(attempt, bundle, episode_id=None):
    attempt, bundle = Path(attempt).resolve(strict=True), Path(bundle).resolve(strict=True)
    if (attempt / 'meta/info.json').exists():
        raise ValueError('LeRobot MP4/Parquet alone lacks robot base and object trajectories. '
                         'Use the retained collection attempt with trace.jsonl[.gz]; '
                         'the original videos can be viewed directly without Isaac. '
                         'Offline brightness/contrast derivatives are reproduced by their saved MP4, not by scene replay.')
    runs = [attempt] if (attempt / 'run.json').is_file() else list(attempt.glob('collection/runs/*'))
    runs = [bound(attempt, p) for p in runs if (p / 'run.json').is_file()]
    require(len(runs) == 1, 'Supply one completed native run or one outer attempt, not an entire batch')
    run = runs[0]
    report = read(run / 'run.json')
    require(report.get('complete') is True, 'Run is incomplete; replay only closed, retained traces')
    episodes = report.get('episodes', [])
    if episode_id:
        episodes = [e for e in episodes if e.get('episode_id') == episode_id]
    require(len(episodes) == 1, 'Select one recorded episode with --episode')
    episode = episodes[0]
    episode_root = bound(attempt, run / episode['episode_id'])
    result = bound(attempt, episode_root / 'result.json')
    require(read(result) == episode, 'Episode result differs from the run report')
    traces = [p for p in (episode_root / 'trace.jsonl', episode_root / 'trace.jsonl.gz') if p.is_file()]
    require(len(traces) == 1, 'Need exactly one retained trace.jsonl or trace.jsonl.gz')
    trace = bound(attempt, traces[0])
    config_path = bound(attempt, run / 'config.json')
    config = read(config_path)
    pair = config.get('user_slot_pair', {}).get('number')
    require(pair in (2, 4), 'Only the packaged slot2/slot4 CRS collection profiles are supported')
    profile = 'left2' if pair == 2 else 'right4'
    runtime = bundle / 'runtime' / profile
    require(runtime.is_dir(), 'Missing packaged replay runtime: ' + str(runtime))
    # Ignore stale absolute paths in run.json. Search only inside the copied run.
    fingerprint = report['robot_gripper']['robot_model_fingerprint']
    models = [p for p in run.glob('robot_model/*/manifest.json') if read(p).get('fingerprint') == fingerprint]
    require(len(models) == 1, 'Retain the recorded robot_model directory with the trace')
    manifest_path = bound(attempt, models[0])
    manifest = read(manifest_path)
    urdf = bound(attempt, manifest_path.parent / 'igris_c_260910_canonical.urdf')
    require(sha(urdf) == manifest['derived_hashes']['urdf'], 'Recorded canonical URDF digest differs')
    scene = bundle / 'assets/workcell/layout.usda'
    require(scene.is_file(), 'Packaged workcell layout is absent')
    expected_scene = config.get('single_arm_layout', {}).get('scene_sha256')
    require(expected_scene, 'The recorded scene digest is absent')
    visual_overlay = None
    if sha(scene) != expected_scene:
        from .visual_overlay import load_visual_overlay
        visual_overlay = load_visual_overlay(attempt, run, config, scene)
    source_files = [run / 'run.json', result, config_path, trace, manifest_path, urdf, scene]
    sources = {str(p): sha(p) for p in source_files}
    if visual_overlay is not None:
        sources.update(visual_overlay['source_files_sha256'])
    receipt_path = run / 'storage_receipt.json'
    receipt_bound = False
    if receipt_path.exists():
        receipt = read(receipt_path)
        for item in receipt.get('traces', []):
            if trace.suffix == '.gz' and item.get('compressed_path') == str(trace.relative_to(run)):
                require(item.get('compressed_sha256') == sources[str(trace)] and
                        item.get('compressed_size_bytes') == trace.stat().st_size,
                        'Compressed trace differs from its storage receipt')
                receipt_bound = True
        require(receipt_bound or trace.suffix != '.gz', 'Compressed trace is not bound by the storage receipt')
        sources[str(receipt_path)] = sha(receipt_path)
    cameras = report.get('camera_configuration', [])
    require({c.get('name') for c in cameras} == {'head', 'wrist_left', 'wrist_right', 'third_person'},
            'Run does not contain the four recorded camera specifications')
    return {'attempt': attempt, 'run': run, 'trace': trace, 'urdf': urdf, 'scene': scene,
            'bundle': bundle, 'runtime': runtime, 'config': config, 'report': report,
            'episode': episode, 'robot_manifest': manifest, 'cameras': cameras,
            'sources': sources, 'storage_trace_binding_verified': receipt_bound, 'profile': profile,
            'visual_overlay': visual_overlay}


def read_states(source):
    """Keep only measured pose/joint fields, not the very large contact diagnostics."""
    result, previous, object_names, joint_names = [], -math.inf, None, None
    decoded_sha = hashlib.sha256()
    opener = gzip.open if source['trace'].suffix == '.gz' else open
    with opener(source['trace'], 'rb') as stream:
        for index, line in enumerate(stream):
            decoded_sha.update(line)
            row = json.loads(line)
            observation = row['observation']
            t = float(observation['time_s'])
            require(math.isfinite(t) and t > previous, f'Trace time is not strictly increasing at row {index}')
            previous = t
            joints = observation['joint_positions']
            require(joints and all(math.isfinite(float(v)) for v in joints.values()), 'Invalid measured joints')
            objects = observation['object_poses_w']
            require(objects and all(p.startswith('/World/Scene/') for p in objects), 'Unsupported object paths')
            if object_names is None:
                object_names, joint_names = set(objects), set(joints)
            require(set(objects) == object_names and set(joints) == joint_names, 'Recorded entity set changed')
            for name, value in objects.items():
                pose(value, name)
            tcps = observation.get('tcp_poses_w', {})
            require(set(tcps) == {'left', 'right'}, 'Both measured TCP poses are required for FK verification')
            for name, value in tcps.items():
                pose(value, name)
            result.append({'trace_index': index, 'raw_line_sha256': hashlib.sha256(line).hexdigest(),
                           'time_s': t, 'stage': row['command']['stage_id'],
                           'base_pose_w': pose(observation['base_pose_w'], 'base'),
                           'joint_positions': joints, 'object_poses_w': objects,
                           'tcp_poses_w': tcps})
    require(len(result) >= 2, 'At least two measured trace rows are required')
    require(sha(source['trace']) == source['sources'][str(source['trace'])], 'Trace changed while reading')
    return result, decoded_sha.hexdigest()


def schedule(states, fps, start_seconds=0., duration_seconds=None):
    require(isinstance(fps, int) and 1 <= fps <= 120, 'FPS must be an integer in [1,120]')
    require(math.isfinite(start_seconds) and start_seconds >= 0, 'Start must be finite and nonnegative')
    times = [row['time_s'] for row in states]
    start = times[0] + start_seconds
    end = times[-1]
    if duration_seconds is not None:
        require(math.isfinite(duration_seconds) and duration_seconds > 0, 'Duration must be positive')
        end = min(end, start + duration_seconds)
    require(start < end, 'Requested replay interval does not intersect the trace')
    selected = []
    for frame in range(max(1, int(math.ceil((end - start) * fps - 1e-9)))):
        target = start + frame / fps
        index = bisect.bisect_left(times, target)
        options = [i for i in (index - 1, index) if 0 <= i < len(times)]
        nearest = min(options, key=lambda i: (abs(times[i] - target), i))
        selected.append({'frame_index': frame, 'video_time_s': frame / fps,
                         'requested_source_time_s': target, 'trace_index': nearest,
                         'source_time_s': times[nearest], 'sample_skew_s': times[nearest] - target})
    return selected


def validate(source, states, selected):
    tree = URDF(source['urdf'])
    expected = set(tree.movable)
    maximum_position, maximum_rotation = 0., 0.
    for row in states:
        require(set(row['joint_positions']) == expected, 'Measured joints differ from recorded URDF active joints')
        world = pose_to_T(row['base_pose_w'])
        links = tree.fk(row['joint_positions'])
        for side in ('left', 'right'):
            error = T_err(world @ links[side.title() + '_Hand'], pose_to_T(row['tcp_poses_w'][side]))
            maximum_position = max(maximum_position, error[0])
            maximum_rotation = max(maximum_rotation, error[1])
            require(error[0] <= .001 and error[1] <= .1,
                    f'URDF FK differs from measured {side} TCP at row {row["trace_index"]}: {error} m/deg')
    steps = np.diff([row['time_s'] for row in states])
    return {'all_rows_checked': len(states), 'joint_count': len(expected),
            'tracked_objects': sorted(states[0]['object_poses_w']),
            'trace_start_time_s': states[0]['time_s'], 'trace_end_time_s': states[-1]['time_s'],
            'median_trace_period_s': float(np.median(steps)), 'maximum_trace_gap_s': float(max(steps)),
            'maximum_fk_tcp_position_error_m': maximum_position,
            'maximum_fk_tcp_rotation_error_deg': maximum_rotation,
            'scheduled_video_frames': len(selected),
            'maximum_nearest_sample_skew_s': max(abs(x['sample_skew_s']) for x in selected),
            'raw_measurements_preserved': True, 'interpolation': False}
