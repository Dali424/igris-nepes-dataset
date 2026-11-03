"""Static USD state restoration, calibrated cameras and isolated MP4 writers."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

from .fk import URDF, pose_to_T, T_err
from .trace import require, read, sha

ROBOT = '/World/envs/env_0/Robot'


def native_pids():
    result = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            maps = (proc / 'maps').read_text()
            if 'libomni.kit.app.plugin.so' in maps or 'libcarb.app.plugin.so' in maps:
                result.append(int(proc.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return result


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def render(args, source, states, selected, plan):
    lock = (Path(tempfile.gettempdir()) / f'igris-nepes-{os.getuid()}.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not native_pids(), 'Another Isaac/Kit process is active; wait before visual replay')
    # Check the encoder before loading the much larger renderer.
    child_env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                     OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    child_env.pop('PYTHONPATH', None)
    subprocess.run([str(args.dataset_python), '-B', '-c', 'import av,numpy'], env=child_env, check=True)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'replay_plan.json', plan)
    runtime = source['runtime']
    sys.path[:0] = [str(runtime / 'source/robros_motion_planning'), str(runtime / 'source/robros_lab')]
    from robros_motion_planning.camera_capture import prepare_calibrated_camera_plugins
    prepare_calibrated_camera_plugins(source['config'])
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    launch = parser.parse_args([])
    launch.headless, launch.device, launch.enable_cameras = not args.gui, 'cpu', True
    launch.kit_args = '--/UJITSO/geometry=true --/renderer/multiGpu/enabled=false --/renderer/multiGpu/maxGpuCount=1'
    app = None
    encoders, products, annotators, cameras = {}, {}, {}, {}
    started = time.monotonic()
    receipt = {'schema': 'nepes_visual_replay_render_receipt_v1', 'status': 'running',
               'native_physics_reexecuted': False, 'action_replay': False,
               'new_success_claim': False, 'original_rgb_reproduced_exactly': False,
               'plan_sha256': sha(args.output / 'replay_plan.json'), 'frames_written': 0,
               'source_files_unchanged': False, 'usd_readback_verified': False, 'cleanup_errors': []}
    maximum_errors = {'link_position_m': 0., 'link_rotation_deg': 0., 'object_position_m': 0.,
                      'object_rotation_deg': 0., 'joint_relative_position_m': 0.,
                      'joint_relative_rotation_deg': 0., 'base_position_m': 0., 'base_rotation_deg': 0.}
    try:
        app = AppLauncher(launch).app
        import omni.timeline
        import omni.usd
        import omni.replicator.core as rep
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics
        import isaaclab.sim as sim_utils
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension('isaacsim.sensors.camera')
        from isaacsim.sensors.camera import Camera
        from robros_motion_planning.camera_capture import (CameraSpec, _set_camera_optics,
                                                           _camera_intrinsics_metadata,
                                                           _ensure_opencv_pinhole_schema)
        from robros_motion_planning.latest_robot import build_latest_robot_variant
        from robros_lab.tasks.base.usd_scene import UsdSceneCfg, apply_usd_scene_render_settings

        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        origin_time = float(timeline.get_current_time())

        def update():
            require(not timeline.is_playing(), 'Replay physics timeline unexpectedly started')
            require(app.is_running(), 'Replay window was closed')
            app.update()
            require(not timeline.is_playing() and abs(float(timeline.get_current_time()) - origin_time) < 1e-12,
                    'Replay advanced physical simulation time')

        def matrix(value):
            return np.asarray(value, dtype=float).T

        def gf(value):
            return Gf.Matrix4d(*np.asarray(value, dtype=float).T.reshape(-1).tolist())

        def rigid(value):
            left, _, right = np.linalg.svd(value[:3, :3])
            result = np.eye(4)
            result[:3, :3] = left @ np.diag([1, 1, np.linalg.det(left @ right)]) @ right
            result[:3, 3] = value[:3, 3]
            return result

        def check(actual, expected, kind):
            distance, angle = T_err(rigid(actual), expected)
            require(distance <= .001 and angle <= .1,
                    f'USD {kind} readback differs: {distance}m, {angle}deg')
            maximum_errors[kind + '_position_m'] = max(maximum_errors[kind + '_position_m'], distance)
            maximum_errors[kind + '_rotation_deg'] = max(maximum_errors[kind + '_rotation_deg'], angle)

        context = omni.usd.get_context()
        context.new_stage()
        for _ in range(5):
            update()
        stage = context.get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.Xform.Define(stage, '/World')
        scene_cfg = UsdSceneCfg(usd_path=str(source['scene']), inactive_prim_paths=('igris_c',))
        scene_cfg.func('/World/Scene', scene_cfg)
        for path in ('/World/envs', '/World/envs/env_0'):
            UsdGeom.Xform.Define(stage, path)
        original = source['report']['robot_gripper']
        variant = build_latest_robot_variant(args.output / 'robot_model',
            model_dir=source['bundle'] / 'assets/robot', gripper_variant=original['variant'],
            pad_material_override=original.get('dev_pad_material_override'),
            pad_contact_patch=original.get('pad_contact_patch'),
            pad_compliant_contact=original.get('pad_compliant_contact'),
            pad_contact_spheres=original.get('pad_contact_spheres'))
        require(variant.metadata['fingerprint'] == original['robot_model_fingerprint'],
                'Packaged robot/gripper differs from the recorded model')
        robot_cfg = sim_utils.UsdFileCfg(usd_path=str(variant.usd_path))
        robot_cfg.func(ROBOT, robot_cfg)
        for _ in range(6000):
            update()
            status = context.get_stage_loading_status()
            if not status or status[2] == 0:
                break
        else:
            raise TimeoutError('Replay assets did not finish loading')
        receipt['render_settings'] = apply_usd_scene_render_settings(str(source['scene']))
        from .visual_overlay import apply_visual_overlay
        receipt['visual_overlay'] = apply_visual_overlay(stage, source.get('visual_overlay'))
        require(not [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)],
                'Replay must not contain an active physics scene')
        tree = URDF(source['urdf'])
        zero = tree.fk({name: 0. for name in tree.movable})
        root = stage.GetPrimAtPath(ROBOT)
        rigid_links = {p.GetName() for p in root.GetChildren() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
        require(set(zero) == rigid_links, 'Recorded URDF and packaged USD rigid link sets differ')
        cache = UsdGeom.XformCache()
        require(np.allclose(matrix(cache.GetLocalToWorldTransform(root)), np.eye(4), rtol=0, atol=1e-10),
                'Robot parent must be identity for measured world link reconstruction')
        link_ops = {}
        for name, expected in zero.items():
            prim = stage.GetPrimAtPath(ROBOT + '/' + name)
            check(matrix(cache.GetLocalToWorldTransform(prim)), expected, 'link')
            link_ops[name] = UsdGeom.Xformable(prim).MakeMatrixXform()
        objects = states[0]['object_poses_w']
        for path in objects:
            prim = stage.GetPrimAtPath(path)
            require(prim.IsValid() and prim.IsActive(), 'Recorded object is absent from packaged scene: ' + path)
            while prim.IsInstanceProxy():
                ancestor = prim.GetParent()
                while ancestor and not ancestor.IsInstance():
                    ancestor = ancestor.GetParent()
                require(bool(ancestor), 'Cannot expand object instance: ' + path)
                ancestor.SetInstanceable(False)
                prim = stage.GetPrimAtPath(path)
        cache = UsdGeom.XformCache()
        object_ops = {}
        for path in sorted(objects, key=lambda p: p.count('/')):
            prim = stage.GetPrimAtPath(path)
            authored = matrix(cache.GetLocalToWorldTransform(prim))
            remainder = np.linalg.inv(rigid(authored)) @ authored
            object_ops[path] = (UsdGeom.Xformable(prim).MakeMatrixXform(), remainder)
        specs = [CameraSpec(**c) for c in source['cameras'] if args.views == 'all' or c['name'] == 'third_person']
        if any(s.calibrated_intrinsics is not None for s in specs):
            _ensure_opencv_pinhole_schema()
        UsdGeom.Xform.Define(stage, '/World/ReplayCameras')
        receipt['camera_optics'] = {}
        for spec in specs:
            path = '/World/ReplayCameras/' + spec.name
            camera = Camera(prim_path=path, name='replay_' + spec.name, resolution=(spec.width, spec.height),
                            frequency=-1, annotator_device='cpu')
            cameras[spec.name] = camera
            _set_camera_optics(camera, spec)
            camera.set_clipping_range(near_distance=spec.near_clip_m, far_distance=spec.far_clip_m)
            # No Camera.initialize(): the Replicator products below render while timeline is stopped.
            intrinsics, extra = _camera_intrinsics_metadata(camera, spec)
            receipt['camera_optics'][spec.name] = {'intrinsics': intrinsics.tolist(), **extra}
            product = rep.create.render_product(path, (spec.width, spec.height))
            products[spec.name] = product
            annotator = rep.AnnotatorRegistry.get_annotator('rgb')
            annotator.attach([product])
            annotators[spec.name] = annotator
            video = args.output / (spec.name + '.mp4')
            log = (args.output / (spec.name + '.encode.log')).open('w')
            command = [str(args.dataset_python), '-B', str(Path(__file__).with_name('encode.py')),
                       '--output', str(video), '--width', str(spec.width), '--height', str(spec.height),
                       '--fps', str(args.fps), '--expected-frames', str(len(selected))]
            encoders[spec.name] = (subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log,
                                                   stderr=subprocess.STDOUT, env=child_env), log)
        try:
            from omni.kit.viewport.utility import get_active_viewport
            viewport = get_active_viewport()
            if viewport is not None:
                viewport.updates_enabled = args.gui
                if args.gui:
                    viewport.camera_path = '/World/ReplayCameras/third_person'
        except Exception as error:
            receipt['viewport_note'] = repr(error)
        usd_axis_conversion = np.eye(4)
        usd_axis_conversion[:3, :3] = [[0, -1, 0], [0, 0, 1], [-1, 0, 0]]
        with (args.output / 'replay_frames.jsonl').open('x') as mapping:
            for frame in selected:
                row = states[frame['trace_index']]
                local_links = tree.fk(row['joint_positions'])
                world = pose_to_T(row['base_pose_w'])
                links = {name: world @ value for name, value in local_links.items()}
                with Sdf.ChangeBlock():
                    for name, op in link_ops.items():
                        op.Set(gf(links[name]))
                for path, (op, remainder) in object_ops.items():
                    cache = UsdGeom.XformCache()
                    parent = stage.GetPrimAtPath(path).GetParent()
                    parent_w = matrix(cache.GetLocalToWorldTransform(parent))
                    op.Set(gf(np.linalg.inv(parent_w) @ pose_to_T(row['object_poses_w'][path]) @ remainder))
                cache = UsdGeom.XformCache()
                actual_links = {name: matrix(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(ROBOT + '/' + name)))
                                for name in links}
                for name, value in actual_links.items():
                    check(value, links[name], 'link')
                check(actual_links[tree.root], world, 'base')
                for joint in tree.joints.values():
                    actual = np.linalg.inv(actual_links[joint['parent']]) @ actual_links[joint['child']]
                    expected = np.linalg.inv(local_links[joint['parent']]) @ local_links[joint['child']]
                    check(actual, expected, 'joint_relative')
                for path in objects:
                    check(matrix(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))),
                          pose_to_T(row['object_poses_w'][path]), 'object')
                for spec in specs:
                    local = pose_to_T([*spec.translation, *spec.orientation_wxyz])
                    camera_w = links[spec.attachment_prim_path.rsplit('/', 1)[-1]] @ local if spec.attachment_prim_path else local
                    camera = cameras[spec.name]
                    prim = stage.GetPrimAtPath('/World/ReplayCameras/' + spec.name)
                    UsdGeom.Xformable(prim).MakeMatrixXform().Set(gf(camera_w @ np.linalg.inv(usd_axis_conversion)))
                    actual = matrix(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
                    require(np.allclose(actual, camera_w @ np.linalg.inv(usd_axis_conversion), atol=1e-9, rtol=0),
                            'Camera pose readback mismatch')
                for _ in range(args.warmup_frames if frame['frame_index'] == 0 else args.render_updates):
                    update()
                for spec in specs:
                    pixels = np.asarray(annotators[spec.name].get_data())
                    require(pixels.dtype == np.uint8 and pixels.shape in ((spec.height, spec.width, 3), (spec.height, spec.width, 4)),
                            'Invalid RGB frame for ' + spec.name)
                    rgb = np.ascontiguousarray(pixels[..., :3])
                    encoders[spec.name][0].stdin.write(rgb.tobytes())
                    if frame['frame_index'] in (0, len(selected) - 1):
                        from PIL import Image
                        Image.fromarray(rgb).save(args.output / f'{spec.name}_{frame["frame_index"]:06d}.png')
                mapping.write(json.dumps({**frame, 'source_raw_line_sha256': row['raw_line_sha256'],
                                          'stage': row['stage'], 'usd_readback_verified': True}) + '\n')
                receipt['frames_written'] += 1
                if receipt['frames_written'] % 30 == 0:
                    print(f'Replay rendered {receipt["frames_written"]}/{len(selected)} frames', flush=True)
            mapping.flush()
        for name, (encoder, log) in encoders.items():
            encoder.stdin.close()
            require(encoder.wait(timeout=120) == 0, 'MP4 encoder/verification failed: ' + name)
            log.flush()
        require(all(sha(path) == expected for path, expected in source['sources'].items()), 'Replay input changed')
        receipt.update(status='complete', source_files_unchanged=True, usd_readback_verified=True,
                       maximum_usd_readback_errors=maximum_errors, timeline_playing=bool(timeline.is_playing()),
                       timeline_time_s=float(timeline.get_current_time()),
                       videos={name: read(args.output / (name + '.verification.json')) for name in encoders})
    except BaseException:
        receipt.update(status='failed', error=traceback.format_exc())
        print(receipt['error'], file=sys.stderr, flush=True)
    finally:
        for name, (encoder, log) in encoders.items():
            if encoder.poll() is None:
                encoder.terminate()
                try:
                    encoder.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    encoder.kill(); encoder.wait(timeout=10)
            log.close()
        for name, annotator in annotators.items():
            try:
                annotator.detach()
            except Exception as error:
                receipt['cleanup_errors'].append(name + ': ' + repr(error))
        for product in products.values():
            try:
                product.destroy()
            except Exception as error:
                receipt['cleanup_errors'].append(repr(error))
        for camera in cameras.values():
            try:
                camera.destroy()
            except Exception as error:
                receipt['cleanup_errors'].append(repr(error))
        receipt['wall_seconds'] = time.monotonic() - started
        save(args.output / 'replay_receipt.json', receipt)
        if app is not None:
            app.close()
        lock.close()
    return 0 if receipt['status'] == 'complete' else 2
