"""Portable replay CPU checks. No Isaac import, GPU or production data required."""
import argparse
from copy import deepcopy
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np

BUNDLE = Path(__file__).resolve().parents[1]
SCRIPTS = BUNDLE / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from replay_support.trace import discover, pose, read_states, schedule
from replay_support.visual_overlay import (CASES, PROPERTIES, apply_visual_overlay,
                                           digest, float32, load_visual_overlay)

spec = importlib.util.spec_from_file_location('portable_replay_launcher', SCRIPTS / 'replay.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class Dispatched(Exception):
    pass


class ReplayDispatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='nepes-replay-dispatch-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sim, self.dataset = self.root / 'sim/python', self.root / 'dataset/python'
        for executable in (self.sim, self.dataset):
            executable.parent.mkdir()
            executable.symlink_to(sys.executable)
        self.args = argparse.Namespace(bundle=self.root, dataset_python=self.dataset,
                                       isaac_python=self.sim, render=False)

    def invoke(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(launcher.os, 'execve', side_effect=Dispatched) as execute:
            with self.assertRaises(Dispatched):
                launcher.dispatch(self.args)
        return execute.call_args.args

    def test_launcher_import_requires_only_standard_library(self):
        code = ('import importlib.util, sys; '
                'spec=importlib.util.spec_from_file_location("replay",sys.argv[1]); '
                'module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); '
                'assert "numpy" not in sys.modules and "isaaclab" not in sys.modules')
        subprocess.run([sys.executable, '-B', '-S', '-c', code, str(SCRIPTS / 'replay.py')], check=True, timeout=10)

    def test_validation_selects_dataset_environment(self):
        self.assertEqual(self.invoke()[0], str(self.dataset))

    def test_render_selects_simulation_environment(self):
        self.args.render = True
        self.assertEqual(self.invoke()[0], str(self.sim))

    def test_dispatch_marker_prevents_alias_loop(self):
        _, _, environment = self.invoke()
        with patch.dict(os.environ, environment, clear=True), patch.object(launcher.os, 'execve') as execute:
            launcher.dispatch(self.args)
            execute.assert_not_called()

    def test_current_interpreter_does_not_reexecute(self):
        self.args.dataset_python = Path(sys.executable)
        with patch.dict(os.environ, {}, clear=True), patch.object(launcher.os, 'execve') as execute:
            launcher.dispatch(self.args)
            execute.assert_not_called()

    def test_missing_interpreter_reports_selected_environment(self):
        self.args.dataset_python = self.root / 'missing'
        with self.assertRaisesRegex(ValueError, 'NEPES_DATASET_PYTHON'):
            launcher.dispatch(self.args)

    def test_defaults_preserve_venv_path_instead_of_resolved_binary(self):
        for kind in ('sim', 'dataset'):
            executable = self.root / '.venv' / kind / 'bin/python'
            executable.parent.mkdir(parents=True)
            executable.symlink_to(sys.executable)
        self.args.dataset_python = self.args.isaac_python = None
        self.assertEqual(self.invoke()[0], str(self.root / '.venv/dataset/bin/python'))


class ReplayMeasuredStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='nepes-replay-state-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_trace(self, duplicate_time=False):
        rows = []
        for index in range(4):
            measurement = [index * .01, .2, .3, 1, 0, 0, 0]
            rows.append({'observation': {'time_s': 6 + index * .02,
                         'base_pose_w': measurement, 'joint_positions': {'test_joint': index * .1},
                         'object_poses_w': {'/World/Scene/Ring_02': measurement},
                         'tcp_poses_w': {'left': measurement, 'right': measurement}},
                         'command': {'stage_id': 'lift', 'base_pose_w': [99] * 7,
                                     'joint_positions': {'test_joint': 99}}})
        if duplicate_time:
            rows[1]['observation']['time_s'] = rows[0]['observation']['time_s']
        payload = b''.join((json.dumps(row) + '\n').encode() for row in rows)
        trace = self.root / 'trace.jsonl.gz'
        with gzip.open(trace, 'wb') as stream:
            stream.write(payload)
        return {'trace': trace, 'sources': {str(trace): hashlib.sha256(trace.read_bytes()).hexdigest()}}, rows, payload

    def test_gzip_retains_measured_states_not_command_targets(self):
        source, originals, payload = self.write_trace()
        states, digest = read_states(source)
        self.assertEqual(digest, hashlib.sha256(payload).hexdigest())
        for state, original in zip(states, originals):
            for key in ('base_pose_w', 'joint_positions', 'object_poses_w', 'tcp_poses_w'):
                self.assertEqual(state[key], original['observation'][key])
            self.assertNotEqual(state['joint_positions'], original['command']['joint_positions'])

    def test_nonmonotonic_time_is_rejected(self):
        source, _, _ = self.write_trace(duplicate_time=True)
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            read_states(source)

    def test_changed_trace_digest_is_rejected(self):
        source, _, _ = self.write_trace()
        source['sources'][str(source['trace'])] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Trace changed'):
            read_states(source)

    def test_30fps_uses_existing_50hz_measurements_without_interpolation(self):
        states = [{'time_s': 6 + index * .02} for index in range(11)]
        before = deepcopy(states)
        selected = schedule(states, 30)
        self.assertEqual(len(selected), 6)
        self.assertEqual([row['trace_index'] for row in selected], [0, 2, 3, 5, 7, 8])
        self.assertEqual(states, before)
        for row in selected:
            self.assertEqual(row['source_time_s'], states[row['trace_index']]['time_s'])
            self.assertLessEqual(abs(row['sample_skew_s']), .01)

    def test_invalid_intervals_and_poses_are_rejected(self):
        states = [{'time_s': 6.0}, {'time_s': 6.02}]
        for kwargs in ({'fps': 0}, {'fps': 30, 'start_seconds': -1},
                       {'fps': 30, 'duration_seconds': 0}, {'fps': 30, 'start_seconds': 100}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                schedule(states, **kwargs)
        for value in ([0] * 7, [0, 0, float('nan'), 1, 0, 0, 0], [0] * 6):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pose(value, 'invalid')

    def test_lerobot_only_reports_missing_geometry_state(self):
        (self.root / 'meta').mkdir()
        (self.root / 'meta/info.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'lacks robot base and object trajectories'):
            discover(self.root, BUNDLE)


class ReplayEncoderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='nepes-replay-video-')
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / 'preview.mp4'
        self.command = [sys.executable, '-B', str(SCRIPTS / 'replay_support/encode.py'),
                        '--output', str(self.output), '--width', '32', '--height', '24',
                        '--fps', '30', '--expected-frames', '4']
        self.pixels = np.zeros((4, 24, 32, 3), dtype=np.uint8)
        self.pixels[:, :, :, 0] = np.arange(4, dtype=np.uint8)[:, None, None] * 50

    def run_encoder(self, payload):
        return subprocess.run(self.command, input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)

    def test_encoding_full_decode_count_dimensions_and_pts(self):
        result = self.run_encoder(self.pixels.tobytes())
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        receipt = json.loads(self.output.with_suffix('.verification.json').read_text())
        self.assertTrue(receipt['verified'])
        self.assertTrue(receipt['full_decode'])
        self.assertEqual(receipt['frames'], 4)
        self.assertEqual(receipt['sha256'], hashlib.sha256(self.output.read_bytes()).hexdigest())
        with av.open(str(self.output)) as container:
            frames = list(container.decode(video=0))
        self.assertEqual(len(frames), 4)
        for index, frame in enumerate(frames):
            self.assertEqual((frame.width, frame.height), (32, 24))
            self.assertEqual(frame.pts * frame.time_base, Fraction(index, 30))

    def test_existing_video_is_never_overwritten(self):
        original = b'preserve original video'
        self.output.write_bytes(original)
        result = self.run_encoder(self.pixels.tobytes())
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.output.read_bytes(), original)
        self.assertFalse(self.output.with_suffix('.verification.json').exists())

    def test_incomplete_video_cannot_receive_verified_receipt(self):
        result = self.run_encoder(self.pixels[:1].tobytes() + b'truncated')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.with_suffix('.verification.json').exists())


class ReplayVisualOverlayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='nepes-replay-overlay-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.trial = self.root / 'relocated_trial'
        self.profile = self.trial / 'profile'
        self.profile.mkdir(parents=True)
        self.scene = self.root / 'packaged_layout.usda'
        self.scene.write_text('#usda 1.0\n')
        self.overlay = self.profile / 'scene_visual_overlay.usda'
        self.config_path = self.profile / 'config.json'
        self.visual_path = self.profile / 'visual.json'
        changes, tree = [], {}
        for attribute, before in PROPERTIES.items():
            value = float32(before * .7 if attribute.endswith('intensity') else .6)
            changes.append({'attribute': attribute, 'before': before, 'after': value})
            prim, name = attribute.split('.', 1)
            node = tree
            for part in prim.strip('/').split('/'):
                node = node.setdefault(part, {})
            node[name] = value
        def serialize(node, depth=0):
            result = ''
            for key, value in node.items():
                prefix = '    ' * depth
                if isinstance(value, dict):
                    result += prefix + 'over "' + key + '"\n' + prefix + '{\n'
                    result += serialize(value, depth + 1) + prefix + '}\n'
                else:
                    result += prefix + 'float ' + key + ' = ' + str(value) + '\n'
            return result
        # An inaccessible old server path must never be opened or composed.
        self.overlay.write_text('#usda 1.0\n(\n subLayers = [@/previous-server/missing/layout.usda@]\n)\n' + serialize(tree))
        self.config = {'single_arm_layout': {'scene_usd': '/previous-server/trial/profile/scene_visual_overlay.usda',
                                            'scene_sha256': digest(self.overlay)}, 'unchanged_control': {'gain': 2.}}
        self.config_path.write_text(json.dumps(self.config))
        self.visual = {'schema': 'nepes_native_render_overlay_v1', 'case': 'dim_matte', 'recipe': CASES['dim_matte'],
                       'source_scene': '/previous-server/missing/layout.usda', 'source_scene_sha256': digest(self.scene),
                       'overlay': self.config['single_arm_layout']['scene_usd'], 'overlay_sha256': digest(self.overlay),
                       'config_sha256': digest(self.config_path), 'changes': changes,
                       'composed_differences': list(PROPERTIES), 'stage_metadata_unchanged': True,
                       'all_physics_geometry_poses_and_cameras_unchanged': True,
                       'only_visual_allowlisted_properties_authored': True}
        self.save_visual()

    def save_visual(self):
        self.visual_path.write_text(json.dumps(self.visual))

    def load(self):
        return load_visual_overlay(self.trial, self.trial, self.config, self.scene)

    def test_relocated_profile_never_resolves_old_asset_paths(self):
        visual = self.load()
        self.assertEqual(visual['case'], 'dim_matte')
        self.assertFalse(visual['old_sublayer_paths_resolved'])
        self.assertEqual(len(visual['changes']), 4)
        self.assertTrue(all(Path(path).is_relative_to(self.trial) for path in visual['source_files_sha256']))

    def test_missing_visual_profile_is_rejected(self):
        self.visual_path.unlink()
        with self.assertRaisesRegex(ValueError, 'retained profile'):
            self.load()

    def test_overlay_byte_tamper_is_rejected(self):
        self.overlay.write_text(self.overlay.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'overlay checksum'):
            self.load()

    def test_visual_receipt_value_tamper_is_rejected(self):
        self.visual['changes'][0]['after'] += 1
        self.save_visual()
        with self.assertRaisesRegex(ValueError, 'Visual values differ'):
            self.load()

    def test_native_run_config_binding_is_required(self):
        config = deepcopy(self.config)
        config['unchanged_control']['gain'] = 99
        with self.assertRaisesRegex(ValueError, 'native run config'):
            load_visual_overlay(self.trial, self.trial, config, self.scene)

    def test_baseline_scene_digest_is_required(self):
        self.scene.write_text('changed baseline')
        with self.assertRaisesRegex(ValueError, 'base scene differs'):
            self.load()

    def test_unsupported_authored_property_rejected_even_with_matching_hashes(self):
        self.overlay.write_text(self.overlay.read_text().replace('float inputs:intensity = 245.0',
                                'float inputs:intensity = 245.0\nfloat physics:mass = 99'))
        self.config['single_arm_layout']['scene_sha256'] = digest(self.overlay)
        self.config_path.write_text(json.dumps(self.config))
        self.visual.update(overlay_sha256=digest(self.overlay), config_sha256=digest(self.config_path))
        self.save_visual()
        with self.assertRaisesRegex(ValueError, 'Unauthorized/duplicate visual override'):
            self.load()

    def test_profile_symlink_cannot_escape_supplied_attempt(self):
        outside = self.root / 'outside_config.json'
        self.config_path.rename(outside)
        self.config_path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'escaped the supplied attempt'):
            self.load()

    def test_only_four_scene_mapped_properties_are_authored_and_read_back(self):
        visual = self.load()
        class Attribute:
            def __init__(self, value): self.value, self.sets = value, []
            def GetTypeName(self): return 'float'
            def GetConnections(self): return []
            def GetTimeSamples(self): return []
            def Get(self): return self.value
            def Set(self, value): self.value = float32(value); self.sets.append(value); return True
        attrs = {'/World/Scene' + key[len('/World'):]: Attribute(value) for key, value in PROPERTIES.items()}
        class Stage:
            def GetAttributeAtPath(self, path): return attrs.get(path)
        receipt = apply_visual_overlay(Stage(), visual)
        self.assertTrue(receipt['readback_verified'])
        self.assertEqual(len(receipt['attributes']), 4)
        self.assertTrue(all(len(attr.sets) == 1 for attr in attrs.values()))


if __name__ == '__main__':
    unittest.main()
