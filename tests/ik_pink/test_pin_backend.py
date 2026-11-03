"""Pinocchio FK/Jacobian parity with the native solver on the recorded canonical URDF.

Ported from the 2026-10-30 experiment (test_pin_backend.py); the baseline is the
runtime's own NativeUpperBodyServo. Timing output is indicative only.
"""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _support as S  # noqa: E402

import numpy as np  # noqa: E402
from pin_backend import PinocchioUpperBodyKinematics  # noqa: E402


class PinBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from robros_motion_planning.task_control import rotation_error
        from robros_motion_planning.upper_body_control import NativeUpperBodyServo
        cls.rotation_error = staticmethod(rotation_error)
        cls.model = S.model()
        cls.baseline = NativeUpperBodyServo(cls.model, waist_command_limits={
            'Joint_Waist_Yaw': [-1.47, 1.47], 'Joint_Waist_Roll': [-.2, .2], 'Joint_Waist_Pitch': [-.4, .15]})
        cls.backend = PinocchioUpperBodyKinematics(S.URDF)
        cls.names = tuple(cls.model.joint_limits)
        cls.low, cls.high = np.asarray([cls.model.joint_limits[n] for n in cls.names]).T

    def sample(self, rng):
        q = self.low + (.1 + .8 * rng.random(len(self.names))) * (self.high - self.low)
        quaternion = rng.normal(size=4)
        quaternion /= np.linalg.norm(quaternion)
        pelvis = np.r_[rng.uniform([-.5, -.5, .7], [.5, .5, 1.1]), quaternion]
        return dict(zip(self.names, q)), pelvis

    def test_random_both_arms_varying_pelvis_parity(self):
        rng = np.random.default_rng(20261030)
        pose_error = jacobian_error = 0.
        for _ in range(64):
            joints, pelvis = self.sample(rng)
            for arm in ('left', 'right'):
                actual, ja = self.backend.pose_and_jacobian(joints, pelvis, arm)
                expected, je = self.baseline.pose_and_jacobian(joints, pelvis, arm)
                pose_error = max(pose_error, float(np.max(np.abs(actual - expected))))
                jacobian_error = max(jacobian_error, float(np.max(np.abs(ja - je))))
                np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)
                np.testing.assert_allclose(ja, je, atol=3e-12, rtol=0)
        print(json.dumps({'case': '128_random_arm_poses', 'max_pose_element_error': pose_error,
                          'max_jacobian_element_error': jacobian_error}), flush=True)

    def test_central_finite_difference_including_inverse_pelvis(self):
        rng = np.random.default_rng(28371)
        maximum, epsilon = 0., 2e-6
        for _ in range(12):
            joints, pelvis = self.sample(rng)
            for arm in ('left', 'right'):
                _, analytic = self.backend.pose_and_jacobian(joints, pelvis, arm)
                numerical = np.zeros_like(analytic)
                for column, name in enumerate(self.backend.owned_names_by_arm[arm]):
                    plus, minus = dict(joints), dict(joints)
                    plus[name] += epsilon
                    minus[name] -= epsilon
                    # Independent native FK supplies the central-difference poses.
                    tp, _ = self.baseline.pose_and_jacobian(plus, pelvis, arm)
                    tm, _ = self.baseline.pose_and_jacobian(minus, pelvis, arm)
                    numerical[:3, column] = (tp[:3, 3] - tm[:3, 3]) / (2 * epsilon)
                    numerical[3:, column] = self.rotation_error(tp[:3, :3], tm[:3, :3]) / (2 * epsilon)
                maximum = max(maximum, float(np.max(np.abs(analytic - numerical))))
                np.testing.assert_allclose(analytic, numerical, atol=2e-8, rtol=0)
        print(json.dumps({'case': '240_central_difference_columns', 'max_element_error': maximum}), flush=True)

    def test_explicit_order_and_cached_arrays_are_independent(self):
        joints, pelvis = self.sample(np.random.default_rng(12))
        order = {arm: tuple(reversed(self.backend.owned_names_by_arm[arm])) for arm in ('left', 'right')}
        reordered = PinocchioUpperBodyKinematics(S.URDF, order)
        for arm in order:
            transform, jacobian = self.backend.pose_and_jacobian(joints, pelvis, arm)
            other, other_jacobian = reordered.pose_and_jacobian(joints, pelvis, arm)
            np.testing.assert_allclose(other, transform, atol=1e-12)
            np.testing.assert_allclose(other_jacobian, jacobian[:, ::-1], atol=1e-12)
            saved = transform.copy(), jacobian.copy()
            self.backend.pose_and_jacobian(*self.sample(np.random.default_rng(33)), arm)
            np.testing.assert_array_equal(transform, saved[0])
            np.testing.assert_array_equal(jacobian, saved[1])

    def test_reject_missing_nonfinite_unknown_and_bad_pose(self):
        joints, pelvis = self.sample(np.random.default_rng(42))
        name = self.names[0]
        bad = dict(joints)
        del bad[name]
        with self.assertRaises(ValueError):
            self.backend.pose_and_jacobian(bad, pelvis, 'left')
        for value in (float('nan'), float('inf'), True, '0', self.model.joint_limits[name][1] + .051):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.backend.pose_and_jacobian({**joints, name: value}, pelvis, 'left')
        with self.assertRaises(ValueError):
            self.backend.pose_and_jacobian({**joints, 'wrong_name': 0.}, pelvis, 'left')
        for pose in ([0.] * 7, [0.] * 6, [float('nan')] * 7):
            with self.assertRaises(ValueError):
                self.backend.pose_and_jacobian(joints, pose, 'left')
        with self.assertRaises(ValueError):
            self.backend.pose_and_jacobian(joints, pelvis, 'middle')
        with self.assertRaises(ValueError):
            PinocchioUpperBodyKinematics(S.URDF, {'left': self.names[:10]})

    def test_small_physical_overshoot_matches_native_clipping(self):
        joints, pelvis = self.sample(np.random.default_rng(42))
        name = self.backend.owned_names_by_arm['right'][-1]
        joints[name] = self.model.joint_limits[name][1] + .025
        for arm in ('left', 'right'):
            pose, jacobian = self.backend.pose_and_jacobian(joints, pelvis, arm)
            expected, je = self.baseline.pose_and_jacobian(joints, pelvis, arm)
            np.testing.assert_allclose(pose, expected, atol=2e-12)
            np.testing.assert_allclose(jacobian, je, atol=3e-12)

    def test_unsupported_urdf_joint_rejected(self):
        text = S.URDF.read_text()
        self.assertIn('type="revolute"', text)
        with tempfile.TemporaryDirectory(prefix='pin-unsupported-joint-') as directory:
            path = Path(directory) / 'continuous.urdf'
            path.write_text(text.replace('type="revolute"', 'type="continuous"', 1))
            with self.assertRaisesRegex(ValueError, 'Unsupported'):
                PinocchioUpperBodyKinematics(path)

    def test_indicative_cpu_timing(self):
        samples = [(*self.sample(np.random.default_rng(300 + i)), arm) for i in range(16) for arm in ('left', 'right')]
        result = {}
        for label, implementation in (('native_numpy', self.baseline), ('pinocchio', self.backend)):
            for sample in samples:
                implementation.pose_and_jacobian(*sample)
            timings = []
            for _ in range(5):
                start = time.perf_counter()
                for sample in samples:
                    implementation.pose_and_jacobian(*sample)
                timings.append((time.perf_counter() - start) / len(samples))
            result[label + '_median_seconds_per_pose_and_jacobian'] = statistics.median(timings)
        result['scope'] = 'CPU FK/Jacobian only; not IK, simulation or collection speedup'
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
