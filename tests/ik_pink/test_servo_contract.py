"""Contact-stage bounds and command-origin semantics are identical for every backend.

Ported from the 2026-10-30 experiment (test_servo_contract.py): baseline (native),
Pinocchio+BVLS and Pinocchio+Pink run the same native solver source taken from
each shipped runtime. No physics; ideal joint tracking only where stated.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _support as S  # noqa: E402

import numpy as np  # noqa: E402

WAIST = ("Joint_Waist_Yaw", "Joint_Waist_Roll", "Joint_Waist_Pitch")
WAIST_LIMITS = dict(zip(WAIST, [(-1.47, 1.47), (-.2, .2), (-.4, .15)]))


def make_inputs(profile, arm, **kwargs):
    from robros_motion_planning.snapshot import matrix_to_pose
    servos = [S.make(profile, backend, waist_command_limits=WAIST_LIMITS, **kwargs) for backend in S.BACKENDS]
    joints = {n: float(np.clip(0, lo, hi)) for n, (lo, hi) in S.model().joint_limits.items()}
    for n in servos[0].names(arm):
        lo, hi = servos[0].limits[n]
        joints[n] = (lo + hi) / 2
    pelvis = [.1, -.2, .85, 1., 0., 0., 0.]
    target, _ = servos[0].pose_and_jacobian(joints, pelvis, arm)
    target[:3, 3] += [.01, -.008, .012]
    return servos, joints, pelvis, matrix_to_pose(target)


class ServoContractTests(unittest.TestCase):
    def test_runtime_dependencies_shared_by_both_profiles_are_identical(self):
        # The native solver imports only these modules; the suite and one-profile workers rely on it.
        for name in ("__init__", "constants", "robot_model", "snapshot", "task_control"):
            digests = {hashlib.sha256((S.runtime_root(p) / "robros_motion_planning" / (name + ".py")).read_bytes()).hexdigest()
                       for p in S.PROFILES}
            self.assertEqual(len(digests), 1, name)

    def test_identical_commands_with_limits_hold_posture_and_feedforward(self):
        for profile in S.PROFILES:
            for arm in ("left", "right"):
                for saturation in (False, True):
                    for hold_waist in (False, True):
                        with self.subTest(profile=profile, arm=arm, saturation=saturation, hold=hold_waist):
                            self._identical(profile, arm, saturation, hold_waist)

    def _identical(self, profile, arm, saturation, hold_waist):
        servos, q, pelvis, target = make_inputs(profile, arm, posture_saturation_aware=saturation)
        names = servos[0].names(arm)
        previous = {n: q[n] + .001 for n in names}
        previous[names[-1]] += .20  # Recenter to the measured-q tracking envelope first.
        posture = {n: q[n] + .08 for n in names}
        hold = {n: q[n] for n in WAIST} if hold_waist else None
        outputs = []
        for servo in servos:
            result, diag = servo.solve_step(q, pelvis, target, arm, .02, previous,
                                            speed_scale=.8, posture_target=posture, posture_gain=.5,
                                            feedforward_twist_world=[.003, -.004, .001, .01, 0., -.005],
                                            waist_hold=hold)
            outputs.append(np.array([result[n] for n in names]))
            self.assertGreater(diag["protective_target_recentering_rad"], .04)
            self.assertIs(diag["root_pose_written"], False)
            self.assertIs(diag["leg_commands_emitted"], False)
            for n in names:
                lo, hi = diag["effective_command_bounds_rad"][n]
                self.assertTrue(lo - 1e-12 <= result[n] <= hi + 1e-12)
            if hold_waist:
                self.assertTrue(all(abs(result[n] - hold[n]) < 1e-12 for n in WAIST))
        for output in outputs[1:]:
            np.testing.assert_allclose(output, outputs[0], atol=2e-9, rtol=0)

    def test_closed_kinematic_loop_converges_without_writing_root(self):
        for arm in ("left", "right"):
            with self.subTest(arm=arm):
                servos, initial, pelvis, target = make_inputs("left2", arm)
                final = []
                for servo in servos:
                    q = dict(initial)
                    previous = {n: q[n] for n in servo.names(arm)}
                    first_error = error = None
                    for _ in range(140):
                        command, diag = servo.solve_step(q, pelvis, target, arm, .02, previous)
                        error = diag["position_error_m"]
                        first_error = error if first_error is None else first_error
                        q.update(command)  # Ideal actuator response only: not a physics success test.
                        previous = command
                    self.assertLess(error, first_error * .02)
                    final.append(np.array([q[n] for n in servo.names(arm)]))
                for output in final[1:]:
                    np.testing.assert_allclose(output, final[0], atol=2e-8, rtol=0)

    def test_factory_does_not_patch_runtime_module_or_scipy(self):
        from scipy.optimize import lsq_linear
        import robros_motion_planning.upper_body_control as runtime
        make_inputs("left2", "left")
        self.assertIs(runtime.lsq_linear, lsq_linear)
        self.assertIs(runtime.NativeUpperBodyServo.solve_step.__globals__["lsq_linear"], lsq_linear)

    def test_keyword_api_matches_native(self):
        servos, joints, pelvis, target = make_inputs("left2", "left")
        output = []
        for servo in servos:
            servo.pose_and_jacobian(joint_positions=joints, pelvis_pose_w=pelvis, arm="left")
            commands, _ = servo.solve_step(joint_positions=joints, pelvis_pose_w=pelvis,
                                           target_pose_w=target, arm="left", dt=.02)
            output.append(list(commands.values()))
        np.testing.assert_allclose(output[1:], [output[0], output[0]], atol=2e-9, rtol=0)

    def test_infeasible_hold_rejected_by_every_backend(self):
        for arm in ("left", "right"):
            servos, joints, pelvis, target = make_inputs("left2", arm)
            hold = {name: joints[name] for name in WAIST}
            hold[WAIST[0]] += .3  # Inside command limits but outside the measured tracking envelope.
            for servo in servos:
                with self.subTest(arm=arm, backend=servo.ik_backend_identity["backend"]), self.assertRaises(ValueError):
                    servo.solve_step(joints, pelvis, target, arm, .02, waist_hold=hold)

    def test_previous_target_outside_stage_limits_rejected(self):
        for arm in ("left", "right"):
            servos, joints, pelvis, target = make_inputs("left2", arm)
            name = servos[0].names(arm)[3]
            for servo in servos:
                servo.stage_id = "transfer"
                servo.stage_command_limits_rad = {name: [joints[name] - .02, joints[name] + .02]}
                servo.limits[name] = tuple(servo.stage_command_limits_rad[name])
                previous = {n: joints[n] for n in servo.names(arm)}
                previous[name] += .03
                with self.subTest(arm=arm, backend=servo.ik_backend_identity["backend"]), self.assertRaises(ValueError):
                    servo.solve_step(joints, pelvis, target, arm, .02, previous)

    def test_native_source_hook_is_required(self):
        from servo_factory import split_native_source, HOOK_MARKER
        with self.assertRaisesRegex(ValueError, "exactly one"):
            split_native_source("x = 1\n")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            split_native_source(("x = 1\n\n\n" + HOOK_MARKER + "\n") * 2)
        self.assertEqual(split_native_source("x = 1\n\n\n" + HOOK_MARKER + "\nhook()\n"), "x = 1\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
