"""The runtime client drives the real stdio worker exactly like an Isaac process.

Ported from the 2026-10-30 experiment (test_bridge.py) and extended with the
recorded fixtures. The worker interpreter is this isolated IK Python.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _support as S  # noqa: E402

import numpy as np  # noqa: E402
import robros_motion_planning.upper_body_ik_backend as backend  # noqa: E402

WAIST = ("Joint_Waist_Yaw", "Joint_Waist_Roll", "Joint_Waist_Pitch")
KWARGS = {"waist_command_limits": {"Joint_Waist_Yaw": [-1.47, 1.47], "Joint_Waist_Roll": [-.15, .15],
                                   "Joint_Waist_Pitch": [-.4, .15]}, "arm_command_margin_rad": .03}


def runtime_module(profile):
    """The shipped upper_body_control.py of a profile, imported with its backend hook."""
    if profile == "left2":
        import robros_motion_planning.upper_body_control as module
        return module
    name = "robros_motion_planning._test_upper_body_control_" + profile
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, S.native_module(profile))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class WorkerStdioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = {key: os.environ.get(key) for key in (backend.PYTHON_ENV, backend.BACKEND_ENV)}
        os.environ[backend.PYTHON_ENV] = sys.executable
        os.environ.pop(backend.BACKEND_ENV, None)

    @classmethod
    def tearDownClass(cls):
        for key, value in cls.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def tearDown(self):
        backend.shutdown_worker()

    def pair(self, profile):
        module = runtime_module(profile)
        self.assertEqual(module.UpperBodyServo.__name__, "PinkUpperBodyServo")
        proxy = module.UpperBodyServo(S.model(), **KWARGS)
        direct = S.make(profile, "pink", **KWARGS)
        return module, proxy, direct

    def inputs(self, servo, arm):
        from robros_motion_planning.snapshot import matrix_to_pose
        joints = {name: (low + high) / 2 for name, (low, high) in S.model().joint_limits.items()}
        for name in servo.names(arm):
            joints[name] = sum(servo.limits[name]) / 2
        pelvis = [.1, -.2, .96, 1., 0., 0., 0.]
        target, _ = servo.pose_and_jacobian(joints, pelvis, arm)
        target[:3, 3] += [.002, -.001, .003]
        return [joints, pelvis, matrix_to_pose(target), arm, .02, dict(joints)]

    def test_proxy_matches_direct_pink_both_profiles_with_stage_and_hold(self):
        for profile, arm in S.PROFILES.items():
            module, proxy, direct = self.pair(profile)
            args = self.inputs(direct, arm)
            for servo in (proxy, direct):
                servo.stage_id = "transfer"
                servo.stage_command_limits_rad = {name: list(servo.limits[name]) for name in servo.names(arm)}
                servo.waist_speed_scope = "test_transfer_scope"
            options = {"waist_hold": {name: args[0][name] for name in WAIST},
                       "posture_target": {name: args[0][name] + (.01 if name not in WAIST else 0.)
                                          for name in direct.names(arm)}, "posture_gain": 2.,
                       "feedforward_twist_world": np.array([.001, 0., -.002, 0., .001, 0.])}
            actual, diagnostic = proxy.solve_step(*args, **options)
            expected, expected_diagnostic = direct.solve_step(*args, **options)
            np.testing.assert_allclose(list(actual.values()), list(expected.values()), atol=1e-12, rtol=0)
            marker = diagnostic.pop("ik_backend")
            self.assertEqual(marker["backend"], "pink")
            self.assertEqual(marker["worker_pid"], backend.shared_worker().process.pid)
            self.assertEqual(json.loads(json.dumps(expected_diagnostic)), diagnostic)
            pose, jacobian = proxy.pose_and_jacobian(args[0], args[1], arm)
            direct_pose, direct_jacobian = direct.pose_and_jacobian(args[0], args[1], arm)
            np.testing.assert_allclose(pose, direct_pose, atol=1e-12)
            np.testing.assert_allclose(jacobian, direct_jacobian, atol=1e-12)
            report = backend.backend_report()
            self.assertEqual((report["backend"], report["status"]), ("pink", "ready"))
            self.assertEqual(report["native_solver"]["native_prefix_sha256"],
                             S.provenance()["native_solver_prefix_sha256"][profile])
            backend.shutdown_worker()  # One worker serves one runtime profile.

    def test_all_stage_and_early_servos_share_one_worker(self):
        from robros_motion_planning.upper_body_control import build_upper_body_servos
        config = S.read_json(S.RELEASE / "configs" / "right4.json")
        module = runtime_module("right4")
        base, stages = module.build_upper_body_servos(S.model(), config["tasks"]["crs_load_jig"]["upper_body_ik"], "right")
        instances = [base, *stages.values()] + [stage.early_servo for stage in stages.values()
                                                if getattr(stage, "early_servo", None) is not None]
        self.assertTrue(all(type(servo).__name__ == "PinkUpperBodyServo" for servo in instances))
        self.assertEqual(len({id(servo._ik_worker) for servo in instances}), 1)
        self.assertEqual(len({servo._ik_remote_id for servo in instances}), len(instances))
        self.assertIsNot(build_upper_body_servos, module.build_upper_body_servos)

    def test_remote_error_closes_worker_and_never_falls_back(self):
        _, proxy, direct = self.pair("left2")
        args = self.inputs(direct, "left")
        args[4] = -1.
        worker = proxy._ik_worker
        with self.assertRaisesRegex(backend.IKBackendError, "ValueError: dt is outside permitted bounds"):
            proxy.solve_step(*args)
        self.assertEqual(worker.process.wait(timeout=10), 2)
        args[4] = .02
        with self.assertRaisesRegex(backend.IKBackendError, "never restarted"):
            proxy.solve_step(*args)
        with self.assertRaisesRegex(backend.IKBackendError, "never restarted"):
            backend.shared_worker()
        report = backend.backend_report()
        self.assertEqual(report["status"], "failed")
        self.assertIn("dt is outside permitted bounds", report["error"])

    def test_recorded_fixtures_round_trip_through_worker(self):
        worst = 0.
        for profile in S.PROFILES:
            module = runtime_module(profile)
            for sample in (s for s in S.snapshots() if s["profile"] == profile):
                proxy = S.configure_stage(module.UpperBodyServo(S.model(), **sample["constructor_kwargs"]), sample)
                command, diagnostics = S.solve(proxy, sample)
                self.assertEqual(diagnostics["ik_backend"]["backend"], "pink")
                worst = max(worst, max(abs(command[n] - sample["recorded_output"][n]) for n in command))
            backend.shutdown_worker()
        print(json.dumps({"case": "50_recorded_calls_via_stdio_worker", "max_command_difference_rad": worst}), flush=True)
        self.assertLessEqual(worst, 1e-9)

    def test_client_import_never_imports_pin_or_pink(self):
        program = ("import sys; sys.path.insert(0, %r); import robros_motion_planning.upper_body_control as m; "
                   "assert m.UpperBodyServo.__name__ == 'PinkUpperBodyServo'; "
                   "assert 'pinocchio' not in sys.modules and 'pink' not in sys.modules; "
                   "from robros_motion_planning import upper_body_ik_backend as b; "
                   "assert b._client is None and b.backend_report()['status'] == 'not_started'") % str(S.runtime_root("left2"))
        environment = dict(os.environ)
        environment.pop(backend.BACKEND_ENV, None)
        subprocess.run([sys.executable, "-I", "-B", "-c", program], check=True, capture_output=True, env=environment)

    def test_worker_check_cli(self):
        result = subprocess.run([sys.executable, "-I", "-B", str(S.IK_DIR / "worker.py"), "--check"],
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        document = json.loads(result.stdout)
        self.assertTrue(document["ok"])
        self.assertEqual(document["mismatches"], {})

    def test_stdio_latency_against_direct_pink(self):
        _, proxy, direct = self.pair("left2")
        args = self.inputs(direct, "left")
        for _ in range(5):
            direct.solve_step(*args)
            proxy.solve_step(*args)
        results = {}
        for name, servo in (("direct_pink", direct), ("stdio_pink", proxy)):
            times = []
            for _ in range(60):
                start = time.perf_counter()
                servo.solve_step(*args)
                times.append(time.perf_counter() - start)
            results[name + "_median_ms"] = 1e3 * statistics.median(times)
        results["stdio_extra_median_ms"] = results["stdio_pink_median_ms"] - results["direct_pink_median_ms"]
        results["scope"] = "Isolated CPU synthetic one-step solve; no physics speed claim"
        print(json.dumps(results), flush=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
