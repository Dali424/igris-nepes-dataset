"""Equivalence on 50 recorded real solver calls (tests/ik_pink/fixtures, see provenance.json).

Each sample holds the measured joints/pelvis, previous commands, world target,
posture/feedforward/held-waist inputs, stage limits and the command the native
solver actually emitted during an accepted seed-10000 episode. Every backend
must reproduce the recorded command, effective bounds and model TCP. The
experiment measured 2.2e-16 rad (Pink) here; 1e-9 rad absorbs CPU/BLAS
rounding differences on other machines and is far below actuator resolution.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _support as S  # noqa: E402

import numpy as np  # noqa: E402

COMMAND_TOLERANCE_RAD = 1e-9
BOUND_TOLERANCE_RAD = 1e-12


class FixtureEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = S.snapshots()
        cls.provenance = S.provenance()

    def test_fixture_files_match_provenance(self):
        for name, entry in self.provenance["files"].items():
            self.assertEqual(S.sha256(S.FIXTURES / name), entry["sha256"], name)
        self.assertEqual(len(self.samples), 50)
        counts = {}
        for sample in self.samples:
            counts[(sample["profile"], sample["stage"])] = counts.get((sample["profile"], sample["stage"]), 0) + 1
        self.assertEqual(set(counts.values()), {5})
        self.assertEqual({profile for profile, _ in counts}, set(S.PROFILES))

    def test_runtime_native_solver_is_the_recorded_production_solver(self):
        from servo_factory import split_native_source
        for profile, digest in self.provenance["native_solver_prefix_sha256"].items():
            prefix = split_native_source(S.native_module(profile).read_text())
            self.assertEqual(hashlib.sha256(prefix.encode()).hexdigest(), digest, profile)

    def test_every_backend_reproduces_recorded_native_commands(self):
        from robros_motion_planning.snapshot import pose_to_matrix
        worst = {backend: {"recorded_command": 0., "native_command": 0., "bound_violation": 0.,
                           "effective_bounds": 0., "weighted_residual": 0., "tcp_position_m": 0.}
                 for backend in S.BACKENDS}
        for index, sample in enumerate(self.samples):
            names = sample["recorded_diagnostics"]["owned_joint_names"]
            recorded_bounds = sample["recorded_diagnostics"]["effective_command_bounds_rad"]
            results = {}
            for backend in S.BACKENDS:
                servo = S.configure_stage(S.make(sample["profile"], backend, **sample["constructor_kwargs"]), sample)
                command, diagnostics = S.solve(servo, sample)
                results[backend] = command
                self.assertEqual(list(command), names)
                row = worst[backend]
                row["recorded_command"] = max(row["recorded_command"],
                                              max(abs(command[n] - sample["recorded_output"][n]) for n in names))
                row["bound_violation"] = max(row["bound_violation"], max(
                    max(recorded_bounds[n][0] - command[n], command[n] - recorded_bounds[n][1], 0.) for n in names))
                row["effective_bounds"] = max(row["effective_bounds"], max(
                    abs(a - b) for n in names
                    for a, b in zip(diagnostics["effective_command_bounds_rad"][n], recorded_bounds[n])))
                row["weighted_residual"] = max(row["weighted_residual"], abs(
                    diagnostics["weighted_velocity_residual"]
                    - sample["recorded_diagnostics"]["weighted_velocity_residual"]))
                tcp = pose_to_matrix(diagnostics["model_tcp_pose_w"])
                recorded_tcp = pose_to_matrix(sample["recorded_diagnostics"]["model_tcp_pose_w"])
                row["tcp_position_m"] = max(row["tcp_position_m"], float(np.linalg.norm(tcp[:3, 3] - recorded_tcp[:3, 3])))
            for backend in S.BACKENDS:
                worst[backend]["native_command"] = max(worst[backend]["native_command"], max(
                    abs(results[backend][n] - results["baseline"][n]) for n in names))
            with self.subTest(sample=index, profile=sample["profile"], stage=sample["stage"]):
                for backend in S.BACKENDS:
                    for n in names:
                        self.assertLessEqual(abs(results[backend][n] - sample["recorded_output"][n]),
                                             COMMAND_TOLERANCE_RAD, (backend, n))
        print(json.dumps({"case": "50_recorded_native_calls", "backends": worst}), flush=True)
        for backend, row in worst.items():
            self.assertLessEqual(row["recorded_command"], COMMAND_TOLERANCE_RAD, backend)
            self.assertLessEqual(row["native_command"], COMMAND_TOLERANCE_RAD, backend)
            self.assertLessEqual(row["bound_violation"], BOUND_TOLERANCE_RAD, backend)
            self.assertLessEqual(row["effective_bounds"], BOUND_TOLERANCE_RAD, backend)
            self.assertLessEqual(row["weighted_residual"], 1e-9, backend)
            self.assertLessEqual(row["tcp_position_m"], 1e-9, backend)


if __name__ == "__main__":
    unittest.main(verbosity=2)
