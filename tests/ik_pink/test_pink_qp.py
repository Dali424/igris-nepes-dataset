"""pink.solve_ik/quadprog reproduces SciPy BVLS on the native bounded objective.

Ported from the 2026-10-30 experiment (test_pink_qp.py, pytest parametrization
replaced by subtests).
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _support  # noqa: E402,F401  Adds ik/pink to sys.path.

import numpy as np  # noqa: E402
from scipy.optimize import lsq_linear  # noqa: E402
from pink_qp import PinkBoundedLeastSquares  # noqa: E402


class PinkQpTests(unittest.TestCase):
    def test_same_bounded_optimum(self):
        for nv in (1, 4, 7, 10):
            for dt in (.01, .02, .04):
                with self.subTest(nv=nv, dt=dt):
                    rng = np.random.default_rng(735 + nv)
                    solver = PinkBoundedLeastSquares()
                    solver.dt = dt
                    for _ in range(15):
                        a = np.vstack([rng.normal(size=(6, nv)), .04 * np.eye(nv)])
                        b = rng.normal(size=len(a))
                        lo = rng.uniform(-.6, .1, size=nv)
                        hi = lo + rng.uniform(.001, 1., size=nv)
                        expected = lsq_linear(a, b, bounds=(lo, hi), method="bvls", tol=1e-10, max_iter=100)
                        actual = solver(a, b, bounds=(lo, hi), method="bvls", tol=1e-10, max_iter=100)
                        np.testing.assert_allclose(actual.x, expected.x, atol=2e-9, rtol=1e-8)
                        self.assertLess(actual.optimality, 1e-7)

    def test_rank_deficient_jacobian_keeps_existing_damping(self):
        solver = PinkBoundedLeastSquares()
        solver.dt = .02
        a = np.vstack([np.zeros((6, 10)), .04 * np.eye(10)])
        b = np.r_[np.ones(6), np.linspace(-.01, .01, 10)]
        out = solver(a, b, bounds=(-np.ones(10), np.ones(10)), method="bvls", tol=1e-10, max_iter=100)
        np.testing.assert_allclose(out.x, b[6:] / .04, atol=1e-10)

    def test_fail_closed_on_uninitialized_dt(self):
        with self.assertRaisesRegex(ValueError, "control dt"):
            PinkBoundedLeastSquares()(np.eye(2), np.ones(2), bounds=(-np.ones(2), np.ones(2)),
                                      method="bvls", tol=1e-10, max_iter=100)

    def test_fail_closed_on_held_joint_or_nonfinite_input(self):
        solver = PinkBoundedLeastSquares()
        solver.dt = .02
        with self.assertRaisesRegex(ValueError, "held joints"):
            solver(np.eye(2), np.ones(2), bounds=(np.zeros(2), np.array([1., 0.])), method="bvls",
                   tol=1e-10, max_iter=100)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            solver(np.eye(2), np.array([1., np.nan]), bounds=(-np.ones(2), np.ones(2)), method="bvls",
                   tol=1e-10, max_iter=100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
