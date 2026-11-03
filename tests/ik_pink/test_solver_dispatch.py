"""The Pink backend really solves with pink.solve_ik (quadprog), never with SciPy's BVLS.

Both optimizers reproduce the native command to about 1e-16 rad, so the
equivalence tests alone cannot tell them apart; spies on both entry points can.
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _support as S  # noqa: E402


class SolverDispatchTests(unittest.TestCase):
    def test_pink_backend_calls_pink_solve_ik_and_never_scipy_lsq_linear(self):
        import pink
        import scipy.optimize
        calls = {"pink.solve_ik": 0, "scipy.optimize.lsq_linear": 0}
        solve_ik, lsq_linear = pink.solve_ik, scipy.optimize.lsq_linear

        def pink_spy(*args, **kwargs):
            calls["pink.solve_ik"] += 1
            return solve_ik(*args, **kwargs)

        def scipy_spy(*args, **kwargs):
            calls["scipy.optimize.lsq_linear"] += 1
            return lsq_linear(*args, **kwargs)

        # Installed before any servo exists: each servo executes its own copy of the native
        # solver source, whose `from scipy.optimize import lsq_linear` then binds the spy.
        pink.solve_ik, scipy.optimize.lsq_linear = pink_spy, scipy_spy
        try:
            samples, adapter_calls = S.snapshots(), 0
            for sample in samples:
                servo = S.configure_stage(S.make(sample["profile"], "pink", **sample["constructor_kwargs"]), sample)
                self.assertEqual(servo.ik_backend_identity["optimization"], "pink.solve_ik/quadprog")
                S.solve(servo, sample)
                adapter_calls += servo.ik_qp.calls
            self.assertEqual(calls["scipy.optimize.lsq_linear"], 0)
            self.assertEqual(calls["pink.solve_ik"], adapter_calls)
            self.assertGreaterEqual(adapter_calls, len(samples))
            # Control: the native baseline does go through SciPy, so the spy would have seen it.
            sample = samples[0]
            S.solve(S.configure_stage(S.make(sample["profile"], "baseline", **sample["constructor_kwargs"]), sample),
                    sample)
            self.assertGreater(calls["scipy.optimize.lsq_linear"], 0)
        finally:
            pink.solve_ik, scipy.optimize.lsq_linear = solve_ik, lsq_linear


if __name__ == "__main__":
    unittest.main(verbosity=2)
