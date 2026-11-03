"""Pink QP adapter for the existing bounded differential-IK objective.

The robot FK/Jacobian is supplied by Pinocchio in pin_backend.py. This module
represents the *selected free joint increments* in a small Euclidean tangent
space. It does not substitute a different robot or integrate measured q.
The caller eliminates held joints and retains all production command bounds.
"""
from types import SimpleNamespace

import numpy as np
import pinocchio as pin
import pink
from pink.limits import Limit
from pink.tasks import Task


class LinearizedCommandTask(Task):
    """Minimize ||A * delta_command - rhs * dt||, including posture damping."""

    def __init__(self, matrix, rhs, dt):
        super().__init__(cost=1.0, gain=1.0, lm_damping=0.0)
        self.matrix = matrix
        self.error = -rhs * dt

    def compute_error(self, configuration):
        return self.error

    def compute_jacobian(self, configuration):
        return self.matrix

    def __repr__(self):
        return f"LinearizedCommandTask(shape={self.matrix.shape})"


class CommandBounds(Limit):
    def __init__(self, lower, upper):
        eye = np.eye(len(lower))
        self.matrix = np.vstack((eye, -eye))
        self.velocity_limits = np.r_[upper, -lower]

    def compute_qp_inequalities(self, configuration, dt):
        return self.matrix, self.velocity_limits * dt


class PinkBoundedLeastSquares:
    """Per-servo replacement for scipy.lsq_linear; no module-wide monkey patch.

Pink receives real control dt. Its output velocity is integrated by the
unchanged controller at the clipped previous command, not measured q.
"""

    def __init__(self):
        self.dt = None
        self.configurations = {}
        self.calls = 0

    def _configuration(self, nv):
        if nv not in self.configurations:
            # This is a coordinate carrier for free command increments, not FK.
            model = pin.Model()
            for i in range(nv):
                model.addJoint(0, pin.JointModelPZ(), pin.SE3.Identity(), f"increment_{i}")
            model.lowerPositionLimit[:] = -1e6
            model.upperPositionLimit[:] = 1e6
            model.velocityLimit[:] = 1e6
            self.configurations[nv] = pink.Configuration(model, model.createData(), np.zeros(nv))
        return self.configurations[nv]

    def __call__(self, a, b, *, bounds, method, tol, max_iter):
        if method != "bvls" or self.dt is None or not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Pink adapter requires validated control dt and bounded-LSQ contract")
        a, b = np.asarray(a, float), np.asarray(b, float)
        lo, hi = (np.asarray(x, float) for x in bounds)
        if a.ndim != 2 or b.shape != (a.shape[0],) or lo.shape != (a.shape[1],) or hi.shape != lo.shape:
            raise ValueError("Invalid linearized IK shapes")
        if not all(np.isfinite(x).all() for x in (a, b, lo, hi)) or np.any(lo >= hi):
            raise ValueError("Nonfinite inputs or held joints not eliminated")
        configuration = self._configuration(a.shape[1])
        task = LinearizedCommandTask(a, b, self.dt)
        velocity = pink.solve_ik(configuration, [task], self.dt, solver="quadprog",
                                 damping=0.0, limits=[CommandBounds(lo, hi)])
        if not np.isfinite(velocity).all() or np.any(velocity < lo - 1e-8) or np.any(velocity > hi + 1e-8):
            raise RuntimeError("Pink returned a nonfinite or out-of-bounds command")
        velocity = np.clip(velocity, lo, hi)
        gradient = a.T @ (a @ velocity - b)
        projected = velocity - np.clip(velocity - gradient, lo, hi)
        optimality = float(np.linalg.norm(projected, ord=np.inf))
        if optimality > 1e-7:
            raise RuntimeError(f"Pink command KKT residual too large: {optimality}")
        self.calls += 1
        return SimpleNamespace(x=velocity, success=True, optimality=optimality)
