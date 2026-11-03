"""Bounded CPU joint-goal candidates; never a collision or motion certificate."""

from __future__ import annotations

import math
import time

import numpy as np

from .robot_model import planning_joint_limits
from .snapshot import matrix_to_pose, pose_to_matrix
from .task_control import rotation_error


def solve_goal_candidate(model, joint_positions, arm, goal_pose_base, *, seeds=16,
                         max_nfev=500, seed=0, position_tolerance_m=0.001,
                         rotation_tolerance_rad=0.01, position_only=False, hold_wrist=False):
    """Find one candidate with exact locked joints and shared planning bounds.

    Deterministic multistart least squares is bounded by seeds/evaluations. A
    failure is not proof of infeasibility. Caller must reject failed candidates,
    check full self/world collision, plan a path, and validate its endpoint.
    """
    from scipy import __version__ as scipy_version
    from scipy.optimize import least_squares

    if not isinstance(seeds, int) or not 1 <= seeds <= 64 or not isinstance(max_nfev, int) or max_nfev < 1:
        raise ValueError("Use 1..64 seeds and positive max_nfev")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(position_only, bool):
        raise ValueError("position_only must be boolean")
    if not isinstance(hold_wrist, bool) or (hold_wrist and not position_only):
        raise ValueError("hold_wrist requires position_only transit IK")
    if not all(math.isfinite(v) and v > 0 for v in (position_tolerance_m, rotation_tolerance_rad)):
        raise ValueError("Goal tolerances must be positive and finite")
    measured = model.validate_snapshot(joint_positions)
    names = model.arm_joint_names(arm)
    limits = planning_joint_limits(model, arm)
    low, high = np.asarray([limits[name] for name in names]).T
    initial = np.array([measured[name] for name in names])
    if np.any(initial < low) or np.any(initial > high):
        raise ValueError("Measured active joints exceed planning limits; no implicit clamping")
    target = pose_to_matrix(goal_pose_base)
    child_to_joint = {joint.child: joint for joint in model.joints.values()}
    chain, child = [], model.default_ee_link(arm)
    while child != model.root_link:
        joint = child_to_joint[child]
        chain.append(joint)
        child = joint.parent
    chain.reverse()
    indices = {name: i for i, name in enumerate(names)}
    decisions = np.arange(4 if hold_wrist else len(names))

    def expand(q):
        full = initial.copy()
        full[decisions] = q
        return full

    def fk(q):
        transform = np.eye(4)
        for joint in chain:
            transform = transform @ joint.origin
            if joint.kind == "fixed":
                continue
            angle = q[indices[joint.name]] if joint.name in indices else measured[joint.name]
            motion = np.eye(4)
            if joint.kind == "prismatic":
                motion[:3, 3] = joint.axis * angle
            else:
                x, y, z = joint.axis
                skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                motion[:3, :3] += math.sin(angle) * skew + (1 - math.cos(angle)) * skew @ skew
            transform = transform @ motion
        return transform

    def error(q):
        actual = fk(expand(q))
        return np.r_[target[:3, 3] - actual[:3, 3], rotation_error(target[:3, :3], actual[:3, :3])]

    rng = np.random.default_rng(seed)
    starts = [initial[decisions]]
    if seeds > 1:
        starts.append(((low + high) / 2)[decisions])
    starts.extend(rng.uniform(low[decisions], high[decisions]) for _ in range(max(0, seeds - 2)))
    scales = np.r_[np.full(3, position_tolerance_m), np.full(3, rotation_tolerance_rad)]
    attempts, best = [], None
    started = time.monotonic()
    for index, start in enumerate(starts):
        # Transit position IK is underdetermined. A tiny posture residual avoids
        # arbitrary wrist motion along zero-position-Jacobian directions; it
        # cannot replace the position tolerance or subsequent collision checks.
        result = least_squares(lambda q: np.r_[(error(q) / scales)[:3], 1e-3 * (q - initial[decisions])]
                               if position_only else error(q) / scales,
                               start, jac="2-point", bounds=(low[decisions], high[decisions]),
                               max_nfev=max_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-10,
                               x_scale=1.0 if position_only else "jac")
        residual = error(result.x)
        position, rotation = float(np.linalg.norm(residual[:3])), float(np.linalg.norm(residual[3:]))
        attempt = {"seed_index": index, "position_error_m": position, "rotation_error_rad": rotation,
                   "nfev": int(result.nfev), "optimizer_success": bool(result.success)}
        attempts.append(attempt)
        score = position / position_tolerance_m if position_only else max(position / position_tolerance_m, rotation / rotation_tolerance_rad)
        if best is None or score < best[0]:
            best = score, expand(result.x), attempt
        if position <= min(position_tolerance_m, 1e-5) and (position_only or rotation <= min(rotation_tolerance_rad, 1e-4)):
            break
    _, q, attempt = best
    active_positions = {name: float(value) for name, value in zip(names, q)}
    return {
        "success": bool(attempt["position_error_m"] <= position_tolerance_m
                        and (position_only or attempt["rotation_error_rad"] <= rotation_tolerance_rad)),
        "method": "bounded_multistart_cpu_least_squares",
        "orientation_constraint": "free_transit_candidate_actual_orientation_returned" if position_only else "full_tcp_pose",
        "held_active_joints": list(names[4:]) if hold_wrist else [],
        "numpy_version": np.__version__, "scipy_version": scipy_version,
        "joint_positions": {**measured, **active_positions},
        "active_joint_positions": active_positions,
        "goal_pose_base": list(goal_pose_base), "candidate_pose_base": matrix_to_pose(fk(q)),
        "position_error_m": attempt["position_error_m"], "rotation_error_rad": attempt["rotation_error_rad"],
        "position_tolerance_m": position_tolerance_m, "rotation_tolerance_rad": rotation_tolerance_rad,
        "seeds_requested": seeds, "max_nfev_per_seed": max_nfev, "seed": seed, "attempts": attempts,
        "wall_time_s": time.monotonic() - started,
        "locked_joints_changed": [], "collision_checked": False, "execution_validated": False,
    }
