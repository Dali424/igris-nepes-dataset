"""Pose-derived feedforward for an arm on a moving robot base.

Differentiate the target expressed in the measured base actor frame. This
includes target motion and base translation/rotation, without confusing the
base link origin with its center of mass. No COM velocity, simulator imports,
actuator writes, filtering, or collision/success decisions belong here.
"""
from __future__ import annotations

import math
import numpy as np

from .snapshot import pose_to_matrix
from .task_control import rotation_error


class RelativePoseFeedforward:
    """Estimate a target's relative Cartesian twist from successive poses.

    ``update`` returns ``[vx, vy, vz, wx, wy, wz]`` in the base coordinate
    convention, in m/s and rad/s. Translation is the finite difference of
    ``inv(base_pose_w) @ target_pose_w``; angular velocity uses the shortest
    rotation vector between those relative rotations. It is a sampled
    geometric derivative, not an instantaneous measured body velocity.

    The first valid update returns zero. Call ``reset`` at episode, phase,
    waypoint, or discontinuous reference changes to avoid interpreting a new
    goal as motion. The caller must preserve joint speed, tracking, position,
    contact, and retention limits when applying the returned twist. This
    helper deliberately does not clip a Cartesian vector before the caller's
    joint-space constraints. Invalid inputs do not advance its history.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._previous_target_in_base = None

    def update(self, base_pose_w, target_pose_w, dt) -> np.ndarray:
        if isinstance(dt, (bool, np.bool_)):
            raise ValueError("Feedforward dt must be positive and finite")
        try:
            dt = float(dt)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Feedforward dt must be positive and finite") from exc
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Feedforward dt must be positive and finite")
        base = pose_to_matrix(base_pose_w)
        target = pose_to_matrix(target_pose_w)
        # The rigid inverse avoids a general matrix solve and uses only the
        # actor poses. In particular, body_lin_vel_w may be a COM velocity.
        relative = np.eye(4)
        relative[:3, :3] = base[:3, :3].T @ target[:3, :3]
        with np.errstate(over="ignore", invalid="ignore"):
            relative[:3, 3] = base[:3, :3].T @ (target[:3, 3] - base[:3, 3])
        if not np.isfinite(relative).all():
            raise ValueError("Feedforward relative pose must be finite")
        previous = self._previous_target_in_base
        if previous is None:
            twist = np.zeros(6, dtype=float)
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                twist = np.r_[
                    (relative[:3, 3] - previous[:3, 3]) / dt,
                    rotation_error(relative[:3, :3], previous[:3, :3]) / dt,
                ]
        if not np.isfinite(twist).all():
            raise ValueError("Feedforward pose difference produced a non-finite twist")
        self._previous_target_in_base = relative
        return twist
