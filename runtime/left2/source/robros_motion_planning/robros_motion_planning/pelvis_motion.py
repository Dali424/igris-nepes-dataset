"""Pose-derived compensation for motion of the measured floating pelvis."""
from numbers import Real

import numpy as np

from .snapshot import pose_to_matrix
from .task_control import rotation_error


class PelvisMotionCompensation:
    """Estimate negative pelvis-induced TCP twist in world coordinates.

    The current TCP point is expressed through both consecutive pelvis poses.
    This retains the rotational lever arm used by the world-frame Jacobian.
    Target motion is deliberately absent. Reset on episode or controller-stage
    entry, not ordinary waypoint changes. Joint limits remain the servo's job;
    this sampled derivative is neither a contact nor a balance controller.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._previous_pelvis = None

    def update(self, pelvis_pose_w, actual_tcp_pose_w, dt):
        if isinstance(dt, (bool, np.bool_)) or not isinstance(dt, Real) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("Pelvis compensation dt must be positive and finite")
        pelvis = pose_to_matrix(pelvis_pose_w)
        tcp = pose_to_matrix(actual_tcp_pose_w)
        if self._previous_pelvis is None:
            twist = np.zeros(6)
        else:
            previous_current_point = self._previous_pelvis @ np.linalg.inv(pelvis) @ tcp
            twist = -np.r_[
                (tcp[:3, 3] - previous_current_point[:3, 3]) / dt,
                rotation_error(pelvis[:3, :3], self._previous_pelvis[:3, :3]) / dt,
            ]
        if not np.isfinite(twist).all():
            raise ValueError("Pelvis compensation derivative must be finite")
        self._previous_pelvis = pelvis.copy()
        return twist
