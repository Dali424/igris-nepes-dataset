"""Measured-motion stagnation detection; never changes control or success gates.

Errors must refer to the fixed final FSM pose for the current stage/waypoint,
not the moving controller reference. Virtual reference progress is diagnostic
only. Initial settling, gripper acquisition and release remain time-driven FSM
stages and are intentionally excluded here.
"""
from __future__ import annotations

from collections import Counter, deque
import math
from typing import Mapping

import numpy as np

from .snapshot import pose_to_matrix


MOTION_STAGES = frozenset(("navigate_source", "pregrasp", "approach", "lift", "recover_carry", "recover_base",
                           "navigate_target", "transfer", "place", "retreat", "open_hinge"))


def _number(value, name, *, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def pose_goal_errors(measured_pose_w, goal_pose_w, *, metric="tcp"):
    """Return position/rotation goal errors; base navigation uses only XY/yaw."""
    if metric not in ("tcp", "object", "base"):
        raise ValueError("metric must be tcp, object, or base")
    actual, goal = pose_to_matrix(measured_pose_w), pose_to_matrix(goal_pose_w)
    if metric == "base":
        position = np.linalg.norm(actual[:2, 3] - goal[:2, 3])
        if min(np.hypot(actual[0, 0], actual[1, 0]), np.hypot(goal[0, 0], goal[1, 0])) < 1e-8:
            raise ValueError("Base yaw is undefined for a vertical forward axis")
        angle = math.atan2(actual[1, 0], actual[0, 0]) - math.atan2(goal[1, 0], goal[0, 0])
        rotation = abs(math.atan2(math.sin(angle), math.cos(angle)))
    else:
        position = np.linalg.norm(actual[:3, 3] - goal[:3, 3])
        rotation = np.arccos(np.clip((np.trace(actual[:3, :3].T @ goal[:3, :3]) - 1) / 2, -1., 1.))
    return {"position_error_m": float(position), "rotation_error_rad": float(rotation)}


def posture_goal_errors(measured_joints, goal_joints):
    """Measured maximum joint error while navigation waits for body posture.

    There is no translation component. Reference or commanded joint motion is
    deliberately excluded from progress evidence.
    """
    if not goal_joints or not set(goal_joints) <= set(measured_joints):
        raise ValueError("Posture progress needs all measured goal joints")
    errors = [abs(float(measured_joints[n]) - float(q)) for n, q in goal_joints.items()]
    if not all(math.isfinite(v) for v in errors):
        raise ValueError("Posture progress joints must be finite")
    return {"position_error_m": 0., "rotation_error_rad": max(errors)}


class MotionProgressWatchdog:
    """Fail after sustained lack of measured progress in a motion stage.

    A new best position error smaller by 2 mm OR rotation error smaller by .01
    rad refreshes the default ten-second timer. Oscillating away and back to an
    already observed best pose does not refresh it. Stage/waypoint/metric changes
    reset the baseline. Repeated sampling at the same physics time adds no time
    or evidence. A missing observation interval is marked invalid, not success.

    ``goal_reached`` may bypass stagnation only when the caller has measured the
    appropriate pose tolerances. This is a pose-only diagnostic, never a task
    completion declaration. Normal FSM contact, dwell, release, and payload
    checks must still run. Leave it false when other commanded posture motion
    must finish before the stage can advance.
    """

    def __init__(self, *, duration_s=10., position_improvement_m=.002,
                 rotation_improvement_rad=.01, max_observation_gap_s=.25):
        self.duration_s = _number(duration_s, "duration_s", positive=True)
        self.position_improvement_m = _number(position_improvement_m, "position_improvement_m", positive=True)
        self.rotation_improvement_rad = _number(rotation_improvement_rad, "rotation_improvement_rad", positive=True)
        self.max_observation_gap_s = _number(max_observation_gap_s, "max_observation_gap_s", positive=True)
        self.reset()

    def reset(self):
        self._key = None
        self._last_time = None
        self._report = None
        self._history = deque(maxlen=10000)
        self._stalled = False

    def update(self, time_s, stage_id, *, position_error_m=None, rotation_error_rad=None,
               metric="tcp", waypoint_index=0, diagnostics: Mapping | None = None,
               goal_reached=False, enabled=True):
        now = _number(time_s, "time_s")
        if metric not in ("tcp", "object", "base", "posture"):
            raise ValueError("metric must be tcp, object, base, or posture")
        if type(waypoint_index) is not int or waypoint_index < 0:
            raise ValueError("waypoint_index must be a nonnegative integer")
        if type(goal_reached) is not bool or type(enabled) is not bool:
            raise ValueError("goal_reached and enabled must be explicit booleans")
        if self._last_time is not None and now < self._last_time - 1e-9:
            raise ValueError("Watchdog physics time moved backwards; reset at episode boundaries")
        if not enabled or stage_id not in MOTION_STAGES:
            self.reset()
            self._last_time = now
            self._report = {"active": False, "valid": True, "stalled": False, "reason": None,
                            "stage_id": stage_id, "scope": "motion_stagnation_only",
                            "physical_success_verified": False}
            return dict(self._report)
        position = _number(position_error_m, "position_error_m")
        rotation = _number(rotation_error_rad, "rotation_error_rad")
        key = (stage_id, waypoint_index, metric)
        if key == self._key and self._last_time is not None and abs(now - self._last_time) <= 1e-9:
            return {**self._report, "duplicate_physics_time": True}
        gap = self._last_time is not None and now - self._last_time > self.max_observation_gap_s + 1e-9
        if key != self._key or gap:
            already_stalled = self._stalled and key == self._key
            self._key = key
            self._start = self._last_progress = now
            self._best_position = self._position_anchor = position
            self._best_rotation = self._rotation_anchor = rotation
            self._history.clear()
            self._stalled = already_stalled
        self._last_time = now
        self._best_position = min(self._best_position, position)
        self._best_rotation = min(self._best_rotation, rotation)
        improved_position = self._position_anchor - self._best_position >= self.position_improvement_m - 1e-12
        improved_rotation = self._rotation_anchor - self._best_rotation >= self.rotation_improvement_rad - 1e-12
        if improved_position:
            self._position_anchor = self._best_position
        if improved_rotation:
            self._rotation_anchor = self._best_rotation
        if improved_position or improved_rotation or goal_reached:
            self._last_progress = now
        details = dict(diagnostics or {})
        diagnostic_values = {}
        for name in ("servo_position_error_m", "servo_rotation_error_rad", "reference_progress"):
            value = details.get(name)
            if value is not None:
                diagnostic_values[name] = _number(value, name)
        pause_reason = details.get("pause_reason")
        if pause_reason is not None and not isinstance(pause_reason, str):
            raise ValueError("pause_reason must be a string or None")
        self._history.append({"time_s": now, "position_error_m": position,
                              "rotation_error_rad": rotation, "pause_reason": pause_reason,
                              **diagnostic_values})
        while self._history and self._history[0]["time_s"] < now - self.duration_s - 1e-9:
            self._history.popleft()
        no_progress = now - self._last_progress
        if not gap and not goal_reached and no_progress >= self.duration_s - 1e-9:
            self._stalled = True
        ranges = {}
        for name in ("position_error_m", "rotation_error_rad", *diagnostic_values):
            values = [r[name] for r in self._history if name in r]
            if values:
                ranges[name] = {"minimum": min(values), "maximum": max(values), "last": values[-1]}
        self._report = {
            "active": True, "valid": not gap, "stalled": self._stalled,
            "reason": f"progress_stalled:{stage_id}" if self._stalled else ("observation_gap" if gap else None),
            "stage_id": stage_id, "waypoint_index": waypoint_index, "metric": metric,
            "stage_elapsed_s": now - self._start, "no_measured_progress_s": no_progress,
            "last_measured_progress_time_s": self._last_progress,
            "position_error_m": position, "rotation_error_rad": rotation,
            "best_position_error_m": self._best_position, "best_rotation_error_rad": self._best_rotation,
            "position_improved": improved_position, "rotation_improved": improved_rotation,
            "pose_goal_reached_only": goal_reached, "window_samples": len(self._history),
            "window_duration_s": now - self._history[0]["time_s"],
            "pause_reason_counts": dict(Counter(r["pause_reason"] for r in self._history if r["pause_reason"])),
            "error_and_reference_ranges": ranges,
            "thresholds": {"duration_s": self.duration_s, "position_improvement_m": self.position_improvement_m,
                           "rotation_improvement_rad": self.rotation_improvement_rad},
            "scope": "motion_stagnation_only", "physical_success_verified": False,
        }
        return dict(self._report)
