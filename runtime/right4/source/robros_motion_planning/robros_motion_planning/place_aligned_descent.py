"""Opt-in lateral alignment gating for lowering the CRS into its JIG slot.

The slot constrains the ring laterally through its two guide notches: about
+-1.5 mm along JIG x at the notch bottoms and +-2 mm along the slot normal.
A straight place reference carries the transfer-end lateral error down into
the notches, where a stiff arm then presses the rim into a guide (run_44).
This helper changes command timing only:

* the transfer stage may end only once the measured ring is laterally aligned
  with the ORIGINAL acceptance slot pose for a continuous dwell;
* in the place stage, while the ring is still clearly above the seat and the
  measured lateral error exceeds a bound, only the nominal descent reference is
  paused (``object_advance_hold_reason``); object feedback keeps correcting.

No acceptance gate, capture reference, JIG anchor, success dwell or physics
parameter is changed, and nothing here proves physical insertion.
"""
from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Real

import numpy as np

from .object_symmetry import symmetric_pose_errors
from .task_control import rotation_error

SCHEMA = "igris_c_place_aligned_descent_v1"
HOLD_REASON = "place_lateral_misaligned"
KEYS = ("transfer_exit_lateral_m", "transfer_exit_rotation_rad", "transfer_exit_dwell_s",
        "descent_pause_lateral_m", "descent_resume_lateral_m", "descent_pause_grace_s",
        "descent_pause_minimum_height_m", "descent_hold_maximum_s")


def _finite(value):
    return (isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
            and math.isfinite(value))


def validate_place_aligned_descent(settings, stationary_spec):
    """Every bound must be explicit, finite and no looser than the acceptance."""
    if not isinstance(settings, Mapping) or set(settings) != set(KEYS):
        raise ValueError("place_aligned_descent requires exactly " + ", ".join(KEYS))
    if not isinstance(stationary_spec, Mapping):
        raise ValueError("place_aligned_descent requires the stationary_task acceptance bounds")
    position_bound = stationary_spec.get("slot_position_tolerance_m")
    rotation_bound = stationary_spec.get("slot_rotation_tolerance_rad")
    if not _finite(position_bound) or not _finite(rotation_bound):
        raise ValueError("place_aligned_descent requires finite slot acceptance tolerances")
    result = {}
    for key in KEYS:
        value = settings[key]
        if not _finite(value):
            raise ValueError(f"place_aligned_descent.{key} must be a finite number")
        result[key] = float(value)
    bounds = {"transfer_exit_lateral_m": (0., position_bound),
              "transfer_exit_rotation_rad": (0., rotation_bound),
              "transfer_exit_dwell_s": (.1, 5.),
              "descent_pause_lateral_m": (0., position_bound),
              "descent_resume_lateral_m": (0., position_bound),
              "descent_pause_grace_s": (0., 1.),
              "descent_pause_minimum_height_m": (0., .05),
              "descent_hold_maximum_s": (.5, 30.)}
    for key, (low, high) in bounds.items():
        value = result[key]
        if value > high or value < low or (value <= 0. and key not in ("descent_pause_grace_s",)):
            raise ValueError(f"place_aligned_descent.{key} must be positive and cannot exceed the acceptance bound")
    if result["descent_resume_lateral_m"] > result["descent_pause_lateral_m"]:
        raise ValueError("place_aligned_descent.descent_resume_lateral_m cannot exceed descent_pause_lateral_m")
    return result


def vertical_descent_start(goal_object_pose_w, control_object_to_tcp, measured_object_pose_w):
    """TCP pose directly above the place goal at the measured ring height.

    A place reference that starts at the measured hand carries the whole
    transfer-end lateral error down the straight descent (run_44/run_46). Starting
    the nominal reference above the goal makes the nominal path vertical, so the
    bounded object feedback removes the lateral error from the first tick. The
    goal, the capture reference and every gate are untouched.
    """
    from .snapshot import matrix_to_pose, pose_to_matrix
    goal = pose_to_matrix(goal_object_pose_w)
    measured = pose_to_matrix(measured_object_pose_w)
    start = goal.copy()
    start[2, 3] = float(measured[2, 3])
    if not np.isfinite(start).all():
        raise ValueError("vertical_descent_start needs finite poses")
    return matrix_to_pose(start @ pose_to_matrix(control_object_to_tcp))


def measure_lateral_alignment(object_pose, jig_pose, slot_pose, axis_object=None):
    """Signed JIG-frame offsets of the measured ring from the acceptance slot pose.

    With a declared symmetry ``axis_object`` the rotation error ignores the ring's spin about that axis.
    """
    relative = np.linalg.inv(jig_pose) @ object_pose
    delta = relative[:3, 3] - slot_pose[:3, 3]
    if axis_object is None:
        rotation = float(np.linalg.norm(rotation_error(relative[:3, :3], slot_pose[:3, :3])))
    else:
        rotation = symmetric_pose_errors(relative, slot_pose, axis_object)[1]
    return {"lateral_x_m": float(delta[0]), "lateral_normal_m": float(delta[1]),
            "height_m": float(delta[2]), "rotation_rad": rotation}


class PlaceAlignedDescent:
    """Transfer-exit alignment predicate and the hysteretic descent hold."""

    def __init__(self, settings, stationary_spec):
        self.settings = validate_place_aligned_descent(settings, stationary_spec)
        self.reset()

    def reset(self):
        self._misaligned_since = None
        self._hold_active = False
        self._hold_since = None
        self._hold_total_s = 0.
        self._hold_exhausted = False
        self._last_time_s = None
        self.evidence = None

    @staticmethod
    def _lateral(errors):
        return max(abs(errors["lateral_x_m"]), abs(errors["lateral_normal_m"]))

    def transfer_aligned(self, errors):
        s = self.settings
        return bool(self._lateral(errors) <= s["transfer_exit_lateral_m"]
                    and errors["rotation_rad"] <= s["transfer_exit_rotation_rad"])

    def update(self, stage, now, errors, *, supported):
        """Return the descent hold reason for this control step, or None.

        ``supported`` means the measured support/hold logic owns the hand goal;
        the hold never interferes with it or with the opening dwell.
        """
        if not _finite(now):
            raise ValueError("place_aligned_descent needs a finite observation time")
        s = self.settings
        lateral = self._lateral(errors)
        aligned = self.transfer_aligned(errors)
        if self._hold_active and self._last_time_s is not None:
            self._hold_total_s += max(0., now - self._last_time_s)
        self._last_time_s = now
        hold = None
        if stage == "place" and not supported and errors["height_m"] > s["descent_pause_minimum_height_m"]:
            # The hold is bounded: the arm may not be able to realise the resume tolerance, and the
            # motion watchdog must not be starved; after the budget the nominal descent continues.
            if self._hold_total_s >= s["descent_hold_maximum_s"] - 1e-10:
                self._hold_exhausted = True
            if self._hold_exhausted:
                self._hold_active, self._misaligned_since = False, None
            elif self._hold_active:
                if lateral <= s["descent_resume_lateral_m"]:
                    self._hold_active, self._misaligned_since = False, None
            elif lateral > s["descent_pause_lateral_m"]:
                if self._misaligned_since is None:
                    self._misaligned_since = now
                if now - self._misaligned_since >= s["descent_pause_grace_s"] - 1e-10:
                    self._hold_active = True
                    self._hold_since = now
            else:
                self._misaligned_since = None
            hold = HOLD_REASON if self._hold_active else None
        else:
            self._hold_active, self._misaligned_since = False, None
        self.evidence = {"schema_version": SCHEMA, "measurement_time_s": float(now), "stage_id": stage,
                         "errors": dict(errors), "lateral_error_m": lateral,
                         "transfer_exit_aligned": aligned, "descent_hold_active": bool(hold is not None),
                         "descent_hold_reason": hold, "supported_or_held": bool(supported),
                         "misaligned_since_s": self._misaligned_since, "hold_since_s": self._hold_since,
                         "hold_total_s": self._hold_total_s, "hold_budget_exhausted": self._hold_exhausted,
                         "scope": "transfer_exit_dwell_and_bounded_nominal_descent_pause_only",
                         "acceptance_or_capture_changed": False, "settings": dict(s)}
        return hold
