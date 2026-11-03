"""Optional stricter seating readiness before holding and opening a loaded CRS.

This changes reference selection and the pre-release condition only. The JIG's
original drift anchor, acquisition transform and released-object acceptance
remain unchanged. It does not prove mechanical stability after opening.

Opt-in ``allow_press_in: true`` (runtime_v26 C2) lets ``command_height_offset_m``
go negative down to -0.0005 m: the place goal then lies up to 0.5 mm below the
seat along JIG-local Z, a bounded push into the seat (the runtime_v26 review cut
the bound from 1.5 mm; a blocked command error is integrated by the arm IK).
Acceptance poses, the alignment readiness and the 5 mm JIG drift gate are
unchanged; the pending placement latch that lets the descent press is bounded in
time and force (placement_hold.py).

The separate experimental ``press_in_mode: bounded_command_overtravel`` permits
an explicit command down to -2 mm with the immediate raw-support stop. It does
not change the old cap, physical contact model, acceptance pose, or geometry
policy. A blocked position goal is not a force-controlled or force-bounded goal.
"""
from collections.abc import Mapping
import math
from numbers import Real

import numpy as np

from .object_symmetry import symmetric_pose_errors
from .snapshot import matrix_to_pose
from .task_control import rotation_error

TARGET_FRAME = "measured_jig_at_place_entry"
# runtime_v26 C2: the deepest commanded press-in below the seat when allow_press_in is true.
PRESS_IN_MAXIMUM_M = .0005
OVERTRAVEL_MODE = "bounded_command_overtravel"
OVERTRAVEL_MAXIMUM_M = .002


def validate_placement_alignment(settings, stationary_spec):
    keys = {"normal_position_m", "height_m", "rotation_rad", "target_frame"}
    optional = {"command_height_offset_m", "support_hold_dropout_grace_s", "allow_press_in", "press_in_mode"}
    if (not isinstance(settings, Mapping) or not keys.issubset(settings)
            or set(settings) - keys - optional):
        raise ValueError("placement_alignment requires normal_position_m, height_m, rotation_rad and target_frame")
    if settings["target_frame"] != TARGET_FRAME:
        raise ValueError("placement_alignment.target_frame must be measured_jig_at_place_entry")
    limits = {"normal_position_m": stationary_spec.get("slot_normal_position_tolerance_m"),
              "height_m": stationary_spec.get("slot_position_tolerance_m"),
              "rotation_rad": stationary_spec.get("slot_rotation_tolerance_rad")}
    result = {"target_frame": TARGET_FRAME}
    for name, limit in limits.items():
        value = settings[name]
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not math.isfinite(value) or not isinstance(limit, Real)
                or not math.isfinite(limit) or not 0 < value <= limit):
            raise ValueError("placement_alignment." + name + " must narrow the existing placement tolerance")
        result[name] = float(value)
    press_in = False
    if "allow_press_in" in settings:
        press_in = settings["allow_press_in"]
        if type(press_in) is not bool:
            raise ValueError("placement_alignment.allow_press_in must be an explicit boolean")
    overtravel = "press_in_mode" in settings
    if overtravel:
        value = settings.get("command_height_offset_m")
        if (settings["press_in_mode"] != OVERTRAVEL_MODE or not press_in
                or isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not math.isfinite(value) or not -OVERTRAVEL_MAXIMUM_M <= value < 0):
            raise ValueError("placement_alignment.press_in_mode requires bounded_command_overtravel, "
                             "allow_press_in: true and an explicit command_height_offset_m in [-0.002, 0)")
        result["press_in_mode"] = OVERTRAVEL_MODE
    if "command_height_offset_m" in settings:
        value = settings["command_height_offset_m"]
        lowest = -OVERTRAVEL_MAXIMUM_M if overtravel else (-PRESS_IN_MAXIMUM_M if press_in else 0)
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not math.isfinite(value) or not lowest <= value <= result["height_m"]):
            # Without the runtime_v26 allow_press_in key this is exactly the runtime_v25 text; the press-in hints
            # only appear when the key is declared (and the declined hint only for a negative offset).
            suffix = ""
            if press_in:
                suffix = " or the bounded 0.5 mm press-in"
            elif ("allow_press_in" in settings and isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
                  and math.isfinite(value) and value < 0):
                suffix = " (a negative press-in down to -0.0005 m needs allow_press_in: true)"
            raise ValueError("placement_alignment.command_height_offset_m must stay inside the original alignment "
                             "height bound" + suffix)
        result["command_height_offset_m"] = float(value)
    if "support_hold_dropout_grace_s" in settings:
        value = settings["support_hold_dropout_grace_s"]
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not math.isfinite(value) or not 0 < value <= .2):
            raise ValueError("placement_alignment.support_hold_dropout_grace_s must be in (0, 0.2] seconds")
        result["support_hold_dropout_grace_s"] = float(value)
    if "allow_press_in" in settings:
        result["allow_press_in"] = press_in
    return result


def placement_overtravel_contract(task):
    """Declared command depth only; never permission for an actual solid overlap.

    The old 0.5 mm limit is deliberately unchanged: blocked error can integrate
    through the arm IK. This experimental mode authorizes a deeper command,
    retains the existing first-support measured-TCP stop, and requires native
    force/drift/contact checks. It does not make contact force bounded.
    """
    settings = task.get("placement_alignment", {})
    if "press_in_mode" not in settings:
        return None
    return {"schema_version": "igris_c_bounded_command_overtravel_v1",
            "mode": settings["press_in_mode"],
            "command_height_offset_m": settings["command_height_offset_m"],
            "maximum_command_depth_m": OVERTRAVEL_MAXIMUM_M,
            "command_frame": "measured_jig_at_place_entry_local_z",
            "support_stop": "existing_first_qualified_support_measured_TCP_hold",
            "arm_IK_continues_tracking_held_TCP": True,
            "actual_penetration_authorized": False,
            "force_control_or_force_bound_claim": False,
            "acceptance_pose_and_raw_final_gates_changed": False,
            "physical_model_and_geometry_masks_changed": False}


def placement_command_slot_pose(settings, target_slot_pose):
    """Offset only the controller goal along JIG-local Z; never the acceptance pose.

    The explicit offset compensates a measured physical seating height. It is
    bounded by the existing readiness range and is not inferred from the object
    under control. No object, JIG or capture transform is rewritten. With
    allow_press_in a negative offset commands a bounded push below the seat
    (normally >= -0.5 mm; the explicit experimental mode permits >= -2 mm).
    The acceptance slot pose is still the unshifted target.
    """
    goal = target_slot_pose.copy()
    goal[2, 3] += settings.get("command_height_offset_m", 0.)
    return goal


def measure_placement_alignment(settings, object_pose, jig_pose, target_slot_pose, axis_object=None):
    """JIG-frame seating errors; with a declared symmetry ``axis_object`` the rotation ignores the spin."""
    relative = np.linalg.inv(jig_pose) @ object_pose
    delta = relative[:3, 3] - target_slot_pose[:3, 3]
    if axis_object is None:
        rotation = float(np.linalg.norm(rotation_error(relative[:3, :3], target_slot_pose[:3, :3])))
    else:
        rotation = symmetric_pose_errors(relative, target_slot_pose, axis_object)[1]
    errors = {"normal_position_m": abs(float(delta[1])), "height_m": abs(float(delta[2])),
              "rotation_rad": rotation}
    passed = {key: value <= settings[key] for key, value in errors.items()}
    return {"schema_version": "igris_c_placement_alignment_v1", "ready": all(passed.values()),
            "errors": errors, "limits": dict(settings), "checks": passed,
            "slot_relative_pose": matrix_to_pose(relative),
            "scope": "additional_place_hold_and_opening_readiness",
            "released_acceptance_changed": False, "jig_drift_anchor_changed": False}
