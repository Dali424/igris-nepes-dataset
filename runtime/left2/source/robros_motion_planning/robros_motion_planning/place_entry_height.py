"""Declared higher PLACE entry using the existing aligned two-speed descent.

This permits an earlier controlled insertion phase. Existing intended target-slot
contact and penetration checks still apply, with no outside-slot permission.
No controller, grasp, collision mask or acceptance pose is changed here.
Undeclared profiles retain the 20--35 mm rule.
"""
from collections.abc import Mapping
from copy import deepcopy
from numbers import Real
import math

import numpy as np


def validate_place_entry_height(document, task):
    """Validate the opt-in after the ordinary task fields have been validated."""
    value = task["place_entry_height_m"]
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(value) or not .035 <= value <= .060):
        raise ValueError("place_entry_height_m must be finite in [0.035, 0.060] metres")
    final = np.asarray(task["transfer_offset_w"], dtype=float)
    points = task.get("object_waypoint_offsets_w", {}).get("transfer", [])
    if (final.shape != (3,) or not np.isfinite(final).all()
            or np.any(np.abs(final[:2]) > 1e-9) or abs(final[2] - value) > 1e-9
            or not points or not np.allclose(points[-1], final, atol=1e-9, rtol=0)):
        raise ValueError("place_entry_height_m must equal the final vertical TRANSFER offset")
    if (document.get("attachment_mode") != "none"
            or task.get("target", {}).get("frame") != "jig"
            or task.get("placement_alignment", {}).get("target_frame") != "measured_jig_at_place_entry"):
        raise ValueError("place_entry_height_m requires no attachment and measured-JIG PLACE alignment")

    from .place_aligned_descent import validate_place_aligned_descent
    aligned = validate_place_aligned_descent(task.get("place_aligned_descent"), document.get("stationary_task"))
    if aligned["transfer_exit_lateral_m"] > .001 or aligned["transfer_exit_rotation_rad"] > .003:
        raise ValueError("place_entry_height_m requires final measured alignment <=1 mm / 0.003 rad")
    from .transfer_arrival_precision import validate_transfer_arrival_precision, first_precision_waypoint
    precision = validate_transfer_arrival_precision(task.get("transfer_arrival_precision"),
                                                    document["thresholds"], len(points))
    if (precision["position_m"] > .001 or precision.get("rotation_rad", math.inf) > .003
            or first_precision_waypoint(precision, len(points)) > len(points) - 1):
        raise ValueError("place_entry_height_m requires final strict arrival <=1 mm / 0.003 rad")

    from .place_release_options import split_two_speed_descent, validate_two_speed_descent
    override, two_speed = split_two_speed_descent(task.get("object_pose_control_overrides", {}).get("place", {}))
    effective = {**task.get("object_pose_control", {}), **override}
    if not two_speed:
        raise ValueError("place_entry_height_m requires an explicit two-speed PLACE descent")
    validate_two_speed_descent(two_speed, effective, task, "place_entry_height_m")
    if two_speed["slow_zone_height_m"] != .008 or effective["linear_speed"] > .002:
        raise ValueError("place_entry_height_m requires exactly 8 mm slow zone and slow speed <=2 mm/s")
    support = task.get("placement_support", {})
    if (not isinstance(support, Mapping) or support.get("object") != "jig"
            or support.get("minimum_load_fraction", 0.) < .8 or support.get("dwell_s", 0.) < 1.
            or any(k in support for k in ("filter_s", "latch_dwell_s"))
            or "support_hold_dropout_grace_s" in task.get("placement_alignment", {})
            or any(k in task for k in ("support_settling_hold_s", "support_gripper_preload_rad", "support_load_control"))):
        raise ValueError("place_entry_height_m requires original raw JIG support >=0.8mg/1s and immediate support stop")
    return float(value)


def place_entry_height_contract(task):
    """Declaration-only contract; admission must reject an ignored declaration."""
    if "place_entry_height_m" not in task:
        return None
    return {"schema_version": "igris_c_declared_place_entry_height_v1",
            "place_entry_height_m": task["place_entry_height_m"],
            "final_transfer_waypoint": len(task["object_waypoint_offsets_w"]["transfer"]) - 1,
            "final_transfer_offset_w": deepcopy(task["transfer_offset_w"]),
            "place_reference": "existing_measured_JIG_aligned_vertical_two_speed_descent",
            "place_aligned_descent": deepcopy(task["place_aligned_descent"]),
            "transfer_arrival_precision": deepcopy(task["transfer_arrival_precision"]),
            "place_control_override": deepcopy(task["object_pose_control_overrides"]["place"]),
            "phase_grasp_reference": "existing_normal_PLACE_entry_rule",
            "support_stop": "existing_first_qualified_support_measured_TCP_hold",
            "acceptance_pose_and_raw_final_gates_changed": False,
            "capture_reference_or_physical_model_changed": False,
            "contact_masks_or_penetration_permission_changed": False,
            "continuous_geometry_or_force_bound_proven": False}
