"""Opt-in tighter measured-object arrival at a suffix of TRANSFER waypoints.

This is an additional arrival condition, not a reference lead limit, collision
monitor or replacement for any grasp, transport, dwell or placement condition.
Waypoint indices are zero-based and inclusive. No controller target is changed.
"""
from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Integral, Real

import numpy as np


def validate_transfer_arrival_precision(settings, thresholds, waypoint_count):
    selectors = {"from_waypoint", "final_waypoints"}
    if (not isinstance(settings, Mapping)
            or len(selectors.intersection(settings)) != 1
            or "position_m" not in settings
            or set(settings) - selectors - {"position_m", "rotation_rad"}):
        raise ValueError("transfer_arrival_precision requires position_m and exactly one of "
                         "from_waypoint or final_waypoints; rotation_rad is optional")
    selector = next(iter(selectors.intersection(settings)))
    index = settings[selector]
    lower, upper = (0, waypoint_count - 1) if selector == "from_waypoint" else (1, waypoint_count)
    if (isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral)
            or not lower <= index <= upper):
        raise ValueError("transfer_arrival_precision." + selector + " is outside the transfer waypoint path")
    result = {selector: int(index)}
    for name, suffix in (("position_m", "position_m"), ("rotation_rad", "rotation_rad")):
        if name not in settings:
            continue
        value = settings[name]
        limit = min(thresholds["tcp_" + suffix], thresholds["object_" + suffix])
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not math.isfinite(value) or not 0 < value <= limit):
            raise ValueError("transfer_arrival_precision." + name + " must narrow the existing "
                             "TCP and object arrival tolerances")
        result[name] = float(value)
    return result


def first_precision_waypoint(settings, waypoint_count):
    """Resolve a validated selector against the actual transfer path length."""
    if "from_waypoint" in settings:
        return settings["from_waypoint"]
    return waypoint_count - settings["final_waypoints"]


def transfer_arrival_precision_contract(task):
    """Pin opt-in semantics so a runtime that silently ignores the key cannot be admitted."""
    if "transfer_arrival_precision" not in task:
        return None
    settings = task["transfer_arrival_precision"]
    count = len(task["object_waypoint_offsets_w"]["transfer"])
    return {"schema_version": "transfer_arrival_precision_v1", "stage": "transfer",
            "settings": deepcopy(settings), "from_waypoint": first_precision_waypoint(settings, count),
            "waypoint_count": count, "measurement": "actual_object_to_current_waypoint",
            "rotation_semantics": "existing_object_symmetry", "existing_conditions_and_dwell_retained": True}
