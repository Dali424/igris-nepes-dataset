"""Optional measured waist targets aligned with upright object waypoints.

This selects commands and arrival criteria only. The runtime must still check
the loaded robot's joint limits and execute bounded feedback; active-arm IK,
physical grasp retention, payload clearance and balance remain separate checks.
"""
from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Real


WAIST_JOINT_NAMES = ("Joint_Waist_Yaw", "Joint_Waist_Roll", "Joint_Waist_Pitch")
WAYPOINT_POSTURE_STAGES = frozenset(("lift", "transfer"))


def _finite(value, name, *, positive=False, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    value = float(value)
    if positive and value <= 0 or maximum is not None and value > maximum:
        raise ValueError(f"{name} is outside its permitted bounds")
    return value


def validate_object_waypoint_postures(task):
    """Return independent normalized stage lists, or {} for an unchanged legacy task."""
    if not isinstance(task, Mapping):
        raise ValueError("object_waypoint_postures requires a task mapping")
    if "object_waypoint_postures" not in task:
        return {}
    from .nepes_tasks import UPRIGHT_PICKUP_SEQUENCE
    if task.get("stage_sequence") != UPRIGHT_PICKUP_SEQUENCE:
        raise ValueError("object_waypoint_postures requires the upright pickup/load stage sequence")
    declared = task["object_waypoint_postures"]
    if (not isinstance(declared, Mapping) or not declared
            or set(declared) - WAYPOINT_POSTURE_STAGES):
        raise ValueError("object_waypoint_postures must contain lift and/or transfer")
    stages = task.get("stage_postures", {})
    if not isinstance(stages, Mapping) or set(stages) & set(declared):
        raise ValueError("object_waypoint_postures conflicts with stage_postures for the same stage")
    offsets = task.get("object_waypoint_offsets_w")
    if not isinstance(offsets, Mapping):
        raise ValueError("object_waypoint_postures requires object_waypoint_offsets_w")
    normalized = {}
    for stage, postures in declared.items():
        waypoints = offsets.get(stage)
        if (not isinstance(waypoints, list) or not waypoints or not isinstance(postures, list)
                or len(postures) != len(waypoints)):
            raise ValueError(f"object_waypoint_postures.{stage} must match the object waypoint count")
        normalized[stage] = []
        for index, posture in enumerate(postures):
            prefix = f"object_waypoint_postures.{stage}[{index}]"
            if (not isinstance(posture, Mapping)
                    or set(posture) != {"joint_positions", "duration_s", "tolerance_rad", "feedback"}):
                raise ValueError(f"{prefix} needs exactly joint_positions, duration_s, tolerance_rad and feedback")
            positions = posture["joint_positions"]
            if not isinstance(positions, Mapping) or set(positions) != set(WAIST_JOINT_NAMES):
                raise ValueError(f"{prefix} must command exactly the three waist joints")
            if posture["feedback"] is not True:
                raise ValueError(f"{prefix}.feedback must be true")
            normalized[stage].append({
                "joint_positions": {name: _finite(positions[name], f"{prefix}.{name}") for name in WAIST_JOINT_NAMES},
                "duration_s": _finite(posture["duration_s"], f"{prefix}.duration_s", positive=True),
                "tolerance_rad": _finite(posture["tolerance_rad"], f"{prefix}.tolerance_rad", positive=True, maximum=.15),
                "feedback": True})
    return normalized


def stage_posture(task, stage_id, waypoint_index=0):
    """Select this object waypoint's waist goal, otherwise the existing stage posture.

    Return a copy so controllers cannot alter the declared profile. An invalid
    index on an opted-in stage must fail instead of falling back to a stale goal.
    """
    postures = validate_object_waypoint_postures(task)
    if stage_id in postures:
        if type(waypoint_index) is not int or not 0 <= waypoint_index < len(postures[stage_id]):
            raise ValueError(f"object_waypoint_postures.{stage_id} has no waypoint index {waypoint_index!r}")
        return deepcopy(postures[stage_id][waypoint_index])
    return deepcopy(task.get("stage_postures", {}).get(stage_id))
