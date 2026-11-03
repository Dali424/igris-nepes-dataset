"""Opt-in in-place CRS handover for the catalog-v2 upright pickup/load (``task["handover"]``).

The pick arm (``task["arm"]``, the GIVING arm) lifts the ring and, after the
carry posture, both arms move along one designed joint path to the pre-handover
posture. The RECEIVING arm then regrasps the ring from its measured pose
(pregrasp, approach, close), the giving jaw opens while the receiving pinch is
verified, the giving hand retreats along declared waypoints, and the machine
continues with the receiving arm as its task arm (side-step, align, insert,
place, release, retreat, rest, settle). Six stages are inserted after
``carry_posture``::

    handover_posture -> handover_pregrasp -> handover_approach -> handover_close
    -> handover_release -> handover_retreat

Absent the block, every profile, machine command, runtime path and admission
rule is unchanged. This module validates the block and provides the shared
helpers; it commands nothing and writes no robot or object state.

Profile schema (``tasks.crs_load_jig.handover``)::

    {"receiving_arm": "left",                      # the arm opposite task.arm
     "grasp_pose_object": [x, y, z, qw, qx, qy, qz],  # ring -> receiving TCP (e.g. Rz(sigma) G*)
     "pregrasp_offset_w": [dx, dy, dz],            # world offset of the receiving pregrasp (>= 1 cm)
     "release_arm_offset_w": [dx, dy, dz],         # giving-hand separation after its jaw opens (5-50 mm)
     "release_retreat_waypoints_w": [[...], ...],  # 1..8 giving-hand world offsets from its release origin;
                                                   # the first collinear with, and at least as long as,
                                                   # release_arm_offset_w
     "posture_paths": {"handover_posture": {...}}, # one 19-joint path (posture_paths schema, no hold mode)
     "dwell_s": 0.5,                               # receiving pinch verified for this long before the retreat
     "minimum_hand_tcp_distance_m": 0.08,          # kinematic hand-hand proxy: measured TCP-TCP distance
     # optional
     "approach_pose_control": {TcpPoseController settings},   # receiving pregrasp/approach reference
     "retreat_pose_control": {TcpPoseController settings},    # giving separation/retreat reference
     "servo_speed_scale": 0.25,                               # (0, 1]; both arms' servos in the handover
     "close_acquisition_position_m": 0.01, "close_acquisition_rotation_rad": 0.1,
     "open_gripper_position_rad": 0.4, "closed_gripper_position_rad": 0.0,  # the receiving jaw
     # optional two-hand rotation: one more stage, handover_rotate, between handover_close and
     # handover_release (phase id 25 appended after handover_retreat's; absent -> no stage)
     "rotation_path": {...},                       # open-loop two-arm joint path, 1..16 waypoints, both jaws closed
     "max_tilt_rad": 0.14,                         # ring plane tilt from vertical allowed during the rotation
     "max_release_offset_deg": 10.0,               # receiving grasp point within this of the top of the rim at the end
     "receiving_grasp_point_object": [x, y, z],    # pinch pivot in the ring frame at the designed receiving grasp
                                                   # (default: the receiving TCP origin, i.e. grasp_pose_object[:3])
     # optional: the giving jaw stays closed while the receiving pinch is re-verified for this long at
     # handover_release entry (physics: close second, dwell, open first); absent -> it opens at entry
     "pre_release_dwell_s": 0.4}
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

import numpy as np

from .nepes_tasks import (HANDOVER_ROTATE_STAGE, HANDOVER_ROTATE_STAGES, HANDOVER_STAGES, PICKUP_LOAD_HANDOVER_STAGES,
                          UPRIGHT_PICKUP_SEQUENCE, _pose, _positive, _validate_posture_path, _vector, handover_stages)

HANDOVER_SCHEMA = "igris_c_crs_handover_v1"
HANDOVER_POSTURE_STAGE = "handover_posture"
# Every stage the handover can run (membership tests); handover_rotate only when rotation_path is declared.
HANDOVER_STAGE_SET = frozenset(HANDOVER_ROTATE_STAGES)
# Joint-space (two-arm posture path) stages of the handover; the others are Cartesian.
HANDOVER_JOINT_PATH_STAGES = (HANDOVER_POSTURE_STAGE, HANDOVER_ROTATE_STAGE)
# handover_release phases: ``verifying`` exists only with pre_release_dwell_s (the giving jaw stays closed).
RELEASE_VERIFYING_PHASE = "verifying"
RELEASE_PHASES = ("opening", "separating")
# Authored CRS frame: the ring plane is local xy, the normal local z and the centre the origin (the cart ring
# Ring_01 stands with its normal along world +Y). ``top`` is the ring-plane direction nearest world up.
RING_NORMAL_AXIS_OBJECT = (0., 0., 1.)
TOP_OFFSET_MEASUREMENT = ("angle_in_the_measured_ring_plane_between_world_up_and_ring_centre_to_receiving_grasp_point;"
                          "grasp_point_rides_with_the_measured_receiving_tcp")
RING_TILT_MEASUREMENT = "angle_of_the_measured_ring_normal_from_horizontal;zero_for_a_vertical_ring"
RING_ROTATION_MEASUREMENT = "ring_spin_about_its_own_normal_since_the_handover_rotate_entry;counter_clockwise_positive"
DEFAULT_MAX_TILT_RAD = .14
DEFAULT_MAX_RELEASE_OFFSET_DEG = 10.
_MAX_TILT_BOUND_RAD = .5
_MAX_RELEASE_OFFSET_BOUND_DEG = 90.
_MAX_ROTATION_WAYPOINTS = 16
_GRASP_POINT_RADIUS_BOUNDS_M = (.02, .5)
# The machine's observation arm becomes the receiving arm here (its pregrasp/approach/close
# are judged like the pick's), and the grasp reference switches at handover_release entry.
HANDOVER_OBSERVATION_SWITCH_STAGE = "handover_pregrasp"
HANDOVER_SWITCH_STAGE = "handover_release"
HANDOVER_CARTESIAN_STAGES = HANDOVER_STAGES[1:]
# Dataset labels (frame phase_name); phase ids stay nepes_tasks.STAGES.index(stage) (19..24).
HANDOVER_PHASES = {
    "handover_posture": "handover_align", "handover_pregrasp": "handover_pregrasp",
    "handover_approach": "handover_approach", "handover_close": "handover_grasp",
    "handover_release": "handover_release", "handover_retreat": "handover_retreat",
    HANDOVER_ROTATE_STAGE: "handover_rotate",  # phase id 25 (appended; never renumbers 19-24)
}
HAND_TCP_DISTANCE_MEASUREMENT = ("euclidean_distance_between_measured_left_and_right_hand_tcp_origins;"
                                 "kinematic_proxy_not_a_mesh_clearance")
REQUIRED_KEYS = frozenset(("receiving_arm", "grasp_pose_object", "pregrasp_offset_w", "release_arm_offset_w",
                           "release_retreat_waypoints_w", "posture_paths", "dwell_s", "minimum_hand_tcp_distance_m"))
OPTIONAL_KEYS = frozenset(("approach_pose_control", "retreat_pose_control", "servo_speed_scale",
                           "close_acquisition_position_m", "close_acquisition_rotation_rad",
                           "open_gripper_position_rad", "closed_gripper_position_rad",
                           "rotation_path", "max_tilt_rad", "max_release_offset_deg", "receiving_grasp_point_object",
                           "pre_release_dwell_s"))
ROTATION_KEYS = frozenset(("max_tilt_rad", "max_release_offset_deg", "receiving_grasp_point_object"))
_MAX_DWELL_S = 5.
_MAX_HAND_DISTANCE_M = .5
_SEPARATION_BOUNDS_M = (.005, .05)
_MAX_RETREAT_WAYPOINTS = 8
_HOLDING_BEFORE_RELEASE = PICKUP_LOAD_HANDOVER_STAGES[:PICKUP_LOAD_HANDOVER_STAGES.index(HANDOVER_SWITCH_STAGE)]
_OBSERVED_BY_GIVING = PICKUP_LOAD_HANDOVER_STAGES[:PICKUP_LOAD_HANDOVER_STAGES.index(HANDOVER_OBSERVATION_SWITCH_STAGE)]


def validate_handover(task: Mapping, thresholds: Mapping) -> dict:
    """Validate ``task["handover"]``; return a normalized copy (never mutates the profile)."""
    from .latest_robot import GRIPPER_CLOSED_TARGETS, GRIPPER_JOINT_NAMES, GRIPPER_OPEN_TARGETS
    from .posture_hold import is_applied_hold
    from .task_control import gripper_closed_position, gripper_open_position
    settings = task.get("handover")
    if not isinstance(settings, Mapping):
        raise ValueError("handover must be a mapping")
    keys = set(settings)
    if not REQUIRED_KEYS <= keys or keys - REQUIRED_KEYS - OPTIONAL_KEYS:
        raise ValueError("handover needs exactly " + ", ".join(sorted(REQUIRED_KEYS))
                         + " (optional: " + ", ".join(sorted(OPTIONAL_KEYS)) + ")")
    if task.get("stage_sequence") != UPRIGHT_PICKUP_SEQUENCE:
        raise ValueError("handover requires the upright pickup/load stage sequence")
    giving = task.get("arm")
    receiving = settings["receiving_arm"]
    if receiving not in ("left", "right") or giving not in ("left", "right") or receiving == giving:
        raise ValueError("handover.receiving_arm must be the arm opposite the task arm")
    grasp = _pose(settings["grasp_pose_object"], "handover.grasp_pose_object")
    pregrasp = _vector(settings["pregrasp_offset_w"], "handover.pregrasp_offset_w")
    if float(np.linalg.norm(pregrasp)) < .01:
        raise ValueError("handover.pregrasp_offset_w needs at least 1 cm clearance")
    separation = _vector(settings["release_arm_offset_w"], "handover.release_arm_offset_w")
    separation_length = float(np.linalg.norm(separation))
    if not _SEPARATION_BOUNDS_M[0] <= separation_length <= _SEPARATION_BOUNDS_M[1]:
        raise ValueError("handover.release_arm_offset_w must be a 5-50 mm giving-hand separation")
    points = settings["release_retreat_waypoints_w"]
    if not isinstance(points, list) or not 1 <= len(points) <= _MAX_RETREAT_WAYPOINTS:
        raise ValueError(f"handover.release_retreat_waypoints_w needs 1..{_MAX_RETREAT_WAYPOINTS} world offsets")
    waypoints = [_vector(point, "handover.release_retreat_waypoints_w") for point in points]
    first_length = float(np.linalg.norm(waypoints[0]))
    if (float(np.linalg.norm(np.cross(waypoints[0], separation))) > 1e-6 * first_length * separation_length
            or float(np.dot(waypoints[0], separation)) <= 0. or first_length + 1e-9 < separation_length):
        raise ValueError("The first handover retreat waypoint must continue the release separation: collinear, "
                         "same direction and at least as long as release_arm_offset_w")
    if any(float(np.linalg.norm(second - first)) < 1e-5 for first, second in zip(waypoints, waypoints[1:])):
        raise ValueError("handover.release_retreat_waypoints_w must make nonzero progress")
    paths = settings["posture_paths"]
    if not isinstance(paths, Mapping) or set(paths) != {HANDOVER_POSTURE_STAGE}:
        raise ValueError(f"handover.posture_paths must define exactly {HANDOVER_POSTURE_STAGE}")
    if is_applied_hold(paths[HANDOVER_POSTURE_STAGE]):
        raise ValueError("handover_posture is a designed two-arm joint path; hold_last_applied is not available")
    _validate_posture_path(HANDOVER_POSTURE_STAGE, paths[HANDOVER_POSTURE_STAGE],
                           prefix=f"handover.posture_paths.{HANDOVER_POSTURE_STAGE}",
                           max_waypoints=256 if "transport_constraints" in task else 8)
    dwell = _positive(settings["dwell_s"], "handover.dwell_s")
    if dwell > _MAX_DWELL_S:
        raise ValueError(f"handover.dwell_s must be at most {_MAX_DWELL_S} s")
    minimum = _positive(settings["minimum_hand_tcp_distance_m"], "handover.minimum_hand_tcp_distance_m")
    if minimum > _MAX_HAND_DISTANCE_M:
        raise ValueError(f"handover.minimum_hand_tcp_distance_m must be at most {_MAX_HAND_DISTANCE_M} m")
    giving_grasp = _pose(task.get("grasp_pose_object"), "grasp_pose_object")
    designed = float(np.linalg.norm(grasp[:3, 3] - giving_grasp[:3, 3]))
    if designed < minimum:
        raise ValueError("The two handover grasps place the hand TCPs closer than minimum_hand_tcp_distance_m "
                         f"({designed:.4f} m)")
    result = {"schema_version": HANDOVER_SCHEMA, "giving_arm": giving, "receiving_arm": receiving,
              "grasp_pose_object": [float(v) for v in settings["grasp_pose_object"]],
              "pregrasp_offset_w": pregrasp.tolist(), "release_arm_offset_w": separation.tolist(),
              "release_retreat_waypoints_w": [point.tolist() for point in waypoints],
              "posture_paths": deepcopy(dict(paths)), "dwell_s": dwell, "minimum_hand_tcp_distance_m": minimum,
              "designed_hand_tcp_distance_m": designed}
    for key in ("approach_pose_control", "retreat_pose_control"):
        if key in settings:
            from .crs_manipulation import TcpPoseController
            if not isinstance(settings[key], Mapping):
                raise ValueError(f"handover.{key} must be a mapping")
            identity = [0., 0., 0., 1., 0., 0., 0.]
            TcpPoseController(identity, identity, **settings[key])
            result[key] = deepcopy(dict(settings[key]))
    if "servo_speed_scale" in settings:
        scale = _positive(settings["servo_speed_scale"], "handover.servo_speed_scale")
        if scale > 1.:
            raise ValueError("handover.servo_speed_scale must be at most 1")
        result["servo_speed_scale"] = scale
    for key, bound in (("close_acquisition_position_m", "capture_position_m"),
                       ("close_acquisition_rotation_rad", "capture_rotation_rad")):
        if key in settings:
            value = _positive(settings[key], f"handover.{key}")
            if value > thresholds[bound]:
                raise ValueError(f"handover.{key} cannot exceed the immutable capture bound thresholds.{bound}")
            result[key] = value
    name = GRIPPER_JOINT_NAMES[0 if receiving == "left" else 1]
    grip = {"open_position_rad": GRIPPER_OPEN_TARGETS[name], "closed_position_rad": GRIPPER_CLOSED_TARGETS[name]}
    evidence = task.get("pinch_geometry_evidence")
    if "open_gripper_position_rad" in settings:
        result["open_gripper_position_rad"] = gripper_open_position(grip, settings["open_gripper_position_rad"])
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("A receiving jaw aperture needs the task pinch_geometry_evidence")
    if "closed_gripper_position_rad" in settings:
        value = gripper_closed_position(grip, settings["closed_gripper_position_rad"],
                                        settings.get("open_gripper_position_rad"))
        if (abs(value - GRIPPER_CLOSED_TARGETS[name]) >= task.get("maximum_pinch_angle_rad", .15)
                or not isinstance(evidence, str) or not evidence.strip()):
            raise ValueError("handover.closed_gripper_position_rad needs contact geometry evidence and must stay "
                             "within the original pinch bound")
        result["closed_gripper_position_rad"] = value
    if set(task.get("stage_postures", {})) & HANDOVER_STAGE_SET:
        raise ValueError("stage_postures cannot shape the handover stages; handover_posture owns the waist and "
                         "both arms are servoed afterwards")
    if "rotation_path" in settings:
        # Opt-in two-hand rotation: an open-loop two-arm joint path with both jaws closed (handover_rotate).
        path = settings["rotation_path"]
        if is_applied_hold(path):
            raise ValueError("handover.rotation_path is a designed two-arm joint path; hold_last_applied is not available")
        _validate_posture_path(HANDOVER_ROTATE_STAGE, path, prefix="handover.rotation_path",
                               max_waypoints=256 if "transport_constraints" in task else _MAX_ROTATION_WAYPOINTS)
        result["rotation_path"] = deepcopy(dict(path))
        result.update(rotation_settings(settings, grasp))
    elif ROTATION_KEYS & keys:
        raise ValueError("handover.max_tilt_rad, max_release_offset_deg and receiving_grasp_point_object describe the "
                         "two-hand rotation and need handover.rotation_path")
    if "pre_release_dwell_s" in settings:
        pre_release = _positive(settings["pre_release_dwell_s"], "handover.pre_release_dwell_s")
        if pre_release > _MAX_DWELL_S:
            raise ValueError(f"handover.pre_release_dwell_s must be at most {_MAX_DWELL_S} s")
        result["pre_release_dwell_s"] = pre_release
    return result


def rotation_settings(settings: Mapping, grasp: np.ndarray | None = None) -> dict:
    """Normalized bounds of the two-hand rotation (defaults filled); ``grasp`` is the receiving grasp matrix."""
    grasp = _pose(settings["grasp_pose_object"], "handover.grasp_pose_object") if grasp is None else grasp
    tilt = _positive(settings.get("max_tilt_rad", DEFAULT_MAX_TILT_RAD), "handover.max_tilt_rad")
    if tilt > _MAX_TILT_BOUND_RAD:
        raise ValueError(f"handover.max_tilt_rad must be at most {_MAX_TILT_BOUND_RAD} rad")
    offset = _positive(settings.get("max_release_offset_deg", DEFAULT_MAX_RELEASE_OFFSET_DEG),
                       "handover.max_release_offset_deg")
    if offset > _MAX_RELEASE_OFFSET_BOUND_DEG:
        raise ValueError(f"handover.max_release_offset_deg must be at most {_MAX_RELEASE_OFFSET_BOUND_DEG} deg")
    if "receiving_grasp_point_object" in settings:
        point_object = _vector(settings["receiving_grasp_point_object"], "handover.receiving_grasp_point_object")
    else:
        point_object = grasp[:3, 3].copy()  # the receiving TCP origin
    radius = float(math.hypot(point_object[0], point_object[1]))
    if not _GRASP_POINT_RADIUS_BOUNDS_M[0] <= radius <= _GRASP_POINT_RADIUS_BOUNDS_M[1]:
        raise ValueError("handover.receiving_grasp_point_object must lie 2-50 cm from the ring centre in the ring plane")
    point_tcp = (np.linalg.inv(grasp) @ np.array([*point_object, 1.]))[:3]
    return {"max_tilt_rad": tilt, "max_release_offset_deg": offset, "max_release_offset_rad": math.radians(offset),
            "receiving_grasp_point_object": [float(v) for v in point_object],
            "receiving_grasp_point_tcp": [float(v) for v in point_tcp]}


def release_phases(task: Mapping) -> tuple:
    """The handover_release phases of one profile: ``verifying`` first when pre_release_dwell_s is declared."""
    if "pre_release_dwell_s" in task["handover"]:
        return (RELEASE_VERIFYING_PHASE, *RELEASE_PHASES)
    return RELEASE_PHASES


def handover_events(task: Mapping) -> tuple:
    """The event names a handover profile records, in order (the admission requires exactly these)."""
    settings = task["handover"]
    events = ["observation_arm_switched"]
    if "rotation_path" in settings:
        events += ["two_hand_rotation_started", "two_hand_rotation_complete"]
    events.append("grasp_reference_switched")
    if "pre_release_dwell_s" in settings:
        events.append("receiving_pinch_verified")
    events += ["giving_jaw_open", "giving_hand_released", "handover_complete"]
    return tuple(events)


def handover_contract(task: Mapping) -> dict:
    """The run.json controller_contract.handover block; admission requires it verbatim."""
    settings = task["handover"]
    contract = {"schema_version": HANDOVER_SCHEMA, "giving_arm": task["arm"], "receiving_arm": settings["receiving_arm"],
                "stages": list(handover_stages(task)), "observation_arm_switch_stage": HANDOVER_OBSERVATION_SWITCH_STAGE,
                "grasp_reference_switch_stage": HANDOVER_SWITCH_STAGE,
                "holding_arm_provenance": "command.holding_arm per control step; frames.jsonl holding_arm per frame",
                "hand_tcp_distance_measurement": HAND_TCP_DISTANCE_MEASUREMENT,
                "minimum_hand_tcp_distance_m": float(settings["minimum_hand_tcp_distance_m"]),
                "hand_camera_arm": task["arm"]}
    if "rotation_path" in settings or "pre_release_dwell_s" in settings:
        # Additive: profiles without the two-hand rotation / pre-release dwell keep the original contract.
        contract["release_phases"] = list(release_phases(task))
        contract["events"] = list(handover_events(task))
    if "rotation_path" in settings:
        rotation = rotation_settings(settings)
        contract["rotation_stage"] = HANDOVER_ROTATE_STAGE
        contract["rotation"] = {
            "inserted_after": "handover_close", "max_tilt_rad": rotation["max_tilt_rad"],
            "max_release_offset_deg": rotation["max_release_offset_deg"],
            "receiving_grasp_point_object": rotation["receiving_grasp_point_object"],
            "ring_normal_axis_object": list(RING_NORMAL_AXIS_OBJECT), "top_offset_measurement": TOP_OFFSET_MEASUREMENT,
            "ring_tilt_measurement": RING_TILT_MEASUREMENT, "ring_rotation_measurement": RING_ROTATION_MEASUREMENT,
            "holding_arm": "both", "base_command": "exactly zero"}
    return contract


def posture_path_for_stage(task: Mapping, stage_id: str):
    """The joint path owning a joint-posture stage: ``posture_paths``, else the handover block.

    navigate_target is a joint-posture stage only before the goal latch of the opt-in
    ``navigation_control.single_step`` (single_step.py): its pre_step_posture_path owns it then.
    """
    path = task.get("posture_paths", {}).get(stage_id)
    if path is None and stage_id == HANDOVER_POSTURE_STAGE:
        path = task.get("handover", {}).get("posture_paths", {}).get(stage_id)
    if path is None and stage_id == HANDOVER_ROTATE_STAGE:
        path = task.get("handover", {}).get("rotation_path")
    if path is None and stage_id == "navigate_target":
        path = task.get("navigation_control", {}).get("single_step", {}).get("pre_step_posture_path")
    if path is None:
        raise ValueError(f"No joint path is declared for stage {stage_id}")
    return path


def jaw_positions(task: Mapping, arm: str):
    """(open, closed) jaw positions declared for ``arm``: the task's for its arm, the handover's for the receiving arm."""
    if arm == task.get("arm"):
        return task.get("open_gripper_position_rad"), task.get("closed_gripper_position_rad")
    settings = task.get("handover")
    if isinstance(settings, Mapping) and settings.get("receiving_arm") == arm:
        return settings.get("open_gripper_position_rad"), settings.get("closed_gripper_position_rad")
    return None, None


def hand_tcp_distance(tcp_poses_w: Mapping, giving: str, receiving: str) -> float:
    """Kinematic hand-hand proxy: the distance between the two measured TCP origins (m)."""
    first = _pose(tcp_poses_w[giving], f"tcp_poses_w[{giving}]")
    second = _pose(tcp_poses_w[receiving], f"tcp_poses_w[{receiving}]")
    return float(np.linalg.norm(first[:3, 3] - second[:3, 3]))


def holding_arm_for_stage(task: Mapping, stage_id: str) -> str:
    """Which hand holds the ring in a stage: giving, ``both`` during the two-hand rotation and while the giving
    jaw opens, then receiving."""
    if stage_id in _HOLDING_BEFORE_RELEASE:
        return task["arm"]
    if stage_id in (HANDOVER_ROTATE_STAGE, HANDOVER_SWITCH_STAGE):
        return "both"
    return task["handover"]["receiving_arm"]


def observation_arm_for_stage(task: Mapping, stage_id: str) -> str:
    """The machine's observation (command ``arm``) in a stage of a handover profile."""
    return task["arm"] if stage_id in _OBSERVED_BY_GIVING else task["handover"]["receiving_arm"]


def hand_commands_for_stage(task: Mapping, stage_id: str, arm_command: str, *, running: bool = True,
                            release_phase: str | None = None) -> dict:
    """Per-arm jaw commands of a handover profile; ``arm_command`` is the observation arm's command.

    ``release_phase`` is the handover_release phase of the command: while it is ``verifying`` (opt-in
    pre_release_dwell_s) the giving jaw stays closed; absent the option the giving jaw opens at entry.
    """
    giving, receiving = task["arm"], task["handover"]["receiving_arm"]
    if not running:
        return {giving: "hold", receiving: "hold"}
    observed = observation_arm_for_stage(task, stage_id)
    other = receiving if observed == giving else giving
    if stage_id in _OBSERVED_BY_GIVING:
        other_command = "hold"  # the receiving jaw is not touched before the handover
    elif stage_id in ("handover_pregrasp", "handover_approach", "handover_close", HANDOVER_ROTATE_STAGE):
        other_command = "close"  # the giving hand keeps the ring (both hands pinch during the rotation)
    elif stage_id == HANDOVER_SWITCH_STAGE and release_phase == RELEASE_VERIFYING_PHASE:
        other_command = "close"  # close second, dwell, open first: the giving jaw waits for the verification
    else:
        other_command = "open"  # the giving jaw opens at handover_release and stays open
    return {observed: arm_command, other: other_command}


# --- two-hand rotation geometry (shared by the machine and the admission) --------------------------------

def ring_tilt_rad(ring_w: np.ndarray) -> float:
    """Tilt of the ring plane from vertical: the angle of the measured ring normal from horizontal (rad)."""
    normal = ring_w[:3, :3] @ np.asarray(RING_NORMAL_AXIS_OBJECT, float)
    return float(math.asin(min(1., abs(float(normal[2])) / max(float(np.linalg.norm(normal)), 1e-12))))


def rim_offset_from_top(ring_w: np.ndarray, point_w) -> tuple[float, float] | None:
    """(unsigned, signed) angle in the measured ring plane between world up and ring centre -> ``point_w``.

    The signed angle is counter-clockwise positive about the measured ring normal; None when the ring is
    horizontal (no top) or the point sits on the ring axis (no rim direction).
    """
    rotation = ring_w[:3, :3]
    normal = rotation @ np.asarray(RING_NORMAL_AXIS_OBJECT, float)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
    up = np.array([0., 0., 1.])
    up = up - float(np.dot(up, normal)) * normal
    radial = np.asarray(point_w, float)[:3] - ring_w[:3, 3]
    radial = radial - float(np.dot(radial, normal)) * normal
    if float(np.linalg.norm(up)) < 1e-6 or float(np.linalg.norm(radial)) < 1e-6:
        return None
    up, radial = up / np.linalg.norm(up), radial / np.linalg.norm(radial)
    cosine = float(np.clip(np.dot(up, radial), -1., 1.))
    signed = float(math.atan2(float(np.dot(np.cross(up, radial), normal)), cosine))
    return float(math.acos(cosine)), signed


def ring_rotation_about_normal(entry_w: np.ndarray, now_w: np.ndarray) -> float:
    """Spin of the ring about its own normal since ``entry_w`` (rad, counter-clockwise positive)."""
    relative = entry_w[:3, :3].T @ now_w[:3, :3]
    return float(math.atan2(relative[1, 0], relative[0, 0]))


def rotation_measurement(settings: Mapping, ring_w: np.ndarray, receiving_tcp_w: np.ndarray,
                         entry_ring_w: np.ndarray | None = None) -> dict:
    """The two-hand rotation's measured quantities from one observation; ``settings`` is validate_handover's result.

    The receiving grasp point rides with the measured receiving TCP (receiving_grasp_point_tcp), so the
    offset from the top is the physical pinch position whatever the hand did in the meantime.
    """
    point_w = (receiving_tcp_w @ np.array([*settings["receiving_grasp_point_tcp"], 1.]))[:3]
    tilt = ring_tilt_rad(ring_w)
    offset = rim_offset_from_top(ring_w, point_w)
    unsigned, signed = (None, None) if offset is None else offset
    rotation = None if entry_ring_w is None else ring_rotation_about_normal(entry_ring_w, ring_w)
    return {"ring_tilt_rad": tilt, "max_tilt_rad": settings["max_tilt_rad"], "ring_tilt_valid": tilt <= settings["max_tilt_rad"],
            "ring_tilt_measurement": RING_TILT_MEASUREMENT,
            "receiving_grasp_point_tcp": list(settings["receiving_grasp_point_tcp"]),
            "receiving_grasp_point_w": [float(v) for v in point_w],
            "receiving_offset_from_top_rad": unsigned,
            "receiving_offset_from_top_deg": None if unsigned is None else math.degrees(unsigned),
            "receiving_offset_from_top_signed_rad": signed,
            "max_release_offset_deg": settings["max_release_offset_deg"],
            "at_top": bool(unsigned is not None and unsigned <= settings["max_release_offset_rad"]),
            "top_offset_measurement": TOP_OFFSET_MEASUREMENT,
            "ring_rotation_rad": rotation, "ring_rotation_deg": None if rotation is None else math.degrees(rotation),
            "ring_rotation_measurement": RING_ROTATION_MEASUREMENT}


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))
