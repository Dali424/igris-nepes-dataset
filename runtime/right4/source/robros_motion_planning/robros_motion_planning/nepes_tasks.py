"""Measured-state task graphs for the five NEPES manipulation tasks.

The runtime adapter supplies synchronized physics poses, measured hand joint
states and object-specific hand contacts. A commanded hand closure is not grasp
evidence. No object pose or constraint is ever written by this module. In
particular, ``allowed_contact_paths`` is a contact *intent*, not permission to
disable all collision checking for that object.

Poses use meters and scalar-first quaternions [x,y,z,qw,qx,qy,qz]. Velocities used
for settling are derived from successive measurements, never from commands.
The bundled geometry-derived calibration is explicitly unvalidated; reaching a
TCP waypoint alone cannot complete a task. A live adapter must fail the machine
on planner/IK failures, collision-monitor faults, robot resets or a fall.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any
import hashlib
import json
import math

import numpy as np

from .snapshot import matrix_to_pose, pose_to_matrix
from .hinged_door import hinge_pose_w, measure_hinge_state, validate_hinge_config
from .object_relations import measure_object_relation, validate_relation_rotation
from .object_symmetry import pinch_point, spin_aligned_object, symmetric_capture_errors, symmetric_pose_errors


TASK_IDS = (
    "crs_load_jig", "open_cleaner_cover", "jig_insert_cleaner",
    "jig_extract_cleaner", "crs_final_place",
)
MANIPULATION_STAGES = (
    "prerequisites", "navigate_source", "pregrasp", "approach", "close",
    "lift", "navigate_target", "transfer", "place", "release", "retreat", "settle",
)
HINGE_STAGES = (*MANIPULATION_STAGES[:5], "open_hinge", "release", "retreat", "settle")
# Joint-space posture phases of the upright-start pickup/load sequence.
POSTURE_STAGES = ("reach_posture", "carry_posture", "place_posture", "rest_posture")
# Stable dataset phase IDs retain the legacy manipulation stages; later phases
# (hinge, recovery, upright postures 15-18) are only ever appended, while each
# task graph uses its own execution order.
# Opt-in in-place handover of the CRS from the pick arm to the other arm (task["handover"],
# handover.py): six stages inserted after carry_posture; their phase ids are appended (19-24).
HANDOVER_STAGES = ("handover_posture", "handover_pregrasp", "handover_approach", "handover_close",
                   "handover_release", "handover_retreat")
# Opt-in two-hand ring rotation (handover.rotation_path): one more recorded stage between handover_close
# and handover_release. Its phase id is appended after handover_retreat's (25); nothing is renumbered.
HANDOVER_ROTATE_STAGE = "handover_rotate"
HANDOVER_ROTATE_STAGES = (*HANDOVER_STAGES[:HANDOVER_STAGES.index("handover_release")], HANDOVER_ROTATE_STAGE,
                          *HANDOVER_STAGES[HANDOVER_STAGES.index("handover_release"):])
STAGES = (*MANIPULATION_STAGES, "open_hinge", "recover_carry", "recover_base", *POSTURE_STAGES, *HANDOVER_STAGES,
          HANDOVER_ROTATE_STAGE)
UPRIGHT_PICKUP_SEQUENCE = "upright_pickup_load_v1"
PICKUP_LOAD_STAGES = (
    "prerequisites", "navigate_source", "reach_posture", "pregrasp", "approach", "close", "lift",
    "carry_posture", "navigate_target", "place_posture", "transfer", "place", "release", "retreat",
    "rest_posture", "settle",
)
PICKUP_LOAD_HANDOVER_STAGES = (*PICKUP_LOAD_STAGES[:PICKUP_LOAD_STAGES.index("carry_posture") + 1], *HANDOVER_STAGES,
                               *PICKUP_LOAD_STAGES[PICKUP_LOAD_STAGES.index("carry_posture") + 1:])
PICKUP_LOAD_HANDOVER_ROTATE_STAGES = (*PICKUP_LOAD_STAGES[:PICKUP_LOAD_STAGES.index("carry_posture") + 1],
                                      *HANDOVER_ROTATE_STAGES,
                                      *PICKUP_LOAD_STAGES[PICKUP_LOAD_STAGES.index("carry_posture") + 1:])
SCHEMA_VERSION = "igris_c_nepes_tasks_v1"
_CARRYING = frozenset(("lift", "recover_base", "recover_carry", "navigate_target", "transfer", "place", "open_hinge",
                       "carry_posture", "place_posture"))
_RELEASED = frozenset(("retreat", "settle", "rest_posture"))
_UPRIGHT_PICKUP_KEYS = ("upright_start", "posture_paths", "object_pose_control_overrides", "retreat_waypoints_w",
                        "handover")
_LEGACY_INITIALIZATION_KEYS = ("initial_upper_joint_positions", "initial_arm_joint_positions", "initial_arm_ramp_seconds")
_POSTURE_PATH_KEYS = frozenset(("joint_names", "waypoints", "segment_min_duration_s", "max_speed_rad_s",
    "max_acceleration_rad_s2", "arrival_tolerance_rad", "settle_velocity_rad_s", "settle_dwell_s",
    "max_tracking_error_rad"))
# (lower, upper) bounds; lower None means strictly positive.
_POSTURE_PATH_BOUNDS = {"max_speed_rad_s": (None, .8), "max_acceleration_rad_s2": (None, 2.),
    "arrival_tolerance_rad": (None, .25), "settle_velocity_rad_s": (None, .2),
    "settle_dwell_s": (.1, 2.), "max_tracking_error_rad": (None, .6)}
# Explicit simulation experiment limits for individual moving references.
# These are command caps, not actuator ratings or physical stability guarantees.
_POSTURE_SIMULATION_FAST_BOUNDS = {"max_speed_rad_s": (None, 1.6),
                                 "max_acceleration_rad_s2": (None, 4.)}
# Optional per-path choice of the stillness measurement compared with settle_velocity_rad_s:
# the joint-velocity readback (default) or the speed implied by consecutive measured positions.
_POSTURE_SETTLE_SPEED_KEY = "settle_speed_source"
_POSTURE_SETTLE_SPEED_SOURCES = ("joint_velocities", "measured_positions")
# Canonical URDF bound; validate_snapshot tolerates only 1e-6 rad beyond it.
_NECK_PITCH_LIMIT_RAD = .523598776
# The canonical foot collision geometry lies 0.96107 m below base_link at the
# HugWBC default legs: a lower upright torso would start with the soles inside
# the floor. The runtime re-measures this with the loaded model.
_UPRIGHT_TORSO_HEIGHT_M = (.961, 1.)
# The unrecorded upright preparation; no joint command may be shaped there.
_UPRIGHT_PREPARATION_STAGES = ("prerequisites", "navigate_source")
_PREDICTIVE_BRAKING_DEFAULT_STAGES = ("navigate_source", "navigate_target", "recover_base")
_OBJECT_POSE_CONTROL_KEYS = frozenset(("linear_speed", "linear_acceleration", "angular_speed", "angular_acceleration",
    "position_lead", "rotation_lead", "pause_slip_position", "pause_slip_rotation",
    "cross_track_position", "cross_track_rotation", "reference_profile", "object_feedback"))


def _positive(value: Any, name: str, *, zero: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number) or number < 0 or (number == 0 and not zero):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'} and finite")
    return number


def _vector(value: Any, name: str, size: int = 3) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must have {size} finite values")
    return vector


def _pose(value: Any, name: str) -> np.ndarray:
    try:
        return pose_to_matrix(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name}: {exc}") from exc


def _errors(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    translation = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
    relative = first[:3, :3].T @ second[:3, :3]
    rotation = float(math.acos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
    return translation, rotation


def _shift(transform: np.ndarray, offset_w: Any) -> np.ndarray:
    result = transform.copy()
    result[:3, 3] += offset_w
    return result


def handover_stages(task):
    """Execution order of the opt-in handover stages of one profile (handover.py): seven with a rotation_path."""
    settings = task.get("handover")
    if isinstance(settings, Mapping) and "rotation_path" in settings:
        return HANDOVER_ROTATE_STAGES
    return HANDOVER_STAGES


def task_stages(task):
    if task.get("stage_sequence") == UPRIGHT_PICKUP_SEQUENCE:
        # The opt-in handover block inserts its stages after carry_posture (handover.py).
        if "handover" not in task:
            return PICKUP_LOAD_STAGES
        return (PICKUP_LOAD_HANDOVER_ROTATE_STAGES if handover_stages(task) is HANDOVER_ROTATE_STAGES
                else PICKUP_LOAD_HANDOVER_STAGES)
    if task.get("mechanism") == "revolute_hinge":
        return HINGE_STAGES
    recovery = (("recover_base",) if "recover_base_offset_w" in task else ()) + (
        ("recover_carry",) if "carry_recovery_offsets_w" in task else ())
    return (*MANIPULATION_STAGES[:6], *recovery, *MANIPULATION_STAGES[6:])


def _finite(value: Any, name: str) -> float:
    """A strict JSON number: booleans and numeric strings are not joint values."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _upright_pickup_identity(document: Mapping[str, Any]) -> bool:
    """The upright sequence belongs only to independent catalog-v2 task 1."""
    spec = document.get("stationary_task")
    return bool(isinstance(spec, Mapping) and spec.get("schema_version") == "igris_c_nepes_stationary_task_v2"
                and type(spec.get("number")) is int and spec["number"] == 1
                and spec.get("id") == "crs_pickup_load_jig" and spec.get("goal") == "pickup_load_jig")


TRANSFER_TRANSIT_ARRIVAL_KEYS = ("position_m", "rotation_rad", "final_waypoints")
TRANSFER_TRANSIT_ARRIVAL_MODES = ("object", "hand_path")
TRANSFER_TRANSIT_DEVIATION_DEFAULT_M = .06
TRANSFER_TRANSIT_DEVIATION_BOUNDS_M = (.01, .1)
TRANSFER_TRANSIT_ARRIVAL_POSITION_MAXIMUM_M = .03
TRANSFER_TRANSIT_ARRIVAL_ROTATION_MAXIMUM_RAD = .3
# Opt-in (runtime_v26) strict dwell of intermediate lift/transfer/retreat waypoints, in [minimum, reach_dwell_s];
# the review raised the floor to 0.04 s (0.02 s let the restart cut 0.3-1.0 mm more off the sharp strict corners).
INTERMEDIATE_DWELL_KEY = "object_waypoint_intermediate_dwell_s"
INTERMEDIATE_DWELL_MINIMUM_S = .04
# Opt-in (runtime_v26) per-segment posture path caps, each in (0, the path's max_speed/max_acceleration].
POSTURE_SEGMENT_LIMIT_KEYS = {"segment_max_speed_rad_s": "max_speed_rad_s",
                              "segment_max_acceleration_rad_s2": "max_acceleration_rad_s2"}


def _transfer_entry_posture(task: Mapping[str, Any]):
    """The command the transfer starts from, when the profile determines it exactly; else None (no check).

    place_posture directly precedes transfer in every upright pick/load sequence, so an explicit place_posture
    path gives the entry command exactly (its last waypoint). With a hold_last_applied (or absent) place_posture
    the carry_posture path end is used only when nothing between carry_posture and transfer can write joints
    (no handover stages, no single step, no carry stabilizer, no navigate_target/place_posture stage posture).
    Malformed containers return None so the schema validators report them with their usual ValueError."""
    try:
        paths = task.get("posture_paths", {})
        if not isinstance(paths, Mapping):
            return None

        def path_end(stage):
            path = paths.get(stage)
            if not isinstance(path, Mapping) or path.get("reference_mode") == "hold_last_applied":
                return None
            names, waypoints = path.get("joint_names"), path.get("waypoints")
            if (not isinstance(names, list) or not isinstance(waypoints, list) or not waypoints
                    or not isinstance(waypoints[-1], list) or len(waypoints[-1]) != len(names)
                    or any(not isinstance(name, str) for name in names)
                    or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                           for value in waypoints[-1])):
                return None
            return {name: float(value) for name, value in zip(names, waypoints[-1])}

        place = paths.get("place_posture")
        if isinstance(place, Mapping) and place.get("reference_mode") != "hold_last_applied":
            return path_end("place_posture")
        if place is not None and not isinstance(place, Mapping):
            return None
        navigation = task.get("navigation_control", {})
        stage_postures = task.get("stage_postures", {})
        if (not isinstance(navigation, Mapping) or not isinstance(stage_postures, Mapping) or "handover" in task
                or "single_step" in navigation or navigation.get("stabilize_carry")
                or {"navigate_target", "place_posture"} & set(stage_postures)):
            return None
        return path_end("carry_posture")
    except (TypeError, AttributeError, ValueError):
        return None


def validate_transfer_transit_arrival(settings: Any, thresholds: Mapping[str, Any], transfer_waypoints: int) -> dict:
    """Looser arrival for the intermediate transfer waypoints of a carried, symmetric payload.

    A ring that pivots inside the pads moves its centre by up to pivot_radius x spin relative to the
    hand (run_57/run_58: 14-24 mm); the bounded object feedback then cannot bring the centre within the
    3 mm tool threshold, so the transfer stalls although the hand tracks its commanded target within 1-2
    mm. ``position_m``/``rotation_rad`` replace the tcp thresholds for every transfer waypoint except the
    last ``final_waypoints`` ones, which keep the strict thresholds (and the aligned-descent gates) for
    the insertion approach. Capture, target and retention thresholds are unchanged.

    ``mode`` (optional, default ``"object"``): with ``"hand_path"`` the intermediate transfer waypoints are
    carried by the HAND along the planned hand path (TCP targets from the original acquisition transform,
    arrival judged on the tool within position_m/rotation_rad, object feedback suppressed), because a
    rim-pivoting ring moves its centre by pivot_radius x spin inside the pads and dragging the centre back
    onto the object path pushed the arm into its command boxes (run_57-59). The last ``final_waypoints``
    waypoints re-capture the measured grasp (spin-aligned) and resume strict object-centred control.
    """
    if (not isinstance(settings, Mapping) or not set(TRANSFER_TRANSIT_ARRIVAL_KEYS) <= set(settings)
            or set(settings) - set(TRANSFER_TRANSIT_ARRIVAL_KEYS) - {"mode", "max_object_deviation_m"}):
        raise ValueError("transfer_transit_arrival requires exactly " + ", ".join(TRANSFER_TRANSIT_ARRIVAL_KEYS)
                         + " and optionally mode and max_object_deviation_m")
    mode = settings.get("mode", "object")
    if mode not in TRANSFER_TRANSIT_ARRIVAL_MODES:
        raise ValueError("transfer_transit_arrival.mode must be one of " + ", ".join(TRANSFER_TRANSIT_ARRIVAL_MODES))
    deviation = settings.get("max_object_deviation_m")
    if deviation is not None and mode != "hand_path":
        raise ValueError("transfer_transit_arrival.max_object_deviation_m applies only to mode hand_path")
    if mode == "hand_path":
        deviation = TRANSFER_TRANSIT_DEVIATION_DEFAULT_M if deviation is None else deviation
        low, high = TRANSFER_TRANSIT_DEVIATION_BOUNDS_M
        if (isinstance(deviation, bool) or not isinstance(deviation, (int, float)) or not math.isfinite(deviation)
                or not low <= deviation <= high):
            raise ValueError(f"transfer_transit_arrival.max_object_deviation_m must be within [{low:g}, {high:g}] m")
    strict_position = float(thresholds.get("tcp_position_m", .003))
    strict_rotation = float(thresholds.get("tcp_rotation_rad", .05))
    position, rotation, final = settings["position_m"], settings["rotation_rad"], settings["final_waypoints"]
    for key, value in (("position_m", position), ("rotation_rad", rotation)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"transfer_transit_arrival.{key} must be a finite number")
    if not strict_position < position <= TRANSFER_TRANSIT_ARRIVAL_POSITION_MAXIMUM_M:
        raise ValueError(f"transfer_transit_arrival.position_m must exceed the tcp threshold {strict_position:g} and be at most "
                         f"{TRANSFER_TRANSIT_ARRIVAL_POSITION_MAXIMUM_M:g} m")
    if not strict_rotation <= rotation <= TRANSFER_TRANSIT_ARRIVAL_ROTATION_MAXIMUM_RAD:
        raise ValueError(f"transfer_transit_arrival.rotation_rad must be within [{strict_rotation:g}, "
                         f"{TRANSFER_TRANSIT_ARRIVAL_ROTATION_MAXIMUM_RAD:g}] rad")
    if isinstance(final, bool) or not isinstance(final, int) or final < 1 or final >= transfer_waypoints:
        raise ValueError("transfer_transit_arrival.final_waypoints must be an integer in [1, transfer waypoint count)")
    result = {"position_m": float(position), "rotation_rad": float(rotation), "final_waypoints": int(final), "mode": mode}
    if mode == "hand_path":
        result["max_object_deviation_m"] = float(deviation)
    return result


APPROACH_AXIS_ALIGNMENT_KEYS = frozenset(("axis_object", "position_m"))
# Opt-in refinements: the gated TCP-frame point (absent: the TCP origin) and a pregrasp completion bound.
APPROACH_AXIS_ALIGNMENT_OPTIONAL_KEYS = frozenset(("point_tcp_m", "pregrasp_position_m"))
APPROACH_AXIS_CONDITION = ("; approach_axis_alignment: measured TCP error along the declared object axis "
                           "within position_m")
APPROACH_AXIS_POINT_CONDITION = " (error of the declared TCP point point_tcp_m)"
PREGRASP_AXIS_CONDITION = ("; approach_axis_alignment.pregrasp_position_m: measured TCP error from the final "
                           "pregrasp pose along the declared object axis within pregrasp_position_m")
GUARDED_APPROACH_COMPLETION_CONDITION = ("; guarded_approach.completion_position_m: measured TCP position error "
                                         "within completion_position_m (instead of lateral_tolerance_m)")
GUARDED_APPROACH_KEYS = frozenset(("max_speed_m_s", "max_target_lead_m", "lateral_tolerance_m",
                                   "rotation_tolerance_rad", "expected_contact_zone_m", "unexpected_contact_force_n",
                                   "max_acceleration_m_s2", "guard_point_tcp_m", "completion_position_m",
                                   # Opt-in (runtime_v26): pressed_key_completion.py and hinge_controller's zones.
                                   "pressed_key_completion", "lateral_tolerance_zones"))
PRESSED_KEY_COMPLETION_CONDITION = ("; guarded_approach.pressed_key_completion: measured TCP error to the grasp "
                                    "goal within max_axis_error_m along finger_axis_tcp and within "
                                    "max_lateral_error_m across it, with the other approach gates, continuously for "
                                    "max(thresholds.reach_dwell_s, dwell_s) (instead of lateral_tolerance_m)")
PRESSED_KEY_FORCE_CONDITION = ("; mean fixed-pad tip force over the trailing dwell_s window at least "
                               "min_mean_tip_force_n")
TCP_POINT_MAXIMUM_M = .3


def _tcp_point(value: Any, name: str) -> list[float]:
    """A TCP-frame point (e.g. the fixed pad tip): three strict finite numbers, each within 0.3 m."""
    if isinstance(value, (str, bytes, Mapping)) or not hasattr(value, "__len__") or len(value) != 3:
        raise ValueError(f"{name} must have 3 finite values")
    point = [_finite(v, name) for v in value]
    if any(abs(v) > TCP_POINT_MAXIMUM_M for v in point):
        raise ValueError(f"{name} components must be within {TCP_POINT_MAXIMUM_M:g} m")
    return point


def validate_approach_axis_alignment(settings: Any, thresholds: Mapping[str, Any],
                                     name: str = "approach_axis_alignment") -> dict:
    """Opt-in approach arrival along one object axis (the pinch/closing axis); returns the axis normalized.

    The approach completes on the TCP error norm, but a 1-DOF jaw closing along the ring axis pushes the ring
    by the error component along that axis (run_103: 1.9 of its 2.3 mm; the jaw pushed the ring 3.1 mm, tilted
    it 1.1 deg in the V-slot and it jammed in the lift). ``axis_object`` (object frame, length 0.5..2) is
    normalized; ``position_m`` (0 < p <= thresholds.tcp_position_m) bounds the TCP error along it in addition.
    Like the guarded_approach bounds it stays out of the motion watchdog's pose-goal test (its 2 mm default
    progress step cannot see sub-mm convergence); stage_timeouts_s.approach bounds a wait on it alone.
    Optional: ``point_tcp_m`` (TCP frame, |component| <= 0.3 m) gates that point instead of the TCP origin,
    e = ((T_tcp @ p) - (T_target @ p)) . a_w: 1 deg of hand tilt moves the pad tip 156.5 mm out by 2.7 mm.
    ``pregrasp_position_m`` (0 < p <= thresholds.tcp_position_m) also gates the pregrasp completion on the same
    error from the final pregrasp pose (the grasp target shifted by pregrasp_offset_w, never a pregrasp
    waypoint), published as command.pregrasp_axis_alignment (the approach evidence's shape, None outside a
    running pregrasp); stage_timeouts_s.pregrasp bounds that wait. Absent, the v24 gate and evidence are unchanged.
    """
    if (not isinstance(settings, Mapping) or not APPROACH_AXIS_ALIGNMENT_KEYS <= set(settings)
            or not set(settings) <= APPROACH_AXIS_ALIGNMENT_KEYS | APPROACH_AXIS_ALIGNMENT_OPTIONAL_KEYS):
        raise ValueError(f"{name} requires exactly axis_object and position_m "
                         "(optional: point_tcp_m, pregrasp_position_m)")
    value = settings["axis_object"]
    if isinstance(value, (str, bytes, Mapping)) or not hasattr(value, "__len__") or len(value) != 3:
        raise ValueError(f"{name}.axis_object must have 3 finite values")
    axis = np.array([_finite(v, f"{name}.axis_object") for v in value])
    length = float(np.linalg.norm(axis))
    if not .5 <= length <= 2.:
        raise ValueError(f"{name}.axis_object must have a length within [0.5, 2] before normalization")
    position = _finite(settings["position_m"], f"{name}.position_m")
    if not 0. < position <= float(thresholds["tcp_position_m"]):
        raise ValueError(f"{name}.position_m must be positive and at most thresholds.tcp_position_m")
    # An already unit axis is kept bit for bit: re-validating a validated profile keeps its digest.
    unit = axis if abs(length - 1.) <= 1e-12 else axis / length
    result = {"axis_object": [float(v) for v in unit], "position_m": position}
    if "point_tcp_m" in settings:
        result["point_tcp_m"] = _tcp_point(settings["point_tcp_m"], f"{name}.point_tcp_m")
    if "pregrasp_position_m" in settings:
        pregrasp = _finite(settings["pregrasp_position_m"], f"{name}.pregrasp_position_m")
        if not 0. < pregrasp <= float(thresholds["tcp_position_m"]):
            raise ValueError(f"{name}.pregrasp_position_m must be positive and at most thresholds.tcp_position_m")
        result["pregrasp_position_m"] = pregrasp
    return result


def approach_axis_error(settings: Mapping[str, Any], tcp: np.ndarray, target: np.ndarray,
                        object_pose: np.ndarray, *, stage: str = "approach") -> dict[str, Any]:
    """approach_axis_alignment evidence: the measured TCP error from the stage target along the declared
    object axis, turned into the world by the object pose that produced the target. With point_tcp_m the
    error is that TCP-frame point's (also published); stage "pregrasp" bounds it by pregrasp_position_m."""
    axis = object_pose[:3, :3] @ np.asarray(settings["axis_object"], dtype=float)
    if "point_tcp_m" in settings:
        point = np.array([*settings["point_tcp_m"], 1.])
        error = float(np.dot((tcp @ point)[:3] - (target @ point)[:3], axis))
    else:
        error = float(np.dot(tcp[:3, 3] - target[:3, 3], axis))
    bound = settings["pregrasp_position_m" if stage == "pregrasp" else "position_m"]
    evidence = {"axis_w": [float(v) for v in axis], "error_m": error, "position_m": float(bound),
                "satisfied": abs(error) <= bound}
    if "point_tcp_m" in settings:
        evidence["point_tcp_m"] = [float(v) for v in settings["point_tcp_m"]]
    return evidence


def pick_gate_conditions(task: Mapping[str, Any]) -> dict[str, str]:
    """Graph success_condition suffixes of the declared opt-in pregrasp/approach gates (none declared: {})."""
    suffixes = {}
    axis = task.get("approach_axis_alignment")
    if axis is not None:
        point = APPROACH_AXIS_POINT_CONDITION if "point_tcp_m" in axis else ""
        suffixes["approach"] = APPROACH_AXIS_CONDITION + point
        if "pregrasp_position_m" in axis:
            suffixes["pregrasp"] = PREGRASP_AXIS_CONDITION + point
    if "completion_position_m" in task.get("guarded_approach", {}):
        suffixes["approach"] = suffixes.get("approach", "") + GUARDED_APPROACH_COMPLETION_CONDITION
    pressed = task.get("guarded_approach", {}).get("pressed_key_completion")
    if pressed is not None:  # opt-in (runtime_v26); exclusive with completion_position_m
        suffixes["approach"] = (suffixes.get("approach", "") + PRESSED_KEY_COMPLETION_CONDITION
                                + (PRESSED_KEY_FORCE_CONDITION if "min_mean_tip_force_n" in pressed else ""))
    return suffixes


def _validate_object_pose_control(settings: Any, thresholds: Mapping[str, Any], name: str) -> None:
    if not isinstance(settings, Mapping) or not set(settings) <= _OBJECT_POSE_CONTROL_KEYS:
        raise ValueError(f"{name} has unsupported settings")
    for key, value in settings.items():
        if key == "object_feedback":
            from .crs_manipulation import CrsPoseController
            identity = [0., 0., 0., 1., 0., 0., 0.]
            feedback = CrsPoseController(identity, identity, identity, object_feedback=value).object_feedback
            if feedback is not None and (feedback["position_limit"] >= thresholds["capture_position_m"]
                    or feedback["rotation_limit"] >= thresholds["capture_rotation_rad"]):
                raise ValueError("Object feedback correction must remain below immutable capture limits")
        elif key == "reference_profile":
            if value not in {"quintic", "online"}:
                raise ValueError("Unknown object pose reference profile")
        else:
            _positive(value, f"{name}.{key}")
    if (settings.get("pause_slip_position", .012) >= thresholds["capture_position_m"]
            or settings.get("pause_slip_rotation", .10) >= thresholds["capture_rotation_rad"]):
        raise ValueError("object reference slip pause must precede grasp-loss limits")


def _validate_upright_start(start: Any) -> None:
    from .latest_robot import GRIPPER_CLOSED_TARGETS, GRIPPER_JOINT_NAMES, GRIPPER_OPEN_TARGETS, NECK_JOINT_NAMES
    if not isinstance(start, Mapping) or set(start) != {"robot_root_pose_w", "joint_overrides", "head_gaze"}:
        raise ValueError("upright_start needs exactly robot_root_pose_w, joint_overrides and head_gaze")
    root = _pose(start["robot_root_pose_w"], "upright_start.robot_root_pose_w")
    if abs(np.linalg.norm(np.asarray(start["robot_root_pose_w"], dtype=float)[3:]) - 1.) > 1e-5:
        raise ValueError("upright_start.robot_root_pose_w needs a unit wxyz quaternion")
    # base_link is the torso: an upright start keeps it level at standing height.
    roll = math.atan2(root[2, 1], root[2, 2])
    pitch = math.asin(float(np.clip(-root[2, 0], -1., 1.)))
    low, high = _UPRIGHT_TORSO_HEIGHT_M
    if not low <= root[2, 3] <= high or abs(roll) > .05 or abs(pitch) > .05:
        raise ValueError(f"upright_start.robot_root_pose_w must be a level torso pose at z {low}-{high} m "
                         "(roll/pitch within 0.05 rad; lower starts the soles inside the floor)")
    if type(start["head_gaze"]) is not bool:
        raise ValueError("upright_start.head_gaze must be an explicit boolean")
    overrides = start["joint_overrides"]
    if (not isinstance(overrides, Mapping)
            or not set(overrides) <= {*GRIPPER_JOINT_NAMES, *NECK_JOINT_NAMES}):
        raise ValueError("upright_start.joint_overrides may set only the two jaws and the two neck joints")
    if start["head_gaze"] and set(overrides) & set(NECK_JOINT_NAMES):
        raise ValueError("upright_start.head_gaze owns the neck; joint_overrides must not name it")
    for name, value in overrides.items():
        value = _finite(value, f"upright_start.joint_overrides.{name}")
        if name in GRIPPER_JOINT_NAMES:
            low, high = sorted((GRIPPER_OPEN_TARGETS[name], GRIPPER_CLOSED_TARGETS[name]))
            if not low <= value <= high:
                raise ValueError(f"upright_start jaw override must lie between closed and default-open: {name}")
        elif name == "Joint_Neck_Pitch" and abs(value) > _NECK_PITCH_LIMIT_RAD:
            raise ValueError("upright_start neck pitch exceeds the 0.523598776 rad joint limit")


def _validate_posture_paths(paths: Any, *, max_waypoints: int = 8) -> None:
    """Offline finite/teleoperation-range checks; URDF limits are rechecked with the loaded model."""
    if not isinstance(paths, Mapping) or set(paths) != set(POSTURE_STAGES):
        raise ValueError("posture_paths must define exactly " + ", ".join(POSTURE_STAGES))
    for stage in POSTURE_STAGES:
        _validate_posture_path(stage, paths[stage], max_waypoints=max_waypoints)


def _validate_posture_path(stage: str, path: Any, *, prefix: str | None = None, max_waypoints: int = 8) -> None:
    """One joint-space path (or a carry/place/rest applied hold): exact keys, all 19 posture joints, bounded waypoints.

    Shared by ``posture_paths`` and the opt-in handover's ``handover_posture`` path.
    """
    from .constants import TELEOPERATION_JOINT_NAMES, TELEOPERATION_RANGES
    from .latest_robot import COMMAND_JOINT_NAMES, GRIPPER_JOINT_NAMES
    from .posture_hold import HOLD_PATH_KEYS, HOLD_STAGES, is_applied_hold
    required = {name for name in COMMAND_JOINT_NAMES if name not in GRIPPER_JOINT_NAMES}
    ranges = dict(zip(TELEOPERATION_JOINT_NAMES, TELEOPERATION_RANGES))
    prefix = f"posture_paths.{stage}" if prefix is None else prefix
    hold = is_applied_hold(path)
    expected = HOLD_PATH_KEYS if hold else _POSTURE_PATH_KEYS
    if hold and stage not in HOLD_STAGES:
        raise ValueError("hold_last_applied is allowed only for " + ", ".join(HOLD_STAGES))
    if not isinstance(path, Mapping):
        raise ValueError(f"{prefix} needs exactly " + ", ".join(sorted(expected)))
    keys = set(path)
    bounds = _POSTURE_PATH_BOUNDS
    if "motion_limit_profile" in keys:
        if hold:
            raise ValueError(f"{prefix}.motion_limit_profile is not available for hold_last_applied paths")
        if path["motion_limit_profile"] != "simulation_fast":
            raise ValueError(f"{prefix}.motion_limit_profile must be simulation_fast")
        bounds = {**bounds, **_POSTURE_SIMULATION_FAST_BOUNDS}
        keys.remove("motion_limit_profile")
    blend_keys = {"reference_profile", "max_joint_deviation_rad"}
    if keys & blend_keys:
        if (hold or not blend_keys <= keys or path["reference_profile"] != "bounded_c2"
                or not 0. <= _finite(path["max_joint_deviation_rad"],
                                    f"{prefix}.max_joint_deviation_rad") <= .02):
            raise ValueError(f"{prefix} bounded_c2 requires max_joint_deviation_rad within [0, 0.02] "
                             "and is unavailable for hold_last_applied")
        keys -= blend_keys
    if _POSTURE_SETTLE_SPEED_KEY in keys:
        if path[_POSTURE_SETTLE_SPEED_KEY] not in _POSTURE_SETTLE_SPEED_SOURCES:
            raise ValueError(f"{prefix}.{_POSTURE_SETTLE_SPEED_KEY} must be one of "
                             + ", ".join(_POSTURE_SETTLE_SPEED_SOURCES))
        keys.discard(_POSTURE_SETTLE_SPEED_KEY)
    segment_limits = sorted(keys & set(POSTURE_SEGMENT_LIMIT_KEYS))
    if segment_limits:
        # Opt-in (runtime_v26): per-segment caps below the path's own speed/acceleration limits.
        if hold:
            raise ValueError(f"{prefix}.{segment_limits[0]} is not available for hold_last_applied paths")
        keys -= set(segment_limits)
    if keys != expected:
        # Exactly the runtime_v25 text; the runtime_v26 hint only when a segment cap is declared.
        raise ValueError(f"{prefix} needs exactly " + ", ".join(sorted(expected))
                         + f" (optional: {_POSTURE_SETTLE_SPEED_KEY}, reference_profile/max_joint_deviation_rad, "
                           "motion_limit_profile)"
                         + (" (runtime_v26 opt-in: segment_max_speed_rad_s, segment_max_acceleration_rad_s2)"
                            if segment_limits else ""))
    names = path["joint_names"]
    if (not isinstance(names, list) or not all(isinstance(name, str) for name in names)
            or len(set(names)) != len(names) or set(names) != required):
        raise ValueError(f"{prefix}.joint_names must order all {len(required)} waist, arm, wrist and "
                         "neck commands and exclude both jaws")
    if hold:
        for key in HOLD_PATH_KEYS - {"reference_mode", "joint_names"}:
            low, high = _POSTURE_PATH_BOUNDS[key]
            value = _finite(path[key], f"{prefix}.{key}")
            if (value <= 0 if low is None else value < low) or value > high:
                raise ValueError(f"{prefix}.{key} is outside the existing posture bounds")
        return
    waypoints = path["waypoints"]
    if not isinstance(waypoints, list) or not 1 <= len(waypoints) <= max_waypoints:
        raise ValueError(f"{prefix} needs 1..{max_waypoints} joint waypoints")
    for waypoint in waypoints:
        if not isinstance(waypoint, list) or len(waypoint) != len(names):
            raise ValueError(f"{prefix} waypoints must align with joint_names")
        for name, value in zip(names, waypoint):
            value = _finite(value, f"{prefix}.waypoints.{name}")
            low, high = ranges.get(name, (-math.inf, math.inf))
            if not low <= value <= high or (name == "Joint_Neck_Pitch" and abs(value) > _NECK_PITCH_LIMIT_RAD):
                raise ValueError(f"{prefix} waypoint exceeds the shared command range: {name}")
    durations = path["segment_min_duration_s"]
    if (not isinstance(durations, list) or len(durations) != len(waypoints)
            or any(_finite(value, f"{prefix}.segment_min_duration_s") <= 0 for value in durations)):
        raise ValueError(f"{prefix}.segment_min_duration_s needs one positive duration per waypoint")
    for key, (low, high) in bounds.items():
        value = _finite(path[key], f"{prefix}.{key}")
        if (value <= 0 if low is None else value < low) or value > high:
            raise ValueError(f"{prefix}.{key} must lie in {'(0' if low is None else f'[{low}'}, {high}]")
    for key in segment_limits:
        values, cap = path[key], _finite(path[POSTURE_SEGMENT_LIMIT_KEYS[key]], f"{prefix}.{key}")
        if (not isinstance(values, list) or len(values) != len(waypoints)
                or any(isinstance(value, bool) or not 0. < _finite(value, f"{prefix}.{key}") <= cap
                       for value in values)):
            raise ValueError(f"{prefix}.{key} needs one value per waypoint segment, each positive and at most "
                             f"{POSTURE_SEGMENT_LIMIT_KEYS[key]} = {cap:g}")


def _validate_upright_pickup(document: Mapping[str, Any], task_id: str, task: Mapping[str, Any],
                             thresholds: Mapping[str, Any]) -> None:
    """Upright-start pickup/load: no legacy pre-posing may hide inside the profile.

    The caller has already pinned the sequence to catalog-v2 task 1 (crs_load_jig).
    """
    if task.get("object") != "crs" or task.get("mechanism") == "revolute_hinge":
        raise ValueError("The upright pickup sequence requires the free CRS")
    legacy = [key for key in _LEGACY_INITIALIZATION_KEYS if key in task]
    if task.get("startup_apply_configured_arm") is True:
        legacy.append("startup_apply_configured_arm")
    if "prepared_grasp_initialization" in document:
        legacy.append("prepared_grasp_initialization")
    if legacy:
        raise ValueError("The upright start cannot be mixed with legacy initialization: " + ", ".join(legacy))
    for key in ("carry_recovery_offsets_w", "recover_base_offset_w", "auxiliary_support_arm"):
        if key in task:
            raise ValueError(f"The upright pickup sequence does not support {key}")
    if task.get("requires_completed") or task.get("requires_events") or task.get("requires_door_open"):
        raise ValueError("The upright pickup sequence cannot require preceding tasks, events or door clearance")
    navigation = task.get("navigation_control", {})
    if navigation.get("stabilize_carry") is True and (
            navigation.get("command_frame") != "pelvis"
            or not isinstance(navigation.get("carry_tracking_gate"), Mapping)
            or not isinstance(task.get("transport_constraints"), Mapping)):
        raise ValueError("navigation_control.stabilize_carry must be false unless the upright profile declares "
                         "pelvis-frame commands, a carry_tracking_gate and measured transport_constraints")
    # The recorded side-step carries the grasped CRS: it must brake ahead of the
    # station instead of overshooting and walking back, and it may end only once
    # the measured torso has settled (otherwise the following posture stage would
    # cut a still-moving base command to zero in one control step).
    if navigation.get("settle_on_arrival") is not True:
        raise ValueError("The upright pickup sequence carries the CRS while stepping; "
                         "navigation_control.settle_on_arrival must be true")
    if (navigation.get("predictive_braking") is not True or "navigate_target" not in
            navigation.get("predictive_braking_stages", _PREDICTIVE_BRAKING_DEFAULT_STAGES)):
        raise ValueError("The upright pickup side-step must brake predictively: navigation_control."
                         "predictive_braking must be true and cover navigate_target")
    stations = task.get("base_stations", {})
    if stations.get("source") is None or stations.get("target") is None:
        raise ValueError("The upright pickup sequence requires source and target torso stations")
    if task.get("target", {}).get("frame") != "jig" or task.get("placement_support", {}).get("object") != "jig":
        raise ValueError("The upright pickup sequence requires a JIG slot target and measured JIG support")
    if set(task.get("stage_postures", {})) & set(POSTURE_STAGES):
        raise ValueError("posture_paths are the only owner of the upright posture stages")
    # A stage posture there would move the waist, inactive arm or neck before the
    # recording starts, and the readiness would then certify that pre-pose as upright.
    if set(task.get("stage_postures", {})) & set(_UPRIGHT_PREPARATION_STAGES):
        raise ValueError("The unrecorded upright preparation (prerequisites, navigate_source) cannot "
                         "pre-pose the robot through stage_postures")
    if set(task.get("object_waypoint_offsets_w", {})) != {"lift", "transfer"}:
        raise ValueError("The upright pickup sequence requires explicit lift and transfer object paths")
    final = _vector(task["transfer_offset_w"], "transfer_offset_w")
    if np.any(np.abs(final[:2]) > 1e-9) or not .02 <= final[2] <= .035:
        raise ValueError("The final transfer waypoint must be the validated vertical 20-35 mm pre-place height")
    _validate_upright_start(task.get("upright_start"))
    _validate_posture_paths(task.get("posture_paths"),
                            max_waypoints=256 if "transport_constraints" in task else 8)
    if "object_pose_control_overrides" in task:
        overrides = task["object_pose_control_overrides"]
        if (not isinstance(overrides, Mapping) or not overrides
                or not set(overrides) <= {"lift", "transfer", "place"}
                or not isinstance(task.get("object_pose_control"), Mapping)):
            raise ValueError("object_pose_control_overrides may refine only object-controlled lift, transfer and place")
        for stage, value in overrides.items():
            items = [value] if stage == "place" else value
            if stage != "place" and (not isinstance(items, list)
                                     or len(items) != len(task["object_waypoint_offsets_w"][stage])):
                raise ValueError(f"object_pose_control_overrides.{stage} needs one entry per object waypoint")
            for item in items:
                if not isinstance(item, Mapping):
                    raise ValueError(f"object_pose_control_overrides.{stage} entries must be mappings")
                two_speed = {}
                if stage == "place":
                    # runtime_v26 C4: the optional two-speed descent keys are not CrsPoseController settings.
                    from .place_release_options import split_two_speed_descent
                    item, two_speed = split_two_speed_descent(item)
                _validate_object_pose_control({**task["object_pose_control"], **item}, thresholds,
                                              f"{task_id}.object_pose_control_overrides.{stage}")
                if two_speed:
                    from .place_release_options import validate_two_speed_descent
                    validate_two_speed_descent(two_speed, {**task["object_pose_control"], **item}, task, task_id)
    if "retreat_waypoints_w" in task:
        points = task["retreat_waypoints_w"]
        if not isinstance(points, list) or not 1 <= len(points) <= 8:
            raise ValueError("retreat_waypoints_w needs 1..8 world offsets")
        vectors = [_vector(point, "retreat_waypoints_w") for point in points]
        if not np.allclose(vectors[0], _vector(task["retreat_offset_w"], "retreat_offset_w"), atol=1e-9, rtol=0):
            raise ValueError("The first retreat waypoint must be the validated retreat_offset_w")
        if any(np.linalg.norm(second - first) < 1e-5 for first, second in zip(vectors, vectors[1:])):
            raise ValueError("Retreat waypoints must make nonzero progress")
    if "handover" in task:
        # Opt-in in-place handover to the other arm (handover.py); absent, nothing changes.
        from .handover import validate_handover
        validate_handover(task, thresholds)


def validate_task_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy a complete five-task configuration without Isaac imports."""
    document = deepcopy(dict(config))
    from .camera_capture import validate_camera_config
    validate_camera_config(document)
    if "jig_slots" in document:
        from .jig_slots import collision_settings
        collision_settings(document)
        if ("collision_profile" in document["jig_slots"]
                and document["jig_slots"].get("restore_missing_guide_collisions") is not True):
            raise ValueError("jig_slots.collision_profile requires restored guide collisions")
    if "actuator_delay_profile" in document:
        from .actuator_delay import validate_actuator_delay_profile
        validate_actuator_delay_profile(document["actuator_delay_profile"])
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version {SCHEMA_VERSION}")
    if document.get("calibration_status") not in ("draft_unvalidated", "validated"):
        raise ValueError("calibration_status must be draft_unvalidated or validated")
    if document.get("quaternion_order") != "wxyz" or document.get("units") != "meters":
        raise ValueError("Configuration must use meters and wxyz quaternions")
    if document.get("attachment_mode", "none") != "none":
        raise ValueError("Only physical grasps are supported; attachment_mode must be none")
    if "enable_scene_ccd" in document and type(document["enable_scene_ccd"]) is not bool:
        raise ValueError("enable_scene_ccd must be an explicit boolean")
    if "support_contact_diagnostics" in document and type(document["support_contact_diagnostics"]) is not bool:
        raise ValueError("support_contact_diagnostics must be an explicit boolean")
    if "contact_distance_profile" in document:
        from .contact_distance import PROFILE
        if document["contact_distance_profile"] != PROFILE:
            raise ValueError("Unknown contact_distance_profile")
    if ("floor_support_profile" in document
            and document["floor_support_profile"] != "original_floor_outside_workcell_box"):
        raise ValueError("Unknown floor_support_profile")
    if "gripper_pad_contact_patch" in document:
        from .latest_robot import validate_pad_contact_patch, DEFAULT_GRIPPER_VARIANT
        validate_pad_contact_patch(document.get("gripper_variant", DEFAULT_GRIPPER_VARIANT),
                                   document["gripper_pad_contact_patch"])
    if "gripper_pad_compliant_contact" in document:
        from .latest_robot import validate_pad_compliant_contact, DEFAULT_GRIPPER_VARIANT
        validate_pad_compliant_contact(document.get("gripper_variant", DEFAULT_GRIPPER_VARIANT),
                                       document["gripper_pad_compliant_contact"])
    if "gripper_pad_contact_spheres" in document:
        from .latest_robot import validate_pad_contact_spheres, DEFAULT_GRIPPER_VARIANT
        validate_pad_contact_spheres(document.get("gripper_variant", DEFAULT_GRIPPER_VARIANT),
                                     document["gripper_pad_contact_spheres"])
    if "gripper_variant" in document:
        # Absent means the authored jaws; a declared jaw model is part of the profile digest.
        from .latest_robot import validate_gripper_variant
        validate_gripper_variant(document["gripper_variant"])
    objects = document.get("objects", {})
    for key in ("crs", "jig", "cleaner", "cover"):
        path = objects.get(key)
        if not isinstance(path, str) or not path.startswith("/") or path == "/":
            raise ValueError(f"objects.{key} must be a full prim path")
    for key, path in objects.items():
        if not isinstance(path, str) or not path.startswith("/") or path == "/":
            raise ValueError(f"objects.{key} must be a full prim path")
    if "contact_distance_profile" in document and objects["crs"] != "/World/Scene/Ring_01":
        raise ValueError("contact_distance_profile requires the original Ring_01 object")
    if "support_contact_awake_object" in document:
        spec = document.get("stationary_task", {})
        placement = spec.get("number") == 2 and spec.get("goal") == "load_jig" and spec.get("id") == "crs_load_jig"
        if document["support_contact_awake_object"] != "crs" or not (placement or _upright_pickup_identity(document)):
            raise ValueError("support_contact_awake_object permits only the selected CRS in stationary task2 "
                             "or the upright pickup/load task1")
    thresholds = document.get("thresholds", {})
    for key in (
        "tcp_position_m", "tcp_rotation_rad", "object_position_m", "object_rotation_rad",
        "base_position_m", "base_yaw_rad", "capture_position_m", "capture_rotation_rad",
        "settle_linear_m_s", "settle_angular_rad_s", "settle_dwell_s", "contact_dwell_s",
        "reach_dwell_s", "max_observation_gap_s", "lost_grasp_grace_s",
    ):
        _positive(thresholds.get(key), f"thresholds.{key}")
    tasks = document.get("tasks", {})
    if set(tasks) != set(TASK_IDS):
        raise ValueError("Configuration must contain exactly the five NEPES task IDs")
    if "lid_fit_clearance_profile" in document:
        if (document["lid_fit_clearance_profile"] != "inset_slabs_0p5mm_per_side"
                or tasks["open_cleaner_cover"].get("mechanism") != "free_removable_lid"
                or document.get("hinged_door")):
            raise ValueError("Lid fit clearance requires the explicit free removable lid calibration profile")
    for task_id, task in tasks.items():
        if "stage_sequence" in task and task["stage_sequence"] != UPRIGHT_PICKUP_SEQUENCE:
            raise ValueError(f"{task_id}.stage_sequence must be {UPRIGHT_PICKUP_SEQUENCE}")
        upright = task.get("stage_sequence") == UPRIGHT_PICKUP_SEQUENCE
        if upright and (task_id != "crs_load_jig" or not _upright_pickup_identity(document)):
            raise ValueError("The upright pickup stage sequence is only supported by independent stationary v2 task1")
        if not upright and any(key in task for key in _UPRIGHT_PICKUP_KEYS):
            raise ValueError(", ".join(_UPRIGHT_PICKUP_KEYS) + " require the upright pickup stage sequence")
        if "object_symmetry" in task:
            # Opt-in: the carried object is rotationally symmetric about one body axis (the CRS ring
            # normal). Orientation comparisons then ignore the spin about that axis; absent, nothing changes.
            from .object_symmetry import validate_object_symmetry
            if task.get("object") != "crs" or task.get("mechanism") == "revolute_hinge":
                raise ValueError(f"{task_id}.object_symmetry is only supported for the free CRS object")
            if "handover" in task:
                raise ValueError(f"{task_id}.object_symmetry does not support the opt-in handover")
            task["object_symmetry"] = validate_object_symmetry(task["object_symmetry"])
            if "pivot_radius_m" in task["object_symmetry"]:
                # The pinch point is derived from the configured grasp: the TCP needs an azimuth about the axis.
                try:
                    grasp = _pose(task.get("grasp_pose_object"), f"{task_id}.grasp_pose_object")
                except ValueError:
                    grasp = None  # a malformed grasp is reported by the grasp validation itself
                if grasp is not None:
                    try:
                        pinch_point(grasp, task["object_symmetry"]["axis_object"], task["object_symmetry"]["pivot_radius_m"])
                    except ValueError as error:
                        raise ValueError(f"{task_id}.object_symmetry.pivot_radius_m requires a grasp off the symmetry axis") from error
        if "supported_grip_relief" in task:
            from .supported_grip_relief import validate_supported_grip_relief
            if not (upright and task_id == "crs_load_jig" and _upright_pickup_identity(document)):
                raise ValueError("supported_grip_relief requires independent upright pickup/load task1")
            if (task.get("placement_support", {}).get("object") != "jig"
                    or any(key in task for key in ("support_gripper_preload_rad", "support_settling_hold_s"))):
                raise ValueError("supported_grip_relief requires JIG support without legacy support-hold controls")
            task["supported_grip_relief"] = validate_supported_grip_relief(task["supported_grip_relief"])
        if "placement_alignment" in task:
            from .placement_alignment import validate_placement_alignment
            if not (upright and task_id == "crs_load_jig" and _upright_pickup_identity(document)
                    and task.get("target", {}).get("frame") == "jig"
                    and task.get("placement_support", {}).get("object") == "jig"
                    and "object_pose_control" in task):
                raise ValueError("placement_alignment requires object-controlled upright pickup/load into the JIG")
            task["placement_alignment"] = validate_placement_alignment(
                task["placement_alignment"], document["stationary_task"])
            if ("support_hold_dropout_grace_s" in task["placement_alignment"]
                    and any(key in task for key in ("support_settling_hold_s",
                        "support_gripper_preload_rad", "support_load_control"))):
                raise ValueError("support_hold_dropout_grace_s cannot combine with legacy support-hold controls")
        if "transfer_transit_arrival" in task:
            if not (upright and task_id == "crs_load_jig" and _upright_pickup_identity(document)
                    and task.get("object_pose_control", {}).get("object_feedback") is not None
                    and "transfer" in task.get("object_feedback_stages", ["lift", "recover_carry", "transfer", "place"])
                    and isinstance(task.get("object_waypoint_offsets_w", {}).get("transfer"), list)):
                raise ValueError("transfer_transit_arrival requires the object-feedback upright pickup/load task with "
                                 "transfer waypoints under object feedback")
            task["transfer_transit_arrival"] = validate_transfer_transit_arrival(
                task["transfer_transit_arrival"], thresholds, len(task["object_waypoint_offsets_w"]["transfer"]))
            if (task["transfer_transit_arrival"]["mode"] == "hand_path"
                    and task.get("measured_phase_grasp_reference") is not True):
                # The switch to the final waypoints re-captures the measured grasp; without the flag
                # _set_control_grasp would silently keep the original acquisition.
                raise ValueError("transfer_transit_arrival mode hand_path requires measured_phase_grasp_reference")
        if "place_aligned_descent" in task:
            from .place_aligned_descent import validate_place_aligned_descent
            if not (upright and task_id == "crs_load_jig" and _upright_pickup_identity(document)
                    and task.get("target", {}).get("frame") == "jig"
                    and task.get("placement_support", {}).get("object") == "jig"
                    and task.get("object_pose_control", {}).get("object_feedback") is not None):
                raise ValueError("place_aligned_descent requires object-feedback upright pickup/load into the JIG")
            task["place_aligned_descent"] = validate_place_aligned_descent(
                task["place_aligned_descent"], document["stationary_task"])
        if any(key in task for key in ("transfer_reference_continuity", "lift_reference_continuity")):
            # Opt-in blended pass-through (reference_continuity.py, runtime_v26): transfer_reference_continuity and
            # the MAPPING form of lift_reference_continuity; the v25 string form keeps its own check below.
            from .reference_continuity import validate_task_reference_continuity
            validate_task_reference_continuity(task_id, task, upright_pickup=bool(
                upright and task_id == "crs_load_jig" and _upright_pickup_identity(document)))
        if "lift_reference_continuity" in task and not isinstance(task["lift_reference_continuity"], Mapping):
            control = task.get("object_pose_control")
            waypoints = task.get("object_waypoint_offsets_w")
            overrides = task.get("object_pose_control_overrides", {})
            if (task["lift_reference_continuity"] != "collinear_online"
                    or not upright or task_id != "crs_load_jig"
                    or not isinstance(control, Mapping) or control.get("reference_profile") != "online"
                    or not isinstance(waypoints, Mapping) or not waypoints.get("lift")
                    or not isinstance(overrides, Mapping)):
                raise ValueError("lift_reference_continuity requires collinear_online and online upright lift waypoints")
            for override in overrides.get("lift", []):
                if isinstance(override, Mapping) and override.get("reference_profile", "online") != "online":
                    raise ValueError("lift_reference_continuity requires online lift overrides")
        if "pregrasp_pose_control" in task:
            from .pregrasp import validate_pregrasp_settings
            validate_pregrasp_settings(task["pregrasp_pose_control"], task.get("pregrasp_waypoints", ()))
        elif "pregrasp_waypoints" in task:
            raise ValueError("pregrasp_waypoints require pregrasp_pose_control")
        if "release_clearance_offset_w" in task:
            spec = document.get("stationary_task", {})
            if ((task_id != "crs_load_jig" or spec.get("number") != 2 or spec.get("goal") != "load_jig")
                    and not (upright and task_id == "crs_load_jig" and _upright_pickup_identity(document))):
                raise ValueError("release_clearance_offset_w is only supported by independent stationary task2 "
                                 "and the upright pickup/load task1")
            offset = _vector(task["release_clearance_offset_w"], "release_clearance_offset_w")
            if not .002 <= np.linalg.norm(offset) <= .005:
                raise ValueError("release_clearance_offset_w must have a measured clearance length of 2-5 mm")
        if "retreat_orientation_wxyz" in task:
            spec = document.get("stationary_task", {})
            if task_id != "crs_load_jig" or spec.get("number") != 2 or spec.get("goal") != "load_jig":
                raise ValueError("retreat_orientation_wxyz is only supported by independent stationary task2")
            quaternion = _vector(task["retreat_orientation_wxyz"], "retreat_orientation_wxyz", 4)
            if abs(np.linalg.norm(quaternion) - 1.) > 1e-5:
                raise ValueError("retreat_orientation_wxyz must be a unit wxyz quaternion")
        if "retreat_pose_control" in task:
            from .crs_manipulation import TcpPoseController
            if not isinstance(task["retreat_pose_control"], Mapping):
                raise ValueError("retreat_pose_control must be a mapping")
            identity = [0., 0., 0., 1., 0., 0., 0.]
            TcpPoseController(identity, identity, **task["retreat_pose_control"])
        if "retreat_pose_control_overrides" in task:
            from .crs_manipulation import TcpPoseController
            from .upright_runtime import retreat_pose_settings
            overrides = task["retreat_pose_control_overrides"]
            # runtime_v26 C5: an optional release_separating mapping (the release 'separating' subphase only).
            separating = isinstance(overrides, Mapping) and "release_separating" in overrides
            stages = ({"retreat", "release_separating"} & set(overrides) if separating else {"retreat"})
            if (not isinstance(overrides, Mapping) or not stages or set(overrides) != stages
                    or "retreat_pose_control" not in task
                    or any(not isinstance(overrides[stage], Mapping) or not overrides[stage] for stage in stages)):
                raise ValueError("retreat_pose_control_overrides requires a nonempty retreat mapping "
                                 "and retreat_pose_control"
                                 + (" (or a nonempty release_separating mapping)" if separating else ""))
            if separating:
                from .place_release_options import validate_release_separating
                if not (upright and task_id == "crs_load_jig" and _upright_pickup_identity(document)):
                    raise ValueError("retreat_pose_control_overrides.release_separating requires the independent "
                                     "upright pickup/load task1")
                validate_release_separating(overrides["release_separating"], task, task_id)
            identity = [0., 0., 0., 1., 0., 0., 0.]
            try:
                if "retreat" in overrides:
                    TcpPoseController(identity, identity, **retreat_pose_settings(task, "retreat"))
                if separating:
                    TcpPoseController(identity, identity,
                                      **retreat_pose_settings(task, "release", release_phase="separating"))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid retreat_pose_control_overrides: {error}") from error
        if "lift_rotation_w" in task:
            _pose([0., 0., 0., *_vector(task["lift_rotation_w"], "lift_rotation_w", 4)], "lift_rotation_w")
            if task.get("mechanism") == "revolute_hinge" or "object_pose_control" not in task:
                raise ValueError("Lift rotation requires object-controlled non-hinge manipulation")
        if "object_waypoint_rotations_w" in task:
            rotations = task["object_waypoint_rotations_w"]
            if (not isinstance(rotations, Mapping) or not rotations
                    or not set(rotations) <= set(task.get("object_waypoint_offsets_w", {}))):
                raise ValueError("Object waypoint rotations require matching stage offsets")
        if "object_waypoint_offsets_w" in task:
            paths = task["object_waypoint_offsets_w"]
            if (not isinstance(paths, Mapping) or not paths or not set(paths) <= {"lift", "transfer"}
                    or task.get("mechanism") == "revolute_hinge" or "object_pose_control" not in task):
                raise ValueError("Object waypoint paths require object-controlled lift/transfer stages")
            for stage, offsets in paths.items():
                coupled = task.get("object_waypoint_postures", {})
                upper = task.get("upper_body_ik", {})
                upper_stages = upper.get("stages", ()) if isinstance(upper, Mapping) else ()
                online_coupled = (isinstance(upper_stages, list) and stage in upper_stages
                                  and "transport_constraints" in task)
                maximum = 128 if (isinstance(coupled, Mapping) and stage in coupled) or online_coupled else 16
                if not isinstance(offsets, list) or not 1 <= len(offsets) <= maximum:
                    raise ValueError(f"Object waypoint paths need 1..{maximum} offsets")
                points = [_vector(point, "object_waypoint_offsets_w") for point in offsets]
                rotations = task.get("object_waypoint_rotations_w", {}).get(stage,
                    [[1., 0., 0., 0.]] * len(points))
                if not isinstance(rotations, list) or len(rotations) != len(points):
                    raise ValueError("Object waypoint rotations must match stage offsets")
                transforms = [_pose([*point, *_vector(q, "object waypoint rotation", 4)],
                    "object waypoint") for point, q in zip(points, rotations)]
                if any(all(error < 1e-5 for error in _errors(a, b))
                       for a, b in zip(transforms, transforms[1:])):
                    raise ValueError("Object waypoints must make nonzero pose progress")
                if not np.allclose(points[-1], _vector(task[f"{stage}_offset_w"], stage), atol=1e-9, rtol=0):
                    raise ValueError("Object waypoint paths must finish at the original stage goal")
                final_rotation = task.get("lift_rotation_w", [1., 0., 0., 0.]) if stage == "lift" else [1., 0., 0., 0.]
                if _errors(transforms[-1], _pose([*points[-1], *final_rotation], "stage rotation"))[1] > 1e-7:
                    raise ValueError("Object waypoint rotations must finish at the explicit stage goal")
        if "required_diagnostic_stop_after_waypoint" in task:
            from .upright_runtime import validate_required_diagnostic_waypoint_stop
            validate_required_diagnostic_waypoint_stop(task, None, execute=False)
        if type(task.get("measured_phase_grasp_reference", False)) is not bool:
            raise ValueError("measured_phase_grasp_reference must be boolean")
        if "recover_base_offset_w" in task:
            offset = _vector(task["recover_base_offset_w"], "recover_base_offset_w", 2)
            if (task.get("mechanism") == "revolute_hinge" or np.linalg.norm(offset) < 1e-5
                    or not task.get("navigation_control", {}).get("stabilize_carry", False)):
                raise ValueError("Base recovery needs nonzero XY motion and carry stabilization")
        if "recovery_clearance" in task:
            clearance = task["recovery_clearance"]
            if "recover_base_offset_w" not in task or set(clearance) != {"axis", "maximum_world_m", "object_bounds_local"}:
                raise ValueError("Recovery clearance needs a base recovery and explicit box/plane")
            bounds = np.asarray(clearance["object_bounds_local"], float)
            if (type(clearance["axis"]) is not int or clearance["axis"] not in (0, 1, 2)
                    or not math.isfinite(float(clearance["maximum_world_m"]))
                    or bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0])):
                raise ValueError("Invalid recovery clearance box/plane")
        if "carry_recovery_rotations_w" in task and "carry_recovery_offsets_w" not in task:
            raise ValueError("Recovery rotations require matching translation offsets")
        if "carry_recovery_offsets_w" in task:
            offsets = task["carry_recovery_offsets_w"]
            if (task.get("mechanism") == "revolute_hinge" or not isinstance(offsets, list)
                    or not 1 <= len(offsets) <= 16 or "object_pose_control" not in task):
                raise ValueError("Carry recovery needs 1..16 object-controlled non-hinge waypoints")
            rotations = task.get("carry_recovery_rotations_w", [[1., 0., 0., 0.]] * len(offsets))
            if not isinstance(rotations, list) or len(rotations) != len(offsets):
                raise ValueError("Recovery rotations must match the number of offsets")
            previous, previous_rotation = np.zeros(3), np.eye(4)
            for value, quaternion in zip(offsets, rotations):
                point = _vector(value, "carry_recovery_offsets_w")
                quaternion = _vector(quaternion, "carry_recovery_rotations_w", 4)
                rotation = _pose([0., 0., 0., *quaternion], "carry_recovery_rotations_w")
                if np.linalg.norm(point - previous) < 1e-5 and _errors(rotation, previous_rotation)[1] < 1e-5:
                    raise ValueError("Carry recovery waypoints must make nonzero progress")
                previous, previous_rotation = point, rotation
        if "initial_robot_pose_w" in task:
            _pose(task["initial_robot_pose_w"], f"{task_id}.initial_robot_pose_w")
        if "startup_apply_configured_arm" in task:
            if type(task["startup_apply_configured_arm"]) is not bool:
                raise ValueError("startup_apply_configured_arm must be boolean")
            if task["startup_apply_configured_arm"] and (not task.get("initial_arm_joint_positions")
                    or task.get("initial_arm_ramp_seconds", 0.) != 0.):
                raise ValueError("Startup arm initialization requires explicit joints and no arm ramp")
        if "contact_detail_limit" in task and (type(task["contact_detail_limit"]) is not int
                or not 0 <= task["contact_detail_limit"] <= 256):
            raise ValueError("contact_detail_limit must be an integer in [0, 256]")
        if "guarded_approach" in task:
            settings = task["guarded_approach"]
            if not isinstance(settings, Mapping) or not set(settings) <= GUARDED_APPROACH_KEYS:
                raise ValueError("Unknown guarded_approach settings")
            for key, value in settings.items():
                if key == "guard_point_tcp_m":
                    # Opt-in: the reference pauses on this TCP-frame point's cross-track error (hinge_controller).
                    _tcp_point(value, f"{task_id}.guarded_approach.{key}")
                elif key == "completion_position_m":
                    # Opt-in: the approach completes on the TCP position error within it, not lateral_tolerance_m.
                    bound = _finite(value, f"{task_id}.guarded_approach.{key}")
                    if not 0. < bound <= float(thresholds["tcp_position_m"]):
                        raise ValueError(f"{task_id}.guarded_approach.{key} must be positive and at most "
                                         "thresholds.tcp_position_m")
                elif key == "pressed_key_completion":
                    # Opt-in (runtime_v26): finger-axis/lateral split completion with the pad-tip axial gate.
                    from .pressed_key_completion import validate_pressed_key_completion
                    validate_pressed_key_completion(value, f"{task_id}.guarded_approach.{key}")
                    if "completion_position_m" in settings:
                        raise ValueError(f"{task_id}.guarded_approach.{key} replaces completion_position_m; "
                                         "declare only one of them")
                    axis_gate = task.get("approach_axis_alignment")
                    if not isinstance(axis_gate, Mapping) or "point_tcp_m" not in axis_gate:
                        raise ValueError(f"{task_id}.guarded_approach.{key} requires approach_axis_alignment with "
                                         "point_tcp_m (the fixed-pad tip axial gate)")
                    if "min_mean_tip_force_n" in value and task.get("contact_detail_limit", 0) < 1:
                        raise ValueError(f"{task_id}.guarded_approach.{key}.min_mean_tip_force_n requires "
                                         "contact_detail_limit >= 1 (the fixed-pad contact patches)")
                elif key == "lateral_tolerance_zones":
                    # Opt-in (runtime_v26): wider alignment-pause tolerances far from the grasp goal.
                    from .hinge_controller import validate_lateral_tolerance_zones
                    validate_lateral_tolerance_zones(value, settings.get("lateral_tolerance_m", .003),
                                                     f"{task_id}.guarded_approach.{key}")
                else:
                    _positive(value, f"{task_id}.guarded_approach.{key}")
        if "approach_axis_alignment" in task:
            # Opt-in: the approach also bounds the TCP error along the pinch axis; absent, nothing changes.
            task["approach_axis_alignment"] = validate_approach_axis_alignment(
                task["approach_axis_alignment"], thresholds, f"{task_id}.approach_axis_alignment")
        acquisition_frame = task.get("close_acquisition_frame", "world_tcp")
        if acquisition_frame not in ("world_tcp", "object_relative"):
            raise ValueError(f"{task_id}.close_acquisition_frame must be world_tcp or object_relative")
        for key in ("close_acquisition_position_m", "close_acquisition_rotation_rad"):
            if acquisition_frame == "object_relative" or key in task:
                task[key] = _positive(task.get(key), f"{task_id}.{key}")
        if "maximum_pinch_angle_rad" in task:
            bound = _positive(task["maximum_pinch_angle_rad"], f"{task_id}.maximum_pinch_angle_rad")
            if bound >= 1.4 or not isinstance(task.get("pinch_geometry_evidence"), str) or not task["pinch_geometry_evidence"].strip():
                raise ValueError("Object-specific pinch angle needs geometry evidence and must exclude an open jaw")
        if "hinge_following_path" in task:
            if task.get("mechanism") != "revolute_hinge" or not isinstance(task["hinge_following_path"], Mapping):
                raise ValueError("An angle-indexed follower path is only valid for a hinged door")
            if task["hinge_following_path"].get("feasible") is not True:
                raise ValueError("Incomplete hinge paths cannot be executed")
        if "calibration_hinge_limit_deg" in task:
            limit = _positive(task["calibration_hinge_limit_deg"], f"{task_id}.calibration_hinge_limit_deg")
            if task.get("mechanism") != "revolute_hinge" or limit <= 1:
                raise ValueError("A hinge calibration bound must exceed one degree")
        navigation = task.get("navigation_control", {})
        if not isinstance(navigation, Mapping):
            raise ValueError(f"{task_id}.navigation_control must be a mapping")
        if "predictive_braking" in navigation and type(navigation["predictive_braking"]) is not bool:
            raise ValueError("navigation_control.predictive_braking must be boolean")
        if "predictive_braking_stages" in navigation:
            stages = navigation["predictive_braking_stages"]
            if (navigation.get("predictive_braking") is not True or not isinstance(stages, list)
                    or not stages or not all(s in {"navigate_source", "navigate_target", "recover_base"} for s in stages)):
                raise ValueError("predictive_braking_stages must name enabled navigation stages")
        if "base_motion_feedforward_stages" in task:
            stages = task["base_motion_feedforward_stages"]
            if (navigation.get("compensate_base_motion") is not True or not isinstance(stages, list)
                    or not stages or not all(s in {"lift", "recover_carry", "recover_base", "navigate_target",
                                                  "transfer", "place", "retreat"} for s in stages)):
                raise ValueError("base_motion_feedforward_stages needs enabled compensation and Cartesian manipulation stages")
        if "torso_motion_feedforward_stages" in task:
            # Opt-in: the arm servo receives the measured torso-relative target twist
            # (RelativePoseFeedforward) in these object-controlled stages, so a waist
            # sweep or torso sway under the arm is compensated beyond the proportional
            # servo lag. It never commands the waist or the base itself.
            stages = task["torso_motion_feedforward_stages"]
            upper = task.get("upper_body_ik", {})
            owned = upper.get("stages", []) if isinstance(upper, Mapping) else []
            if (not isinstance(stages, list) or not stages or not all(isinstance(s, str) for s in stages)
                    or len(stages) != len(set(stages)) or not set(stages) <= {"lift", "transfer", "place"}
                    or task.get("stage_sequence") != UPRIGHT_PICKUP_SEQUENCE or "object_pose_control" not in task):
                raise ValueError("torso_motion_feedforward_stages needs distinct object-controlled lift/transfer/place "
                                 "stages of the upright pickup/load sequence")
            if (isinstance(owned, list) and set(stages) & set(owned)) or (
                    navigation.get("compensate_base_motion") is True
                    and set(stages) & set(task.get("base_motion_feedforward_stages", ["navigate_target", "recover_base"]))):
                raise ValueError("torso_motion_feedforward_stages cannot share a stage with upper_body_ik or "
                                 "base_motion_feedforward_stages")
        if "retreat_motion_feedforward" in task:
            enabled = task["retreat_motion_feedforward"]
            if not isinstance(enabled, bool):
                raise ValueError("retreat_motion_feedforward must be a boolean")
            if enabled:
                if (task.get("stage_sequence") != UPRIGHT_PICKUP_SEQUENCE
                        or "retreat_pose_control" not in task
                        or task_id != "crs_load_jig" or not _upright_pickup_identity(document)):
                    raise ValueError("retreat_motion_feedforward requires the upright pickup/load "
                                     "sequence with retreat_pose_control")
                upper = task.get("upper_body_ik", {})
                owned = upper.get("stages", []) if isinstance(upper, Mapping) else []
                if ("retreat" in owned or "retreat" in task.get("torso_motion_feedforward_stages", [])
                        or navigation.get("compensate_base_motion") is True
                        and "retreat" in task.get("base_motion_feedforward_stages", ["navigate_target", "recover_base"])):
                    raise ValueError("retreat_motion_feedforward cannot share retreat with another "
                                     "feedforward or upper_body_ik")
        for key in ("max_speed_m_s", "max_yaw_rate_rad_s", "linear_acceleration_m_s2",
                    "yaw_acceleration_rad_s2", "carry_tcp_speed_m_s", "carry_tcp_yaw_rate_rad_s",
                    "position_gain_s_inv", "yaw_gain_s_inv"):
            if key in navigation:
                _positive(navigation[key], f"{task_id}.navigation_control.{key}")
        for key in ("arrival_linear_speed_m_s", "arrival_yaw_speed_rad_s", "arrival_dwell_s",
                    "station_position_tolerance_m", "station_yaw_tolerance_rad", "recovery_position_tolerance_m",
                    "target_position_tolerance_m", "target_yaw_tolerance_rad"):
            if key in navigation:
                _positive(navigation[key], f"{task_id}.navigation_control.{key}")
        if "recovery_position_tolerance_m" in navigation and "recover_base_offset_w" not in task:
            raise ValueError("recovery_position_tolerance_m requires a base recovery stage")
        for key in ("recovery_position_tolerance_m", "target_position_tolerance_m", "target_yaw_tolerance_rad"):
            if isinstance(navigation.get(key), bool):
                raise ValueError(f"{key} must be a physical tolerance, not a boolean")
        if "settle_on_arrival" in navigation and type(navigation["settle_on_arrival"]) is not bool:
            raise ValueError("settle_on_arrival must be boolean")
        if "command_frame" in navigation and navigation["command_frame"] not in ("torso", "pelvis"):
            raise ValueError(f"{task_id}.navigation_control.command_frame must be 'torso' or 'pelvis'")
        if "turn_in_place" in navigation:
            # Opt-in yaw-only goal handling (a station or route waypoint at the current
            # planar position with a new heading); absent, navigation is unchanged.
            from .task_control import validate_turn_in_place
            validate_turn_in_place(navigation["turn_in_place"])
        if "single_step" in navigation:
            # Opt-in single short loaded step in navigate_target (single_step.py); absent, nothing changes.
            from .single_step import validate_single_step
            validate_single_step(task)
        if "stabilize_carry" in navigation and type(navigation["stabilize_carry"]) is not bool:
            raise ValueError(f"{task_id}.navigation_control.stabilize_carry must be boolean")
        if "carry_tracking_gate" in navigation:
            gate = navigation["carry_tracking_gate"]
            keys = {"stop_position_error_m", "resume_position_error_m",
                    "stop_rotation_error_rad", "resume_rotation_error_rad"}
            if (navigation.get("stabilize_carry") is not True or not isinstance(gate, Mapping)
                    or set(gate) != keys):
                raise ValueError("carry_tracking_gate requires stabilized carry and all stop/resume thresholds")
            values = {key: _positive(gate[key], f"carry_tracking_gate.{key}") for key in keys}
            for suffix in ("position_error_m", "rotation_error_rad"):
                if values[f"resume_{suffix}"] >= values[f"stop_{suffix}"]:
                    raise ValueError("carry_tracking_gate resume thresholds must be smaller than stop thresholds")
        if "transport_constraints" in task:
            if task.get("stage_sequence") != UPRIGHT_PICKUP_SEQUENCE:
                raise ValueError("transport_constraints currently require the upright pickup/load sequence")
            from .payload_transport import validate_transport_constraints
            validate_transport_constraints(task["transport_constraints"])
        if "object_waypoint_postures" in task:
            from .waypoint_postures import validate_object_waypoint_postures
            validate_object_waypoint_postures(task)
        if "upper_body_ik" in task:
            from .upper_body_control import validate_upper_body_ik
            upper = validate_upper_body_ik(task["upper_body_ik"])
            if "arm_command_limits_rad" in upper:
                from .robot_model import arm_joint_names
                if set(upper["arm_command_limits_rad"]) - set(arm_joint_names(task["arm"])):
                    raise ValueError("upper_body_ik.arm_command_limits_rad may name only the task's active arm")
            if "transfer_secondary_arm_schedule" in upper or "transfer_early_command_limits" in upper:
                from .robot_model import arm_joint_names
                from .upper_body_control import WAIST
                from .upper_body_guidance import guidance_arm
                transfer_arm = guidance_arm(task, "transfer")
                inactive = "left" if transfer_arm == "right" else "right"
                transfer_count = len(task.get("object_waypoint_offsets_w", {}).get("transfer") or [])
                schedule = upper.get("transfer_secondary_arm_schedule")
                if schedule is not None:
                    if set(schedule["joint_positions"]) - set(arm_joint_names(inactive)):
                        raise ValueError("upper_body_ik.transfer_secondary_arm_schedule may name only the inactive arm "
                                         "(the arm not servoed in transfer)")
                    if transfer_count < 2 or schedule["to_waypoint"] > transfer_count - 1:
                        raise ValueError("upper_body_ik.transfer_secondary_arm_schedule.to_waypoint must index a transfer waypoint")
                early = upper.get("transfer_early_command_limits")
                if early is not None:
                    if set(early["limits"]) - (set(WAIST) | set(arm_joint_names(transfer_arm))):
                        raise ValueError("upper_body_ik.transfer_early_command_limits may name only waist/transfer active arm")
                    if transfer_count < 2 or early["until_waypoint"] > transfer_count - 1:
                        raise ValueError("upper_body_ik.transfer_early_command_limits.until_waypoint must index a transfer waypoint")
                    entry = _transfer_entry_posture(task)
                    for name, (low, high) in early["limits"].items():
                        if entry is not None and name in entry and not low - 1e-9 <= entry[name] <= high + 1e-9:
                            raise ValueError("upper_body_ik.transfer_early_command_limits excludes the transfer-entry command "
                                             f"{name}={entry[name]:g} (last posture-path waypoint before transfer)")
            if "stage_command_limits_rad" in upper:
                from .upper_body_control import WAIST
                from .upper_body_guidance import guidance_arm
                from .robot_model import arm_joint_names
                for stage, limits in upper["stage_command_limits_rad"].items():
                    allowed = set(WAIST) | set(arm_joint_names(guidance_arm(task, stage)))
                    if set(limits) - allowed:
                        raise ValueError("upper_body_ik.stage_command_limits_rad may name only waist/stage active arm")
            if (task.get("stage_sequence") != UPRIGHT_PICKUP_SEQUENCE
                    or "stationary_pelvis_hold" not in navigation):
                raise ValueError("upper_body_ik requires upright pickup/load with stationary_pelvis_hold")
            # Inspect ownership only after checking the container types. Their
            # full schemas are validated below; malformed optional guidance
            # must not bypass validation with an AttributeError/TypeError here.
            stage_postures = task.get("stage_postures", {})
            guidance = task.get("arm_posture_guidance", {})
            if (not isinstance(stage_postures, Mapping) or not isinstance(guidance, Mapping)
                    or not isinstance(guidance.get("stages", {}), Mapping)):
                raise ValueError("upper_body_ik joint ownership requires mapping stage_postures and arm_posture_guidance.stages")
            owned_stages = set(upper["stages"])
            if owned_stages & {"lift", "transfer", "place"} and (
                    "object_pose_control" not in task or "transport_constraints" not in task
                    or upper.get("arm_command_margin_rad", 0.) <= 0.):
                raise ValueError("upper_body_ik manipulation requires object_pose_control, transport_constraints and a positive arm command margin")
            if (owned_stages & set(stage_postures)
                    or owned_stages & set(task.get("object_waypoint_postures", {}))
                    or owned_stages & set(guidance.get("stages", {}))
                    or task.get("arm_nullspace_gain_s_inv", 0.) != 0.
                    or "auxiliary_support_arm" in task):
                raise ValueError("upper_body_ik must be the sole waist/active-arm controller in its stages")
        if "upper_body_posture_guidance" in task:
            from .upper_body_guidance import validate_upper_body_guidance
            validate_upper_body_guidance(task)
        if "stationary_pelvis_hold" in navigation:
            hold = navigation["stationary_pelvis_hold"]
            keys = {"maximum_position_drift_m", "maximum_yaw_drift_rad"}
            if (task.get("stage_sequence") != UPRIGHT_PICKUP_SEQUENCE
                    or navigation.get("command_frame") != "pelvis"
                    or not isinstance(hold, Mapping) or set(hold) != keys):
                raise ValueError("stationary_pelvis_hold requires upright pickup/load, pelvis commands and drift limits")
            for key in keys:
                _positive(hold[key], f"stationary_pelvis_hold.{key}")
            source, target = (_pose(task["base_stations"][name], name) for name in ("source", "target"))
            yaw = math.atan2(source[1, 0], source[0, 0]) - math.atan2(target[1, 0], target[0, 0])
            # Opt-in single_step (validated above): navigate_target steps to a goal latched from the measured
            # stance, so base_stations.target is unused and need not repeat the source station.
            if "single_step" not in navigation and (np.linalg.norm(source[:2, 3] - target[:2, 3]) > 1e-6
                    or abs(math.atan2(math.sin(yaw), math.cos(yaw))) > 1e-6):
                raise ValueError("stationary_pelvis_hold requires identical source and target planar stations")
        if "no_walk" in navigation:
            # Opt-in single stance: zero base command in every stage (no_walk_stance).
            from .no_walk_stance import validate_no_walk
            validate_no_walk(task)
        if "stationary_feet_hold" in navigation:
            if "stationary_pelvis_hold" not in navigation:
                raise ValueError("stationary_feet_hold requires stationary_pelvis_hold")
            if not config.get("floor_support_profile"):
                raise ValueError("stationary_feet_hold requires floor_support_profile for native ground contacts")
            from .stationary_feet import validate_stationary_feet_settings
            validate_stationary_feet_settings(navigation["stationary_feet_hold"])
        if "compensate_base_motion" in navigation:
            if type(navigation["compensate_base_motion"]) is not bool:
                raise ValueError("compensate_base_motion must be boolean")
            if navigation["compensate_base_motion"] and not navigation.get("stabilize_carry", False):
                raise ValueError("Base motion compensation requires a stabilized carry reference")
        if "contact_servo_speed_scale" in task:
            scale = _positive(task["contact_servo_speed_scale"], f"{task_id}.contact_servo_speed_scale")
            if scale > 1:
                raise ValueError(f"{task_id}.contact_servo_speed_scale must be at most 1")
        if "arm_nullspace_gain_s_inv" in task:
            gain = _positive(task["arm_nullspace_gain_s_inv"], "arm_nullspace_gain_s_inv", zero=True)
            if gain > 1:
                raise ValueError("arm_nullspace_gain_s_inv must be at most 1")
        if "hinge_posture_gain_s_inv" in task:
            gain = _positive(task["hinge_posture_gain_s_inv"], "hinge_posture_gain_s_inv")
            if gain > 2 or "hinge_following_path" not in task:
                raise ValueError("Hinge posture guidance needs a complete path and gain at most 2")
        if "arm_posture_guidance" in task:
            guide = task["arm_posture_guidance"]
            if not isinstance(guide, Mapping) or set(guide) != {"gain_s_inv", "stages"}:
                raise ValueError("Arm posture guidance needs gain_s_inv and stages")
            if _positive(guide["gain_s_inv"], "gain_s_inv") > 2:
                raise ValueError("Arm posture gain must be at most 2")
            if (not isinstance(guide["stages"], Mapping) or not guide["stages"]
                    or not set(guide["stages"]) <= {"lift", "recover_carry", "transfer", "place"}
                    or "object_pose_control" not in task):
                raise ValueError("Arm posture stages require object-pose reference progress")
        if "object_pose_control" in task:
            _validate_object_pose_control(task["object_pose_control"], thresholds, f"{task_id}.object_pose_control")
        if "motion_watchdog" in task:
            settings = task["motion_watchdog"]
            if not isinstance(settings, Mapping) or not set(settings) <= {
                    "duration_s", "position_improvement_m", "rotation_improvement_rad", "max_observation_gap_s"}:
                raise ValueError("Unknown motion_watchdog settings")
            for key, value in settings.items():
                _positive(value, f"{task_id}.motion_watchdog.{key}")
        if "object_feedback_stages" in task:
            stages = task["object_feedback_stages"]
            if (not isinstance(stages, list) or not stages or not all(isinstance(s, str) for s in stages)
                    or len(stages) != len(set(stages))
                    or not set(stages) <= {"lift", "recover_carry", "transfer", "place"}
                    or task.get("object_pose_control", {}).get("object_feedback") is None):
                raise ValueError("Object feedback stages require configured feedback and distinct controlled stages")
        if task.get("arm_solver", "dls") not in {"dls", "bounded_lsq"}:
            raise ValueError("Unknown arm solver")
        if "arm_solver_stages" in task:
            stages = task["arm_solver_stages"]
            if (not isinstance(stages, list) or not stages or not all(isinstance(s, str) for s in stages)
                    or len(stages) != len(set(stages))
                    or not set(stages) <= set(task_stages(task))):
                raise ValueError("Arm solver stages must name distinct task stages")
        if task.get("arm") not in ("left", "right"):
            raise ValueError(f"{task_id}: arm must be left or right")
        if "open_gripper_position_rad" in task:
            from .latest_robot import GRIPPER_CLOSED_TARGETS, GRIPPER_JOINT_NAMES, GRIPPER_OPEN_TARGETS
            from .task_control import gripper_open_position
            name = GRIPPER_JOINT_NAMES[0 if task["arm"] == "left" else 1]
            task["open_gripper_position_rad"] = gripper_open_position(
                {"open_position_rad": GRIPPER_OPEN_TARGETS[name],
                 "closed_position_rad": GRIPPER_CLOSED_TARGETS[name]}, task["open_gripper_position_rad"])
            if not isinstance(task.get("pinch_geometry_evidence"), str) or not task["pinch_geometry_evidence"].strip():
                raise ValueError("A task-specific gripper aperture needs pinch_geometry_evidence")
        if task.get("object") not in objects:
            raise ValueError(f"{task_id}: unknown object alias")
        if "closed_gripper_position_rad" in task:
            from .latest_robot import GRIPPER_CLOSED_TARGETS, GRIPPER_JOINT_NAMES, GRIPPER_OPEN_TARGETS
            from .task_control import gripper_closed_position
            name = GRIPPER_JOINT_NAMES[0 if task["arm"] == "left" else 1]
            value = gripper_closed_position(
                {"open_position_rad": GRIPPER_OPEN_TARGETS[name], "closed_position_rad": GRIPPER_CLOSED_TARGETS[name]},
                task["closed_gripper_position_rad"], task.get("open_gripper_position_rad"))
            if (abs(value - GRIPPER_CLOSED_TARGETS[name]) >= task.get("maximum_pinch_angle_rad", .15)
                    or not isinstance(task.get("pinch_geometry_evidence"), str) or not task["pinch_geometry_evidence"].strip()):
                raise ValueError("closed_gripper_position_rad needs contact geometry evidence and must be within the original pinch bound")
            task["closed_gripper_position_rad"] = value
        _pose(task.get("grasp_pose_object"), f"{task_id}.grasp_pose_object")
        target = task.get("target", {})
        if target.get("frame") not in {"world", *objects}:
            raise ValueError(f"{task_id}: unknown target frame")
        if target.get("frame") == task.get("object"):
            raise ValueError(f"{task_id}: placement target cannot follow the manipulated object")
        if "placement_support" in task:
            from .place_release_options import PLACEMENT_SUPPORT_OPTIONAL, validate_placement_support_options
            support = task["placement_support"]
            expected = {"object", "normal_w", "minimum_load_fraction", "maximum_linear_m_s",
                        "maximum_angular_rad_s", "dwell_s"}
            if (not isinstance(support, Mapping) or not expected <= set(support)
                    or set(support) - expected - PLACEMENT_SUPPORT_OPTIONAL):
                # Exactly the runtime_v25 text unless a runtime_v26 key (latch_dwell_s, filter_s) is declared.
                declared = isinstance(support, Mapping) and bool(set(support) & PLACEMENT_SUPPORT_OPTIONAL)
                unknown = (sorted(map(str, set(support) - expected - PLACEMENT_SUPPORT_OPTIONAL))
                           if isinstance(support, Mapping) else [])
                raise ValueError("placement_support needs an explicit support, normal, load, speeds and dwell"
                                 + (" (optional: latch_dwell_s, filter_s" + ("; unknown: " + ", ".join(unknown)
                                                                             if unknown else "") + ")"
                                    if declared else ""))
            # runtime_v26 C2/C3 opt-in keys: the upright pickup/load task only.
            validate_placement_support_options(support, upright_pickup=bool(
                upright and task_id == "crs_load_jig" and _upright_pickup_identity(document)),
                placement_alignment=task.get("placement_alignment"))
            if support["object"] not in objects or support["object"] == task["object"]:
                raise ValueError("placement_support must name a separate known object")
            normal = _vector(support["normal_w"], "placement_support.normal_w")
            if abs(np.linalg.norm(normal) - 1.) > 1e-6 or normal[2] <= 0:
                raise ValueError("placement_support normal must be a unit vector pointing upwards")
            for key in expected - {"object", "normal_w"}:
                _positive(support[key], "placement_support." + key)
            if not .5 <= support["minimum_load_fraction"] <= 1.:
                raise ValueError("placement_support must carry at least half the measured object weight")
        if "support_settling_hold_s" in task:
            duration = _positive(task["support_settling_hold_s"], "support_settling_hold_s")
            if "placement_support" not in task or duration > 5:
                raise ValueError("Support settling hold needs placement_support and at most 5 seconds")
        if "support_load_control" in task:
            from .support_settling import validate_load_control
            validate_load_control(task["support_load_control"])
            if "support_settling_hold_s" not in task:
                raise ValueError("Support load control requires a bounded support settling hold")
        if "support_gripper_preload_rad" in task:
            value = _positive(task["support_gripper_preload_rad"], "support_gripper_preload_rad")
            if "support_settling_hold_s" not in task or not .005 <= value <= .05:
                raise ValueError("Supported gripper preload needs a bounded hold and 0.005..0.05 rad")
        _pose(target.get("pose"), f"{task_id}.target.pose")
        for key in ("pregrasp_offset_w", "lift_offset_w", "transfer_offset_w", "retreat_offset_w"):
            offset = _vector(task.get(key), f"{task_id}.{key}")
            if float(np.linalg.norm(offset)) < 0.01:
                raise ValueError(f"{task_id}.{key} needs at least 1 cm clearance")
        if task["lift_offset_w"][2] <= thresholds["object_position_m"]:
            raise ValueError(f"{task_id}: lift must rise above the object position tolerance")
        _positive(task.get("min_lift_m"), f"{task_id}.min_lift_m")
        if task["min_lift_m"] > task["lift_offset_w"][2]:
            raise ValueError(f"{task_id}: minimum lift exceeds commanded lift")
        _positive(task.get("timeout_s"), f"{task_id}.timeout_s")
        timeouts = task.get("stage_timeouts_s", {})
        stages = task_stages(task)
        if set(timeouts) != set(stages):
            raise ValueError(f"{task_id}: stage timeouts must cover every stage")
        for key, value in timeouts.items():
            _positive(value, f"{task_id}.stage_timeouts_s.{key}")
        postures = task.get("stage_postures", {})
        if not isinstance(postures, Mapping) or not set(postures) <= set(stages):
            raise ValueError(f"{task_id}.stage_postures has unknown stages")
        for stage, posture in postures.items():
            joints = posture.get("joint_positions", {})
            if (not isinstance(joints, Mapping) or not joints
                    or any(not isinstance(n, str) or not math.isfinite(float(v)) for n, v in joints.items())):
                raise ValueError(f"{task_id}.stage_postures.{stage} needs finite named joints")
            _positive(posture.get("duration_s", 3.), "posture.duration_s")
            _positive(posture.get("tolerance_rad", .03), "posture.tolerance_rad")
            if type(posture.get("feedback", False)) is not bool:
                raise ValueError("posture.feedback must be an explicit boolean")
        if "auxiliary_support_arm" in task:
            from .auxiliary_support import validate_auxiliary_support
            settings = validate_auxiliary_support(task["auxiliary_support_arm"], task["arm"])
            if ("object_pose_control" not in task or task.get("mechanism") == "revolute_hinge"
                    or task.get("measured_phase_grasp_reference") is not True):
                raise ValueError("Auxiliary support requires object-controlled non-hinge lift with measured_phase_grasp_reference")
            suffix = "_" + settings["arm"].title()
            jaw = ("l" if settings["arm"] == "left" else "r") + "_1_dof_gripper_moving"
            for posture in postures.values():
                if any(n.endswith(suffix) or n == jaw for n in posture["joint_positions"]):
                    raise ValueError("Stage posture conflicts with auxiliary support arm or jaw")
            if task["lift_offset_w"][2] <= settings["max_object_lift_m"] + thresholds["object_position_m"]:
                raise ValueError("Auxiliary diagnostic stop must precede the unchanged full lift goal")
            task["auxiliary_support_arm"] = settings
        prerequisites = task.get("requires_completed", [])
        if not isinstance(prerequisites, list) or any(p not in TASK_IDS[:TASK_IDS.index(task_id)] for p in prerequisites):
            raise ValueError(f"{task_id}: task prerequisites must refer to earlier tasks")
        events = task.get("requires_events", [])
        if not isinstance(events, list) or any(not isinstance(e, str) or not e for e in events):
            raise ValueError(f"{task_id}: requires_events must contain nonempty names")
        for station in ("source", "target"):
            value = task.get("base_stations", {}).get(station)
            if value is not None:
                _pose(value, f"{task_id}.base_stations.{station}")
            route = task.get("base_routes", {}).get(station, [])
            if not isinstance(route, list) or (route and value is None):
                raise ValueError(f"{task_id}.base_routes.{station} requires a list and final station")
            for waypoint in route:
                _pose(waypoint, f"{task_id}.base_routes.{station}")
        jitter = _vector(task.get("target_position_jitter_m", [0, 0, 0]), f"{task_id}.target_position_jitter_m")
        if np.any(jitter < 0):
            raise ValueError(f"{task_id}: target jitter must be nonnegative")
        if ("slot_approach" in task.get("transport_constraints", {}).get("stage_rules", {}).get("transfer", {})
                and np.any(jitter != 0)):
            raise ValueError("slot_approach requires zero target_position_jitter_m and the original anchored slot")
        for relation in task.get("required_object_relations", []):
            if relation.get("object") not in objects or relation.get("reference") not in objects or relation["object"] == relation["reference"]:
                raise ValueError(f"{task_id}: object relations need distinct known object aliases")
            _pose(relation.get("pose"), f"{task_id}.object_relation.pose")
            _positive(relation.get("position_tolerance_m"), "relation.position_tolerance_m")
            _positive(relation.get("rotation_tolerance_rad"), "relation.rotation_tolerance_rad")
            validate_relation_rotation(relation)
            if relation.get("check_until_stage", "settle") not in STAGES:
                raise ValueError(f"{task_id}: invalid relation check_until_stage")
        for path in task.get("required_fixture_paths", []):
            if not isinstance(path, str) or path not in objects.values():
                raise ValueError(f"{task_id}: required fixture must be a configured full object prim path")
        if upright:
            _validate_upright_pickup(document, task_id, task, thresholds)
        if "place_pinch_reduction" in task and not (upright and task_id == "crs_load_jig"
                                                    and _upright_pickup_identity(document)):
            raise ValueError(f"{task_id}.place_pinch_reduction requires the independent upright pickup/load task1")
        # runtime_v26 C cross-key bounds (no-op without the opt-in keys).
        from .place_release_options import validate_place_release_options
        validate_place_release_options(task, task_id)
    if INTERMEDIATE_DWELL_KEY in thresholds:
        # Opt-in (runtime_v26): a shorter strict dwell for intermediate lift/transfer/retreat waypoints of the
        # upright pickup/load task only; every other dwell, and every final waypoint, keeps reach_dwell_s.
        crs = tasks.get("crs_load_jig", {})
        if crs.get("stage_sequence") != UPRIGHT_PICKUP_SEQUENCE or not _upright_pickup_identity(document):
            raise ValueError(f"thresholds.{INTERMEDIATE_DWELL_KEY} requires the independent upright pickup/load task1")
        value, reach = thresholds[INTERMEDIATE_DWELL_KEY], float(thresholds["reach_dwell_s"])
        low = INTERMEDIATE_DWELL_MINIMUM_S
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or not low <= value <= reach):
            raise ValueError(f"thresholds.{INTERMEDIATE_DWELL_KEY} must be a finite number in "
                             f"[{low:g}, reach_dwell_s = {reach:g}] s")
    if "wash_complete" not in tasks["jig_extract_cleaner"].get("requires_events", []):
        raise ValueError("JIG extraction must require the wash_complete event")
    if not all(tasks[task].get("requires_door_open") is True for task in ("jig_insert_cleaner", "jig_extract_cleaner")):
        raise ValueError("Insertion and extraction must require measured door clearance")
    mechanism = tasks["open_cleaner_cover"].get("mechanism")
    if mechanism not in ("free_removable_lid", "revolute_hinge"):
        raise ValueError("Door mechanism must be free_removable_lid or explicit revolute_hinge")
    if mechanism == "revolute_hinge":
        hinge = validate_hinge_config(document)
        task = tasks["open_cleaner_cover"]
        if task["target"]["frame"] != "world" or not np.allclose(
                _pose(task["target"]["pose"], "hinge target"),
                _pose(hinge_pose_w(document, math.radians(hinge["open_angle_deg"])), "hinge target"), atol=1e-6):
            raise ValueError("Hinged door target must agree with its anchored open pose")
        if task.get("base_stations", {}).get("target") is not None:
            raise ValueError("Hinged door must not navigate while holding its handle")
        if any(task.get("target_position_jitter_m", [0, 0, 0])):
            raise ValueError("Hinged door target cannot have unconstrained position jitter")
    elif document.get("hinged_door"):
        raise ValueError("A free removable lid must not author a hinge constraint")
    door = document.get("door_clearance", {})
    _vector(door.get("opening_half_extents_m"), "door_clearance.opening_half_extents_m", 2)
    extent = _vector(door.get("cover_half_extents_m"), "door_clearance.cover_half_extents_m")
    if np.any(extent <= 0) or np.any(np.asarray(door["opening_half_extents_m"]) <= 0):
        raise ValueError("Door clearance extents must be positive")
    _positive(door.get("margin_m"), "door_clearance.margin_m", zero=True)
    # Reject non-JSON or nonfinite calibration annotations as well.
    json.dumps(document, allow_nan=False)
    return document


def load_task_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        path = Path(__file__).resolve().parents[1] / "config" / "nepes_tasks_twofinger_hinge.json"
    with Path(path).open(encoding="utf-8") as stream:
        return validate_task_config(json.load(stream))


class NepesTaskMachine:
    """One task attempt. Terminal results are sticky; reset creates a new instance.

    ``step`` consumes a mapping containing ``time_s``, ``object_poses_w`` keyed
    by full prim paths, ``tcp_poses_w`` keyed by arm, measured ``hand_open`` and
    ``hand_closed`` booleans keyed by arm, and ``contacts`` keyed by arm whose
    entries are full object prim paths. ``base_pose_w`` is required when the
    configuration has base stations. Optional ``events`` and ``completed_tasks``
    satisfy explicit prerequisites; ``faults`` or ``unreachable`` terminate the
    attempt. Hand state/contact omissions never imply release or capture.
    """

    def __init__(self, config: Mapping[str, Any], task_id: str, *, seed: int = 0,
                 completed_tasks: Any = ()) -> None:
        self.config = validate_task_config(config)
        if task_id not in TASK_IDS:
            raise ValueError(f"Unknown task {task_id!r}")
        if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.task_id = task_id
        self.task = self.config["tasks"][task_id]
        self._hinged = self.task.get("mechanism") == "revolute_hinge"
        self.stages = task_stages(self.task)
        self.seed = int(seed)
        self.completed_tasks = set(completed_tasks)
        if not self.completed_tasks <= set(TASK_IDS):
            raise ValueError("Unknown completed task IDs")
        self.arm = self.task["arm"]
        self.object_path = self.config["objects"][self.task["object"]]
        # Declared rotational symmetry axis (object frame) or None: object/capture comparisons ignore spin.
        symmetry = self.task.get("object_symmetry")
        self._symmetry_axis = None if symmetry is None else np.asarray(symmetry["axis_object"], dtype=float)
        # Optional pinch-circle radius: capture slip is then the rim's motion at the pads, not the centre's.
        self._symmetry_pivot_radius_m = None if symmetry is None else symmetry.get("pivot_radius_m")
        self.stage_id = STAGES[0]
        self.status = "running"
        self.reason: str | None = None
        self.history: list[dict[str, Any]] = []
        self._stage_index = 0
        self._start_time: float | None = None
        self._stage_start = 0.0
        self._previous_time: float | None = None
        self._previous_object: np.ndarray | None = None
        self._condition_since: float | None = None
        self._lost_grasp_since: float | None = None
        self._source_object: np.ndarray | None = None
        self._target_object: np.ndarray | None = None
        self._grasp_tcp: np.ndarray | None = None
        self._object_to_tcp: np.ndarray | None = None
        self._control_object_to_tcp: np.ndarray | None = None
        # Optional transit arrival/hand-path policy for intermediate transfer waypoints (validated mapping or None).
        self._transit_arrival = self.task.get("transfer_transit_arrival")
        self._supported_release_tcp: np.ndarray | None = None
        self._support_hold = None
        self._support_hold_active = False
        if "support_settling_hold_s" in self.task:
            from .support_settling import SupportSettlingHold
            self._support_hold = SupportSettlingHold(self.task["support_settling_hold_s"],
                                                     self.task.get("support_load_control"))
        self._support_active = False
        self._tcp_target: np.ndarray | None = None
        self._last_observation: dict[str, Any] | None = None
        self._linear_speed: float | None = None
        self._angular_speed: float | None = None
        self._hinge_command_angle: float | None = None
        self._base_route_index = 0
        self._base_braking = False
        self._base_settle_since = None
        self._recovery_origin = self._recovery_goal = None
        self._recovery_index = 0
        self._recovery_base_goal = None
        self._object_path_goals = []
        self._object_path_index = 0
        self._object_path_lengths = np.array([], dtype=float)
        offsets = np.asarray([[0., 0., 0.], *self.task.get("carry_recovery_offsets_w", [])])
        self._recovery_lengths = np.linalg.norm(np.diff(offsets, axis=0), axis=1)
        rotations = [np.eye(4), *[_pose([0., 0., 0., *q], "recovery rotation")
                     for q in self.task.get("carry_recovery_rotations_w", [])]]
        if len(rotations) > 1:
            # This parameter only indexes an arm-posture guide; the Cartesian
            # controller independently enforces its physical linear/angular
            # speed limits. Pure rotations need nonzero parameter length.
            self._recovery_lengths += .1 * np.asarray([_errors(b, a)[1] for a, b in zip(rotations, rotations[1:])])
        self._jitter = np.random.default_rng(self.seed).uniform(
            -np.asarray(self.task.get("target_position_jitter_m", [0, 0, 0])),
            np.asarray(self.task.get("target_position_jitter_m", [0, 0, 0])),
        )
        # Opt-in guarded_approach.pressed_key_completion (runtime_v26); None keeps the approach completion unchanged.
        self._pressed_key = None
        if "pressed_key_completion" in self.task.get("guarded_approach", {}):
            from .pressed_key_completion import PressedKeyCompletion
            self._pressed_key = PressedKeyCompletion(self.task["guarded_approach"]["pressed_key_completion"], self.arm)

    def graph(self) -> dict[str, Any]:
        """Dry-run graph; no simulation observation or success is fabricated."""
        reasons = {
            "prerequisites": "declared task/event prerequisites and measured door clearance",
            "navigate_source": "measured base XY/yaw within source station tolerance",
            "pregrasp": "measured TCP at clearance pose and measured hand open",
            "approach": "measured TCP at grasp pose and measured hand open",
            "close": "measured hand closed and object-specific contacts for contact dwell",
            "lift": "measured object rises with TCP while preserving grasp transform",
            "recover_carry": "each measured object/TCP clearance waypoint is reached with grasp and payload retained before walking",
            "recover_base": "measured base translation at captured yaw with payload held above the source and measured payload clearance",
            "navigate_target": "measured base XY/yaw at target station with grasp retained",
            "transfer": "measured TCP at target clearance pose with grasp retained",
            "place": "measured object and TCP at placement poses with grasp retained",
            "release": "measured hand open and object-specific contacts absent",
            "retreat": "measured TCP clear of released object; no hand contact",
            "settle": "released object inside pose tolerance with measured low speed for dwell",
            "open_hinge": "measured anchored hinge reaches open angle while preserving physical handle grasp",
            "reach_posture": "joint-space reach path complete; measured posture joints within arrival tolerance and "
                             "nearly still, base quiet, no hand contact with the object",
            "carry_posture": "joint-space carry path complete with measured posture arrival and the physical grasp retained",
            "place_posture": "joint-space path to the outside-JIG alignment posture complete with measured arrival "
                             "and the physical grasp retained",
            "rest_posture": "joint-space rest path complete with the hand open, no object contact and the placement retained",
            # Opt-in in-place handover (handover.py); the pick/load machine refines these conditions.
            "handover_posture": "joint-space two-arm pre-handover path complete with the giving hand's grasp retained",
            "handover_pregrasp": "measured receiving TCP at its clearance pose of the measured ring, receiving hand open",
            "handover_approach": "measured receiving TCP at its grasp pose of the measured ring, receiving hand open",
            "handover_close": "receiving same-object bilateral pinch within the acquisition bounds for the contact dwell",
            "handover_rotate": ("two-arm joint path complete with both pinches retained, the ring tilt bounded and the "
                                "receiving grasp point at the top of the rim"),
            "handover_release": "giving jaw measured open and separated while the receiving pinch holds for dwell_s",
            "handover_retreat": "giving hand reaches each retreat waypoint, open and contact-free, receiving grasp retained",
        }
        if self._hinged:
            reasons["settle"] = "released door holds measured open angle and hinge geometry at low speed for dwell"
        for stage, condition in pick_gate_conditions(self.task).items():
            reasons[stage] += condition
        return {
            "schema_version": SCHEMA_VERSION, "task_id": self.task_id, "seed": self.seed,
            "calibration_status": self.config["calibration_status"],
            "calibration_notes": deepcopy(self.config.get("calibration_notes", [])),
            "task": deepcopy(self.task), "object_path": self.object_path,
            "target_position_jitter_m": self._jitter.tolist(),
            "stages": [{"id": stage, "success_condition": reasons[stage],
                        "timeout_s": self.task["stage_timeouts_s"][stage]} for stage in self.stages],
            "edges": [[a, b] for a, b in zip(self.stages, (*self.stages[1:], "succeeded"))],
            "failure_edges": [{"from": stage, "to": "failed", "reasons": [
                "stage_timeout", "task_timeout", "invalid_observation", "target_unreachable",
                "runtime_fault", "grasp_lost", "door_closed_during_task",
            ]} for stage in self.stages],
            "attachment_mode": "none", "collision_free_execution_verified": False,
        }

    def fail(self, reason: str) -> dict[str, Any]:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Failure reason must be a nonempty string")
        if self.status == "running":
            self.status, self.reason = "failed", reason
            self.history.append({"time_s": self._previous_time, "stage_id": self.stage_id,
                                 "status": "failed", "reason": reason})
        return self._command()

    def _read_observation(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        now = _positive(observation.get("time_s"), "time_s", zero=True)
        if self._previous_time is not None and now <= self._previous_time:
            raise ValueError("time_s must increase strictly")
        raw_objects = observation.get("object_poses_w", {})
        required = {self.object_path}
        target_frame = self.task["target"]["frame"]
        if target_frame != "world":
            required.add(self.config["objects"][target_frame])
        if self.task.get("requires_door_open") or self.task_id == "open_cleaner_cover":
            required.update((self.config["objects"]["cleaner"], self.config["objects"]["cover"]))
        required.update(self.task.get("required_fixture_paths", []))
        for relation in self.task.get("required_object_relations", []):
            required.update(self.config["objects"][relation[key]] for key in ("object", "reference"))
        objects = {path: _pose(raw_objects[path], f"object_poses_w[{path}]") for path in required}
        tcp = _pose(observation["tcp_poses_w"][self.arm], "tcp_poses_w")
        opened, closed = observation["hand_open"][self.arm], observation["hand_closed"][self.arm]
        if not isinstance(opened, (bool, np.bool_)) or not isinstance(closed, (bool, np.bool_)):
            raise ValueError("hand_open/hand_closed must be measured booleans")
        if opened and closed:
            raise ValueError("hand cannot be both open and closed")
        # Older measured-state providers did not expose jaw velocity. The live
        # two-finger runner supplies this explicitly; require it only when
        # acquiring the grip so load-induced jaw compliance remains permitted.
        settled = True if "hand_settled" not in observation else observation["hand_settled"][self.arm]
        if not isinstance(settled, (bool, np.bool_)):
            raise ValueError("hand_settled must be a measured boolean")
        raw_contacts = observation["contacts"][self.arm]
        if isinstance(raw_contacts, (str, bytes)) or not isinstance(raw_contacts, (list, tuple, set, frozenset)):
            raise ValueError("contacts must contain full object prim paths")
        if any(not isinstance(p, str) or not p.startswith("/") for p in raw_contacts):
            raise ValueError("contacts must contain full object prim paths")
        sensor_available = None
        if "contact_evidence" in observation:
            evidence = observation["contact_evidence"]
            if not isinstance(evidence, Mapping) or not isinstance(evidence.get(self.arm), Mapping):
                raise ValueError("contact_evidence must contain per-arm sensor evidence")
            item = evidence[self.arm].get(self.object_path, {})
            sensor_available = isinstance(item, Mapping) and item.get("available") is True
        object_velocity = None
        if "object_states" in observation:
            state = observation["object_states"][self.object_path]
            if state.get("velocity_frame") != "world_com_linear_angular":
                raise ValueError("Object velocity must use the measured world COM frame")
            object_velocity = _vector(state["velocity_w"], "object velocity", 6)
        base = None
        if any(value is not None for value in self.task.get("base_stations", {}).values()):
            base = _pose(observation["base_pose_w"], "base_pose_w")
        events = observation.get("events", [])
        completed = observation.get("completed_tasks", [])
        if isinstance(events, str) or any(not isinstance(e, str) for e in events):
            raise ValueError("events must be a collection of names")
        if isinstance(completed, str) or not set(completed) <= set(TASK_IDS):
            raise ValueError("completed_tasks contains unknown IDs")
        self.completed_tasks.update(completed)
        base_velocity = (_vector(observation["base_velocity_w"], "base_velocity_w", 6)
                         if "base_velocity_w" in observation else None)
        return {"time_s": now, "objects": objects, "tcp": tcp, "base": base, "base_velocity": base_velocity,
                "open": bool(opened), "closed": bool(closed), "hand_settled": bool(settled),
                "contacts": set(raw_contacts),
                "contact_sensor_available": sensor_available, "object_velocity": object_velocity,
                "events": set(events)}

    def _door_open(self, observation: dict[str, Any]) -> bool:
        """Conservative XY separation of the lid from the cleaner opening.

        Merely lifting the lid vertically does not permit an insertion under it.
        Rotation-aware lid AABB includes its thickness; this is a prerequisite
        geometry check, not a whole-body or payload collision certificate.
        """
        objects = observation["objects"]
        if self.config["tasks"]["open_cleaner_cover"].get("mechanism") == "revolute_hinge":
            return measure_hinge_state(self.config, matrix_to_pose(objects[self.config["objects"]["cover"]]))["open"]
        cleaner = objects[self.config["objects"]["cleaner"]]
        cover = objects[self.config["objects"]["cover"]]
        relative = np.linalg.inv(cleaner) @ cover
        cfg = self.config["door_clearance"]
        extent = np.abs(relative[:3, :3]) @ np.asarray(cfg["cover_half_extents_m"])
        opening = np.asarray(cfg["opening_half_extents_m"])
        return bool(np.any(np.abs(relative[:2, 3]) > opening + extent[:2] + cfg["margin_m"]))

    def _object_relation_evidence(self, observation: dict[str, Any]) -> list[dict[str, Any]]:
        evidence = []
        for relation in self.task.get("required_object_relations", []):
            until = relation.get("check_until_stage", "settle")
            if until not in self.stages:
                raise ValueError(f"Object relation check_until_stage {until} is absent from this task graph")
            reference = observation["objects"][self.config["objects"][relation["reference"]]]
            object_pose = observation["objects"][self.config["objects"][relation["object"]]]
            metrics = measure_object_relation(
                np.linalg.inv(reference) @ object_pose, _pose(relation["pose"], "object relation"), relation)
            evidence.append({"object": relation["object"], "reference": relation["reference"],
                             "object_path": self.config["objects"][relation["object"]],
                             "reference_path": self.config["objects"][relation["reference"]],
                             "measurement_time_s": observation["time_s"],
                             "check_until_stage": until,
                             "checked": self._stage_index <= self.stages.index(until), **metrics})
        return evidence

    def _bad_object_relations(self, observation: dict[str, Any]) -> list[str]:
        return [f"{item['object']}:{item['reference']}"
                for item in self._object_relation_evidence(observation)
                if item["checked"] and not item["valid"]]

    def _near(self, current: np.ndarray, target: np.ndarray, kind: str = "tcp", tolerance=None) -> bool:
        """Pose arrival against the configured thresholds, or against an explicit (position_m, rotation_rad) pair."""
        axis = getattr(self, "_symmetry_axis", None)
        if axis is not None and kind == "capture":
            # Object pose in the hand frame: spin about the symmetry axis cancels, slip and tilt remain
            # (with a declared pivot radius the slip is the rim's motion at the pinch, not the centre's).
            position, rotation, _ = symmetric_capture_errors(
                current, target, axis, getattr(self, "_symmetry_pivot_radius_m", None))
        elif axis is not None and kind == "object":
            position, rotation, _ = symmetric_pose_errors(current, target, axis)
        else:
            position, rotation = _errors(current, target)
        if tolerance is not None:
            return position <= tolerance[0] and rotation <= tolerance[1]
        tolerance = self.config["thresholds"]
        return position <= tolerance[f"{kind}_position_m"] and rotation <= tolerance[f"{kind}_rotation_rad"]

    def navigation_tolerances(self, station: str) -> tuple[float, float]:
        """Navigation stopping region; never substitutes for object placement accuracy."""
        if station not in {"source", "target", "recovery"}:
            raise ValueError(f"Unknown navigation station: {station}")
        navigation = self.task.get("navigation_control", {})
        position = navigation.get("station_position_tolerance_m", self.config["thresholds"]["base_position_m"])
        yaw = navigation.get("station_yaw_tolerance_rad", self.config["thresholds"]["base_yaw_rad"])
        if station == "recovery":
            position = navigation.get("recovery_position_tolerance_m", position)
        elif station == "target":
            position = navigation.get("target_position_tolerance_m", position)
            yaw = navigation.get("target_yaw_tolerance_rad", yaw)
        return float(position), float(yaw)

    def _base_near(self, observation: dict[str, Any], station: str) -> bool:
        route = [] if station == "recovery" else self.task.get("base_routes", {}).get(station, [])
        goal = self._recovery_base_goal if station == "recovery" else (route[self._base_route_index] if self._base_route_index < len(route)
                else self.task.get("base_stations", {}).get(station))
        if goal is None:
            return True
        current, target = observation["base"], _pose(goal, "base station")
        delta = float(np.linalg.norm(current[:2, 3] - target[:2, 3]))
        yaw_current = math.atan2(current[1, 0], current[0, 0])
        yaw_target = math.atan2(target[1, 0], target[0, 0])
        yaw = abs(math.atan2(math.sin(yaw_current - yaw_target), math.cos(yaw_current - yaw_target)))
        navigation = self.task.get("navigation_control", {})
        position_tolerance, yaw_tolerance = self.navigation_tolerances(station)
        near = delta <= position_tolerance and yaw <= yaw_tolerance
        if navigation.get("settle_on_arrival", False):
            self._base_braking = near
            velocity = observation["base_velocity"]
            slow = (velocity is not None and np.linalg.norm(velocity[:3]) <= navigation.get("arrival_linear_speed_m_s", .03)
                    and abs(velocity[5]) <= navigation.get("arrival_yaw_speed_rad_s", .12))
            if near and slow:
                if self._base_settle_since is None:
                    self._base_settle_since = observation["time_s"]
                near = observation["time_s"] - self._base_settle_since >= navigation.get("arrival_dwell_s", .4)
            else:
                self._base_settle_since = None
                near = False
        if near and self._base_route_index < len(route):
            self._base_route_index += 1
            self._base_braking = False
            self._base_settle_since = None
            self._condition_since = None
            return False
        return near

    def _dwell(self, condition: bool, now: float, duration: float) -> bool:
        if not condition:
            self._condition_since = None
            return False
        if self._condition_since is None:
            self._condition_since = now
        return now - self._condition_since >= duration - 1e-10

    def _grasp_valid(self, observation: dict[str, Any]) -> bool:
        if not observation["closed"] or self.object_path not in observation["contacts"]:
            return False
        if self._object_to_tcp is None:
            return False
        measured = np.linalg.inv(observation["objects"][self.object_path]) @ observation["tcp"]
        return self._near(measured, self._object_to_tcp, "capture")

    def _tracking_goal_near(self, observed, tolerance=None) -> bool:
        """Keep the same tool accuracy while correcting measured compliance.

        Object feedback commands the hand through the current measured grasp.
        Testing its arrival against the obsolete phase-entry hand transform
        would reject an accurately placed object. This changes the control goal
        only; acquisition, object target and retention thresholds stay fixed.
        """
        goal = self._tcp_target
        if (self.task.get("object_pose_control", {}).get("object_feedback") is not None
                and self.stage_id in self.task.get("object_feedback_stages", ["lift", "recover_carry", "transfer", "place"])):
            object_pose = observed["objects"][self.object_path]
            desired_object = goal @ np.linalg.inv(self._control_object_to_tcp)
            axis = getattr(self, "_symmetry_axis", None)
            if axis is not None:
                # The hand must not roll to undo a spin of the symmetric object inside the pads.
                object_pose = spin_aligned_object(object_pose, desired_object[:3, :3], axis)
            goal = desired_object @ np.linalg.inv(object_pose) @ observed["tcp"]
        if tolerance is None:
            return self._near(observed["tcp"], goal)
        return self._near(observed["tcp"], goal, tolerance=tolerance)

    def _placement_supported(self, raw, observed, *, require_stationary=True) -> bool:
        """A measured load transfer can unload a jaw without losing the payload.

        Geometry alone is insufficient: the named support must carry the actual
        object's weight, with bounded velocity and the original capture intact.
        Release, retreat and final settled placement remain separate stages.
        """
        settings = self.task.get("placement_support")
        if self.stage_id != "place" or settings is None or self._object_to_tcp is None:
            return False
        item = raw.get("support_evidence", {}).get(self.object_path, {})
        if (item.get("available") is not True or item.get("object_path") != self.object_path
                or item.get("support_path") != self.config["objects"][settings["object"]]
                or item.get("source") != "physx_filtered_object_support_contacts"):
            return False
        try:
            force = _vector(item["force_w"], "support force")
            linear = _vector(item["linear_velocity_w"], "object velocity")
            angular = _vector(item["angular_velocity_w"], "object angular velocity")
            mass = _positive(item["object_mass_kg"], "object mass")
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        object_pose = observed["objects"][self.object_path]
        capture = np.linalg.inv(object_pose) @ observed["tcp"]
        return bool(np.dot(force, settings["normal_w"]) >= settings["minimum_load_fraction"] * mass * 9.81
                    and (not require_stationary or (np.linalg.norm(linear) <= settings["maximum_linear_m_s"]
                         and np.linalg.norm(angular) <= settings["maximum_angular_rad_s"]))
                    and not observed["open"]
                    and self._near(object_pose, self._target_object, "object")
                    and self._near(capture, self._object_to_tcp, "capture"))

    def _close_pose_valid(self, observation: dict[str, Any]) -> bool:
        """Bound contact compliance during acquisition without changing retention.

        The optional object-relative gate still bounds world TCP displacement
        from the already reached approach target. Once acquired, the existing
        lift entry captures the measured grasp transform exactly once.
        """
        if self.task.get("close_acquisition_frame", "world_tcp") == "world_tcp":
            return self._near(observation["tcp"], self._tcp_target)
        measured = np.linalg.inv(observation["objects"][self.object_path]) @ observation["tcp"]
        nominal = _pose(self.task["grasp_pose_object"], "grasp pose")
        limits = (self.task["close_acquisition_position_m"],
                  self.task["close_acquisition_rotation_rad"])
        return all(position <= limits[0] and rotation <= limits[1]
                   for position, rotation in (_errors(measured, nominal),
                                               _errors(observation["tcp"], self._tcp_target)))

    def _approach_axis_alignment(self, observation: dict[str, Any], stage: str = "approach") -> dict[str, Any]:
        """Opt-in approach_axis_alignment evidence. The approach target is the pregrasp entry's measured
        object pose times the grasp (_source_object @ grasp_pose_object); that pose turns the axis. In
        pregrasp the target is the final pregrasp pose (that grasp target shifted by pregrasp_offset_w)."""
        return approach_axis_error(self.task["approach_axis_alignment"], observation["tcp"], self._tcp_target,
                                   self._source_object, stage=stage)

    def _enter(self, index: int, observation: dict[str, Any]) -> None:
        self._stage_index = index
        self.stage_id = self.stages[index]
        self._stage_start = observation["time_s"]
        self._condition_since = None
        if self._pressed_key is not None and self.stage_id == "approach":
            self._pressed_key.reset()
        self._lost_grasp_since = None
        self._object_path_goals = []
        self._object_path_index = 0
        self._object_path_lengths = np.array([], dtype=float)
        if self.stage_id in ("navigate_source", "navigate_target", "recover_base"):
            self._base_route_index = 0
            self._base_braking = False
            self._base_settle_since = None
        object_pose = observation["objects"][self.object_path]
        if self.stage_id == "pregrasp":
            self._source_object = object_pose.copy()
            self._grasp_tcp = object_pose @ _pose(self.task["grasp_pose_object"], "grasp pose")
            target = _pose(self.task["target"]["pose"], "target pose")
            target[:3, 3] += self._jitter
            if self.task["target"]["frame"] != "world":
                target = observation["objects"][self.config["objects"][self.task["target"]["frame"]]] @ target
            self._target_object = target
            self._tcp_target = _shift(self._grasp_tcp, self.task["pregrasp_offset_w"])
        elif self.stage_id in ("approach", "close"):
            self._tcp_target = self._grasp_tcp.copy()
        elif self.stage_id == "lift":
            self._object_to_tcp = np.linalg.inv(object_pose) @ observation["tcp"]
            self._control_object_to_tcp = self._object_to_tcp.copy()
            self._source_object = object_pose.copy()
            self._tcp_target = _shift(observation["tcp"], self.task["lift_offset_w"])
            if "lift_rotation_w" in self.task:
                goal = _shift(object_pose, self.task["lift_offset_w"])
                rotation = _pose([0., 0., 0., *self.task["lift_rotation_w"]], "lift rotation")
                goal[:3, :3] = rotation[:3, :3] @ object_pose[:3, :3]
                self._tcp_target = goal @ self._control_object_to_tcp
        elif self.stage_id == "open_hinge":
            self._object_to_tcp = np.linalg.inv(object_pose) @ observation["tcp"]
            self._control_object_to_tcp = self._object_to_tcp.copy()
            measured_angle = measure_hinge_state(self.config, matrix_to_pose(object_pose))["angle_rad"]
            self._hinge_command_angle = float(np.clip(measured_angle, *np.radians(self.config["hinged_door"]["limits_deg"])))
            self._tcp_target = observation["tcp"].copy()
        elif self.stage_id == "recover_carry":
            self._set_control_grasp(object_pose, observation["tcp"])
            self._recovery_origin = object_pose.copy()
            self._recovery_index = 0
            self._update_recovery_goal()
        elif self.stage_id == "recover_base":
            goal = observation["base"].copy()
            goal[:2, 3] += self.task["recover_base_offset_w"]
            self._recovery_base_goal = matrix_to_pose(goal)
            self._tcp_target = None
        elif self.stage_id in ("navigate_target", *POSTURE_STAGES):
            # Posture stages are joint-space references; no Cartesian goal may linger.
            self._tcp_target = None
        elif self.stage_id == "transfer":
            if self._transit_hand_path(0):
                # The hand replays the planned hand path from the ORIGINAL acquisition; the pivoted ring is not
                # dragged back onto the object path until the final waypoints re-capture the grasp.
                self._control_object_to_tcp = self._object_to_tcp.copy()
            else:
                self._set_control_grasp(object_pose, observation["tcp"])
            self._tcp_target = _shift(self._target_object @ self._control_object_to_tcp, self.task["transfer_offset_w"])
        elif self.stage_id in ("place", "release"):
            if self.stage_id == "place":
                self._set_control_grasp(object_pose, observation["tcp"])
            self._tcp_target = self._target_object @ self._control_object_to_tcp
            if self.stage_id == "release" and self._supported_release_tcp is not None:
                self._tcp_target = self._supported_release_tcp.copy()
        elif self.stage_id in ("retreat", "settle"):
            release_tcp = (self._supported_release_tcp if self._supported_release_tcp is not None
                           else self._target_object @ self._control_object_to_tcp)
            self._tcp_target = _shift(release_tcp, self.task["retreat_offset_w"])
        offsets = self.task.get("object_waypoint_offsets_w", {}).get(self.stage_id)
        if offsets is not None:
            origin = self._source_object if self.stage_id == "lift" else self._target_object
            self._object_path_goals = [_shift(origin, offset) for offset in offsets]
            rotations = self.task.get("object_waypoint_rotations_w", {}).get(self.stage_id)
            if rotations is not None:
                for goal, quaternion in zip(self._object_path_goals, rotations):
                    delta = _pose([0., 0., 0., *quaternion], "object waypoint rotation")
                    goal[:3, :3] = delta[:3, :3] @ origin[:3, :3]
            poses = [object_pose, *self._object_path_goals]
            self._object_path_lengths = np.asarray([max(position, .1 * rotation) if rotations is not None else position
                for position, rotation in (_errors(a, b) for a, b in zip(poses, poses[1:]))])
            self._tcp_target = self._object_path_goals[0] @ self._control_object_to_tcp
        self.history.append({"time_s": observation["time_s"], "stage_id": self.stage_id, "status": "running"})

    def _update_recovery_goal(self):
        self._recovery_goal = _shift(self._recovery_origin,
                                    self.task["carry_recovery_offsets_w"][self._recovery_index])
        rotations = self.task.get("carry_recovery_rotations_w")
        if rotations is not None:
            delta = _pose([0., 0., 0., *rotations[self._recovery_index]], "recovery rotation")
            self._recovery_goal[:3, :3] = delta[:3, :3] @ self._recovery_origin[:3, :3]
        self._tcp_target = self._recovery_goal @ self._control_object_to_tcp

    def _transit_settings(self, index):
        """The declared transit policy if transfer waypoint ``index`` is an intermediate (non-final) one, else None."""
        settings = self._transit_arrival
        if settings is None:
            return None
        count = len(self.task.get("object_waypoint_offsets_w", {}).get("transfer") or [])
        if count == 0 or index < 0 or count - 1 - index < settings["final_waypoints"]:
            return None
        return settings

    def _transit_hand_path(self, index) -> bool:
        """True when transfer waypoint ``index`` is carried along the hand path (transfer_transit_arrival hand_path)."""
        settings = self._transit_settings(index)
        return settings is not None and settings.get("mode") == "hand_path"

    def _set_control_grasp(self, object_pose, tcp):
        # Bounded compliance can change the current tool/object relationship.
        # Use it for a continuous next-phase reference only; all retention and
        # payload acceptance checks continue using the ORIGINAL acquisition.
        if not self.task.get("measured_phase_grasp_reference", False):
            self._control_object_to_tcp = self._object_to_tcp.copy()
            return
        if self._symmetry_axis is not None:
            # A spin of the symmetric object about its axis must not enter the phase reference, or every
            # later hand goal (desired_object @ control) would roll the hand to undo it: spin the measured
            # object back onto the orientation the original acquisition implies for the current hand.
            object_pose = spin_aligned_object(
                object_pose, (tcp @ np.linalg.inv(self._object_to_tcp))[:3, :3], self._symmetry_axis)
        self._control_object_to_tcp = np.linalg.inv(object_pose) @ tcp

    def _recovery_clear(self, object_pose):
        specification = self.task.get("recovery_clearance")
        if specification is None:
            return True
        import itertools
        low, high = np.asarray(specification["object_bounds_local"], float)
        corners = np.asarray(list(itertools.product(*zip(low, high))))
        transformed = corners @ object_pose[:3, :3].T + object_pose[:3, 3]
        return bool(transformed[:, specification["axis"]].max() <= specification["maximum_world_m"])

    def step(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        if self.status != "running":
            return self._command()
        if observation.get("unreachable") is True:
            return self.fail("target_unreachable")
        faults = observation.get("faults", [])
        if faults:
            return self.fail("runtime_fault:" + ";".join(str(value) for value in faults))
        try:
            observed = self._read_observation(observation)
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            return self.fail(f"invalid_observation:{exc}")
        now = observed["time_s"]
        current_object = observed["objects"][self.object_path]
        dt = 0.0
        if self._previous_time is not None:
            dt = now - self._previous_time
            if dt > self.config["thresholds"]["max_observation_gap_s"]:
                return self.fail("observation_gap")
            distance, angle = _errors(current_object, self._previous_object)
            self._linear_speed, self._angular_speed = distance / dt, angle / dt
        self._previous_time, self._previous_object = now, current_object.copy()
        self._last_observation = observed
        hinge_state = None
        if self._hinged:
            hinge_state = measure_hinge_state(self.config, matrix_to_pose(current_object))
            if not hinge_state["valid"]:
                return self.fail("hinge_constraint_violation")
            if self.stage_id == "open_hinge":
                hinge = self.config["hinged_door"]
                target_angle = math.radians(hinge["open_angle_deg"])
                if "calibration_hinge_limit_deg" in self.task:
                    limit = math.radians(self.task["calibration_hinge_limit_deg"])
                    if hinge_state["angle_rad"] <= -(limit - math.radians(1)):
                        return self.fail("calibration_limit:hinge_opening")
                    target_angle = max(target_angle, -limit)
                self._hinge_command_angle = max(
                    target_angle,
                    hinge_state["angle_rad"] - hinge["max_target_lead_rad"],
                    self._hinge_command_angle - hinge["angular_speed_rad_s"] * dt,
                )
                self._tcp_target = _pose(hinge_pose_w(self.config, self._hinge_command_angle), "hinge arc") @ self._object_to_tcp
        if self._start_time is None:
            self._start_time, self._stage_start = now, now
            self.history.append({"time_s": now, "stage_id": self.stage_id, "status": "running"})
        if now - self._start_time >= self.task["timeout_s"]:
            return self.fail("task_timeout")
        if now - self._stage_start >= self.task["stage_timeouts_s"][self.stage_id]:
            return self.fail(f"stage_timeout:{self.stage_id}")
        if self.stage_id != "prerequisites" and self.task.get("requires_door_open") and not self._door_open(observed):
            return self.fail("door_closed_during_task")
        bad_relations = self._bad_object_relations(observed)
        if self.stage_id != "prerequisites" and bad_relations:
            return self.fail("payload_displaced:" + ",".join(bad_relations))
        self._support_active = self._placement_supported(observation, observed)
        self._support_hold_active = False
        if self.stage_id == "place" and "placement_support" in self.task:
            held_tcp = None
            if self._support_hold is not None:
                loaded = self._placement_supported(observation, observed, require_stationary=False)
                evidence = observation.get("support_evidence", {}).get(self.object_path, {})
                keep_valid = (evidence.get("available") is True
                    and evidence.get("source") == "physx_filtered_object_support_contacts"
                    and evidence.get("object_path") == self.object_path
                    and evidence.get("support_path") == self.config["objects"][self.task["placement_support"]["object"]]
                    and self._near(observed["objects"][self.object_path], self._target_object, "object")
                    and (self._grasp_valid(observed) or loaded))
                held_tcp = self._support_hold.update(now, loaded=loaded, valid=keep_valid,
                    measured_tcp=observed["tcp"], normal_w=self.task["placement_support"]["normal_w"],
                    force_w=evidence.get("force_w"), mass_kg=evidence.get("object_mass_kg"))
                self._support_hold_active = held_tcp is not None
            if self._support_active or self._support_hold_active:
                if held_tcp is not None or self._supported_release_tcp is None:
                    self._supported_release_tcp = (held_tcp if held_tcp is not None else observed["tcp"]).copy()
                self._tcp_target = self._supported_release_tcp.copy()
            else:
                self._supported_release_tcp = None
                self._tcp_target = self._target_object @ self._control_object_to_tcp
        if self.stage_id in _CARRYING:
            if self._grasp_valid(observed) or self._support_active:
                self._lost_grasp_since = None
            elif self._lost_grasp_since is None:
                self._lost_grasp_since = now
            elif now - self._lost_grasp_since >= self.config["thresholds"]["lost_grasp_grace_s"]:
                return self.fail("grasp_lost")
        if self.stage_id in (*_RELEASED, "release") and observed["contact_sensor_available"] is False:
            return self.fail("contact_sensor_unavailable")
        if self.stage_id in _RELEASED and (not observed["open"] or self.object_path in observed["contacts"]):
            return self.fail("release_lost")
        threshold = self.config["thresholds"]
        duration = threshold["reach_dwell_s"]
        condition = False
        if self.stage_id == "prerequisites":
            missing = set(self.task.get("requires_completed", [])) - self.completed_tasks
            events = set(self.task.get("requires_events", [])) - observed["events"]
            door_ok = not self.task.get("requires_door_open") or self._door_open(observed)
            condition = not missing and not events and door_ok and not bad_relations
            self.reason = None if condition else "waiting_prerequisites:" + ",".join(
                sorted(missing | events | ({"door_open"} if not door_ok else set()) |
                       {f"object_relation:{name}" for name in bad_relations}))
        elif self.stage_id == "navigate_source":
            condition = self._base_near(observed, "source")
        elif self.stage_id in ("pregrasp", "approach"):
            condition = observed["open"] and self._near(observed["tcp"], self._tcp_target)
            if self.stage_id == "pregrasp" and "pregrasp_pose_control" in self.task:
                from .pregrasp import pregrasp_contact_failure
                contact = observation.get("contact_evidence", {}).get(self.arm, {}).get(self.object_path, {})
                failure = pregrasp_contact_failure(contact, self.arm, self.object_path)
                if failure:
                    return self.fail("pregrasp:" + failure)
                path = observation.get("pregrasp_reference", {})
                if not isinstance(path, Mapping):
                    return self.fail("pregrasp:invalid_reference_evidence")
                condition = (condition and path.get("status") == "running"
                             and path.get("stage_id") == "pregrasp"
                             and path.get("arm") == self.arm and path.get("object_path") == self.object_path
                             and path.get("waypoints_complete") is True)
            if self.stage_id == "pregrasp" and "pregrasp_position_m" in self.task.get("approach_axis_alignment", {}):
                # Opt-in: the standoff must also hold the (pad-tip) error along the pinch axis.
                condition = condition and self._approach_axis_alignment(observed, "pregrasp")["satisfied"]
            if self.stage_id == "approach" and "guarded_approach" in self.task:
                # The runtime guard must not be skipped by a looser FSM angle
                # threshold on the very step that transitions into close.
                settings = self.task["guarded_approach"]
                distance, angle = _errors(observed["tcp"], self._tcp_target)
                contact = observation.get("contact_evidence", {}).get(self.arm, {}).get(self.object_path, {})
                if self._pressed_key is not None:
                    # Opt-in (runtime_v26): the finger-axis/lateral split replaces the distance bound; every other
                    # gate stays, and the dwell (and mean tip force) are checked below.
                    keyed = self._pressed_key.measure(observed["tcp"], self._tcp_target, contact)
                    condition = (condition and contact.get("available") is True and keyed
                                 and angle <= settings.get("rotation_tolerance_rad", .025))
                    duration = max(duration, self._pressed_key.dwell_s)
                else:
                    # Opt-in completion_position_m replaces lateral_tolerance_m here (a pin key may stop the hand short).
                    condition = (condition and contact.get("available") is True
                                 and distance <= settings.get("completion_position_m",
                                                              settings.get("lateral_tolerance_m", .003))
                                 and angle <= settings.get("rotation_tolerance_rad", .025))
            if self.stage_id == "approach" and "approach_axis_alignment" in self.task:
                # The jaw closes along this axis: an error along it would shove the object while pinching.
                condition = condition and self._approach_axis_alignment(observed)["satisfied"]
        elif self.stage_id == "close":
            condition = (observed["closed"] and observed["hand_settled"]
                         and self.object_path in observed["contacts"] and self._close_pose_valid(observed))
            duration = threshold["contact_dwell_s"]
        elif self.stage_id == "lift":
            lifted = current_object[2, 3] - self._source_object[2, 3] >= self.task["min_lift_m"]
            condition = lifted and self._grasp_valid(observed) and self._tracking_goal_near(observed)
        elif self.stage_id == "open_hinge":
            condition = (hinge_state["open"] and
                         abs(hinge_state["angle_rad"] - math.radians(self.config["hinged_door"]["open_angle_deg"]))
                         <= self.config["hinged_door"]["angle_tolerance_rad"] and self._grasp_valid(observed))
        elif self.stage_id == "navigate_target":
            condition = self._base_near(observed, "target") and self._grasp_valid(observed)
        elif self.stage_id == "recover_base":
            condition = (self._base_near(observed, "recovery") and self._grasp_valid(observed)
                         and self._recovery_clear(current_object))
        elif self.stage_id == "recover_carry":
            condition = (self._near(current_object, self._recovery_goal, "object")
                         and self._tracking_goal_near(observed) and self._grasp_valid(observed))
        elif self.stage_id == "transfer":
            condition = self._tracking_goal_near(observed) and self._grasp_valid(observed)
        elif self.stage_id == "place":
            condition = self._near(current_object, self._target_object, "object") and self._tracking_goal_near(observed) and self._grasp_valid(observed)
            if "placement_support" in self.task:
                condition = self._support_active
                duration = self.task["placement_support"]["dwell_s"]
        elif self.stage_id == "release":
            condition = observed["open"] and self.object_path not in observed["contacts"]
            duration = threshold["contact_dwell_s"]
        elif self.stage_id == "retreat":
            condition = self._near(observed["tcp"], self._tcp_target)
        elif self.stage_id == "settle":
            condition = self._near(current_object, self._target_object, "object") and self._linear_speed is not None and self._angular_speed is not None and self._linear_speed <= threshold["settle_linear_m_s"] and self._angular_speed <= threshold["settle_angular_rad_s"]
            if observed["object_velocity"] is not None:
                condition = (condition and np.linalg.norm(observed["object_velocity"][:3]) <= threshold["settle_linear_m_s"]
                             and np.linalg.norm(observed["object_velocity"][3:]) <= threshold["settle_angular_rad_s"])
            if self.task_id == "open_cleaner_cover":
                condition = condition and self._door_open(observed)
            duration = threshold["settle_dwell_s"]
        if self._object_path_goals:
            # Intermediate points require measured payload arrival, not only a
            # commanded tool pose. Final lift still requires the original rise.
            condition = (self._near(current_object, self._object_path_goals[self._object_path_index], "object")
                         and self._tracking_goal_near(observed) and self._grasp_valid(observed))
            if self.stage_id == "lift" and self._object_path_index == len(self._object_path_goals) - 1:
                condition = condition and current_object[2, 3] - self._source_object[2, 3] >= self.task["min_lift_m"]
        if "auxiliary_support_arm" in self.task and self.stage_id == "pregrasp":
            # Additional measured secondary readiness never overrides the primary gate.
            extra = observation.get("auxiliary_support", {})
            condition = (condition and extra.get("precontact_ready") is True
                         and extra.get("status") == "running"
                         and extra.get("stage_id") == "pregrasp")
        posture = self.task.get("stage_postures", {}).get(self.stage_id)
        if posture is not None:
            joints = observation.get("joint_positions", {})
            condition = condition and all(
                n in joints and math.isfinite(float(joints[n]))
                and abs(float(joints[n]) - q) <= posture.get("tolerance_rad", .03)
                for n, q in posture["joint_positions"].items())
        passed = self._dwell(condition, now, duration)
        if self._pressed_key is not None and self.stage_id == "approach" and "guarded_approach" in self.task:
            # Opt-in: the same continuous run, plus the mean tip force over its trailing dwell_s window.
            passed = self._pressed_key.complete(now, condition, passed)
        if passed:
            self.reason = None
            if self._object_path_index + 1 < len(self._object_path_goals):
                self._object_path_index += 1
                self._condition_since = None
                self._tcp_target = self._object_path_goals[self._object_path_index] @ self._control_object_to_tcp
                self.history.append({"time_s": now, "stage_id": self.stage_id,
                    "status": "waypoint_reached", "next_waypoint_index": self._object_path_index})
                return self._command()
            if (self.stage_id == "recover_carry"
                    and self._recovery_index + 1 < len(self._recovery_lengths)):
                self._recovery_index += 1
                self._condition_since = None
                self._update_recovery_goal()
                self.history.append({"time_s": now, "stage_id": self.stage_id,
                    "status": "waypoint_reached", "next_waypoint_index": self._recovery_index})
                return self._command()
            if self._stage_index == len(self.stages) - 1:
                self.status = "succeeded"
                self.history.append({"time_s": now, "stage_id": self.stage_id, "status": self.status})
            else:
                self._enter(self._stage_index + 1, observed)
        return self._command()

    def _command(self) -> dict[str, Any]:
        stage = self.stage_id
        holding = stage in _CARRYING or stage == "close"
        hand = "close" if holding else ("hold" if stage == "prerequisites" else "open")
        mode = "hold"
        if stage in ("navigate_source", "navigate_target", "recover_base"):
            mode = "navigate_base"
        elif stage == "pregrasp":
            mode = "plan"
        elif stage in ("approach", "close", "lift", "recover_carry", "place", "release", "retreat", "open_hinge"):
            mode = "contact_servo"
        elif stage == "transfer":
            mode = "transport"
        elif stage in POSTURE_STAGES:
            mode = "joint_posture"
        base = None
        if stage == "recover_base":
            base = self._recovery_base_goal
        elif stage in ("navigate_source", "navigate_target"):
            station = "source" if stage == "navigate_source" else "target"
            route = self.task.get("base_routes", {}).get(station, [])
            base = (route[self._base_route_index] if self._base_route_index < len(route)
                    else self.task.get("base_stations", {}).get(station))
        if mode == "navigate_base" and base is None:
            # A missing optional station means manipulation starts in place.
            # The stage still advances through measured prerequisites/dwell;
            # never send a null pose to the physical navigation controller.
            mode = "hold"
        if self.status != "running":
            mode, hand, base = "hold", "hold", None
        config_hash = hashlib.sha256(json.dumps(self.config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        command = {
            "task_id": self.task_id, "stage_id": stage, "status": self.status,
            "arm": self.arm, "object_path": self.object_path,
            "tcp_target_pose_w": matrix_to_pose(self._tcp_target) if (self._tcp_target is not None and self.status == "running"
                                                                     and stage not in POSTURE_STAGES) else None,
            "target_object_pose_w": matrix_to_pose(self._target_object) if self._target_object is not None else None,
            "base_goal_pose_w": deepcopy(base), "hand_command": hand, "motion_mode": mode,
            "base_route_waypoint_index": self._base_route_index,
            "object_waypoint_index": self._recovery_index if stage == "recover_carry" else self._object_path_index,
            "object_path_metric": ("max_translation_m_rotation_rad_times_0.1m"
                if stage in self.task.get("object_waypoint_rotations_w", {}) else "translation_m"),
            "object_path_length_m": float(self._object_path_lengths.sum()),
            "object_prior_path_length_m": float(self._object_path_lengths[:self._object_path_index].sum()),
            "object_segment_length_m": float(self._object_path_lengths[self._object_path_index]) if self._object_path_goals else None,
            "control_object_to_tcp": matrix_to_pose(self._control_object_to_tcp) if self._control_object_to_tcp is not None else None,
            "retention_object_to_tcp": matrix_to_pose(self._object_to_tcp) if self._object_to_tcp is not None else None,
            "object_symmetry_axis_object": None if self._symmetry_axis is None else self._symmetry_axis.tolist(),
            "object_symmetry_pivot_radius_m": self._symmetry_pivot_radius_m,
            "placement_support_active": self._support_active,
            "placement_support_hold_active": self._support_hold_active,
            "placement_support_hold_activation": (None if self._support_hold is None else self._support_hold.activation),
            "supported_release_tcp_pose_w": matrix_to_pose(self._supported_release_tcp) if self._supported_release_tcp is not None else None,
            "recovery_object_goal_pose_w": matrix_to_pose(self._recovery_goal) if stage == "recover_carry" else None,
            "recovery_path_length_m": float(self._recovery_lengths.sum()),
            "recovery_path_metric": "translation_m_plus_0.1_m_per_rad" if "carry_recovery_rotations_w" in self.task else "translation_m",
            "recovery_prior_path_length_m": float(self._recovery_lengths[:self._recovery_index].sum()),
            "recovery_segment_length_m": float(self._recovery_lengths[self._recovery_index]) if stage == "recover_carry" else None,
            "base_arrival_braking": self._base_braking if stage in {"navigate_source", "navigate_target", "recover_base"} else False,
            "allowed_contact_paths": [self.object_path] if stage in (*_CARRYING, "approach", "close", "release") else [],
            "reason": self.reason, "seed": self.seed, "config_sha256": config_hash,
            "calibration_status": self.config["calibration_status"],
            "collision_free_execution_verified": False,
            "measured_object_speed_m_s": self._linear_speed,
            "measured_object_angular_speed_rad_s": self._angular_speed,
            "object_relation_evidence": self._object_relation_evidence(self._last_observation)
                if self._last_observation is not None else [],
            "measured_hinge_state": measure_hinge_state(self.config, matrix_to_pose(self._last_observation["objects"][self.object_path]))
                if self._hinged and self._last_observation is not None else None,
        }
        if "approach_axis_alignment" in self.task:
            # Declared only (undeclared profiles keep their command schema): why a running approach waits.
            command["approach_axis_alignment"] = (self._approach_axis_alignment(self._last_observation)
                if stage == "approach" and self.status == "running" and self._last_observation is not None else None)
            if "pregrasp_position_m" in self.task["approach_axis_alignment"]:
                # Declared only: the same evidence in a running pregrasp, bounded by pregrasp_position_m.
                command["pregrasp_axis_alignment"] = (self._approach_axis_alignment(self._last_observation, "pregrasp")
                    if stage == "pregrasp" and self.status == "running" and self._last_observation is not None else None)
        if self._pressed_key is not None:
            # Declared only (runtime_v26): the live evidence of a running approach and, on the step that completed
            # it (already in close), the completion record; None otherwise.
            command["pressed_key_completion"] = self._pressed_key.published(stage, self.status, self._previous_time)
        if mode == "joint_posture":
            command["posture_stage"] = stage
        return command


def dry_run_graph(config: Mapping[str, Any], *, seed: int = 0) -> dict[str, Any]:
    """Return all task graphs and explicit supervisory wash prerequisites."""
    validated = validate_task_config(config)
    return {
        "schema_version": SCHEMA_VERSION, "seed": seed,
        "calibration_status": validated["calibration_status"],
        "task_order": list(TASK_IDS),
        "tasks": [NepesTaskMachine(validated, task, seed=seed).graph() for task in TASK_IDS],
        "supervisor_contract": {
            "wash_complete": "Explicit external or simulated event; never inferred from elapsed task time.",
            "washer_lifecycle": "Adapter owns lid close/start/wash/stop/reopen if required by the washer; extraction independently checks measured lid clearance.",
            "failure": "Stop the attempt and record its cause. No automatic teleport or success flag.",
        },
    }
