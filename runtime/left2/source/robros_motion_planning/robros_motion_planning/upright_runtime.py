"""Isaac-free runtime decisions for the catalog-v2 upright pickup/load task.

task_runtime owns the simulator and every state write. This module decides
commands and provenance from plain values only, so each rule is CPU-testable:
the joint-space posture-stage executor, per-segment object pose control, the
retreat and watchdog segment keys, the upright-start plan and its provenance,
and the diagnostic stop. Nothing here writes robot or object state.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
import re

import numpy as np

from .nepes_tasks import POSTURE_STAGES
# runtime_v26 C4: opt-in keys of object_pose_control_overrides.place (validated by nepes_tasks).
from .place_release_options import TWO_SPEED_DESCENT_KEYS
from .posture_control import JointPathReference
from .posture_hold import (AppliedTargetHoldReference, HOLD_MODE, HOLD_STAGES, HOLD_PATH_KEYS,
                           applied_hold_targets, is_applied_hold)
from .single_step import PRE_STEP_PATH_KEY, SINGLE_STEP_STAGE
from .snapshot import pose_to_matrix

EPISODE_UPRIGHT_START_SCOPE = "episode_boundary_before_physics_settle_and_recording"
STARTUP_UPRIGHT_START_SCOPE = "before_startup_physics_settle"
# ``upright_joint_targets`` is the upper command of the 19 posture-path joints
# right after the write; every unrecorded preparation step must keep exactly it.
UPRIGHT_START_PROVENANCE_KEYS = frozenset(("robot_root_pose_w", "joint_overrides_applied", "head_gaze", "scope",
                                           "demonstration_pose_write", "upright_joint_targets"))
# The NEPES floor is horizontal at world Z=0 (floor_support); the upright spawn
# may leave the lowest collision geometry inside it only by float32 round-off.
FLOOR_HEIGHT_M = 0.
FLOOR_PENETRATION_TOLERANCE_M = 1e-4
# Published every posture control step and injected into the next observation.
POSTURE_REFERENCE_KEYS = frozenset(("stage_id", "status", "elapsed_s", "duration_s", "complete", "segment_index",
                                    "max_tracking_error_rad", "max_tracking_error_joint", "tracking_error_limit_rad",
                                    "final_joint_positions"))
_OBJECT_CONTROL_STAGES = ("lift", "recover_carry", "transfer", "place")
# Joint-space posture stages the PostureStageExecutor may own: the four upright posture
# stages and the opt-in handover's two-arm paths (handover.py): the pre-handover posture and,
# with handover.rotation_path, the two-hand ring rotation.
JOINT_PATH_STAGES = (*POSTURE_STAGES, "handover_posture", "handover_rotate")
# The executor also owns the opt-in navigation_control.single_step pre-step path, which runs in navigate_target
# before the step goal is latched (single_step.py). JOINT_PATH_STAGES itself is unchanged: it also selects the
# observations that carry the posture reference, and a profile without single_step never runs that path.
POSTURE_EXECUTOR_STAGES = (*JOINT_PATH_STAGES, SINGLE_STEP_STAGE)
# ArmServo tolerates this much PhysX overshoot of a measured joint before rejecting it.
_MEASURED_LIMIT_SLACK_RAD = .05
_LIMIT_EPSILON_RAD = 1e-9


def _within(value, limits, slack=_LIMIT_EPSILON_RAD):
    low, high = limits
    return math.isfinite(value) and low - slack <= value <= high + slack


def _model_snapshot(model, positions):
    """Measured joints for URDF validation, clipped like ArmServo; material excursions are rejected."""
    if not isinstance(positions, Mapping):
        raise ValueError("Upright runtime validation needs measured joint positions")
    snapshot = {}
    for name in model.joint_names:
        low, high = model.joint_limits[name]
        if name not in positions:
            raise ValueError(f"Measured snapshot is missing joint {name}")
        value = float(positions[name])
        if not _within(value, (low, high), _MEASURED_LIMIT_SLACK_RAD):
            raise ValueError(f"Measured joint outside physical limits: {name}={value}")
        snapshot[name] = min(max(value, low), high)
    return snapshot


def validate_posture_waypoints(path, *, joint_limits, model=None, positions=None):
    """Every waypoint inside the shared control limits and, with a loaded model, the URDF.

    Only the path joints are replaced; the other measured joints complete the
    snapshot. Returns the waypoints as name -> value mappings.
    """
    names = list(path["joint_names"])
    if is_applied_hold(path):
        if set(path) - {"settle_speed_source"} != HOLD_PATH_KEYS:
            raise ValueError("hold_last_applied cannot also configure a joint path")
        if ("settle_speed_source" in path
                and path["settle_speed_source"] not in ("joint_velocities", "measured_positions")):
            raise ValueError("hold_last_applied settle_speed_source must be joint_velocities or measured_positions")
        # The future last-applied target is not available at startup. Every
        # owned target is checked against the loaded model on actual entry.
        return []
    waypoints = []
    for index, values in enumerate(path["waypoints"]):
        if len(values) != len(names):
            raise ValueError("Posture waypoint must align with joint_names")
        waypoint = {name: float(value) for name, value in zip(names, values)}
        for name, value in waypoint.items():
            if name not in joint_limits or not _within(value, joint_limits[name], 0.):
                raise ValueError(f"Posture waypoint {index} exceeds shared control limits: {name}")
        if model is not None:
            model.validate_snapshot({**_model_snapshot(model, positions), **waypoint})
        waypoints.append(waypoint)
    return waypoints


def lowest_collision_point_z(model, root_pose_w, positions):
    """World height of the robot's lowest collision geometry at a root pose and joint snapshot.

    Meshes contribute their (closed or convex-hull) vertices and primitives or
    AABB fallbacks their box corners, so the value never lies above the geometry.
    """
    from .robot_model import _collision_shapes
    shapes, _ = _collision_shapes(model)
    transforms = model.forward_kinematics(_model_snapshot(model, positions))
    root = pose_to_matrix(root_pose_w)
    lowest = math.inf
    for shape in shapes:
        if shape.mesh is not None:
            points = np.asarray(shape.mesh.vertices, dtype=float)
        else:
            low, high = np.asarray(shape.bounds, dtype=float)
            points = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1])
                               for z in (low[2], high[2])])
        pose = root @ transforms[shape.link] @ shape.transform
        lowest = min(lowest, float((points @ pose[:3, :3].T + pose[:3, 3])[:, 2].min()))
    if not math.isfinite(lowest):
        raise ValueError("The loaded robot model has no collision geometry to place on the floor")
    return lowest


def validate_upright_task_runtime(task, *, command_names, joint_limits, model, positions):
    """Startup check of every posture path, the upright overrides and the spawn height with the loaded model.

    The declared torso height is written verbatim, so the measured reset joints
    (with the overrides) must keep every collision body above the floor.
    """
    for stage in POSTURE_STAGES:
        path = task["posture_paths"][stage]
        if not set(path["joint_names"]) <= set(command_names):
            raise ValueError(f"posture_paths.{stage} must name only upper command joints")
        validate_posture_waypoints(path, joint_limits=joint_limits, model=model, positions=positions)
    if "handover" in task:
        # The opt-in two-arm pre-handover path is checked against the loaded model like the others.
        path = task["handover"]["posture_paths"]["handover_posture"]
        if not set(path["joint_names"]) <= set(command_names):
            raise ValueError("handover.posture_paths.handover_posture must name only upper command joints")
        validate_posture_waypoints(path, joint_limits=joint_limits, model=model, positions=positions)
        if "rotation_path" in task["handover"]:
            # The opt-in two-hand rotation path (handover_rotate) is checked against the loaded model too.
            path = task["handover"]["rotation_path"]
            if not set(path["joint_names"]) <= set(command_names):
                raise ValueError("handover.rotation_path must name only upper command joints")
            validate_posture_waypoints(path, joint_limits=joint_limits, model=model, positions=positions)
    single_step = task.get("navigation_control", {}).get("single_step")
    if isinstance(single_step, Mapping) and PRE_STEP_PATH_KEY in single_step:
        # The opt-in single-step pre-step path (navigate_target) is checked against the loaded model too.
        path = single_step[PRE_STEP_PATH_KEY]
        if not set(path["joint_names"]) <= set(command_names):
            raise ValueError("navigation_control.single_step.pre_step_posture_path must name only upper command joints")
        validate_posture_waypoints(path, joint_limits=joint_limits, model=model, positions=positions)
    overrides = upright_overrides(task["upright_start"], positions, command_names=command_names,
                                  joint_limits=joint_limits, model=model)
    lowest = lowest_collision_point_z(model, upright_spawn_pose(task["upright_start"]), {**positions, **overrides})
    if lowest < FLOOR_HEIGHT_M - FLOOR_PENETRATION_TOLERANCE_M:
        raise ValueError(f"upright_start.robot_root_pose_w starts the robot {(FLOOR_HEIGHT_M - lowest) * 1e3:.1f} mm "
                         "inside the floor at the reset joints; raise the declared torso height")
    return lowest


class PostureStageExecutor:
    """Runtime owner of one joint-space posture stage (reach, carry, place or rest).

    It also runs the two-arm handover paths and the opt-in single-step pre-step path (navigate_target).

    The path starts at the live commanded targets of stage entry, so the first
    command continues exactly what the actuators were already given. Each
    update first compares the measured joints with the targets applied in the
    previous control step (the command that produced the measurement). A trip
    returns a failed status WITHOUT advancing, so the caller can log the
    evidence and stop the episode; otherwise the reference advances by ``dt``.
    """

    def __init__(self, stage_id, path, *, start_targets, measured_positions, command_names, joint_limits,
                 model=None, excluded_joint_names=()):
        if stage_id not in POSTURE_EXECUTOR_STAGES:
            raise ValueError(f"Not a joint-space posture stage: {stage_id}")
        names = tuple(path["joint_names"])
        self.hold_last_applied = is_applied_hold(path)
        if "reference_mode" in path and (not self.hold_last_applied or stage_id not in HOLD_STAGES):
            raise ValueError("hold_last_applied is allowed only for " + ", ".join(HOLD_STAGES))
        if self.hold_last_applied:
            from .latest_robot import COMMAND_JOINT_NAMES, GRIPPER_JOINT_NAMES
            required = set(COMMAND_JOINT_NAMES) - set(GRIPPER_JOINT_NAMES)
            if len(names) != len(required) or set(names) != required:
                raise ValueError("hold_last_applied owns exactly the 19 canonical posture joints")
            start_targets = applied_hold_targets(start_targets, names)
        if not set(names) <= set(command_names) or set(names) & set(excluded_joint_names):
            raise ValueError("Posture paths must name upper command joints and exclude the jaws")
        validate_posture_waypoints(path, joint_limits=joint_limits, model=model, positions=measured_positions)
        start = {}
        for name in names:
            value = float(start_targets[name])
            if not _within(value, joint_limits[name]):
                raise ValueError(f"Posture stage starts outside shared control limits: {name}={value}")
            start[name] = value
        self.stage_id = stage_id
        self.joint_names = names
        self.tracking_limit_rad = float(path["max_tracking_error_rad"])
        if self.hold_last_applied and model is not None:
            model.validate_snapshot({**_model_snapshot(model, measured_positions), **start})
        if self.hold_last_applied:
            self.reference = AppliedTargetHoldReference(start)
        else:
            reference_type, options = JointPathReference, {}
            if path.get("reference_profile") == "bounded_c2":
                from .continuous_posture import ContinuousJointPathReference
                reference_type = ContinuousJointPathReference
                options["max_joint_deviation_rad"] = path["max_joint_deviation_rad"]
            elif "reference_profile" in path:
                raise ValueError("Unknown posture reference_profile")
            # Opt-in per-segment caps (runtime_v26); absent, the reference gets exactly the v25 arguments.
            for key, option in (("segment_max_speed_rad_s", "segment_max_speeds_rad_s"),
                                ("segment_max_acceleration_rad_s2", "segment_max_accelerations_rad_s2")):
                if key in path:
                    options[option] = path[key]
            self.reference = reference_type(
                start, path["waypoints"], joint_names=names, max_speed_rad_s=path["max_speed_rad_s"],
                max_acceleration_rad_s2=path["max_acceleration_rad_s2"],
                segment_min_durations_s=path["segment_min_duration_s"], **options)
        # Declared per-segment caps only (runtime_v26 review fix, B5 trace evidence): the caps, the resulting
        # segment durations and, for bounded_c2, the certified per-segment bounds, on every published status.
        self._segment_limits = None
        if not self.hold_last_applied and any(key in path for key in ("segment_max_speed_rad_s",
                                                                       "segment_max_acceleration_rad_s2")):
            def certified(name):
                values = getattr(self.reference, name, None)
                return None if values is None else [float(value) for value in values]
            self._segment_limits = {
                "segment_max_speeds_rad_s": (None if "segment_max_speed_rad_s" not in path
                                             else [float(value) for value in path["segment_max_speed_rad_s"]]),
                "segment_max_accelerations_rad_s2": (None if "segment_max_acceleration_rad_s2" not in path else
                                                     [float(value) for value in path["segment_max_acceleration_rad_s2"]]),
                "segment_durations_s": [float(value) for value in self.reference.segment_durations],
                "segment_speed_bounds_rad_s": certified("segment_speed_bounds_rad_s"),
                "segment_acceleration_bounds_rad_s2": certified("segment_acceleration_bounds_rad_s2")}
        self._status = None

    @property
    def status(self):
        """The last published posture reference (a copy), or None before the first update."""
        return deepcopy(self._status)

    def update(self, measured_positions, applied_targets, dt):
        """Return (targets or None, status) for one control step."""
        if self.hold_last_applied:
            applied = applied_hold_targets(applied_targets, self.joint_names)
            if any(abs(applied[n] - self.reference.final[n]) > 1e-9 for n in self.joint_names):
                raise ValueError("hold_last_applied command owner changed after entry")
        errors = {}
        for name in self.joint_names:
            measured, applied = float(measured_positions[name]), float(applied_targets[name])
            if not math.isfinite(measured) or not math.isfinite(applied):
                raise ValueError(f"Posture tracking needs finite measured and applied joints: {name}")
            errors[name] = abs(measured - applied)
        joint = max(errors, key=errors.get)
        failed = errors[joint] > self.tracking_limit_rad
        targets = None if failed else self.reference.update(dt)
        self._status = {
            "stage_id": self.stage_id, "status": "failed" if failed else "running",
            "elapsed_s": self.reference.elapsed, "duration_s": self.reference.duration,
            "complete": self.reference.complete, "segment_index": self.reference.segment_index,
            "max_tracking_error_rad": errors[joint], "max_tracking_error_joint": joint,
            "tracking_error_limit_rad": self.tracking_limit_rad,
            "final_joint_positions": self.reference.final}
        if self.hold_last_applied:
            self._status["reference_mode"] = HOLD_MODE
        if self._segment_limits is not None:
            self._status["segment_limits"] = deepcopy(self._segment_limits)
        if failed:
            self._status["reason"] = f"posture_tracking_error:{self.stage_id}"
        return targets, self.status


def object_pose_settings(task, stage_id, waypoint_index):
    """CrsPoseController settings for one (stage, object waypoint) segment.

    The task object_pose_control is refined by object_pose_control_overrides for
    that segment (lift/transfer: per waypoint; place: one mapping). Object
    feedback is then removed outside the task's object-feedback stages, exactly
    as without overrides.
    """
    settings = dict(task["object_pose_control"])
    overrides = task.get("object_pose_control_overrides", {})
    if stage_id in overrides:
        segment = overrides[stage_id]
        if stage_id != "place":
            if type(waypoint_index) is not int or not 0 <= waypoint_index < len(segment):
                raise ValueError(f"No object_pose_control override for {stage_id} waypoint {waypoint_index}")
            segment = segment[waypoint_index]
        else:
            # runtime_v26 C4: the two-speed descent keys configure place_two_speed_descent, not the speeds here.
            segment = {key: value for key, value in segment.items() if key not in TWO_SPEED_DESCENT_KEYS}
        settings.update(segment)
    if stage_id not in task.get("object_feedback_stages", list(_OBJECT_CONTROL_STAGES)):
        settings.pop("object_feedback", None)
    return settings


def place_two_speed_descent(task):
    """The declared two-speed place descent {fast_linear_speed, slow_zone_height_m}, or None (runtime_v25)."""
    place = task.get("object_pose_control_overrides", {}).get("place", {})
    if not any(key in place for key in TWO_SPEED_DESCENT_KEYS):
        return None
    return {key: float(place[key]) for key in TWO_SPEED_DESCENT_KEYS}


def retreat_pose_settings(task, stage_id, release_phase=None):
    """Keep contact release unchanged when speeding up the empty-hand retreat.

    runtime_v26 C5: an opt-in ``retreat_pose_control_overrides.release_separating`` refines only the release
    'separating' subphase (hand open and contact-free after the clearing phase); opening and clearing keep
    retreat_pose_control.
    """
    settings = dict(task.get("retreat_pose_control", {
        "linear_speed": .002, "linear_acceleration": .004,
        "angular_speed": .12, "angular_acceleration": .3,
        "position_lead": .004, "rotation_lead": .02}))
    overrides = task.get("retreat_pose_control_overrides", {})
    settings.update(overrides.get(stage_id, {}))
    if stage_id == "release" and release_phase == "separating":
        settings.update(overrides.get("release_separating", {}))
    return settings


def retreat_segment_key(stage_id, command):
    """Release sub-phase and retreat waypoint: every segment gets a new TCP reference."""
    return stage_id, command.get("release_phase"), command.get("retreat_waypoint_index")


class RetreatMotionFeedforward:
    """Pose-derived twist for the released-hand retreat only, reset per waypoint.

    The existing arm servo still owns every speed, tracking and joint bound.
    Contact release and handover never receive a twist from this helper.
    """

    def __init__(self):
        self._segment = None
        self._reference = None

    def update(self, stage_id, command, base_pose_w, target_pose_w, dt):
        if stage_id != "retreat":
            self._segment = self._reference = None
            return None
        from .carry_feedforward import RelativePoseFeedforward
        segment = retreat_segment_key(stage_id, command)
        if segment != self._segment:
            self._reference = RelativePoseFeedforward()
            self._segment = segment
        return self._reference.update(base_pose_w, target_pose_w, dt)


def motion_waypoint_index(stage_id, command):
    """Watchdog waypoint key: a new waypoint goal restarts the stagnation baseline."""
    if stage_id in _OBJECT_CONTROL_STAGES:
        return command.get("object_waypoint_index", 0)
    if stage_id == "pregrasp":
        return command.get("pregrasp_pose_reference", {}).get("waypoint_index", 0)
    if stage_id == "retreat" and command.get("retreat_waypoint_index") is not None:
        return command["retreat_waypoint_index"]
    return command.get("base_route_waypoint_index", 0)


def validate_diagnostic_stop(stop_after, stages):
    """A calibration stop must name a non-final stage: leaving the final stage is success."""
    if stop_after is None:
        return None
    stages = tuple(stages)
    if stop_after not in stages[:-1]:
        raise ValueError("--diagnostic_stop_after must name a non-final stage of this task: " + ", ".join(stages[:-1]))
    return stop_after


def diagnostic_stop_reason(stop_after, previous_stage_id, command):
    """The failure reason once the machine has left ``stop_after`` for the next stage, else None."""
    if (stop_after is None or previous_stage_id != stop_after or command.get("status") != "running"
            or command.get("stage_id") == stop_after):
        return None
    return f"diagnostic_stop_after:{stop_after}"


def validate_diagnostic_waypoint_stop(value, task, stages=None):
    """Parse a zero-based, non-final waypoint of an explicitly configured path.

    This describes a measured FSM transition, not a reference progress or a
    timeout. The last waypoint must use the existing stage diagnostic stop.
    """
    if value is None:
        return None
    if not isinstance(value, str) or re.fullmatch(r"(lift|transfer):(0|[1-9][0-9]*)", value) is None:
        raise ValueError("--diagnostic_stop_after_waypoint must be lift:INDEX or transfer:INDEX (zero-based)")
    stage, index_text = value.split(":")
    index = int(index_text)
    paths = task.get("object_waypoint_offsets_w", {})
    path = paths.get(stage) if isinstance(paths, Mapping) else None
    if not isinstance(path, list) or len(path) < 2:
        raise ValueError("--diagnostic_stop_after_waypoint requires an explicit path with at least two waypoints")
    if stages is not None and stage not in stages:
        raise ValueError("--diagnostic_stop_after_waypoint stage is not executed by this task")
    if index >= len(path) - 1:
        raise ValueError("--diagnostic_stop_after_waypoint must name a non-final configured waypoint; "
                         "use --diagnostic_stop_after for the final waypoint")
    return stage, index


def validate_required_diagnostic_waypoint_stop(task, value, *, execute):
    """A partial-path profile can be inspected but can only execute its pinned prefix."""
    if "required_diagnostic_stop_after_waypoint" not in task:
        return
    required = task["required_diagnostic_stop_after_waypoint"]
    if required is None:
        raise ValueError("required_diagnostic_stop_after_waypoint must name a non-final configured waypoint")
    validate_diagnostic_waypoint_stop(required, task)
    if execute and value != required:
        raise ValueError(f"This partial-path profile requires --diagnostic_stop_after_waypoint {required}")


def diagnostic_waypoint_stop_reason(stop, previous_command, command):
    """Stop only on i -> i+1 within the same running stage, before applying i+1.

    A physical failure takes precedence; merely entering a stage, reaching a
    reference endpoint or changing stages does not establish this transition.
    """
    if stop is None:
        return None
    stage, index = stop
    if (previous_command.get("status") == command.get("status") == "running"
            and previous_command.get("stage_id") == command.get("stage_id") == stage
            and type(previous_command.get("object_waypoint_index")) is int
            and type(command.get("object_waypoint_index")) is int
            and previous_command["object_waypoint_index"] == index
            and command["object_waypoint_index"] == index + 1):
        return f"diagnostic_stop_after_waypoint:{stage}:{index}"
    return None


def upright_spawn_pose(upright_start):
    """The declared torso (base_link) pose, height included; never the asset reset height."""
    pose = [float(value) for value in upright_start["robot_root_pose_w"]]
    if len(pose) != 7:
        raise ValueError("upright_start.robot_root_pose_w must be xyz plus a wxyz quaternion")
    pose_to_matrix(pose)
    return pose


def upright_overrides(upright_start, positions, *, command_names, joint_limits, model=None):
    """Validated jaw/neck overrides of the env default joints, as floats."""
    overrides = {name: float(value) for name, value in upright_start["joint_overrides"].items()}
    for name, value in overrides.items():
        if name not in command_names or name not in joint_limits or not _within(value, joint_limits[name], 0.):
            raise ValueError(f"upright_start override must be a bounded upper command joint: {name}")
    if model is not None:
        model.validate_snapshot({**_model_snapshot(model, positions), **overrides})
    return overrides


def gaze_target_w(task, object_pose_w):
    """The source grasp point the head camera aims at, as the legacy episode initialization uses."""
    return (pose_to_matrix(object_pose_w) @ pose_to_matrix(task["grasp_pose_object"]))[:3, 3]


def upright_joint_writes(overrides, gaze, *, joint_limits):
    """Joint state written at the boundary: the declared overrides plus the gaze neck targets."""
    writes = dict(overrides)
    if gaze is not None:
        neck = {name: float(value) for name, value in gaze["joint_targets_rad"].items()}
        if set(neck) & set(writes):
            raise ValueError("upright_start head gaze owns the neck; overrides must not name it")
        for name, value in neck.items():
            if name not in joint_limits or not _within(value, joint_limits[name]):
                raise ValueError(f"Head gaze target exceeds shared control limits: {name}")
        writes.update(neck)
    return writes


def upright_start_provenance(*, root_pose_w, overrides, gaze, scope, joint_targets):
    """What the boundary wrote: the declared root pose, overrides and gaze; never a demonstration.

    ``joint_targets`` is the resulting upper command of the posture-path joints
    (env defaults with the overrides and gaze applied). The machine readiness
    and admission require the unrecorded preparation to command exactly it.
    """
    if scope not in (EPISODE_UPRIGHT_START_SCOPE, STARTUP_UPRIGHT_START_SCOPE):
        raise ValueError(f"Unknown upright start scope: {scope}")
    targets = {name: float(value) for name, value in joint_targets.items()}
    if not targets or not all(map(math.isfinite, targets.values())):
        raise ValueError("The upright start must record finite posture-joint command targets")
    return {"robot_root_pose_w": [float(value) for value in root_pose_w],
            "joint_overrides_applied": {name: float(value) for name, value in overrides.items()},
            "head_gaze": deepcopy(gaze), "scope": scope, "demonstration_pose_write": False,
            "upright_joint_targets": targets}
