"""Optional measured-state gates for a two-hand, single-object task.

This is an adapter for the existing task FSM, not a second success machine.
It commands no simulator objects and performs no attachment, pose write, force
injection or independent declaration of task success. Geometry and reachability
must be checked separately for both arms.

Integration order (the existing single-hand default needs no instance):
* Call ``enter`` at each FSM stage entry with the ORIGINAL raw observation.
* Evaluate once before the FSM's grasp guards/dwell. AND ``stage_ready`` with
  the existing condition, retain original task/object/posture/time predicates,
  and use ``both_grasps_valid`` for carrying retention. Propagate any failure.
* On close->lift, ``enter`` captures both measured relations once. The existing
  primary capture must come from that SAME observation. Do not overwrite either
  retention transform at subsequent manipulation phases.
* Feed ``pause_motion`` into the common object-reference and base controller:
  missing pinch/slip pauses both arms and walking immediately, even during the
  existing loss grace. Do not advance CarryStabilizer while paused. Holding the
  last references may require an explicitly logged protective acceleration stop.
* Derive the secondary TCP from the SAME bounded object reference, then run its
  own ArmServo. Do not run an independent object trajectory for the second arm.
* Both arms need live object-filtered contacts and PinchSettlingTracker input.
  The legacy runtime's active-primary-only closed/settled calculation is not
  sufficient. Log this module's diagnostics and both arm servo diagnostics.

The FSM still owns dwell and final settled-object success. No support-contact
exception is implemented here: support alone cannot bypass either hand's
retention guard before release. This module is deliberately not imported by the
existing runtime until its integration is explicitly enabled and reviewed.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

import numpy as np

from .live_scene import two_finger_pinch_evidence
from .snapshot import matrix_to_pose, pose_to_matrix

CARRYING_STAGES = frozenset({"lift", "open_hinge", "recover_carry", "recover_base",
                            "navigate_target", "transfer", "place"})
RELEASED_STAGES = frozenset({"retreat", "settle"})
STAGES = CARRYING_STAGES | RELEASED_STAGES | {
    "prerequisites", "navigate_source", "pregrasp", "approach", "close", "release"}


def _positive(value, name, *, zero=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (number == 0 and not zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return number


def _vector(value, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain three finite values")
    return array.copy()


def _boolean(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a measured boolean")
    return bool(value)


def _errors(current, target):
    translation = float(np.linalg.norm(current[:3, 3] - target[:3, 3]))
    cosine = (np.trace(current[:3, :3] @ target[:3, :3].T) - 1.) / 2.
    return translation, float(np.arccos(np.clip(cosine, -1., 1.)))


def validate_bimanual_settings(task, thresholds):
    """Validate the optional secondary specification without changing a task.

    The secondary inherits the primary acquisition, capture and contact limits;
    it cannot use this extension to widen existing acceptance criteria.
    """
    if "bimanual" not in task:
        return None
    settings = deepcopy(task["bimanual"])
    if not isinstance(settings, dict):
        raise ValueError("bimanual must be a mapping")
    allowed = {"secondary_arm", "grasp_pose_object", "pregrasp_offset_w", "retreat_offset_w"}
    if set(settings) - allowed:
        raise ValueError("Unknown bimanual settings: " + ",".join(sorted(set(settings) - allowed)))
    primary = task.get("arm")
    secondary = settings.get("secondary_arm")
    if primary not in {"left", "right"} or secondary not in {"left", "right"} or primary == secondary:
        raise ValueError("bimanual requires distinct left and right arms")
    for name in ("grasp_pose_object",):
        settings[name] = matrix_to_pose(pose_to_matrix(settings[name]))
    for name in ("pregrasp_offset_w", "retreat_offset_w"):
        settings[name] = _vector(settings[name], "bimanual." + name).tolist()
    pose_to_matrix(task["grasp_pose_object"])
    for name in ("tcp_position_m", "tcp_rotation_rad", "capture_position_m", "capture_rotation_rad",
                 "contact_dwell_s", "lost_grasp_grace_s", "max_observation_gap_s"):
        _positive(thresholds[name], name)
    frame = task.get("close_acquisition_frame", "world_tcp")
    if frame not in {"world_tcp", "object_relative"}:
        raise ValueError("Invalid close_acquisition_frame")
    if frame == "object_relative":
        for name in ("close_acquisition_position_m", "close_acquisition_rotation_rad"):
            _positive(task[name], name)
    bound = _positive(task.get("maximum_pinch_angle_rad", .15), "maximum_pinch_angle_rad")
    if bound >= 1.4:
        raise ValueError("maximum_pinch_angle_rad must be below the open aperture")
    return settings


class BimanualCoordinator:
    """Fail-closed secondary gates and immutable dual grasp relations.

    ``evaluate`` returns an additional gate, never a replacement for existing
    FSM conditions. ``enter`` must not be called speculatively: stage changes
    remain the primary FSM's responsibility after its original dwell.
    """

    def __init__(self, task, thresholds, object_path):
        self.settings = validate_bimanual_settings(task, thresholds)
        if self.settings is None:
            raise ValueError("BimanualCoordinator requires an explicit bimanual specification")
        if not isinstance(object_path, str) or not object_path.startswith("/"):
            raise ValueError("object_path must be an absolute prim path")
        self.task, self.thresholds = deepcopy(task), deepcopy(thresholds)
        self.primary_arm, self.secondary_arm = task["arm"], self.settings["secondary_arm"]
        self.arms = (self.primary_arm, self.secondary_arm)
        self.object_path = object_path
        self._nominal = {self.primary_arm: pose_to_matrix(task["grasp_pose_object"]),
                         self.secondary_arm: pose_to_matrix(self.settings["grasp_pose_object"])}
        # Opt-in approach_axis_alignment (nepes_tasks) belongs to the primary task grasp; the secondary
        # grasp declares no pinch axis and keeps its original approach gate (and pregrasp gate).
        self._approach_axis = None
        if "approach_axis_alignment" in task:
            from .nepes_tasks import validate_approach_axis_alignment
            self._approach_axis = validate_approach_axis_alignment(task["approach_axis_alignment"], thresholds)
        self._pregrasp_axis = self._approach_axis is not None and "pregrasp_position_m" in self._approach_axis
        self._grasp_object = None
        self._grasp_goals = {}
        self._retention = {}
        self._release_tcp = None
        self._retreat_tcp = None
        self._last_secondary_target = None
        self._stage = None
        self._last_time = None
        self._close_since = None
        self._loss_since = {arm: None for arm in self.arms}
        self._failure = None
        self._last_result = None

    @property
    def retention_object_to_tcp(self):
        """Copies of acquisition relations; callers cannot rebase retention."""
        return {arm: matrix_to_pose(value) for arm, value in self._retention.items()}

    def _read(self, raw):
        now = _positive(raw["time_s"], "time_s", zero=True)
        obj = pose_to_matrix(raw["object_poses_w"][self.object_path])
        hands = {}
        for arm in self.arms:
            tcp = pose_to_matrix(raw["tcp_poses_w"][arm])
            opened = _boolean(raw["hand_open"][arm], "hand_open")
            closed = _boolean(raw["hand_closed"][arm], "hand_closed")
            settled = _boolean(raw["hand_settled"][arm], "hand_settled")
            if opened and closed:
                raise ValueError(f"{arm} hand cannot be both open and closed")
            contacts = raw["contacts"][arm]
            if isinstance(contacts, (str, bytes)) or not isinstance(contacts, (list, tuple, set, frozenset)):
                raise ValueError("contacts must be full object prim paths")
            if any(not isinstance(p, str) or not p.startswith("/") for p in contacts):
                raise ValueError("contacts must be full object prim paths")
            evidence = raw["contact_evidence"][arm][self.object_path]
            if (not isinstance(evidence, Mapping) or evidence.get("available") is not True
                    or evidence.get("object_path") != self.object_path or evidence.get("arm") != arm
                    or evidence.get("source") != "physx_filtered_hand_object_contacts"):
                raise ValueError(f"{arm}:contact_sensor_unavailable_or_wrong_object")
            angle = float(evidence["joint_position_rad"])
            pinch = two_finger_pinch_evidence(
                evidence, arm, angle, closed_position_rad=0.,
                maximum_closed_angle_rad=self.task.get("maximum_pinch_angle_rad", .15))
            forces = np.asarray(evidence["forces_w"], dtype=float)
            paths = evidence["sensor_paths"]
            threshold = _positive(evidence["force_threshold_n"], "force_threshold_n")
            if (forces.ndim != 3 or forces.shape[0] != len(paths) or forces.shape[1] != 1
                    or forces.shape[2] != 3 or not np.isfinite(forces).all()
                    or len(pinch["jaw_force_magnitudes_n"]) != 2):
                raise ValueError(f"{arm}:invalid_filtered_jaw_force_matrix")
            any_contact = self.object_path in contacts or float(np.linalg.norm(forces, axis=-1).sum()) > threshold
            hands[arm] = {"tcp": tcp, "open": opened, "closed": closed, "settled": settled,
                          "any_contact": any_contact, "pinch": bool(closed and pinch["pinch_confirmed"]),
                          "pinch_evidence": pinch}
        return {"time_s": now, "object": obj, "hands": hands}

    def _acquisition(self, observed, arm):
        hand = observed["hands"][arm]
        if arm not in self._grasp_goals:
            return False, None
        world_error = _errors(hand["tcp"], self._grasp_goals[arm])
        relative = np.linalg.inv(observed["object"]) @ hand["tcp"]
        relative_error = _errors(relative, self._nominal[arm])
        if self.task.get("close_acquisition_frame", "world_tcp") == "object_relative":
            limits = (self.task["close_acquisition_position_m"], self.task["close_acquisition_rotation_rad"])
            pose_ok = all(p <= limits[0] and r <= limits[1] for p, r in (world_error, relative_error))
        else:
            pose_ok = (world_error[0] <= self.thresholds["tcp_position_m"]
                       and world_error[1] <= self.thresholds["tcp_rotation_rad"])
        return bool(hand["pinch"] and hand["settled"] and pose_ok), {
            "world_position_error_m": world_error[0], "world_rotation_error_rad": world_error[1],
            "relative_position_error_m": relative_error[0], "relative_rotation_error_rad": relative_error[1]}

    def enter(self, stage, observation):
        """Synchronize an actual FSM stage entry, capturing only close->carry."""
        if stage not in STAGES:
            raise ValueError(f"Unknown bimanual stage: {stage}")
        if self._failure:
            raise ValueError("A failed bimanual coordinator cannot be restarted")
        observed = self._read(observation)
        if self._last_time is not None and observed["time_s"] < self._last_time:
            raise ValueError("Bimanual stage entry time must not go backwards")
        old = self._stage
        if stage == "pregrasp":
            if self._retention:
                raise ValueError("Cannot reacquire an already captured bimanual episode")
            self._grasp_object = observed["object"].copy()
            self._grasp_goals = {arm: observed["object"] @ self._nominal[arm] for arm in self.arms}
        if stage in CARRYING_STAGES and not self._retention:
            if old != "close" or self._close_since is None:
                raise ValueError("Both measured close conditions must precede capture")
            if (observed["time_s"] - self._close_since < self.thresholds["contact_dwell_s"] - 1e-10
                    or not all(self._acquisition(observed, arm)[0] for arm in self.arms)):
                raise ValueError("Both settled acquisitions must satisfy the original contact dwell")
            inverse = np.linalg.inv(observed["object"])
            self._retention = {arm: inverse @ observed["hands"][arm]["tcp"] for arm in self.arms}
        if stage == "release":
            if not self._retention:
                raise ValueError("Release requires an acquired bimanual object")
            self._release_tcp = observed["hands"][self.secondary_arm]["tcp"].copy()
            self._retreat_tcp = self._release_tcp.copy()
            self._retreat_tcp[:3, 3] += _vector(self.settings["retreat_offset_w"], "retreat offset")
        if stage in RELEASED_STAGES and self._release_tcp is None:
            raise ValueError("Retreat/settle require a measured release entry")
        self._stage = stage
        self._last_result = None
        if stage != "close":
            self._close_since = None
        # Loss timers intentionally survive carrying stage transitions.

    def evaluate(self, stage, observation):
        """Additional stage gate plus per-hand grace diagnostics; never success."""
        if self._failure is not None:
            return deepcopy(self._last_result)
        try:
            if stage != self._stage:
                raise ValueError("Call enter before evaluating a new stage")
            observed = self._read(observation)
            now = observed["time_s"]
            if self._last_time is not None:
                gap = now - self._last_time
                if gap < 0:
                    raise ValueError("Observation time went backwards")
                if gap > self.thresholds["max_observation_gap_s"]:
                    raise ValueError("observation_gap")
            self._last_time = now
            diagnostics, all_valid = {}, bool(self._retention)
            for arm, hand in observed["hands"].items():
                acquisition_ok, acquisition_errors = self._acquisition(observed, arm)
                errors = (_errors(np.linalg.inv(observed["object"]) @ hand["tcp"], self._retention[arm])
                          if arm in self._retention else None)
                retained = bool(errors is not None and errors[0] <= self.thresholds["capture_position_m"]
                                and errors[1] <= self.thresholds["capture_rotation_rad"])
                valid = bool(hand["pinch"] and retained)
                all_valid = all_valid and valid
                if stage in CARRYING_STAGES:
                    if valid:
                        self._loss_since[arm] = None
                    elif self._loss_since[arm] is None:
                        self._loss_since[arm] = now
                elapsed = 0. if self._loss_since[arm] is None else now - self._loss_since[arm]
                diagnostics[arm] = {"sensor_available": True, "hand_open": hand["open"],
                    "hand_closed": hand["closed"], "hand_settled": hand["settled"],
                    "any_object_contact": hand["any_contact"], "pinch_confirmed": hand["pinch"],
                    "acquisition_valid": acquisition_ok, "acquisition_errors": acquisition_errors,
                    "retention_valid": retained, "grasp_valid": valid,
                    "capture_position_error_m": errors[0] if errors else None,
                    "capture_rotation_error_rad": errors[1] if errors else None,
                    "loss_duration_s": elapsed, **hand["pinch_evidence"]}
            failure = None
            if stage in CARRYING_STAGES:
                for arm in self.arms:
                    if self._loss_since[arm] is not None and diagnostics[arm]["loss_duration_s"] >= self.thresholds["lost_grasp_grace_s"] - 1e-10:
                        failure = f"bimanual_grasp_lost:{arm}"
                        break
            released = all(h["open"] and not h["any_contact"] for h in observed["hands"].values())
            if stage in RELEASED_STAGES and not released:
                failure = "bimanual_release_lost"
            ready, alignment, pregrasp_alignment = True, None, None
            if stage in {"pregrasp", "approach"}:
                targets = self._approach_targets(stage)
                ready = all(h["open"] and all(v <= t for v, t in zip(_errors(h["tcp"], targets[a]),
                            (self.thresholds["tcp_position_m"], self.thresholds["tcp_rotation_rad"])))
                            for a, h in observed["hands"].items())
                if stage == "pregrasp" and self._pregrasp_axis:
                    # The primary FSM's opt-in pregrasp gate: the same error from the final primary pregrasp target.
                    from .nepes_tasks import approach_axis_error
                    pregrasp_alignment = approach_axis_error(
                        self._approach_axis, observed["hands"][self.primary_arm]["tcp"], targets[self.primary_arm],
                        self._grasp_object, stage="pregrasp")
                    ready = ready and pregrasp_alignment["satisfied"]
                if stage == "approach" and self._approach_axis is not None:
                    # The primary FSM's opt-in pinch-axis gate on the same target and pregrasp-entry object pose.
                    from .nepes_tasks import approach_axis_error
                    alignment = approach_axis_error(self._approach_axis, observed["hands"][self.primary_arm]["tcp"],
                                                    targets[self.primary_arm], self._grasp_object)
                    ready = ready and alignment["satisfied"]
            elif stage == "close":
                ready = all(diagnostics[arm]["acquisition_valid"] for arm in self.arms)
                if ready and self._close_since is None:
                    self._close_since = now
                elif not ready:
                    self._close_since = None
            elif stage in CARRYING_STAGES:
                ready = all_valid
            elif stage == "release":
                ready = released
            elif stage == "retreat":
                errors = _errors(observed["hands"][self.secondary_arm]["tcp"], self._retreat_tcp)
                ready = released and errors[0] <= self.thresholds["tcp_position_m"] and errors[1] <= self.thresholds["tcp_rotation_rad"]
            elif stage == "settle":
                ready = released
            self._failure = failure
            result = {"enabled": True, "stage_id": stage, "measurement_time_s": now,
                      "stage_ready": bool(ready and failure is None), "both_grasps_valid": bool(all_valid),
                      "pause_motion": bool(failure or (stage in CARRYING_STAGES and not all_valid)),
                      "failure_reason": failure, "primary_arm": self.primary_arm,
                      "secondary_arm": self.secondary_arm, "hands": diagnostics,
                      "retention_object_to_tcp": self.retention_object_to_tcp,
                      "attachment_mode": "none", "physical_success_verified": False}
            if self._approach_axis is not None:  # declared only: primary-arm evidence in approach, else None
                result["approach_axis_alignment"] = alignment
            if self._pregrasp_axis:  # declared pregrasp_position_m only: primary-arm evidence in pregrasp, else None
                result["pregrasp_axis_alignment"] = pregrasp_alignment
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            self._failure = "bimanual_invalid_observation:" + str(exc)
            result = {"enabled": True, "stage_id": stage, "stage_ready": False,
                      "both_grasps_valid": False, "pause_motion": True,
                      "failure_reason": self._failure, "hands": {}, "attachment_mode": "none"}
            if self._approach_axis is not None:  # the declared key stays; no evidence without a valid observation
                result["approach_axis_alignment"] = None
            if self._pregrasp_axis:
                result["pregrasp_axis_alignment"] = None
        self._last_result = deepcopy(result)
        return result

    def _approach_targets(self, stage):
        if len(self._grasp_goals) != 2:
            raise ValueError("Pregrasp entry must establish both fixed grasp goals")
        targets = {arm: value.copy() for arm, value in self._grasp_goals.items()}
        if stage == "pregrasp":
            targets[self.primary_arm][:3, 3] += _vector(self.task["pregrasp_offset_w"], "primary pregrasp offset")
            targets[self.secondary_arm][:3, 3] += _vector(self.settings["pregrasp_offset_w"], "secondary pregrasp offset")
        return targets

    def secondary_target(self, stage, *, object_reference_pose_w=None):
        """TCP target from a shared bounded object reference, never current drift.

        During a transient carry pause the last target is held. The caller must
        also hold the primary reference and stop the base; this helper cannot
        make an independently advancing primary arm safe.
        """
        if stage != self._stage:
            raise ValueError("Target stage does not match coordinator stage")
        if self._failure:
            return None
        if stage in {"pregrasp", "approach", "close"}:
            target = self._approach_targets("approach" if stage == "close" else stage)[self.secondary_arm]
        elif stage in CARRYING_STAGES:
            if not self._retention:
                raise ValueError("A shared reference requires both immutable captures")
            if self._last_result and self._last_result["pause_motion"]:
                return (matrix_to_pose(self._last_secondary_target) if self._last_secondary_target is not None else None)
            if object_reference_pose_w is None:
                raise ValueError("Carrying requires the existing bounded object reference")
            target = pose_to_matrix(object_reference_pose_w) @ self._retention[self.secondary_arm]
        elif stage == "release":
            target = self._release_tcp
        elif stage in RELEASED_STAGES:
            target = self._retreat_tcp
        else:
            return None
        self._last_secondary_target = target.copy()
        return matrix_to_pose(target)
