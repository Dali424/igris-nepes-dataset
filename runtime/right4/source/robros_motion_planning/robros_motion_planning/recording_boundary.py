"""Read-only readiness gate for beginning a prepared-scene demonstration.

Preparation remains in the full attempt trace, including unintended contacts.
This gate changes recording scope only; it never commands or resets physics.
"""
from collections.abc import Mapping
from copy import deepcopy
import math
import numpy as np

from .snapshot import pose_to_matrix


BOUNDARY_KINDS = ("after_measured_pregrasp", "after_measured_grasp_hold", "after_measured_upright_stance")


def validate_recording_boundary(settings):
    if settings is None:
        return None
    if (not isinstance(settings, Mapping)
            or set(settings) != {"kind", "readiness_dwell_s"}
            or settings["kind"] not in BOUNDARY_KINDS):
        raise ValueError("recording_boundary requires a measured preparation kind and readiness_dwell_s")
    value = settings["readiness_dwell_s"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < .5:
        raise ValueError("Recording readiness requires at least 0.5 s of continuous measured stability")
    return {"kind": settings["kind"], "readiness_dwell_s": float(value)}


def _speed(value):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (6,) or not np.isfinite(vector).all():
        raise ValueError("Measured world linear/angular velocity must contain six finite values")
    return float(np.linalg.norm(vector[:3])), float(np.linalg.norm(vector[3:])), abs(float(vector[5]))


def _pose_errors(first, second):
    return (float(np.linalg.norm(second[:3, 3] - first[:3, 3])),
            float(math.acos(np.clip((np.trace(first[:3, :3].T @ second[:3, :3]) - 1) / 2, -1, 1))))


class PreparedRecordingBoundary:
    """Accept only a fresh, measured pregrasp-to-approach transition.

    Call update after the task machine consumes each observation, but before
    replacing any applied upper-body/base/hand targets. A start request allows
    frame zero to use the last actually applied action and measured state.
    """
    def __init__(self, settings, *, arm, object_path, thresholds, stationary_spec, attempt_start_time_s):
        self.settings = validate_recording_boundary(settings)
        if self.settings is None or arm not in {"left", "right"} or not object_path.startswith("/"):
            raise ValueError("Prepared recording requires explicit settings, arm and object path")
        self.arm, self.object_path = arm, object_path
        self.attempt_start_time_s = float(attempt_start_time_s)
        self.limits = {
            "tcp_position_m": float(thresholds["tcp_position_m"]),
            "tcp_rotation_rad": float(thresholds["tcp_rotation_rad"]),
            "max_object_com_linear_m_s": float(stationary_spec["max_object_linear_m_s"]),
            "max_object_com_angular_rad_s": float(stationary_spec["max_object_angular_rad_s"]),
            "max_object_pose_linear_m_s": .015,
            "max_object_pose_angular_rad_s": float(stationary_spec["max_object_angular_rad_s"]),
            "max_base_linear_m_s": float(stationary_spec["max_base_linear_m_s"]),
            "max_base_yaw_rad_s": float(stationary_spec["max_base_yaw_rad_s"]),
            "max_hand_contact_force_n": .05,
        }
        self.max_gap_s = float(thresholds["max_observation_gap_s"])
        self._previous_time = self._previous_object = self._ready_since = None
        self._metadata = None
        self._pending = None
        self.preparation = {"target_object_path": object_path, "samples": 0,
            "hand_contact_samples": 0, "sensors_unavailable_samples": 0,
            "max_hand_contact_force_n": 0., "peak_contact": None,
            "observed_from_time_s": None, "observed_through_time_s": None}

    def _contacts(self, observation, stage_id, now):
        available, absent, any_contact = True, True, False
        for arm in ("left", "right"):
            item = observation.get("contact_evidence", {}).get(arm, {}).get(self.object_path, {})
            valid = (isinstance(item, Mapping) and item.get("available") is True
                and item.get("source") == "physx_filtered_hand_object_contacts"
                and item.get("arm") == arm and item.get("object_path") == self.object_path)
            force = None
            if valid:
                try:
                    forces = np.asarray(item["forces_w"], dtype=float)
                    paths = item["sensor_paths"]
                    valid = (forces.ndim == 3 and forces.shape[0] == len(paths)
                        and forces.shape[0] > 0 and forces.shape[1] > 0
                        and forces.shape[2] == 3 and np.isfinite(forces).all())
                    if valid:
                        force = float(np.linalg.norm(forces, axis=-1).sum())
                except (KeyError, ValueError, TypeError):
                    valid = False
            valid = bool(valid)
            available = available and valid
            if valid:
                touched = force > self.limits["max_hand_contact_force_n"]
                any_contact = any_contact or touched
                absent = absent and not touched and item.get("contact_confirmed") is False
                if force > self.preparation["max_hand_contact_force_n"]:
                    self.preparation["max_hand_contact_force_n"] = force
                    self.preparation["peak_contact"] = {"time_s": now, "stage_id": stage_id,
                        "arm": arm, "force_magnitude_n": force,
                        "hand_open": observation.get("hand_open", {}).get(arm),
                        "pinch_confirmed": item.get("pinch_confirmed"),
                        "sensor_paths": deepcopy(item["sensor_paths"])}
            else:
                absent = False
        self.preparation["hand_contact_samples"] += int(any_contact)
        self.preparation["sensors_unavailable_samples"] += int(not available)
        return available, absent

    def update(self, observation, command, *, previous_stage_id, pregrasp_goal_pose_w):
        if self._metadata is not None:
            return {"status": "started", "start_recording": False, "metadata": self.metadata}
        now = float(observation["time_s"])
        if not math.isfinite(now) or self._previous_time is not None and now <= self._previous_time:
            raise ValueError("Recording readiness observation times must increase strictly")
        self.preparation["samples"] += 1
        if self.preparation["observed_from_time_s"] is None:
            self.preparation["observed_from_time_s"] = now
        self.preparation["observed_through_time_s"] = now
        available, absent = self._contacts(observation, previous_stage_id, now)
        evidence = {"valid": False, "continuous_dwell_s": 0.,
            "required_dwell_s": self.settings["readiness_dwell_s"],
            "hand_open": observation.get("hand_open", {}).get(self.arm) is True
                and observation.get("hand_closed", {}).get(self.arm) is False,
            "same_object_contacts_absent": absent, "sensors_available": available,
            "tcp_position_error_m": None, "tcp_rotation_error_rad": None,
            "object_com_linear_speed_m_s": None, "object_com_angular_speed_rad_s": None,
            "object_pose_linear_speed_m_s": None, "object_pose_angular_speed_rad_s": None,
            "base_linear_speed_m_s": None, "base_yaw_speed_rad_s": None,
            "limits": deepcopy(self.limits)}
        pose = None
        try:
            pose = pose_to_matrix(observation["object_poses_w"][self.object_path])
            state = observation["object_states"][self.object_path]
            if state["velocity_frame"] != "world_com_linear_angular":
                raise ValueError("Object velocity must use measured world COM frame")
            linear, angular, _ = _speed(state["velocity_w"])
            base_linear, _, base_yaw = _speed(observation["base_velocity_w"])
            evidence.update(object_com_linear_speed_m_s=linear,
                object_com_angular_speed_rad_s=angular,
                base_linear_speed_m_s=base_linear, base_yaw_speed_rad_s=base_yaw)
            if self._previous_time is not None and self._previous_object is not None:
                dt = now - self._previous_time
                if dt <= self.max_gap_s:
                    distance, angle = _pose_errors(self._previous_object, pose)
                    evidence.update(object_pose_linear_speed_m_s=distance / dt,
                                    object_pose_angular_speed_rad_s=angle / dt)
            pairs = (("object_com_linear_speed_m_s", "max_object_com_linear_m_s"),
                     ("object_com_angular_speed_rad_s", "max_object_com_angular_rad_s"),
                     ("object_pose_linear_speed_m_s", "max_object_pose_linear_m_s"),
                     ("object_pose_angular_speed_rad_s", "max_object_pose_angular_rad_s"),
                     ("base_linear_speed_m_s", "max_base_linear_m_s"),
                     ("base_yaw_speed_rad_s", "max_base_yaw_rad_s"))
            stable = (evidence["hand_open"] and absent and available and not observation.get("faults")
                and all(evidence[key] is not None and evidence[key] <= self.limits[bound] for key, bound in pairs))
        except (KeyError, ValueError, TypeError, OverflowError):
            stable = False
        self._previous_time, self._previous_object = now, pose
        if stable:
            if self._ready_since is None:
                self._ready_since = now
            evidence["continuous_dwell_s"] = now - self._ready_since
        else:
            self._ready_since = None
        fresh = (previous_stage_id == "pregrasp" and command.get("stage_id") == "approach"
                 and command.get("status") == "running")
        if fresh:
            try:
                goal = pose_to_matrix(pregrasp_goal_pose_w)
                measured = pose_to_matrix(observation["tcp_poses_w"][self.arm])
                distance, angle = _pose_errors(goal, measured)
                evidence.update(tcp_position_error_m=distance, tcp_rotation_error_rad=angle)
                evidence["valid"] = bool(stable
                    and evidence["continuous_dwell_s"] >= self.settings["readiness_dwell_s"] - 1e-10
                    and distance <= self.limits["tcp_position_m"]
                    and angle <= self.limits["tcp_rotation_rad"])
            except (KeyError, ValueError, TypeError):
                pass
        decision = {"status": "ready" if fresh and evidence["valid"] else "failed" if fresh else "preparing",
            "start_recording": fresh and evidence["valid"],
            "reason": "recording_boundary_not_ready" if fresh and not evidence["valid"] else None,
            "readiness_evidence": evidence}
        if decision["start_recording"]:
            self._pending = {"schema_version": "igris_c_prepared_recording_boundary_v1",
                "kind": self.settings["kind"], "started": True,
                "start_stage": "approach", "start_observation_time_s": now,
                "attempt_start_observation_time_s": self.attempt_start_time_s,
                "preparation_duration_s": now - self.attempt_start_time_s,
                "measured_tcp_pose_w": deepcopy(observation["tcp_poses_w"][self.arm]),
                "target_pregrasp_pose_w": deepcopy(pregrasp_goal_pose_w),
                "object_pose_w": deepcopy(observation["object_poses_w"][self.object_path]),
                "readiness_evidence": deepcopy(evidence),
                "preparation_contact_summary": deepcopy(self.preparation),
                "preparation_trace": "trace.jsonl",
                "world_or_joint_state_written_at_boundary": False,
                "object_attachment_created_at_boundary": False}
        return decision

    def mark_started(self, sim_time_s):
        if self._metadata is not None or self._pending is None:
            raise RuntimeError("Recording must begin exactly once at a verified fresh approach transition")
        time = float(sim_time_s)
        if not math.isfinite(time):
            raise ValueError("Recording start requires the actual finite simulation clock")
        self._metadata = {**self._pending, "start_sim_time_s": time}
        return self.metadata

    @property
    def metadata(self):
        if self._metadata is not None:
            return deepcopy(self._metadata)
        return {"schema_version": "igris_c_prepared_recording_boundary_v1",
            "kind": self.settings["kind"], "started": False,
            "preparation_trace": "trace.jsonl",
            "preparation_contact_summary": deepcopy(self.preparation)}


class HeldRecordingBoundary(PreparedRecordingBoundary):
    """Begin transfer only after an actual retained and unsupported grasp hold.

    The state restore is explicitly outside this boundary; this class performs
    no state writes and retains the preparation trace, contacts and capture.
    """
    def __init__(self, settings, **kwargs):
        super().__init__(settings, **kwargs)
        if self.settings["kind"] != "after_measured_grasp_hold":
            raise ValueError("Held recording requires after_measured_grasp_hold")
        self.stationary_spec = deepcopy(kwargs["stationary_spec"])

    def update(self, observation, command, *, previous_stage_id, pregrasp_goal_pose_w=None):
        from .stationary_lift import _bilateral_contact
        if self._metadata is not None:
            return {"status": "started", "start_recording": False, "metadata": self.metadata}
        now = float(observation["time_s"])
        if not math.isfinite(now) or self._previous_time is not None and now <= self._previous_time:
            raise ValueError("Recording readiness observation times must increase strictly")
        self._previous_time = now
        self.preparation["samples"] += 1
        if self.preparation["observed_from_time_s"] is None:
            self.preparation["observed_from_time_s"] = now
        self.preparation["observed_through_time_s"] = now
        available, _ = self._contacts(observation, previous_stage_id, now)
        evidence = deepcopy(command.get("preparation_readiness") or {})
        fresh = (previous_stage_id == "navigate_target" and command.get("stage_id") == "transfer"
                 and command.get("status") == "running")
        valid = (fresh and evidence.get("valid") is True and available
            and _bilateral_contact(observation, self.arm, self.object_path)
            and observation.get("hand_closed", {}).get(self.arm) is True
            and observation.get("hand_open", {}).get(self.arm) is False
            and not observation.get("faults")
            and evidence.get("physical_grasp_retained") is True
            and evidence.get("no_jig_support") is True
            and evidence.get("continuous_dwell_s", -1) >= self.settings["readiness_dwell_s"] - 1e-10)
        evidence["valid"] = bool(valid)
        evidence["contact_evidence"] = deepcopy(observation.get("contact_evidence", {}).get(self.arm, {}).get(self.object_path))
        evidence["limits"] = {**deepcopy(self.limits),
            "minimum_initial_clearance_m": self.stationary_spec["minimum_initial_clearance_m"],
            "max_anchor_position_error_m": .005, "max_anchor_rotation_error_rad": .05,
            "maximum_jig_support_force_n": .05}
        decision = {"status": "ready" if valid else "failed" if fresh else "preparing",
            "start_recording": bool(valid),
            "reason": "recording_boundary_not_ready" if fresh and not valid else None,
            "readiness_evidence": evidence}
        if valid:
            self._pending = {"schema_version": "igris_c_prepared_recording_boundary_v1",
                "kind": self.settings["kind"], "started": True,
                "start_stage": "transfer", "start_observation_time_s": now,
                "attempt_start_observation_time_s": self.attempt_start_time_s,
                "preparation_duration_s": now - self.attempt_start_time_s,
                "measured_tcp_pose_w": deepcopy(observation["tcp_poses_w"][self.arm]),
                "object_pose_w": deepcopy(observation["object_poses_w"][self.object_path]),
                "capture_object_to_tcp": deepcopy(evidence["capture_object_to_tcp"]),
                "readiness_evidence": evidence,
                "preparation_contact_summary": deepcopy(self.preparation),
                "preparation_trace": "trace.jsonl", "pickup_demonstrated": False,
                "world_or_joint_state_written_at_boundary": False,
                "object_attachment_created_at_boundary": False}
        return decision


def upright_readiness_limits(stationary_spec):
    """Acceptance bounds recorded with, and enforced by, the upright-stance boundary.

    The source-object bounds (enforced by the machine, re-checked by admission)
    keep the CRS at its first measured, initialized cart-slot pose until the
    jaws close, like the 5 mm / 0.05 rad hold anchor.
    """
    return {"max_object_com_linear_m_s": float(stationary_spec["max_object_linear_m_s"]),
            "max_object_com_angular_rad_s": float(stationary_spec["max_object_angular_rad_s"]),
            "max_object_pose_linear_m_s": .015,
            "max_object_pose_angular_rad_s": float(stationary_spec["max_object_angular_rad_s"]),
            "max_base_linear_m_s": float(stationary_spec["max_base_linear_m_s"]),
            "max_base_yaw_rad_s": float(stationary_spec["max_base_yaw_rad_s"]),
            "max_hand_contact_force_n": .05,
            "upright_joint_tolerance_rad": float(stationary_spec["upright_joint_tolerance_rad"]),
            "maximum_pelvis_tilt_rad": float(stationary_spec["maximum_pelvis_tilt_rad"]),
            "max_source_object_position_error_m": .005, "max_source_object_rotation_error_rad": .05}


class UprightRecordingBoundary(PreparedRecordingBoundary):
    """Begin recording at the first commanded motion out of a measured upright stance.

    The machine certifies upright posture, pelvis tilt and dwell during the
    unrecorded navigate_source stage; this gate re-checks the same observation
    independently (both hands free of the CRS, task hand open, no faults, quiet
    object and base) and requires the fresh navigate_source -> reach_posture
    transition. The pickup itself is recorded; no state is written here.
    """
    _PAIRS = (("object_com_linear_speed_m_s", "max_object_com_linear_m_s"),
              ("object_com_angular_speed_rad_s", "max_object_com_angular_rad_s"),
              ("object_pose_linear_speed_m_s", "max_object_pose_linear_m_s"),
              ("object_pose_angular_speed_rad_s", "max_object_pose_angular_rad_s"),
              ("base_linear_speed_m_s", "max_base_linear_m_s"),
              ("base_yaw_speed_rad_s", "max_base_yaw_rad_s"))

    def __init__(self, settings, **kwargs):
        super().__init__(settings, **kwargs)
        if self.settings["kind"] != "after_measured_upright_stance":
            raise ValueError("Upright recording requires after_measured_upright_stance")
        self.stationary_spec = deepcopy(kwargs["stationary_spec"])
        self.limits = upright_readiness_limits(self.stationary_spec)

    def update(self, observation, command, *, previous_stage_id, pregrasp_goal_pose_w=None):
        if self._metadata is not None:
            return {"status": "started", "start_recording": False, "metadata": self.metadata}
        now = float(observation["time_s"])
        if not math.isfinite(now) or self._previous_time is not None and now <= self._previous_time:
            raise ValueError("Recording readiness observation times must increase strictly")
        self.preparation["samples"] += 1
        if self.preparation["observed_from_time_s"] is None:
            self.preparation["observed_from_time_s"] = now
        self.preparation["observed_through_time_s"] = now
        available, absent = self._contacts(observation, previous_stage_id, now)
        machine = command.get("preparation_readiness")
        machine = deepcopy(dict(machine)) if isinstance(machine, Mapping) else None
        dwell = None if machine is None else machine.get("continuous_dwell_s")
        if isinstance(dwell, bool) or not isinstance(dwell, (int, float)) or not math.isfinite(dwell):
            dwell = None
        # The machine's own readiness stays nested; top-level fields are this
        # gate's independent measurements of the same observation.
        evidence = {"valid": False, "machine_readiness": machine,
            "machine_readiness_valid": machine is not None and machine.get("valid") is True,
            "continuous_dwell_s": dwell, "required_dwell_s": self.settings["readiness_dwell_s"],
            "hand_open": observation.get("hand_open", {}).get(self.arm) is True
                and observation.get("hand_closed", {}).get(self.arm) is False,
            "same_object_contacts_absent": absent, "sensors_available": available,
            **{key: None for key, _ in self._PAIRS},
            "contact_evidence": {arm: deepcopy(observation.get("contact_evidence", {}).get(arm, {}).get(self.object_path))
                                 for arm in ("left", "right")},
            "limits": deepcopy(self.limits)}
        pose = None
        try:
            pose_to_matrix(observation["tcp_poses_w"][self.arm])
            pose = pose_to_matrix(observation["object_poses_w"][self.object_path])
            state = observation["object_states"][self.object_path]
            if state["velocity_frame"] != "world_com_linear_angular":
                raise ValueError("Object velocity must use measured world COM frame")
            linear, angular, _ = _speed(state["velocity_w"])
            base_linear, _, base_yaw = _speed(observation["base_velocity_w"])
            evidence.update(object_com_linear_speed_m_s=linear, object_com_angular_speed_rad_s=angular,
                            base_linear_speed_m_s=base_linear, base_yaw_speed_rad_s=base_yaw)
            if self._previous_time is not None and self._previous_object is not None:
                dt = now - self._previous_time
                if dt <= self.max_gap_s:
                    distance, angle = _pose_errors(self._previous_object, pose)
                    evidence.update(object_pose_linear_speed_m_s=distance / dt,
                                    object_pose_angular_speed_rad_s=angle / dt)
            quiet = all(evidence[key] is not None and evidence[key] <= self.limits[bound]
                        for key, bound in self._PAIRS)
        except (KeyError, ValueError, TypeError, OverflowError):
            quiet = False
        self._previous_time, self._previous_object = now, pose
        fresh = (previous_stage_id == "navigate_source" and command.get("stage_id") == "reach_posture"
                 and command.get("status") == "running")
        valid = bool(fresh and evidence["machine_readiness_valid"]
                     and dwell is not None and dwell >= self.settings["readiness_dwell_s"] - 1e-10
                     and available and absent and evidence["hand_open"]
                     and not observation.get("faults") and quiet)
        evidence["valid"] = valid
        decision = {"status": "ready" if valid else "failed" if fresh else "preparing",
            "start_recording": valid,
            "reason": "recording_boundary_not_ready" if fresh and not valid else None,
            "readiness_evidence": evidence}
        if valid:
            self._pending = {"schema_version": "igris_c_prepared_recording_boundary_v1",
                "kind": self.settings["kind"], "started": True,
                "start_stage": "reach_posture", "start_observation_time_s": now,
                "attempt_start_observation_time_s": self.attempt_start_time_s,
                "preparation_duration_s": now - self.attempt_start_time_s,
                "measured_tcp_pose_w": deepcopy(observation["tcp_poses_w"][self.arm]),
                "object_pose_w": deepcopy(observation["object_poses_w"][self.object_path]),
                "readiness_evidence": deepcopy(evidence),
                "preparation_contact_summary": deepcopy(self.preparation),
                "preparation_trace": "trace.jsonl", "pickup_demonstrated": True,
                "world_or_joint_state_written_at_boundary": False,
                "object_attachment_created_at_boundary": False}
        return decision


_BOUNDARY_TYPES = {"after_measured_pregrasp": PreparedRecordingBoundary,
                   "after_measured_grasp_hold": HeldRecordingBoundary,
                   "after_measured_upright_stance": UprightRecordingBoundary}


def recording_boundary_type(settings):
    """Select the boundary class by its validated kind, never by task identity."""
    validated = validate_recording_boundary(settings)
    if validated is None:
        raise ValueError("A recording boundary type requires explicit boundary settings")
    return _BOUNDARY_TYPES[validated["kind"]]
