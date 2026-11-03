"""Independent, physically released CRS-to-JIG placement from a declared held start.

The held state is an episode initialization, never a demonstrated pickup. Actual
bilateral contact, capture retention and measured stability qualify it before
recording. The task machine only emits commands; it never writes physical state.
"""
from collections.abc import Mapping
from copy import deepcopy
import math

import numpy as np

from .nepes_tasks import NepesTaskMachine, _errors
from .object_symmetry import symmetric_capture_errors, symmetric_pose_errors
from .snapshot import matrix_to_pose, pose_to_matrix
from .stationary_lift import _bilateral_contact

PLACEMENT_CONTRACT_VERSION = "igris_c_crs_jig_placement_v1"
_IDENTITY = {"schema_version": "igris_c_nepes_stationary_task_v1", "number": 2,
    "id": "crs_load_jig", "legacy_task_id": "crs_load_jig", "goal": "load_jig",
    "require_hugwbc": True, "independent_scene": True}
_DEFAULTS = {"preparation_dwell_s": .5, "settle_dwell_s": 1.,
    "max_object_linear_m_s": .02, "max_object_angular_rad_s": .12,
    "max_base_linear_m_s": .03, "max_base_yaw_rad_s": .12,
    "slot_position_tolerance_m": .008, "slot_normal_position_tolerance_m": .0025,
    "slot_rotation_tolerance_rad": .05,
    "maximum_jig_position_drift_m": .005, "maximum_jig_rotation_drift_rad": .05,
    "release_separation_m": .015, "minimum_retreat_m": .04,
    "minimum_initial_clearance_m": .02}
# The authored JIG notches are 10 mm wide and engage only the 3.75 mm CRS sectors
# (the 2.25 mm pins lie in the entry taper), so a released ring resting against a
# notch wall sits 3.0 / 3.25 mm off the sector-centred target (exact-prism audit,
# planning/exact_prisms_vs_raw_v3.json). A profile may therefore widen the slot
# normal acceptance up to this measured play; the default stays 2.5 mm.
SLOT_NORMAL_TOLERANCE_MAXIMUM_M = .004
PLACEMENT_BOUNDS = {**_DEFAULTS, "slot_normal_position_tolerance_m": SLOT_NORMAL_TOLERANCE_MAXIMUM_M}


def validate_place_settings(spec, task):
    if not isinstance(spec, Mapping) or set(spec) - set(_IDENTITY) - set(_DEFAULTS):
        raise ValueError("Unknown stationary placement settings")
    result = {**_IDENTITY, **_DEFAULTS, **deepcopy(dict(spec))}
    for key, value in _IDENTITY.items():
        if type(result[key]) is not type(value) or result[key] != value:
            raise ValueError(f"stationary_task.{key} must be {value!r}")
    for key, value in PLACEMENT_BOUNDS.items():
        current = result[key]
        if isinstance(current, bool) or not isinstance(current, (int, float)) or not math.isfinite(current) or current <= 0:
            raise ValueError(f"stationary_task.{key} must be positive and finite")
        result[key] = float(current)
        if (key.startswith(("max_", "maximum_", "slot_")) and current > value
                or key.startswith(("minimum_",)) and current < value):
            raise ValueError(f"stationary_task.{key} cannot relax the placement acceptance bound")
    if result["preparation_dwell_s"] < .5 or result["settle_dwell_s"] < 1.:
        raise ValueError("Placement requires 0.5 s physical preparation and 1 s released settling")
    if not .005 <= result["release_separation_m"] <= .03:
        raise ValueError("Release separation must be a bounded 5-30 mm movement")
    if np.linalg.norm(task["retreat_offset_w"]) < result["minimum_retreat_m"]:
        raise ValueError("Retreat target must clear the released CRS by the required distance")
    if task.get("target", {}).get("frame") != "jig" or task.get("placement_support", {}).get("object") != "jig":
        raise ValueError("Task2 requires the original named JIG for both target and measured support")
    if task.get("requires_completed") or task.get("requires_events"):
        raise ValueError("Independent placement cannot require preceding task execution")
    return result


def _contact_absent(raw, arm, object_path):
    item = raw.get("contact_evidence", {}).get(arm, {}).get(object_path, {})
    try:
        force = np.asarray(item["forces_w"], dtype=float)
        paths = item["sensor_paths"]
        return bool(item.get("available") is True and item.get("arm") == arm
            and item.get("object_path") == object_path
            and item.get("source") == "physx_filtered_hand_object_contacts"
            and item.get("contact_confirmed") is False
            and force.ndim == 3 and force.shape[0] == len(paths) and len(paths) > 0
            and force.shape[1] > 0 and force.shape[2] == 3 and np.isfinite(force).all()
            and np.linalg.norm(force, axis=-1).sum() <= .05)
    except (KeyError, TypeError, ValueError):
        return False


class StationaryPlaceMachine(NepesTaskMachine):
    """Held-object preparation, insertion, supported opening, separation and settle."""
    def __init__(self, config, legacy_task_id="crs_load_jig", *, seed=0,
                 completed_tasks=(), goal_settings=None):
        if legacy_task_id != "crs_load_jig":
            raise ValueError("StationaryPlaceMachine implements only task2 CRS JIG loading")
        super().__init__(config, legacy_task_id, seed=seed, completed_tasks=completed_tasks)
        self.goal_settings = validate_place_settings(config.get("stationary_task", {})
            if goal_settings is None else goal_settings, self.task)
        if config["thresholds"]["capture_position_m"] > .025 or config["thresholds"]["capture_rotation_rad"] > .2:
            raise ValueError("Original 25 mm / 0.2 rad capture retention must remain")
        self.stages = ("navigate_target", "transfer", "place", "release", "retreat", "settle")
        self.stage_id, self._stage_index = self.stages[0], 0
        self.jig_path = self.config["objects"]["jig"]
        self._slot_pose = pose_to_matrix(self.task["target"]["pose"])
        self._slot_pose[:3, 3] += self._jitter
        self._jig_anchor = self._release_origin = self._preparation_anchor = None
        self._preparation_anchor_time_s = None
        self._capture_time_s = None
        self._release_phase = "opening"
        self._release_subphase_events = []
        self._clearance_completion = None
        self._readiness = None
        self._metrics = None
        self._raw = None
        self._stage_times = {}
        self._recording_ready = False

    def _read_observation(self, raw):
        observed = super()._read_observation(raw)
        observed["bilateral"] = _bilateral_contact(raw, self.arm, self.object_path)
        observed["closed"] = observed["closed"] and observed["bilateral"]
        return observed

    def _grasp_valid(self, observed):
        return bool(observed.get("bilateral") and super()._grasp_valid(observed))

    def _support(self, raw):
        item = raw.get("support_evidence", {}).get(self.object_path, {})
        result = {"available": False, "object_path": self.object_path, "support_path": self.jig_path,
            "source": item.get("source"), "normal_force_n": None,
            "object_mass_kg": None, "minimum_force_n": None, "load_supported": False}
        try:
            force = np.asarray(item["force_w"], float)
            mass = float(item["object_mass_kg"])
            valid = (item.get("available") is True and item.get("object_path") == self.object_path
                and item.get("support_path") == self.jig_path
                and item.get("source") == "physx_filtered_object_support_contacts"
                and force.shape == (3,) and np.isfinite(force).all() and math.isfinite(mass) and mass > 0)
            if valid:
                normal_force = float(np.dot(force, self.task["placement_support"]["normal_w"]))
                minimum = self.task["placement_support"]["minimum_load_fraction"] * mass * 9.81
                result.update(available=True, normal_force_n=normal_force, object_mass_kg=mass,
                    minimum_force_n=minimum, load_supported=normal_force >= minimum)
        except (KeyError, ValueError, TypeError, OverflowError):
            pass
        return result

    def _measure(self, raw, obs):
        cfg = self.goal_settings
        object_pose, jig = obs["objects"][self.object_path], obs["objects"][self.jig_path]
        relative = np.linalg.inv(jig) @ object_pose
        position, rotation = _errors(relative, self._slot_pose)
        spin = capture_spin = None
        if self._symmetry_axis is not None:
            # Declared ring symmetry: the slot rotation error is the ring-axis error only; the spin about
            # the axis (slot frame, and inside the hand against the original capture) is reported separately.
            position, rotation, spin = symmetric_pose_errors(relative, self._slot_pose, self._symmetry_axis)
            if self._object_to_tcp is not None and obs["closed"]:
                # The spin inside the closed pads against the original capture (meaningless once released).
                capture_spin = symmetric_capture_errors(np.linalg.inv(object_pose) @ obs["tcp"], self._object_to_tcp,
                                                       self._symmetry_axis, self._symmetry_pivot_radius_m)[2]
        jig_position, jig_rotation = _errors(jig, self._jig_anchor)
        linear = angular = base_linear = base_yaw = None
        if obs["object_velocity"] is not None:
            linear, angular = (float(np.linalg.norm(obs["object_velocity"][:3])),
                               float(np.linalg.norm(obs["object_velocity"][3:])))
        if obs["base_velocity"] is not None:
            base_linear = float(np.linalg.norm(obs["base_velocity"][:3]))
            base_yaw = abs(float(obs["base_velocity"][5]))
        support = self._support(raw)
        retreat = None if self._release_origin is None else float(np.linalg.norm(obs["tcp"][:3, 3] - self._release_origin[:3, 3]))
        clearance_offset = self.task.get("release_clearance_offset_w")
        clearance = None
        if clearance_offset is not None:
            direction = np.asarray(clearance_offset, float)
            length = float(np.linalg.norm(direction))
            projected = (None if self._release_origin is None else
                float(np.dot(obs["tcp"][:3, 3] - self._release_origin[:3, 3], direction / length)))
            clearance = {"offset_w": list(clearance_offset), "required_projected_m": length - .0005,
                "projected_distance_m": projected, "completion": deepcopy(self._clearance_completion),
                "current_phase": self._release_phase, "continuous_dwell_s": 0.,
                "contact_free_dwell_s": self.config["thresholds"]["contact_dwell_s"]}
        gates = {
            "slot_pose_reached": position <= cfg["slot_position_tolerance_m"] and rotation <= cfg["slot_rotation_tolerance_rad"],
            "slot_normal_reached": abs(float(relative[1, 3] - self._slot_pose[1, 3])) <= cfg["slot_normal_position_tolerance_m"],
            "jig_anchor_retained": jig_position <= cfg["maximum_jig_position_drift_m"] and jig_rotation <= cfg["maximum_jig_rotation_drift_rad"],
            "jig_support_confirmed": support["load_supported"],
            "hand_open": obs["open"] and not obs["closed"],
            "same_object_contacts_absent": all(_contact_absent(raw, arm, self.object_path) for arm in ("left", "right")),
            "retreat_clearance": retreat is not None and retreat >= cfg["minimum_retreat_m"],
            "object_com_stationary": linear is not None and linear <= cfg["max_object_linear_m_s"] and angular <= cfg["max_object_angular_rad_s"],
            "object_pose_stationary": self._linear_speed is not None and self._linear_speed <= .015 and self._angular_speed <= cfg["max_object_angular_rad_s"],
            "base_stationary": base_linear is not None and base_linear <= cfg["max_base_linear_m_s"] and base_yaw <= cfg["max_base_yaw_rad_s"],
        }
        return {"placement_contract_version": PLACEMENT_CONTRACT_VERSION,
            "measurement_time_s": obs["time_s"], "slot_relative_pose": matrix_to_pose(relative),
            "target_slot_relative_pose": matrix_to_pose(self._slot_pose),
            "slot_position_error_m": position, "slot_rotation_error_rad": rotation,
            "object_spin_rad": spin, "capture_spin_rad": capture_spin,
            "slot_normal_position_error_m": abs(float(relative[1, 3] - self._slot_pose[1, 3])),
            "jig_anchor_pose_w": matrix_to_pose(self._jig_anchor),
            "jig_position_drift_m": jig_position, "jig_rotation_drift_rad": jig_rotation,
            "object_com_linear_speed_m_s": linear, "object_com_angular_speed_rad_s": angular,
            "object_pose_linear_speed_m_s": self._linear_speed, "object_pose_angular_speed_rad_s": self._angular_speed,
            "base_linear_speed_m_s": base_linear, "base_yaw_speed_rad_s": base_yaw,
            "retreat_distance_m": retreat, "support_evidence": support,
            "release_clearance": clearance, "release_subphase_events": deepcopy(self._release_subphase_events),
            "capture_time_s": self._capture_time_s,
            "capture_object_to_tcp": None if self._object_to_tcp is None else matrix_to_pose(self._object_to_tcp),
            "stage_entry_times_s": deepcopy(self._stage_times), "pickup_demonstrated": False,
            "gates": gates, "all_conditions_met": all(gates.values()),
            "required_dwell_s": cfg["settle_dwell_s"], "continuous_dwell_s": 0.,
            "fixed_acceptance_bounds": {"max_object_pose_linear_m_s": .015, "max_hand_contact_force_n": .05}}

    def _retreat_goal(self, fraction=1.):
        goal = self._release_origin.copy()
        goal[:3, 3] += np.asarray(self.task.get("release_clearance_offset_w", [0., 0., 0.]))
        goal[:3, 3] += np.asarray(self.task["retreat_offset_w"]) * fraction
        if "retreat_orientation_wxyz" in self.task:
            first = np.asarray(matrix_to_pose(self._release_origin)[3:])
            second = np.asarray(self.task["retreat_orientation_wxyz"], float)
            dot = float(np.dot(first, second))
            if dot < 0:
                second, dot = -second, -dot
            if dot > .9995:
                quaternion = first + fraction * (second - first)
                quaternion /= np.linalg.norm(quaternion)
            else:
                angle = math.acos(np.clip(dot, -1., 1.))
                quaternion = (math.sin((1 - fraction) * angle) * first
                    + math.sin(fraction * angle) * second) / math.sin(angle)
            goal[:3, :3] = pose_to_matrix([0., 0., 0., *quaternion])[:3, :3]
        return goal

    def _transition(self, obs):
        self._stage_index += 1
        self.stage_id = self.stages[self._stage_index]
        self._stage_start, self._condition_since, self._lost_grasp_since = obs["time_s"], None, None
        self._stage_times[self.stage_id] = obs["time_s"]
        if self.stage_id in {"transfer", "place"}:
            self._set_control_grasp(obs["objects"][self.object_path], obs["tcp"])
            self._tcp_target = self._target_object @ self._control_object_to_tcp
            if self.stage_id == "transfer":
                self._tcp_target[:3, 3] += self.task["transfer_offset_w"]
        elif self.stage_id == "release":
            self._release_origin = obs["tcp"].copy()
            self._supported_release_tcp = self._release_origin.copy()
            self._tcp_target = self._release_origin.copy()
            self._release_subphase_events.append({"phase": "opening", "time_s": obs["time_s"]})
        elif self.stage_id in {"retreat", "settle"}:
            self._tcp_target = self._retreat_goal()
        self.history.append({"time_s": obs["time_s"], "stage_id": self.stage_id, "status": "running"})

    def step(self, raw):
        if self.status != "running":
            return self._command()
        if raw.get("faults"):
            return self.fail("runtime_fault:" + ";".join(map(str, raw["faults"])))
        if raw.get("unreachable") is True:
            return self.fail("target_unreachable")
        try:
            obs = self._read_observation(raw)
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            return self.fail(f"invalid_observation:{exc}")
        now, obj = obs["time_s"], obs["objects"][self.object_path]
        if self._previous_time is not None:
            dt = now - self._previous_time
            if dt > self.config["thresholds"]["max_observation_gap_s"]:
                return self.fail("observation_gap")
            distance, angle = _errors(obj, self._previous_object)
            self._linear_speed, self._angular_speed = distance / dt, angle / dt
        self._previous_time, self._previous_object = now, obj.copy()
        self._last_observation, self._raw = obs, raw
        if self._start_time is None:
            self._start_time = self._stage_start = now
            self._tcp_target = obs["tcp"].copy()
            self._source_object = obj.copy()
            self._jig_anchor = obs["objects"][self.jig_path].copy()
            self._target_object = self._jig_anchor @ self._slot_pose
            self._stage_times[self.stage_id] = now
            self.history.append({"time_s": now, "stage_id": self.stage_id, "status": "running"})
        if now - self._start_time >= self.task["timeout_s"]:
            return self.fail("task_timeout")
        if now - self._stage_start >= self.task["stage_timeouts_s"][self.stage_id]:
            return self.fail("stage_timeout:" + self.stage_id)
        if self.stage_id == "navigate_target" and self._object_to_tcp is None and obs["closed"] and obs["hand_settled"]:
            self._object_to_tcp = np.linalg.inv(obj) @ obs["tcp"]
            self._control_object_to_tcp = self._object_to_tcp.copy()
            self._capture_time_s = now
        self._metrics = self._measure(raw, obs)
        gates = self._metrics["gates"]
        if not gates["jig_anchor_retained"]:
            return self.fail("jig_displaced")
        threshold, cfg = self.config["thresholds"], self.goal_settings
        condition, duration = False, threshold["reach_dwell_s"]
        if self.stage_id == "navigate_target":
            stable = all(gates[k] for k in ("object_com_stationary", "object_pose_stationary", "base_stationary"))
            clearance = float(obj[2, 3] - self._target_object[2, 3])
            retained = self._grasp_valid(obs)
            no_support = self._metrics["support_evidence"]["available"] and abs(self._metrics["support_evidence"]["normal_force_n"]) <= .05
            quiet = bool(stable and retained and no_support and clearance >= cfg["minimum_initial_clearance_m"])
            # Capture G is immutable from actual acquisition. World stationarity
            # instead measures one uninterrupted quiet window after WBC settles;
            # preparation motion cannot permanently poison that future window.
            anchor_position = anchor_rotation = None
            if quiet:
                if self._preparation_anchor is None:
                    self._preparation_anchor = obj.copy()
                    self._preparation_anchor_time_s = now
                anchor_position, anchor_rotation = _errors(obj, self._preparation_anchor)
            condition = bool(quiet and anchor_position <= .005 and anchor_rotation <= .05)
            if not condition:
                self._preparation_anchor = self._preparation_anchor_time_s = None
            duration = cfg["preparation_dwell_s"]
            self._readiness = {"valid": condition, "physical_grasp_retained": retained,
                "bilateral_contact_confirmed": obs["bilateral"], "hand_closed": obs["closed"],
                "no_jig_support": no_support, "jig_support_evidence": deepcopy(self._metrics["support_evidence"]),
                "sensors_available": obs["contact_sensor_available"] is True,
                "clearance_above_slot_m": clearance, "anchor_position_error_m": anchor_position,
                "anchor_rotation_error_rad": anchor_rotation,
                "stability_anchor_time_s": self._preparation_anchor_time_s,
                "stability_anchor_pose_w": None if self._preparation_anchor is None else matrix_to_pose(self._preparation_anchor),
                "required_dwell_s": duration,
                "continuous_dwell_s": 0., "measurement_time_s": now,
                "capture_time_s": self._capture_time_s,
                "capture_object_to_tcp": self._metrics["capture_object_to_tcp"],
                **{k: self._metrics[k] for k in ("object_com_linear_speed_m_s", "object_com_angular_speed_rad_s",
                    "object_pose_linear_speed_m_s", "object_pose_angular_speed_rad_s", "base_linear_speed_m_s", "base_yaw_speed_rad_s")}}
        elif self.stage_id in {"transfer", "place"}:
            self._support_active = self.stage_id == "place" and gates["slot_pose_reached"] and gates["slot_normal_reached"] and gates["jig_support_confirmed"]
            measured_capture = np.linalg.inv(obj) @ obs["tcp"]
            supported_retained = self._support_active and self._near(measured_capture, self._object_to_tcp, "capture") and not obs["open"]
            if self._grasp_valid(obs) or supported_retained:
                self._lost_grasp_since = None
            elif self._lost_grasp_since is None:
                self._lost_grasp_since = now
            elif now - self._lost_grasp_since >= threshold["lost_grasp_grace_s"]:
                return self.fail("grasp_lost")
            if self.stage_id == "transfer":
                condition = self._tracking_goal_near(obs) and self._grasp_valid(obs)
            else:
                condition = supported_retained and gates["object_com_stationary"] and gates["object_pose_stationary"]
                duration = self.task["placement_support"]["dwell_s"]
                if self._support_active:
                    if self._supported_release_tcp is None:
                        self._supported_release_tcp = obs["tcp"].copy()
                    self._tcp_target = self._supported_release_tcp.copy()
                elif self._supported_release_tcp is not None:
                    self._supported_release_tcp = None
                    self._tcp_target = self._target_object @ self._control_object_to_tcp
        elif self.stage_id == "release":
            if not gates["slot_pose_reached"] or not gates["slot_normal_reached"]:
                return self.fail("placement_displaced_during_release")
            if obs["contact_sensor_available"] is not True:
                return self.fail("contact_sensor_unavailable")
            if obs["open"] and self._release_phase == "opening":
                self._condition_since = None
                if "release_clearance_offset_w" in self.task:
                    self._release_phase = "clearing"
                    self._tcp_target = self._release_origin.copy()
                    self._tcp_target[:3, 3] += self.task["release_clearance_offset_w"]
                else:
                    self._release_phase = "separating"
                    fraction = cfg["release_separation_m"] / np.linalg.norm(self.task["retreat_offset_w"])
                    self._tcp_target = self._retreat_goal(fraction)
                self._release_subphase_events.append({"phase": self._release_phase, "time_s": now})
            if self._release_phase == "clearing":
                clearance = self._metrics["release_clearance"]
                clearance["gates"] = {key: gates[key] for key in ("hand_open", "same_object_contacts_absent",
                    "jig_support_confirmed", "slot_pose_reached", "slot_normal_reached")}
                clearance["gates"].update(
                    projected_clearance_reached=clearance["projected_distance_m"] >= clearance["required_projected_m"],
                    tcp_goal_reached=self._near(obs["tcp"], self._tcp_target))
                condition = all(clearance["gates"].values())
            else:
                # Lateral clearance never substitutes for the existing upward
                # separation distance. Its offset stays in every later goal.
                cleared_origin = self._release_origin[:3, 3] + np.asarray(
                    self.task.get("release_clearance_offset_w", [0., 0., 0.]))
                separation_distance = float(np.linalg.norm(obs["tcp"][:3, 3] - cleared_origin))
                condition = (self._release_phase == "separating" and gates["hand_open"]
                    and gates["same_object_contacts_absent"] and gates["jig_support_confirmed"]
                    and separation_distance >= cfg["release_separation_m"] - .003
                    and self._near(obs["tcp"], self._tcp_target))
            duration = threshold["contact_dwell_s"]
        elif self.stage_id in {"retreat", "settle"}:
            if not gates["hand_open"] or not gates["same_object_contacts_absent"]:
                return self.fail("release_lost")
            if not gates["slot_pose_reached"] or not gates["slot_normal_reached"]:
                return self.fail("placement_displaced_after_release")
            if self.stage_id == "retreat":
                condition = gates["retreat_clearance"] and self._near(obs["tcp"], self._tcp_target)
            else:
                condition, duration = all(gates.values()), cfg["settle_dwell_s"]
        passed = self._dwell(condition, now, duration)
        dwell = 0. if self._condition_since is None else now - self._condition_since
        if self.stage_id == "navigate_target":
            self._readiness["continuous_dwell_s"] = dwell
        if self.stage_id == "settle":
            self._metrics["continuous_dwell_s"] = dwell
        if self.stage_id == "release" and self._release_phase == "clearing":
            self._metrics["release_clearance"]["continuous_dwell_s"] = dwell
        if passed and self.stage_id == "release" and self._release_phase == "clearing":
            self._clearance_completion = {"time_s": now, "continuous_dwell_s": dwell,
                "projected_distance_m": self._metrics["release_clearance"]["projected_distance_m"],
                "measured_tcp_pose_w": matrix_to_pose(obs["tcp"]),
                "hand_open": gates["hand_open"], "same_object_contacts_absent": gates["same_object_contacts_absent"],
                "jig_support_confirmed": gates["jig_support_confirmed"],
                "slot_pose_reached": gates["slot_pose_reached"], "slot_normal_reached": gates["slot_normal_reached"]}
            self._release_phase = "separating"
            self._release_subphase_events.append({"phase": "separating", "time_s": now})
            self._condition_since = None
            fraction = cfg["release_separation_m"] / np.linalg.norm(self.task["retreat_offset_w"])
            self._tcp_target = self._retreat_goal(fraction)
            self._metrics["release_clearance"]["completion"] = deepcopy(self._clearance_completion)
            self._metrics["release_clearance"]["current_phase"] = "separating"
            self._metrics["release_subphase_events"] = deepcopy(self._release_subphase_events)
            return self._command()
        if passed:
            if self.stage_id == "settle":
                self.status = "succeeded"
                self.history.append({"time_s": now, "stage_id": self.stage_id, "status": self.status})
            else:
                if self.stage_id == "navigate_target":
                    self._recording_ready = True
                self._transition(obs)
        return self._command()

    def _command(self):
        command = super()._command()
        phase = "grasp_hold" if self.stage_id == "navigate_target" else self.stage_id
        command.update(base_goal_pose_w=None, stationary_phase=phase,
            goal_id="crs_load_jig", dataset_task_id="crs_load_jig", legacy_task_id="crs_load_jig",
            episode_scope="stationary_task", full_task_completed=self.status == "succeeded",
            legacy_full_task_completed=False, pickup_demonstrated=False,
            stationary_metrics=deepcopy(self._metrics), preparation_readiness=deepcopy(self._readiness),
            release_phase=self._release_phase if self.stage_id == "release" else None)
        if self.status == "running" and self.stage_id == "navigate_target":
            command.update(motion_mode="contact_servo", hand_command="close")
        return command

    def graph(self):
        graph = super().graph()
        graph.update(episode_scope="stationary_task", stationary_task=deepcopy(self.goal_settings),
            initial_condition="declared held-object boundary; measured physical bilateral readiness required",
            pickup_demonstrated=False, placement_contract_version=PLACEMENT_CONTRACT_VERSION)
        graph["stages"][0]["success_condition"] = "Actual bilateral grasp, original capture retention, above-slot clearance and stationary dwell"
        graph["stages"][-3]["success_condition"] = "Measured jaw opening followed by bounded separation, actual JIG support and no hand contact"
        graph["stages"][-1]["success_condition"] = "JIG-relative placement, JIG support, open/contact-free hand and measured retreat/stability for one second"
        return graph
