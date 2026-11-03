"""Measured acceptance for the independent CRS pick-and-lift dataset task.

The existing source alignment, approach, physical acquisition and lift execute
unchanged. The former target-navigation stage instead holds the measured TCP;
it sends no target base pose and requires sustained measured lift and grasp.
This module never moves simulated objects, writes joint states or adds joints.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from typing import Any

import numpy as np

from .nepes_tasks import MANIPULATION_STAGES, NepesTaskMachine
from .object_symmetry import pinch_point, symmetric_contact_errors, symmetric_pose_errors
from .snapshot import matrix_to_pose, pose_to_matrix


GOAL_ID = "crs_pick_lift"
LEGACY_TASK_ID = "crs_load_jig"
_HOLD_STAGE = "navigate_target"
HOLD_CONTRACT_VERSION = "igris_c_crs_lift_hold_v2"
# Native attempt 01 had median instantaneous COM speed 0.01645 m/s but median
# measured pose-stream speed 0.00078 m/s, and timed out under the 0.015 m/s COM
# contract. Independent pose-stream and immutable-anchor bounds accompany the
# revised COM limit; this new contract does not reclassify that attempt.
_MAX_POSE_LINEAR_M_S = .015
_MAX_HOLD_POSITION_DRIFT_M = .005
_MAX_HOLD_ROTATION_DRIFT_RAD = .05
_DEFAULTS = {
    "hold_dwell_s": 1.0,
    "minimum_lift_m": .08,
    "max_object_linear_m_s": .02,
    "max_object_angular_rad_s": .12,
    "max_base_linear_m_s": .03,
    "max_base_yaw_rad_s": .12,
}
_IDENTITY = {
    "schema_version": "igris_c_nepes_stationary_task_v1",
    "number": 1,
    "id": GOAL_ID,
    "legacy_task_id": LEGACY_TASK_ID,
    "goal": "lift_hold",
    "require_hugwbc": True,
    "independent_scene": True,
}


def _settings(value: Mapping[str, Any], task: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("stationary_task must be a mapping")
    unknown = set(value) - set(_DEFAULTS) - set(_IDENTITY)
    if unknown:
        raise ValueError("Unknown stationary lift settings: " + ", ".join(sorted(unknown)))
    settings = {**_IDENTITY, **_DEFAULTS, **deepcopy(dict(value))}
    for key, expected in _IDENTITY.items():
        if settings[key] != expected or type(settings[key]) is not type(expected):
            raise ValueError(f"stationary_task.{key} must be {expected!r}")
    for key in _DEFAULTS:
        value = settings[key]
        if isinstance(value, bool):
            raise ValueError(f"stationary_task.{key} must be positive and finite")
        try:
            value = float(value)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"stationary_task.{key} must be positive and finite") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"stationary_task.{key} must be positive and finite")
        settings[key] = value
    if settings["hold_dwell_s"] < 1.0:
        raise ValueError("Stationary lift requires at least one continuous second of hold")
    if not max(.08, task["min_lift_m"]) <= settings["minimum_lift_m"] <= task["lift_offset_w"][2]:
        raise ValueError("Stationary minimum lift must preserve the original minimum and fit the lift goal")
    for key in _DEFAULTS:
        if key.startswith("max_") and settings[key] > _DEFAULTS[key]:
            raise ValueError(f"stationary_task.{key} may tighten but not relax the stationary speed bound")
    if settings["hold_dwell_s"] >= task["stage_timeouts_s"][_HOLD_STAGE]:
        raise ValueError("Hold dwell must be shorter than the hold stage timeout")
    return settings


def _bilateral_contact(raw: Mapping[str, Any], arm: str, object_path: str) -> bool:
    """Require both actual named jaw bodies to press against this same object."""
    try:
        contact = raw["contact_evidence"][arm][object_path]
        if not isinstance(contact, Mapping) or not (
            contact.get("available") is True
            and contact.get("arm") == arm
            and contact.get("object_path") == object_path
            and contact.get("source") == "physx_filtered_hand_object_contacts"
            and contact.get("contact_confirmed") is True
            and contact.get("pinch_confirmed") is True
            and contact.get("bilateral_contact_confirmed") is True
            and contact.get("angle_within_closed_bound") is True
            and contact.get("pinch_source") == "same_object_two_jaw_filtered_forces_and_angle"
        ):
            return False
        paths = contact["sensor_paths"]
        forces = np.asarray(contact["forces_w"], dtype=float)
        threshold = float(contact["force_threshold_n"])
        if (not isinstance(paths, (list, tuple)) or forces.ndim != 3
                or forces.shape[0] != len(paths) or forces.shape[1] == 0 or forces.shape[2] != 3
                or not np.isfinite(forces).all() or not math.isfinite(threshold) or threshold <= 0):
            return False
        for name in (f"{arm.title()}_Hand", f"{arm[0]}_1_dof_gripper_moving"):
            indices = [index for index, path in enumerate(paths)
                       if isinstance(path, str) and path.rsplit("/", 1)[-1] == name]
            if len(indices) != 1 or np.linalg.norm(forces[indices[0]], axis=-1).sum() <= threshold:
                return False
        return True
    except (KeyError, ValueError, TypeError, OverflowError):
        return False


class StationaryLiftMachine(NepesTaskMachine):
    """Source alignment, physical CRS acquisition, lift, then a measured hold.

    ``task_id`` remains the legacy ID for runtime calibration lookups. Public
    dataset identity is ``dataset_task_id`` / ``goal_id``. Completion only
    certifies this new goal, never the legacy CRS-to-JIG loading task.
    """

    def __init__(self, config: Mapping[str, Any], legacy_task_id: str, *, seed: int = 0,
                 completed_tasks: Any = (), goal_settings: Mapping[str, Any] | None = None) -> None:
        if legacy_task_id != LEGACY_TASK_ID:
            raise ValueError("StationaryLiftMachine currently implements only CRS pick-and-lift")
        super().__init__(config, legacy_task_id, seed=seed, completed_tasks=completed_tasks)
        self.goal_settings = _settings(config.get("stationary_task", {})
                                       if goal_settings is None else goal_settings, self.task)
        thresholds = self.config["thresholds"]
        if thresholds["capture_position_m"] > .025 or thresholds["capture_rotation_rad"] > .2:
            raise ValueError("Stationary lift must retain the original 25 mm / 0.2 rad grasp bounds")
        if any(key in self.task for key in ("carry_recovery_offsets_w", "recover_base_offset_w")):
            raise ValueError("Stationary lift cannot silently skip a configured recovery path")
        self.stages = (*MANIPULATION_STAGES[:6], _HOLD_STAGE)
        if any(relation.get("check_until_stage", "settle") not in self.stages
               for relation in self.task.get("required_object_relations", [])):
            raise ValueError("Stationary object relations must explicitly end within the stationary task")
        self._hold_evidence: dict[str, Any] | None = None
        self._hold_object_anchor: np.ndarray | None = None
        self._hold_anchor_time_s: float | None = None

    def _read_observation(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        observed = super()._read_observation(observation)
        observed["stationary_bilateral_grasp"] = _bilateral_contact(observation, self.arm, self.object_path)
        contact = observation.get("contact_evidence", {}).get(self.arm, {}).get(self.object_path, {})
        observed["stationary_contact_summary"] = {
            key: deepcopy(contact.get(key)) for key in (
                "source", "available", "object_path", "arm", "force_threshold_n",
                "jaw_force_magnitudes_n", "bilateral_contact_confirmed", "pinch_confirmed")
        } if isinstance(contact, Mapping) else {}
        # The legacy close gate already requires settled jaws and contact dwell.
        # It must additionally reject a closed command or a one-jaw contact.
        observed["closed"] = observed["closed"] and observed["stationary_bilateral_grasp"]
        return observed

    def _grasp_valid(self, observation: dict[str, Any]) -> bool:
        return bool(observation.get("stationary_bilateral_grasp") and super()._grasp_valid(observation))

    def _enter(self, index: int, observation: dict[str, Any]) -> None:
        super()._enter(index, observation)
        if self.stage_id == "lift":
            self._target_object = self._source_object.copy()
            self._target_object[:3, 3] += self.task["lift_offset_w"]
            if "lift_rotation_w" in self.task:
                rotation = pose_to_matrix([0., 0., 0., *self.task["lift_rotation_w"]])
                self._target_object[:3, :3] = rotation[:3, :3] @ self._source_object[:3, :3]
        if self.stage_id == _HOLD_STAGE:
            self._tcp_target = observation["tcp"].copy()
            self._hold_object_anchor = observation["objects"][self.object_path].copy()
            self._hold_anchor_time_s = observation["time_s"]
            self.history[-1]["stationary_phase"] = "hold"

    def _hold_measurements(self, observed: dict[str, Any]) -> dict[str, Any]:
        settings = self.goal_settings
        object_pose = observed["objects"][self.object_path]
        lift = float(object_pose[2, 3] - self._source_object[2, 3])
        anchor_spin = None
        if self._symmetry_axis is None:
            anchor_position_error = float(np.linalg.norm(object_pose[:3, 3] - self._hold_object_anchor[:3, 3]))
            relative_rotation = self._hold_object_anchor[:3, :3].T @ object_pose[:3, :3]
            anchor_rotation_error = float(math.acos(np.clip((np.trace(relative_rotation) - 1) / 2, -1, 1)))
        elif self._symmetry_pivot_radius_m is None:
            # Declared ring symmetry: the immutable anchor keeps the centre and the axis direction; a spin
            # about the axis inside the still pads is reported, never a drift.
            anchor_position_error, anchor_rotation_error, anchor_spin = symmetric_pose_errors(
                object_pose, self._hold_object_anchor, self._symmetry_axis)
        else:
            # With a pivot radius the anchor keeps the rim at the pinch (the original acquisition's pinch
            # point) and the axis direction: a ring that settled at a new hanging angle is still anchored.
            anchor_position_error, anchor_rotation_error, anchor_spin = symmetric_contact_errors(
                object_pose, self._hold_object_anchor, self._symmetry_axis,
                pinch_point(self._object_to_tcp, self._symmetry_axis, self._symmetry_pivot_radius_m))
        velocity, base_velocity = observed["object_velocity"], observed["base_velocity"]
        linear = None if velocity is None else float(np.linalg.norm(velocity[:3]))
        angular = None if velocity is None else float(np.linalg.norm(velocity[3:]))
        base_linear = None if base_velocity is None else float(np.linalg.norm(base_velocity[:3]))
        base_yaw = None if base_velocity is None else abs(float(base_velocity[5]))
        gates = {
            "minimum_lift": lift >= settings["minimum_lift_m"],
            "physical_grasp_retained": self._grasp_valid(observed),
            "object_com_stationary": bool(linear is not None and angular is not None
                and linear <= settings["max_object_linear_m_s"]
                and angular <= settings["max_object_angular_rad_s"]),
            # Cross-check the measured pose stream as well as PhysX COM speeds;
            # stale zero velocities cannot certify a visibly moving object.
            "object_pose_stationary": bool(self._linear_speed is not None and self._angular_speed is not None
                and self._linear_speed <= _MAX_POSE_LINEAR_M_S
                and self._angular_speed <= settings["max_object_angular_rad_s"]),
            "hold_anchor_retained": (anchor_position_error <= _MAX_HOLD_POSITION_DRIFT_M
                and anchor_rotation_error <= _MAX_HOLD_ROTATION_DRIFT_RAD),
            "base_stationary": bool(base_linear is not None and base_yaw is not None
                and base_linear <= settings["max_base_linear_m_s"]
                and base_yaw <= settings["max_base_yaw_rad_s"]),
        }
        return {"measurement_time_s": observed["time_s"], "lift_m": lift,
                "hold_contract_version": HOLD_CONTRACT_VERSION,
                "hold_anchor_time_s": self._hold_anchor_time_s,
                "hold_object_anchor_pose_w": matrix_to_pose(self._hold_object_anchor),
                "hold_anchor_position_error_m": anchor_position_error,
                "hold_anchor_rotation_error_rad": anchor_rotation_error,
                "hold_anchor_spin_rad": anchor_spin,
                "fixed_acceptance_bounds": {
                    "max_object_pose_linear_m_s": _MAX_POSE_LINEAR_M_S,
                    "max_hold_position_drift_m": _MAX_HOLD_POSITION_DRIFT_M,
                    "max_hold_rotation_drift_rad": _MAX_HOLD_ROTATION_DRIFT_RAD},
                "object_com_linear_speed_m_s": linear, "object_com_angular_speed_rad_s": angular,
                "object_pose_linear_speed_m_s": self._linear_speed,
                "object_pose_angular_speed_rad_s": self._angular_speed,
                "base_linear_speed_m_s": base_linear, "base_yaw_speed_rad_s": base_yaw,
                "contact_evidence": deepcopy(observed["stationary_contact_summary"]),
                "gates": gates, "all_conditions_met": all(gates.values()),
                "required_dwell_s": settings["hold_dwell_s"]}

    def _base_near(self, observation: dict[str, Any], station: str) -> bool:
        if self.stage_id == _HOLD_STAGE and station == "target":
            self._hold_evidence = self._hold_measurements(observation)
            return self._hold_evidence["all_conditions_met"]
        return super()._base_near(observation, station)

    def _dwell(self, condition: bool, now: float, duration: float) -> bool:
        if self.stage_id == _HOLD_STAGE:
            duration = self.goal_settings["hold_dwell_s"]
        return super()._dwell(condition, now, duration)

    def _command(self) -> dict[str, Any]:
        command = super()._command()
        phase = ("hold" if self.stage_id == _HOLD_STAGE else
                 "station_alignment" if self.stage_id == "navigate_source" else self.stage_id)
        if self.stage_id == _HOLD_STAGE:
            command["base_goal_pose_w"] = None
            command["base_arrival_braking"] = False
            if self.status == "running":
                command["motion_mode"] = "contact_servo"
                command["hand_command"] = "close"
        evidence = (self._hold_measurements(self._last_observation)
                    if self.stage_id == _HOLD_STAGE and self._last_observation is not None
                    else deepcopy(self._hold_evidence))
        if evidence is not None:
            evidence["continuous_dwell_s"] = (0.0 if self._condition_since is None else
                max(0.0, self._previous_time - self._condition_since))
        command.update({"legacy_task_id": self.task_id, "goal_id": GOAL_ID,
                        "dataset_task_id": GOAL_ID, "stationary_phase": phase,
                        "episode_scope": "stationary_task",
                        "full_task_completed": self.status == "succeeded",
                        "legacy_full_task_completed": False,
                        "stationary_hold_evidence": evidence,
                        "stationary_metrics": deepcopy(evidence),
                        "stationary_goal_settings": deepcopy(self.goal_settings)})
        return command

    def graph(self) -> dict[str, Any]:
        graph = super().graph()
        graph.update({"legacy_task_id": self.task_id, "goal_id": GOAL_ID,
                      "dataset_task_id": GOAL_ID, "episode_scope": "stationary_task",
                      "legacy_full_task_completed": False,
                      "hold_contract_version": HOLD_CONTRACT_VERSION,
                      "stationary_goal_settings": deepcopy(self.goal_settings)})
        for stage in graph["stages"]:
            if stage["id"] == _HOLD_STAGE:
                stage.update(stationary_phase="hold", success_condition=(
                    "measured minimum lift, same-object bilateral grasp, original retention bounds, "
                    "low measured COM/pose/base speeds and an immutable hold-entry object anchor "
                    "within 5 mm / 0.05 rad continuously for the configured hold dwell (a declared "
                    "object_symmetry excludes the spin about its axis from the anchor rotation and, with "
                    "pivot_radius_m, anchors the rim at the pinch instead of the ring centre)"))
            elif stage["id"] == "navigate_source":
                stage["stationary_phase"] = "station_alignment"
        return graph
