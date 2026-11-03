"""Bounded empty-hand pregrasp paths with measured contact and waypoint gates.

These references are not collision planners. They never move an object, change
the final task goal, or create grasp evidence. Original task guards stay active.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
import numpy as np

from .crs_manipulation import TcpPoseController
from .motion_watchdog import pose_goal_errors
from .snapshot import matrix_to_pose, pose_to_matrix


DEFAULT_SETTINGS = dict(linear_speed=.03, linear_acceleration=.06,
    angular_speed=.12, angular_acceleration=.3, position_lead=.01,
    rotation_lead=.02, cross_track_position=.01, cross_track_rotation=.05,
    reference_profile="online")


def validate_pregrasp_settings(settings, waypoints=()):
    if not isinstance(settings, Mapping) or set(settings) - set(DEFAULT_SETTINGS):
        raise ValueError("Unknown pregrasp_pose_control settings")
    checked = {**DEFAULT_SETTINGS, **settings}
    for key, maximum in DEFAULT_SETTINGS.items():
        if key == "reference_profile":
            if checked[key] not in ("online", "quintic"):
                raise ValueError("Invalid pregrasp reference_profile")
            continue
        value = checked[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float, np.number))
                or not math.isfinite(float(value)) or not 0 < value <= maximum):
            raise ValueError(f"pregrasp {key} must be positive and <= {maximum}")
    if not isinstance(waypoints, (list, tuple)) or len(waypoints) > 4:
        raise ValueError("pregrasp_waypoints must contain at most four poses")
    for waypoint in waypoints:
        if (not isinstance(waypoint, Mapping) or set(waypoint) != {"frame", "pose"}
                or waypoint["frame"] not in ("world", "target")):
            raise ValueError("Pregrasp waypoint needs frame world/target and pose")
        pose_to_matrix(waypoint["pose"])
    return checked


def pregrasp_contact_failure(contact, arm, object_path):
    """No task-object contact is expected anywhere on the empty pregrasp path."""
    if (not isinstance(contact, Mapping) or contact.get("available") is not True
            or contact.get("source") != "physx_filtered_hand_object_contacts"
            or contact.get("arm") != arm or contact.get("object_path") != object_path):
        return "contact_sensor_unavailable"
    force = contact.get("force_magnitude_sum_n")
    if (isinstance(force, bool) or not isinstance(force, (float, int, np.number))
            or not math.isfinite(float(force)) or force < 0):
        return "invalid_contact_force"
    return "unexpected_object_contact" if force > .05 else None


class BoundedPregraspController:
    def __init__(self, initial_tcp, final_tcp, *, arm, object_path, settings,
                 waypoints=(), position_tolerance_m=.003, rotation_tolerance_rad=.05):
        if arm not in ("left", "right") or not isinstance(object_path, str) or not object_path.startswith("/"):
            raise ValueError("Pregrasp needs an arm and full object prim path")
        self.settings = validate_pregrasp_settings(settings, waypoints)
        self.arm, self.object_path = arm, object_path
        self.position_tolerance = float(position_tolerance_m)
        self.rotation_tolerance = float(rotation_tolerance_rad)
        if any(not math.isfinite(v) or v <= 0 for v in (self.position_tolerance, self.rotation_tolerance)):
            raise ValueError("Pregrasp arrival tolerances must be positive and finite")
        final = pose_to_matrix(final_tcp)
        self.final = matrix_to_pose(final)
        self.goals = [matrix_to_pose((final if w["frame"] == "target" else np.eye(4))
                                    @ pose_to_matrix(w["pose"])) for w in waypoints] + [self.final]
        self.index = 0
        self.last_target = matrix_to_pose(pose_to_matrix(initial_tcp))
        self.controller = TcpPoseController(self.last_target, self.goals[0], **self.settings)
        self.failure = None
        self.last_result = None

    def update(self, actual_tcp, dt, contact_evidence):
        if isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Pregrasp dt must be positive and finite")
        pose_to_matrix(actual_tcp)
        self.failure = self.failure or pregrasp_contact_failure(contact_evidence, self.arm, self.object_path)
        reference = None
        arrived = False
        if not self.failure:
            reference = self.controller.update(actual_tcp, dt)
            self.last_target = reference["tcp_target_pose_w"]
            errors = pose_goal_errors(actual_tcp, self.goals[self.index])
            arrived = (errors["position_error_m"] <= self.position_tolerance
                       and errors["rotation_error_rad"] <= self.rotation_tolerance
                       and reference["progress"] >= 1. - 1e-9)
            if arrived and self.index < len(self.goals) - 1:
                self.index += 1
                # Preserve the last commanded reference across the segment boundary.
                self.controller = TcpPoseController(self.last_target, self.goals[self.index], **self.settings)
                arrived = False
        result = dict(status="failed" if self.failure else "running", reason=self.failure,
            stage_id="pregrasp", arm=self.arm, object_path=self.object_path,
            tcp_target_pose_w=list(self.last_target), final_tcp_goal_pose_w=list(self.final),
            waypoint_goal_pose_w=list(self.goals[self.index]), waypoint_index=self.index,
            waypoint_count=len(self.goals), waypoints_complete=bool(arrived and self.index == len(self.goals) - 1),
            reference=reference, maximum_expected_object_contact_force_n=.05,
            protective_stop_overrides_acceleration_limit=bool(self.failure),
            reference_scope="empty_hand_pregrasp", collision_checked=False,
            physical_success_verified=False)
        self.last_result = result
        return result
