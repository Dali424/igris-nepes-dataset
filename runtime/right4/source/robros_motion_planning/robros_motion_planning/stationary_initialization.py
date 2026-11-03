"""Explicit held-object initial conditions for independent placement episodes.

This module describes and validates an episode boundary, not a grasp planner.
The runtime must apply the state before stepping physics, then verify actual
bilateral contact under gravity and live HugWBC before recording any placement.
No inferred contact, successful pickup, or attachment is returned here.
"""

from collections.abc import Mapping
from copy import deepcopy
import math

import numpy as np

from .snapshot import pose_to_matrix


def validate_prepared_grasp_initialization(config, *, model=None, command_names=None,
                                          control_limits=None):
    value = config.get("prepared_grasp_initialization")
    if value is None:
        return None
    spec = config.get("stationary_task", {})
    if (spec.get("number") != 2 or spec.get("id") != "crs_load_jig"
            or spec.get("goal") != "load_jig" or config.get("attachment_mode") != "none"):
        raise ValueError("Prepared held state is only an independent physical task 2 initial condition")
    keys = {"kind", "object_pose_w", "robot_root_pose_w", "joint_positions",
            "command_joint_positions", "source"}
    if not isinstance(value, Mapping) or set(value) != keys or value["kind"] != "measured_grasp_boundary":
        raise ValueError("Prepared grasp needs explicit root, object, measured joints, commands and source")
    result = deepcopy(dict(value))
    for key in ("object_pose_w", "robot_root_pose_w"):
        pose = np.asarray(value[key], dtype=float)
        if pose.shape != (7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:]) - 1.) > 1e-5:
            raise ValueError(f"Prepared grasp {key} must be xyz and a unit wxyz quaternion")
        pose_to_matrix(pose)
        result[key] = pose.tolist()
    for key in ("joint_positions", "command_joint_positions"):
        joints = value[key]
        if (not isinstance(joints, Mapping) or not joints or any(
                not isinstance(n, str) or isinstance(q, bool) or not isinstance(q, (int, float))
                or not math.isfinite(q) for n, q in joints.items())):
            raise ValueError(f"Prepared grasp {key} must contain named finite positions")
        result[key] = {n: float(q) for n, q in joints.items()}
    if not set(result["command_joint_positions"]) <= set(result["joint_positions"]):
        raise ValueError("Prepared commands must name measured robot joints")
    source = value["source"]
    if (not isinstance(source, Mapping) or not isinstance(source.get("description"), str)
            or not source["description"].strip() or not isinstance(source.get("snapshot_path"), str)
            or not isinstance(source.get("snapshot_sha256"), str)
            or len(source["snapshot_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in source["snapshot_sha256"])):
        raise ValueError("Prepared state needs a named, hashed source snapshot and an explicit derivation")
    if model is not None:
        result["joint_positions"] = model.validate_snapshot(result["joint_positions"])
        model.validate_snapshot({**result["joint_positions"], **result["command_joint_positions"]})
        task = config["tasks"]["crs_load_jig"]
        actual_tcp = (pose_to_matrix(result["robot_root_pose_w"])
                      @ model.forward_kinematics(result["joint_positions"])[model.default_ee_link(task["arm"])])
        expected_tcp = pose_to_matrix(result["object_pose_w"]) @ pose_to_matrix(task["grasp_pose_object"])
        position_error = np.linalg.norm(actual_tcp[:3, 3] - expected_tcp[:3, 3])
        rotation_error = math.acos(np.clip((np.trace(actual_tcp[:3, :3].T @ expected_tcp[:3, :3]) - 1) / 2, -1, 1))
        if position_error > .0005 or rotation_error > .005:
            raise ValueError("Prepared object and actual joint FK do not preserve the declared measured grasp")
    if command_names is not None and set(result["command_joint_positions"]) != set(command_names):
        raise ValueError("Prepared state must specify every upper-body command, including both jaws")
    if control_limits is not None and any(
            n not in control_limits or not control_limits[n][0] <= q <= control_limits[n][1]
            for n, q in result["command_joint_positions"].items()):
        raise ValueError("Prepared command exceeds the shared live-controller limits")
    return result


def prepared_grasp_provenance(prepared):
    """Evidence labels attached to actual runtime boundary state writes."""
    return {
        "kind": "measured_grasp_boundary",
        "scope": "episode_boundary_before_physics_settle_and_recording",
        "object_pose_w": deepcopy(prepared["object_pose_w"]),
        "robot_root_pose_w": deepcopy(prepared["robot_root_pose_w"]),
        "joint_positions": deepcopy(prepared["joint_positions"]),
        "command_joint_positions": deepcopy(prepared["command_joint_positions"]),
        "source": deepcopy(prepared["source"]),
        "initialization_only": True,
        "pickup_demonstrated": False,
        "physical_grasp_verified": False,
        "requires_measured_bilateral_grasp_and_stability": True,
        "attachment_created": False,
        "demonstration_pose_write": False,
    }
