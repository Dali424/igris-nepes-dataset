"""Passive cleaner-cover hinge geometry and lazy USD joint authoring.

The source workcell has a removable lid. This explicit, provisional simulation
hardware conversion anchors that rigid body with a revolute joint. No drive,
motor, per-step pose write or automatic opening is provided by this module.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np

from .snapshot import matrix_to_pose, pose_to_matrix


def validate_hinge_config(config: Mapping[str, Any]) -> dict[str, Any]:
    hinge = dict(config.get("hinged_door", {}))
    if not hinge or hinge.get("enabled") is not True:
        raise ValueError("hinged_door.enabled must be true for a hinged door")
    anchor = np.asarray(hinge.get("anchor_w"), dtype=float)
    axis = np.asarray(hinge.get("axis_w"), dtype=float)
    if anchor.shape != (3,) or not np.isfinite(anchor).all():
        raise ValueError("Hinge anchor_w must contain three finite meters")
    if axis.shape != (3,) or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis), 1, atol=1e-6):
        raise ValueError("Hinge axis_w must be a unit vector")
    if not np.allclose(axis, [1, 0, 0], atol=1e-8):
        raise ValueError("This hardware conversion requires a world +X hinge axis")
    pose_to_matrix(hinge.get("closed_cover_pose_w"))
    limits = np.asarray(hinge.get("limits_deg"), dtype=float)
    if limits.shape != (2,) or not np.isfinite(limits).all() or not (-180 < limits[0] < limits[1] == 0):
        raise ValueError("Hinge limits_deg must be [negative opening limit, 0] above -180 degrees")
    for field in ("min_open_angle_deg", "angular_speed_rad_s", "max_target_lead_rad",
                  "pivot_tolerance_m", "axis_tolerance_rad", "angle_tolerance_rad"):
        value = float(hinge.get(field, float("nan")))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"hinged_door.{field} must be positive and finite")
    angle = float(hinge.get("open_angle_deg", float("nan")))
    if not limits[0] <= angle < 0 or not 90 <= float(hinge["min_open_angle_deg"]) <= abs(angle):
        raise ValueError("Hinge opening must reach its configured clearance angle of at least 90 degrees")
    if not str(hinge.get("joint_path", "")).startswith("/"):
        raise ValueError("Hinge joint_path must be an absolute prim path")
    return hinge


def hinge_pose_w(config: Mapping[str, Any], angle_rad: float) -> list[float]:
    """Compute an angle-consistent door pose; only initialization may write it."""
    hinge = validate_hinge_config(config)
    if not math.isfinite(angle_rad):
        raise ValueError("Hinge angle must be finite")
    lower, upper = np.radians(hinge["limits_deg"])
    if not lower - 1e-8 <= angle_rad <= upper + 1e-8:
        raise ValueError("Hinge angle exceeds mechanical limits")
    closed = pose_to_matrix(hinge["closed_cover_pose_w"])
    anchor = np.asarray(hinge["anchor_w"])
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    rotation = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    result = closed.copy()
    result[:3, :3] = rotation @ closed[:3, :3]
    result[:3, 3] = anchor + rotation @ (closed[:3, 3] - anchor)
    return matrix_to_pose(result)


def measure_hinge_state(config: Mapping[str, Any], cover_pose_w: Any) -> dict[str, Any]:
    """Infer signed angle and reject detached/translating or off-axis covers."""
    hinge = validate_hinge_config(config)
    closed = pose_to_matrix(hinge["closed_cover_pose_w"])
    measured = pose_to_matrix(cover_pose_w)
    relative = measured[:3, :3] @ closed[:3, :3].T
    angle = math.atan2(relative[2, 1] - relative[1, 2], relative[1, 1] + relative[2, 2])
    anchor_local = np.linalg.inv(closed) @ np.r_[hinge["anchor_w"], 1.0]
    pivot_error = float(np.linalg.norm((measured @ anchor_local)[:3] - hinge["anchor_w"]))
    c, s = math.cos(angle), math.sin(angle)
    ideal_rotation = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    axis_error = float(math.acos(np.clip((np.trace(ideal_rotation.T @ relative) - 1) / 2, -1, 1)))
    lower, upper = np.radians(hinge["limits_deg"])
    valid = (pivot_error <= hinge["pivot_tolerance_m"] and axis_error <= hinge["axis_tolerance_rad"]
             and lower - hinge["angle_tolerance_rad"] <= angle <= upper + hinge["angle_tolerance_rad"])
    return {"angle_rad": angle, "angle_deg": math.degrees(angle),
            "pivot_error_m": pivot_error, "axis_error_rad": axis_error,
            "valid": bool(valid),
            "open": bool(valid and -angle >= math.radians(hinge["min_open_angle_deg"]))}


def author_hinged_door(stage: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    """Create the passive world-to-cover joint before simulation reset/start."""
    if config.get("tasks", {}).get("open_cleaner_cover", {}).get("mechanism") != "revolute_hinge":
        return {"enabled": False, "reason": "legacy_free_removable_lid"}
    hinge = validate_hinge_config(config)
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics
    cover_path = config["objects"]["cover"]
    cover = stage.GetPrimAtPath(cover_path)
    if not cover.IsValid() or not cover.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError(f"Hinged cover must be an existing rigid body: {cover_path}")
    if stage.GetPrimAtPath(hinge["joint_path"]).IsValid():
        raise ValueError(f"Hinge joint already exists: {hinge['joint_path']}")
    if not UsdGeom.LinearUnitsAre(UsdGeom.GetStageMetersPerUnit(stage), UsdGeom.LinearUnits.meters):
        raise ValueError("Hinge authoring requires a meter-scale stage")
    closed = pose_to_matrix(hinge["closed_cover_pose_w"])
    anchor = np.asarray(hinge["anchor_w"], dtype=float)
    local_anchor = (np.linalg.inv(closed) @ np.r_[anchor, 1])[:3]
    local_rotation = np.eye(4)
    local_rotation[:3, :3] = closed[:3, :3].T
    quaternion = matrix_to_pose(local_rotation)[3:]
    joint = UsdPhysics.RevoluteJoint.Define(stage, hinge["joint_path"])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(cover_path)])
    # Empty body0 means world. Both joint frames coincide at angle zero.
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*anchor.tolist()))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*local_anchor.tolist()))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(quaternion[0], Gf.Vec3f(*quaternion[1:])))
    joint.CreateAxisAttr().Set("X")
    joint.CreateLowerLimitAttr().Set(float(hinge["limits_deg"][0]))
    joint.CreateUpperLimitAttr().Set(float(hinge["limits_deg"][1]))
    joint.CreateJointEnabledAttr().Set(True)
    joint.CreateExcludeFromArticulationAttr().Set(True)
    return {"enabled": True, "joint_path": hinge["joint_path"], "body1": cover_path,
            "anchor_w": anchor.tolist(), "axis_w": list(hinge["axis_w"]),
            "limits_deg": list(hinge["limits_deg"]), "actuation": "passive_no_drive",
            "source": "explicit_simulation_hardware_conversion",
            "calibration_status": "draft_unvalidated", "validated_physical_hardware": False}
