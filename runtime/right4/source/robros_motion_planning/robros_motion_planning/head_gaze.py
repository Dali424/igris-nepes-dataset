"""CPU-only neck initialization using the actual mounted camera's optical axis.

The result is an episode-boundary initialization recommendation. Apply it before
settling and planning, then hold the neck fixed for the whole planned path.
This solves pointing direction only; it does not test visibility or collisions.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
from typing import Any

import numpy as np

from .robot_model import RobotModel
from .snapshot import matrix_to_pose, pose_to_matrix


def recommend_head_joint_targets(
    model: RobotModel,
    joint_positions: Mapping[str, float],
    base_pose_w: Sequence[float],
    target_w: Sequence[float],
    camera_attachments: Mapping[str, Mapping[str, Any]],
    *,
    head_camera: str = "LeftCamera",
    neck_joint_names: tuple[str, str] = ("Joint_Neck_Yaw", "Joint_Neck_Pitch"),
) -> dict[str, Any]:
    """Minimize optical-axis angular error within the two URDF joint limits.

    ``base_pose_w`` is the measured pose of ``model.root_link``, xyz + wxyz.
    All non-neck joints remain at ``joint_positions``. The mounted camera's
    authored USD (-Z forward, +Y up) extrinsic is retained exactly. A finite
    global grid followed by bounded local grids needs only NumPy, makes no
    convergence claim, and always retains the original valid pose as a candidate.
    """
    positions = model.validate_snapshot(joint_positions)
    base = pose_to_matrix(base_pose_w)
    target = np.asarray(target_w, dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("Head target must contain three finite world coordinates")
    if len(neck_joint_names) != 2 or len(set(neck_joint_names)) != 2:
        raise ValueError("Exactly two distinct neck joints are required")
    for name in neck_joint_names:
        if name not in model.joints or model.joints[name].kind != "revolute":
            raise ValueError(f"Missing revolute neck joint: {name}")
    if head_camera not in camera_attachments:
        raise ValueError(f"Authored head camera missing: {head_camera}")
    source = camera_attachments[head_camera]
    if source.get("pose_convention") != "usd_camera_minus_z_forward_plus_y_up":
        raise ValueError("Head camera must retain its authored USD pose convention")
    body = source["body_name"]
    if body not in model.link_names:
        raise ValueError(f"Head camera mount missing from URDF: {body}")
    # Reject a camera unrelated to the requested neck joints (e.g. a wrist).
    parent_joints = {joint.child: joint for joint in model.joints.values()}
    ancestors = set()
    link = body
    while link != model.root_link:
        joint = parent_joints[link]
        ancestors.add(joint.name)
        link = joint.parent
    if not set(neck_joint_names) <= ancestors:
        raise ValueError("Both neck joints must be ancestors of the head camera")
    relative = pose_to_matrix(source["relative_pose_wxyz"])
    limits = np.asarray([model.joint_limits[name] for name in neck_joint_names])
    initial = np.asarray([positions[name] for name in neck_joint_names])
    evaluations = 0

    def evaluate(values):
        nonlocal evaluations
        candidate = {**positions, **dict(zip(neck_joint_names, map(float, values)))}
        camera = base @ model.forward_kinematics(candidate)[body] @ relative
        forward = -camera[:3, 2]  # Authored USD camera optical direction.
        delta = target - camera[:3, 3]
        distance = float(np.linalg.norm(delta))
        evaluations += 1
        if distance < 1e-9:
            return math.inf, camera, forward, distance
        cosine = float(np.clip(np.dot(forward, delta / distance), -1, 1))
        return 1 - cosine, camera, forward, distance

    initial_result = evaluate(initial)
    if not math.isfinite(initial_result[0]):
        raise ValueError("Head target coincides with the initial camera origin")
    best, best_result = initial.copy(), initial_result

    def search(yaw_grid, pitch_grid):
        nonlocal best, best_result
        for yaw in yaw_grid:
            for pitch in pitch_grid:
                candidate = np.array([yaw, pitch])
                result = evaluate(candidate)
                # Prefer the smaller change from the initial neck pose on ties.
                if result[0] < best_result[0] or (result[0] == best_result[0]
                        and np.linalg.norm(candidate - initial) < np.linalg.norm(best - initial)):
                    best, best_result = candidate, result

    search(*(np.linspace(low, high, 17) for low, high in limits))
    step = (limits[:, 1] - limits[:, 0]) / 16
    for _ in range(10):
        grids = [np.unique(np.clip(value + np.linspace(-size, size, 5), low, high))
                 for value, size, (low, high) in zip(best, step, limits)]
        search(*grids)
        step /= 2

    cost, camera, forward, distance = best_result
    return {
        "schema": "igris_c_head_gaze_initialization_v1",
        "joint_targets_rad": dict(zip(neck_joint_names, map(float, best))),
        "initial_joint_positions_rad": dict(zip(neck_joint_names, map(float, initial))),
        "joint_limits_rad": {name: list(map(float, bounds)) for name, bounds in zip(neck_joint_names, limits)},
        "initial_angular_error_rad": math.acos(float(np.clip(1 - initial_result[0], -1, 1))),
        "angular_error_rad": math.acos(float(np.clip(1 - cost, -1, 1))),
        "camera_pose_w_usd": matrix_to_pose(camera), "camera_forward_w": forward.tolist(),
        "target_w": target.tolist(), "target_distance_m": distance,
        "camera_name": head_camera, "camera_mount_body": body,
        "source_camera_relative_pose_wxyz": list(source["relative_pose_wxyz"]),
        "source_camera_relative_prim_path": source.get("relative_prim_path"),
        "model_urdf_sha256": hashlib.sha256(model.urdf_path.read_bytes()).hexdigest(),
        "base_link": model.root_link, "base_pose_w": list(map(float, base_pose_w)),
        "method": "17x17_global_grid_then_10_bounded_5x5_refinements",
        "evaluations": evaluations, "application_scope": "episode_boundary_before_settle_and_planning",
        "hold_neck_during_execution": True,
        "visibility_checked": False, "collision_checked": False,
    }
