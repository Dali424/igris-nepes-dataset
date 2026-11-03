"""World-only collision filtering for a frozen-base, single-arm planning query.

cuRobo 0.7 shares robot spheres between self and world collision. Disabling link
spheres on the robot would also hide the torso from the moving arm. This module
instead masks a COPY of each world query and leaves robot/self-collision data
unchanged. Frozen feet touching the floor therefore do not invalidate arm plans.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from .robot_model import RobotModel


def moving_link_names(model: RobotModel, arm: str) -> tuple[str, ...]:
    """All descendants of the selected shoulder, including palm and fingers."""
    shoulder = model.joints[model.arm_joint_names(arm)[0]].child
    moving = {shoulder}
    while True:
        extended = moving | {j.child for j in model.joints.values() if j.parent in moving}
        if extended == moving:
            break
        moving = extended
    return tuple(name for name in model.link_names if name in moving)


def _mask_query_spheres(query_sphere: Any, active_mask: Any, extra_radius: float = 0.0) -> Any:
    """Return differentiable independent storage; preserve all query dimensions."""
    import torch

    if not math.isfinite(extra_radius) or extra_radius < 0:
        raise ValueError("World extra radius must be finite and nonnegative")
    if query_sphere.ndim != 4 or query_sphere.shape[-1] != 4:
        raise ValueError("Expected [batch, horizon, spheres, 4] collision query")
    if active_mask.ndim != 1 or query_sphere.shape[-2] != active_mask.numel():
        raise ValueError("World collision sphere order/count does not match the robot model")
    if active_mask.dtype != torch.bool or active_mask.device != query_sphere.device:
        raise ValueError("World collision mask must be boolean and on the query device")
    masked = query_sphere.clone()
    masked[..., 3] = torch.where(active_mask, query_sphere[..., 3] + extra_radius, -100.0)
    return masked.contiguous()


def make_world_checker(
    world_config_dict: dict[str, Any],
    tensor_args: Any,
    kinematics_config: Any,
    moving_link_names: Iterable[str],
    *, extra_radius: float = 0.0,
) -> Any:
    """Create a legacy cuRobo mesh+primitive checker with an immutable link mask.

    Pass ``RobotConfig.kinematics.kinematics_config`` to kinematics_config and
    inject the result through ``MotionGenConfig.load_from_robot_config(...,
    world_coll_checker=checker)``. Construct it before warmup/CUDA graph capture.
    Mask selection uses cuRobo's actual sphere indices, never dictionary order.
    ``extra_radius`` adds nonnegative meters only to active world query copies;
    the robot's stored radii and self-collision offsets remain unchanged.

    This is valid only while the robot root and all nonselected joints are frozen
    at the same planning snapshot. It does not validate support contacts, balance,
    locomotion, or collisions between already-frozen bodies and the world.
    """
    if not math.isfinite(extra_radius) or extra_radius < 0:
        raise ValueError("World extra radius must be finite and nonnegative")
    import torch
    from curobo.geom.sdf.world import CollisionCheckerType, WorldCollisionConfig
    from curobo.geom.sdf.world_mesh import WorldMeshCollision
    from curobo.geom.types import WorldConfig

    names = tuple(dict.fromkeys(moving_link_names))
    if not names:
        raise ValueError("At least one moving link is required")
    spheres = kinematics_config.link_spheres
    if spheres.ndim != 2 or spheres.shape[-1] != 4:
        raise ValueError("Expected cuRobo kinematic link_spheres with shape [spheres, 4]")
    mask = torch.zeros(spheres.shape[0], device=spheres.device, dtype=torch.bool)
    for name in names:
        if name not in kinematics_config.link_name_to_idx_map:
            raise ValueError(f"Moving link missing from cuRobo kinematics: {name}")
        indices = kinematics_config.get_sphere_index_from_link_name(name)
        mask[indices] = True
    active_count = int(mask.sum().item())
    if active_count == 0:
        raise ValueError("Moving links have no collision spheres")

    class MovingArmWorldCollision(WorldMeshCollision):
        # Override all four public paths used by optimization, constraints and
        # swept validation. SelfCollisionCost never calls these methods.
        def get_sphere_distance(self, query_sphere, *args, **kwargs):
            return super().get_sphere_distance(_mask_query_spheres(query_sphere, mask, extra_radius), *args, **kwargs)

        def get_sphere_collision(self, query_sphere, *args, **kwargs):
            return super().get_sphere_collision(_mask_query_spheres(query_sphere, mask, extra_radius), *args, **kwargs)

        def get_swept_sphere_distance(self, query_sphere, *args, **kwargs):
            return super().get_swept_sphere_distance(_mask_query_spheres(query_sphere, mask, extra_radius), *args, **kwargs)

        def get_swept_sphere_collision(self, query_sphere, *args, **kwargs):
            return super().get_swept_sphere_collision(_mask_query_spheres(query_sphere, mask, extra_radius), *args, **kwargs)

    config = WorldCollisionConfig(
        tensor_args=tensor_args,
        world_model=WorldConfig.from_dict(world_config_dict),
        checker_type=CollisionCheckerType.MESH,
        cache={"mesh": max(1, len(world_config_dict.get("mesh", {}))),
               "obb": max(1, len(world_config_dict.get("cuboid", {})))},
    )
    checker = MovingArmWorldCollision(config)
    checker.world_collision_manifest = {
        "mode": "moving_arm_world_full_body_self",
        "moving_link_names": list(names),
        "excluded_world_link_names": sorted(set(kinematics_config.link_name_to_idx_map) - set(names)),
        "active_world_spheres": active_count,
        "excluded_world_spheres": int(mask.numel()) - active_count,
        "assumption": "Robot root and all nonselected joints remain frozen at the planning snapshot.",
        "self_collision_spheres_modified": False,
        "extra_world_radius_m": extra_radius,
    }
    return checker
