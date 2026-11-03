"""Immutable scene selection and measured slot provenance for a two-ring layout.

This module never writes USD or physics state. The separately authored USD
overlay places the cart and rings before PhysX starts. Existing reset, grasp,
support, release and success checks remain responsible for physical evidence.
Slot numbers are one-based; JIG indices retain their existing zero-based order.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import math
from pathlib import Path

from .jig_slots import SLOT_CENTERS_Y_M, SLOT_CLEAR_WIDTH_M, CRS_THICKNESS_ENVELOPE_M
from .no_walk_stance import validate_no_walk
from .single_step import validate_single_step
from .snapshot import matrix_to_pose, pose_to_matrix

LAYOUT_SCHEMA = "igris_c_single_arm_slot_layout_v1"
CART_PATH = "/World/Scene/Cart"
_KEYS = {"scene_usd", "scene_sha256", "source_slot_number", "target_slot_number",
         "occupied_cart_slots", "allowed_jig_slots"}


def resolve_layout_scene(config, default_scene):
    """Return (resolved scene, validated contract), before loading Isaac assets.

    A configured scene intentionally overrides the CLI default. Re-running this
    function immediately before scene configuration detects intervening edits.
    The SHA binds this USD file, not all of its referenced asset dependencies.
    """
    settings = config.get("single_arm_layout")
    if settings is None:
        return Path(default_scene), None
    if not isinstance(settings, Mapping) or set(settings) != _KEYS:
        raise ValueError("single_arm_layout must contain exactly " + ", ".join(sorted(_KEYS)))
    for key in ("source_slot_number", "target_slot_number"):
        if type(settings[key]) is not int or settings[key] not in (2, 4):
            raise ValueError(f"single_arm_layout.{key} must be the one-based slot number 2 or 4")
    for key in ("occupied_cart_slots", "allowed_jig_slots"):
        if (not isinstance(settings[key], list) or settings[key] != [2, 4]
                or any(type(value) is not int for value in settings[key])):
            raise ValueError(f"single_arm_layout.{key} must be [2, 4]")
    if not isinstance(settings["scene_usd"], str) or not Path(settings["scene_usd"]).is_absolute():
        raise ValueError("single_arm_layout.scene_usd must be an absolute USD path")
    scene = Path(settings["scene_usd"]).resolve(strict=True)
    if not scene.is_file() or scene.suffix.lower() not in (".usd", ".usda", ".usdc"):
        raise ValueError("single_arm_layout.scene_usd must name an existing USD file")
    digest = settings["scene_sha256"]
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("single_arm_layout.scene_sha256 must be a lowercase SHA256 digest")
    if hashlib.sha256(scene.read_bytes()).hexdigest() != digest:
        raise ValueError("single_arm_layout scene SHA256 mismatch")
    source_path = f"/World/Scene/Ring_{settings['source_slot_number']:02d}"
    if config.get("objects", {}).get("crs") != source_path:
        raise ValueError("single_arm_layout source slot and objects.crs do not match")
    task = config.get("tasks", {}).get("crs_load_jig", {})
    if task.get("object") != "crs" or task.get("arm") not in ("left", "right"):
        raise ValueError("single_arm_layout requires one CRS pickup/load arm")
    if task.get("handover") is not None:
        raise ValueError("single_arm_layout forbids a handover")
    if validate_no_walk(task) is None:
        step = validate_single_step(task)
        if step is None or step.get("source_hold") is not True:
            raise ValueError("single_arm_layout requires navigation_control.no_walk or single_step.source_hold=true")
    target_slot = settings["target_slot_number"] - 1
    slot_index = config.get("jig_slots", {}).get("slot_index")
    if type(slot_index) is not int or slot_index != target_slot:
        raise ValueError("single_arm_layout target slot and jig_slots.slot_index do not match")
    target = task.get("target", {})
    if target.get("frame") != "jig":
        raise ValueError("single_arm_layout target must use the measured jig frame")
    pose = target.get("pose")
    if (not isinstance(pose, (list, tuple)) or len(pose) != 7
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in pose)):
        raise ValueError("single_arm_layout requires a finite JIG-relative target pose")
    # A face-origin pose may differ from its slot center by half the ring
    # thickness. This catches mislabeled adjacent slots without replacing the
    # existing much tighter measured placement/admission gates.
    maximum_origin_offset = (SLOT_CLEAR_WIDTH_M + CRS_THICKNESS_ENVELOPE_M) / 2
    if abs(pose[1] - SLOT_CENTERS_Y_M[target_slot]) > maximum_origin_offset:
        raise ValueError("single_arm_layout target pose is outside its numbered JIG slot")
    contract = {"schema": LAYOUT_SCHEMA, **deepcopy(dict(settings)), "scene_usd": str(scene),
                "source_object_prim_path": source_path,
                "target_object_prim_path": config["objects"]["jig"],
                "cart_prim_path": CART_PATH,
                "active_ring_paths": [f"/World/Scene/Ring_{slot:02d}" for slot in (2, 4)],
                "target_slot_index": target_slot, "target_pose_jig": list(pose),
                "manipulation_arm": task["arm"], "handover_enabled": False,
                "scene_placement_scope": "authored_overlay_before_physics",
                "scene_hash_scope": "overlay_file_only_not_transitive_asset_dependencies"}
    return scene, contract


def discover_layout_objects(stage, contract):
    """Strict discovery for the explicit layout; other cart rings must be inactive."""
    from pxr import UsdPhysics
    names = [f"Ring_{slot:02d}" for slot in contract["occupied_cart_slots"]]
    for slot in (1, 3, 5):
        path = f"/World/Scene/Ring_{slot:02d}"
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid() and prim.IsActive():
            raise ValueError(f"Unoccupied cart ring must be absent or inactive in the layout: {path}")
    result = {}
    for name in names + ["CleanerCover", "asset_jig_rev01"]:
        path = "/World/Scene/" + name
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsActive() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"NEPES dynamic rigid body is missing: {path}")
        body = UsdPhysics.RigidBodyAPI(prim)
        if body.GetRigidBodyEnabledAttr().Get() is False or body.GetKinematicEnabledAttr().Get() is True:
            raise ValueError(f"Task object must be a dynamic enabled rigid body: {path}")
        result[name] = path
    cart = stage.GetPrimAtPath(CART_PATH)
    if not cart.IsValid() or not cart.IsActive():
        raise ValueError("single_arm_layout requires the original active cart fixture")
    return result


def measured_layout_provenance(contract, object_poses_w, *, phase):
    """Bind selected slots to actual measured actors and the resolved JIG target."""
    if contract is None:
        return None
    result = deepcopy(contract)
    result["measurement_phase"] = str(phase)
    result["source_initial_pose_w"] = list(object_poses_w[contract["source_object_prim_path"]])
    result["jig_initial_pose_w"] = list(object_poses_w[contract["target_object_prim_path"]])
    result["cart_pose_w"] = list(object_poses_w[CART_PATH])
    result["active_ring_initial_poses_w"] = {
        path: list(object_poses_w[path]) for path in contract["active_ring_paths"]}
    result["target_pose_w"] = matrix_to_pose(
        pose_to_matrix(result["jig_initial_pose_w"]) @ pose_to_matrix(result["target_pose_jig"]))
    result["pose_sources"] = {"rings_and_jig": "measured_physx_rigid_body_tensors",
                              "cart": "static_composed_usd_world_transform"}
    return result
