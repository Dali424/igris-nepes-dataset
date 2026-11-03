"""Measured payload envelope guards, independent of Isaac and robot control.

The declared vertices or solid cylinder are a conservative proxy supplied by
the caller, not a mesh inferred or certified by this module. World AABB overlap
is deliberately conservative: it can reject empty corners of a rotated proxy.
These instantaneous checks do not certify swept robot/payload collision,
balance, grip forces, or that the proxy encloses the authored asset.
"""
from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Real

import numpy as np

from .snapshot import pose_to_matrix

SCHEMA_VERSION = "igris_c_payload_transport_constraints_v1"
TRANSPORT_STAGES = frozenset(("lift", "carry_posture", "navigate_target", "place_posture", "transfer"))


def _number(value, name, *, minimum=None, maximum=None, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    value = float(value)
    if ((positive and value <= 0) or (minimum is not None and value < minimum)
            or (maximum is not None and value > maximum)):
        raise ValueError(f"{name} is outside its permitted bounds")
    return value


def _vector(value, name, *, unit=False):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} needs three numbers")
    result = [_number(item, name) for item in value]
    if unit and not math.isclose(float(np.linalg.norm(result)), 1., abs_tol=1e-6, rel_tol=0.):
        raise ValueError(f"{name} must be a unit axis")
    return result


def _keys(value, required, optional, name):
    if not isinstance(value, Mapping) or not required <= set(value) or set(value) - required - optional:
        raise ValueError(f"{name} has missing or unsupported keys")


def _rule(value, name):
    allowed = {"minimum_world_z_m", "maximum_world_z_m", "orientation", "forbidden_aabbs", "slot_approach"}
    _keys(value, set(), allowed, name)
    if not value:
        raise ValueError(f"{name} needs a transport constraint")
    result = {}
    for key in ("minimum_world_z_m", "maximum_world_z_m"):
        if key in value:
            result[key] = _number(value[key], f"{name}.{key}")
    if ("minimum_world_z_m" in result and "maximum_world_z_m" in result
            and result["minimum_world_z_m"] >= result["maximum_world_z_m"]):
        raise ValueError(f"{name} needs minimum world Z below maximum world Z")
    if "slot_approach" in value:
        if name != "stage_rules.transfer" or "minimum_world_z_m" not in value:
            raise ValueError("slot_approach requires the transfer stage and its minimum_world_z_m")
        slot = value["slot_approach"]
        keys = {"maximum_abs_world_xy_error_m", "maximum_rotation_error_rad",
                "minimum_bottom_height_above_slot_m"}
        _keys(slot, keys, set(), "slot_approach")
        xy = slot["maximum_abs_world_xy_error_m"]
        if not isinstance(xy, (list, tuple)) or len(xy) != 2:
            raise ValueError("slot_approach.maximum_abs_world_xy_error_m needs two positive numbers")
        result["slot_approach"] = {
            "maximum_abs_world_xy_error_m": [_number(x, "slot_approach.xy", positive=True) for x in xy],
            "maximum_rotation_error_rad": _number(slot["maximum_rotation_error_rad"],
                "slot_approach.maximum_rotation_error_rad", positive=True, maximum=math.pi),
            "minimum_bottom_height_above_slot_m": _number(slot["minimum_bottom_height_above_slot_m"],
                "slot_approach.minimum_bottom_height_above_slot_m", positive=True)}
    if "orientation" in value:
        orientation = value["orientation"]
        vertical = isinstance(orientation, Mapping) and "maximum_abs_vertical_component" in orientation
        required = ({"axis_object", "maximum_abs_vertical_component"} if vertical else
                    {"axis_object", "axis_world", "maximum_angle_rad"})
        _keys(orientation, required, set(), f"{name}.orientation")
        axis = _vector(orientation["axis_object"], "orientation.axis_object", unit=True)
        if vertical:
            result["orientation"] = {"axis_object": axis, "maximum_abs_vertical_component": _number(
                orientation["maximum_abs_vertical_component"], "maximum_abs_vertical_component", minimum=0., maximum=1.)}
        else:
            result["orientation"] = {"axis_object": axis,
                "axis_world": _vector(orientation["axis_world"], "orientation.axis_world", unit=True),
                "maximum_angle_rad": _number(orientation["maximum_angle_rad"], "maximum_angle_rad",
                                              positive=True, maximum=math.pi)}
    if "forbidden_aabbs" in value:
        obstacles = value["forbidden_aabbs"]
        if not isinstance(obstacles, list) or not obstacles:
            raise ValueError(f"{name}.forbidden_aabbs needs a nonempty list")
        result["forbidden_aabbs"], names = [], set()
        for obstacle in obstacles:
            _keys(obstacle, {"id", "minimum_w_m", "maximum_w_m", "margin_m"}, set(), "forbidden AABB")
            identifier = obstacle["id"]
            if not isinstance(identifier, str) or not identifier.strip() or identifier in names:
                raise ValueError("forbidden AABB ids must be nonempty and unique within a rule")
            names.add(identifier)
            low = _vector(obstacle["minimum_w_m"], "forbidden minimum_w_m")
            high = _vector(obstacle["maximum_w_m"], "forbidden maximum_w_m")
            if not np.all(np.asarray(low) < high):
                raise ValueError("forbidden AABB must have positive extent on all three axes")
            margin = _number(obstacle["margin_m"], "forbidden margin_m", minimum=0.)
            if not np.isfinite(np.r_[np.asarray(low) - margin, np.asarray(high) + margin]).all():
                raise ValueError("expanded forbidden AABB must remain finite")
            result["forbidden_aabbs"].append({"id": identifier, "minimum_w_m": low,
                                             "maximum_w_m": high, "margin_m": margin})
    return result


def validate_transport_constraints(settings):
    """Validate an explicit world-frame proxy contract; return an independent normalized copy."""
    _keys(settings, {"schema_version", "coordinate_frame", "geometry", "stage_rules"},
          {"walk_entry_rule"}, "transport_constraints")
    if settings["schema_version"] != SCHEMA_VERSION or settings["coordinate_frame"] != "world":
        raise ValueError("transport_constraints requires its declared schema_version and world coordinate_frame")
    geometry = settings["geometry"]
    if not isinstance(geometry, Mapping):
        raise ValueError("transport_constraints.geometry must be a mapping")
    kind = geometry.get("kind")
    if kind == "vertices":
        _keys(geometry, {"kind", "vertices_object_m"}, set(), "transport geometry")
        vertices = geometry["vertices_object_m"]
        if not isinstance(vertices, list) or len(vertices) < 4:
            raise ValueError("vertices_object_m needs at least four object-local vertices")
        vertices = [_vector(vertex, "vertices_object_m") for vertex in vertices]
        if np.linalg.matrix_rank(np.asarray(vertices) - vertices[0]) < 3:
            raise ValueError("vertices_object_m must span a three-dimensional payload envelope")
        geometry = {"kind": kind, "vertices_object_m": vertices}
    elif kind == "cylinder":
        _keys(geometry, {"kind", "center_object_m", "axis_object", "radius_m", "thickness_m"}, set(),
              "transport geometry")
        geometry = {"kind": kind, "center_object_m": _vector(geometry["center_object_m"], "center_object_m"),
                    "axis_object": _vector(geometry["axis_object"], "geometry.axis_object", unit=True),
                    "radius_m": _number(geometry["radius_m"], "radius_m", positive=True),
                    "thickness_m": _number(geometry["thickness_m"], "thickness_m", positive=True)}
    else:
        raise ValueError("transport geometry kind must be vertices or cylinder")
    rules = settings["stage_rules"]
    if (not isinstance(rules, Mapping) or not {"carry_posture", "navigate_target"} <= set(rules)
            or set(rules) - TRANSPORT_STAGES):
        raise ValueError("stage_rules requires carry_posture and navigate_target, with only transport stages")
    result = {"schema_version": SCHEMA_VERSION, "coordinate_frame": "world", "geometry": geometry,
              "stage_rules": {stage: _rule(rule, f"stage_rules.{stage}") for stage, rule in rules.items()}}
    if "walk_entry_rule" in settings:
        result["walk_entry_rule"] = _rule(settings["walk_entry_rule"], "walk_entry_rule")
    return result


def _world_bounds(geometry, pose):
    rotation, translation = pose[:3, :3], pose[:3, 3]
    if geometry["kind"] == "vertices":
        points = np.asarray(geometry["vertices_object_m"]) @ rotation.T + translation
        low, high = points.min(axis=0), points.max(axis=0)
    else:
        center = rotation @ geometry["center_object_m"] + translation
        axis = rotation @ geometry["axis_object"]
        axis /= np.linalg.norm(axis)
        extent = (geometry["radius_m"] * np.sqrt(np.maximum(0., 1. - axis**2))
                  + .5 * geometry["thickness_m"] * abs(axis))
        low, high = center - extent, center + extent
    if not np.isfinite(np.r_[low, high]).all():
        raise ValueError("payload world bounds must remain finite")
    return low, high


def _measure_rule(rule, low, high, rotation):
    gates, evidence = {}, {}
    if "minimum_world_z_m" in rule:
        margin = float(low[2] - rule["minimum_world_z_m"])
        gates["minimum_world_z"] = margin >= 0.
        evidence["minimum_world_z_margin_m"] = margin
    if "maximum_world_z_m" in rule:
        margin = float(rule["maximum_world_z_m"] - high[2])
        gates["maximum_world_z"] = margin >= 0.
        evidence["maximum_world_z_margin_m"] = margin
    if "orientation" in rule:
        orientation = rule["orientation"]
        axis = rotation @ orientation["axis_object"]
        axis /= np.linalg.norm(axis)
        if "maximum_abs_vertical_component" in orientation:
            measured = float(abs(axis[2]))
            gates["orientation"] = measured <= orientation["maximum_abs_vertical_component"]
            evidence["axis_abs_vertical_component"] = measured
        else:
            target = np.asarray(orientation["axis_world"])
            angle = math.acos(float(np.clip(axis @ target / np.linalg.norm(target), -1., 1.)))
            gates["orientation"] = angle <= orientation["maximum_angle_rad"]
            evidence["axis_angle_rad"] = angle
        evidence["measured_axis_w"] = axis.tolist()
    if "forbidden_aabbs" in rule:
        evidence["forbidden_aabbs"] = []
        for obstacle in rule["forbidden_aabbs"]:
            bottom = np.asarray(obstacle["minimum_w_m"]) - obstacle["margin_m"]
            top = np.asarray(obstacle["maximum_w_m"]) + obstacle["margin_m"]
            # Positive on any axis proves disjoint envelopes. Touching is unsafe.
            separation = np.maximum(bottom - high, low - top)
            safe = bool(np.any(separation > 0.))
            gates["forbidden_aabb:" + obstacle["id"]] = safe
            evidence["forbidden_aabbs"].append({"id": obstacle["id"], "separated": safe,
                "axis_separation_m": separation.tolist(), "expanded_minimum_w_m": bottom.tolist(),
                "expanded_maximum_w_m": top.tolist()})
    return {"valid": all(gates.values()), "gates": gates, **evidence}


def evaluate_payload_transport(settings, object_pose_w, stage_id, *, at_walk_entry=False, slot_target_pose_w=None):
    """Evaluate measured geometry. Missing stage rules are explicitly unchecked.

    ``at_walk_entry`` combines the navigate_target rule and optional cart-exit
    rule on the *same measured pose*, before any walking command is emitted.
    A transfer ``slot_approach`` requires the episode-anchored original slot
    target, never the current object pose or a target read from command claims.
    It can lower only the existing minimum height inside its XY/orientation
    zone; every other orientation and forbidden-volume rule still applies.
    Malformed measured poses fail closed; malformed configuration raises.
    """
    settings = validate_transport_constraints(settings)
    if at_walk_entry and stage_id != "navigate_target":
        raise ValueError("at_walk_entry is only valid for navigate_target")
    rules = {}
    if stage_id in settings["stage_rules"]:
        rules[stage_id] = settings["stage_rules"][stage_id]
    if at_walk_entry and "walk_entry_rule" in settings:
        rules["walk_entry_rule"] = settings["walk_entry_rule"]
    result = {"schema_version": SCHEMA_VERSION, "coordinate_frame": "world", "stage_id": stage_id,
              "at_walk_entry": bool(at_walk_entry), "checked": bool(rules), "valid": True,
              "geometry_kind": settings["geometry"]["kind"], "proxy_enclosure_verified": False,
              "full_mesh_collision_checked": False, "swept_collision_checked": False,
              "collision_method": "conservative_payload_world_aabb_vs_declared_forbidden_aabbs",
              "rules": {}, "violations": []}
    if not rules:
        return result
    try:
        pose = pose_to_matrix(object_pose_w)
        low, high = _world_bounds(settings["geometry"], pose)
    except (ValueError, TypeError, OverflowError):
        result.update(valid=False, violations=["invalid_measured_payload_pose"])
        return result
    result.update(measured_object_pose_w=deepcopy(list(map(float, object_pose_w))),
                  minimum_w_m=low.tolist(), maximum_w_m=high.tolist())
    for name, rule in rules.items():
        effective_rule, slot_evidence = rule, None
        if "slot_approach" in rule:
            try:
                target = pose_to_matrix(slot_target_pose_w)
                target_low, _ = _world_bounds(settings["geometry"], target)
            except (ValueError, TypeError, OverflowError):
                result.update(valid=False, violations=[name + ":invalid_slot_target_pose"])
                return result
            slot = rule["slot_approach"]
            xy = abs(pose[:2, 3] - target[:2, 3])
            angle = math.acos(float(np.clip((np.trace(target[:3, :3].T @ pose[:3, :3]) - 1.) / 2., -1., 1.)))
            eligible = bool(np.all(xy <= slot["maximum_abs_world_xy_error_m"])
                            and angle <= slot["maximum_rotation_error_rad"])
            candidate_minimum = float(target_low[2] + slot["minimum_bottom_height_above_slot_m"])
            effective_minimum = (min(rule["minimum_world_z_m"], candidate_minimum)
                                 if eligible else rule["minimum_world_z_m"])
            effective_rule = {**rule, "minimum_world_z_m": effective_minimum}
            slot_evidence = {
                "slot_target_pose_w": deepcopy(list(map(float, slot_target_pose_w))),
                "absolute_world_xy_error_m": xy.tolist(), "rotation_error_rad": angle,
                "eligible": eligible, "target_payload_bottom_world_z_m": float(target_low[2]),
                "slot_minimum_world_z_m": candidate_minimum,
                "original_minimum_world_z_m": rule["minimum_world_z_m"],
                "effective_minimum_world_z_m": effective_minimum,
                "height_bound_lowered": effective_minimum < rule["minimum_world_z_m"],
                "target_source": "caller_supplied_episode_anchored_original_slot"}
        measured = _measure_rule(effective_rule, low, high, pose[:3, :3])
        if slot_evidence is not None:
            measured["slot_approach"] = slot_evidence
        result["rules"][name] = {**measured, "limits": deepcopy(rule)}
        result["violations"].extend(name + ":" + gate for gate, valid in measured["gates"].items() if not valid)
    result["valid"] = not result["violations"]
    return result
