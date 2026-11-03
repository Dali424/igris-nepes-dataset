"""Measured rigid-object relations with explicit orientation semantics.

The default compares the full SO(3) rotation. ``directed_axis`` leaves rotation
around a configured object-local axis unconstrained, while rejecting a flipped
axis. This is a task specification, not a claim of geometric symmetry or a
replacement for hand capture, collision, support, or final placement checks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from .snapshot import matrix_to_pose


def validate_relation_rotation(relation: Mapping) -> tuple[str, np.ndarray]:
    """Validate without inserting defaults or changing a configuration hash."""
    mode = relation.get("rotation_constraint", "full_so3")
    if mode not in ("full_so3", "directed_axis"):
        raise ValueError("rotation_constraint must be full_so3 or directed_axis")
    if mode == "full_so3":
        if "local_axis_object" in relation:
            raise ValueError("local_axis_object requires rotation_constraint=directed_axis")
        # This axis is diagnostic only for legacy/full-SO(3) relations.
        return mode, np.array([0., 0., 1.])
    raw = relation.get("local_axis_object")
    try:
        axis = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("local_axis_object must contain three finite unit-axis values") from exc
    if (axis.shape != (3,) or not np.isfinite(axis).all()
            or any(isinstance(v, (bool, np.bool_)) for v in raw)
            or abs(float(np.linalg.norm(axis)) - 1.) > 1e-6):
        raise ValueError("local_axis_object must be a finite unit vector")
    return mode, axis / np.linalg.norm(axis)


def _rigid(value, label):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0., 0., 0., 1.], atol=1e-8, rtol=0)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or abs(float(np.linalg.det(matrix[:3, :3])) - 1.) > 1e-6):
        raise ValueError(f"{label} must be a finite rigid transform")
    return matrix


def measure_object_relation(actual_relative, expected_relative, relation: Mapping) -> dict:
    """Evaluate a relation and retain full rotation, directed-axis and twist data.

    Twist is the signed swing/twist component about the configured object-local
    axis in the expected orientation. It is undefined for an exact 180-degree
    swing about a perpendicular axis; such a flipped normal is still rejected.
    """
    mode, axis = validate_relation_rotation(relation)
    actual = _rigid(actual_relative, "actual_relative")
    expected = _rigid(expected_relative, "expected_relative")
    tolerances = []
    for name in ("position_tolerance_m", "rotation_tolerance_rad"):
        value = relation.get(name)
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be positive and finite")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be positive and finite") from exc
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"{name} must be positive and finite")
        tolerances.append(number)
    relative_rotation = expected[:3, :3].T @ actual[:3, :3]
    full_rotation = float(math.acos(np.clip((np.trace(relative_rotation) - 1.) / 2., -1., 1.)))
    expected_axis = expected[:3, :3] @ axis
    actual_axis = actual[:3, :3] @ axis
    axis_error = float(math.acos(np.clip(np.dot(expected_axis, actual_axis), -1., 1.)))
    delta = np.eye(4)
    delta[:3, :3] = relative_rotation
    quaternion = matrix_to_pose(delta)[3:]
    projected = float(np.dot(quaternion[1:], axis))
    twist_defined = math.hypot(quaternion[0], projected) > 1e-10
    raw_twist = 2. * math.atan2(projected, quaternion[0]) if twist_defined else None
    twist = math.atan2(math.sin(raw_twist), math.cos(raw_twist)) if twist_defined else None
    position = float(np.linalg.norm(actual[:3, 3] - expected[:3, 3]))
    selected_rotation = axis_error if mode == "directed_axis" else full_rotation
    position_ok = position <= tolerances[0]
    rotation_ok = selected_rotation <= tolerances[1]
    return {
        "rotation_constraint": mode,
        "local_axis_object": axis.tolist(),
        "axis_is_diagnostic_only": mode == "full_so3",
        "position_error_m": position,
        "rotation_error_rad": selected_rotation,
        "full_rotation_error_rad": full_rotation,
        "directed_axis_error_rad": axis_error,
        "in_plane_twist_rad": twist,
        "twist_defined": twist_defined,
        "actual_axis_in_reference": actual_axis.tolist(),
        "expected_axis_in_reference": expected_axis.tolist(),
        "position_tolerance_m": tolerances[0],
        "rotation_tolerance_rad": tolerances[1],
        "position_ok": position_ok,
        "rotation_ok": rotation_ok,
        "valid": position_ok and rotation_ok,
    }
