"""Read cooked rigid-body properties without importing Isaac or changing physics.

Call after physics reset, while the supplied simulation view is valid, in the
NEPES metre/kilogram scene. Results contain plain Python data and retain no native
views. Single-body views avoid material-array padding across unequal bodies.

API contract verified against installed omni.physics.tensors 107.3.26:
``RigidBodyView.get_coms`` returns xyz+xyzw in the rigid-body-prim frame;
``get_material_properties`` returns (count, max_shapes, 3), ordered static
friction, dynamic friction, restitution. ``max_shapes`` is the actual shape
count for a view containing exactly one body, including cooked convex pieces.
Shape indices have no exposed USD-path mapping in this API.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.array(value, dtype=float, copy=True)


def _unavailable(error: Exception) -> dict:
    # Do not retain exceptions/tracebacks that can keep native views alive.
    return {"available": False, "reason": f"{type(error).__name__}: {error}"}


def _values(view: Any, method: str, shape: tuple[int, ...]) -> np.ndarray:
    values = _numpy(getattr(view, method)())
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError(f"{method} expected finite shape {shape}, got {values.shape}")
    return values


def _read_body(simulation_view: Any, path: str) -> dict:
    view = None
    try:
        view = simulation_view.create_rigid_body_view(path)
        resolved = list(view.prim_paths)
        if view.count != 1 or resolved != [path]:
            raise ValueError(f"Expected exactly {path}; resolved count={view.count}, paths={resolved}")
    except Exception as error:
        return _unavailable(error)
    try:
        result = {"prim_path": path, "resolved_prim_path": resolved[0]}
        try:
            mass = float(_values(view, "get_masses", (1, 1))[0, 0])
            if mass <= 0:
                raise ValueError("Dynamic/articulation body mass must be positive")
            result["mass"] = {"available": True, "mass_kg": mass}
        except Exception as error:
            result["mass"] = _unavailable(error)
        try:
            pose = _values(view, "get_coms", (1, 7))[0]
            norm = float(np.linalg.norm(pose[3:]))
            if abs(norm - 1.) > 1e-4:
                raise ValueError(f"COM principal-axes quaternion is not unit length: {norm}")
            result["com"] = {
                "available": True,
                "frame": "rigid_body_prim_local",
                "translation_m": pose[:3].tolist(),
                "principal_axes_quaternion_wxyz": pose[[6, 3, 4, 5]].tolist(),
                "raw_tensor_pose_xyz_xyzw": pose.tolist(),
            }
        except Exception as error:
            result["com"] = _unavailable(error)
        try:
            count = view.max_shapes
            if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 0:
                raise ValueError(f"Invalid cooked shape count: {count!r}")
            count = int(count)
            materials = _values(view, "get_material_properties", (1, count, 3))[0]
            if np.any(materials < 0) or np.any(materials[:, 2] > 1):
                raise ValueError("Friction must be nonnegative and restitution must be in [0, 1]")
            result["materials"] = {
                "available": True,
                "shape_count": count,
                "shape_index_basis": "cooked PhysX shape order within this exact single-body view",
                "shape_prim_paths_available": False,
                "combine_modes_available": False,
                "shapes": [
                    {"shape_index": index, "static_friction": float(row[0]),
                     "dynamic_friction": float(row[1]), "restitution": float(row[2])}
                    for index, row in enumerate(materials)
                ],
            }
        except Exception as error:
            result["materials"] = _unavailable(error)
        result["available"] = all(result[key]["available"] for key in ("mass", "com", "materials"))
        return result
    finally:
        # This helper owns no persistent views and must finish before app close.
        view = None


def read_rigid_body_properties(simulation_view: Any, paths: Sequence[str]) -> dict:
    """Read exact hand/finger/object body paths, reporting partial API failures.

    Uses the existing simulation view and only getters. This is intended for
    the current SI-unit scene; it does not convert arbitrary USD stage units.
    No world COM is inferred from USD, and no authored material substitutes for
    missing cooked data. Call again after any material/mass randomization.
    """
    if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence):
        raise ValueError("paths must be a sequence of exact absolute rigid-body paths")
    paths = list(paths)
    for path in paths:
        if (not isinstance(path, str) or not path.startswith("/") or path.endswith("/")
                or "//" in path or any(c in path for c in "*?[]()|+^$\\.{} ") or path == "/"):
            raise ValueError(f"Expected an exact absolute prim path, got {path!r}")
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate rigid-body paths are not permitted")
    bodies = {path: _read_body(simulation_view, path) for path in paths}
    return {
        "schema_version": "igris_c_physx_body_properties_v1",
        "source": "omni.physics.tensors.RigidBodyView getters",
        "units": {"mass": "kg", "translation": "m", "material_coefficients": "dimensionless"},
        "unit_contract": "Caller uses the NEPES metre/kilogram scene; no stage-unit conversion performed",
        "physics_writes_performed": False,
        "all_requested_available": all(body["available"] for body in bodies.values()),
        "bodies": bodies,
    }


_JOINT_DRIVE_SOURCES = {
    "isaaclab_articulation_data": {"stiffness": "joint_stiffness", "damping": "joint_damping",
                                   "effort_limit": "joint_effort_limits", "velocity_limit": "joint_vel_limits"},
    "physx_articulation_view": {"stiffness": "get_dof_stiffnesses", "damping": "get_dof_dampings",
                                "effort_limit": "get_dof_max_forces", "velocity_limit": "get_dof_max_velocities"},
}


def read_joint_drives(articulation: Any, joint_names: Sequence[str]) -> dict:
    """Drive gains and limits of named revolute joints as applied, without any write.

    Reads Isaac Lab's articulation data (what its actuators wrote) and the PhysX
    articulation view getters (what the solver holds); implicit actuators set both,
    explicit ones leave zero PhysX stiffness/damping. A failing source is reported
    unavailable instead of raised. Single-environment articulation (row 0).
    """
    names = list(articulation.joint_names)
    if isinstance(joint_names, (str, bytes)) or not set(joint_names) <= set(names) or len(set(joint_names)) != len(joint_names):
        raise ValueError("joint_names must be distinct joints of the articulation")
    indices = [names.index(name) for name in joint_names]
    sources = {}
    for source, fields in _JOINT_DRIVE_SOURCES.items():
        try:
            owner = articulation.data if source == "isaaclab_articulation_data" else articulation.root_physx_view
            values = {}
            for key, attribute in fields.items():
                value = getattr(owner, attribute)
                row = _numpy(value() if callable(value) else value)[0]
                if row.shape != (len(names),) or not np.isfinite(row[indices]).all():
                    raise ValueError(f"{attribute} expected {len(names)} finite joint values, got {row.shape}")
                values[key] = row[indices].tolist()
            sources[source] = {"available": True, **{key: dict(zip(joint_names, v)) for key, v in values.items()}}
        except Exception as error:
            sources[source] = _unavailable(error)
    return {
        "schema_version": "igris_c_joint_drive_readback_v1",
        "units": {"stiffness": "N*m/rad", "damping": "N*m*s/rad", "effort_limit": "N*m", "velocity_limit": "rad/s"},
        "joint_names": list(joint_names), "physics_writes_performed": False,
        "all_requested_available": all(entry["available"] for entry in sources.values()),
        **sources,
    }
