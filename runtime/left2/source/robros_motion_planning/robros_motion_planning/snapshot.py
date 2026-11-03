"""Portable, hashed collision snapshots from a synchronized USD reset state.

This module imports neither Isaac Sim nor USD at module import time. The caller
must reset the simulation and synchronize USD transforms (disable Fabric) before
capture. Capturing a running simulation from USD alone is not supported: dynamic
object transforms may otherwise lag the physics state.

Meshes contain vertices in the robot URDF ``base_link`` frame, with identity
poses. Only enabled CollisionAPI geometry is exported; render geometry, including
NuRec, does not become an obstacle merely because it is visible. Convex physics
approximations are conservatively enclosed by a convex hull. Triangle meshes keep
their openings, notably the cleaner cavity and the space below the cart.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import re

import numpy as np


SCHEMA_VERSION = "igris_c_planning_snapshot_v1"


def pose_to_matrix(pose: Any) -> np.ndarray:
    """Convert an xyz + scalar-first quaternion pose to a column-vector matrix."""
    value = np.asarray(pose, dtype=float)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("Pose must contain seven finite values [x,y,z,qw,qx,qy,qz]")
    norm = np.linalg.norm(value[3:])
    if not np.isclose(norm, 1.0, atol=1e-3, rtol=0):
        raise ValueError(f"Pose quaternion is not normalized: norm={norm}")
    w, x, y, z = value[3:] / norm
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    matrix[:3, 3] = value[:3]
    return matrix


def matrix_to_pose(matrix: Any) -> list[float]:
    """Convert a rigid column-vector matrix to xyz + scalar-first quaternion."""
    value = np.asarray(matrix, dtype=float)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("Transform must be a finite 4x4 matrix")
    rotation = value[:3, :3]
    if not np.allclose(value[3], [0, 0, 0, 1], atol=1e-7) or not np.allclose(
        rotation.T @ rotation, np.eye(3), atol=1e-6
    ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError("Transform must be rigid; scale/shear cannot be represented as a pose")
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = 2 * math.sqrt(trace + 1)
        quaternion = np.array([
            scale / 4, (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        ])
    else:
        index = int(np.argmax(np.diag(rotation)))
        following, last = (index + 1) % 3, (index + 2) % 3
        scale = 2 * math.sqrt(1 + rotation[index, index] - rotation[following, following] - rotation[last, last])
        quaternion = np.empty(4)
        quaternion[0] = (rotation[last, following] - rotation[following, last]) / scale
        quaternion[index + 1] = scale / 4
        quaternion[following + 1] = (rotation[following, index] + rotation[index, following]) / scale
        quaternion[last + 1] = (rotation[last, index] + rotation[index, last]) / scale
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0:
        quaternion *= -1
    return [*value[:3, 3].tolist(), *quaternion.tolist()]


def snapshot_digest(snapshot: Mapping[str, Any]) -> str:
    """Canonical SHA256 excluding only the self-referential digest field."""
    document = {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _rigid_part(matrix: np.ndarray) -> np.ndarray:
    """Remove authored scale while retaining the body's translation/orientation."""
    left, _, right = np.linalg.svd(matrix[:3, :3])
    rigid = matrix.copy()
    rigid[:3, :3] = left @ np.diag([1.0, 1.0, np.linalg.det(left @ right)]) @ right
    return rigid


def _live_world_transform(prim, usd_world, overrides, xform_cache, UsdPhysics):
    if overrides is None:
        return usd_world
    ancestor = prim
    while ancestor.IsValid() and not ancestor.IsPseudoRoot():
        if ancestor.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(ancestor).GetRigidBodyEnabledAttr().Get() is not False:
            path = str(ancestor.GetPath())
            if path not in overrides:
                raise ValueError(f"Live snapshot has no PhysX pose for dynamic body {path}")
            authored = np.asarray(xform_cache.GetLocalToWorldTransform(ancestor), dtype=float).T
            return pose_to_matrix(overrides[path]) @ np.linalg.inv(_rigid_part(authored)) @ usd_world
        ancestor = ancestor.GetParent()
    return usd_world


def _triangulate(counts: Any, indices: Any, point_count: int, holes: Any = ()) -> np.ndarray:
    """Fan-triangulate authored faces, rejecting malformed USD topology."""
    counts = np.asarray(counts, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    if counts.ndim != 1 or indices.ndim != 1 or np.any(counts < 3) or counts.sum() != len(indices):
        raise ValueError("Mesh face counts/indices are malformed")
    if len(indices) == 0 or np.any(indices < 0) or np.any(indices >= point_count):
        raise ValueError("Mesh has no faces or out-of-range indices")
    hole_set = set(int(index) for index in holes)
    faces = []
    offset = 0
    for face_index, count in enumerate(counts):
        polygon = indices[offset:offset + count]
        if face_index not in hole_set:
            faces.extend((int(polygon[0]), int(polygon[i]), int(polygon[i + 1])) for i in range(1, int(count) - 1))
        offset += int(count)
    if not faces:
        raise ValueError("Mesh has no non-hole faces")
    return np.asarray(faces, dtype=np.int64)


def _outer_sphere(radius: float, trimesh: Any) -> Any:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    # An ordinary polygon sphere is inscribed. Move every face outside the true
    # sphere so the exported collision shape cannot underestimate its surface.
    distances = np.abs(np.einsum("ij,ij->i", mesh.face_normals, mesh.triangles[:, 0]))
    mesh.apply_scale(radius / float(distances.min()))
    return mesh


def _collider_mesh(prim: Any, UsdGeom: Any, UsdPhysics: Any, trimesh: Any) -> tuple[Any, str]:
    kind = prim.GetTypeName()
    approximation = "primitive"
    if prim.IsA(UsdGeom.Mesh):
        geometry = UsdGeom.Mesh(prim)
        points = np.asarray(geometry.GetPointsAttr().Get(), dtype=float)
        if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
            raise ValueError("Mesh points are missing or nonfinite")
        faces = _triangulate(geometry.GetFaceVertexCountsAttr().Get(), geometry.GetFaceVertexIndicesAttr().Get(), len(points), geometry.GetHoleIndicesAttr().Get() or ())
        if geometry.GetOrientationAttr().Get() == "leftHanded":
            faces = faces[:, ::-1]
        mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
        approximation = str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() or "none")
        if approximation in ("convexHull", "convexDecomposition", "meshSimplification"):
            mesh = mesh.convex_hull
        elif approximation == "boundingCube":
            bounds = mesh.bounds
            mesh = trimesh.creation.box(extents=bounds[1] - bounds[0])
            mesh.apply_translation(bounds.mean(axis=0))
        elif approximation == "boundingSphere":
            center = mesh.bounds.mean(axis=0)
            mesh = _outer_sphere(float(np.linalg.norm(mesh.vertices - center, axis=1).max()), trimesh)
            mesh.apply_translation(center)
        elif approximation != "none":
            raise ValueError(f"Unsupported physics mesh approximation {approximation!r}")
    elif prim.IsA(UsdGeom.Cube):
        size = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
        if not math.isfinite(size) or size <= 0:
            raise ValueError("Cube size must be positive and finite")
        mesh = trimesh.creation.box(extents=[size] * 3)
    elif kind in ("Sphere", "Capsule", "Cylinder"):
        geometry = getattr(UsdGeom, kind)(prim)
        radius = float(geometry.GetRadiusAttr().Get())
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("Primitive radius must be positive and finite")
        if kind == "Sphere":
            mesh = _outer_sphere(radius, trimesh)
        else:
            height = float(geometry.GetHeightAttr().Get())
            if not math.isfinite(height) or height <= 0:
                raise ValueError("Primitive height must be positive and finite")
            if kind == "Cylinder":
                mesh = trimesh.creation.cylinder(radius=radius / math.cos(math.pi / 32), height=height, sections=32)
            else:
                sphere = _outer_sphere(radius, trimesh)
                vertices = np.concatenate((sphere.vertices + [0, 0, height / 2], sphere.vertices - [0, 0, height / 2]))
                mesh = trimesh.convex.convex_hull(vertices)
            axis = str(geometry.GetAxisAttr().Get())
            if axis == "X":
                mesh.vertices = mesh.vertices[:, [2, 0, 1]]
            elif axis == "Y":
                mesh.vertices = mesh.vertices[:, [1, 2, 0]]
            elif axis != "Z":
                raise ValueError(f"Unsupported primitive axis {axis!r}")
    else:
        raise ValueError(f"Unsupported collision geometry {kind!r}")
    if len(mesh.vertices) < 3 or len(mesh.faces) < 1:
        raise ValueError("Empty collider")
    return mesh, approximation


def capture_snapshot(
    stage: Any,
    robot: Any,
    output_path: str | Path,
    *,
    scene_usd: str,
    scene_root: str = "/World/Scene",
    robot_root: str = "/World/envs/env_0/Robot",
    rigid_body_poses_w: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture the reset robot and all enabled non-robot USD collision geometry.

    Without overrides, call immediately after reset with synchronized USD transforms.
    With ``rigid_body_poses_w``, every enabled environment rigid body's actor pose
    must come from PhysX tensors at the current paused simulation state. Collider
    transforms are corrected numerically; no USD or physics objects are moved.
    Unknown
    collider types fail the entire capture; no incomplete snapshot is published.
    Mesh files are stored beside the JSON with relative paths and SHA256 hashes.
    """
    from pxr import Usd, UsdGeom, UsdPhysics
    import trimesh

    output_path = Path(output_path).expanduser().resolve()
    scene_path = Path(scene_usd).expanduser().resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)
    scene_prim = stage.GetPrimAtPath(scene_root)
    if not scene_prim.IsValid() or not scene_prim.IsActive():
        raise ValueError(f"Active scene root not found: {scene_root}")
    if not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0) or UsdGeom.GetStageUpAxis(stage) != "Z":
        raise ValueError("Planning snapshot requires a Z-up USD stage in meters")
    joint_names = list(robot.joint_names)
    positions = _array(robot.data.joint_pos)[0]
    if len(set(joint_names)) != len(joint_names) or positions.shape != (len(joint_names),) or not np.isfinite(positions).all():
        raise ValueError("Robot joint names/positions are malformed")
    root_pose = _array(robot.data.root_pose_w)[0].tolist()
    pose_to_matrix(root_pose)
    body_names = list(robot.body_names)
    body_positions = _array(robot.data.body_pos_w)[0]
    body_quaternions = _array(robot.data.body_quat_w)[0]
    if len(set(body_names)) != len(body_names) or body_positions.shape != (len(body_names), 3) or body_quaternions.shape != (len(body_names), 4):
        raise ValueError("Robot body names/poses are malformed")
    body_poses = {name: [*position.tolist(), *quaternion.tolist()] for name, position, quaternion in zip(body_names, body_positions, body_quaternions)}
    for pose in body_poses.values():
        pose_to_matrix(pose)
    if "base_link" not in body_poses:
        raise ValueError("Robot must expose the URDF base_link body; root alignment cannot be assumed")
    planning_pose = body_poses["base_link"]
    world_to_base = np.linalg.inv(pose_to_matrix(planning_pose))
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    if rigid_body_poses_w is not None:
        for path, pose in rigid_body_poses_w.items():
            body = stage.GetPrimAtPath(path)
            if not body.IsValid() or not body.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError(f"Live pose does not name a USD rigid body: {path}")
            pose_to_matrix(pose)
    meshes = []
    failures = []
    disabled = []
    # Traverse the complete stage to include external terrain. Instance proxies
    # carry their instance-specific world transforms and must be traversed too.
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if _inside(path, robot_root) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            disabled.append(path)
            continue
        try:
            mesh, approximation = _collider_mesh(prim, UsdGeom, UsdPhysics, trimesh)
            # Gf matrices transform row vectors; the numerical helper uses columns.
            local_to_world = np.asarray(xform_cache.GetLocalToWorldTransform(prim), dtype=float).T
            local_to_world = _live_world_transform(prim, local_to_world, rigid_body_poses_w, xform_cache, UsdPhysics)
            mesh.apply_transform(world_to_base @ local_to_world)
            if not np.isfinite(mesh.vertices).all():
                raise ValueError("Nonfinite transformed geometry")
            meshes.append((path, mesh, approximation))
        except Exception as error:
            failures.append(f"{path}: {error}")
    if failures:
        raise ValueError("Unsupported or invalid colliders; refusing an incomplete snapshot:\n" + "\n".join(failures))
    if not meshes:
        raise ValueError("No enabled environment colliders were found")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_directory = output_path.parent / f"{output_path.stem}_meshes"
    mesh_directory.mkdir(exist_ok=True)
    records = []
    objects: dict[str, Any] = {}
    for index, (source_prim, mesh, approximation) in enumerate(meshes):
        name = f"collider_{index:04d}_{re.sub('[^A-Za-z0-9_]', '_', source_prim.rsplit('/', 1)[-1])}"
        content = trimesh.exchange.obj.export_obj(mesh, include_normals=False, include_color=False, include_texture=False, digits=17).encode("utf-8")
        digest = hashlib.sha256(content).hexdigest()
        mesh_path = mesh_directory / f"{name}_{digest[:16]}.obj"
        mesh_path.write_bytes(content)
        records.append({
            "name": name, "file_path": mesh_path.relative_to(output_path.parent).as_posix(),
            "pose": [0, 0, 0, 1, 0, 0, 0], "sha256": digest,
            "source_prim": source_prim, "physics_approximation": approximation,
            "export_approximation": "convexHull" if approximation in ("convexDecomposition", "meshSimplification") else approximation,
            "watertight": bool(mesh.is_watertight),
            "bounds_base": mesh.bounds.tolist(),
        })
        if _inside(source_prim, scene_root):
            relative = source_prim[len(scene_root):].strip("/")
            object_name = relative.split("/")[0]
            object_path = scene_root.rstrip("/") + "/" + object_name
            if object_path not in objects:
                object_prim = stage.GetPrimAtPath(object_path)
                transform = np.asarray(xform_cache.GetLocalToWorldTransform(object_prim), dtype=float).T
                transform = _live_world_transform(object_prim, transform, rigid_body_poses_w, xform_cache, UsdPhysics)
                # Object Xforms may include unit/mesh scale. Keep the full matrix
                # for geometry, and expose a proper rigid orientation for goals.
                rigid = _rigid_part(transform)
                objects[object_path] = {
                    "prim_path": object_path,
                    "pose_w": matrix_to_pose(rigid),
                    "transform_w": transform.tolist(),
                    "bounds_base": mesh.bounds.tolist(),
                    "colliders": [],
                }
            metadata = objects[object_path]
            bounds = np.asarray(metadata["bounds_base"])
            metadata["bounds_base"] = [np.minimum(bounds[0], mesh.bounds[0]).tolist(), np.maximum(bounds[1], mesh.bounds[1]).tolist()]
            metadata["colliders"].append(name)
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "scene_usd": str(scene_path), "scene_usd_sha256": _file_digest(scene_path),
        "scene_root": scene_root, "robot_prim_path": robot_root,
        "capture_contract": "live_physx_actor_poses" if rigid_body_poses_w is not None else "reset_state_usd_transforms_synchronized",
        "coordinate_frame": "base_link", "units": "meters", "quaternion_order": "wxyz",
        "joint_positions": dict(zip(joint_names, positions.tolist())),
        "robot_root_pose_w": root_pose,
        "planning_base_pose_w": planning_pose,
        "body_poses_w": body_poses,
        "world": {"mesh": records}, "objects": objects,
        "disabled_collision_prims": disabled,
    }
    if rigid_body_poses_w is not None:
        snapshot["rigid_body_poses_w"] = {path: list(pose) for path, pose in rigid_body_poses_w.items()}
    snapshot["snapshot_sha256"] = snapshot_digest(snapshot)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return snapshot


def load_snapshot(path: str | Path) -> dict[str, Any]:
    """Read and validate a snapshot and all referenced meshes without importing USD."""
    path = Path(path).expanduser().resolve()
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported planning snapshot schema")
    if snapshot.get("snapshot_sha256") != snapshot_digest(snapshot):
        raise ValueError("Snapshot SHA256 mismatch")
    if snapshot.get("coordinate_frame") != "base_link" or snapshot.get("units") != "meters" or snapshot.get("quaternion_order") != "wxyz":
        raise ValueError("Snapshot must use meters and scalar-first poses in base_link coordinates")
    if snapshot.get("capture_contract") not in ("reset_state_usd_transforms_synchronized", "live_physx_actor_poses"):
        raise ValueError("Snapshot lacks a supported physics-state capture contract")
    if snapshot["capture_contract"] == "live_physx_actor_poses":
        if not isinstance(snapshot.get("rigid_body_poses_w"), dict):
            raise ValueError("Live snapshot is missing rigid body poses")
        for pose in snapshot["rigid_body_poses_w"].values():
            pose_to_matrix(pose)
    pose_to_matrix(snapshot["robot_root_pose_w"])
    pose_to_matrix(snapshot["planning_base_pose_w"])
    joint_positions = snapshot.get("joint_positions")
    if not isinstance(joint_positions, dict) or not joint_positions:
        raise ValueError("Snapshot has no joint positions")
    for name, position in joint_positions.items():
        if not isinstance(name, str) or not isinstance(position, (int, float)) or isinstance(position, bool) or not math.isfinite(position):
            raise ValueError(f"Invalid joint position: {name}")
    body_poses = snapshot.get("body_poses_w", {})
    if "base_link" not in body_poses:
        raise ValueError("Snapshot is missing the base_link body pose")
    for pose in body_poses.values():
        pose_to_matrix(pose)
    if not np.allclose(pose_to_matrix(body_poses["base_link"]), pose_to_matrix(snapshot["planning_base_pose_w"]), atol=1e-7):
        raise ValueError("Planning base pose does not match base_link body pose")
    records = snapshot.get("world", {}).get("mesh", [])
    if not records:
        raise ValueError("Snapshot has no collision meshes")
    names: set[str] = set()
    for record in records:
        name = record["name"]
        if name in names:
            raise ValueError(f"Duplicate collision mesh name: {name}")
        names.add(name)
        relative = Path(record["file_path"])
        mesh_path = (path.parent / relative).resolve()
        if relative.is_absolute() or not mesh_path.is_relative_to(path.parent):
            raise ValueError(f"Collision mesh path escapes snapshot directory: {relative}")
        if _file_digest(mesh_path) != record["sha256"]:
            raise ValueError(f"Collision mesh SHA256 mismatch: {relative}")
        if not np.allclose(pose_to_matrix(record["pose"]), np.eye(4), atol=1e-9):
            raise ValueError("Snapshot meshes must contain base-frame vertices with identity poses")
    return snapshot
