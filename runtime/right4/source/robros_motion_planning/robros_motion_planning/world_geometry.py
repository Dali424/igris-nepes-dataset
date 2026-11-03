"""Reduce frozen compound obstacle count by validated solid boolean union.

The snapshot remains unchanged. Only objects with at least 16 individually
watertight, positive-volume colliders are candidates. A failed union keeps every
original collider. This is a set union, never a convex hull or bounding box, so
holes and passages are retained. Derived meshes and source hashes are saved for
inspection. No cuRobo, CUDA, or Isaac imports are needed here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import importlib.metadata
import json
import re

import numpy as np

from .snapshot import pose_to_matrix, snapshot_digest


MIN_UNION_COLLIDERS = 16


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_volume(mesh: Any) -> bool:
    return bool(
        len(mesh.vertices) >= 4 and len(mesh.faces) >= 4
        and np.isfinite(mesh.vertices).all()
        and mesh.is_volume and np.isfinite(mesh.volume) and mesh.volume > 0
    )


def _exact_union(meshes: list[Any], trimesh: Any) -> tuple[Any, str]:
    """Use manifold solid union, preserving topology when trimesh processing fails.

    trimesh 4.5.1's manifold wrapper automatically merges coincident vertices on
    return. That can make touching solids nonmanifold despite a valid manifold
    result. The fallback uses the same engine's float64 API and disables that
    postprocessing. Neither path repairs, simplifies, or convexifies geometry.
    """
    wrapper_failure = None
    try:
        result = trimesh.boolean.union(meshes, engine="manifold", check_volume=True)
        if _is_volume(result):
            return result, "trimesh.boolean.union:manifold"
        wrapper_failure = "trimesh manifold wrapper returned a non-volume mesh"
    except Exception as error:
        wrapper_failure = f"{type(error).__name__}: {error}"
    try:
        import manifold3d

        manifolds = [
            manifold3d.Manifold(mesh=manifold3d.Mesh64(
                vert_properties=np.asarray(mesh.vertices, dtype=np.float64),
                tri_verts=np.asarray(mesh.faces, dtype=np.uint64),
            )) for mesh in meshes
        ]
        if any(mesh.status() != manifold3d.Error.NoError for mesh in manifolds):
            raise ValueError("manifold rejected an input volume")
        result = manifold3d.Manifold.batch_boolean(manifolds, manifold3d.OpType.Add)
        if result.status() != manifold3d.Error.NoError:
            raise ValueError(f"manifold union failed: {result.status()}")
        raw = result.to_mesh64()
        mesh = trimesh.Trimesh(vertices=raw.vert_properties, faces=raw.tri_verts, process=False)
        if not _is_volume(mesh):
            raise ValueError("manifold union did not produce a watertight positive volume")
        return mesh, "manifold3d.Mesh64.batch_boolean:preserve_topology"
    except Exception as error:
        raise ValueError(f"{wrapper_failure}; topology-preserving union failed: {error}") from error


def prepare_world(
    snapshot: dict[str, Any],
    snapshot_path: str | Path,
    output_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a cuRobo world dictionary and auditable union metadata.

    Input meshes are verified against their snapshot hashes even when they are
    not merged. Source corruption is an error; only optimization failures fall
    back to original geometry. Source poses are applied before each union.

    Derived meshes are passed to cuRobo as explicit vertices/faces: cuRobo 0.7.8
    otherwise reloads mesh files with trimesh's automatic vertex merging, which
    can damage the topology of solids touching at edges or points.
    """
    import trimesh

    snapshot_path = Path(snapshot_path).expanduser().resolve()
    source_directory = snapshot_path.parent
    output_dir = Path(output_dir).expanduser().resolve()
    if snapshot.get("snapshot_sha256") != snapshot_digest(snapshot):
        raise ValueError("Snapshot SHA256 mismatch before world preparation")
    records = snapshot["world"]["mesh"]
    by_name = {record["name"]: record for record in records}
    if len(by_name) != len(records) or not records:
        raise ValueError("Collision records must be nonempty and uniquely named")
    source_paths = {}
    raw_world = {}
    for record in records:
        relative = Path(record["file_path"])
        path = (source_directory / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(source_directory):
            raise ValueError(f"Collision path escapes snapshot directory: {relative}")
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"Collision mesh SHA256 mismatch: {relative}")
        pose_to_matrix(record["pose"])
        source_paths[record["name"]] = path
        raw_world[record["name"]] = {"file_path": str(path), "pose": record["pose"]}
    try:
        manifold_version = importlib.metadata.version("manifold3d")
    except importlib.metadata.PackageNotFoundError:
        manifold_version = None
    metadata: dict[str, Any] = {
        "schema_version": "igris_c_world_union_v1",
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "operation": "solid_boolean_union_per_frozen_object",
        "minimum_colliders_per_object": MIN_UNION_COLLIDERS,
        "trimesh_version": trimesh.__version__, "manifold3d_version": manifold_version,
        "input_mesh_count": len(records), "groups": [],
        "unchanged_sources": [],
        "assumption": "All object transforms remain fixed at the source planning snapshot.",
    }
    used_names: set[str] = set()
    merged_names: set[str] = set()
    for object_path, obj in sorted(snapshot.get("objects", {}).items()):
        names = list(obj["colliders"])
        if len(names) != len(set(names)) or any(name not in by_name for name in names):
            raise ValueError(f"Invalid collider membership for {object_path}")
        if used_names.intersection(names):
            raise ValueError(f"A collider belongs to multiple object groups: {object_path}")
        used_names.update(names)
        if len(names) < MIN_UNION_COLLIDERS:
            continue
        sources = [{"name": name, "file_path": by_name[name]["file_path"],
                    "sha256": by_name[name]["sha256"], "pose": by_name[name]["pose"]}
                   for name in names]
        source_digest = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        group: dict[str, Any] = {
            "object_path": object_path, "sources": sources,
            "sources_sha256": source_digest, "input_mesh_count": len(names),
        }
        try:
            meshes = []
            for name in names:
                mesh = trimesh.load(str(source_paths[name]), force="mesh", process=False)
                if not _is_volume(mesh):
                    raise ValueError(f"Source collider is not a watertight positive volume: {name}")
                mesh.apply_transform(pose_to_matrix(by_name[name]["pose"]))
                meshes.append(mesh)
            union, engine = _exact_union(meshes, trimesh)
            # A union's bounds are exactly the extrema of its inputs, and its
            # volume cannot exceed their sum. Allow float32 wrapper roundoff.
            source_bounds = np.asarray([mesh.bounds for mesh in meshes])
            expected_bounds = np.stack((source_bounds[:, 0].min(axis=0), source_bounds[:, 1].max(axis=0)))
            source_volume = float(sum(mesh.volume for mesh in meshes))
            if not np.allclose(union.bounds, expected_bounds, atol=2e-6, rtol=1e-7):
                raise ValueError("Union changed the source geometry bounds")
            volume_tolerance = max(source_volume * 1e-4, 1e-12)
            if union.volume > source_volume + volume_tolerance or union.volume < max(mesh.volume for mesh in meshes) - volume_tolerance:
                raise ValueError("Union volume is inconsistent with source volumes")
            content = trimesh.exchange.obj.export_obj(
                union, include_normals=False, include_color=False,
                include_texture=False, digits=17,
            ).encode("utf-8")
            derived_digest = hashlib.sha256(content).hexdigest()
            safe_name = re.sub("[^A-Za-z0-9_]", "_", object_path.strip("/"))
            union_name = f"union_{safe_name}_{source_digest[:12]}"
            if union_name in raw_world:
                raise ValueError(f"Union mesh name collides with source mesh: {union_name}")
            derived_dir = output_dir / "world_unions"
            derived_dir.mkdir(parents=True, exist_ok=True)
            mesh_path = derived_dir / f"{union_name}_{derived_digest[:12]}.obj"
            mesh_path.write_bytes(content)
            # Do not point cuRobo at mesh_path: its default file loader merges
            # distinct coincident vertices. The direct data path preserves them.
            raw_world[union_name] = {
                "vertices": union.vertices.tolist(), "faces": union.faces.tolist(),
                "pose": [0, 0, 0, 1, 0, 0, 0],
            }
            for name in names:
                del raw_world[name]
            merged_names.update(names)
            group.update({
                "status": "merged", "engine": engine, "output_mesh_name": union_name,
                "output_file_path": mesh_path.relative_to(output_dir).as_posix(),
                "output_sha256": derived_digest,
                "output_mesh_count": 1, "watertight": True, "is_volume": True,
                "input_volume_sum_m3": source_volume, "union_volume_m3": float(union.volume),
                "input_face_count": sum(len(mesh.faces) for mesh in meshes),
                "output_face_count": len(union.faces),
                "curobo_geometry_transport": "inline_vertices_faces_preserve_topology",
            })
        except Exception as error:
            # Never turn missing optional dependencies, invalid topology or a
            # failed optimization into missing obstacles.
            group.update({"status": "kept_original", "output_mesh_count": len(names),
                          "reason": f"{type(error).__name__}: {error}"})
        metadata["groups"].append(group)
    metadata["output_mesh_count"] = len(raw_world)
    metadata["unchanged_sources"] = [
        {"name": record["name"], "file_path": record["file_path"], "sha256": record["sha256"]}
        for record in records if record["name"] not in merged_names
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "world_geometry_manifest.json"
    manifest_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {"mesh": raw_world}, metadata
