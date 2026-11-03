"""Restore collisions on the NEPES JIG's existing slotted guide meshes.

The source scene binds plastic materials to these two meshes but omits their
collision schemas. Only collision opinions are authored, in the session layer;
the mesh geometry, source USD files, masses and object poses are unchanged.
Convex decomposition still needs a physical settling check: adding its schemas
does not prove that PhysX cooking preserved the narrow slots.

``exact_prisms_v3`` avoids cooking the thin slotted plates altogether: instead
of a convex decomposition of each guide mesh it authors one convex-hull collider
per exact convex prism of the raw 5 mm plate extrusion (package asset
``assets/jig_guide_prisms_v1.json``) and leaves the original guide meshes without
any collision API. The prisms are transform-free descendants of the JIG prim, so
their points are expressed directly in the JIG rigid-body frame.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


GUIDE_MESH_RELATIVE_PATHS = (
    "assm_jig/assm_jig/jig/tn__6_X1kxm3zm9kx3lyK/Mesh",
    "assm_jig/assm_jig/jig_1/tn__6_X1kxm3zm9kx3lyK/Mesh",
)

# Match the high-resolution decomposition already authored on jig_support,
# jig_support_1 and jig_frame in the supplied Scene.usda.
COLLISION_SETTINGS = {
    "physics:approximation": "convexDecomposition",
    "physics:collisionEnabled": True,
    "physxCollision:contactOffset": 0.0005,
    "physxCollision:restOffset": 0.0,
    "physxConvexDecompositionCollision:errorPercentage": 0.1,
    "physxConvexDecompositionCollision:hullVertexLimit": 64,
    "physxConvexDecompositionCollision:maxConvexHulls": 128,
    "physxConvexDecompositionCollision:minThickness": 0.0001,
    "physxConvexDecompositionCollision:shrinkWrap": True,
    "physxConvexDecompositionCollision:voxelResolution": 1_000_000,
}

EXACT_PRISM_PROFILE = "exact_prisms_v3"
COLLISION_PROFILES = ("authored_v1", "high_resolution_v2", EXACT_PRISM_PROFILE)
GUIDE_PRISM_ASSET_PATH = Path(__file__).resolve().parent / "assets" / "jig_guide_prisms_v1.json"
GUIDE_PRISM_ASSET_SCHEMA = "igris_c_jig_guide_prisms_v1"
GUIDE_PRISM_CONTAINER = "GuideCollisionPrisms"
GUIDE_PRISMS_PER_GUIDE = 22
GUIDE_PRISM_MAX_VERTICES = 32
GUIDE_PRISM_VOLUME_TOLERANCE_M3 = 1e-9
# Same contact/rest offsets as the decomposition profiles. Exact convex prisms
# need no decomposition budget; ``profile`` is a marker, never a USD attribute.
EXACT_PRISM_SETTINGS = {
    "profile": EXACT_PRISM_PROFILE,
    "physics:approximation": "convexHull",
    "physics:collisionEnabled": True,
    "physxCollision:contactOffset": COLLISION_SETTINGS["physxCollision:contactOffset"],
    "physxCollision:restOffset": COLLISION_SETTINGS["physxCollision:restOffset"],
}
_GUIDE_COLLISION_SCHEMAS = ("PhysicsCollisionAPI", "PhysicsMeshCollisionAPI",
                            "PhysxCollisionAPI", "PhysxConvexDecompositionCollisionAPI")
_GUIDE_COLLISION_PROPERTY_PREFIXES = ("physics:", "physxCollision:", "physxConvexDecompositionCollision:")


def _collision_profile(config: Mapping[str, Any]) -> str:
    slot_settings = config.get("jig_slots", {})
    if not isinstance(slot_settings, Mapping):
        raise ValueError("jig_slots must be a mapping")
    profile = slot_settings.get("collision_profile", "authored_v1")
    if profile not in COLLISION_PROFILES:
        raise ValueError("jig_slots.collision_profile must be authored_v1, high_resolution_v2 or exact_prisms_v3")
    return profile


def collision_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Choose a bounded cooking profile without altering CAD or global defaults.

    The opt-in profile resolves the millimetre-scale guide slots more faithfully.
    Its physical use still requires a measured support/placement check; it does
    not change material, contact offsets, source topology or object transforms.
    ``exact_prisms_v3`` returns only the shared offsets plus a ``profile`` marker:
    its colliders are exact convex prisms, so no decomposition budget applies.
    """
    profile = _collision_profile(config)
    if profile == EXACT_PRISM_PROFILE:
        return dict(EXACT_PRISM_SETTINGS)
    settings = dict(COLLISION_SETTINGS)
    if profile == "high_resolution_v2":
        settings.update({"physxConvexDecompositionCollision:voxelResolution": 16_000_000,
                         "physxConvexDecompositionCollision:maxConvexHulls": 256,
                         "physxConvexDecompositionCollision:errorPercentage": .01})
    return settings

SLOT_CENTERS_Y_M = (-0.115, -0.070, -0.025, 0.025, 0.070, 0.115)
SLOT_CLEAR_WIDTH_M = 0.010
CRS_THICKNESS_ENVELOPE_M = 0.006
ESTIMATED_SETTLED_CRS_CENTER_Z_M = 0.051955 + math.sqrt(0.127**2 - 0.067724**2)


def vertical_slot_pose(
    slot_index: int = 0, *, center_z_m: float = 0.175,
    normal_direction: str = "negative_y",
) -> list[float]:
    """Return a geometry-derived CRS origin pose in the JIG frame (xyz+wxyz).

    The visible slots constrain the ring in the XZ plane. The CRS origin is on
    one face; its 6 mm mesh envelope is centered in a 10 mm slot. ``center_z_m``
    is an explicit candidate release/reset height, not a measured stable pose.
    No contact or successful placement is inferred by this function.
    """
    if isinstance(slot_index, bool) or not isinstance(slot_index, (int, np.integer)):
        raise ValueError("slot_index must be an integer")
    if not 0 <= slot_index < len(SLOT_CENTERS_Y_M):
        raise ValueError("slot_index must select one of the six measured slots")
    if not math.isfinite(center_z_m) or not 0.127 <= center_z_m < 0.436:
        raise ValueError("center_z_m must place the candidate ring above the floor and below the handle")
    half = math.sqrt(0.5)
    center_y = SLOT_CENTERS_Y_M[slot_index]
    if normal_direction == "negative_y":
        return [0.0, center_y + CRS_THICKNESS_ENVELOPE_M / 2,
                float(center_z_m), half, half, 0.0, 0.0]
    if normal_direction == "positive_y":
        return [0.0, center_y - CRS_THICKNESS_ENVELOPE_M / 2,
                float(center_z_m), 0.0, 0.0, half, half]
    raise ValueError("normal_direction must be negative_y or positive_y")


def _convex_triangle_mesh_volume(points: np.ndarray, faces: np.ndarray, name: str, tol: float = 1e-9) -> float:
    """Return the volume of a closed, outward-wound, strictly convex triangle mesh.

    Raises ValueError for anything else: open or non-manifold edges, inconsistent
    winding, degenerate triangles, a vertex outside another face's plane, or a
    vertex that is not a true corner (extra points on an edge or inside a face).
    """
    if (points.ndim != 2 or points.shape[1] != 3 or not 4 <= len(points) <= GUIDE_PRISM_MAX_VERTICES
            or not np.all(np.isfinite(points))):
        raise ValueError(f"{name}: a prism needs 4..{GUIDE_PRISM_MAX_VERTICES} finite vertices")
    if len(np.unique(points.round(12), axis=0)) != len(points):
        raise ValueError(f"{name}: duplicate prism vertices")
    if (faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 4 or faces.min() < 0 or faces.max() >= len(points)
            or np.any(faces[:, 0] == faces[:, 1]) or np.any(faces[:, 1] == faces[:, 2]) or np.any(faces[:, 0] == faces[:, 2])):
        raise ValueError(f"{name}: faces must be triangles over the prism vertices")
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    keys = directed[:, 0] * len(points) + directed[:, 1]
    reverse = directed[:, 1] * len(points) + directed[:, 0]
    if len(np.unique(keys)) != len(keys) or not np.all(np.isin(reverse, keys)):
        raise ValueError(f"{name}: prism surface is open or inconsistently wound")
    if len(points) - len(keys) // 2 + len(faces) != 2:
        raise ValueError(f"{name}: prism surface is not a single closed shell")
    if np.bincount(faces.ravel(), minlength=len(points)).min() < 3:
        raise ValueError(f"{name}: every prism vertex must belong to at least three faces")
    tri = points[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if lengths.min() <= 1e-14:
        raise ValueError(f"{name}: degenerate prism triangle")
    unit = normals / lengths[:, None]
    volume = float(np.einsum("ij,ij->i", tri[:, 0], normals).sum() / 6.0)
    if volume <= 0:
        raise ValueError(f"{name}: prism faces must wind outward")
    signed = points @ unit.T - np.einsum("ij,ij->i", unit, tri[:, 0])[None, :]
    if signed.max() > tol:
        raise ValueError(f"{name}: prism is not convex")
    for vertex in range(len(points)):
        if np.linalg.matrix_rank(unit[np.any(faces == vertex, axis=1)], tol=1e-6) < 3:
            raise ValueError(f"{name}: vertex {vertex} is not a corner of the prism")
    return volume


def _unique_directions(vectors: np.ndarray) -> np.ndarray:
    unit = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    first = unit[np.arange(len(unit)), np.argmax(np.abs(unit) > 1e-12, axis=1)]
    unit = unit * np.where(first < 0, -1.0, 1.0)[:, None]
    return np.unique(unit.round(9), axis=0)


def _prism_summary(points: np.ndarray, faces: np.ndarray) -> dict[str, np.ndarray]:
    tri = points[faces]
    normals = _unique_directions(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]))
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return {"points": points, "normals": normals,
            "edges": _unique_directions(points[edges[:, 1]] - points[edges[:, 0]]),
            "low": points.min(axis=0), "high": points.max(axis=0)}


def _interiors_disjoint(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray], tol: float = 1e-9) -> bool:
    """Separating-axis test for two convex polytopes; shared faces/edges count as disjoint."""
    if np.any(a["low"] > b["high"] + tol) or np.any(b["low"] > a["high"] + tol):
        return True
    crosses = np.cross(a["edges"][:, None, :], b["edges"][None, :, :]).reshape(-1, 3)
    crosses = crosses[np.linalg.norm(crosses, axis=1) > 1e-9]
    axes = np.vstack([a["normals"], b["normals"], crosses / np.linalg.norm(crosses, axis=1)[:, None]])
    pa, pb = a["points"] @ axes.T, b["points"] @ axes.T
    overlap = np.minimum(pa.max(axis=0) - pb.min(axis=0), pb.max(axis=0) - pa.min(axis=0))
    return bool(np.any(overlap <= tol))


def load_guide_prism_asset(path: str | Path | None = None) -> dict[str, Any]:
    """Load and verify the exact guide-prism asset with numpy only (no USD).

    Invariants: the two original guides in order, exactly 22 prisms per guide,
    each a closed, outward, strictly convex triangle mesh whose volume matches
    its record, pairwise disjoint interiors, and a union volume equal to the raw
    guide mesh volume within 1e-9 m^3. Returns numpy points/faces per prism plus
    the asset's sha256 for provenance.
    """
    asset_path = Path(path) if path is not None else GUIDE_PRISM_ASSET_PATH
    raw = asset_path.read_bytes()
    try:
        data = json.loads(raw)
    except ValueError as error:
        raise ValueError(f"Unreadable guide prism asset: {asset_path}") from error
    if not isinstance(data, Mapping) or data.get("schema") != GUIDE_PRISM_ASSET_SCHEMA:
        raise ValueError(f"Guide prism asset must use schema {GUIDE_PRISM_ASSET_SCHEMA}: {asset_path}")
    guides = data.get("guides")
    if (not isinstance(guides, list) or not all(isinstance(g, Mapping) for g in guides)
            or [g.get("mesh_relative_path") for g in guides] != list(GUIDE_MESH_RELATIVE_PATHS)):
        raise ValueError("Guide prism asset must describe the two original guide meshes in order")
    loaded = []
    for guide in guides:
        name, prisms = guide.get("name"), guide.get("prisms")
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("Guide prism asset entries need an identifier name")
        if (not isinstance(prisms, list) or len(prisms) != GUIDE_PRISMS_PER_GUIDE
                or guide.get("prism_count") != GUIDE_PRISMS_PER_GUIDE):
            raise ValueError(f"{name}: expected exactly {GUIDE_PRISMS_PER_GUIDE} prisms")
        raw_volume = guide.get("raw_volume_m3")
        if not isinstance(raw_volume, float) or not math.isfinite(raw_volume) or raw_volume <= 0:
            raise ValueError(f"{name}: raw_volume_m3 must be a positive finite float")
        digest = guide.get("source_mesh_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or set(digest) - set("0123456789abcdef"):
            raise ValueError(f"{name}: source_mesh_sha256 must be a sha256 hex digest")
        entries, summaries = [], []
        for index, prism in enumerate(prisms):
            expected = f"{name}_{index:02d}"
            if not isinstance(prism, Mapping) or prism.get("name") != expected:
                raise ValueError(f"{name}: prism {index} must be named {expected}")
            try:
                points = np.asarray(prism.get("points_jig_m"), dtype=np.float64)
                faces = np.asarray(prism.get("faces"), dtype=np.int64)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{expected}: malformed points or faces") from error
            volume = _convex_triangle_mesh_volume(points, faces, expected)
            recorded = prism.get("volume_m3")
            if not isinstance(recorded, float) or abs(volume - recorded) > 1e-12:
                raise ValueError(f"{expected}: recorded volume differs from the triangulated volume")
            entries.append({"name": expected, "points": points, "faces": faces, "volume_m3": volume})
            summaries.append(_prism_summary(points, faces))
        for i in range(len(summaries)):
            for j in range(i + 1, len(summaries)):
                if not _interiors_disjoint(summaries[i], summaries[j]):
                    raise ValueError(f"{name}: prisms {i} and {j} overlap")
        union = float(sum(entry["volume_m3"] for entry in entries))
        if abs(union - raw_volume) > GUIDE_PRISM_VOLUME_TOLERANCE_M3:
            raise ValueError(f"{name}: prism union volume {union} differs from the raw guide volume {raw_volume}")
        loaded.append({"name": name, "mesh_relative_path": guide["mesh_relative_path"],
                       "source_mesh_sha256": digest, "raw_volume_m3": raw_volume,
                       "union_volume_m3": union, "prisms": entries})
    return {"path": str(asset_path), "sha256": hashlib.sha256(raw).hexdigest(), "schema": data["schema"],
            "sources": data.get("sources", []), "guides": loaded}


def _validated_mesh(prim: Any, jig: Any, cache: Any, usd_geom: Any) -> dict[str, Any]:
    """Validate all input geometry before any session-layer mutation."""
    path = str(prim.GetPath()) if prim else "<missing>"
    if not prim or not prim.IsActive() or not prim.IsDefined() or not prim.IsA(usd_geom.Mesh):
        raise ValueError(f"Expected an active authored guide Mesh: {path}")
    if prim.IsInstanceProxy():
        raise ValueError(f"Guide mesh is an instance proxy; use the supplied non-instanceable scene override: {path}")
    mesh = usd_geom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 4 or
            not np.all(np.isfinite(points)) or counts.ndim != 1 or len(counts) == 0 or
            np.any(counts < 3) or indices.ndim != 1 or int(counts.sum()) != len(indices) or
            len(indices) == 0 or np.min(indices) < 0 or np.max(indices) >= len(points)):
        raise ValueError(f"Invalid guide mesh points or face topology: {path}")
    matrix = (np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(jig)).T) @
              np.asarray(cache.GetLocalToWorldTransform(prim)).T)
    if not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix[:3, :3])) < 1e-15:
        raise ValueError(f"Guide mesh has a singular or non-finite transform: {path}")
    local = points @ matrix[:3, :3].T + matrix[:3, 3]
    if np.any(np.ptp(local, axis=0) <= 0):
        raise ValueError(f"Guide mesh must have nonzero volume bounds: {path}")
    digest = hashlib.sha256()
    for value, dtype in ((points, "<f8"), (counts, "<i8"), (indices, "<i8")):
        array = np.asarray(value, dtype=dtype)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    schemas = prim.GetMetadata("apiSchemas")
    applied = list(schemas.GetAppliedItems()) if schemas else []
    enabled = prim.GetAttribute("physics:collisionEnabled").Get()
    return {
        "prim_path": path,
        "mesh_sha256": digest.hexdigest(),
        "point_count": len(points),
        "face_count": len(counts),
        "mesh_to_jig_matrix": matrix.tolist(),
        "bounds_in_jig_m": [local.min(axis=0).tolist(), local.max(axis=0).tolist()],
        "prior_collision_api": "PhysicsCollisionAPI" in applied,
        "prior_collision_enabled": enabled,
        "physics_material_targets": [str(p) for p in prim.GetRelationship("material:binding:physics").GetTargets()],
    }


def _strip_session_collision_opinions(session: Any, prim_path: Any, sdf: Any) -> None:
    """Drop collision opinions an earlier profile authored in the session on an original guide mesh."""
    spec = session.GetPrimAtPath(prim_path)
    if not spec:
        return
    for name in list(spec.properties.keys()):
        if name.startswith(_GUIDE_COLLISION_PROPERTY_PREFIXES):
            spec.RemoveProperty(spec.properties[name])
    if not spec.HasInfo("apiSchemas"):
        return
    current, kept, remaining = spec.GetInfo("apiSchemas"), sdf.TokenListOp(), False
    for field in ("explicitItems", "addedItems", "prependedItems", "appendedItems", "deletedItems", "orderedItems"):
        items = [token for token in getattr(current, field) if token not in _GUIDE_COLLISION_SCHEMAS]
        if items or (field == "explicitItems" and current.isExplicit):
            setattr(kept, field, items)
            remaining = True
    if remaining:
        spec.SetInfo("apiSchemas", kept)
    else:
        spec.ClearInfo("apiSchemas")


def _author_exact_prisms(stage: Any, jig_path: Any, meshes: list, records: list, asset: Mapping[str, Any],
                         settings: Mapping[str, Any]) -> tuple[Any, list[str]]:
    """Author one convex-hull prism collider per asset prism under <jig>/GuideCollisionPrisms."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    session = stage.GetSessionLayer()
    container_path = jig_path.AppendChild(GUIDE_PRISM_CONTAINER)
    container = UsdGeom.Xform.Define(stage, container_path)
    container.CreatePurposeAttr(UsdGeom.Tokens.guide)
    container.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    planned = {prism["name"] for guide in asset["guides"] for prism in guide["prisms"]}
    for child in container.GetPrim().GetChildren():
        # Session prisms left by an older asset revision would otherwise keep colliding.
        if child.GetName() not in planned and session.GetPrimAtPath(child.GetPath()):
            stage.RemovePrim(child.GetPath())
    prim_paths = []
    for guide, mesh_prim, record in zip(asset["guides"], meshes, records):
        _strip_session_collision_opinions(session, mesh_prim.GetPath(), Sdf)
        record["collision_disabled_in_session"] = bool(mesh_prim.HasAPI(UsdPhysics.CollisionAPI))
        if record["collision_disabled_in_session"]:
            # A weaker layer made the original guide a collider: switch it off so only the prisms collide.
            mesh_prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool, custom=False).Set(False)
        source_rel = mesh_prim.GetRelationship("material:binding:physics")
        strength = source_rel.GetCustomDataByKey("bindMaterialAs") if source_rel else None
        for prism in guide["prisms"]:
            prim_path = container_path.AppendChild(prism["name"])
            mesh = UsdGeom.Mesh.Define(stage, prim_path)
            prim = mesh.GetPrim()
            points = prism["points"]
            mesh.CreatePointsAttr([Gf.Vec3f(*map(float, point)) for point in points])
            mesh.CreateFaceVertexCountsAttr([3] * len(prism["faces"]))
            mesh.CreateFaceVertexIndicesAttr([int(index) for index in prism["faces"].ravel()])
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateExtentAttr([Gf.Vec3f(*map(float, points.min(axis=0))),
                                   Gf.Vec3f(*map(float, points.max(axis=0)))])
            mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
            mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(settings["physics:approximation"])
            prim.AddAppliedSchema("PhysxCollisionAPI")
            for name, value in settings.items():
                if name.startswith("physics:") or name == "profile":
                    continue
                type_name = (Sdf.ValueTypeNames.Bool if isinstance(value, bool) else
                             Sdf.ValueTypeNames.Int if isinstance(value, int) else Sdf.ValueTypeNames.Float)
                prim.CreateAttribute(name, type_name, custom=True).Set(value)
            UsdShade.MaterialBindingAPI.Apply(prim)
            rel = prim.CreateRelationship("material:binding:physics", bool(source_rel and source_rel.IsCustom()))
            rel.SetTargets(record["physics_material_targets"])
            if strength:
                rel.SetCustomDataByKey("bindMaterialAs", strength)
            prim_paths.append(str(prim_path))
    return container_path, prim_paths


def author_jig_slot_collisions(stage: Any, config: Mapping[str, Any], *,
                               require_source_mesh_match: bool = True) -> dict[str, Any]:
    """Enable collision on the two original JIG guide meshes, in session only.

    Call before starting or resetting physics so PhysX cooks the new colliders.
    Requires ``config['objects']['jig']`` to name the supplied asset's rigid body.
    Missing geometry or invalid topology fails before any edit. The previous USD
    edit target is restored even on failure. No object attachment is created.

    ``authored_v1`` and ``high_resolution_v2`` apply a convex decomposition to the
    guide meshes themselves. ``exact_prisms_v3`` instead defines transform-free
    convex-hull Mesh colliders under ``<jig>/GuideCollisionPrisms`` from the
    verified package asset, binds them to the guide's physics material and leaves
    the original guide meshes without a collision API. By default that profile
    refuses a stage whose guide meshes differ from the ones the prisms were built
    from (``require_source_mesh_match``), since the prisms would not fit them.

    USD imports are lazy. PhysX schema tokens and attributes use the same authored
    representation as Scene.usda, allowing validation with core USD without
    starting Isaac Sim or importing its extension registry.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    if not isinstance(config, Mapping) or not isinstance(config.get("objects"), Mapping):
        raise ValueError("config.objects must contain the JIG prim path")
    settings = collision_settings(config)
    exact = settings.get("profile") == EXACT_PRISM_PROFILE
    jig_path = config["objects"].get("jig")
    if not isinstance(jig_path, str) or not Sdf.Path.IsValidPathString(jig_path):
        raise ValueError("config.objects.jig must be an absolute USD prim path")
    path = Sdf.Path(jig_path)
    if not path.IsAbsolutePath() or not path.IsPrimPath() or path == Sdf.Path.absoluteRootPath:
        raise ValueError("config.objects.jig must be an absolute USD prim path")
    jig = stage.GetPrimAtPath(path)
    if not jig or not jig.IsActive() or not jig.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError(f"JIG must be an existing active rigid body: {jig_path}")
    if not math.isclose(float(UsdGeom.GetStageMetersPerUnit(stage)), 1.0, abs_tol=1e-12):
        raise ValueError("JIG collision settings require a stage expressed in meters")
    session = stage.GetSessionLayer()
    if not session or not session.permissionToEdit:
        raise ValueError("A writable USD session layer is required")
    asset = load_guide_prism_asset() if exact else None
    cache = UsdGeom.XformCache()
    meshes, records = [], []
    for relative in GUIDE_MESH_RELATIVE_PATHS:
        full_path = path.AppendPath(relative)
        prim = stage.GetPrimAtPath(full_path)
        if not prim:
            raise ValueError(f"Missing original JIG guide mesh: {full_path}")
        records.append(_validated_mesh(prim, jig, cache, UsdGeom))
        meshes.append(prim)
    if exact:
        mismatched = [record["prim_path"] for guide, record in zip(asset["guides"], records)
                      if record["mesh_sha256"] != guide["source_mesh_sha256"]]
        if mismatched and require_source_mesh_match:
            raise ValueError("exact_prisms_v3 prisms were built for different guide mesh geometry: "
                             + ", ".join(mismatched))

    before = session.ExportToString()
    try:
        with Usd.EditContext(stage, session):
            if exact:
                container_path, prim_paths = _author_exact_prisms(stage, path, meshes, records, asset, settings)
            else:
                for prim in meshes:
                    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexDecomposition")
                    prim.AddAppliedSchema("PhysxCollisionAPI")
                    prim.AddAppliedSchema("PhysxConvexDecompositionCollisionAPI")
                    for name, value in settings.items():
                        if name.startswith("physics:"):
                            continue
                        type_name = (Sdf.ValueTypeNames.Bool if isinstance(value, bool) else
                                     Sdf.ValueTypeNames.Int if isinstance(value, int) else Sdf.ValueTypeNames.Float)
                        prim.CreateAttribute(name, type_name, custom=True).Set(value)
    except Exception:
        # Restore other existing session opinions too; never leave one guide
        # enabled while the second failed to author.
        session.ImportFromString(before)
        raise
    report = {
        "schema_version": "igris_c_jig_slot_collision_v1",
        "jig_prim_path": jig_path,
        "edit_layer": session.identifier,
        "source": "exact_convex_prisms_of_raw_guide_extrusion" if exact else "existing_authored_jig_guide_meshes",
        "settings": settings,
        "meshes": records,
        "source_assets_modified": False,
        "geometry_modified": False,
        "attachments_created": False,
        "physics_cooking_verified": False,
        "physical_support_verified": False,
        "requires_physics_restart_or_reset": True,
    }
    if exact:
        report["collision_profile"] = EXACT_PRISM_PROFILE
        report["original_guide_meshes_collide"] = False
        report["guide_collision_prisms"] = {
            "container_prim_path": str(container_path),
            "count": len(prim_paths),
            "prim_paths": prim_paths,
            "asset_path": asset["path"],
            "asset_sha256": asset["sha256"],
            "asset_schema": asset["schema"],
            "approximation": settings["physics:approximation"],
            "frame": "jig_prim_local",
            "per_guide": [{
                "name": guide["name"],
                "mesh_prim_path": record["prim_path"],
                "prism_count": len(guide["prisms"]),
                "union_volume_m3": guide["union_volume_m3"],
                "raw_volume_m3": guide["raw_volume_m3"],
                "source_mesh_sha256": guide["source_mesh_sha256"],
                "source_mesh_sha256_matches": record["mesh_sha256"] == guide["source_mesh_sha256"],
                "physics_material_targets": record["physics_material_targets"],
                "collision_disabled_in_session": record["collision_disabled_in_session"],
            } for guide, record in zip(asset["guides"], records)],
        }
    return report
