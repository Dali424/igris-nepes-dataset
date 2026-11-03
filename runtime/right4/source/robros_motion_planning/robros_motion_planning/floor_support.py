"""Restore original floor support without overlapping the workcell support Cube.

The original downward-facing floor is replaced for collision only by its
portion outside the existing support Cube. The Cube preserves interior holes.
No new support is added beyond the pre-existing union of these two shapes.
Original vertices, face indices, source USD and visual geometry remain unchanged.
Runtime uses NumPy and lazy USD imports; polygon audit hashes avoid adding
a Shapely dependency to the simulator environment.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
SOURCE_SHA256 = 'f8304c3316f377c33506c13e1f5560d703de18d05833dd607730121d57a41c5e'
POINT_SHA256 = '6b7e4989afead184c88326a886aa7099a170fec9b7177858f4938b40be642831'
COUNT_SHA256 = '37f24a4b38881c685c4ecb93dbf90f352eaccedf9893b0da31c26b3892e91d35'
INDEX_SHA256 = '47084e463790f3d02614620d8aed28ace0109d0a16a5c770e75e6f8ca671858d'
SCALE = np.array([2.2, 1.75, 0.05], dtype=np.float32)
TRANSLATE = np.array([0.0, -0.175, -0.025])
MATERIAL = {'physics:dynamicFriction': 0.44999998807907104, 'physics:staticFriction': 0.6000000238418579, 'physics:restitution': 0.0, 'physxMaterial:frictionCombineMode': 'average', 'physxMaterial:restitutionCombineMode': 'average'}

def _sha(value, dtype):
    return hashlib.sha256(np.asarray(value, dtype=dtype).tobytes()).hexdigest()

def _locked_source_coverage_audit():
    """Offline polygon-union metrics bound to the validated source/Box hashes."""
    return {'algorithm': 'Exact locked-source triangle XY union, audited offline; verified by hashes at runtime.', 'floor_area_m2': 9.672582395545554, 'box_area_m2': 3.8500000834465027, 'missing_area_m2': 0.017775530417960157, 'overlap_area_m2': 3.8322245530285426, 'whole_box_coverage_verified': False}

def inspect_floor_support_source(stage, *, scene_path='/World/Scene'):
    from pxr import Usd, UsdGeom, UsdPhysics
    if scene_path not in ('/World', '/World/Scene'):
        raise ValueError('Unexpected exact scene root')
    path = scene_path + '/Colliders/Floor_00'
    boxpath = scene_path + '/WorkcellFloorCollider'
    floor = stage.GetPrimAtPath(path)
    guide = stage.GetPrimAtPath(boxpath)
    if not floor or not floor.IsA(UsdGeom.Mesh) or (not floor.HasAPI(UsdPhysics.CollisionAPI)):
        raise ValueError('Expected existing Floor_00 collision Mesh')
    if not guide or not guide.IsA(UsdGeom.Cube) or (not guide.HasAPI(UsdPhysics.CollisionAPI)):
        raise ValueError('Expected existing WorkcellFloorCollider Cube')
    sources = [Path(s.layer.realPath).resolve() for s in floor.GetPrimStack() if str(s.path) == '/World/Colliders/Floor_00' and s.layer.realPath]
    sources = [p for p in sources if hashlib.sha256(p.read_bytes()).hexdigest() == SOURCE_SHA256]
    if not sources:
        raise ValueError('Original floor source SHA256/prim mismatch')
    if UsdGeom.GetStageUpAxis(stage) != 'Z' or UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        raise ValueError('Expected meter-scale Z-up stage')
    for p in (floor, guide):
        if not UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get():
            raise ValueError('Expected original enabled collision/repeat guard')
        a = p
        while a and (not a.IsPseudoRoot()):
            if a.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError('Expected static floor')
            a = a.GetParent()
        im = UsdGeom.Imageable(p)
        if im.ComputeVisibility() != 'invisible' or im.ComputePurpose() != 'guide':
            raise ValueError('Only original invisible guide collision geometry may change')
        materialpath = scene_path + '/PhysicsMaterials/Environment'
        if [str(t) for t in p.GetRelationship('material:binding:physics').GetTargets()] != [materialpath]:
            raise ValueError('Original physics material binding changed')
    material = stage.GetPrimAtPath(materialpath)
    if any((material.GetAttribute(k).Get() != v for (k, v) in MATERIAL.items())):
        raise ValueError('Original floor material values changed')
    mesh = UsdGeom.Mesh(floor)
    if mesh.GetOrientationAttr().Get() != 'rightHanded':
        raise ValueError('Expected original rightHanded floor/repeat guard')
    if mesh.GetSubdivisionSchemeAttr().Get() != 'none' or UsdPhysics.MeshCollisionAPI(floor).GetApproximationAttr().Get() != 'none':
        raise ValueError('Expected original triangle-mesh collision')
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype='<f4')
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype='<i4')
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype='<i4')
    if _sha(points, '<f4') != POINT_SHA256 or _sha(counts, '<i4') != COUNT_SHA256 or _sha(indices, '<i4') != INDEX_SHA256:
        raise ValueError('Original floor points/topology SHA256 mismatch')
    if any((a.GetNumTimeSamples() for a in (mesh.GetPointsAttr(), mesh.GetFaceVertexCountsAttr(), mesh.GetFaceVertexIndicesAttr(), mesh.GetOrientationAttr()))):
        raise ValueError('Animated floor is not supported')
    cache = UsdGeom.XformCache()
    T = np.asarray(cache.GetLocalToWorldTransform(floor), dtype=float).T
    if not np.array_equal(T, np.eye(4)):
        raise ValueError('Original floor world transform must be identity')
    triangles = points.astype(float)[indices.reshape(-1, 3)]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    if not (np.all(normals[:, 2] < 0) and np.all(normals[:, :2] == 0) and np.all(points[:, 2] == 0)):
        raise ValueError('Expected horizontal floor at Z=0 with every triangle facing down')
    cube = UsdGeom.Cube(guide)
    bt = np.asarray(cache.GetLocalToWorldTransform(guide), dtype=float).T
    expected = np.diag([*SCALE.astype(float), 1.0])
    expected[:3, 3] = TRANSLATE
    if cube.GetSizeAttr().Get() != 1.0 or not np.array_equal(bt, expected):
        raise ValueError('Original workcell support box geometry changed')
    coverage = _locked_source_coverage_audit()
    return {'schema_version': 'igris_c_floor_support_source_v1', 'floor_path': path, 'workcell_box_path': boxpath, 'source_usd': str(sources[0]), 'source_sha256': SOURCE_SHA256, 'point_sha256': POINT_SHA256, 'counts_sha256': COUNT_SHA256, 'indices_sha256': INDEX_SHA256, 'source_orientation': 'rightHanded', 'all_world_triangles_down': True, 'floor_z_m': 0.0, 'world_transform_determinant': 1.0, 'physics_material_path': materialpath, 'physics_material_values': MATERIAL.copy(), 'coverage': coverage, 'support_area_or_height_expansion': False, 'vertices_indices_visuals_source_unchanged': True, 'assumption': 'Original CAD has downward-facing invisible floor triangles. Geometry and existing support Cube are locked before deriving a non-overlapping upward-facing replacement outside that Cube.', 'physical_success_verified': False, 'duplicate_upward_support_permitted': False}
# These metrics were independently recomputed from the serialized USD overlay.
# Exact point/index hashes lock the audit to the generated runtime geometry.
DERIVED_NAME = 'FloorOutsideWorkcellSupport'
EXPECTED_PARTITION = {'derived_point_sha256': '597f769a3c01944e2784a863177565963cc0476d776cf7b22ae00d7bf0de80ed',
 'derived_indices_sha256': '8df103d6142dc72cd8c91d3d97c2033ce401950da28f1cf11deae0d7055e3471',
 'derived_box_overlap_area_m2': 0.0,
 'support_union_symmetric_difference_m2': 4.5553499632624614e-07,
 'support_union_tolerance_m2': 1e-06,
 'source_union_area_m2': 9.672582395545554,
 'source_triangle_sum_minus_union_m2': 1.360489534363296e-07,
 'source_overlap_tolerance_m2': 1e-06,
 'derived_union_area_m2': 5.8403578621279735,
 'derived_triangle_sum_minus_union_m2': 1.5808103626113734e-07,
 'before_support_union_area_m2': 9.690357925963522,
 'after_support_union_area_m2': 9.690357945574474,
 'method': 'Independent-of-clip Shapely/GEOS polygon unions of actual source/derived '
           'float32-authored coordinates; compare symmetric difference and overlap.',
 'physical_support_verified': False}

def _clip_half_plane(polygon, axis, bound, keep_greater):
    if not len(polygon):
        return np.empty((0, 3), dtype=float)
    out = []
    previous = np.asarray(polygon[-1], dtype=float)
    prev_inside = previous[axis] >= bound if keep_greater else previous[axis] <= bound
    for current in polygon:
        current = np.asarray(current, dtype=float)
        inside = current[axis] >= bound if keep_greater else current[axis] <= bound
        if inside != prev_inside:
            fraction = (bound - previous[axis]) / (current[axis] - previous[axis])
            point = previous + fraction * (current - previous)
            point[axis] = bound
            out.append(point)
        if inside:
            out.append(current.copy())
        previous = current
        prev_inside = inside
    if not out:
        return np.empty((0, 3), dtype=float)
    clean = [out[0]]
    for point in out[1:]:
        if not np.array_equal(point, clean[-1]):
            clean.append(point)
    if len(clean) > 1 and np.array_equal(clean[0], clean[-1]):
        clean.pop()
    return np.asarray(clean, dtype=float)

def partition_triangles_outside_box(points, indices, box_bounds):
    """Clip each source triangle into four mutually exclusive XY regions.

    Return float32 local points, triangle indices, origin and quantitative
    conversion metadata. Origin's Y equals the Box center, making both Y
    clipping boundaries +/-0.875 exactly representable in USD point floats.
    """
    points = np.asarray(points, dtype=float)
    raw_indices = np.asarray(indices)
    if raw_indices.dtype.kind not in 'iu':
        raise ValueError('Triangle indices must have integer dtype')
    indices = raw_indices.astype(np.int64)
    bounds = np.asarray(box_bounds, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or (not np.isfinite(points).all()):
        raise ValueError('Finite Nx3 points required')
    if indices.ndim != 1 or len(indices) % 3 or np.any(indices < 0) or np.any(indices >= len(points)):
        raise ValueError('Triangle index array required')
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1, :2] <= bounds[0, :2]):
        raise ValueError('Valid Box world bounds required')
    origin = np.array([0.0, (bounds[0, 1] + bounds[1, 1]) / 2.0, 0.0])
    (lo, hi) = (bounds[0] - origin, bounds[1] - origin)
    if not np.array_equal(np.asarray([lo[:2], hi[:2]], dtype=np.float32).astype(float), [lo[:2], hi[:2]]):
        raise ValueError('Box boundaries must be exact float32 values in the selected local frame')
    regions = (((0, lo[0], False),), ((0, hi[0], True),), ((0, lo[0], True), (0, hi[0], False), (1, lo[1], False)), ((0, lo[0], True), (0, hi[0], False), (1, hi[1], True)))
    out_points = []
    out_indices = []
    lookup = {}
    source_area = 0.0
    clip_area = 0.0
    max_quantization = 0.0
    discarded_zero_area = 0
    quantized_degenerate_area = 0.0
    quantized_degenerate_count = 0
    for face in indices.reshape(-1, 3):
        source = points[face] - origin
        source_area += float(np.linalg.norm(np.cross(source[1] - source[0], source[2] - source[0])) / 2)
        for region in regions:
            polygon = source.copy()
            for (axis, bound, greater) in region:
                polygon = _clip_half_plane(polygon, axis, bound, greater)
            if len(polygon) < 3:
                continue
            for j in range(1, len(polygon) - 1):
                tri = polygon[[0, j, j + 1]].copy()
                normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                if abs(normal[2]) <= 1e-18:
                    discarded_zero_area += 1
                    continue
                if normal[2] < 0:
                    tri = tri[[0, 2, 1]]
                quantized = tri.astype(np.float32)
                qnormal = np.cross(quantized[1].astype(float) - quantized[0], quantized[2].astype(float) - quantized[0])
                if qnormal[2] <= 0:
                    lost_area = float(abs(normal[2]) / 2)
                    quantized_degenerate_area += lost_area
                    quantized_degenerate_count += 1
                    if lost_area > 1e-08 or quantized_degenerate_area > 1e-07:
                        raise ValueError(f'Float32 degeneration exceeds explicit area budget: {lost_area}, total={quantized_degenerate_area}, tri={tri.tolist()}, quantized={quantized.tolist()}')
                    continue
                max_quantization = max(max_quantization, float(np.max(np.abs(quantized.astype(float) - tri))))
                clip_area += float(np.linalg.norm(qnormal) / 2)
                for vertex in quantized:
                    key = tuple(map(float, vertex))
                    if key not in lookup:
                        lookup[key] = len(out_points)
                        out_points.append(vertex)
                    out_indices.append(lookup[key])
    vertices = np.asarray(out_points, dtype=np.float32)
    faces = np.asarray(out_indices, dtype=np.int32)
    return (vertices, faces, origin, {'source_triangle_area_sum_m2': source_area, 'derived_triangle_area_sum_m2': clip_area, 'maximum_coordinate_quantization_m': max_quantization, 'discarded_zero_area_triangles': discarded_zero_area, 'float32_degenerate_triangle_count': quantized_degenerate_count, 'float32_degenerate_triangle_area_m2': quantized_degenerate_area, 'float32_degenerate_area_budget_m2': 1e-07, 'source_triangle_count': len(indices) // 3, 'derived_triangle_count': len(faces) // 3, 'source_duplicate_area_tolerance_m2': 1e-06, 'derived_box_overlap_area_tolerance_m2': 1e-10})

def build_floor_support_partition(stage, *, scene_path='/World/Scene'):
    from pxr import Usd, UsdGeom
    report = inspect_floor_support_source(stage, scene_path=scene_path)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(report['floor_path']))
    guide = stage.GetPrimAtPath(report['workcell_box_path'])
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy', 'guide']).ComputeWorldBound(guide).ComputeAlignedRange()
    (points, indices, origin, stats) = partition_triangles_outside_box(mesh.GetPointsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get(), np.array([bounds.GetMin(), bounds.GetMax()]))
    duplicate_area = stats['source_triangle_area_sum_m2'] - report['coverage']['floor_area_m2']
    if duplicate_area < -1e-10 or duplicate_area > 1e-06:
        raise ValueError('Original triangle overlap exceeds native-precision audit tolerance')
    normals = np.cross(points[indices.reshape(-1, 3)][:, 1].astype(float) - points[indices.reshape(-1, 3)][:, 0], points[indices.reshape(-1, 3)][:, 2].astype(float) - points[indices.reshape(-1, 3)][:, 0])
    if not np.all(normals[:, 2] > 0) or not np.all(normals[:, :2] == 0) or (not np.all(points[:, 2] == 0)):
        raise ValueError('Derived floor must be horizontal at original Z=0 with +Z triangles')
    report.update(profile='original_floor_outside_workcell_box', schema_version='igris_c_floor_support_partition_v1', proposed_orientation='original orientation remains unchanged; derived mesh is rightHanded with +Z winding', derived_path=scene_path + '/Colliders/' + DERIVED_NAME, derived_point_sha256=_sha(points, '<f4'), derived_indices_sha256=_sha(indices, '<i4'), derived_origin_w=origin.tolist(), partition_metrics=stats, original_source_triangle_overlap_m2=duplicate_area, assumption='Restore collision support only on original floor mesh outside the existing support Cube, using native-float clipped triangles. Existing Cube remains unchanged and preserves all its previous support. Source visual geometry is unmodified. Actual PhysX support remains unverified.', original_floor_collision_disabled=True, workcell_box_unchanged=True)
    return (points, indices, origin, report)

def author_floor_support_partition(stage, *, scene_path='/World/Scene'):
    """Author one collision-only derived mesh and disable original floor.

    Must run before physics initialization. No broad box expansion or duplicate
    active coplanar region is permitted. Atomic session rollback on failure.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt, Gf
    (points, indices, origin, report) = build_floor_support_partition(stage, scene_path=scene_path)
    if EXPECTED_PARTITION is None:
        raise ValueError('Derived mesh independent audit has not been locked')
    for key in ('derived_point_sha256', 'derived_indices_sha256'):
        if report[key] != EXPECTED_PARTITION[key]:
            raise ValueError('Derived partition hash differs from audited geometry')
    if EXPECTED_PARTITION['derived_box_overlap_area_m2'] > 1e-10:
        raise ValueError('Audited derived mesh overlaps workcell Box')
    if EXPECTED_PARTITION['support_union_symmetric_difference_m2'] > EXPECTED_PARTITION['support_union_tolerance_m2']:
        raise ValueError('Audited support union is not preserved within native-float tolerance')
    if stage.GetPrimAtPath(report['derived_path']):
        raise ValueError('Derived support already exists/repeat guard')
    session = stage.GetSessionLayer()
    if session is None or not session.permissionToEdit:
        raise ValueError('Editable session layer required')
    before = session.ExportToString()
    original = stage.GetPrimAtPath(report['floor_path'])
    source_hashes = (_sha(UsdGeom.Mesh(original).GetPointsAttr().Get(), '<f4'), _sha(UsdGeom.Mesh(original).GetFaceVertexIndicesAttr().Get(), '<i4'))
    try:
        with Usd.EditContext(stage, session):
            derived = UsdGeom.Mesh.Define(stage, report['derived_path'])
            p = derived.GetPrim()
            derived.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
            derived.CreateFaceVertexCountsAttr([3] * (len(indices) // 3))
            derived.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices))
            derived.CreateSubdivisionSchemeAttr('none')
            derived.CreateOrientationAttr('rightHanded')
            derived.CreateVisibilityAttr('invisible')
            derived.CreatePurposeAttr('guide')
            derived.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, points.min(0))), Gf.Vec3f(*map(float, points.max(0)))]))
            UsdGeom.Xformable(p).AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, origin)))
            UsdPhysics.CollisionAPI.Apply(p).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr('none')
            UsdShade.MaterialBindingAPI.Apply(p)
            rel = original.GetRelationship('material:binding:physics')
            dest = p.CreateRelationship('material:binding:physics')
            dest.SetTargets(rel.GetTargets())
            for (key, value) in rel.GetAllMetadata().items():
                if key not in ('custom', 'variability'):
                    dest.SetMetadata(key, value)
            UsdPhysics.CollisionAPI(original).GetCollisionEnabledAttr().Set(False)
            if UsdPhysics.CollisionAPI(stage.GetPrimAtPath(report['workcell_box_path'])).GetCollisionEnabledAttr().Get() is not True:
                raise RuntimeError('Existing Box collision must remain enabled')
        if source_hashes != (_sha(UsdGeom.Mesh(original).GetPointsAttr().Get(), '<f4'), _sha(UsdGeom.Mesh(original).GetFaceVertexIndicesAttr().Get(), '<i4')):
            raise RuntimeError('Original floor geometry changed')
    except Exception:
        session.ImportFromString(before)
        raise
    report.update(edit_layer='session', independent_geometry_audit=EXPECTED_PARTITION.copy(), modified_existing_properties=[report['floor_path'] + '.physics:collisionEnabled'], created_prims=[report['derived_path']], physical_success_verified=False)
    return report


def prepare_floor_support(env, env_ids=None):
    """IsaacLab prestartup hook; author before first PhysX initialization.

    One environment and disabled physics replication are required because the
    derived collision mesh is scene-level geometry. The caller must register
    this event before gym.make(), which initializes the physics scene.
    """
    del env_ids
    scene_cfg = env.cfg.scene
    if scene_cfg.num_envs != 1 or scene_cfg.replicate_physics is not False:
        raise ValueError("Floor support repair requires num_envs=1 and replicate_physics=False")
    import omni.usd
    report = author_floor_support_partition(omni.usd.get_context().get_stage())
    env.nepes_floor_support_provenance = report

