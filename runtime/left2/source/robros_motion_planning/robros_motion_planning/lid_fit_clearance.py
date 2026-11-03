"""Unvalidated simulation clearance for the authored zero-gap removable lid.

Opt-in calibration profile; the runtime default leaves it disabled. It changes
only the two named invisible collision meshes in the USD session layer, before
physics initialization. It is NOT a measured manufacturing tolerance or a
physical task success. The source USD, visible geometry, handle, upper flange,
Z coordinates, mass, materials, friction, contact offsets and controllers are
not authored by this helper.
"""
from __future__ import annotations
import hashlib
import numpy as np

CLEARANCE_PER_SIDE_M = 0.0005
EXPECTED = {
    'Slab_00': {'sha256': '45fa74db2a4014dfbce2911f9af1d9cfa825beb40a99d96ac1a69876fcd19b9c',
                'point_count': 28, 'bounds': [[-.24000000953674316,-.23995928466320038,-.010856463573873043], [.24000000953674316,.24004073441028595,-.00985646340996027]]},
    'Slab_01': {'sha256': '8bca8be5d84b5eac7e484cb6298c406282d0908b12085b872dbd14e72693e3e7',
                'point_count': 32, 'bounds': [[-.24000000953674316,-.23995928466320038,-.00985646340996027], [.24000000953674316,.24004073441028595,-.0018564631463959813]]},
}


def prepare_lid_fit_clearance(env, env_ids=None):
    """IsaacLab prestartup event, before the first PhysX initialization."""
    del env_ids
    import omni.usd
    env.nepes_lid_fit_clearance_provenance = author_lid_fit_clearance(
        omni.usd.get_context().get_stage())


def geometry_sha256(points, counts, indices):
    digest = hashlib.sha256(b'igris_c_lid_collision_geometry_v1\0')
    for value, dtype in ((points, '<f4'), (counts, '<i4'), (indices, '<i4')):
        digest.update(np.asarray(value, dtype=dtype).tobytes(order='C'))
    return digest.hexdigest()


def inset_points(points):
    """Move each lateral side inward by 0.5 mm; preserve chamfer and all Z."""
    original = np.asarray(points, dtype=np.float32)
    if original.ndim != 2 or original.shape[1] != 3 or not np.isfinite(original).all():
        raise ValueError('Expected finite Nx3 collision points')
    updated = original.copy()
    center = (original[:, :2].min(axis=0).astype(float) + original[:, :2].max(axis=0)) / 2.
    delta = original[:, :2].astype(float) - center
    if np.any(np.abs(delta) < CLEARANCE_PER_SIDE_M):
        raise ValueError('Unexpected interior vertices: fixed side-inset contract does not apply')
    updated[:, :2] = original[:, :2] - np.sign(delta) * CLEARANCE_PER_SIDE_M
    return updated


def author_lid_fit_clearance(stage, *, cover_path='/World/Scene/CleanerCover'):
    """Validate exact authored geometry, then modify only session-layer points.

    Must be called before the first physics initialization/reset. Repeated
    application fails source-hash validation, preventing accidental double
    shrink. The caller records the returned provenance; no success is implied.
    """
    from pxr import Usd, UsdGeom, UsdPhysics, Vt

    if cover_path not in ('/World/Scene/CleanerCover', '/World/CleanerCover'):
        raise ValueError('Unexpected CleanerCover actor path')
    body = stage.GetPrimAtPath(cover_path)
    if not body.IsValid() or not body.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError('Expected dynamic CleanerCover rigid body')
    api = UsdPhysics.RigidBodyAPI(body)
    if api.GetRigidBodyEnabledAttr().Get() is False or api.GetKinematicEnabledAttr().Get() is True:
        raise ValueError('Clearance prototype requires a free dynamic lid')
    mass = UsdPhysics.MassAPI(body).GetMassAttr().Get()
    if mass is None or not np.isclose(float(mass), 1., rtol=0, atol=1e-6):
        raise ValueError('Expected authored CleanerCover mass of 1 kg; no mass edits are permitted')
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            targets = list(joint.GetBody0Rel().GetTargets()) + list(joint.GetBody1Rel().GetTargets())
            if any(str(path) == cover_path for path in targets):
                raise ValueError('Lid is connected to a physics joint')
    cache = UsdGeom.XformCache()
    body_matrix = np.asarray(cache.GetLocalToWorldTransform(body), dtype=float).T
    prepared = []
    for name, expected in EXPECTED.items():
        path = cover_path + '/CollisionFit/' + name
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError('Expected collision Mesh: ' + path)
        if UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() != 'convexHull':
            raise ValueError('Unexpected collision approximation: ' + path)
        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputeVisibility() != UsdGeom.Tokens.invisible or imageable.ComputePurpose() != UsdGeom.Tokens.guide:
            raise ValueError('Only the authored invisible collision guide is eligible: ' + path)
        local_transform = np.linalg.inv(body_matrix) @ np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float).T
        if not np.allclose(local_transform, np.eye(4), atol=1e-10, rtol=0):
            raise ValueError('Unexpected collision mesh transform: ' + path)
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        bounds = np.array([points.min(0), points.max(0)])
        if (len(points) != expected['point_count'] or len(counts) != 12
                or not np.allclose(bounds, expected['bounds'], atol=1e-9, rtol=0)
                or geometry_sha256(points, counts, indices) != expected['sha256']):
            raise ValueError('Original lid collision geometry hash/bounds mismatch: ' + path)
        if mesh.GetPointsAttr().GetNumTimeSamples() or mesh.GetExtentAttr().HasAuthoredValueOpinion():
            raise ValueError('Animated or pre-authored extent geometry is not supported: ' + path)
        updated = inset_points(points)
        prepared.append((path, mesh, points, updated, counts, indices))
    session = stage.GetSessionLayer()
    if session is None:
        raise ValueError('A USD session layer is required')
    previous_session = session.ExportToString()
    try:
        with Usd.EditContext(stage, session):
            for _, mesh, _, updated, _, _ in prepared:
                if not mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(updated)):
                    raise RuntimeError('Unable to author collision points')
    except Exception:
        session.ImportFromString(previous_session)
        raise
    return {
        'schema_version': 'igris_c_lid_fit_clearance_prototype_v1',
        'assumption': 'Unvalidated simulation fit clearance; actual manufacturing clearance is unknown.',
        'cover_path': cover_path, 'edit_layer': 'session', 'clearance_per_side_m': CLEARANCE_PER_SIDE_M,
        'physical_success_verified': False, 'manufacturing_tolerance_verified': False,
        'mass_friction_stiffness_visuals_source_usd_unchanged': True,
        'meshes': [{'path': path, 'source_geometry_sha256': geometry_sha256(old, counts, indices),
                    'modified_geometry_sha256': geometry_sha256(new, counts, indices),
                    'before_bounds_local_m': [old.min(0).tolist(), old.max(0).tolist()],
                    'after_bounds_local_m': [new.min(0).tolist(), new.max(0).tolist()],
                    'maximum_lateral_coordinate_change_m': float(np.max(np.abs(new[:, :2] - old[:, :2]))),
                    'z_unchanged': bool(np.array_equal(old[:, 2], new[:, 2]))}
                   for path, _, old, new, counts, indices in prepared],
    }
