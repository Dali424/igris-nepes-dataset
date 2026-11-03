#!/usr/bin/env python3
"""make_pads.py -- parametric model of the beige rubber pads of the NEW IGRIS-C 1-DOF gripper jaws (both hands).

Model (decided from the original meshes + photo img_01_line1497.jpg, see pads.json 'evidence'):
  * The old jaw meshes are assemblies.  Each finger carries a separate INNER INSERT over the same footprint
    (fixed: tip insert, z -156.5..-86.5, base plane s*y = -2 mm, top = V-channel; moving: inner insert with V-ridge +
    a 6.7 mm arc recess near the tip, bounded by the insert/body interface plane).  In the photo these volumes are
    beige: the new pads REPLACE the old inserts.  The two pads meet along a flat parting plane (fixed-frame
    s*y = fixed_face_mm, photo 4.0 mm) at joint angle q_touch (photo 0.02 rad).
  * A beige strip (photo: notched) replaces the old 3 mm back plate on the fixed jaw's OUTER (back) face, thicker
    (photo 6.6 mm) with transverse grooves.
  * Every output piece is one CONVEX polytope (half-space construction -> convex hull), so PhysX convexHull is exact.

Frames / units: every STL is in its own URDF link frame, meters
  fixed pieces  -> {l,r}_1_dof_gripper_fixed  (canonical Left_Hand / Right_Hand)
  moving pieces -> {l,r}_1_dof_gripper_moving (joint frame; posed by origin (0, s*0.02625, -0.04225) + Rx(q))
  s = +1 left, -1 right (left moving jaw at +y, right at -y; right joint q is negative when opening).

Usage:
  python make_pads.py [--repo DIR] [--out DIR] [--set name=value ...] [--no-strip] [--no-grooves] [--white-bodies]
Example (replace the photo estimate by a caliper value):  --set fixed_face_mm=3.5 --set q_touch_rad=0.0
--repo defaults to the nearest parent of this file holding igrisc_model_cam_260910/; --out defaults to
<repo>/outputs/gripper_pads/make_pads, outside the Python package (the pinned package directory is refused).
Only numpy/scipy/trimesh are needed.  The original meshes are read-only inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
MODEL = 'igrisc_model_cam_260910'
REPO = MESH_DIR = URDF = None


def use_repo(repo):
    """Read the supplied jaw meshes from ``repo``, a checkout that contains igrisc_model_cam_260910/."""
    global REPO, MESH_DIR, URDF
    REPO = Path(repo).resolve()
    MESH_DIR = REPO / MODEL / 'meshes/igris_c_end_effector/1_dof_gripper'
    URDF = REPO / MODEL / 'urdf/igris_c_v2_torso_1dof.urdf'
    if not MESH_DIR.is_dir():
        raise FileNotFoundError(f'{MESH_DIR} not found; pass --repo <checkout containing {MODEL}/>')


_found = next((parent for parent in HERE.parents if (parent / MODEL).is_dir()), None)
if _found is not None:
    use_repo(_found)

HINGE_Y_MM, HINGE_Z_MM = 26.25, -42.25          # URDF moving-joint origin (|y|, z) in the fixed link frame

# ----------------------------------------------------------------------------------------------- parameters
# value, unit, basis, 1-sigma-ish uncertainty (mm or rad), what to measure to replace it
PARAMS = {
    'fixed_face_mm': (3.9, 'mm', 'photo-estimated (seam back-projected on the jaw side faces with the fitted camera: '
                      '3.94 mm mean at q 0.02; coupled with q_touch: 3.44 + 25.5*q_touch over q 0..0.03)', 0.6,
                      'fixed-jaw pad gripping face height s*y in the fixed link frame (= fixed pad thickness - 2 mm). '
                      'Measure: fixed pad thickness normal to the fixed finger plate (caliper)'),
    'fixed_base_mm': (-2.0, 'mm', 'mesh-derived', 0.0,
                      'pad base = plane of the old tip insert / structural fixed finger (s*y = -2 mm)'),
    'q_touch_rad': (0.020, 'rad', 'photo-estimated (camera fit with the moving angle free: best 0.02-0.03 rad; focal length '
                    'then 1280-1620 px = 24-26 mm-equiv. phone lens; a perfectly uniform fixed pad would favour 0.00)', 0.02,
                    'joint angle |q| at which the two inner pads touch (no object). Measure: joint reading when closed on '
                    'nothing, or the jaw opening at the tips when the pads touch'),
    'pad_z_tip_mm': (-156.5, 'mm', 'mesh-derived (old insert tip = jaw tip); photo: fixed pad flush with its white jaw tip', 1.0,
                     'pad tip end along the finger (fixed frame z)'),
    'pad_z_root_mm': (-88.3, 'mm', 'mesh-derived clearance (fixed pad must end at z <= -88.30 for 0.2 mm clearance to the '
                      'moving jaw body swinging over q in [q_touch, limit]; at the old insert end -86.5 it penetrates 1.04 mm); '
                      'photo: both pad root ends aligned at the seam, back-projected z -87.5..-89.3', 1.5,
                      'pad root end along the finger (fixed-frame z at q=0; the moving pad uses the same value in its joint '
                      'frame, z\' = z - (-42.25))'),
    'moving_tip_setback_mm': (2.3, 'mm', 'photo-estimated (lower pad tip at z -156.8..-157.1 vs lower white tip -159.2 and '
                              'upper pad tip -159.2..-159.4; fixed pad is flush with its jaw tip)', 1.0,
                              'moving pad tip set back from the moving jaw tip, along the finger'),
    'strip_thickness_mm': (6.6, 'mm', 'photo-estimated (back-projection, median of 400 columns)', 1.0,
                           'back strip thickness normal to the fixed jaw back face (old CAD back plate: 3.0 mm)'),
    'strip_groove_centres_z_mm': ([-116.7, -97.4, -79.2, -60.7, -42.4], 'mm', 'photo-estimated (shadow/silhouette)', 2.0,
                                  'z of the transverse grooves on the back strip (pitch ~18.3 mm)'),
    'strip_groove_width_mm': (2.5, 'mm', 'photo-estimated', 1.0, 'groove width along the finger'),
    'strip_groove_depth_mm': (2.5, 'mm', 'photo-estimated', 1.0, 'groove depth into the strip'),
    'density_g_cm3': (1.2, 'g/cm3', 'given (task)', 0.1, 'pad material density'),
}


def P(name):
    return PARAMS[name][0]


# ----------------------------------------------------------------------------------------------- source features
def _load(hand, part):
    m = trimesh.load(MESH_DIR / f'{hand}_1_dof_gripper_{part}.stl', force='mesh', process=True)
    m.apply_scale(0.01)
    return m


def _components(m):
    return [c for c in m.split(only_watertight=False) if len(c.faces) > 1]


def source_features(hand):
    """Extract (mm) the old-insert footprints and planes from the ORIGINAL meshes of this hand."""
    s = 1.0 if hand == 'l' else -1.0
    fx, mv = _load(hand, 'fixed'), _load(hand, 'moving')
    feats = {'sign': s}
    tip_insert = back_plate = None
    for c in _components(fx):
        b = c.bounds * 1000
        if b[0][0] > 39:
            continue                                            # wrist-camera housing
        if b[0][2] < -156 and abs(b[1][2] + 86.5) < 0.1 and abs((b[1][1] - b[0][1]) - 12.5) < 0.2:
            tip_insert = c
        elif (s * b[1][1] < -8 if s > 0 else b[0][1] > 8):
            back_plate = c
    inner_insert = None
    for c in _components(mv):
        b = c.bounds * 1000
        if b[0][0] < 17 and abs(b[1][2] + 44.25) < 0.1:
            inner_insert = c
    assert tip_insert is not None and back_plate is not None and inner_insert is not None, 'component detection failed'
    # fixed tip insert: base plane and x-z outline
    V = tip_insert.vertices * 1000
    base_y = V[:, 1].min() if s > 0 else V[:, 1].max()
    base = V[np.abs(V[:, 1] - base_y) < 1e-3][:, [0, 2]]
    feats['fixed_outline_xz'] = base[ConvexHull(base).vertices]
    feats['fixed_base_s_y'] = float(s * base_y)
    # moving inner insert: x-z' outline + interface plane (faces whose normal points toward the moving body)
    Vm = inner_insert.vertices * 1000
    feats['moving_outline_xz'] = Vm[ConvexHull(Vm[:, [0, 2]]).vertices][:, [0, 2]]
    n = inner_insert.face_normals
    Pm = inner_insert.triangles[(s * n[:, 1]) > 0.9].reshape(-1, 3) * 1000
    feats['moving_interface_y_eq_a_bz'] = np.linalg.lstsq(np.c_[np.ones(len(Pm)), Pm[:, 2]], Pm[:, 1], rcond=None)[0]
    # back plate: base plane on the back face and its x-z outline
    Vb = back_plate.vertices * 1000
    A = np.c_[np.ones(len(Vb)), Vb[:, 2]]
    cb = np.linalg.lstsq(A, Vb[:, 1], rcond=None)[0]
    for _ in range(10):                                          # keep the side nearest the jaw (base)
        r = s * (Vb[:, 1] - A @ cb)
        keep = r > np.percentile(r, 50)
        cb = np.linalg.lstsq(A[keep], Vb[keep, 1], rcond=None)[0]
    onb = np.abs(Vb[:, 1] - A @ cb) < 0.05
    ob = Vb[onb][:, [0, 2]]
    feats['back_plane_y_eq_a_bz'] = cb
    feats['back_outline_xz'] = ob[ConvexHull(ob).vertices]
    feats['old_tip_insert_volume_mm3'] = float(tip_insert.volume * 1e9) if tip_insert.is_watertight else None
    feats['old_back_plate_volume_mm3'] = float(back_plate.volume * 1e9) if back_plate.is_watertight else None
    feats['old_back_plate_thickness_mm'] = float(np.abs(Vb[:, 1] - A @ cb).max() / np.sqrt(1 + cb[1] ** 2))
    return feats


# ----------------------------------------------------------------------------------------------- geometry helpers
def clip_polygon_z(poly, zmin, zmax):
    """Clip a convex polygon given as (x, z) rows to zmin <= z <= zmax (Sutherland-Hodgman, two half-planes)."""
    def clip(pts, keep, inter):
        out = []
        for i in range(len(pts)):
            a, b = pts[i - 1], pts[i]
            ka, kb = keep(a), keep(b)
            if kb:
                if not ka:
                    out.append(inter(a, b))
                out.append(b)
            elif ka:
                out.append(inter(a, b))
        return np.array(out)
    def at(zc):
        return lambda a, b: a + (b - a) * ((zc - a[1]) / (b[1] - a[1]))
    p = clip(np.asarray(poly, float), lambda v: v[1] >= zmin - 1e-12, at(zmin))
    if len(p):
        p = clip(p, lambda v: v[1] <= zmax + 1e-12, at(zmax))
    return p


def convex_piece(points_mm):
    """Convex hull of points (mm) -> watertight convex trimesh in meters.  Vertices are rounded to float32 BEFORE
    the hull is taken, so the exported binary STL (float32) is exactly the convex hull of its own vertices."""
    pts = np.unique(np.asarray(points_mm, float) / 1000.0, axis=0).astype(np.float32).astype(np.float64)
    pts = np.unique(pts, axis=0)
    h = ConvexHull(pts)
    used = np.unique(h.simplices)
    remap = -np.ones(len(pts), int); remap[used] = np.arange(len(used))
    faces = remap[h.simplices]
    V = pts[used]
    # orient every facet outward using the qhull plane equations
    for i, (f, eq) in enumerate(zip(faces, h.equations)):
        n = np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]])
        if np.dot(n, eq[:3]) < 0:
            faces[i] = f[[0, 2, 1]]
    return trimesh.Trimesh(vertices=V, faces=faces, process=False)


def convexity_check(m):
    """(is_convex, max_vertex_outside_hull_m, rel_volume_gap): all vertices on/below every hull plane."""
    h = ConvexHull(m.vertices)
    d = m.vertices @ h.equations[:, :3].T + h.equations[:, 3]
    outside = float(d.max())
    rel = float((h.volume - m.volume) / h.volume)
    return bool(outside < 1e-9 and rel < 1e-6 and m.is_watertight and m.is_winding_consistent and m.volume > 0), outside, rel


def min_width_mm(m):
    """Exact minimum width of a convex polytope: min over facet normals and edge-pair directions."""
    V = m.vertices * 1000
    dirs = [n for n in m.face_normals]
    E = m.vertices[m.edges_unique] * 1000
    e = E[:, 1] - E[:, 0]
    e = e[np.linalg.norm(e, axis=1) > 1e-9]
    e /= np.linalg.norm(e, axis=1)[:, None]
    for i in range(len(e)):
        c = np.cross(e[i], e[i + 1:])
        nn = np.linalg.norm(c, axis=1)
        dirs.extend(c[nn > 1e-6] / nn[nn > 1e-6, None])
    D = np.asarray(dirs)
    proj = V @ D.T
    return float((proj.max(0) - proj.min(0)).min())


# ----------------------------------------------------------------------------------------------- pad pieces
def build_pieces(hand, feats, strip=True, grooves=True):
    s = feats['sign']
    pieces = []
    zt, zr = P('pad_z_tip_mm'), P('pad_z_root_mm')
    # 1) fixed inner pad: prism over the old tip-insert footprint between the base plane and the gripping face
    foot = clip_polygon_z(feats['fixed_outline_xz'], zt, zr)
    yb, yf = s * P('fixed_base_mm'), s * P('fixed_face_mm')
    pts = [(x, y, z) for x, z in foot for y in (yb, yf)]
    pieces.append(dict(name='inner_gripping_pad', link=f'{hand}_1_dof_gripper_fixed', jaw='fixed',
                       mesh=convex_piece(pts), role='inner gripping pad (replaces old tip insert)',
                       min_thickness_mm=abs(P('fixed_face_mm') - P('fixed_base_mm'))))
    # 2) moving inner pad (moving joint frame): footprint of the old inner insert, between the insert/body interface
    #    plane and the plane that coincides with the fixed pad face when q = s*q_touch.
    qt = P('q_touch_rad')
    a, b = feats['moving_interface_y_eq_a_bz']
    hf = P('fixed_face_mm')
    ztip_m = zt - HINGE_Z_MM + P('moving_tip_setback_mm')        # moving-frame z of the pad tip end
    zroot_m = zr - HINGE_Z_MM
    footm = clip_polygon_z(feats['moving_outline_xz'], ztip_m, zroot_m)
    pts = []
    nf = np.array([0.0, s * np.cos(qt), -np.sin(qt)])             # face-plane unit normal (moving frame), toward the body
    thick = []
    for x, z in footm:
        y_face = (hf - HINGE_Y_MM + np.sin(qt) * z) / (s * np.cos(qt))   # s*cos(qt)*y' - sin(qt)*z' = hf - 26.25
        y_int = a + b * z
        assert s * (y_int - y_face) > 1.0, 'moving pad thinner than 1 mm at z=%.2f' % z
        pts += [(x, y_face, z), (x, y_int, z)]
        thick.append(float(nf @ (np.array([x, y_int, z]) - np.array([x, y_face, z]))))
    pieces.append(dict(name='inner_gripping_pad', link=f'{hand}_1_dof_gripper_moving', jaw='moving',
                       mesh=convex_piece(pts), role='inner gripping pad (replaces old inner insert incl. arc recess)',
                       min_thickness_mm=min(thick), max_thickness_mm=max(thick)))
    # 3) back strip on the fixed jaw outer face: base layer + raised segments between transverse grooves
    if strip:
        ab, bb = feats['back_plane_y_eq_a_bz']
        nrm = -s * np.array([0.0, 1.0, -bb]) / np.sqrt(1 + bb ** 2)    # outward normal of the back face (away from jaw)
        outline = feats['back_outline_xz']
        base3 = np.array([(x, ab + bb * z, z) for x, z in outline])
        t = P('strip_thickness_mm')
        dg = P('strip_groove_depth_mm') if grooves else 0.0
        if dg > 0:
            assert t - dg >= 1.0 and dg >= 1.0, 'strip layers must be >= 1 mm'
            pieces.append(dict(name='back_strip_base', link=f'{hand}_1_dof_gripper_fixed', jaw='fixed',
                               mesh=convex_piece(np.vstack([base3, base3 + (t - dg) * nrm])), role='back strip base layer',
                               min_thickness_mm=t - dg))
            zs = sorted(P('strip_groove_centres_z_mm'))
            w = P('strip_groove_width_mm')
            zmin, zmax = outline[:, 1].min(), outline[:, 1].max()
            edges = [zmin] + [v for c in zs for v in (c - w / 2, c + w / 2)] + [zmax]
            for k in range(0, len(edges), 2):
                lo, hi = edges[k], edges[k + 1]
                seg = clip_polygon_z(outline, lo, hi)
                if len(seg) < 3 or hi - lo < 1.0:
                    continue
                seg3 = np.array([(x, ab + bb * z, z) for x, z in seg])
                pieces.append(dict(name=f'back_strip_rib_{k // 2}', link=f'{hand}_1_dof_gripper_fixed', jaw='fixed',
                                   mesh=convex_piece(np.vstack([seg3 + (t - dg) * nrm, seg3 + t * nrm])),
                                   role='back strip raised segment between grooves', min_thickness_mm=dg,
                                   z_range_mm=[float(lo), float(hi)]))
        else:
            pieces.append(dict(name='back_strip', link=f'{hand}_1_dof_gripper_fixed', jaw='fixed',
                               mesh=convex_piece(np.vstack([base3, base3 + t * nrm])), role='back strip (no grooves)',
                               min_thickness_mm=t))
    return pieces


# ----------------------------------------------------------------------------------------------- white bodies
def _close_with_interface_faces(comp, singles):
    """Close a non-watertight component with the stray single-face interface triangles of the assembly:
    group them by plane, try (none / +n / -n) per group, keep the smallest valid combination."""
    import itertools
    if comp.is_watertight:
        return comp, 0
    groups = []
    for c in singles:
        n = c.face_normals[0]; off = float(np.dot(n, c.vertices[0]))
        for g in groups:
            if abs(abs(np.dot(g['n'], n)) - 1) < 1e-6 and abs(abs(off) - abs(g['off'])) < 1e-7:
                g['faces'].append(c); break
        else:
            groups.append({'n': n, 'off': off, 'faces': [c]})
    best = None
    for combo in itertools.product((0, 1, -1), repeat=len(groups)):
        add = [c for g, o in zip(groups, combo) if o for c in g['faces'] if np.dot(c.face_normals[0], o * g['n']) > 0]
        if best is not None and len(add) >= best[1]:
            continue
        b = trimesh.util.concatenate([comp] + add)
        b.merge_vertices()
        if b.is_watertight and b.is_winding_consistent and b.volume > 0:
            best = (b, len(add))
    if best is None:
        raise RuntimeError('could not close component with interface faces')
    return best


def white_bodies(hand):
    """Original jaw meshes with the old inserts removed (fixed: tip insert + old 3 mm back plate, which the new
    strip encloses; moving: inner insert): the white structural parts under the pads.  Every kept piece is closed
    with the assembly's own interface triangles (no new geometry).  Returns {part: (mesh, [per-shell info])}."""
    s = 1.0 if hand == 'l' else -1.0
    out = {}
    for part in ('fixed', 'moving'):
        m = _load(hand, part)
        comps = m.split(only_watertight=False)
        singles = [c for c in comps if len(c.faces) == 1]
        keep = []
        for c in comps:
            if len(c.faces) == 1:
                continue
            b = c.bounds * 1000
            if part == 'fixed':
                is_insert = b[0][2] < -156 and abs(b[1][2] + 86.5) < 0.1 and abs((b[1][1] - b[0][1]) - 12.5) < 0.2
                is_plate = (s * b[1][1] < -8) if s > 0 else (b[0][1] > 8)
                if b[0][0] <= 39 and (is_insert or is_plate):
                    continue
            elif b[0][0] < 17 and abs(b[1][2] + 44.25) < 0.1:
                continue
            keep.append(c)
        shells, info = [], []
        for c in keep:
            cc, nadd = _close_with_interface_faces(c, singles)
            shells.append(cc)
            info.append({'faces': int(len(cc.faces)), 'watertight': bool(cc.is_watertight), 'volume_m3': float(cc.volume),
                         'interface_faces_added': int(nadd), 'bounds_m': cc.bounds.round(5).tolist()})
        out[part] = (trimesh.util.concatenate(shells), info)
    return out


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo', help=f'checkout containing {MODEL}/ (default: the nearest parent of this file)')
    ap.add_argument('--out', help='output directory (default: <repo>/outputs/gripper_pads/make_pads)')
    ap.add_argument('--set', action='append', default=[], help='override a parameter: name=value (JSON value)')
    ap.add_argument('--no-strip', action='store_true')
    ap.add_argument('--no-grooves', action='store_true')
    ap.add_argument('--white-bodies', action='store_true', help='also export insert-free white jaw bodies')
    args = ap.parse_args()
    for kv in args.set:
        k, v = kv.split('=', 1)
        assert k in PARAMS, f'unknown parameter {k}'
        PARAMS[k] = (json.loads(v), PARAMS[k][1], 'user-set', 0.0, PARAMS[k][4])
    if args.repo is not None:
        use_repo(args.repo)
    elif REPO is None:
        ap.error(f'no {MODEL}/ above {HERE}; pass --repo')
    out = Path(args.out).resolve() if args.out else REPO / 'outputs' / 'gripper_pads' / 'make_pads'
    if out.is_relative_to(HERE.parent):
        ap.error(f'--out {out} is inside the pinned pad package {HERE.parent}')
    out.mkdir(parents=True, exist_ok=True)
    rho = P('density_g_cm3') * 1000.0                           # kg/m^3
    report = {'pieces': [], 'features': {}, 'white_bodies': []}
    for hand in 'lr':
        feats = source_features(hand)
        report['features'][hand] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in feats.items()}
        pieces = build_pieces(hand, feats, strip=not args.no_strip, grooves=not args.no_grooves)
        counters = {'fixed': 0, 'moving': 0}
        for pc in pieces:
            m = pc['mesh']
            k = counters[pc['jaw']]
            counters[pc['jaw']] += 1
            path = out / f'{hand}_{pc["jaw"]}_pad_{k}.stl'
            m.export(path)
            back = trimesh.load(path, force='mesh', process=True)   # merges identical float32 vertices only
            is_cvx, outside, relgap = convexity_check(back)
            rec = {
                'hand': {'l': 'left', 'r': 'right'}[hand], 'source_urdf_link': pc['link'], 'piece': pc['name'],
                'role': pc['role'], 'stl': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'convex': is_cvx, 'max_vertex_outside_hull_m': outside, 'hull_volume_rel_gap': relgap,
                'watertight': bool(back.is_watertight), 'n_vertices': int(len(back.vertices)), 'n_faces': int(len(back.faces)),
                'volume_m3': float(back.volume), 'bbox_m': back.bounds.round(6).tolist(),
                'mass_kg_est': float(back.volume * rho), 'min_width_mm': round(min_width_mm(back), 3),
                'min_thickness_mm': round(float(pc['min_thickness_mm']), 3),
            }
            for extra in ('max_thickness_mm', 'z_range_mm'):
                if extra in pc:
                    rec[extra] = pc[extra]
            report['pieces'].append(rec)
            print(f"{path.name:22s} {pc['name']:22s} V={rec['volume_m3'] * 1e6:7.3f} cm3 m={rec['mass_kg_est'] * 1000:6.2f} g "
                  f"convex={rec['convex']} watertight={rec['watertight']} min_thickness={rec['min_thickness_mm']:.2f} mm")
        if args.white_bodies:
            wb = white_bodies(hand)
            for part, (body, info) in wb.items():
                path = out / 'white_bodies' / f'{hand}_{part}_white_body.stl'
                path.parent.mkdir(exist_ok=True)
                body.export(path)
                report['white_bodies'].append({'hand': hand, 'link': f'{hand}_1_dof_gripper_{part}', 'stl': f'white_bodies/{path.name}',
                                               'shells': info, 'faces': int(len(body.faces)),
                                               'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
                print('white body', path.name, 'shells', [(i['faces'], i['watertight'], round(i['volume_m3'] * 1e6, 2)) for i in info])
    report['parameters'] = {k: {'value': v[0], 'unit': v[1], 'basis': v[2], 'uncertainty': v[3], 'meaning': v[4]} for k, v in PARAMS.items()}
    (out / 'make_pads_report.json').write_text(json.dumps(report, indent=1))
    print('wrote', out / 'make_pads_report.json')


if __name__ == '__main__':
    main()
