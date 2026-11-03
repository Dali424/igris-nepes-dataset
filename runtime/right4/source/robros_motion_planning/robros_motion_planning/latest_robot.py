"""Canonical naming adapter for the supplied 260910 IGRIS-C two-finger model.

The authored meshes, inertias, joint frames, limits, and cameras are retained.
This is a derived asset, not a newly designed or parallel-jaw gripper. Isaac
imports are confined to configure_latest_robot; USD imports are lazy as well.

``gripper_variant`` selects a registered jaw model. ``authored_v0`` (default) is
the supplied jaws, unchanged: same fingerprint, URDF bytes and manifest. A pad
variant installs a versioned, hash-pinned jaw package. Each jaw's authored mesh
is replaced by the package's white jaw body (USD: the authored jaw visual and
collider are deactivated; URDF: the jaw mesh is swapped), and the package's
rubber pad pieces are added as convex colliders bound to their own physics
material. Joints, limits, gripper targets, masses, inertias and cameras are not
changed; each variant has its own gripper actuator profile.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import re
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from .constants import TELEOPERATION_JOINT_NAMES, WRIST_HAND_JOINT_NAMES
from .robot_model import default_urdf_path, load_robot_model, resolve_mesh_path
from .snapshot import matrix_to_pose


WRIST_JOINT_NAMES = WRIST_HAND_JOINT_NAMES[:6]
GRIPPER_JOINT_NAMES = ("l_1_dof_gripper_moving", "r_1_dof_gripper_moving")
NECK_JOINT_NAMES = ("Joint_Neck_Yaw", "Joint_Neck_Pitch")
AUXILIARY_JOINT_NAMES = WRIST_JOINT_NAMES + GRIPPER_JOINT_NAMES + NECK_JOINT_NAMES
COMMAND_JOINT_NAMES = TELEOPERATION_JOINT_NAMES + AUXILIARY_JOINT_NAMES
GRIPPER_OPEN_TARGETS = dict(zip(GRIPPER_JOINT_NAMES, (1.5, -1.5)))
GRIPPER_CLOSED_TARGETS = dict.fromkeys(GRIPPER_JOINT_NAMES, 0.0)
_ADAPTER_VERSION = "igris_c_260910_canonical_v9"
_EMPTY_FIXED_FRAMES = {"base_link", "l_camera_link", "r_camera_link", "l_wrist_camera_link", "r_wrist_camera_link"}
_FRAME_MASS_KG = 1e-6
_FRAME_INERTIA_KG_M2 = 1e-9
_JAW_CONTACT_OFFSET_M = 0.0005
_CONTACT_SOLVER_ITERATIONS = {"position": 32, "velocity": 8}
_JAW_DECOMPOSITION_SETTINGS = {
    "errorPercentage": 0.1, "hullVertexLimit": 64, "maxConvexHulls": 128,
    "minThickness": 0.0001, "shrinkWrap": True, "voxelResolution": 1_000_000,
}

DEFAULT_GRIPPER_VARIANT = "authored_v0"
FRICTION_COMBINE_MODES = ("average", "min", "multiply", "max")
_PAD_PACKAGE_SCHEMA = "igris_c_gripper_pad_package_v1"
_PAD_BUILDER_VERSION = "igris_c_gripper_pads_builder_v2"
_PAD_ASSET_ROOT = Path(__file__).resolve().parent / "assets" / "gripper_pads"
_JAW_MESH_DIR = "meshes/igris_c_end_effector/1_dof_gripper"
# Canonical jaw body -> source STL stem of its authored visual/collision mesh.
_JAW_LINK_MESHES = {"Left_Hand": "l_1_dof_gripper_fixed", "l_1_dof_gripper_moving": "l_1_dof_gripper_moving",
                    "Right_Hand": "r_1_dof_gripper_fixed", "r_1_dof_gripper_moving": "r_1_dof_gripper_moving"}
_PAD_ID = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,62}")
_PAD_HULL_VERTEX_LIMIT = 64          # PhysX convexHull cooking default (physxConvexHullCollision:hullVertexLimit)
# Authored limits, the closed target 0 and the disabled articulation self-collision are retained, so a jaw
# closed on nothing passes its first pad contact by this angle at most (0.025 rad ~ 2.9 mm at the 0.117 m tip).
_PAD_MAX_EMPTY_CLOSE_TOUCH_RAD = .025
_PAD_DENSITY_ESTIMATE_KG_M3 = 1200.  # rubber estimate; recorded only (explicit link masses are retained)
_ACTUATOR_UNITS = {"stiffness": "N*m/rad", "damping": "N*m*s/rad", "effort_limit_sim": "N*m", "velocity_limit_sim": "rad/s"}
# Visual-only: 0.5 is the OmniPBR default the authored jaw materials use; matte rubber pads.
_VISUAL_ROUGHNESS = {"pad": .85, "jaw_body": .5}
# Pad-variant collider choices; _pad_builder_settings puts every one of them into the pad fingerprint.
_PAD_COLLIDER_APPROXIMATION = "convexHull"           # each pad piece is one exactly convex STL
_JAW_BODY_APPROXIMATION = "convexDecomposition"      # white bodies keep their channels and camera housing
# White jaw bodies: the authored-jaw settings at 4x the voxel resolution (same 128-hull cap); basis recorded below.
_JAW_BODY_DECOMPOSITION_SETTINGS = {**_JAW_DECOMPOSITION_SETTINGS, "voxelResolution": 4_000_000}
_JAW_BODY_DECOMPOSITION_BASIS = (
    "isaac_5.1_cooked_hull_dump: at 4M voxels/128 hulls no hull rises above a gripping or back-strip pad face (0 mm) "
    "and other pad faces see <= 0.006 mm within their footprints (1M voxels: 0.12 mm on a pad side face; 256-1024 "
    "hulls: up to 7.4 mm at back-strip facets); every setting tried, like the authored_v0 jaws, partly fills the "
    "jaw V-channel beyond the fixed-pad root end (z > -88.3 mm)")
_PAD_MATERIAL_BINDING_STRENGTH = "weakerThanDescendants"
# Runtime/physics contract per jaw model; geometry and colours live in the hash-pinned asset package.
_GRIPPER_VARIANT_SPECS = {
    "authored_v0": {
        "runtime_implicit_actuator": {"stiffness": 60., "damping": 1., "effort_limit_sim": 5., "velocity_limit_sim": 1.2},
    },
    "filled_pad_v1": {
        # 3 N*m: manufacturer rating (user supplied). 100 N*m/rad: in the Isaac closed-loop pinch sweep (TGS
        # 250 Hz, 32/8) d=1 held 2.9-3.35 N*m while k*e stayed in [4.5, 10] N*m and fell to 1.3-2.0 N*m at
        # k*e >= 12 N*m. The pads take a 3.75 mm CRS rim between parallel faces at q 0.0628 rad -> 6.3 N*m.
        "runtime_implicit_actuator": {"stiffness": 100., "damping": 1., "effort_limit_sim": 3., "velocity_limit_sim": 1.2},
        "actuator_basis": {
            "effort_source": "manufacturer_rating_3Nm_user_supplied",
            "stiffness_source": "isaac_closed_loop_pinch_sweep_k100_d1_held_2.88_to_3.20Nm_for_error_0.03_to_0.08rad",
            "design_contact_q_rad": .0628,
            "design_contact_basis": "pads.json evaluation: 3.75 mm CRS rim between parallel pads",
            "held_command_window_Nm": [4.5, 10.],
        },
        "pad_material": {"static_friction": 1.2, "dynamic_friction": .9, "restitution": 0.,
                         "friction_combine_mode": "max"},
        "pad_material_source": "user_supplied_rubber_estimate_static_1.2_dynamic_0.9",
        "asset_dir": _PAD_ASSET_ROOT / "filled_pad_v1",
        # New geometry is a new variant (new directory, entry and pin), never an edit of these files.
        "package_json_sha256": "f12ba3312e00a9aa85098b3c0099f86e22447be2f08bd74e8fefc385b1b1001e",
    },
}
GRIPPER_VARIANTS = tuple(_GRIPPER_VARIANT_SPECS)
_PAD_MATERIAL_KEYS = ("static_friction", "dynamic_friction", "restitution", "friction_combine_mode")


def _pad_builder_settings() -> dict:
    """Builder constants that shape a pad variant's derived USD; fingerprinted and recorded with each build."""
    return {"collider_offsets_m": {"contact": _JAW_CONTACT_OFFSET_M, "rest": 0.},
            "jaw_body_approximation": _JAW_BODY_APPROXIMATION,
            "jaw_body_decomposition": dict(_JAW_BODY_DECOMPOSITION_SETTINGS),
            "pad_approximation": _PAD_COLLIDER_APPROXIMATION,
            "pad_material_binding_strength": _PAD_MATERIAL_BINDING_STRENGTH,
            "visual_roughness": dict(_VISUAL_ROUGHNESS)}


def _fingerprint(identity: Mapping) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def validate_gripper_variant(name) -> str:
    """Return a registered gripper variant name; no file or USD access."""
    if not isinstance(name, str) or name not in _GRIPPER_VARIANT_SPECS:
        raise ValueError("gripper_variant must be one of: " + ", ".join(_GRIPPER_VARIANT_SPECS))
    return name


def require_gripper_package(name) -> str:
    """Fail before Isaac starts if a pad variant's pinned package is absent or was edited."""
    spec = _GRIPPER_VARIANT_SPECS[validate_gripper_variant(name)]
    if "asset_dir" in spec:
        path = Path(spec["asset_dir"]) / "package.json"
        if not path.is_file() or _sha(path) != spec["package_json_sha256"]:
            raise ValueError(f"{name} pad package is not installed or differs from its pinned sha256: {path}")
    return name


def _pad_material(name: str, override: Mapping | None) -> dict | None:
    """Validated pad physics material of a variant, optionally overridden for sensitivity runs."""
    spec = _GRIPPER_VARIANT_SPECS[validate_gripper_variant(name)]
    if "pad_material" not in spec:
        if override is not None:
            raise ValueError("pad material overrides require a gripper variant with pads")
        return None
    material = dict(spec["pad_material"])
    if override is not None:
        if not isinstance(override, Mapping) or not override or set(override) - set(material):
            raise ValueError("pad_material_override may set only " + ", ".join(sorted(material)))
        material.update(override)
    for key in ("static_friction", "dynamic_friction", "restitution"):
        value = material[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"pad {key} must be finite and nonnegative")
        material[key] = float(value)
    if not material["dynamic_friction"] <= material["static_friction"] <= 5. or material["restitution"] > 1.:
        raise ValueError("pad material requires 0 <= dynamic <= static <= 5 friction and restitution <= 1")
    if material["friction_combine_mode"] not in FRICTION_COMBINE_MODES:
        raise ValueError("friction_combine_mode must be one of: " + ", ".join(FRICTION_COMBINE_MODES))
    return material


_PAD_CONTACT_PATCH_KEYS = ("min_torsional_patch_radius_m", "torsional_patch_radius_m")
_PAD_CONTACT_PATCH_MAX_M = .02


def validate_pad_contact_patch(name: str, settings: Mapping | None) -> dict | None:
    """Validated PhysX torsional contact patch of a pad variant (profile setting), or None.

    A rigid pad meets the thin CRS rim in one or two points, so PhysX applies no
    torsional friction and the ring spins about the pinch axis under gravity
    (run_45). The real pads are filled rubber with a finite contact patch; this
    setting authors ``physxCollision:minTorsionalPatchRadius`` /
    ``torsionalPatchRadius`` on the pad colliders only. It is part of the
    profile's jaw model (fingerprinted, admissible), never a dev override.
    """
    if settings is None:
        return None
    spec = _GRIPPER_VARIANT_SPECS[validate_gripper_variant(name)]
    if "pad_material" not in spec:
        raise ValueError("gripper_pad_contact_patch requires a gripper variant with pads")
    if not isinstance(settings, Mapping) or "min_torsional_patch_radius_m" not in settings \
            or set(settings) - set(_PAD_CONTACT_PATCH_KEYS):
        raise ValueError("gripper_pad_contact_patch may set only " + ", ".join(_PAD_CONTACT_PATCH_KEYS)
                         + " and requires min_torsional_patch_radius_m")
    result = {}
    minimum = settings["min_torsional_patch_radius_m"]
    radius = settings.get("torsional_patch_radius_m", minimum)
    for key, value in (("min_torsional_patch_radius_m", minimum), ("torsional_patch_radius_m", radius)):
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or not 0. < value <= _PAD_CONTACT_PATCH_MAX_M):
            raise ValueError(f"gripper_pad_contact_patch.{key} must be in (0, {_PAD_CONTACT_PATCH_MAX_M}] metres")
        result[key] = float(value)
    return result


_PAD_CONTACT_SPHERE_KEYS = ("radius_m", "pitch_m", "min_torsional_patch_radius_m", "torsional_patch_radius_m",
                            "protrusion_m", "retain_pad_collider")
_PAD_CONTACT_SPHERE_RADIUS_BOUNDS_M = (.002, .015)
_PAD_CONTACT_SPHERE_PITCH_BOUNDS_M = (.002, .03)
_PAD_CONTACT_SPHERE_PROTRUSION_BOUNDS_M = (0., .001)
_PAD_CONTACT_SPHERE_PROTRUSION_DEFAULT_M = .0003
_PAD_GRIPPING_PIECE_ID = "inner_gripping_pad"
# Sign of the jaw-frame y component of the gripping face normal (the face that meets the opposing jaw).
# The moving jaw hangs at +y (left) / -y (right) of its fixed jaw (URDF hinge origin (0, +-0.02625, -0.04225)),
# so a fixed pad grips with its +y (left) / -y (right) face and a moving pad with the opposite sign in its own
# link frame; pads.json: "fixed-jaw pad gripping face height s*y = 3.9", base plane s*y = -2 on the jaw body.
_PAD_GRIPPING_FACE_NORMAL_SIGN_Y = {"Left_Hand": 1., "l_1_dof_gripper_moving": -1.,
                                    "Right_Hand": -1., "r_1_dof_gripper_moving": 1.}
_PAD_GRIPPING_FACE_COPLANARITY_M = 1e-4


def validate_pad_contact_spheres(name: str, settings: Mapping | None) -> dict | None:
    """Validated sphere-array collider of the inner gripping pads (profile setting), or None.

    PhysX patch friction applies at most two friction anchors per shape pair and
    torsional friction only to single-point patches, so a convex pad face on the
    thin CRS rim gives 1-2 anchors a few millimetres apart and the ring creeps about
    the pinch axis under its 0.14-0.26 N m gravity torque (run_45/run_55; compliant
    contacts made it worse, run_56). The real pads are filled rubber with a
    distributed contact patch. This setting replaces each ``inner_gripping_pad``
    collider by a grid of sphere colliders whose apexes lie on the pad's gripping
    face plane (same first-touch geometry, pitch ``pitch_m``, radius ``radius_m``):
    every sphere meets the ring in a single-point patch with its own anchor and
    torsional friction (``min_torsional_patch_radius_m``/``torsional_patch_radius_m``).
    By default the convex pad collider is retained and the sphere apexes stand
    ``protrusion_m`` (default 0.3 mm) above its face, so the spheres carry the
    pinch while the pad face remains a backstop that limits ring tilt and keeps
    thin rim overlaps in contact; ``retain_pad_collider: false`` and
    ``protrusion_m: 0`` give sphere-only pads with apexes on the face plane.
    Material, offsets, visuals, URDF collision envelope, masses and gates are
    untouched; the layout is part of the fingerprinted jaw model (admissible).
    """
    if settings is None:
        return None
    spec = _GRIPPER_VARIANT_SPECS[validate_gripper_variant(name)]
    if "pad_material" not in spec:
        raise ValueError("gripper_pad_contact_spheres requires a gripper variant with pads")
    required = set(_PAD_CONTACT_SPHERE_KEYS[:3])
    if (not isinstance(settings, Mapping) or not required <= set(settings)
            or set(settings) - set(_PAD_CONTACT_SPHERE_KEYS)):
        raise ValueError("gripper_pad_contact_spheres requires radius_m, pitch_m and min_torsional_patch_radius_m"
                         " and may add only torsional_patch_radius_m")
    values = dict(settings)
    values.setdefault("torsional_patch_radius_m", values["min_torsional_patch_radius_m"])
    values.setdefault("protrusion_m", _PAD_CONTACT_SPHERE_PROTRUSION_DEFAULT_M)
    values.setdefault("retain_pad_collider", True)
    bounds = {"radius_m": _PAD_CONTACT_SPHERE_RADIUS_BOUNDS_M, "pitch_m": _PAD_CONTACT_SPHERE_PITCH_BOUNDS_M,
              "min_torsional_patch_radius_m": (1e-6, _PAD_CONTACT_PATCH_MAX_M),
              "torsional_patch_radius_m": (1e-6, _PAD_CONTACT_PATCH_MAX_M),
              "protrusion_m": _PAD_CONTACT_SPHERE_PROTRUSION_BOUNDS_M}
    result = {}
    for key in _PAD_CONTACT_SPHERE_KEYS[:5]:
        value = values[key]
        low, high = bounds[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or not low <= value <= high):
            raise ValueError(f"gripper_pad_contact_spheres.{key} must be within [{low:g}, {high:g}] metres")
        result[key] = float(value)
    if not isinstance(values["retain_pad_collider"], bool):
        raise ValueError("gripper_pad_contact_spheres.retain_pad_collider must be a boolean")
    result["retain_pad_collider"] = values["retain_pad_collider"]
    if result["protrusion_m"] >= result["radius_m"]:
        raise ValueError("gripper_pad_contact_spheres.protrusion_m must be smaller than radius_m")
    return result


def pad_gripping_face(piece: Mapping, normal_sign_y: float) -> dict:
    """Largest planar face of a convex pad piece whose outward normal points toward the opposing jaw.

    Returns the face plane (``origin``, unit ``normal``), in-plane axes ``u`` (jaw x
    projected) and ``v`` = normal x u, and the face extents along them; pure geometry.
    """
    vertices = np.asarray(piece["vertices"], dtype=float)
    faces = np.asarray(piece["faces"], dtype=np.int64)
    normals = np.asarray(piece["face_normals"], dtype=float)
    triangles = vertices[faces]
    area = .5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    groups: dict = {}
    for index, normal in enumerate(normals):
        groups.setdefault(tuple(np.round(normal, 4)), []).append(index)
    candidates = [indices for key, indices in groups.items() if normal_sign_y * key[1] > .9]
    if not candidates:
        raise ValueError(f"pad piece {piece.get('id')} has no gripping face toward the opposing jaw")
    best = max(candidates, key=lambda indices: float(area[indices].sum()))
    points = triangles[best].reshape(-1, 3)
    normal = normals[best].mean(0)
    normal /= np.linalg.norm(normal)
    origin = points.mean(0)
    if np.abs((points - origin) @ normal).max() > 1e-6:
        raise ValueError(f"pad piece {piece.get('id')} gripping face is not planar")
    u = np.array([1., 0., 0.]) - normal[0] * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    uv = (points - origin) @ np.stack([u, v], 1)
    return {"normal": normal, "origin": origin, "u": u, "v": v,
            "u_range": (float(uv[:, 0].min()), float(uv[:, 0].max())),
            "v_range": (float(uv[:, 1].min()), float(uv[:, 1].max())), "area_m2": float(area[best].sum())}


def gripping_face_mismatch(fixed_face: Mapping, moving_face: Mapping, moving_in_fixed: np.ndarray) -> tuple[float, float]:
    """(normal dot product, plane distance in metres) of a moving gripping face expressed in the fixed jaw frame.

    Facing pads give a dot product of -1 and a zero distance at the package first-touch angle.
    """
    rotation, translation = np.asarray(moving_in_fixed)[:3, :3], np.asarray(moving_in_fixed)[:3, 3]
    normal = rotation @ np.asarray(moving_face["normal"], dtype=float)
    origin = rotation @ np.asarray(moving_face["origin"], dtype=float) + translation
    fixed_normal = np.asarray(fixed_face["normal"], dtype=float)
    return float(normal @ fixed_normal), float(abs((origin - np.asarray(fixed_face["origin"], dtype=float)) @ fixed_normal))


def _gripper_hinge_transform(tree, moving_link: str, fixed_link: str, angle: float) -> np.ndarray:
    """Fixed-jaw-frame pose of a moving jaw link from the source URDF revolute joint at ``angle``."""
    from .robot_model import _origin, _vector
    for joint in tree.getroot().findall("joint"):
        child, parent = joint.find("child"), joint.find("parent")
        if child is None or child.get("link") != moving_link:
            continue
        if parent is None or parent.get("link") != fixed_link or joint.get("type") != "revolute":
            raise ValueError(f"Gripper joint of {moving_link} must be a revolute child of {fixed_link}")
        axis = _vector(joint.find("axis").get("xyz") if joint.find("axis") is not None else None, (1., 0., 0.))
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        motion = np.eye(4)
        motion[:3, :3] += math.sin(angle) * skew + (1. - math.cos(angle)) * skew @ skew
        return _origin(joint.find("origin")) @ motion
    raise ValueError(f"Source URDF has no joint whose child is {moving_link}")


def _check_gripping_faces(pads: Mapping, tree) -> dict:
    """Raise unless each hand's selected fixed and moving gripping faces face each other and coincide at first touch."""
    report = {}
    for fixed_link, moving_link in (("Left_Hand", "l_1_dof_gripper_moving"), ("Right_Hand", "r_1_dof_gripper_moving")):
        pieces = {link: next(piece for piece in pads["jaws"][link]["pads"] if piece["id"] == _PAD_GRIPPING_PIECE_ID)
                  for link in (fixed_link, moving_link)}
        faces = {link: pad_gripping_face(pieces[link], _PAD_GRIPPING_FACE_NORMAL_SIGN_Y[link])
                 for link in (fixed_link, moving_link)}
        touch = pads["first_touch_q_rad"][moving_link]
        hinge = _gripper_hinge_transform(tree, _JAW_LINK_MESHES[moving_link], _JAW_LINK_MESHES[fixed_link], touch)
        dot, distance = gripping_face_mismatch(faces[fixed_link], faces[moving_link], hinge)
        if dot > -.999 or distance > _PAD_GRIPPING_FACE_COPLANARITY_M:
            raise ValueError(f"{fixed_link}/{moving_link} gripping faces do not meet at the package first touch "
                             f"(normal dot {dot:.4f}, plane distance {distance * 1000:.3f} mm)")
        report[fixed_link] = {"moving_link": moving_link, "first_touch_q_rad": touch, "normal_dot": dot,
                              "plane_distance_m": distance}
    return report


def pad_contact_sphere_centres(face: Mapping, radius: float, pitch: float, protrusion: float = 0.) -> np.ndarray:
    """Centred grid of sphere centres (jaw frame) whose apexes stand ``protrusion`` above the gripping face plane,
    inset from the face outline by one radius."""
    axes = []
    for low, high in (face["u_range"], face["v_range"]):
        span = high - low - 2. * radius
        if span < pitch:
            raise ValueError("gripper_pad_contact_spheres: the pad face holds fewer than two spheres per axis")
        count = int(math.floor(span / pitch + 1e-9)) + 1
        start = low + radius + (span - (count - 1) * pitch) / 2.
        axes.append([start + k * pitch for k in range(count)])
    centres = [np.asarray(face["origin"]) + a * np.asarray(face["u"]) + b * np.asarray(face["v"])
               - (radius - protrusion) * np.asarray(face["normal"]) for a in axes[0] for b in axes[1]]
    return np.array(centres, dtype=float)


_PAD_COMPLIANT_KEYS = ("stiffness_n_m", "damping_n_s_m")
_PAD_COMPLIANT_STIFFNESS_BOUNDS_N_M = (1e3, 1e7)
_PAD_COMPLIANT_DAMPING_BOUNDS_N_S_M = (0., 1e4)


def validate_pad_compliant_contact(name: str, settings: Mapping | None) -> dict | None:
    """Validated PhysX compliant (spring) contact of the pad material (profile setting), or None.

    Rigid pads meet the thin CRS rim as a line: under a 1-2 degree wrist compliance
    the pad plane lifts off everywhere but one edge, the contact manifold collapses
    to 1-2 points 3-6 mm apart and the friction anchors can no longer resist the
    0.25 N m gravity torque about the pinch axis (run_45/run_55). The real pads are
    filled rubber. This authors ``physxMaterial:compliantContactStiffness`` and
    ``compliantContactDamping`` on the pad physics material only, so pad-ring
    contacts become penetrating springs that keep a spread manifold under tilt.
    It is part of the profile's fingerprinted jaw model (admissible), never a dev
    override; friction, offsets, masses and gates are untouched.
    """
    if settings is None:
        return None
    spec = _GRIPPER_VARIANT_SPECS[validate_gripper_variant(name)]
    if "pad_material" not in spec:
        raise ValueError("gripper_pad_compliant_contact requires a gripper variant with pads")
    if not isinstance(settings, Mapping) or set(settings) != set(_PAD_COMPLIANT_KEYS):
        raise ValueError("gripper_pad_compliant_contact requires exactly " + ", ".join(_PAD_COMPLIANT_KEYS))
    result = {}
    for key, (low, high) in (("stiffness_n_m", _PAD_COMPLIANT_STIFFNESS_BOUNDS_N_M),
                             ("damping_n_s_m", _PAD_COMPLIANT_DAMPING_BOUNDS_N_S_M)):
        value = settings[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or not low <= value <= high):
            raise ValueError(f"gripper_pad_compliant_contact.{key} must be within [{low:g}, {high:g}]")
        result[key] = float(value)
    return result


def resolve_gripper_selection(config: Mapping, cli_variant: str | None = None, *,
                              dev_pad_friction=None, dev_pad_friction_combine: str | None = None) -> dict:
    """Resolve one run's jaw model; the profile's ``gripper_variant`` is authoritative.

    A CLI variant selects a model only for a profile that declares none; a
    conflicting declaration is an error. Dev pad-material overrides are
    sensitivity inputs: fingerprinted, recorded, never collection-admissible.
    """
    declared = config.get("gripper_variant")
    if declared is not None:
        validate_gripper_variant(declared)
    if cli_variant is not None:
        validate_gripper_variant(cli_variant)
        if declared is not None and cli_variant != declared:
            raise ValueError(f"--gripper_variant {cli_variant} conflicts with the profile's gripper_variant {declared}")
    name = declared or cli_variant or DEFAULT_GRIPPER_VARIANT
    override = None
    if dev_pad_friction is not None or dev_pad_friction_combine is not None:
        override = {}
        if dev_pad_friction is not None:
            static, dynamic = dev_pad_friction
            override.update(static_friction=static, dynamic_friction=dynamic)
        if dev_pad_friction_combine is not None:
            override["friction_combine_mode"] = dev_pad_friction_combine
        _pad_material(name, override)
    require_gripper_package(name)
    contact_patch = validate_pad_contact_patch(name, config.get("gripper_pad_contact_patch"))
    compliant_contact = validate_pad_compliant_contact(name, config.get("gripper_pad_compliant_contact"))
    contact_spheres = validate_pad_contact_spheres(name, config.get("gripper_pad_contact_spheres"))
    source = "profile" if declared is not None else "cli" if cli_variant is not None else "default"
    return {"variant": name, "selection_source": source, "profile_gripper_variant": declared,
            "cli_gripper_variant": cli_variant, "dev_pad_material_override": override,
            "pad_contact_patch": contact_patch, "pad_compliant_contact": compliant_contact,
            "pad_contact_spheres": contact_spheres,
            "collection_admissible": override is None and (source != "cli" or name == DEFAULT_GRIPPER_VARIANT)}


def _manifest_identity(manifest: Mapping) -> dict:
    """The fingerprint identity a pad-variant manifest records (KeyError/TypeError when a field is missing)."""
    pads = manifest["gripper_pads"]
    material = pads["physics_material"]
    identity = {"version": manifest["schema"], "sources": manifest["source_hashes"],
                "joint_aliases": manifest["source_joint_to_canonical"],
                "link_aliases": manifest["source_link_to_canonical"],
                "runtime_gripper_actuator": manifest["gripper_actuator_profile"]["runtime_implicit_actuator"],
                "gripper_variant": {"name": manifest["gripper_variant"], "builder": pads["builder"],
                                    "builder_settings": pads["builder_settings"], "asset_sha256": pads["asset_sha256"],
                                    "pad_material": {key: material[key] for key in _PAD_MATERIAL_KEYS},
                                    "pad_material_override": material["dev_override"]}}
    if pads.get("contact_patch") is not None:  # absent/None keeps the pre-patch identity and fingerprint
        identity["gripper_variant"]["pad_contact_patch"] = dict(pads["contact_patch"])
    if pads.get("compliant_contact") is not None:
        identity["gripper_variant"]["pad_compliant_contact"] = dict(pads["compliant_contact"])
    if pads.get("contact_spheres") is not None:
        identity["gripper_variant"]["pad_contact_spheres"] = dict(pads["contact_spheres"])
    return identity


def verify_gripper_manifest(manifest, name, pad_contact_patch: Mapping | None = None,
                            pad_compliant_contact: Mapping | None = None,
                            pad_contact_spheres: Mapping | None = None) -> None:
    """Raise ValueError unless ``manifest`` is a clean build of pad variant ``name``.

    Clean: the spec's actuator and pad material, the pinned package, this builder
    and its settings, no development override, exactly the profile-declared
    contact patch (``pad_contact_patch``; None = none authored), and a
    fingerprint that is the hash of exactly these recorded inputs.
    """
    spec = _GRIPPER_VARIANT_SPECS[validate_gripper_variant(name)]
    if "asset_dir" not in spec:
        raise ValueError(f"{name} has no pad package to verify")
    expected_patch = validate_pad_contact_patch(name, pad_contact_patch)
    expected_compliance = validate_pad_compliant_contact(name, pad_compliant_contact)
    expected_spheres = validate_pad_contact_spheres(name, pad_contact_spheres)
    try:
        identity = _manifest_identity(manifest)
        variant = identity["gripper_variant"]
        clean = (identity["version"] == _ADAPTER_VERSION and variant["name"] == name
                 and variant["builder"] == _PAD_BUILDER_VERSION and variant["builder_settings"] == _pad_builder_settings()
                 and variant["asset_sha256"].get("package.json") == spec["package_json_sha256"]
                 and variant["pad_material"] == spec["pad_material"] and variant["pad_material_override"] is None
                 and variant.get("pad_contact_patch") == expected_patch
                 and variant.get("pad_compliant_contact") == expected_compliance
                 and variant.get("pad_contact_spheres") == expected_spheres
                 and identity["runtime_gripper_actuator"] == spec["runtime_implicit_actuator"])
        fingerprint = manifest["fingerprint"]
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"{name} robot-model manifest lacks its gripper identity: {error!r}") from None
    if not clean:
        raise ValueError(f"{name} robot-model manifest is not a clean build of its variant spec")
    if _fingerprint(identity) != fingerprint:
        raise ValueError(f"{name} robot-model manifest fingerprint is not the hash of its recorded inputs")


@dataclass(frozen=True)
class LatestRobotVariant:
    usd_path: Path
    urdf_path: Path
    manifest_path: Path
    metadata: dict
    command_joint_names: tuple[str, ...] = COMMAND_JOINT_NAMES
    auxiliary_joint_names: tuple[str, ...] = AUXILIARY_JOINT_NAMES

    @property
    def camera_attachments(self):
        return self.metadata["camera_attachments"]

    @property
    def gripper_limits(self):
        return self.metadata["gripper_limits_rad"]

    @property
    def grippers(self):
        return self.metadata["grippers"]


def default_model_dir() -> Path:
    repository = Path(__file__).resolve().parents[3]
    candidate = repository.parents[1] / "assets" / "robot"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError("Packaged robot asset directory not found; supply model_dir")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _aliases(source_urdf: Path):
    old = ET.parse(default_urdf_path()).getroot()
    source = ET.parse(source_urdf).getroot()
    old_joints = {j.get("name"): j for j in old.findall("joint")}
    joints, links = {}, {"l_1_dof_gripper_fixed": "Left_Hand", "r_1_dof_gripper_fixed": "Right_Hand",
                          "neck": "Link_Neck_Yaw", "head": "Link_Neck_Pitch"}
    for joint in source.findall("joint"):
        name = joint.get("name")
        if name.startswith("waist_"):
            alias = "Joint_" + "_".join(word.title() for word in name.split("_"))
        elif re.fullmatch(r"[lr]_(?:hip|knee|ankle|shoulder|elbow|wrist)_(?:pitch|roll|yaw)", name):
            side, limb, axis = name.split("_")
            alias = f"Joint_{limb.title()}_{axis.title()}_{'Left' if side == 'l' else 'Right'}"
        elif name in ("neck_yaw", "neck_pitch"):
            alias = "Joint_" + "_".join(word.title() for word in name.split("_"))
        elif name in ("l_1_dof_gripper_fixed", "r_1_dof_gripper_fixed"):
            alias = links[name]
        else:
            alias = name
        joints[name] = alias
        if alias in old_joints:
            links[joint.find("child").get("link")] = old_joints[alias].find("child").get("link")
    for link in source.findall("link"):
        links.setdefault(link.get("name"), link.get("name"))
    if len(set(links.values())) != len(links) or len(set(joints.values())) != len(joints):
        raise ValueError("Latest-model canonical aliases are not one-to-one")
    return joints, links


def _unit_rgb(value, name):
    if (not isinstance(value, list) or len(value) != 3
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not 0. <= v <= 1. for v in value)):
        raise ValueError(f"package.json colors_linear_rgb.{name} must be three linear RGB values in [0, 1]")
    return [float(v) for v in value]


def _package_path(directory: Path, relative) -> Path:
    """A file listed by a package: relative, inside the package directory, existing."""
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"Package files must be listed by relative path: {relative!r}")
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError(f"Package file is missing or outside the package: {relative}")
    return path


def _package_mesh(directory: Path, item: dict, asset_sha256: dict, *, kind: str) -> dict:
    """Load one hash-checked package STL. ``kind``: body (decomposed), collider (one convex hull) or visual."""
    import trimesh  # Pad variants only; the default build never imports it.

    path = _package_path(directory, item["stl"])
    if path.suffix.lower() != ".stl":
        raise ValueError(f"Package mesh must be an STL file: {item['stl']}")
    if _sha(path) != item["sha256"]:
        raise ValueError(f"Package mesh differs from its package.json sha256: {item['stl']}")
    asset_sha256[item["stl"]] = item["sha256"]
    mesh = trimesh.load(str(path), force="mesh", process=True)
    vertices = np.asarray(mesh.vertices, dtype=float)
    if not len(mesh.faces) or not np.isfinite(vertices).all() or np.abs(vertices).max() > .5:
        raise ValueError(f"Package mesh is empty, nonfinite or not in jaw-link metres: {item['stl']}")
    volume = None
    if kind != "visual":
        # Closed: every edge has two or more faces (the white bodies are two closed shells sharing edges).
        _, uses = np.unique(np.sort(np.asarray(mesh.edges), axis=1), axis=0, return_counts=True)
        if not (uses.min() >= 2 and mesh.is_winding_consistent and mesh.volume > 0):
            raise ValueError(f"Package collider must be a closed outward surface: {item['stl']}")
        volume = float(mesh.volume)
    if kind == "collider":
        hull = mesh.convex_hull
        if (not mesh.is_watertight or abs(hull.volume - mesh.volume) > 1e-3 * hull.volume
                or len(hull.vertices) > _PAD_HULL_VERTEX_LIMIT):
            raise ValueError(f"Pad collider must be one convex piece with <= 64 hull vertices: {item['stl']}")
    return {"id": item["id"], "file": item["stl"], "source": path, "role": item["role"],
            "collider": kind != "visual", "vertices": vertices, "faces": np.asarray(mesh.faces, dtype=np.int64),
            "face_normals": np.asarray(mesh.face_normals, dtype=float), "volume_m3": volume}


def _load_gripper_pads(name: str, source_hashes: Mapping[str, str]) -> dict:
    """Read and verify one immutable jaw package (no USD). Raises before any asset is written."""
    spec = _GRIPPER_VARIANT_SPECS[name]
    directory = Path(spec["asset_dir"]).resolve()
    document_path = directory / "package.json"
    if not document_path.is_file():
        raise FileNotFoundError(f"Gripper pad package is missing: {document_path}")
    digest = _sha(document_path)
    if digest != spec["package_json_sha256"]:
        raise ValueError(f"{name}: package.json differs from its pinned sha256; new pad geometry is a new variant")
    document = json.loads(document_path.read_text(encoding="utf-8"))
    required = {"schema", "variant", "status", "units", "frame", "designed_against", "colors_linear_rgb",
                "closure", "jaws", "provenance"}
    if not isinstance(document, dict) or not required <= set(document) or set(document) - required - {"notes"}:
        raise ValueError(f"package.json must contain exactly the {_PAD_PACKAGE_SCHEMA} fields")
    if (document["schema"] != _PAD_PACKAGE_SCHEMA or document["variant"] != name or document["units"] != "meters"
            or document["frame"] != "jaw_link"):
        raise ValueError("package.json schema, variant, units (meters) or frame (jaw_link) mismatch")
    designed = document["designed_against"]
    if (not isinstance(designed, dict)
            or set(designed) != {f"{_JAW_MESH_DIR}/{stem}.stl" for stem in _JAW_LINK_MESHES.values()}
            or any(source_hashes.get(path) != value for path, value in designed.items())):
        raise ValueError("Pad package was designed against different source jaw meshes")
    colors = document["colors_linear_rgb"]
    if not isinstance(colors, dict) or set(colors) != {"pad", "jaw_body"}:
        raise ValueError("package.json colors_linear_rgb needs exactly pad and jaw_body")
    colors = {key: _unit_rgb(colors[key], key) for key in ("pad", "jaw_body")}
    touch = document["closure"].get("empty_close_first_touch_q_rad") if isinstance(document["closure"], dict) else None
    if not isinstance(touch, dict) or set(touch) != set(GRIPPER_JOINT_NAMES) or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or not 0. <= value * math.copysign(1., GRIPPER_OPEN_TARGETS[joint]) <= _PAD_MAX_EMPTY_CLOSE_TOUCH_RAD
            for joint, value in touch.items()):
        raise ValueError("Pads must first touch between the closed target 0 and 0.025 rad toward open on both jaws")
    asset_sha256 = {"package.json": digest}
    if not isinstance(document["provenance"], dict):
        raise ValueError("package.json provenance must be an object")
    for item in document["provenance"].values():  # copied generator inputs/outputs are hash-checked too
        if isinstance(item, dict) and "file" in item:
            if _sha(_package_path(directory, item["file"])) != item.get("sha256"):
                raise ValueError(f"Package provenance file differs from its sha256: {item['file']}")
            asset_sha256[item["file"]] = item["sha256"]
    if not isinstance(document["jaws"], dict) or set(document["jaws"]) != set(_JAW_LINK_MESHES):
        raise ValueError("package.json must define exactly the four canonical jaw links")
    jaws = {}
    for link, stem in _JAW_LINK_MESHES.items():
        entry = document["jaws"][link]
        if (not isinstance(entry, dict) or set(entry) != {"body", "pads"} or not isinstance(entry["pads"], list)
                or not entry["pads"]):
            raise ValueError(f"package.json jaws.{link} needs a body and a nonempty pads list")
        body = entry["body"]
        if (not isinstance(body, dict) or set(body) != {"id", "stl", "sha256", "replaces", "role"}
                or body["id"] != "jaw_body" or body["replaces"] != f"{_JAW_MESH_DIR}/{stem}.stl"):
            raise ValueError(f"package.json jaws.{link}.body must be the jaw_body that replaces {stem}.stl")
        pieces, ids = [], {"jaw_body"}
        for item in entry["pads"]:
            if (not isinstance(item, dict) or set(item) != {"id", "stl", "sha256", "role", "collider"}
                    or not isinstance(item["collider"], bool)):
                raise ValueError(f"package.json jaws.{link}.pads entries need exactly id, stl, sha256, role, collider")
            if not isinstance(item["id"], str) or not _PAD_ID.fullmatch(item["id"]) or item["id"] in ids:
                raise ValueError(f"Invalid or duplicate pad piece id: {item['id']!r}")
            ids.add(item["id"])
            pieces.append(_package_mesh(directory, item, asset_sha256,
                                        kind="collider" if item["collider"] else "visual"))
        jaws[link] = {"body": {**_package_mesh(directory, body, asset_sha256, kind="body"),
                               "replaces": body["replaces"]},
                      "pads": pieces}
    return {"variant": name, "directory": directory, "document": document, "colors": colors,
            "first_touch_q_rad": dict(touch), "asset_sha256": asset_sha256, "jaws": jaws}


def _pad_copy(pads: dict, piece: dict) -> str:
    return f"gripper_pads/{pads['variant']}/{piece['file']}"


def _pad_pieces(pads: dict):
    for entry in pads["jaws"].values():
        yield entry["body"]
        yield from entry["pads"]


def _copy_pad_files(pads: dict, destination: Path) -> None:
    """Byte-identical copies the derived URDF references relative to its own directory (relocatable cache)."""
    for piece in _pad_pieces(pads):
        target = destination / _pad_copy(pads, piece)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(piece["source"], target)


def _referenced_meshes(urdf_path: Path) -> set:
    """Every mesh file a URDF names, resolved as load_robot_model does; FileNotFoundError if one is missing."""
    return {resolve_mesh_path(mesh.get("filename"), urdf_path) for mesh in ET.parse(urdf_path).getroot().iter("mesh")}


def _derived_hashes(usd_path: Path, urdf_path: Path, destination: Path, pads: dict | None) -> dict:
    """Exactly {"usd", "urdf"} for authored_v0; pad variants add their copied jaw and pad meshes."""
    result = {"usd": _sha(usd_path), "urdf": _sha(urdf_path)}
    if pads is not None:
        copies = sorted({_pad_copy(pads, piece) for piece in _pad_pieces(pads)})
        result["gripper_pad_files"] = {relative: _sha(destination / relative) if (destination / relative).is_file()
                                       else None for relative in copies}
    return result


def _adapt_urdf_jaws(root, pads: dict, model_dir: Path) -> None:
    """Swap each jaw link's authored mesh for the white body; append identity-pose pad elements.

    Package meshes are named relative to the derived URDF (its cache copies), so a moved or archived
    cache still loads; the supplied model meshes keep their resolved absolute paths.
    """
    variant = pads["variant"]
    rgba = {key: " ".join(format(v, ".6g") for v in pads["colors"][key]) + " 1" for key in ("pad", "jaw_body")}
    for link_name, entry in pads["jaws"].items():
        link = root.find(f"link[@name='{link_name}']")
        if link is None:
            raise ValueError(f"Derived URDF has no jaw link {link_name}")
        replaced = (model_dir / entry["body"]["replaces"]).resolve()
        for role in ("visual", "collision"):
            elements = link.findall(role)
            mesh = elements[0].find("geometry/mesh") if len(elements) == 1 else None
            if mesh is None or Path(mesh.get("filename")).resolve() != replaced:
                raise ValueError(f"Derived URDF {link_name} must have exactly one authored {role} jaw mesh")
            elements[0].set("name", f"{variant}_jaw_body")
            mesh.set("filename", _pad_copy(pads, entry["body"]))
            mesh.set("scale", "1 1 1")
            if role == "visual":
                for material in elements[0].findall("material"):
                    elements[0].remove(material)
                material = ET.SubElement(elements[0], "material", name=f"{variant}_jaw_body")
                ET.SubElement(material, "color", rgba=rgba["jaw_body"])
        for piece in entry["pads"]:
            for role in ("collision", "visual") if piece["collider"] else ("visual",):
                element = ET.SubElement(link, role, name=f"{variant}_{piece['id']}")
                ET.SubElement(element, "origin", xyz="0 0 0", rpy="0 0 0")
                geometry = ET.SubElement(element, "geometry")
                ET.SubElement(geometry, "mesh", filename=_pad_copy(pads, piece), scale="1 1 1")
                if role == "visual":
                    material = ET.SubElement(element, "material", name=f"{variant}_pad")
                    ET.SubElement(material, "color", rgba=rgba["pad"])


def _apply_collider_offsets(prim, offsets: Mapping) -> None:
    from pxr import Sdf
    prim.AddAppliedSchema("PhysxCollisionAPI")
    prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(offsets["contact"])
    prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(offsets["rest"])


def _apply_jaw_decomposition(prim, decomposition: Mapping | None = None, offsets: Mapping | None = None) -> None:
    """Jaw-body collider settings: explicit contact / zero rest offsets and convex decomposition.

    Defaults are the authored-jaw repair (0.5 mm / 0); pad variants pass their fingerprinted builder settings.
    """
    from pxr import Sdf
    _apply_collider_offsets(prim, {"contact": _JAW_CONTACT_OFFSET_M, "rest": 0.} if offsets is None else offsets)
    prim.AddAppliedSchema("PhysxConvexDecompositionCollisionAPI")
    for name, value in (_JAW_DECOMPOSITION_SETTINGS if decomposition is None else decomposition).items():
        value_type = (Sdf.ValueTypeNames.Bool if isinstance(value, bool) else
                      Sdf.ValueTypeNames.Int if isinstance(value, int) else Sdf.ValueTypeNames.Float)
        prim.CreateAttribute("physxConvexDecompositionCollision:" + name, value_type).Set(value)


def _define_mesh(stage, path, piece):
    """Triangle mesh in metres with per-face normals (the URDF importer's STL convention)."""
    from pxr import Gf, UsdGeom, Vt
    vertices, faces = piece["vertices"], piece["faces"]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in faces.reshape(-1)]))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, vertices.min(0))), Gf.Vec3f(*map(float, vertices.max(0)))]))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.repeat(piece["face_normals"], 3, axis=0).astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    return mesh


def _define_visual_material(stage, path, rgb, roughness):
    """OmniPBR (the authored robot's MDL shading) plus a UsdPreviewSurface with the same colour."""
    from pxr import Gf, Sdf, UsdShade
    material = UsdShade.Material.Define(stage, path)
    omni_pbr = UsdShade.Shader.Define(stage, path.AppendChild("OmniPBR"))
    omni_pbr.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    omni_pbr.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    omni_pbr.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    omni_pbr.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(roughness)
    out = omni_pbr.CreateOutput("out", Sdf.ValueTypeNames.Token)
    for output in (material.CreateSurfaceOutput("mdl"), material.CreateDisplacementOutput("mdl"),
                   material.CreateVolumeOutput("mdl")):
        output.ConnectToSource(out)
    preview = UsdShade.Shader.Define(stage, path.AppendChild("PreviewSurface"))
    preview.CreateIdAttr("UsdPreviewSurface")
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    preview.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.)
    material.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
    return material


def _author_usd_gripper(stage, root_path, pads: dict, material: dict, settings: dict) -> dict:
    """Replace the four jaw meshes by the package's white bodies and add its pads.

    The authored jaw visual instance and jaw collider are deactivated, not deleted:
    inactive prims are not composed, so PhysX and the renderer ignore them, and no
    other prim changes. The pad physics material is bound to pad colliders only.
    Every collider and visual choice comes from ``settings`` (_pad_builder_settings).
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

    name = pads["variant"]
    looks = root_path.AppendChild("Looks")
    if not stage.GetPrimAtPath(looks).IsValid():
        UsdGeom.Scope.Define(stage, looks)
    physics = UsdShade.Material.Define(stage, looks.AppendChild(f"{name}_pad_physics"))
    physics_api = UsdPhysics.MaterialAPI.Apply(physics.GetPrim())
    physics_api.CreateStaticFrictionAttr(material["static_friction"])
    physics_api.CreateDynamicFrictionAttr(material["dynamic_friction"])
    physics_api.CreateRestitutionAttr(material["restitution"])
    physics.GetPrim().AddAppliedSchema("PhysxMaterialAPI")
    physics.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set(
        material["friction_combine_mode"])
    if settings.get("pad_compliant_contact") is not None:
        compliance = settings["pad_compliant_contact"]
        physics.GetPrim().CreateAttribute("physxMaterial:compliantContactStiffness", Sdf.ValueTypeNames.Float).Set(
            compliance["stiffness_n_m"])
        physics.GetPrim().CreateAttribute("physxMaterial:compliantContactDamping", Sdf.ValueTypeNames.Float).Set(
            compliance["damping_n_s_m"])
    looks_by_key = {key: _define_visual_material(stage, looks.AppendChild(f"{name}_{key}_visual"),
                                                 pads["colors"][key], settings["visual_roughness"][key])
                    for key in ("pad", "jaw_body")}
    colliders, visuals, bodies = [], [], []
    sphere_layouts = {}

    def visual_mesh(group, piece, key):
        mesh = _define_mesh(stage, group.AppendChild(piece["id"]), piece)
        mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*pads["colors"][key])]))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(looks_by_key[key])
        return str(mesh.GetPath())

    for link, entry in pads["jaws"].items():
        body = stage.GetPrimAtPath(root_path.AppendChild(link))
        if not body.IsValid() or not body.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"Pad target is not a canonical jaw rigid body: {link}")
        authored_visual = body.GetChild("visuals")
        authored_collider = stage.GetPrimAtPath(body.GetPath().AppendPath(f"collisions/{_JAW_LINK_MESHES[link]}"))
        authored = [p.GetPath() for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
                    if p.HasAPI(UsdPhysics.CollisionAPI)]
        if (not authored_visual.IsValid() or not authored_collider.IsValid() or len(authored) != 1
                or not authored[0].HasPrefix(authored_collider.GetPath())):
            raise ValueError(f"Unexpected authored jaw visual/collider layout: {link}")
        authored_visual.SetActive(False)
        authored_collider.SetActive(False)
        group = UsdGeom.Xform.Define(stage, body.GetPath().AppendChild(name)).GetPath()
        collision_group = UsdGeom.Xform.Define(stage, group.AppendChild("collisions"))
        collision_group.CreatePurposeAttr(UsdGeom.Tokens.guide)  # rendered by the visual meshes, like the importer
        visual_group = UsdGeom.Xform.Define(stage, group.AppendChild("visuals")).GetPath()
        jaw = entry["body"]
        prim = _define_mesh(stage, collision_group.GetPath().AppendChild(jaw["id"]), jaw).GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(settings["jaw_body_approximation"])
        _apply_jaw_decomposition(prim, settings["jaw_body_decomposition"], settings["collider_offsets_m"])
        bodies.append({"link": link, "file": jaw["file"], "replaces": jaw["replaces"],
                       "usd_collider_path": str(prim.GetPath()), "approximation": settings["jaw_body_approximation"],
                       "usd_visual_path": visual_mesh(visual_group, jaw, "jaw_body"),
                       "urdf_mesh": _pad_copy(pads, jaw), "volume_m3": jaw["volume_m3"],
                       "deactivated_authored_prims": [str(authored_visual.GetPath()),
                                                      str(authored_collider.GetPath())]})
        for piece in entry["pads"]:
            if piece["collider"] and settings.get("pad_contact_spheres") is not None \
                    and piece["id"] == _PAD_GRIPPING_PIECE_ID:
                spheres = settings["pad_contact_spheres"]
                face = pad_gripping_face(piece, _PAD_GRIPPING_FACE_NORMAL_SIGN_Y[link])
                centres = pad_contact_sphere_centres(face, spheres["radius_m"], spheres["pitch_m"], spheres["protrusion_m"])
                group_path = collision_group.GetPath().AppendChild(piece["id"] + "_spheres")
                UsdGeom.Xform.Define(stage, group_path)
                share = 0. if spheres["retain_pad_collider"] else piece["volume_m3"] * _PAD_DENSITY_ESTIMATE_KG_M3 / len(centres)
                if spheres["retain_pad_collider"]:  # the convex pad stays as the backstop behind the sphere apexes
                    prim = _define_mesh(stage, collision_group.GetPath().AppendChild(piece["id"]), piece).GetPrim()
                    UsdPhysics.CollisionAPI.Apply(prim)
                    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(settings["pad_approximation"])
                    _apply_collider_offsets(prim, settings["collider_offsets_m"])
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(physics, settings["pad_material_binding_strength"], "physics")
                    colliders.append({"link": link, "id": piece["id"], "role": piece["role"] + " (backstop behind the contact spheres)",
                                      "usd_path": str(prim.GetPath()), "urdf_mesh": _pad_copy(pads, piece),
                                      "approximation": settings["pad_approximation"], "volume_m3": piece["volume_m3"],
                                      "estimated_mass_kg": piece["volume_m3"] * _PAD_DENSITY_ESTIMATE_KG_M3})
                for index, centre in enumerate(centres):
                    sphere = UsdGeom.Sphere.Define(stage, group_path.AppendChild(f"sphere_{index:02d}"))
                    sphere.CreateRadiusAttr(spheres["radius_m"])
                    sphere.AddTranslateOp().Set(Gf.Vec3d(*map(float, centre)))
                    extent = spheres["radius_m"]
                    sphere.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(-extent, -extent, -extent), Gf.Vec3f(extent, extent, extent)]))
                    prim = sphere.GetPrim()
                    UsdPhysics.CollisionAPI.Apply(prim)
                    _apply_collider_offsets(prim, settings["collider_offsets_m"])
                    prim.CreateAttribute("physxCollision:minTorsionalPatchRadius", Sdf.ValueTypeNames.Float).Set(
                        spheres["min_torsional_patch_radius_m"])
                    prim.CreateAttribute("physxCollision:torsionalPatchRadius", Sdf.ValueTypeNames.Float).Set(
                        spheres["torsional_patch_radius_m"])
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(physics, settings["pad_material_binding_strength"], "physics")
                    colliders.append({"link": link, "id": f"{piece['id']}_sphere_{index:02d}",
                                      "role": "inner gripping pad contact sphere (replaces the convex pad collider)",
                                      "usd_path": str(prim.GetPath()), "urdf_mesh": _pad_copy(pads, piece),
                                      "approximation": "sphere", "volume_m3": 4. / 3. * math.pi * spheres["radius_m"] ** 3,
                                      "estimated_mass_kg": share,
                                      "centre_jaw_frame_m": [float(x) for x in centre]})
                sphere_layouts[link] = {"count": len(centres), "radius_m": spheres["radius_m"], "pitch_m": spheres["pitch_m"],
                                        "protrusion_m": spheres["protrusion_m"], "retain_pad_collider": spheres["retain_pad_collider"],
                                        "face_normal_jaw_frame": [float(x) for x in face["normal"]],
                                        "face_origin_jaw_frame_m": [float(x) for x in face["origin"]],
                                        "face_u_range_m": list(face["u_range"]), "face_v_range_m": list(face["v_range"]),
                                        "face_area_m2": face["area_m2"], "group_usd_path": str(group_path),
                                        "replaced_collider_mesh": _pad_copy(pads, piece)}
            elif piece["collider"]:
                prim = _define_mesh(stage, collision_group.GetPath().AppendChild(piece["id"]), piece).GetPrim()
                UsdPhysics.CollisionAPI.Apply(prim)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(settings["pad_approximation"])
                _apply_collider_offsets(prim, settings["collider_offsets_m"])
                if settings.get("pad_contact_patch") is not None:
                    patch = settings["pad_contact_patch"]
                    prim.CreateAttribute("physxCollision:minTorsionalPatchRadius", Sdf.ValueTypeNames.Float).Set(
                        patch["min_torsional_patch_radius_m"])
                    prim.CreateAttribute("physxCollision:torsionalPatchRadius", Sdf.ValueTypeNames.Float).Set(
                        patch["torsional_patch_radius_m"])
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(physics, settings["pad_material_binding_strength"], "physics")
                colliders.append({"link": link, "id": piece["id"], "role": piece["role"], "usd_path": str(prim.GetPath()),
                                  "urdf_mesh": _pad_copy(pads, piece), "approximation": settings["pad_approximation"],
                                  "volume_m3": piece["volume_m3"],
                                  "estimated_mass_kg": piece["volume_m3"] * _PAD_DENSITY_ESTIMATE_KG_M3})
            visuals.append({"link": link, "id": piece["id"], "usd_path": visual_mesh(visual_group, piece, "pad"),
                            "urdf_mesh": _pad_copy(pads, piece)})
    return {"physics_material_path": str(physics.GetPath()),
            "visual_material_paths": {key: str(value.GetPath()) for key, value in looks_by_key.items()},
            "colliders": colliders, "visuals": visuals, "jaw_bodies": bodies, "sphere_layouts": sphere_layouts}


def build_latest_robot_variant(output_dir: str | Path, *, model_dir: str | Path | None = None,
                               gripper_stiffness: float | None = None,
                               gripper_variant: str = DEFAULT_GRIPPER_VARIANT,
                               pad_material_override: Mapping | None = None,
                               pad_contact_patch: Mapping | None = None,
                               pad_compliant_contact: Mapping | None = None,
                               pad_contact_spheres: Mapping | None = None) -> LatestRobotVariant:
    """Build a hash-addressed naming adapter without touching supplied assets.

    ``gripper_stiffness=None`` uses the variant's gain (60 N*m/rad for authored_v0);
    a pad variant records any other gain as an ``override`` stiffness source.
    ``pad_material_override`` is a development-only sensitivity input for pad
    variants; the override itself (even one equal to the spec) changes the
    fingerprint and is recorded in the manifest.
    """
    spec = _GRIPPER_VARIANT_SPECS[validate_gripper_variant(gripper_variant)]
    actuator = spec["runtime_implicit_actuator"]
    if gripper_stiffness is None:
        gripper_stiffness = actuator["stiffness"]
    if isinstance(gripper_stiffness, bool) or not np.isfinite(gripper_stiffness) or gripper_stiffness <= 0:
        raise ValueError("gripper_stiffness must be finite and positive")
    gripper_actuator = {"stiffness": float(gripper_stiffness), "damping": actuator["damping"],
                        "effort_limit_sim": actuator["effort_limit_sim"],
                        "velocity_limit_sim": actuator["velocity_limit_sim"]}
    pad_material = _pad_material(gripper_variant, pad_material_override)
    contact_patch = validate_pad_contact_patch(gripper_variant, pad_contact_patch)
    compliant_contact = validate_pad_compliant_contact(gripper_variant, pad_compliant_contact)
    contact_spheres = validate_pad_contact_spheres(gripper_variant, pad_contact_spheres)
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    model_dir = (default_model_dir() if model_dir is None else Path(model_dir)).resolve()
    source_urdf = model_dir / "urdf/igris_c_v2_torso_1dof.urdf"
    source_usd = model_dir / "usd/1d_gripper_cam/igrisc_1dgripper.usd"
    if not source_usd.is_file() or not source_urdf.is_file():
        raise FileNotFoundError("Latest model requires the authored 1d_gripper_cam USD and torso_1dof URDF")
    tree = ET.parse(source_urdf)
    source_mass = sum(float(element.get("value")) for element in tree.getroot().findall("link/inertial/mass"))
    empty_frames = {link.get("name") for link in tree.getroot().findall("link")
                    if all(link.find(tag) is None for tag in ("inertial", "visual", "collision"))}
    if empty_frames != _EMPTY_FIXED_FRAMES:
        raise ValueError(f"Unexpected inertia-free source frames; manual review required: {sorted(empty_frames)}")
    for joint in tree.getroot().findall("joint"):
        endpoints = {joint.find(tag).get("link") for tag in ("parent", "child")}
        if endpoints & empty_frames and joint.get("type") != "fixed":
            raise ValueError("An inertia-free placeholder frame is connected by an active joint")
    source_joint_aliases, source_link_aliases = _aliases(source_urdf)
    source_files = {source_urdf, *[p for p in (model_dir / "usd/1d_gripper_cam").rglob("*") if p.is_file()]}
    for mesh in tree.getroot().iter("mesh"):
        source_files.add(resolve_mesh_path(mesh.get("filename"), source_urdf))
    hashes = {str(path.relative_to(model_dir)): _sha(path) for path in sorted(source_files)}
    pads = None if pad_material is None else _load_gripper_pads(gripper_variant, hashes)
    gripping_face_check = None
    if pads is not None and contact_spheres is not None:
        # The sphere grids sit on the faces that meet the opposing jaw; prove it with the URDF hinge at first touch.
        gripping_face_check = _check_gripping_faces(pads, tree)
    identity = {"version": _ADAPTER_VERSION, "sources": hashes,
                "joint_aliases": source_joint_aliases, "link_aliases": source_link_aliases,
                "runtime_gripper_actuator": gripper_actuator}
    if pads is not None:  # authored_v0 keeps the exact pre-variant identity (and fingerprint).
        builder_settings = _pad_builder_settings()
        dev_override = (None if pad_material_override is None
                        else {key: pad_material[key] for key in sorted(pad_material_override)})
        identity["gripper_variant"] = {"name": gripper_variant, "builder": _PAD_BUILDER_VERSION,
                                       "builder_settings": builder_settings, "asset_sha256": pads["asset_sha256"],
                                       "pad_material": pad_material, "pad_material_override": dev_override}
        if contact_patch is not None:  # a declared patch is its own jaw model; absence keeps the old fingerprint
            identity["gripper_variant"]["pad_contact_patch"] = contact_patch
        if compliant_contact is not None:
            identity["gripper_variant"]["pad_compliant_contact"] = compliant_contact
        if contact_spheres is not None:
            identity["gripper_variant"]["pad_contact_spheres"] = contact_spheres
    fingerprint = _fingerprint(identity)
    destination = Path(output_dir).resolve() / fingerprint[:16]
    destination.mkdir(parents=True, exist_ok=True)
    usd_path = destination / "igris_c_260910_canonical.usda"
    urdf_path = destination / "igris_c_260910_canonical.urdf"
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        metadata = json.loads(manifest_path.read_text())
        # The cached URDF must name exactly the files this identity covers: the supplied meshes in this
        # model_dir and, for a pad variant, the hashed copies inside this cache directory.
        expected = {resolve_mesh_path(mesh.get("filename"), source_urdf) for link in tree.getroot().findall("link")
                    if pads is None or source_link_aliases[link.get("name")] not in pads["jaws"]
                    for mesh in link.iter("mesh")}
        if pads is not None:
            expected |= {destination / _pad_copy(pads, piece) for piece in _pad_pieces(pads)}
        try:
            referenced = _referenced_meshes(urdf_path)
        except (OSError, ValueError, ET.ParseError):
            referenced = None
        if (metadata.get("fingerprint") == fingerprint and referenced == expected
                and metadata.get("derived_hashes") == _derived_hashes(usd_path, urdf_path, destination, pads)):
            return LatestRobotVariant(usd_path, urdf_path, manifest_path, metadata)
        raise ValueError("Cached latest robot assets differ from their manifest")

    # URDF adaptation changes only names and resolves the original mesh paths.
    root = tree.getroot()
    root.set("name", "igris_c_260910_canonical")
    for link in root.findall("link"):
        original_name = link.get("name")
        link.set("name", source_link_aliases[original_name])
        if original_name in empty_frames:
            # PhysX imports empty rigid frames with 1 kg default mass each.
            # These five URDF frames have no physical body/inertia/geometry.
            # Positive epsilon mass avoids that importer fallback while keeping
            # their named actor/camera frames and fixed-joint relationships.
            inertia = ET.SubElement(link, "inertial")
            ET.SubElement(inertia, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(inertia, "mass", value=str(_FRAME_MASS_KG))
            ET.SubElement(inertia, "inertia", ixx=str(_FRAME_INERTIA_KG_M2), iyy=str(_FRAME_INERTIA_KG_M2),
                          izz=str(_FRAME_INERTIA_KG_M2), ixy="0", ixz="0", iyz="0")
    for joint in root.findall("joint"):
        joint.set("name", source_joint_aliases[joint.get("name")])
        for tag in ("parent", "child"):
            element = joint.find(tag)
            element.set("link", source_link_aliases[element.get("link")])
    for mesh in root.iter("mesh"):
        mesh.set("filename", str(resolve_mesh_path(mesh.get("filename"), source_urdf)))
    if pads is not None:
        _copy_pad_files(pads, destination)
        _adapt_urdf_jaws(root, pads, model_dir)
    ET.indent(tree, space="  ")
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)

    # Flatten before namespace edits: source references and their properties
    # must remain intact, and relative asset paths are resolved by USD flatten.
    source_stage = Usd.Stage.Open(str(source_usd))
    articulation_roots = [prim for prim in source_stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(articulation_roots) != 1:
        raise ValueError("The supplied robot must have exactly one authored articulation root")
    self_collision_setting = articulation_roots[0].GetAttribute("physxArticulation:enabledSelfCollisions").Get()
    if not isinstance(self_collision_setting, bool):
        raise ValueError("The supplied robot must explicitly define its self-collision setting")
    root_path = source_stage.GetDefaultPrim().GetPath()
    source_solver_iterations = {
        name: articulation_roots[0].GetAttribute(f"physxArticulation:solver{name.title()}IterationCount").Get()
        for name in ("position", "velocity")}
    layer = source_stage.Flatten()
    changes = {root_path.AppendChild(old): root_path.AppendChild(new)
               for old, new in source_link_aliases.items() if old != new}
    changes.update({root_path.AppendPath("joints/" + old): root_path.AppendPath("joints/" + new)
                    for old, new in source_joint_aliases.items() if old != new})
    edits = Sdf.BatchNamespaceEdit()
    for old, new in changes.items():
        if layer.GetPrimAtPath(old) is None:
            raise ValueError(f"Authored USD missing mapped URDF prim: {old}")
        edits.Add(old, new)
    if not layer.Apply(edits):
        raise RuntimeError("Could not apply latest-model canonical USD names")
    stage = Usd.Stage.Open(layer)
    for name in empty_frames:
        prim = stage.GetPrimAtPath(root_path.AppendChild(source_link_aliases[name]))
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI) or any(
                child.HasAPI(UsdPhysics.CollisionAPI) or child.IsA(UsdGeom.Mesh)
                for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())):
            raise ValueError(f"Placeholder frame unexpectedly contains physical geometry: {name}")
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(_FRAME_MASS_KG)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(_FRAME_INERTIA_KG_M2))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(1, Gf.Vec3f(0, 0, 0)))

    # Whole-jaw convex hulls bridge the authored gripping recess. At the
    # 125 mm pad depth they add 9.59 mm of nonexistent fixed-jaw material.
    # Decompose the same original four meshes; keep every physical vertex,
    # transform, mass, joint and material. This repair is derived-asset only.
    jaw_repairs = []
    for source_body in ("l_1_dof_gripper_fixed", "l_1_dof_gripper_moving",
                        "r_1_dof_gripper_fixed", "r_1_dof_gripper_moving"):
        body = stage.GetPrimAtPath(root_path.AppendChild(source_link_aliases[source_body]))
        # A local override cannot be authored inside an instance proxy.
        instances = [p for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
                     if p.IsInstance() and p.GetName() == "collisions"]
        for prim in instances:
            prim.SetInstanceable(False)
        colliders = [p for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
                     if p.HasAPI(UsdPhysics.CollisionAPI)]
        if len(colliders) != 1:
            raise ValueError(f"Expected one original collision mesh for actual jaw: {source_body}")
        prim = colliders[0]
        # The importer applies CollisionAPI/MeshCollisionAPI to an Xform
        # containing its one Mesh, rather than to the Mesh prim itself.
        if len([p for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Mesh)]) != 1:
            raise ValueError(f"Expected one mesh below the original jaw collider: {source_body}")
        mesh_collision = UsdPhysics.MeshCollisionAPI(prim)
        original_approximation = mesh_collision.GetApproximationAttr().Get()
        if original_approximation != "convexHull":
            raise ValueError(f"Unexpected source jaw collision approximation: {source_body}: {original_approximation}")
        mesh_collision.CreateApproximationAttr("convexDecomposition")
        source_offsets = {name: prim.GetAttribute(f"physxCollision:{name}").Get()
                          if prim.GetAttribute(f"physxCollision:{name}").HasAuthoredValueOpinion() else None
                          for name in ("contactOffset", "restOffset")}
        _apply_jaw_decomposition(prim)
        jaw_repairs.append({"source_body_name": source_body,
                            "derived_collider_path": str(prim.GetPath()),
                            "source_approximation": original_approximation,
                            "source_authored_offsets_m": source_offsets,
                            "derived_approximation": "convexDecomposition"})

    for articulation in stage.Traverse():
        if articulation.HasAPI(UsdPhysics.ArticulationRootAPI):
            for name, value in _CONTACT_SOLVER_ITERATIONS.items():
                articulation.CreateAttribute(f"physxArticulation:solver{name.title()}IterationCount",
                                             Sdf.ValueTypeNames.Int).Set(value)

    def map_path(path):
        for old, new in changes.items():
            if path.HasPrefix(old):
                return path.ReplacePrefix(old, new)
        return path

    # Sdf namespace edits do not retarget physics/material relationships.
    for prim in stage.Traverse():
        for rel in prim.GetRelationships():
            original = rel.GetTargets()
            mapped = [map_path(path) for path in original]
            if mapped != original:
                rel.SetTargets(mapped)
        for attr in prim.GetAttributes():
            original = attr.GetConnections()
            mapped = [map_path(path) for path in original]
            if mapped != original:
                attr.SetConnections(mapped)
    # After retargeting: jaw bodies and pads are authored directly at canonical paths.
    pad_usd = None if pads is None else _author_usd_gripper(
        stage, root_path, pads, pad_material, {**builder_settings, "pad_contact_patch": contact_patch,
                                               "pad_compliant_contact": compliant_contact,
                                               "pad_contact_spheres": contact_spheres})
    camera_attachments = {}
    cache = UsdGeom.XformCache()
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            parent = prim.GetParent()
            relative = np.asarray(cache.GetLocalToWorldTransform(prim)).T
            # Camera mounts have actual measured rigid actor frames in this USD.
            relative = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(parent)).T) @ relative
            camera_attachments[prim.GetName()] = {
                "body_name": parent.GetName(), "relative_pose_wxyz": matrix_to_pose(relative),
                "relative_prim_path": str(prim.GetPath().MakeRelativePath(root_path)),
                "optics_units": "usd_tenths_of_stage_unit",
                "meters_per_stage_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
                "focal_length_usd_tenths": float(UsdGeom.Camera(prim).GetFocalLengthAttr().Get()),
                "horizontal_aperture_usd_tenths": float(UsdGeom.Camera(prim).GetHorizontalApertureAttr().Get()),
                "vertical_aperture_usd_tenths": float(UsdGeom.Camera(prim).GetVerticalApertureAttr().Get()),
                "clipping_range_m": [float(value) * UsdGeom.GetStageMetersPerUnit(stage)
                                     for value in UsdGeom.Camera(prim).GetClippingRangeAttr().Get()],
                "pose_convention": "usd_camera_minus_z_forward_plus_y_up",
            }
    stage.GetRootLayer().Export(str(usd_path))
    model = load_robot_model(urdf_path)
    active = [name for name, joint in model.joints.items() if joint.kind != "fixed"]
    usd_active = [p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    if set(active) != set(usd_active) or len(active) != 33:
        raise ValueError("Latest-model USD/URDF must have the same 33 revolute joints")
    jaw_collision_repair = {
        "colliders": jaw_repairs, "decomposition_settings": _JAW_DECOMPOSITION_SETTINGS,
        "reason": "Single convex hull fills the real finger recess and prevents mesh-consistent pinching",
        "original_vertices_and_topology_preserved": True,
        "materials_masses_joints_preserved": True,
        "cooked_contact_surfaces_require_physics_validation": True,
        "contact_offset_m": _JAW_CONTACT_OFFSET_M, "rest_offset_m": 0.,
        "offset_basis": "Explicit 0.5 mm contact generation margin matches authored JIG/CRS thin-part colliders; does not change the zero-rest physical surface",
        "source_automatic_contact_offset_measured": False,
    }
    metadata = {
        "schema": _ADAPTER_VERSION, "fingerprint": fingerprint,
        "source_model_dir": str(model_dir), "source_hashes": hashes,
        "source_joint_to_canonical": source_joint_aliases, "source_link_to_canonical": source_link_aliases,
        "active_joint_names": active, "active_joint_count": len(active),
        "command_joint_names": list(COMMAND_JOINT_NAMES), "auxiliary_joint_names": list(AUXILIARY_JOINT_NAMES),
        "gripper_type": "authored_fixed_jaw_and_one_revolute_jaw_per_hand",
        "gripper_limits_rad": {name: list(model.joint_limits[name]) for name in GRIPPER_JOINT_NAMES},
        "gripper_open_targets_rad": GRIPPER_OPEN_TARGETS, "gripper_closed_targets_rad": GRIPPER_CLOSED_TARGETS,
        "grippers": {arm: {"joint_name": name, "open_position_rad": GRIPPER_OPEN_TARGETS[name],
                            "closed_position_rad": 0., "speed_rad_s": .7}
                     for arm, name in zip(("left", "right"), GRIPPER_JOINT_NAMES)},
        "gripper_actuator_profile": {
            "runtime_implicit_actuator": gripper_actuator,
            "units": {"stiffness": "N*m/rad", "damping": "N*m*s/rad",
                      "effort_limit_sim": "N*m", "velocity_limit_sim": "rad/s"},
            "prior_provisional_stiffness": 25.,
            "qualification": "provisional_simulation_controller_gains_not_manufacturer_motor_ratings",
            "reason": "Use the existing capped 5 N*m actuator effort during loaded pinching; prior gain delivered only 2.25 N*m at 0.09 rad error",
            "motor_effort_limit_increased": False,
            "geometry_friction_and_target_slew_changed": False,
        },
        "camera_attachments": camera_attachments,
        "source_enabled_self_collisions": self_collision_setting,
        "physics_self_collision_enabled": self_collision_setting,
        "jaw_collision_repair": jaw_collision_repair,
        "contact_solver_profile": {
            "source_asset_iterations": source_solver_iterations,
            "derived_and_runtime_iterations": _CONTACT_SOLVER_ITERATIONS,
            "prior_runtime_iterations": {"position": 8, "velocity": 4},
            "reason": "Improve finger/payload constraint convergence; measured contact had high-frequency jaw-velocity chatter",
            "physical_geometry_friction_force_limits_changed": False,
        },
        "empty_fixed_frame_import_repair": {
            "frame_names": sorted(empty_frames), "mass_per_frame_kg": _FRAME_MASS_KG,
            "diagonal_inertia_per_frame_kg_m2": _FRAME_INERTIA_KG_M2,
            "reason": "Source URDF nonphysical frames otherwise acquire 1 kg each from PhysX default mass",
            "eligibility": "no source URDF visual, collision, or inertial; only fixed joints; no USD physical geometry",
            "source_physical_body_mass_kg": source_mass,
            "derived_total_mass_kg": source_mass + len(empty_frames) * _FRAME_MASS_KG,
            "physical_body_properties_modified": False,
        },
        "qualification": "naming_and_geometry_adaptation_only_policy_dynamics_not_yet_validated",
        "derived_hashes": _derived_hashes(usd_path, urdf_path, destination, pads),
    }
    if pads is not None:  # authored_v0 manifests keep exactly the pre-variant keys and values.
        basis = spec["actuator_basis"]
        commanded = round(gripper_actuator["stiffness"] * basis["design_contact_q_rad"], 9)
        low, high = basis["held_command_window_Nm"]
        metadata.update({
            "gripper_variant": gripper_variant,
            "gripper_type": "white_jaw_bodies_with_rubber_pads_one_revolute_jaw_per_hand",
            "gripper_actuator_profile": {
                "runtime_implicit_actuator": gripper_actuator, "units": dict(_ACTUATOR_UNITS),
                "qualification": "effort_cap_is_the_user_supplied_manufacturer_rating; gains_are_simulation_choices",
                "effort_source": basis["effort_source"],
                # Another gain (API only) is a sensitivity input, not the Isaac-validated spec gain.
                "stiffness_source": (basis["stiffness_source"] if gripper_actuator["stiffness"] == actuator["stiffness"]
                                     else "override"),
                "pinch_command_check": {
                    "design_contact_q_rad": basis["design_contact_q_rad"],
                    "design_contact_basis": basis["design_contact_basis"],
                    "commanded_torque_at_design_contact_Nm": commanded,
                    "held_command_window_Nm": [low, high],
                    "commanded_torque_within_held_window": low <= commanded <= high},
                "motor_effort_limit_increased": False,
                "geometry_friction_and_target_slew_changed": True,
                "targets_and_joint_limits_changed": False,
            },
            "jaw_collision_repair": {**jaw_collision_repair, "superseded_by_gripper_variant": gripper_variant,
                                     "repaired_authored_colliders_active": False},
            "gripper_pads": {
                "builder": _PAD_BUILDER_VERSION, "builder_settings": builder_settings,
                "asset_dir": str(pads["directory"]),
                "asset_sha256": pads["asset_sha256"], "status": pads["document"]["status"],
                "provenance": pads["document"]["provenance"],
                "closure": {
                    "empty_close_first_touch_q_rad": pads["first_touch_q_rad"],
                    "basis": pads["document"]["closure"].get("basis"),
                    "authored_limits_and_closed_target_retained": True,
                    "articulation_self_collision_enabled": self_collision_setting,
                    "empty_close_behaviour": ("closing on nothing reaches the closed target 0; the pads pass their "
                                              "first contact by the first-touch angle without a pad-pad force"),
                    "max_first_touch_q_rad_accepted": _PAD_MAX_EMPTY_CLOSE_TOUCH_RAD},
                "colors_linear_rgb": pads["colors"],
                "contact_patch": contact_patch,
                "compliant_contact": compliant_contact,
                "contact_spheres": contact_spheres,
                "contact_sphere_layouts": pad_usd["sphere_layouts"],
                "contact_sphere_gripping_face_check": gripping_face_check,
                "contact_spheres_basis": (None if contact_spheres is None else
                                          "profile gripper_pad_contact_spheres: the inner gripping pad colliders are a "
                                          "grid of sphere colliders with apexes on the pad face plane, so each pad-ring "
                                          "contact is a single-point PhysX patch with its own friction anchor and "
                                          "torsional friction (filled rubber pad approximation); material, offsets, "
                                          "visuals and the URDF envelope unchanged"),
                "compliant_contact_basis": (None if compliant_contact is None else
                                            "profile gripper_pad_compliant_contact: PhysX compliant contact on the pad "
                                            "physics material models the filled rubber pad; friction/offsets unchanged"),
                "contact_patch_basis": (None if contact_patch is None else
                                        "profile gripper_pad_contact_patch: PhysX torsional patch radius on pad "
                                        "colliders models the filled rubber pad contact patch; no friction/offset change"),
                "physics_material": {**pad_material, "usd_path": pad_usd["physics_material_path"],
                                     "binding": "material:binding:physics on pad colliders only",
                                     "source": spec["pad_material_source"],
                                     "dev_override": dev_override},
                "visual_material_paths": pad_usd["visual_material_paths"],
                "visual_roughness": builder_settings["visual_roughness"],
                "jaw_bodies": pad_usd["jaw_bodies"],
                "jaw_body_decomposition_settings": builder_settings["jaw_body_decomposition"],
                "jaw_body_decomposition_basis": _JAW_BODY_DECOMPOSITION_BASIS,
                "colliders": pad_usd["colliders"], "visuals": pad_usd["visuals"],
                "collider_offsets_m": builder_settings["collider_offsets_m"],
                "mass_policy": "explicit_authored_jaw_link_mass_com_inertia_retained; jaw_bodies_and_pads_add_no_mass",
                "estimated_pad_mass_kg_not_applied": {
                    link: sum(item["estimated_mass_kg"] for item in pad_usd["colliders"] if item["link"] == link)
                    for link in _JAW_LINK_MESHES},
                "estimated_density_kg_m3": _PAD_DENSITY_ESTIMATE_KG_M3,
                "authored_jaw_meshes_active": False,
                "joint_limits_and_gripper_targets_changed": False,
            },
            "qualification": "naming_geometry_and_gripper_pad_adaptation_policy_dynamics_not_yet_validated",
        })
    manifest_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return LatestRobotVariant(usd_path, urdf_path, manifest_path, metadata)


def configure_latest_robot(env_cfg, variant: LatestRobotVariant) -> None:
    """Select the derived asset before gym.make; keep existing lower-body PD gains."""
    from isaaclab.actuators import ImplicitActuatorCfg

    cfg = env_cfg.scene.robot
    positions = {pattern: value for pattern, value in cfg.init_state.joint_pos.items()
                 if not any(re.fullmatch(pattern, name) for name in WRIST_HAND_JOINT_NAMES[6:])}
    positions.update(GRIPPER_OPEN_TARGETS)
    positions.update(dict.fromkeys(NECK_JOINT_NAMES, 0.0))
    actuators = {name: actuator for name, actuator in cfg.actuators.items()
                 if name not in ("active_hand", "distal_hand")}
    # Bounded provisional command gains, not claimed manufacturer motor ratings.
    actuators["two_finger_gripper"] = ImplicitActuatorCfg(
        joint_names_expr=list(GRIPPER_JOINT_NAMES),
        **variant.metadata["gripper_actuator_profile"]["runtime_implicit_actuator"])
    actuators["neck_hold"] = ImplicitActuatorCfg(
        joint_names_expr=list(NECK_JOINT_NAMES), stiffness=60., damping=4.,
        effort_limit_sim=10., velocity_limit_sim=1.)
    solver_iterations = variant.metadata["contact_solver_profile"]["derived_and_runtime_iterations"]
    articulation_settings = {
        "enabled_self_collisions": variant.metadata["source_enabled_self_collisions"],
        "solver_position_iteration_count": solver_iterations["position"],
        "solver_velocity_iteration_count": solver_iterations["velocity"],
    }
    if cfg.spawn.articulation_props is None:
        from isaaclab.sim import ArticulationRootPropertiesCfg
        articulation_props = ArticulationRootPropertiesCfg(**articulation_settings)
    else:
        articulation_props = cfg.spawn.articulation_props.replace(**articulation_settings)
    env_cfg.scene.robot = cfg.replace(spawn=cfg.spawn.replace(usd_path=str(variant.usd_path),
        articulation_props=articulation_props),
        init_state=cfg.init_state.replace(joint_pos=positions), actuators=actuators)
