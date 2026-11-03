"""Opt-in rotational symmetry of the carried object about one body axis.

The CRS is a 254 mm ring: in simulation the rigid pads touch its rim at one or
two points and torsional friction about the pinch axis is ~0, so whenever the
hand reorients, gravity turns the ring about its own normal inside the closed
jaws (run_45 / run_55: 0.2 rad of "capture rotation" while 36-46 N jaw forces
stayed). That spin is physically irrelevant for carrying, inserting and placing
a ring. What matters is the rim staying between the pads (radial/axial slip),
the ring axis direction relative to the hand (tilt) and, for placement, the
centre and the axis direction relative to the slot.

A task declares ``object_symmetry = {"axis_object": [0, 0, 1]}`` (a unit vector
in the object frame), optionally with ``"pivot_radius_m": 0.127`` (the radius of
the circle on which the pads pinch the object, the CRS rim). Absent the
declaration every caller keeps its original full-rotation comparison, byte for
byte. With it, the helpers below remove the best-fitting rotation about the axis
before measuring a rotation error and report that spin separately.

Two spin modes exist for a pinched ring. The rim may slide through the pads
(rotation about the ring centre: the centre stays put in the hand), or the ring
may turn about the pinch itself (rotation about the axis direction THROUGH the
contact: the pads keep the same rim material point while the centre swings by
2 sin(spin/2) x radius, 25 mm at 0.2 rad). The recorded runs show the second mode
(the fixed point of the hand-relative motion lies on the rim). Without a pivot
radius the capture position error is the ring-centre motion in the hand frame,
which only cancels the first mode; with ``pivot_radius_m`` it is measured at the
pinch contact (``symmetric_contact_errors``), which cancels both and still
reports a radial or axial slip of the rim at the pads in full.

Conventions: poses are xyz + wxyz lists (snapshot.pose_to_matrix /
matrix_to_pose), matrices are 4x4 column-vector transforms, rotation vectors
follow task_control.rotation_error. ``spin`` is always the angle by which the
MEASURED orientation is turned about the axis relative to the REFERENCE:
measured.R ~ reference.R @ rotation_about(axis, spin), up to the residual tilt.
Nothing here writes physical state or proves a grasp; it only changes how an
orientation difference is measured once the profile declares the symmetry.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real

import numpy as np

AXIS_KEY = "axis_object"
PIVOT_KEY = "pivot_radius_m"
UNIT_TOLERANCE = 1e-6
# A reference TCP closer than this to the symmetry axis has no pinch azimuth (a rim pinch never has).
MIN_PINCH_AZIMUTH_M = 1e-6


def unit_axis(value, name="axis"):
    """Return ``value`` as a float array after checking it is a finite unit 3-vector."""
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{name} must be three finite numbers")
    try:
        axis = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be three finite numbers") from error
    if axis.shape != (3,) or not np.isfinite(axis).all():
        raise ValueError(f"{name} must be three finite numbers")
    if abs(float(np.linalg.norm(axis)) - 1.) > UNIT_TOLERANCE:
        raise ValueError(f"{name} must be a unit vector (length within 1e-6 of one)")
    return axis


def pivot_radius(value, name="pivot_radius_m"):
    """Return ``value`` as a float after checking it is a positive finite number (never a bool)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    radius = float(value)
    if not math.isfinite(radius) or radius <= 0.:
        raise ValueError(f"{name} must be a positive finite number")
    return radius


def validate_object_symmetry(settings):
    """Validate ``{"axis_object": [x, y, z]}`` (a finite unit vector) with the optional ``pivot_radius_m``."""
    if not isinstance(settings, Mapping) or AXIS_KEY not in settings or set(settings) - {AXIS_KEY, PIVOT_KEY}:
        raise ValueError("object_symmetry requires axis_object and accepts only the optional pivot_radius_m")
    value = settings[AXIS_KEY]
    if (isinstance(value, (str, bytes, Mapping)) or not hasattr(value, "__len__") or len(value) != 3
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in value)):
        raise ValueError("object_symmetry.axis_object must be three finite numbers")
    axis = unit_axis(value, "object_symmetry.axis_object")
    validated = {AXIS_KEY: [float(v) for v in axis]}
    if PIVOT_KEY in settings:
        validated[PIVOT_KEY] = pivot_radius(settings[PIVOT_KEY], "object_symmetry.pivot_radius_m")
    return validated


def symmetry_axis(task):
    """The declared unit axis of a task mapping as an array, or None when undeclared."""
    settings = task.get("object_symmetry") if isinstance(task, Mapping) else None
    if settings is None:
        return None
    return np.asarray(validate_object_symmetry(settings)[AXIS_KEY], dtype=float)


def symmetry_pivot_radius(task):
    """The declared pinch-circle radius of a task mapping, or None when undeclared."""
    settings = task.get("object_symmetry") if isinstance(task, Mapping) else None
    if settings is None:
        return None
    return validate_object_symmetry(settings).get(PIVOT_KEY)


def rotation_about(axis, theta):
    """Rodrigues rotation matrix (3x3) by ``theta`` radians about the unit ``axis``."""
    x, y, z = unit_axis(axis)
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(theta) * skew + (1. - math.cos(theta)) * (skew @ skew)


def rotation_about_point(axis, theta, point):
    """4x4 rotation by ``theta`` about the line through ``point`` parallel to the unit ``axis``."""
    point = np.asarray(point, dtype=float)
    rotation = rotation_about(axis, theta)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = point - rotation @ point
    return transform


def axis_basis(axis):
    """Right-handed orthonormal basis ``B = [u, v, axis]`` (columns) whose third column is the axis."""
    axis = unit_axis(axis)
    seed = np.zeros(3)
    seed[int(np.argmin(np.abs(axis)))] = 1.
    u = np.cross(axis, seed)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return np.column_stack((u, v, axis))


def residual_angle(first, second):
    """Angle of the rotation between two rotation matrices (as nepes_tasks._errors measures it)."""
    relative = np.asarray(first, dtype=float)[:3, :3].T @ np.asarray(second, dtype=float)[:3, :3]
    return float(math.acos(float(np.clip((np.trace(relative) - 1.) / 2., -1., 1.))))


def align_spin(R_from, R_to, axis_in_from_frame):
    """Spin ``R_from`` about its own ``axis`` so that it comes closest to ``R_to``.

    Returns ``(theta, R_from @ rotation_about(axis, theta))`` with ``theta`` in
    (-pi, pi] minimising the residual rotation between the spun frame and ``R_to``.
    Closed form: ``Rd = R_from.T @ R_to`` expressed in a basis whose third axis is
    the symmetry axis, ``theta = atan2(Rd'[1,0] - Rd'[0,1], Rd'[0,0] + Rd'[1,1])``.
    When the two axes are antiparallel every spin leaves the same residual and
    ``theta`` is reported as 0.
    """
    R_from = np.asarray(R_from, dtype=float)[:3, :3]
    R_to = np.asarray(R_to, dtype=float)[:3, :3]
    basis = axis_basis(axis_in_from_frame)
    planar = basis.T @ (R_from.T @ R_to) @ basis
    theta = math.atan2(planar[1, 0] - planar[0, 1], planar[0, 0] + planar[1, 1])
    if theta <= -math.pi:
        theta += 2. * math.pi
    return float(theta), R_from @ rotation_about(axis_in_from_frame, theta)


def symmetric_pose_errors(pose_matrix, target_matrix, axis_object):
    """``(position_m, rotation_rad, spin_rad)`` of an object pose against a target in the same frame.

    Position is the centre distance (unchanged). Rotation is the residual after the
    best spin about the object's symmetry axis (the axis-direction error). Spin is
    that angle: ``pose.R ~ target.R @ rotation_about(axis, spin)``.
    """
    pose = np.asarray(pose_matrix, dtype=float)
    target = np.asarray(target_matrix, dtype=float)
    position = float(np.linalg.norm(pose[:3, 3] - target[:3, 3]))
    spin, aligned_target = align_spin(target[:3, :3], pose[:3, :3], axis_object)
    return position, residual_angle(aligned_target, pose[:3, :3]), spin


def pinch_point(reference_object_to_tcp, axis_object, pivot_radius_m):
    """The rim point (object frame) where the pads pinch the object, derived from the reference grasp.

    It lies on the circle of radius ``pivot_radius_m`` about the symmetry axis in
    the plane of the object origin, at the azimuth of the reference TCP origin
    (``object_to_tcp`` = the TCP pose in the object frame). A reference TCP on the
    axis has no azimuth and is rejected: a rim pinch never has one there.
    """
    axis = unit_axis(axis_object)
    radius = pivot_radius(pivot_radius_m)
    tcp = np.asarray(reference_object_to_tcp, dtype=float)[:3, 3]
    perpendicular = tcp - float(np.dot(tcp, axis)) * axis
    length = float(np.linalg.norm(perpendicular))
    if not math.isfinite(length) or length < MIN_PINCH_AZIMUTH_M:
        raise ValueError("the reference TCP lies on the object's symmetry axis: no pinch point")
    return radius * perpendicular / length


def _cylindrical(point, axis):
    axial = float(np.dot(point, axis))
    return float(np.linalg.norm(point - axial * axis)), axial


def symmetric_contact_errors(pose_matrix, reference_matrix, axis_object, contact_object):
    """``(position_m, rotation_rad, spin_rad)`` of an object pose against a reference, measured at the pinch.

    Both poses are in the same frame; ``contact_object`` is the pinched rim point in
    the object frame (``pinch_point``). Position is the distance from the
    reference's contact point to the pose's rim circle (the circle through the
    contact about the symmetry axis), i.e. the radial and axial offset of the rim
    at the pads. It is zero for a spin of the ring about its centre (the rim
    sliding through the pads) AND for a rotation about the pads themselves (the
    ring hanging at a new angle, its centre swung), while a radial or axial slip
    of the rim at the pads is reported in full. Rotation is the residual tilt
    after the best spin, spin the angle itself (``pose.R ~ reference.R @
    rotation_about(axis, spin)``).
    """
    axis = unit_axis(axis_object)
    contact = np.asarray(contact_object, dtype=float)
    if contact.shape != (3,) or not np.isfinite(contact).all():
        raise ValueError("contact_object must be a finite point in the object frame")
    pose = np.asarray(pose_matrix, dtype=float)
    reference = np.asarray(reference_matrix, dtype=float)
    # The reference's pinch point expressed in the pose's object frame.
    point = (np.linalg.inv(pose) @ reference @ np.array([*contact, 1.]))[:3]
    radial, axial = _cylindrical(point, axis)
    radial_reference, axial_reference = _cylindrical(contact, axis)
    position = math.hypot(radial - radial_reference, axial - axial_reference)
    spin, aligned_reference = align_spin(reference[:3, :3], pose[:3, :3], axis)
    return position, residual_angle(aligned_reference, pose[:3, :3]), spin


def symmetric_capture_errors(measured_object_to_tcp, reference_object_to_tcp, axis_object, pivot_radius_m=None):
    """``(position_m, rotation_rad, spin_rad)`` of a measured grasp transform against its capture.

    Both object_to_tcp transforms ``G`` are inverted to the object pose in the HAND
    frame ``H = inv(G)``, where a spin of the object about its own axis cancels.
    Rotation is the ring-axis tilt relative to the hand and spin the angle the
    object has turned about its axis inside the closed pads
    (``H.R ~ H0.R @ rotation_about(axis, spin)``). Without ``pivot_radius_m``
    position is the ring-centre motion relative to the hand, so only a spin about
    the centre gives ``(0, 0, spin)``. With it, position is measured at the pinch
    contact derived from the reference grasp (``symmetric_contact_errors``), so a
    spin about the centre and a rotation about the pads both give ``(0, 0, spin)``
    while a radial or axial slip of the rim at the pads keeps its full size.
    """
    measured = np.linalg.inv(np.asarray(measured_object_to_tcp, dtype=float))
    reference = np.linalg.inv(np.asarray(reference_object_to_tcp, dtype=float))
    if pivot_radius_m is None:
        return symmetric_pose_errors(measured, reference, axis_object)
    contact = pinch_point(reference_object_to_tcp, axis_object, pivot_radius_m)
    return symmetric_contact_errors(measured, reference, axis_object, contact)


def spin_aligned_object(object_matrix, reference_rotation, axis_object):
    """``object_matrix`` spun about its own axis to best match ``reference_rotation``; position unchanged.

    This is the object "as if it had not spun" inside the pads. Hand goals derived
    through ``desired_object @ inv(object) @ tcp`` use it so that they never roll
    the hand to undo a physically irrelevant spin. The measured centre is kept, so
    a ring that swung about the pinch is still carried to where it is wanted.
    """
    aligned = np.array(object_matrix, dtype=float, copy=True)
    aligned[:3, :3] = align_spin(aligned[:3, :3], np.asarray(reference_rotation, dtype=float)[:3, :3],
                                 axis_object)[1]
    return aligned


def remove_axis_component(vector, axis_w):
    """``vector`` minus its component along the unit ``axis_w`` (a rotation correction without spin)."""
    axis = unit_axis(axis_w)
    vector = np.asarray(vector, dtype=float)
    return vector - float(np.dot(vector, axis)) * axis
