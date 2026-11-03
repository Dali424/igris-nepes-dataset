"""Isolated measured-pelvis FK/Jacobian adapter; no controller or simulator writes.

One instance owns mutable Pinocchio Data and must not be shared by concurrent
threads. The URDF is torso-rooted: pelvis is a measured boundary condition, not
a commanded floating base. Jacobian columns include its inverse derivative.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
from numbers import Real
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin


WAIST = ("Joint_Waist_Yaw", "Joint_Waist_Roll", "Joint_Waist_Pitch")
ARM_PARTS = ("Shoulder_Pitch", "Shoulder_Roll", "Shoulder_Yaw", "Elbow_Pitch",
             "Wrist_Yaw", "Wrist_Roll", "Wrist_Pitch")
PELVIS = "Link_Waist_Pitch"


def _default_names(arm):
    return WAIST + tuple(f"Joint_{part}_{arm.title()}" for part in ARM_PARTS)


def _pose_matrix(pose):
    """Match the production scalar-first pose normalization contract."""
    try:
        value = np.asarray(pose, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Pelvis pose must contain seven finite numbers") from exc
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("Pelvis pose must be [x,y,z,qw,qx,qy,qz]")
    norm = np.linalg.norm(value[3:])
    if not np.isclose(norm, 1., atol=1e-3, rtol=0):
        raise ValueError("Pelvis quaternion must be normalized")
    w, x, y, z = value[3:] / norm
    rotation = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    return pin.SE3(rotation, value[:3])


class PinocchioUpperBodyKinematics:
    """World TCP and [linear; angular] world Jacobian for waist3 + arm7.

    ``owned_names_by_arm`` may reorder each arm's exact ten owned joints; the
    default reproduces UpperBodyServo.names(). All movable URDF joints must be
    supplied by name. Like production, measured physical-limit overshoot up to
    0.05 is clipped to authored limits for FK; larger excursions are rejected.
    Command limits, IK, joint speeds, contact and balance remain the caller's
    responsibility. Returned arrays are copies independent of mutable Data.
    """

    def __init__(self, urdf_path, owned_names_by_arm=None):
        self.urdf_path = Path(urdf_path).expanduser().resolve(strict=True)
        xml = ET.parse(self.urdf_path).getroot()
        authored = {}
        self._known_joint_names = set()
        for joint in xml.findall("joint"):
            name, kind = joint.get("name"), joint.get("type")
            if not name or name in self._known_joint_names:
                raise ValueError("URDF joint names must be present and unique")
            self._known_joint_names.add(name)
            if kind not in ("fixed", "revolute", "prismatic") or joint.find("mimic") is not None:
                raise ValueError(f"Unsupported joint type or mimic: {name}")
            if kind == "fixed":
                continue
            limit = joint.find("limit")
            if limit is None:
                raise ValueError(f"Missing physical joint limits: {name}")
            try:
                low, high = float(limit.attrib['lower']), float(limit.attrib['upper'])
            except (ValueError, KeyError) as exc:
                raise ValueError(f"Invalid physical joint limits: {name}") from exc
            if not np.isfinite([low, high]).all() or low >= high:
                raise ValueError(f"Invalid physical joint limits: {name}")
            authored[name] = (low, high)

        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()
        self.q_index, self.v_index = {}, {}
        for jid, name in enumerate(self.model.names):
            if jid == 0:
                continue
            joint = self.model.joints[jid]
            if joint.nq != 1 or joint.nv != 1 or name not in authored:
                raise ValueError(f"Unsupported non-scalar or unauthored joint: {name}")
            self.q_index[name], self.v_index[name] = joint.idx_q, joint.idx_v
        if set(self.q_index) != set(authored):
            raise ValueError("Pinocchio and URDF movable-joint names differ")
        self.joint_names = tuple(self.q_index)
        self.joint_limits = authored
        self._low = np.asarray([authored[n][0] for n in self.joint_names])
        self._high = np.asarray([authored[n][1] for n in self.joint_names])
        self._q_columns = np.asarray([self.q_index[n] for n in self.joint_names], dtype=int)
        self._neutral = pin.neutral(self.model)

        requested = ({arm: _default_names(arm) for arm in ('left', 'right')}
                     if owned_names_by_arm is None else owned_names_by_arm)
        if not isinstance(requested, Mapping) or set(requested) != {'left', 'right'}:
            raise ValueError("owned_names_by_arm must contain exactly left and right")
        self.owned_names_by_arm, self._v_columns = {}, {}
        for arm, names in requested.items():
            if isinstance(names, str):
                raise ValueError("Owned joint names must be a sequence")
            names = tuple(names)
            if len(names) != 10 or set(names) != set(_default_names(arm)) or not set(names) <= set(self.q_index):
                raise ValueError(f"Expected the exact ten waist/active-arm joint names for {arm}")
            self.owned_names_by_arm[arm] = names
            self._v_columns[arm] = np.asarray([self.v_index[n] for n in names], dtype=int)
        self.pelvis_id = self._frame_id(PELVIS)
        self.hand_ids = {arm: self._frame_id(f'{arm.title()}_Hand') for arm in ('left', 'right')}
        self.provenance = {
            'backend': 'isolated_pinocchio_measured_pelvis_v1', 'pinocchio_version': pin.__version__,
            'urdf': str(self.urdf_path), 'urdf_sha256': hashlib.sha256(self.urdf_path.read_bytes()).hexdigest(),
            'owned_names_by_arm': {arm: list(names) for arm, names in self.owned_names_by_arm.items()},
            'pelvis_frame': PELVIS, 'jacobian_rows': 'world_linear_then_world_angular',
            'physical_overshoot_clip_tolerance': .05, 'root_or_joint_state_written': False,
        }

    def _frame_id(self, name):
        fid = self.model.getFrameId(name, pin.FrameType.BODY)
        if fid >= self.model.nframes:
            raise ValueError(f"Missing BODY frame: {name}")
        return fid

    def _q(self, joints):
        if not isinstance(joints, Mapping):
            raise ValueError("Measured joints must be a name/value mapping")
        if set(joints) - self._known_joint_names:
            raise ValueError("Unknown measured joint names")
        values = []
        for name in self.joint_names:
            value = joints.get(name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"Missing or nonfinite measured joint: {name}")
            values.append(float(value))
        values = np.asarray(values)
        if np.any(values < self._low - .05) or np.any(values > self._high + .05):
            raise ValueError("Measured joint outside authored physical limits")
        q = self._neutral.copy()
        q[self._q_columns] = np.clip(values, self._low, self._high)
        return q

    def pose_and_jacobian(self, joints, pelvis_pose_w, arm):
        if arm not in self.hand_ids:
            raise ValueError("arm must be left or right")
        q, pelvis_w = self._q(joints), _pose_matrix(pelvis_pose_w)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        a = self.data.oMf[self.pelvis_id]
        b = self.data.oMf[self.hand_ids[arm]]
        root_w = pelvis_w * a.inverse()
        columns = self._v_columns[arm]
        ja = pin.getFrameJacobian(self.model, self.data, self.pelvis_id,
                                 pin.LOCAL_WORLD_ALIGNED)[:, columns]
        jb = pin.getFrameJacobian(self.model, self.data, self.hand_ids[arm],
                                 pin.LOCAL_WORLD_ALIGNED)[:, columns]
        delta = b.translation - a.translation
        jv = jb[:3] - ja[:3] - np.cross(ja[3:].T, delta).T
        jw = jb[3:] - ja[3:]
        jacobian = np.vstack((root_w.rotation @ jv, root_w.rotation @ jw))
        return (root_w * b).homogeneous.copy(), jacobian.copy()
