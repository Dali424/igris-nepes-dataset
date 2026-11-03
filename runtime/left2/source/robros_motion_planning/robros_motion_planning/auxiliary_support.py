"""Experimental fixed-jaw support for a bounded, unsuccessful lift diagnostic.

The primary FSM remains the authority for pinch, immutable capture, payload
retention and task success. This controller never writes object state. A 1 N
acquisition threshold is sensor readiness only, not a load-sharing certificate.
Only source acquisition and the first <=10 cm of lift are supported.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
import numpy as np

from .crs_manipulation import TcpPoseController
from .snapshot import matrix_to_pose, pose_to_matrix
from .task_control import rotation_error

ALLOWED_STAGES = frozenset({'prerequisites', 'navigate_source', 'pregrasp', 'approach', 'close', 'lift'})
DEFAULTS = dict(minimum_support_force_n=1., support_dwell_s=.1,
                approach_speed_m_s=.005, approach_acceleration_m_s2=.01,
                position_tolerance_m=.003, rotation_tolerance_rad=.025,
                joint_tolerance_rad=.03, acquisition_timeout_s=10.,
                lost_support_grace_s=.1, max_object_lift_m=.10)


def validate_auxiliary_support(settings, primary_arm):
    if not isinstance(settings, Mapping):
        raise ValueError('auxiliary_support_arm must be a mapping')
    allowed = set(DEFAULTS) | {'arm', 'object_to_tcp', 'precontact_offset_w',
                              'joint_position_rad', 'geometry_evidence', 'pregrasp_pose_control', 'pregrasp_waypoint_offsets_object'}
    if set(settings) - allowed:
        raise ValueError('Unknown auxiliary_support_arm fields')
    arm = settings.get('arm')
    if primary_arm not in {'left', 'right'} or arm not in {'left', 'right'} or arm == primary_arm:
        raise ValueError('Auxiliary support must use the opposite arm')
    pose = np.asarray(settings.get('object_to_tcp'), float)
    if pose.shape != (7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:]) - 1.) > 1e-5:
        raise ValueError('Auxiliary object_to_tcp needs a finite unit-quaternion pose')
    pose_to_matrix(pose)
    offset = np.asarray(settings.get('precontact_offset_w'), float)
    if offset.shape != (3,) or not np.isfinite(offset).all() or not np.allclose(offset, [0, 0, -.005], atol=1e-10, rtol=0):
        raise ValueError('Auxiliary prefix requires a world -Z 5 mm precontact gap')
    q = settings.get('joint_position_rad')
    if isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(q) or not .1 <= (q if arm == 'left' else -q) <= 1.5:
        raise ValueError('Auxiliary jaw must retain its geometry-checked open angle')
    if not isinstance(settings.get('geometry_evidence'), str) or not settings['geometry_evidence'].strip():
        raise ValueError('Auxiliary support requires geometry_evidence')
    result = {**DEFAULTS, **dict(settings), 'object_to_tcp': pose.tolist(),
              'precontact_offset_w': offset.tolist(), 'joint_position_rad': float(q)}
    for name in DEFAULTS:
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Auxiliary ' + name + ' must be positive and finite')
        result[name] = float(value)
    bounds = {'approach_speed_m_s': .005, 'approach_acceleration_m_s2': .01,
              'position_tolerance_m': .003, 'rotation_tolerance_rad': .025,
              'joint_tolerance_rad': .03, 'max_object_lift_m': .10,
              'acquisition_timeout_s': 15., 'lost_support_grace_s': .1}
    if any(result[n] > high for n, high in bounds.items()):
        raise ValueError('Auxiliary prefix exceeds its bounded acquisition/diagnostic limits')
    if result['minimum_support_force_n'] < 1. or result['support_dwell_s'] < .1:
        raise ValueError('Auxiliary support acquisition requires at least 1 N for 0.1 s')
    if 'pregrasp_pose_control' in result:
        free = result['pregrasp_pose_control']
        upper = {'linear_speed': .03, 'linear_acceleration': .06,
                 'angular_speed': .12, 'angular_acceleration': .3,
                 'position_lead': .01, 'rotation_lead': .02,
                 'cross_track_position': .01, 'cross_track_rotation': .05}
        if not isinstance(free, Mapping) or set(free) - upper.keys():
            raise ValueError('Unknown auxiliary pregrasp_pose_control fields')
        for name, value in free.items():
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 < value <= upper[name]):
                raise ValueError('Auxiliary free pregrasp exceeds bounded reference limits')
        result['pregrasp_pose_control'] = {n: float(v) for n, v in free.items()}
    if 'pregrasp_waypoint_offsets_object' in result:
        offsets = result['pregrasp_waypoint_offsets_object']
        if not isinstance(offsets, list) or not 1 <= len(offsets) <= 3:
            raise ValueError('Auxiliary pregrasp needs 1..3 explicit exterior waypoints')
        validated = []
        for offset in offsets:
            value = np.asarray(offset, float)
            if (value.shape != (3,) or not np.isfinite(value).all()
                    or not .01 <= np.linalg.norm(value) <= .3):
                raise ValueError('Auxiliary exterior waypoint must be a finite 1..30 cm object-frame offset')
            validated.append(value.tolist())
        result['pregrasp_waypoint_offsets_object'] = validated
    return result


def fixed_jaw_support_evidence(contact, arm, object_path):
    """Signed force on the object: negative of the filtered force ON the hand.

    Never sum moving-jaw forces or use an unsigned force norm for readiness.
    Missing, malformed and mismatched views fail closed.
    """
    unavailable = {'available': False, 'source': 'physx_filtered_fixed_jaw_object_contact',
                   'object_path': object_path, 'arm': arm, 'force_on_object_w': None,
                   'upward_force_n': None, 'all_hand_contact_force_magnitude_n': None}
    if not isinstance(contact, Mapping) or contact.get('available') is not True:
        return {**unavailable, 'reason': 'contact_sensor_unavailable'}
    if contact.get('source') != 'physx_filtered_hand_object_contacts' or contact.get('object_path') != object_path or contact.get('arm') != arm:
        return {**unavailable, 'reason': 'contact_sensor_identity_mismatch'}
    paths = contact.get('sensor_paths', [])
    fixed_name = 'Left_Hand' if arm == 'left' else 'Right_Hand'
    if not isinstance(paths, list) or any(not isinstance(p, str) for p in paths):
        return {**unavailable, 'reason': 'contact_sensor_paths_invalid'}
    indices = [i for i, path in enumerate(paths) if path.rsplit('/', 1)[-1] == fixed_name]
    try:
        forces = np.asarray(contact.get('forces_w'), float)
    except (TypeError, ValueError):
        return {**unavailable, 'reason': 'contact_force_matrix_invalid'}
    if len(indices) != 1 or forces.shape != (len(paths), 1, 3) or not np.isfinite(forces).all():
        return {**unavailable, 'reason': 'contact_force_matrix_invalid'}
    force = -forces[indices[0], 0]
    return {**unavailable, 'available': True, 'reason': None,
            'sensor_path': paths[indices[0]], 'force_on_object_w': force.tolist(),
            'upward_force_n': float(force[2]),
            'all_hand_contact_force_magnitude_n': float(np.linalg.norm(forces, axis=-1).sum()),
            'interpretation': 'Filtered fixed-jaw force; 1 N readiness is not verified load sharing.'}


class AuxiliarySupportController:
    """Acquire only after primary close; follow an immutable nominal lift frame.

    ``observe`` can be called again at a stage transition at the same timestamp.
    ``target`` advances the bounded acquisition reference exactly once per step.
    During lift, a missing support pauses the primary reference immediately,
    then fails after grace. No pose measured from the sagging payload is used
    for the secondary reference after acquisition starts.
    """
    def __init__(self, settings, primary_arm, object_path):
        self.settings = validate_auxiliary_support(settings, primary_arm)
        self.arm, self.object_path = self.settings['arm'], object_path
        self.joint_name = ('l' if self.arm == 'left' else 'r') + '_1_dof_gripper_moving'
        self._grasp = pose_to_matrix(self.settings['object_to_tcp'])
        self.stage = None
        self._anchor = self._nominal = None
        self._reference = None
        self._approach_goals = []
        self._approach_index = 0
        self._free_settings = None
        self._good_since = self._lost_since = self._lift_since = None
        self._lift_z = None
        self._acquired = False
        self._last_time = None
        self._last = {}

    def _goal(self, *, gap=False):
        goal = self._anchor @ self._grasp
        if gap:
            goal[:3, 3] += self.settings['precontact_offset_w']
        return goal

    def _pose_ready(self, actual, target):
        return (np.linalg.norm(actual[:3, 3] - target[:3, 3]) <= self.settings['position_tolerance_m']
                and np.linalg.norm(rotation_error(target[:3, :3], actual[:3, :3])) <= self.settings['rotation_tolerance_rad'])

    def observe(self, stage, observation):
        now = float(observation['time_s'])
        if not math.isfinite(now) or (self._last_time is not None and now < self._last_time):
            raise ValueError('Auxiliary support requires monotonic finite measurement time')
        self._last_time = now
        if stage not in ALLOWED_STAGES:
            self._last = {'status': 'failed', 'reason': 'auxiliary_support_unsupported_stage:' + stage,
                          'stage_id': stage, 'precontact_ready': False, 'ready': False}
            return dict(self._last)
        actual = pose_to_matrix(observation['tcp_poses_w'][self.arm])
        obj = pose_to_matrix(observation['object_poses_w'][self.object_path])
        if stage != self.stage:
            self.stage = stage
            if stage == 'pregrasp':
                self._anchor = obj.copy()
                # Free-space approach may start from a separately audited park.
                # Its optional bounds NEVER alter the subsequent 5 mm contact approach.
                free = dict(linear_speed=self.settings['approach_speed_m_s'],
                    linear_acceleration=self.settings['approach_acceleration_m_s2'],
                    angular_speed=.05, angular_acceleration=.1, position_lead=.001,
                    rotation_lead=.01, cross_track_position=.003, cross_track_rotation=.025)
                free.update(self.settings.get('pregrasp_pose_control', {}))
                self._free_settings = free
                final_goal = self._goal(gap=True)
                self._approach_goals = []
                for offset in self.settings.get('pregrasp_waypoint_offsets_object', []):
                    exterior = final_goal.copy()
                    exterior[:3, 3] += self._anchor[:3, :3] @ offset
                    self._approach_goals.append(exterior)
                self._approach_goals.append(final_goal)
                self._reference = TcpPoseController(matrix_to_pose(actual), matrix_to_pose(self._approach_goals[0]),
                    reference_profile='online', **free)
            elif stage == 'lift':
                # Primary capture is already fixed by the FSM at this point.
                self._anchor = obj.copy()
                self._nominal = obj.copy()
                self._lift_z, self._lift_since = float(obj[2, 3]), now
                self._reference = TcpPoseController(matrix_to_pose(actual), matrix_to_pose(self._goal()),
                    linear_speed=self.settings['approach_speed_m_s'],
                    linear_acceleration=self.settings['approach_acceleration_m_s2'],
                    angular_speed=.05, angular_acceleration=.1, position_lead=.001,
                    rotation_lead=.01, cross_track_position=.003, cross_track_rotation=.025,
                    reference_profile='online')
        raw = observation.get('contact_evidence', {}).get(self.arm, {}).get(self.object_path)
        evidence = fixed_jaw_support_evidence(raw, self.arm, self.object_path)
        q = observation.get('joint_positions', {}).get(self.joint_name)
        jaw_ready = (isinstance(q, (int, float)) and math.isfinite(q)
                     and abs(q - self.settings['joint_position_rad']) <= self.settings['joint_tolerance_rad'])
        active = stage in {'pregrasp', 'approach', 'close', 'lift'}
        precontact = bool(self._anchor is not None and jaw_ready and evidence['available']
                          and evidence['all_hand_contact_force_magnitude_n'] <= .1
                          and self._approach_index == len(self._approach_goals) - 1
                          and self._pose_ready(actual, self._goal(gap=True)))
        ready = False
        reason = None
        rise = 0. if self._lift_z is None else float(obj[2, 3] - self._lift_z)
        if active and not evidence['available']:
            reason = evidence['reason']
        if stage == 'lift':
            target = self._nominal @ self._grasp
            good = (evidence['available'] and jaw_ready
                    and evidence['upward_force_n'] >= self.settings['minimum_support_force_n']
                    and self._pose_ready(actual, target))
            if good:
                self._good_since = now if self._good_since is None else self._good_since
                self._lost_since = None
                ready = now - self._good_since + 1e-12 >= self.settings['support_dwell_s']
                self._acquired = self._acquired or ready
            else:
                self._good_since = None
                if self._acquired:
                    self._lost_since = now if self._lost_since is None else self._lost_since
                    if now - self._lost_since + 1e-12 >= self.settings['lost_support_grace_s']:
                        reason = reason or 'auxiliary_support_lost'
            if not self._acquired and now - self._lift_since >= self.settings['acquisition_timeout_s']:
                reason = reason or 'auxiliary_support_acquisition_timeout'
        self._last = {'status': 'failed' if reason else 'running', 'reason': reason,
            'stage_id': stage, 'arm': self.arm, 'joint_name': self.joint_name,
            'joint_position_rad': self.settings['joint_position_rad'],
            'precontact_ready': precontact, 'ready': bool(ready), 'ever_acquired': self._acquired,
            'pregrasp_waypoint_index': self._approach_index,
            'pregrasp_waypoint_count': len(self._approach_goals),
            'hold_primary_reference': stage == 'lift' and not ready,
            'hold_reason': 'auxiliary_support_not_ready' if stage == 'lift' and not ready else None,
            'object_lift_m': rise, 'diagnostic_stop': bool(stage == 'lift' and self._acquired and rise >= self.settings['max_object_lift_m']),
            'evidence': evidence, 'reference_source': 'nominal_primary_object_path',
            'load_sharing_success_verified': False, 'full_task_success_supported': False}
        return dict(self._last)

    def target(self, observation, dt, *, nominal_object_pose_w=None):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('Auxiliary reference dt must be finite and positive')
        if self.stage not in {'pregrasp', 'approach', 'close', 'lift'}:
            return None
        if self._last.get('status') != 'running':
            return None
        if self.stage == 'lift' and self._acquired:
            if nominal_object_pose_w is None:
                raise ValueError('Acquired auxiliary support requires the primary nominal object reference')
            self._nominal = pose_to_matrix(nominal_object_pose_w)
            return matrix_to_pose(self._nominal @ self._grasp)
        reference = self._reference.update(observation['tcp_poses_w'][self.arm], dt)
        if (self.stage == 'pregrasp' and reference['reference_complete']
                and self._approach_index + 1 < len(self._approach_goals)
                and self._pose_ready(pose_to_matrix(observation['tcp_poses_w'][self.arm]),
                                     self._approach_goals[self._approach_index])):
            self._approach_index += 1
            # Start from the last command, preserving reference continuity.
            self._reference = TcpPoseController(reference['tcp_target_pose_w'],
                matrix_to_pose(self._approach_goals[self._approach_index]),
                reference_profile='online', **self._free_settings)
        return reference['tcp_target_pose_w']
