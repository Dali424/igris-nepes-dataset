"""Measured-angle following for a passively hinged, physically grasped cover.

This module only calculates robot targets. It never drives a hinge or writes an
object pose. A CPU IK path is a calibration candidate, not permission to claim
collision-free or successful execution. The caller must monitor contacts,
floating-base tracking, robot collisions and the final release/settling stage.
"""
from __future__ import annotations

from copy import deepcopy
import math
from collections.abc import Mapping

import numpy as np

from .hinged_door import hinge_pose_w, measure_hinge_state, validate_hinge_config
from .snapshot import matrix_to_pose, pose_to_matrix


def _pose_interpolate(a, b, fraction):
    a, b = np.asarray(a, float), np.asarray(b, float)
    qa, qb = a[3:], b[3:]
    dot = float(qa @ qb)
    if dot < 0:
        qb, dot = -qb, -dot
    if dot > .9995:
        q = (1 - fraction) * qa + fraction * qb
        q /= np.linalg.norm(q)
    else:
        angle = math.acos(np.clip(dot, -1, 1))
        q = (math.sin((1 - fraction) * angle) * qa + math.sin(fraction * angle) * qb) / math.sin(angle)
    return np.r_[(1 - fraction) * a[:3] + fraction * b[:3], q].tolist()


class HingeFollowingPath:
    """Validated angle-indexed robot targets, including optional waist posture.

    Waypoints require ``hinge_angle_rad``, ``base_pose_w`` and
    ``arm_joint_positions``. ``waist_joint_positions`` is optional. Joint limits
    must come from the actual robot model, not from the candidate artifact.
    Only paths spanning closed through the configured full opening are accepted.
    """

    def __init__(self, config: Mapping, artifact: Mapping, joint_limits: Mapping):
        self.config = deepcopy(dict(config))
        self.hinge = validate_hinge_config(config)
        if artifact.get('feasible') is not True:
            raise ValueError('A complete feasible kinematic path is required')
        self.grasp = pose_to_matrix(artifact.get('grasp_pose_object'))
        self.waypoints = deepcopy(artifact.get('waypoints', []))
        if len(self.waypoints) < 2:
            raise ValueError('A hinge path needs at least two waypoints')
        self.angles = np.asarray([w['hinge_angle_rad'] for w in self.waypoints], float)
        end = math.radians(self.hinge['open_angle_deg'])
        if (not np.isfinite(self.angles).all() or np.any(np.diff(self.angles) >= 0)
                or abs(self.angles[0]) > 1e-6 or abs(self.angles[-1] - end) > 1e-6):
            raise ValueError('Path angles must strictly decrease from zero to full opening')
        expected = None
        for point in self.waypoints:
            pose_to_matrix(point['base_pose_w'])
            arm = point.get('arm_joint_positions')
            waist = point.get('waist_joint_positions', {})
            if not isinstance(arm, Mapping) or not arm or not isinstance(waist, Mapping):
                raise ValueError('Finite arm and optional waist joint targets are required')
            if set(arm) & set(waist):
                raise ValueError('Arm and waist names must be distinct')
            names = (frozenset(arm), frozenset(waist))
            if expected is not None and names != expected:
                raise ValueError('Every waypoint must contain the same named joints')
            expected = names
            for name, value in {**arm, **waist}.items():
                if name not in joint_limits:
                    raise ValueError(f'Unknown robot joint {name}')
                low, high = joint_limits[name]
                if not math.isfinite(value) or not low <= value <= high:
                    raise ValueError(f'Candidate outside physical joint limits: {name}')
        self.provenance = {
            'source_robot_urdf_sha256': artifact.get('robot_urdf_sha256'),
            'collision_free_execution_verified': False,
            'physical_grasp_verified': False,
            'posture_assumption': artifact.get('posture_assumption'),
        }

    def sample(self, angle_rad: float) -> dict:
        if not math.isfinite(angle_rad) or not self.angles[-1] - 1e-8 <= angle_rad <= 1e-8:
            raise ValueError('Requested angle lies outside the complete hinge path')
        index = min(int(np.searchsorted(-self.angles, -angle_rad, side='right')) - 1, len(self.angles) - 2)
        index = max(index, 0)
        a, b = self.waypoints[index:index + 2]
        fraction = float(np.clip((angle_rad - self.angles[index]) / (self.angles[index+1] - self.angles[index]), 0, 1))
        interpolate = lambda field: {n: (1 - fraction) * v + fraction * b.get(field, {})[n]
                                     for n, v in a.get(field, {}).items()}
        return {
            'hinge_angle_rad': float(angle_rad),
            'base_goal_pose_w': _pose_interpolate(a['base_pose_w'], b['base_pose_w'], fraction),
            'arm_joint_seed': interpolate('arm_joint_positions'),
            'waist_joint_targets': interpolate('waist_joint_positions'),
            'tcp_goal_pose_w': matrix_to_pose(pose_to_matrix(hinge_pose_w(self.config, angle_rad)) @ self.grasp),
        }


class HingeFollower:
    """Advance only as fast as measured hinge motion and base tracking permit.

    Start after physical grasp capture. ``hand_captured`` must be based on real
    bilateral contacts and measured hand/object relation, never a close command.
    ``opening_reached`` only completes this controller's arc; release, retreat
    and a settled-open success check belong to the task state machine.
    """

    def __init__(self, path: HingeFollowingPath, *, base_position_tolerance_m=.035,
                 base_rotation_tolerance_rad=.12, stall_timeout_s=8., total_timeout_s=90.,
                 base_speed_m_s=.12, base_angular_speed_rad_s=.15, joint_speed_rad_s=.3,
                 lost_grasp_grace_s=.1, waist_speed_rad_s=.08, waist_tracking_tolerance_rad=.04):
        values = [base_position_tolerance_m, base_rotation_tolerance_rad, stall_timeout_s,
                  total_timeout_s, base_speed_m_s, base_angular_speed_rad_s, joint_speed_rad_s]
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError('Tracking thresholds, timeouts and rates must be positive and finite')
        if not math.isfinite(lost_grasp_grace_s) or lost_grasp_grace_s < 0:
            raise ValueError('Contact-loss grace must be nonnegative and finite')
        if not all(math.isfinite(v) and v > 0 for v in (waist_speed_rad_s, waist_tracking_tolerance_rad)):
            raise ValueError('Waist rate and tracking tolerance must be finite and positive')
        self.path = path
        self.position_tolerance, self.rotation_tolerance = values[:2]
        self.stall_timeout, self.total_timeout = values[2:4]
        self.base_speed, self.base_angular_speed, self.joint_speed = values[4:]
        self.reference = 0.
        self.elapsed = self.stalled = 0.
        self.best_angle = 0.
        self.failure = None
        self.contact_gap = 0.
        self.contact_grace = lost_grasp_grace_s
        self.waist_speed = waist_speed_rad_s
        self.waist_tracking_tolerance = waist_tracking_tolerance_rad

    def step(self, *, cover_pose_w, base_pose_w, hand_captured: bool, dt: float,
             joint_positions: Mapping | None = None) -> dict:
        if not math.isfinite(dt) or not 0 < dt <= .1:
            raise ValueError('Controller dt must be finite and within (0, .1] seconds')
        if not isinstance(hand_captured, (bool, np.bool_)):
            raise ValueError('hand_captured must be measured boolean evidence')
        measured = measure_hinge_state(self.path.config, cover_pose_w)
        base = pose_to_matrix(base_pose_w)
        self.elapsed += dt
        if not measured['valid']:
            self.failure = self.failure or 'hinge_constraint_deviation'
        self.contact_gap = 0. if hand_captured else self.contact_gap + dt
        if not hand_captured and self.contact_gap + 1e-12 >= self.contact_grace:
            self.failure = self.failure or 'grasp_lost'
        angle = measured['angle_rad']
        if angle < self.best_angle - .005:
            self.best_angle, self.stalled = angle, 0.
        else:
            self.stalled += dt
        if self.elapsed > self.total_timeout:
            self.failure = self.failure or 'hinge_motion_timeout'
        reached = abs(angle - float(self.path.angles[-1])) <= self.path.hinge['angle_tolerance_rad']
        if self.stalled > self.stall_timeout and not reached:
            self.failure = self.failure or 'hinge_motion_stalled'
        if angle - self.reference > self.path.hinge['max_target_lead_rad'] + 1e-6:
            self.failure = self.failure or 'hinge_recoil_exceeded_lead'
        current = self.path.sample(self.reference)
        waist_error = 0.
        for name, target in current['waist_joint_targets'].items():
            if joint_positions is None or name not in joint_positions or not math.isfinite(joint_positions[name]):
                self.failure = self.failure or 'waist_measurement_unavailable'
                waist_error = None
                break
            waist_error = max(waist_error, abs(joint_positions[name]-target))
        waist_tracking = waist_error is not None and waist_error <= self.waist_tracking_tolerance
        target_base = pose_to_matrix(current['base_goal_pose_w'])
        position_error = float(np.linalg.norm(base[:2, 3] - target_base[:2, 3]))
        # Floating-base roll/pitch is handled by the arm's measured-base servo;
        # the walking target is horizontal position and yaw.
        yaw_error = abs(math.atan2(math.sin(math.atan2(base[1, 0], base[0, 0]) - math.atan2(target_base[1, 0], target_base[0, 0])),
                                  math.cos(math.atan2(base[1, 0], base[0, 0]) - math.atan2(target_base[1, 0], target_base[0, 0]))))
        tracking = position_error <= self.position_tolerance and yaw_error <= self.rotation_tolerance
        end = float(self.path.angles[-1])
        if not self.failure and not reached and tracking and waist_tracking and hand_captured:
            wanted = max(end, self.reference - self.path.hinge['angular_speed_rad_s'] * dt,
                         angle - self.path.hinge['max_target_lead_rad'])
            wanted = min(0., wanted)
            next_target = self.path.sample(wanted)
            delta_base = np.linalg.norm(np.array(next_target['base_goal_pose_w'][:2]) - current['base_goal_pose_w'][:2])
            ba, bb = pose_to_matrix(current['base_goal_pose_w']), pose_to_matrix(next_target['base_goal_pose_w'])
            angle_delta = math.acos(np.clip((np.trace(ba[:3, :3].T @ bb[:3, :3]) - 1) / 2, -1, 1))
            deltas = [abs(v - current['arm_joint_seed'][n]) for n, v in next_target['arm_joint_seed'].items()]
            waist_deltas = [abs(v - current['waist_joint_targets'][n]) for n, v in next_target['waist_joint_targets'].items()]
            ratio = max(1., delta_base / (self.base_speed * dt), angle_delta / (self.base_angular_speed * dt),
                        max(deltas, default=0.) / (self.joint_speed * dt),
                        max(waist_deltas, default=0.) / (self.waist_speed * dt))
            self.reference += (wanted - self.reference) / ratio
        result = self.path.sample(self.reference)
        result.update(status='failed' if self.failure else ('opening_reached' if reached else 'running'),
                      reason=self.failure, measured_hinge=measured,
                      paused_for_base_tracking=not tracking, base_position_error_m=position_error,
                      paused_for_contact=not hand_captured, contact_gap_s=self.contact_gap,
                      paused_for_waist_tracking=not waist_tracking, waist_tracking_error_rad=waist_error,
                      base_yaw_error_rad=yaw_error, elapsed_s=self.elapsed,
                      collision_free_execution_verified=False, full_task_completed=False,
                      actuation='robot_targets_only_passive_hinge')
        if self.failure:
            # Fail closed: the caller gets no new motion targets on failure.
            for key in ('tcp_goal_pose_w', 'base_goal_pose_w', 'arm_joint_seed', 'waist_joint_targets'):
                result[key] = None
        return result


# Opt-in guarded_approach.lateral_tolerance_zones (runtime_v26): bounds of the widened alignment-pause tolerances.
LATERAL_TOLERANCE_ZONE_MAXIMUM_M = .003
LATERAL_TOLERANCE_ZONE_MAXIMUM_DISTANCE_M = .05
LATERAL_TOLERANCE_ZONE_MAXIMUM_COUNT = 8
# runtime_v26 review fix: a WIDENED zone tolerance may not exceed the pad-to-pin cross-track allowance at its zone
# distance, allowed(d) = (d - 1.3 mm) / 3 (contact_strategies c04, run_116 approach line: fixed-pad STL against the
# ring pins, worst case over 16 cross-track directions; the linear form matches the 0.25 mm-sampled profile within
# 0.035 mm: 3.0 mm at 10.3 mm, 2.5 mm at 8.8 mm, 2.0 mm at 7.3 mm, 1.0 mm at 4.3 mm, nothing below 1.05 mm).
LATERAL_TOLERANCE_ZONE_GEOMETRY_OFFSET_M = .0013
LATERAL_TOLERANCE_ZONE_GEOMETRY_SLOPE = 3.


def _zone_number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def validate_lateral_tolerance_zones(value, lateral_tolerance_m, name="guarded_approach.lateral_tolerance_zones"):
    """Opt-in widened alignment-pause tolerances away from the grasp goal; returns [(distance, tolerance), ...].

    ``value`` lists 1..8 ``[remaining_distance_m, lateral_tolerance_m]`` pairs with strictly decreasing remaining
    distances in (0, 0.05] m and tolerances in [the base ``lateral_tolerance_m``, 0.003] m that never increase
    toward the goal. While the remaining distance along the approach line exceeds a zone's distance, the tolerance
    of the first such zone replaces the base one in the reference's alignment pause; inside the last distance the
    base tolerance applies (the completion bounds never change). Geometry basis: contact_strategies c04 (run_116
    line, fixed-pad STL against the ring pins: 2.5 mm until 8.8 mm before G, 2.0 mm until 7.3 mm, then 1.2 mm); a
    tolerance wider than the base must stay within that allowance at its distance, (d - 1.3 mm) / 3.
    """
    base = _zone_number(lateral_tolerance_m, name + " base lateral_tolerance_m")
    if (isinstance(value, (str, bytes, Mapping)) or not isinstance(value, (list, tuple))
            or not 1 <= len(value) <= LATERAL_TOLERANCE_ZONE_MAXIMUM_COUNT):
        raise ValueError(f"{name} must be a list of 1..{LATERAL_TOLERANCE_ZONE_MAXIMUM_COUNT} "
                         "[remaining_distance_m, lateral_tolerance_m] pairs")
    zones = []
    for index, pair in enumerate(value):
        label = f"{name}[{index}]"
        if isinstance(pair, (str, bytes, Mapping)) or not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(label + " must be a [remaining_distance_m, lateral_tolerance_m] pair")
        distance, tolerance = (_zone_number(item, label) for item in pair)
        if not 0. < distance <= LATERAL_TOLERANCE_ZONE_MAXIMUM_DISTANCE_M:
            raise ValueError(f"{label} remaining_distance_m must be in (0, {LATERAL_TOLERANCE_ZONE_MAXIMUM_DISTANCE_M:g}] m")
        if not base <= tolerance <= LATERAL_TOLERANCE_ZONE_MAXIMUM_M:
            raise ValueError(f"{label} lateral_tolerance_m must be within [lateral_tolerance_m, "
                             f"{LATERAL_TOLERANCE_ZONE_MAXIMUM_M:g}] m")
        if zones and distance >= zones[-1][0]:
            raise ValueError(name + " remaining distances must strictly decrease")
        if zones and tolerance > zones[-1][1]:
            raise ValueError(name + " tolerances must not increase toward the goal")
        allowed = (distance - LATERAL_TOLERANCE_ZONE_GEOMETRY_OFFSET_M) / LATERAL_TOLERANCE_ZONE_GEOMETRY_SLOPE
        if tolerance > base + 1e-12 and tolerance > allowed + 1e-12:
            raise ValueError(f"{label} lateral_tolerance_m {tolerance:g} m exceeds the pad-to-pin cross-track "
                             f"allowance {max(allowed, 0.):.6g} m at remaining_distance_m {distance:g} m "
                             "((d - 1.3 mm) / 3, contact_strategies c04)")
        zones.append((distance, tolerance))
    return zones


class GuardedHandleApproach:
    """Slow, measured-progress insertion into an open gripper.

    The measured pregrasp must already be aligned with the grasp. A short reference
    along their line avoids commanding the full insertion while the physical
    arm has lateral or angular tracking error. This is a contact/pose gate,
    not collision planning. It never authorizes grasp success or moves objects.

    Opt-in ``guard_point_tcp_m`` (TCP frame, e.g. the fixed pad tip): the lateral
    error is that point's cross-track distance from the line the reference
    carries it along (goal rotation @ p + start position -> goal @ p, parallel
    to the origin's line), so a hand tilt the origin cannot see pauses the
    reference too (lateral_error_m; the origin's is origin_lateral_error_m). A
    measured start tilt pauses it until the hand turns to the commanded grasp
    orientation: a line from the measured start @ p would build that tilt into
    the path, and the hand would leave it for good once it obeyed (a stall).
    Progress and the commanded reference stay the origin's.
    ``completion_position_m`` is the task machines' approach completion bound;
    it is accepted with the validated settings and never changes the reference.
    So is the opt-in ``pressed_key_completion`` (runtime_v26, pressed_key_completion.py).
    Opt-in ``lateral_tolerance_zones`` (runtime_v26): while the remaining
    distance along the line, measured from whichever of the reference and
    the measured TCP has progressed farther toward the goal (so a zone never
    widens for where the hand already is or is being carried to), exceeds a
    zone's distance, that first zone's wider tolerance replaces
    lateral_tolerance_m in the alignment pause; the evidence then adds
    ``lateral_tolerance_zone``.
    """

    def __init__(self, pregrasp_pose_w, grasp_pose_w, *, max_speed_m_s=.015,
                 max_target_lead_m=.003, lateral_tolerance_m=.003,
                 rotation_tolerance_rad=.025, expected_contact_zone_m=.006,
                 unexpected_contact_force_n=2., max_acceleration_m_s2=.04,
                 guard_point_tcp_m=None, completion_position_m=None,
                 lateral_tolerance_zones=None, pressed_key_completion=None):
        self.start = pose_to_matrix(pregrasp_pose_w)
        self.goal = pose_to_matrix(grasp_pose_w)
        angle = math.acos(np.clip((np.trace(self.start[:3, :3].T @ self.goal[:3, :3]) - 1) / 2, -1, 1))
        delta = self.goal[:3, 3] - self.start[:3, 3]
        self.length = float(np.linalg.norm(delta))
        if angle > rotation_tolerance_rad or self.length < .01:
            raise ValueError('Insertion needs an already aligned pregrasp and at least 10 mm travel')
        values = [max_speed_m_s, max_target_lead_m, lateral_tolerance_m,
                  rotation_tolerance_rad, expected_contact_zone_m, unexpected_contact_force_n]
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError('Insertion rates, pose gates and force limits must be positive and finite')
        if max_target_lead_m >= self.length or expected_contact_zone_m >= self.length:
            raise ValueError('Lead and contact zone must be shorter than the insertion')
        if not math.isfinite(max_acceleration_m_s2) or max_acceleration_m_s2 <= 0:
            raise ValueError('Reference acceleration must be finite and positive')
        self.speed, self.lead, self.lateral_tolerance, self.rotation_tolerance, self.contact_zone, self.contact_limit = values
        self.acceleration = max_acceleration_m_s2
        self.axis = delta / self.length
        if completion_position_m is not None and not 0 < completion_position_m < math.inf:
            raise ValueError('The approach completion bound must be positive and finite')
        self.guard_point = None
        if guard_point_tcp_m is not None:
            point = np.asarray(guard_point_tcp_m, float)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError('The guard point needs three finite TCP-frame coordinates')
            self.guard_point = point
        self.zones = None
        if lateral_tolerance_zones is not None:
            self.zones = validate_lateral_tolerance_zones(lateral_tolerance_zones, self.lateral_tolerance,
                                                          "lateral_tolerance_zones")
        if pressed_key_completion is not None and not isinstance(pressed_key_completion, Mapping):
            raise ValueError('The pressed-key completion settings must be a mapping')
        self.progress = 0.
        self.velocity = 0.
        self.failure = None

    def step(self, *, tcp_pose_w, contact_force_w, dt, contact_available=True,
             contact_force_magnitude_n=None):
        if not math.isfinite(dt) or not 0 < dt <= .1:
            raise ValueError('Insertion dt must lie within (0, .1] seconds')
        tcp = pose_to_matrix(tcp_pose_w)
        if not isinstance(contact_available, (bool, np.bool_)):
            raise ValueError('Contact availability must be explicit boolean evidence')
        force = np.asarray(contact_force_w, float) if contact_available else np.zeros(3)
        if contact_available and (force.shape != (3,) or not np.isfinite(force).all()):
            raise ValueError('Measured world contact force must contain three finite components')
        if not contact_available:
            self.failure = self.failure or 'contact_evidence_unavailable'
        force_magnitude = float(np.linalg.norm(force))
        if contact_available and contact_force_magnitude_n is not None:
            force_magnitude = float(contact_force_magnitude_n)
            if not math.isfinite(force_magnitude) or force_magnitude < 0:
                raise ValueError('Summed jaw-force magnitude must be finite and nonnegative')
        delta = tcp[:3, 3] - self.start[:3, 3]
        measured_progress = float(delta @ self.axis)
        lateral = float(np.linalg.norm(delta - measured_progress * self.axis))
        if self.guard_point is not None:
            # Offset from the reference's point at the start (goal rotation, start position); its cross-track part.
            origin_lateral = lateral
            point_delta = delta + (tcp[:3, :3] - self.goal[:3, :3]) @ self.guard_point
            lateral = float(np.linalg.norm(point_delta - float(point_delta @ self.axis) * self.axis))
        rotation = math.acos(np.clip((np.trace(self.goal[:3, :3].T @ tcp[:3, :3]) - 1) / 2, -1, 1))
        tolerance = self.lateral_tolerance
        if self.zones is not None:
            remaining, zone = max(0., self.length - max(self.progress, measured_progress)), None
            for index, (distance, widened) in enumerate(self.zones):
                if remaining > distance:
                    tolerance, zone = widened, index
                    break
        aligned = lateral <= tolerance and rotation <= self.rotation_tolerance
        if measured_progress < self.length - self.contact_zone and force_magnitude > self.contact_limit:
            self.failure = self.failure or 'unexpected_approach_contact'
        old_velocity = self.velocity
        safety_override = False
        if aligned and not self.failure:
            budget = max(0., min(self.length - self.progress, measured_progress + self.lead - self.progress))
            # Reserve enough distance to brake before either the final goal or
            # the measured-motion lead bound. Discrete one-step lookahead is
            # included instead of relying on continuous stopping distance.
            braking_speed = max(0., math.sqrt((self.acceleration * dt)**2 + 2*self.acceleration*budget)
                                - self.acceleration * dt)
            desired = min(self.speed, braking_speed)
            self.velocity += float(np.clip(desired-self.velocity, -self.acceleration*dt, self.acceleration*dt))
            advance = min(self.velocity*dt, budget)
            self.velocity = advance / dt
            safety_override = abs(self.velocity-old_velocity) > self.acceleration*dt + 1e-10
            self.progress += advance
        else:
            # Safety holds override the normal acceleration bound. Resume from
            # zero, never silently advance while correcting a misaligned hand.
            self.velocity = 0.
            safety_override = old_velocity > self.acceleration*dt + 1e-10
        target = self.goal.copy()
        target[:3, 3] = self.start[:3, 3] + self.progress * self.axis
        result = {'status': 'failed' if self.failure else 'running', 'reason': self.failure,
                  'tcp_goal_pose_w': None if self.failure else matrix_to_pose(target),
                  'paused_for_alignment': not aligned, 'reference_progress_m': self.progress,
                  'measured_progress_m': measured_progress, 'lateral_error_m': lateral,
                  'rotation_error_rad': rotation, 'measured_contact_force_n': force_magnitude if contact_available else None,
                  'contact_evidence_available': bool(contact_available), 'reference_speed_m_s': self.velocity,
                  'acceleration_limit_overridden_for_safety': safety_override,
                  'full_task_completed': False, 'collision_free_execution_verified': False}
        if self.guard_point is not None:
            # Declared only: lateral_error_m is the guard point's; the origin's is kept for comparison.
            result.update(guard_point_tcp_m=self.guard_point.tolist(), origin_lateral_error_m=origin_lateral)
        if self.zones is not None:
            # Declared only: the alignment-pause tolerance this step used and why.
            result["lateral_tolerance_zone"] = {"remaining_distance_m": remaining, "zone_index": zone,
                                                "lateral_tolerance_m": tolerance}
        return result
