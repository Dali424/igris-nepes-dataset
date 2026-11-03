"""Opt-in rest-to-rest collinear TRANSFER handoff without reacquisition.

The existing FSM still measures every arrival and dwell. A new nominal segment
starts at the completed nominal endpoint, not the measured or corrected TCP.
Only the bounded object-correction filters and quaternion signs are carried.
Nominal path velocity starts at rest with the new segment's declared rates.
The explicit allow_online_extension option also admits a same-rate, collinear
extension before rest, preserving nominal physical velocity through the existing
online trajectory extension. Measured FSM arrival and dwell still apply.
"""
from collections.abc import Mapping
from copy import deepcopy
import math
import numpy as np

from .crs_manipulation import CrsPoseController
from .snapshot import matrix_to_pose, pose_to_matrix

KEY = 'transfer_completed_reference_continuity'
REPORT = 'completed_transfer_reference_handoff'
SCHEMA = 'igris_c_completed_transfer_continuity_v1'


def validate_settings(value, count):
    required = {'from_waypoint', 'through_waypoint', 'axis_w'}
    if (not isinstance(value, Mapping) or not required <= set(value)
            or set(value) - required - {'allow_online_extension'}):
        raise ValueError(KEY + ' requires from_waypoint, through_waypoint and axis_w')
    if 'allow_online_extension' in value and type(value['allow_online_extension']) is not bool:
        raise ValueError(KEY + '.allow_online_extension must be a boolean')
    first, last = value['from_waypoint'], value['through_waypoint']
    if (type(first) is not int or type(last) is not int or first < 1 or last < first
            or last != count - 1 or last - first + 1 > 8):
        raise ValueError(KEY + ' requires at most the final eight consecutive noninitial waypoints')
    axis = value['axis_w']
    if (not isinstance(axis, (list, tuple)) or len(axis) != 3
            or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float))
                   or not math.isfinite(x) for x in axis)
            or list(axis) != [0., 0., -1.]):
        raise ValueError(KEY + ' initially supports only descending world-Z segments')
    result = {'from_waypoint': first, 'through_waypoint': last, 'axis_w': [0., 0., -1.]}
    if 'allow_online_extension' in value:
        result['allow_online_extension'] = value['allow_online_extension']
    return result


def validate_task(task, *, upright_pickup):
    settings = validate_settings(task[KEY], len(task.get('object_waypoint_offsets_w', {}).get('transfer', [])))
    control = task.get('object_pose_control', {})
    precision = task.get('transfer_arrival_precision', {})
    if (not upright_pickup or control.get('reference_profile') != 'online'
            or not isinstance(control.get('object_feedback'), Mapping)
            or 'transfer_reference_continuity' in task
            or precision.get('from_waypoint', settings['from_waypoint']) > settings['from_waypoint'] - 1
            or precision.get('position_m', float('inf')) > .001
            or precision.get('rotation_rad', float('inf')) > .003):
        raise ValueError(KEY + ' requires existing upright online feedback and final precision gates, without blended transfer')
    offsets = task['object_waypoint_offsets_w']['transfer']
    rotations = task.get('object_waypoint_rotations_w', {}).get('transfer')
    if rotations is None or len(rotations) != len(offsets):
        raise ValueError(KEY + ' requires explicit rotations for every transfer goal')
    from .upright_runtime import object_pose_settings
    previous_feedback = None
    for i in range(settings['from_waypoint'] - 1, settings['through_waypoint'] + 1):
        effective = object_pose_settings(task, 'transfer', i)
        if effective.get('reference_profile') != 'online' or not effective.get('object_feedback'):
            raise ValueError(KEY + ' requires online feedback in every selected segment')
        if previous_feedback is not None and effective['object_feedback'] != previous_feedback:
            raise ValueError(KEY + ' cannot change feedback filter limits')
        previous_feedback = effective['object_feedback']
        if i < settings['from_waypoint']: continue
        d = np.asarray(offsets[i], float) - np.asarray(offsets[i-1], float)
        if not np.isfinite(d).all() or d[2] >= -1e-12 or np.linalg.norm(d[:2]) > 1e-9:
            raise ValueError(KEY + ' requires strictly descending collinear authored goals')
        prior = pose_to_matrix([0., 0., 0., *rotations[i-1]])
        current = pose_to_matrix([0., 0., 0., *rotations[i]])
        if not np.allclose(prior[:3,:3], current[:3,:3], atol=1e-10, rtol=0.):
            raise ValueError(KEY + ' cannot rotate an authored selected segment')
    task[KEY] = settings


def contract(task):
    if KEY not in task: return None
    result = dict(schema_version=SCHEMA, settings=deepcopy(task[KEY]),
                stage='transfer', previous_nominal_reference_must_be_complete=True,
                nominal_start='previous_nominal_goal_endpoint', nominal_velocity_restart='rest',
                preserved='bounded_filter_values_velocities_accelerations_and_quaternion_signs',
                acquisition_and_control_G_changed=False, measured_arrival_and_dwell_changed=False,
                rates_bounds_and_physical_gates_changed=False, rejection='original_measured_TCP_reset',
                geometry_or_native_validation_claim=False)
    if task[KEY].get('allow_online_extension', False):
        result.update(previous_nominal_reference_must_be_complete=False,
            nominal_start="previous_nominal_reference_for_online_extension_otherwise_completed_endpoint",
            nominal_velocity_restart='preserved_only_for_same_rate_collinear_extension_otherwise_completed_rest',
            online_extension=dict(same_effective_settings_required=True,
                prior_path_must_be_nonrotating_and_collinear=True,
                speed_and_stopping_distance_checked=True,
                implementation='CrsPoseController.retarget_collinear_goal_on_deep_copy',
                original_measured_arrival_and_dwell_required=True))
    return result


def context(stage, index, arm, object_path, grasp, control_grasp, settings, now):
    return dict(stage=stage, waypoint_index=index, arm=arm, object_path=object_path,
                grasp=deepcopy(grasp), control_grasp=deepcopy(control_grasp),
                settings=deepcopy(settings), measurement_time_s=now)


def offer(controller, previous, current, goal, declared, controller_factory=CrsPoseController):
    """Return (candidate or None, evidence); rejection never changes old state."""
    def reject(reason): return None, dict(accepted=False, reason=reason)
    index = current['waypoint_index']
    if not declared['from_waypoint'] <= index <= declared['through_waypoint']:
        return reject('outside_declared_range')
    if controller is None or previous is None: return reject('initial_segment')
    if previous['stage'] != 'transfer' or current['stage'] != 'transfer': return reject('stage_boundary')
    if index != previous['waypoint_index'] + 1: return reject('nonadjacent_waypoint')
    for key in ('arm', 'object_path'):
        if previous[key] != current[key]: return reject(key + '_changed')
    for key in ('grasp', 'control_grasp'):
        if not np.allclose(previous[key], current[key], atol=1e-12, rtol=0.): return reject(key + '_changed')
    dt = current['measurement_time_s'] - previous['measurement_time_s']
    if not math.isfinite(dt) or abs(dt - .02) > 1e-9: return reject('measurement_not_fresh_20ms')
    old_settings, new_settings = previous['settings'], current['settings']
    if old_settings.get('object_feedback') != new_settings.get('object_feedback'): return reject('feedback_limits_changed')
    # Only nominal speed/acceleration may change at a completed stop.
    rate_keys={'linear_speed','linear_acceleration','angular_speed','angular_acceleration'}
    if {k:v for k,v in old_settings.items() if k not in rate_keys} != {k:v for k,v in new_settings.items() if k not in rate_keys}:
        return reject('nonrate_settings_changed')
    if (controller.reference_profile != 'online' or controller._two_speed is not None
            or controller._symmetry_axis is not None or controller.object_feedback is None):
        return reject('unsupported_controller')
    completed_at_rest = not (controller._progress != 1. or controller._paused
            or abs(controller._path_velocity) > 1e-12
            or not np.allclose(controller._reference, controller._goal_tcp, atol=1e-10, rtol=0.))
    online_extension = bool(declared.get('allow_online_extension', False) and not completed_at_rest)
    if not completed_at_rest and not online_extension:
        return reject('prior_nominal_reference_not_complete_at_rest')
    if online_extension and controller._paused:
        return reject('prior_nominal_reference_paused')
    if online_extension and old_settings != new_settings:
        return reject('online_extension_rates_changed')
    try: destination = pose_to_matrix(goal)
    except (ValueError, TypeError, OverflowError): return reject('invalid_goal')
    if not np.allclose(destination[:3,:3], controller._goal_object[:3,:3], atol=1e-10, rtol=0.):
        return reject('goal_rotation_changed')
    displacement = destination[:3,3] - controller._goal_object[:3,3]
    axis = np.asarray(declared['axis_w'], float); forward = float(displacement @ axis)
    if not np.isfinite(displacement).all() or forward <= 1e-12 or np.linalg.norm(displacement - forward*axis) > 1e-9:
        return reject('goal_not_forward_collinear')
    if online_extension:
        candidate = deepcopy(controller)
        extension = candidate.retarget_collinear_goal(goal)
        if not extension['accepted']:
            return reject('online_extension_' + extension['reason'])
        evidence = dict(accepted=True, reason='same_rate_collinear_online_extension',
            previous_waypoint=previous['waypoint_index'], waypoint=index,
            nominal_start_pose_w=matrix_to_pose(controller._reference),
            forward_distance_m=forward, extension=extension,
            correction_translation_w_m=candidate._position_correction.value.tolist(),
            correction_rotation_vector_w_rad=candidate._rotation_correction.value.tolist(),
            correction_linear_velocity_w_m_s=candidate._position_correction.velocity.tolist(),
            correction_rotation_vector_velocity_w_rad_s=candidate._rotation_correction.velocity.tolist(),
            correction_linear_acceleration_w_m_s2=candidate._position_correction.acceleration.tolist(),
            correction_rotation_vector_acceleration_w_rad_s2=candidate._rotation_correction.acceleration.tolist(),
            corrected_target_reused_as_nominal=False, acquisition_or_control_G_changed=False)
        return candidate, evidence
    candidate = controller_factory(current['grasp'], matrix_to_pose(controller._reference), goal,
                    reference_object_to_tcp=current['control_grasp'], **new_settings)
    if candidate.object_feedback != controller.object_feedback: return reject('effective_feedback_limits_changed')
    for name in ('_position_correction', '_rotation_correction', '_previous_quaternion', '_previous_command_quaternion'):
        setattr(candidate, name, deepcopy(getattr(controller, name)))
    evidence = dict(accepted=True, reason='completed_nominal_collinear_handoff',
                    previous_waypoint=previous['waypoint_index'], waypoint=index,
                    nominal_start_pose_w=matrix_to_pose(controller._reference), forward_distance_m=forward,
                    nominal_velocity_start=0., correction_translation_w_m=candidate._position_correction.value.tolist(),
                    correction_rotation_vector_w_rad=candidate._rotation_correction.value.tolist(),
                    correction_linear_velocity_w_m_s=candidate._position_correction.velocity.tolist(),
                    correction_rotation_vector_velocity_w_rad_s=candidate._rotation_correction.velocity.tolist(),
                    correction_linear_acceleration_w_m_s2=candidate._position_correction.acceleration.tolist(),
                    correction_rotation_vector_acceleration_w_rad_s2=candidate._rotation_correction.acceleration.tolist(),
                    corrected_target_reused_as_nominal=False, acquisition_or_control_G_changed=False)
    return candidate, evidence


class CompletedTransferAdmission:
    """Reconstruct declared TRANSFER references from raw measured rows.

    Runs only for the opt-in. Checks the entire TRANSFER history to independently
    recover the filter state used by the first selected handoff. This is an
    evidence check, not a new completion gate or geometric acceptance exception.
    """
    def __init__(self, task, equal):
        self.task, self.equal = task, equal
        self.settings = task.get(KEY)
        self.controller = self.previous = self.segment = self.event = None
        self.last_time = None
        self.seen = set()
        from .upper_body_feedforward import UpperBodyTargetMotionFeedforward
        self.feedforward = UpperBodyTargetMotionFeedforward(task.get('upper_body_ik', {}).get('target_motion_feedforward_stages', []))

    def check(self, row, previous_stage=None):
        observation, command = row['observation'], row['command']
        if self.settings is None:
            if REPORT in command: raise ValueError('completed_transfer_handoff_undeclared')
            return
        now = observation['time_s'];stage = command.get('stage_id')
        dt = .02 if self.last_time is None else now - self.last_time
        self.last_time = now
        if stage != 'transfer':
            if REPORT in command: raise ValueError('completed_transfer_handoff_outside_transfer')
            self.controller = self.previous = self.segment = self.event = None
            self.feedforward.reset()
            return
        if abs(dt - .02) > 1e-9: raise ValueError('completed_transfer_handoff_nonfresh_trace')
        arm, index = command['arm'], command['object_waypoint_index']
        object_path = command['object_path']
        grasp, control_grasp = command['retention_object_to_tcp'], command['control_object_to_tcp']
        from .upright_runtime import object_pose_settings
        from .crs_manipulation import goal_object_from_tcp
        settings = object_pose_settings(self.task, stage, index)
        current = context(stage, index, arm, object_path, grasp, control_grasp, settings, now)
        scoped = self.settings['from_waypoint'] <= index <= self.settings['through_waypoint']
        if self.segment != (stage, index):
            goal = goal_object_from_tcp(command['fsm_tcp_goal_pose_w'], control_grasp)
            candidate = None;self.event = None
            if scoped:
                candidate, decision = offer(self.controller, self.previous, current, goal, self.settings)
                self.event = dict(schema_version=SCHEMA, measurement_time_s=now,
                                  previous_segment=None if self.segment is None else list(self.segment), current_segment=[stage,index],
                                  controller_state_preserved=decision['accepted'], decision=decision,
                                  measured_arrival_and_dwell_changed=False)
                self.seen.add(index)
            if candidate is None:
                candidate = CrsPoseController(grasp, observation['tcp_poses_w'][arm], goal,
                    reference_object_to_tcp=control_grasp, object_symmetry_axis=command.get('object_symmetry_axis_object'),
                    object_pivot_radius_m=command.get('object_symmetry_pivot_radius_m'), **settings)
            self.controller = candidate;self.segment = (stage,index)
        if scoped:
            if not self.equal(command.get(REPORT), self.event):
                raise ValueError('completed_transfer_handoff_replayed_event_mismatch')
        elif REPORT in command:
            raise ValueError('completed_transfer_handoff_outside_declared_range')
        stored = command.get('object_pose_reference')
        if not isinstance(stored, Mapping) or stored.get('external_hold_reason') is not None:
            raise ValueError('completed_transfer_handoff_reference_unavailable_or_external_hold')
        expected = self.controller.update(observation['object_poses_w'][object_path], observation['tcp_poses_w'][arm], .02,
            pinch_confirmed=bool(observation['hand_closed'][arm] and object_path in observation['contacts'][arm]),
            advance_hold_reason=command.get('object_advance_hold_reason'), feedback_hold_reason=command.get('object_feedback_hold_reason'))
        if not self.equal(stored, expected): raise ValueError('completed_transfer_handoff_reference_replay_mismatch')
        if not self.equal(command.get('tcp_target_pose_w'), expected['tcp_target_pose_w']):
            raise ValueError('completed_transfer_handoff_command_mismatch')
        ff_command=dict(command,object_pose_reference=expected)
        if self.event is not None: ff_command[REPORT]=self.event
        self.feedforward.update(stage,ff_command,arm,expected['tcp_target_pose_w'],.02)
        expected_feedforward = (None if self.feedforward.last_result is None else
                                {**self.feedforward.last_result, 'measurement_time_s': now})
        if expected_feedforward is not None and not self.equal(command.get('upper_body_target_motion_feedforward'), expected_feedforward):
            raise ValueError('completed_transfer_handoff_feedforward_replay_mismatch')
        self.previous=current

    def finish(self):
        if self.settings is not None and self.seen != set(range(self.settings['from_waypoint'],self.settings['through_waypoint']+1)):
            raise ValueError('completed_transfer_handoff_incomplete_scope')
