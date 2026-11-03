"""Causal raw admission for opt-in carry/place/rest applied-target holds."""
from collections.abc import Mapping
from copy import deepcopy
import math
import numpy as np

from .posture_hold import (HOLD_MODE, applied_hold_targets, canonical_upper_command_limits,
                           hold_anchor, is_applied_hold)
from .upright_runtime import PostureStageExecutor
from .snapshot import pose_to_matrix
from .nepes_tasks import _errors
from .stationary_lift import _bilateral_contact
from .stationary_place import _contact_absent


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _joints(value, names, label):
    try:
        return applied_hold_targets(value, names)
    except ValueError as exc:
        raise ValueError('pickup_applied_hold_missing_' + label) from exc


class AppliedHoldAdmission:
    """Derive immutable goals from raw prior commands, not claimed controller goals.

    Every existing episode admission rule still applies. This additional replay
    never turns a failed/diagnostic attempt into an accepted demonstration.
    """
    def __init__(self, task, spec, object_path, thresholds, equal, *, jig_path=None):
        self.paths = {s:p for s,p in task.get('posture_paths', {}).items() if is_applied_hold(p)}
        self.enabled = bool(self.paths)
        # Stage neighbours come from the profile's own sequence (an opt-in handover inserts stages after carry_posture).
        from .nepes_tasks import task_stages
        self.sequence = tuple(task_stages(task)) if self.enabled else ()
        self.task, self.spec, self.object_path = task, spec, object_path
        self.jig_path = jig_path
        self.place_arm = task.get('handover', {}).get('receiving_arm', task['arm'])
        if 'rest_posture' in self.paths and not isinstance(jig_path, str):
            raise ValueError('pickup_applied_hold_missing_configured_jig_path')
        self.thresholds, self.equal = thresholds, equal
        from .object_symmetry import symmetry_axis, symmetry_pivot_radius
        self.symmetry_axis = symmetry_axis(task)  # declared ring symmetry: capture/slot rotations ignore spin
        self.symmetry_pivot_radius_m = symmetry_pivot_radius(task)  # capture slip measured at the pinch when declared
        self.previous_time = self.previous_new_targets = self.capture = None
        self.anchors, self.executors, self.since, self.completed = {}, {}, {}, set()
        self.position_samples = {}
        self.limits = canonical_upper_command_limits() if self.enabled else None

    def _same(self, actual, expected, label):
        if not self.equal(actual, expected):
            raise ValueError('pickup_applied_hold_' + label + '_mismatch')

    def _grasp(self, observation):
        arm = self.task['arm']
        opened, closed = observation.get('hand_open', {}), observation.get('hand_closed', {})
        contacts = observation.get('contacts', {})
        if (not isinstance(opened, Mapping) or not isinstance(closed, Mapping)
                or type(opened.get(arm)) is not bool or type(closed.get(arm)) is not bool
                or opened[arm] and closed[arm] or not isinstance(contacts, Mapping)
                or not isinstance(contacts.get(arm), list)
                or any(not isinstance(p, str) or not p.startswith('/') for p in contacts[arm])):
            raise ValueError('pickup_applied_hold_malformed_measured_hand_state')
        if (self.capture is None or closed[arm] is not True
                or self.object_path not in contacts[arm]
                or not _bilateral_contact(observation, arm, self.object_path)):
            return False
        relative = (np.linalg.inv(pose_to_matrix(observation['object_poses_w'][self.object_path]))
                    @ pose_to_matrix(observation['tcp_poses_w'][arm]))
        if self.symmetry_axis is None:
            position, rotation = _errors(relative, self.capture)
        else:
            from .object_symmetry import symmetric_capture_errors
            position, rotation, _ = symmetric_capture_errors(relative, self.capture, self.symmetry_axis,
                                                             self.symmetry_pivot_radius_m)
        return (position <= self.thresholds['capture_position_m']
                and rotation <= self.thresholds['capture_rotation_rad'])

    def _arrival(self, observation, stage):
        path, run = self.paths[stage], self.executors[stage]
        names, now = path['joint_names'], observation['time_s']
        measured = _joints(observation.get('joint_positions'), names, 'measured_joints')
        applied = _joints(observation.get('commanded_upper_joint_positions'), names, 'previous_applied_targets')
        velocities = _joints(observation.get('joint_velocities'), names, 'joint_velocities')
        reference = observation.get('posture_reference')
        self._same(reference, run.status, 'previous_reference')
        goal = self.anchors[stage]['joint_positions']
        self._same(applied, goal, 'immutable_applied_targets')
        errors = {n:abs(measured[n] - goal[n]) for n in names}
        worst = max(errors, key=errors.get)
        readback = max(abs(v) for v in velocities.values())
        source = path.get('settle_speed_source')
        speed = readback
        if source == 'measured_positions':
            # Independently reconstruct the runtime's consecutive measured-position
            # derivative. Entry does not count as a sample; the first arrival
            # observation has no speed evidence and cannot begin the dwell.
            previous = self.position_samples.get(stage)
            self.position_samples[stage] = (now, dict(measured))
            speed = (None if previous is None else
                     max(abs(measured[n] - previous[1][n]) for n in names) / (now - previous[0]))
        tracking = max(abs(measured[n] - applied[n]) for n in names)
        velocity = observation.get('base_velocity_w')
        if not isinstance(velocity, list) or len(velocity) != 6 or not all(map(_number, velocity)):
            raise ValueError('pickup_applied_hold_base_velocity_missing')
        quiet = (float(np.linalg.norm(velocity[:3])) <= self.spec['max_base_linear_m_s']
                 and abs(velocity[5]) <= self.spec['max_base_yaw_rad_s'])
        # A released rest holds the last actuator targets, not the object.
        # Its original FSM gates differ from the carrying holds and must be
        # reconstructed from raw observations rather than claimed metrics.
        retained = None if stage == 'rest_posture' else self._grasp(observation)
        physical = self._released(observation) if stage == 'rest_posture' else retained
        complete = bool(reference['stage_id'] == stage and reference['status'] == 'running'
                        and reference['complete'] is True and reference['reference_mode'] == HOLD_MODE
                        and reference['final_joint_positions'] == goal)
        condition = bool(complete and errors[worst] <= path['arrival_tolerance_rad']
                         and speed is not None and speed <= path['settle_velocity_rad_s'] and quiet and physical)
        if tracking > path['max_tracking_error_rad']:
            raise ValueError('pickup_applied_hold_posture_tracking_error')
        if condition:
            if self.since[stage] is None: self.since[stage] = now
        else:
            self.since[stage] = None
        dwell = 0. if self.since[stage] is None else now - self.since[stage]
        evidence = {'stage_id':stage, 'reference_complete':complete,
            'reference_elapsed_s':reference['elapsed_s'], 'reference_duration_s':reference['duration_s'],
            'max_arrival_error_rad':errors[worst], 'max_arrival_error_joint':worst,
            'max_joint_speed_rad_s':speed, 'max_tracking_error_rad':tracking,
            'arrival_tolerance_rad':path['arrival_tolerance_rad'],
            'settle_velocity_rad_s':path['settle_velocity_rad_s'],
            'tracking_error_limit_rad':path['max_tracking_error_rad'],
            'base_quiet':quiet, 'grasp_retained':retained,
            'required_dwell_s':path['settle_dwell_s'], 'continuous_dwell_s':dwell,
            'measurement_time_s':now, 'hold_reference':deepcopy(self.anchors[stage])}
        if source is not None:
            evidence['settle_speed_source'] = source
            evidence['max_joint_velocity_readback_rad_s'] = readback
        return evidence, condition and dwell >= path['settle_dwell_s'] - 1e-10

    def _released(self, observation):
        """The unchanged released-rest hand and measured JIG-relative pose gates."""
        opened, closed = observation.get('hand_open', {}), observation.get('hand_closed', {})
        if (not isinstance(opened, Mapping) or not isinstance(closed, Mapping)
                or opened.get(self.place_arm) is not True or closed.get(self.place_arm) is not False
                or not all(_contact_absent(observation, arm, self.object_path) for arm in ('left', 'right'))):
            return False
        poses = observation.get('object_poses_w', {})
        if not isinstance(poses, Mapping):
            return False
        try:
            relative = (np.linalg.inv(pose_to_matrix(poses[self.jig_path]))
                        @ pose_to_matrix(poses[self.object_path]))
            slot = pose_to_matrix(self.task['target']['pose'])
            if self.symmetry_axis is None:
                position, rotation = _errors(relative, slot)
            else:
                from .object_symmetry import symmetric_pose_errors
                position, rotation, _ = symmetric_pose_errors(relative, slot, self.symmetry_axis)
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
            return False
        return bool(np.isfinite(relative).all()
                    and position <= self.spec['slot_position_tolerance_m']
                    and rotation <= self.spec['slot_rotation_tolerance_rad']
                    and abs(float(relative[1, 3] - slot[1, 3])) <= self.spec['slot_normal_position_tolerance_m'])

    def check(self, row, previous_stage):
        if not self.enabled: return
        observation, command = row['observation'], row['command']
        now, stage = observation.get('time_s'), command.get('stage_id')
        if not _number(now) or self.previous_time is not None and now <= self.previous_time:
            raise ValueError('pickup_applied_hold_clock_invalid')
        dt = None if self.previous_time is None else now - self.previous_time
        previous_targets = self.previous_new_targets
        self.previous_time, self.previous_new_targets = now, deepcopy(row.get('commanded_upper_joint_positions'))
        if stage == 'lift' and previous_stage == 'close':
            if self.capture is not None: raise ValueError('pickup_applied_hold_capture_replaced')
            self.capture = (np.linalg.inv(pose_to_matrix(observation['object_poses_w'][self.object_path]))
                            @ pose_to_matrix(observation['tcp_poses_w'][self.task['arm']]))
            if not self._grasp(observation): raise ValueError('pickup_applied_hold_capture_not_physical')
        if stage in self.paths and stage != previous_stage:
            if stage in self.anchors: raise ValueError('pickup_applied_hold_reentered_stage')
            expected_previous = self.sequence[self.sequence.index(stage) - 1]
            if previous_stage != expected_previous: raise ValueError('pickup_applied_hold_entry_missing')
            path = self.paths[stage]
            names = path['joint_names']
            applied = _joints(observation.get('commanded_upper_joint_positions'), names, 'previous_applied_targets')
            self._same(applied, _joints(previous_targets, names, 'previous_trace_new_targets'), 'causal_entry_targets')
            self.anchors[stage] = hold_anchor(applied, names, now)
            self.executors[stage] = PostureStageExecutor(stage, path, start_targets=applied,
                measured_positions=observation.get('joint_positions'), command_names=names, joint_limits=self.limits)
            self.since[stage] = None
        self._same(command.get('posture_hold_references'), self.anchors, 'immutable_anchors')
        relevant = previous_stage if previous_stage in self.paths else stage
        if relevant not in self.paths: return
        # The live FSM fails immediately on lost release or displaced placement,
        # including entry/exit observations; it does not merely restart a dwell.
        if relevant == 'rest_posture' and not self._released(observation):
            raise ValueError('pickup_applied_hold_released_state_invalid')
        if command.get('status') != 'running': raise ValueError('pickup_applied_hold_failed_or_diagnostic_control')
        if observation.get('faults') or observation.get('unreachable') is True:
            raise ValueError('pickup_applied_hold_raw_runtime_fault')
        if dt is None or abs(dt - .02) > 1e-9 or dt > self.thresholds['max_observation_gap_s']:
            raise ValueError('pickup_applied_hold_control_interval_invalid')
        if previous_stage in self.paths:
            expected, passed = self._arrival(observation, previous_stage)
            self._same(command.get('posture_evidence', {}).get(previous_stage), expected, 'measured_arrival')
            if (stage != previous_stage) != passed:
                raise ValueError('pickup_applied_hold_measured_dwell_transition_mismatch')
            if stage != previous_stage:
                if stage != self.sequence[self.sequence.index(previous_stage) + 1]:
                    raise ValueError('pickup_applied_hold_unexpected_next_stage')
                self.completed.add(previous_stage)
                return
        names, run = self.paths[stage]['joint_names'], self.executors[stage]
        measured = _joints(observation.get('joint_positions'), names, 'measured_joints')
        applied = _joints(observation.get('commanded_upper_joint_positions'), names, 'previous_applied_targets')
        targets, reference = run.update(measured, applied, .02)
        if targets is None: raise ValueError('pickup_applied_hold_replayed_controller_failed')
        self._same(_joints(row.get('commanded_upper_joint_positions'), names, 'new_applied_targets'), targets, 'held_targets')
        self._same(command.get('posture_reference'), reference, 'reference')

    def finish(self):
        if self.enabled and self.completed != set(self.paths):
            raise ValueError('pickup_applied_hold_measured_transitions_missing')
