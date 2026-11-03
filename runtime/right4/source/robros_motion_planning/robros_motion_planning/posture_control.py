"""Upper-body posture references and foot-preserving boundary initialization.

These are joint commands and reset mathematics, not a balance controller. The
live HugWBC policy still controls the free base and legs. Never apply the reset
transform during a demonstration.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
import numpy as np

from .snapshot import matrix_to_pose, pose_to_matrix


def resolve_auxiliary_commands(command_names, auxiliary_names, policy_names, teleop_names):
    """Route only previously unowned upper commands; never override policy joints."""
    policy, teleop, auxiliary = set(policy_names), set(teleop_names), set(auxiliary_names)
    if auxiliary & (policy | teleop):
        raise ValueError("Auxiliary and action-manager joint ownership overlap")
    missing = set(command_names) - (policy | teleop | auxiliary)
    if not missing <= {"Joint_Waist_Roll", "Joint_Waist_Pitch"}:
        raise ValueError("Unsupported upper command without actuator routing")
    return tuple(auxiliary_names) + tuple(n for n in command_names if n in missing)


def pelvis_preserving_boundary_pose(model, base_pose_w, old_joints, new_joints):
    pelvis = "Link_Waist_Pitch"
    old = model.forward_kinematics(old_joints)[pelvis]
    new = model.forward_kinematics(new_joints)[pelvis]
    return matrix_to_pose(pose_to_matrix(base_pose_w) @ old @ np.linalg.inv(new))


class PostureReference:
    """A zero-endpoint-speed quintic joint reference with bounded rates."""

    def __init__(self, start, goal, *, minimum_duration_s=3., max_speed_rad_s=.08,
                 max_acceleration_rad_s2=.16):
        values = (minimum_duration_s, max_speed_rad_s, max_acceleration_rad_s2)
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Posture durations and limits must be positive and finite")
        if not goal or not set(goal) <= set(start):
            raise ValueError("Posture goal needs measured initial joint values")
        self.start = {n: float(start[n]) for n in goal}
        self.goal = {n: float(v) for n, v in goal.items()}
        if not all(math.isfinite(v) for v in (*self.start.values(), *self.goal.values())):
            raise ValueError("Posture joints must be finite")
        delta = max(abs(self.goal[n] - self.start[n]) for n in goal)
        self.duration = max(minimum_duration_s, 1.875 * delta / max_speed_rad_s,
                            math.sqrt(10 / math.sqrt(3) * delta / max_acceleration_rad_s2))
        self.elapsed = 0.

    def update(self, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Posture dt must be positive and finite")
        self.elapsed = min(self.elapsed + dt, self.duration)
        u = self.elapsed / self.duration
        blend = u**3 * (10 + u * (-15 + 6*u))
        return {n: self.start[n] + blend * (self.goal[n] - self.start[n]) for n in self.goal}


def _finite_joint_value(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _segment_caps(values, limit, count, label):
    """Optional per-segment caps (runtime_v26): None, or one finite value in (0, limit] per segment."""
    if values is None:
        return None
    if not isinstance(values, (list, tuple, np.ndarray)) or np.ndim(values) != 1 or len(values) != count:
        raise ValueError(f"Joint path {label} needs one cap per waypoint segment")
    caps = tuple(_finite_joint_value(value, f"Joint path {label}") for value in values)
    if any(not 0. < cap <= limit for cap in caps):
        raise ValueError(f"Joint path {label} must be positive and at most the path limit {limit:g}")
    return caps


class JointPathReference:
    """Stop-and-go synchronized quintic through joint waypoints (zero vel/acc at each waypoint).

    Each segment moves every joint on one shared quintic blend, so the per-joint
    peaks are 1.875*|delta|/T (speed) and 10/sqrt(3)*|delta|/T**2 (acceleration)
    at u=0.5 and u=(3-sqrt(3))/6. Segment durations make the largest joint delta
    respect both limits. Only ``update`` advances the clock; no wall time is read.
    The reference is a command; measured arrival is the caller's responsibility.
    Optional ``segment_max_speeds_rad_s``/``segment_max_accelerations_rad_s2`` (runtime_v26) give one cap per
    segment, each positive and at most the path limit; a segment then uses its own cap. Absent, every segment
    uses the path limits exactly as before.
    """

    def __init__(self, start, waypoints, *, joint_names, max_speed_rad_s, max_acceleration_rad_s2,
                 segment_min_durations_s, segment_max_speeds_rad_s=None, segment_max_accelerations_rad_s2=None):
        if isinstance(joint_names, (str, bytes)):
            raise ValueError("Joint path names must be a sequence of joint names")
        names = tuple(joint_names)
        if (not names or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("Joint path needs distinct nonempty joint names")
        speed = _finite_joint_value(max_speed_rad_s, "Joint path max_speed_rad_s")
        acceleration = _finite_joint_value(max_acceleration_rad_s2, "Joint path max_acceleration_rad_s2")
        if speed <= 0 or acceleration <= 0:
            raise ValueError("Joint path speed and acceleration limits must be positive")
        if not isinstance(start, Mapping) or not set(names) <= set(start):
            raise ValueError("Joint path start must provide every path joint")
        if not isinstance(waypoints, (list, tuple, np.ndarray)) or len(waypoints) == 0:
            raise ValueError("Joint path needs at least one waypoint")
        points = [[_finite_joint_value(start[name], f"Joint path start {name}") for name in names]]
        for waypoint in waypoints:
            if isinstance(waypoint, Mapping):
                if set(waypoint) != set(names):
                    raise ValueError("Joint path waypoint names must match joint_names exactly")
                values = [waypoint[name] for name in names]
            elif (isinstance(waypoint, (list, tuple, np.ndarray)) and np.ndim(waypoint) == 1
                    and len(waypoint) == len(names)):
                values = list(waypoint)
            else:
                raise ValueError("Joint path waypoint must align with joint_names")
            points.append([_finite_joint_value(value, f"Joint path waypoint {name}")
                           for name, value in zip(names, values)])
        if not isinstance(segment_min_durations_s, (list, tuple, np.ndarray)):
            raise ValueError("Joint path needs one minimum duration per waypoint")
        minimums = [_finite_joint_value(value, "Joint path segment minimum duration")
                    for value in segment_min_durations_s]
        if len(minimums) != len(points) - 1 or any(value <= 0 for value in minimums):
            raise ValueError("Joint path needs one positive minimum duration per waypoint")
        self.segment_max_speeds_rad_s = _segment_caps(segment_max_speeds_rad_s, speed, len(minimums),
                                                      "segment_max_speeds_rad_s")
        self.segment_max_accelerations_rad_s2 = _segment_caps(segment_max_accelerations_rad_s2, acceleration,
                                                              len(minimums), "segment_max_accelerations_rad_s2")
        self.joint_names = names
        self.max_speed_rad_s, self.max_acceleration_rad_s2 = speed, acceleration
        self._points = np.asarray(points, dtype=float)
        peaks = np.max(np.abs(np.diff(self._points, axis=0)), axis=1)
        speeds = self.segment_max_speeds_rad_s or (speed,) * len(minimums)
        accelerations = self.segment_max_accelerations_rad_s2 or (acceleration,) * len(minimums)
        self.segment_durations = tuple(
            max(minimum, 1.875 * float(delta) / cap, math.sqrt(10 / math.sqrt(3) * float(delta) / cap_acceleration))
            for minimum, delta, cap, cap_acceleration in zip(minimums, peaks, speeds, accelerations))
        self._boundaries = np.concatenate(([0.], np.cumsum(self.segment_durations)))
        self._elapsed = 0.

    @property
    def duration(self):
        return float(self._boundaries[-1])

    @property
    def elapsed(self):
        return self._elapsed

    @property
    def complete(self):
        return self._elapsed >= self.duration

    @property
    def segment_index(self):
        """Index of the waypoint segment being executed; the last one once complete."""
        index = int(np.searchsorted(self._boundaries, self._elapsed, side="right")) - 1
        return min(max(index, 0), len(self.segment_durations) - 1)

    @property
    def start(self):
        return dict(zip(self.joint_names, map(float, self._points[0])))

    @property
    def final(self):
        return dict(zip(self.joint_names, map(float, self._points[-1])))

    def update(self, dt):
        """Advance by ``dt`` seconds and return the commanded targets for joint_names."""
        dt = _finite_joint_value(dt, "Joint path dt")
        if dt <= 0:
            raise ValueError("Joint path dt must be positive and finite")
        self._elapsed = min(self._elapsed + dt, self.duration)
        if self.complete:
            return self.final
        index = self.segment_index
        u = (self._elapsed - self._boundaries[index]) / self.segment_durations[index]
        blend = u**3 * (10 + u * (-15 + 6*u))
        first, second = self._points[index], self._points[index + 1]
        return dict(zip(self.joint_names, map(float, first + blend * (second - first))))


class BoundedPostureFeedback:
    """Adjust position targets using measured error, without changing actuators.

    Initialize from the last commanded targets to retain their supporting
    torque. The stored command is the only integral state, so saturation cannot
    accumulate a hidden error. Limits supplied here must include both physical
    joint limits and any narrower controller limits.
    """

    def __init__(self, initial_commands, joint_limits, *, gain_s_inv=.5,
                 max_command_speed_rad_s=.08, tracking_envelope_rad=.2):
        if not initial_commands:
            raise ValueError("Posture feedback needs at least one controlled joint")
        settings = {"gain": gain_s_inv, "command speed": max_command_speed_rad_s,
                    "tracking envelope": tracking_envelope_rad}
        if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0
               for v in settings.values()):
            raise ValueError("Posture feedback settings must be positive and finite")
        if max_command_speed_rad_s > .08 or tracking_envelope_rad > .2:
            raise ValueError("Posture feedback speed/envelope cannot exceed .08 rad/s and .2 rad")
        self.names = tuple(initial_commands)
        self.gain = float(gain_s_inv)
        self.speed = float(max_command_speed_rad_s)
        self.envelope = float(tracking_envelope_rad)
        self.limits, self.commands = {}, {}
        for name in self.names:
            if name not in joint_limits or len(joint_limits[name]) != 2:
                raise ValueError("Posture joint needs two finite limits: " + name)
            lo, hi = map(float, joint_limits[name])
            command = float(initial_commands[name])
            if not all(map(math.isfinite, (lo, hi, command))) or lo >= hi or not lo <= command <= hi:
                raise ValueError("Invalid posture initial command or joint bounds: " + name)
            self.limits[name] = (lo, hi)
            self.commands[name] = command

    def update(self, reference_positions, measured_positions, dt):
        """Return (commands, evidence), committing state only after all checks.

        A discontinuous measurement can make the rate and tracking bounds
        incompatible. This raises without mutating state; callers must stop
        the stage instead of silently violating either bound.
        """
        if isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Posture feedback dt must be positive and finite")
        if set(reference_positions) != set(self.names) or not set(self.names) <= set(measured_positions):
            raise ValueError("Posture references must match controlled joints and measurements must include them")
        commands, details = {}, {}
        for name in self.names:
            reference = float(reference_positions[name])
            measured = float(measured_positions[name])
            lo, hi = self.limits[name]
            if not math.isfinite(reference) or not math.isfinite(measured) or not lo <= reference <= hi:
                raise ValueError("Invalid posture reference or measurement: " + name)
            previous = self.commands[name]
            raw_rate = self.gain * (reference - measured)
            rate = max(-self.speed, min(self.speed, raw_rate))
            proposed = previous + rate * dt
            lower = max(lo, measured - self.envelope, previous - self.speed * dt)
            upper = min(hi, measured + self.envelope, previous + self.speed * dt)
            if lower > upper:
                raise RuntimeError("posture_tracking_rate_constraints_incompatible:" + name)
            command = max(lower, min(upper, proposed))
            commands[name] = command
            details[name] = {
                "reference_rad": reference, "measured_rad": measured,
                "reference_error_rad": reference - measured,
                "command_error_rad": command - measured, "command_rad": command,
                "command_rate_rad_s": (command - previous) / dt,
                "rate_limited": abs(raw_rate) > self.speed,
                "tracking_envelope_limited": abs(command - measured) >= self.envelope - 1e-12,
                "physical_bound_limited": command <= lo + 1e-12 or command >= hi - 1e-12,
            }
        self.commands = commands
        return dict(commands), {"joints": details, "physical_success_verified": False,
                                "scope": "bounded_position_target_feedback_only"}


class MeasuredPostureController:
    """A measured-start quintic reference with bounded outer target feedback.

    Only goal joints are commanded. Continue updating after the reference
    finishes to compensate a steady load; elapsed time never proves measured
    arrival, grasp retention, or balance. Those checks belong to the caller.
    """

    def __init__(self, measured_positions, previous_commands, goal, joint_limits, *,
                 gain_s_inv=.5, max_command_speed_rad_s=.08, tracking_envelope_rad=.2,
                 minimum_duration_s=3., max_reference_speed_rad_s=.08,
                 max_reference_acceleration_rad_s2=.16):
        if not goal or not set(goal) <= set(measured_positions) or not set(goal) <= set(previous_commands):
            raise ValueError("Posture goals need measured joints and previous commanded targets")
        self.feedback = BoundedPostureFeedback(
            {n: previous_commands[n] for n in goal}, joint_limits,
            gain_s_inv=gain_s_inv, max_command_speed_rad_s=max_command_speed_rad_s,
            tracking_envelope_rad=tracking_envelope_rad)
        self.reference = PostureReference(
            measured_positions, goal, minimum_duration_s=minimum_duration_s,
            max_speed_rad_s=max_reference_speed_rad_s,
            max_acceleration_rad_s2=max_reference_acceleration_rad_s2)
        for name in goal:
            lo, hi = self.feedback.limits[name]
            if not lo <= self.reference.start[name] <= hi or not lo <= self.reference.goal[name] <= hi:
                raise ValueError("Posture start and goal must be within joint limits: " + name)
            if abs(self.feedback.commands[name] - self.reference.start[name]) > self.feedback.envelope:
                raise ValueError("Initial posture command exceeds the measured tracking envelope: " + name)

    @property
    def elapsed(self):
        return self.reference.elapsed

    @property
    def duration(self):
        return self.reference.duration

    @property
    def goal(self):
        return dict(self.reference.goal)

    def update(self, measured_positions, dt):
        elapsed = self.reference.elapsed
        try:
            reference = self.reference.update(dt)
            commands, evidence = self.feedback.update(reference, measured_positions, dt)
        except Exception:
            self.reference.elapsed = elapsed
            raise
        evidence.update({"reference_elapsed_s": self.elapsed,
                         "reference_duration_s": self.duration,
                         "reference_complete": self.elapsed >= self.duration,
                         "goal_errors_rad": {n: self.reference.goal[n] - float(measured_positions[n])
                                             for n in self.reference.goal}})
        return commands, evidence


class ArmPostureGuide:
    """An IK branch hint indexed by the contact controller's reference progress.

    The servo projects this hint into the exact Cartesian Jacobian nullspace.
    It is never played back directly as an arm trajectory or used as measured
    completion evidence. Collision and live floating-base checks remain separate.
    """

    def __init__(self, waypoints, joint_limits, joint_names):
        self.names = tuple(joint_names)
        if len(waypoints) < 2:
            raise ValueError("An arm posture guide needs at least two waypoints")
        self.progress = np.asarray([p["progress"] for p in waypoints], float)
        if (not np.isfinite(self.progress).all() or np.any(np.diff(self.progress) <= 0)
                or self.progress[0] != 0 or self.progress[-1] != 1):
            raise ValueError("Arm guide progress must strictly increase from zero to one")
        values = []
        for point in waypoints:
            q = point["joint_positions"]
            if set(q) != set(self.names) or any(not math.isfinite(v) or not joint_limits[n][0] <= v <= joint_limits[n][1]
                                              for n, v in q.items()):
                raise ValueError("Arm guide must contain exactly the bounded active arm joints")
            values.append([q[n] for n in self.names])
        self.values = np.asarray(values)

    def sample(self, progress):
        if not math.isfinite(progress) or not 0 <= progress <= 1:
            raise ValueError("Arm guide reference progress must lie in [0, 1]")
        return {name: float(np.interp(progress, self.progress, self.values[:, index]))
                for index, name in enumerate(self.names)}
