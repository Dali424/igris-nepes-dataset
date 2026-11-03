"""Bounded feedback commands for the NEPES task runner (no simulator imports).

cuRobo supplies free-space approach paths. This local servo is for the short
contact and placement phases; it is not a collision-avoiding motion planner.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real
from types import SimpleNamespace
import numpy as np

from .constants import COMMAND_JOINT_NAMES, TELEOPERATION_JOINT_NAMES, TELEOPERATION_RANGES
from .robot_model import load_robot_model
from .snapshot import matrix_to_pose, pose_to_matrix


def rotation_error(target, actual):
    """Shortest rotation vector in the reference frame, including near pi."""
    relative = np.asarray(target) @ np.asarray(actual).T
    angle = math.acos(float(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
    skew = np.array([relative[2, 1] - relative[1, 2],
                     relative[0, 2] - relative[2, 0],
                     relative[1, 0] - relative[0, 1]])
    if angle < 1e-6:
        return skew / 2
    if math.pi - angle < 1e-5:
        values, vectors = np.linalg.eigh((relative + np.eye(3)) / 2)
        axis = vectors[:, np.argmax(values)]
        if np.dot(axis, skew) < 0:
            axis = -axis
        return angle * axis
    return angle / (2 * math.sin(angle)) * skew


def base_velocity_command(current_pose, target_pose, *, max_speed=0.12, max_yaw_rate=0.25,
                          position_gain=.6, yaw_gain=.6):
    """Slow planar goal tracking; no navigation obstacle avoidance is implied."""
    current, target = pose_to_matrix(current_pose), pose_to_matrix(target_pose)
    yaw = math.atan2(current[1, 0], current[0, 0])
    goal_yaw = math.atan2(target[1, 0], target[0, 0])
    error_yaw = math.atan2(math.sin(goal_yaw - yaw), math.cos(goal_yaw - yaw))
    delta_world = target[:2, 3] - current[:2, 3]
    rotation = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
    xy = rotation @ delta_world * position_gain
    norm = np.linalg.norm(xy)
    if norm > max_speed:
        xy *= max_speed / norm
    # Turn before translating far, reducing sweeping into a work surface.
    if abs(error_yaw) > 0.5:
        xy *= 0.15
    return np.r_[xy, np.clip(yaw_gain * error_yaw, -max_yaw_rate, max_yaw_rate)]


def base_command_in_heading_frame(command3, command_frame_pose, realized_frame_pose):
    """Re-express a planar [vx, vy, yaw_rate] command from one body's heading frame in another's.

    ``base_velocity_command`` returns XY in the torso (base_link) heading frame. With a strongly
    twisted waist the HugWBC checkpoint realizes XY in the leg (pelvis, Link_Waist_Pitch) heading
    frame instead (Task-1 side-step at waist yaw -1.34 rad: measured walking direction -78 deg off
    the torso-frame command, -1 deg off the same command read in the pelvis frame). The yaw rate is
    unchanged: the waist is held, so both bodies turn together.
    """
    command = np.asarray(command3, dtype=float)
    if command.shape != (3,) or not np.isfinite(command).all():
        raise ValueError("command3 must contain three finite velocity commands")
    source, realized = pose_to_matrix(command_frame_pose), pose_to_matrix(realized_frame_pose)
    for pose in (source, realized):
        if math.hypot(pose[0, 0], pose[1, 0]) < 1e-8:
            raise ValueError("Base yaw is undefined with a vertical forward axis")
    delta = math.atan2(source[1, 0], source[0, 0]) - math.atan2(realized[1, 0], realized[0, 0])
    c, s = math.cos(delta), math.sin(delta)
    return np.array([c * command[0] - s * command[1], s * command[0] + c * command[1], command[2]])


def _positive_finite(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def arrival_limited_base_velocity(desired3, current_pose, goal_pose, *,
                                  position_tolerance, yaw_tolerance,
                                  linear_acceleration, yaw_acceleration):
    """Cap a base command by its ideal remaining stopping distance.

    Apply this before ``BaseCommandLimiter``. XY keeps the supplied command
    frame and direction; distance uses world XY and yaw uses the shortest
    signed-angle magnitude. Half the unchanged arrival tolerance is reserved
    for braking. This stateless envelope assumes velocity tracking and does
    not guarantee measured stopping distance, balance, or policy gait behavior.
    """
    command = np.asarray(desired3, dtype=float)
    if command.shape != (3,) or not np.isfinite(command).all():
        raise ValueError("desired3 must contain three finite velocity commands")
    command = command.copy()
    tolerances = []
    for value, name in ((position_tolerance, "position_tolerance"),
                        (yaw_tolerance, "yaw_tolerance")):
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be nonnegative and finite")
        tolerances.append(value)
    linear_acceleration = _positive_finite(linear_acceleration, "linear_acceleration")
    yaw_acceleration = _positive_finite(yaw_acceleration, "yaw_acceleration")
    current, goal = pose_to_matrix(current_pose), pose_to_matrix(goal_pose)
    for pose in (current, goal):
        if math.hypot(pose[0, 0], pose[1, 0]) < 1e-8:
            raise ValueError("Base yaw is undefined with a vertical forward axis")
    yaw = math.atan2(current[1, 0], current[0, 0])
    goal_yaw = math.atan2(goal[1, 0], goal[0, 0])
    distance = math.hypot(*(goal[:2, 3] - current[:2, 3]))
    angle = abs(math.atan2(math.sin(goal_yaw - yaw), math.cos(goal_yaw - yaw)))
    speed_cap = math.sqrt(2 * linear_acceleration * max(distance - .5 * tolerances[0], 0.))
    yaw_cap = math.sqrt(2 * yaw_acceleration * max(angle - .5 * tolerances[1], 0.))
    speed = math.hypot(*command[:2])
    if speed > speed_cap:
        command[:2] *= speed_cap / speed
    command[2] = float(np.clip(command[2], -yaw_cap, yaw_cap))
    return command


TURN_IN_PLACE_KEYS = frozenset(("max_yaw_rate_rad_s", "yaw_acceleration_rad_s2", "minimum_yaw_rate_rad_s",
                                "maximum_distance_m"))


def validate_turn_in_place(settings):
    """Explicit yaw-only (turn in place) navigation settings, normalized to floats.

    While a base goal lies within ``maximum_distance_m`` (default: the station
    position tolerance) of the measured base but outside the yaw tolerance,
    ``max_yaw_rate_rad_s`` replaces the walking yaw cap and
    ``yaw_acceleration_rad_s2`` the stopping-envelope and limiter yaw
    acceleration. The optional ``minimum_yaw_rate_rad_s`` floors the commanded
    yaw-rate magnitude there, so a policy command deadband (HugWBC ignores
    command norms below about 0.1) cannot stall the turn short of the heading.
    Nothing here certifies that the policy realizes the commanded yaw rate.
    """
    required = {"max_yaw_rate_rad_s", "yaw_acceleration_rad_s2"}
    if (not isinstance(settings, Mapping) or not required <= set(settings)
            or set(settings) - TURN_IN_PLACE_KEYS):
        raise ValueError("turn_in_place requires max_yaw_rate_rad_s and yaw_acceleration_rad_s2; only "
                         "minimum_yaw_rate_rad_s and maximum_distance_m are optional")
    result = {}
    for key, value in settings.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"turn_in_place.{key} must be positive and finite")
        result[key] = _positive_finite(value, f"turn_in_place.{key}")
    if result["max_yaw_rate_rad_s"] > 1.:
        raise ValueError("turn_in_place.max_yaw_rate_rad_s must be at most 1 rad/s")
    if "minimum_yaw_rate_rad_s" in result and result["minimum_yaw_rate_rad_s"] >= result["max_yaw_rate_rad_s"]:
        raise ValueError("turn_in_place.minimum_yaw_rate_rad_s must be below max_yaw_rate_rad_s")
    return result


def turn_in_place_base_velocity(desired3, current_pose, goal_pose, *, settings, position_tolerance,
                                yaw_tolerance, yaw_gain=.6):
    """Re-shape the yaw command of a yaw-only base goal; XY passes through unchanged.

    Apply after ``arrival_limited_base_velocity`` and before ``BaseCommandLimiter``.
    The turn is active while the goal lies within the turn radius of the
    measured base and the wrapped yaw error exceeds ``yaw_tolerance``: the yaw
    rate is then yaw_gain * error, capped by ``max_yaw_rate_rad_s`` and by the
    stopping envelope sqrt(2 * yaw_acceleration * (|error| - yaw_tolerance / 2)),
    and floored at ``minimum_yaw_rate_rad_s`` when declared. Inside the yaw
    tolerance the command returns unchanged (the caller's arrival braking
    zeroes it). Returns (command, evidence); ``evidence["yaw_acceleration_rad_s2"]``
    is the limiter yaw acceleration to apply while the goal is within the turn
    radius (ramp-up and braking), else None.
    """
    settings = validate_turn_in_place(settings)
    command = np.asarray(desired3, dtype=float)
    if command.shape != (3,) or not np.isfinite(command).all():
        raise ValueError("desired3 must contain three finite velocity commands")
    command = command.copy()
    tolerances = []
    for value, name in ((position_tolerance, "position_tolerance"), (yaw_tolerance, "yaw_tolerance")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name} must be nonnegative and finite")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be nonnegative and finite")
        tolerances.append(value)
    yaw_gain = _positive_finite(yaw_gain, "yaw_gain")
    current, goal = pose_to_matrix(current_pose), pose_to_matrix(goal_pose)
    for pose in (current, goal):
        if math.hypot(pose[0, 0], pose[1, 0]) < 1e-8:
            raise ValueError("Base yaw is undefined with a vertical forward axis")
    yaw = math.atan2(current[1, 0], current[0, 0])
    goal_yaw = math.atan2(goal[1, 0], goal[0, 0])
    error = math.atan2(math.sin(goal_yaw - yaw), math.cos(goal_yaw - yaw))
    distance = math.hypot(*(goal[:2, 3] - current[:2, 3]))
    radius = settings.get("maximum_distance_m", tolerances[0])
    within_radius = distance <= radius
    active = bool(within_radius and abs(error) > tolerances[1])
    evidence = {"active": active, "within_turn_radius": bool(within_radius), "planar_distance_m": distance,
                "turn_radius_m": radius, "yaw_error_rad": error, "yaw_tolerance_rad": tolerances[1],
                "input_yaw_rate_rad_s": float(command[2]), "yaw_rate_rad_s": float(command[2]),
                "stopping_cap_rad_s": None, "floored": False,
                "yaw_acceleration_rad_s2": settings["yaw_acceleration_rad_s2"] if within_radius else None,
                "limits": dict(settings), "policy_yaw_tracking_verified": False}
    if active:
        maximum = settings["max_yaw_rate_rad_s"]
        rate = float(np.clip(yaw_gain * error, -maximum, maximum))
        cap = math.sqrt(2. * settings["yaw_acceleration_rad_s2"] * max(abs(error) - .5 * tolerances[1], 0.))
        rate = float(np.clip(rate, -cap, cap))
        minimum = settings.get("minimum_yaw_rate_rad_s")
        floored = minimum is not None and abs(rate) < minimum
        if floored:
            rate = math.copysign(minimum, error)
        command[2] = rate
        evidence.update(yaw_rate_rad_s=rate, stopping_cap_rad_s=cap, floored=bool(floored))
    return command, evidence


class BaseCommandLimiter:
    """Limit changes in commanded [vx, vy, yaw_rate], starting from rest.

    XY acceleration is bounded by its vector norm, in the caller's command
    frame. This limits command changes, not measured robot acceleration or
    world-frame acceleration while the body itself rotates. Apply any desired
    velocity cap before this limiter. ``reset`` is for episode boundaries;
    ordinary braking should pass a zero desired command through ``update``.
    """

    def __init__(self, linear_acceleration=.05, yaw_acceleration=.08):
        self.linear_acceleration = _positive_finite(linear_acceleration, "linear_acceleration")
        self.yaw_acceleration = _positive_finite(yaw_acceleration, "yaw_acceleration")
        self.reset()

    def reset(self):
        self._command = np.zeros(3, dtype=float)

    def update(self, desired3, dt, *, yaw_acceleration=None):
        """Advance one step; ``yaw_acceleration`` optionally replaces the yaw bound for this step only."""
        dt = _positive_finite(dt, "dt")
        desired = np.asarray(desired3, dtype=float)
        if desired.shape != (3,) or not np.isfinite(desired).all():
            raise ValueError("desired3 must contain three finite velocity commands")
        yaw_bound = (self.yaw_acceleration if yaw_acceleration is None
                     else _positive_finite(yaw_acceleration, "yaw_acceleration")) * dt
        change = desired - self._command
        xy_norm = float(np.linalg.norm(change[:2]))
        xy_bound = self.linear_acceleration * dt
        if xy_norm > xy_bound:
            change[:2] *= xy_bound / xy_norm
        change[2] = np.clip(change[2], -yaw_bound, yaw_bound)
        self._command = self._command + change
        return self._command.copy()


class CarryStabilizer:
    """Follow measured base XY/yaw smoothly while preserving TCP world height.

    A virtual planar base follows measured XY/yaw at bounded rates. The TCP
    retains its initial planar offset from that base, its initial world Z and
    its initial roll/pitch. Measured base height, roll and pitch never enter the
    target; an arm servo must compensate those disturbances using the actual
    full base pose. Recreate this object when a new carry phase starts.

    Translation and rotation share a conservative TCP displacement budget:
    |base XY step| + |TCP planar offset| * |yaw step| <= speed * dt. Thus the
    output TCP translation also obeys max_linear_speed during a turn. This
    filter does not certify balance, reachability, grasp stability or collision
    clearance; lag may make the target unreachable if navigation runs ahead.
    """

    def __init__(self, base_pose_w, tcp_pose_w, max_linear_speed=.08, max_yaw_rate=.10):
        self.max_linear_speed = _positive_finite(max_linear_speed, "max_linear_speed")
        self.max_yaw_rate = _positive_finite(max_yaw_rate, "max_yaw_rate")
        base = pose_to_matrix(base_pose_w)
        tcp = pose_to_matrix(tcp_pose_w)
        self._initial_yaw = self._yaw(base)
        self._yaw_reference = self._initial_yaw
        self._xy_reference = base[:2, 3].copy()
        self._initial_tcp_offset_w = tcp[:2, 3] - base[:2, 3]
        self._tcp_height_w = float(tcp[2, 3])
        self._initial_tcp_rotation_w = tcp[:3, :3].copy()
        self._previous_quaternion = np.asarray(matrix_to_pose(tcp)[3:])
        self._target_pose_w = np.asarray(matrix_to_pose(tcp))
        self._measured_base = base.copy()

    @staticmethod
    def _yaw(base):
        if np.hypot(base[0, 0], base[1, 0]) < 1e-8:
            raise ValueError("Base yaw is undefined with a vertical forward axis")
        return math.atan2(base[1, 0], base[0, 0])

    def update(self, base_pose_w, dt):
        dt = _positive_finite(dt, "dt")
        base = pose_to_matrix(base_pose_w)
        measured_yaw = self._yaw(base)
        yaw_error = math.atan2(math.sin(measured_yaw - self._yaw_reference),
                               math.cos(measured_yaw - self._yaw_reference))
        yaw_step = float(np.clip(yaw_error, -self.max_yaw_rate * dt,
                                self.max_yaw_rate * dt))
        xy_step = base[:2, 3] - self._xy_reference
        xy_norm = float(np.linalg.norm(xy_step))
        translation_bound = self.max_linear_speed * dt
        if xy_norm > translation_bound:
            xy_step *= translation_bound / xy_norm
            xy_norm = translation_bound
        displacement_bound = xy_norm + np.linalg.norm(self._initial_tcp_offset_w) * abs(yaw_step)
        if displacement_bound > translation_bound:
            scale = translation_bound / displacement_bound
            xy_step *= scale
            yaw_step *= scale
        self._xy_reference += xy_step
        self._yaw_reference += yaw_step
        yaw_change = self._yaw_reference - self._initial_yaw
        c, s = math.cos(yaw_change), math.sin(yaw_change)
        rotation_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        target = np.eye(4)
        target[:2, 3] = self._xy_reference + rotation_z[:2, :2] @ self._initial_tcp_offset_w
        target[2, 3] = self._tcp_height_w
        target[:3, :3] = rotation_z @ self._initial_tcp_rotation_w
        pose = np.asarray(matrix_to_pose(target))
        # Keep quaternion signs continuous across +/- pi for downstream users.
        if np.dot(pose[3:], self._previous_quaternion) < 0:
            pose[3:] *= -1
        self._previous_quaternion = pose[3:].copy()
        self._target_pose_w = pose.copy()
        self._measured_base = base.copy()
        return pose

    def diagnostics(self, measured_tcp_pose_w):
        """Compare the current reference with the real TCP, without recapturing it.

        The caller chooses the heading body (pelvis or torso) consistently at
        construction and update. Base-filter lag is reported separately from
        actual arm error: a well-tracked, lagging reference is not evidence that
        navigation has remained within reach. Height error measures real TCP
        motion against the original fixed world height, never the current base Z.
        """
        measured = pose_to_matrix(measured_tcp_pose_w)
        target = pose_to_matrix(self._target_pose_w)
        yaw_lag = self._yaw(self._measured_base) - self._yaw_reference
        height_error = self._tcp_height_w - float(measured[2, 3])
        return {
            "position_error_m": float(np.linalg.norm(target[:3, 3] - measured[:3, 3])),
            "rotation_error_rad": float(np.linalg.norm(rotation_error(target[:3, :3], measured[:3, :3]))),
            "height_error_m": abs(height_error),
            "height_error_signed_m": height_error,
            "base_xy_lag_m": float(np.linalg.norm(self._measured_base[:2, 3] - self._xy_reference)),
            "base_yaw_lag_rad": abs(math.atan2(math.sin(yaw_lag), math.cos(yaw_lag))),
            "fixed_world_height_m": self._tcp_height_w,
            "reference_tcp_pose_w": self._target_pose_w.tolist(),
            "measured_tcp_pose_w": matrix_to_pose(measured),
        }


class CarryTrackingGate:
    """A speed request with hysteresis; the caller owns actual actuator commands.

    Errors up to both resume bounds request full speed; between resume and stop
    bounds the request decreases linearly. Reaching either stop bound latches a
    pause until both errors return within their resume bounds. Invalid/missing
    measurements also latch a pause. This is a tracking guard, not a grasp,
    collision, balance or payload-retention predicate.
    """

    def __init__(self, *, stop_position_error_m, resume_position_error_m,
                 stop_rotation_error_rad, resume_rotation_error_rad):
        settings = dict(stop_position_error_m=stop_position_error_m,
                        resume_position_error_m=resume_position_error_m,
                        stop_rotation_error_rad=stop_rotation_error_rad,
                        resume_rotation_error_rad=resume_rotation_error_rad)
        for name, value in settings.items():
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be positive and finite")
            settings[name] = _positive_finite(value, name)
        if not settings["resume_position_error_m"] < settings["stop_position_error_m"]:
            raise ValueError("resume_position_error_m must be below stop_position_error_m")
        if not settings["resume_rotation_error_rad"] < settings["stop_rotation_error_rad"]:
            raise ValueError("resume_rotation_error_rad must be below stop_rotation_error_rad")
        self.settings = settings
        self.reset()

    def reset(self):
        """Start a new carry phase; never recapture any physical grasp transform."""
        self.paused = False

    @staticmethod
    def _measurement(value):
        if value is None or isinstance(value, (bool, np.bool_)):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) and number >= 0 else None

    def update(self, position_error_m=None, rotation_error_rad=None):
        position = self._measurement(position_error_m)
        rotation = self._measurement(rotation_error_rad)
        reason = None
        if position is None or rotation is None:
            self.paused = True
            scale, reason = 0., "invalid_tracking_measurement"
        else:
            stop_p, stop_r = (self.settings["stop_position_error_m"], self.settings["stop_rotation_error_rad"])
            resume_p, resume_r = (self.settings["resume_position_error_m"], self.settings["resume_rotation_error_rad"])
            if position >= stop_p or rotation >= stop_r:
                self.paused = True
                reason = ("position_and_rotation_stop" if position >= stop_p and rotation >= stop_r
                          else "position_stop" if position >= stop_p else "rotation_stop")
            elif self.paused:
                if position <= resume_p and rotation <= resume_r:
                    self.paused = False
                else:
                    reason = "tracking_recovery_pending"
            scale = 0. if self.paused else float(np.clip(min(
                (stop_p - position) / (stop_p - resume_p),
                (stop_r - rotation) / (stop_r - resume_r)), 0., 1.))
            if not self.paused and scale < 1.:
                reason = "tracking_slowdown"
        return {"position_error_m": position, "rotation_error_rad": rotation,
                "paused": self.paused, "speed_scale": scale, "reason": reason,
                "limits": dict(self.settings)}


class ArmServo:
    def __init__(self, model=None, *, max_joint_speed=0.6, max_tracking_error=0.15):
        self.model = model or load_robot_model()
        self.max_joint_speed = max_joint_speed
        self.max_tracking_error = max_tracking_error
        self.limits = dict(self.model.joint_limits)
        for name, limits in zip(TELEOPERATION_JOINT_NAMES, TELEOPERATION_RANGES):
            low, high = self.limits[name]
            self.limits[name] = max(low, limits[0]), min(high, limits[1])

    def solve_step(self, joint_positions, base_pose_w, target_pose_w, arm, dt, previous_targets=None,
                   *, speed_scale=1.0, tracking_error_limits=None, nullspace_gain=0.0,
                   posture_target=None, posture_gain=0.0, feedforward_twist_base=None,
                   solver="dls"):
        if solver not in {"dls", "bounded_lsq"}:
            raise ValueError("Unknown arm servo solver")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Servo dt must be positive")
        if not math.isfinite(speed_scale) or not 0 < speed_scale <= 1:
            raise ValueError("Servo speed_scale must be in (0, 1]")
        if not math.isfinite(nullspace_gain) or not 0 <= nullspace_gain <= 1:
            raise ValueError("Nullspace gain must be in [0, 1]")
        if not math.isfinite(posture_gain) or not 0 <= posture_gain <= 2:
            raise ValueError("Posture gain must be in [0, 2]")
        feedforward = np.zeros(6) if feedforward_twist_base is None else np.asarray(feedforward_twist_base, float)
        if feedforward.shape != (6,) or not np.isfinite(feedforward).all():
            raise ValueError("Feedforward twist must contain six finite base-frame velocities")
        # PhysX may overshoot authored limits slightly; reject material excursions.
        measured = dict(joint_positions)
        for name, (low, high) in self.model.joint_limits.items():
            value = measured[name]
            if not np.isfinite(value) or value < low - 0.05 or value > high + 0.05:
                raise ValueError(f"Measured joint outside physical limits: {name}={value}")
            measured[name] = float(np.clip(value, low, high))
        transforms = self.model.forward_kinematics(measured)
        ee = transforms[self.model.default_ee_link(arm)]
        target = np.linalg.inv(pose_to_matrix(base_pose_w)) @ pose_to_matrix(target_pose_w)
        names = self.model.arm_joint_names(arm)
        if posture_target is not None:
            if set(posture_target) != set(names) or any(not math.isfinite(v) or not self.limits[n][0] <= v <= self.limits[n][1]
                                                       for n, v in posture_target.items()):
                raise ValueError("Posture target must contain the active arm's finite, bounded joints")
        elif posture_gain:
            raise ValueError("Posture gain requires a posture target")
        tracking_error_limits = dict(tracking_error_limits or {})
        if set(tracking_error_limits) - set(names):
            raise ValueError("Tracking limits may name only the active arm joints")
        for value in tracking_error_limits.values():
            if not math.isfinite(value) or not 0 < value <= .3:
                raise ValueError("Per-joint tracking error limits must be in (0, .3] rad")
        jacobian = np.zeros((6, len(names)))
        for i, name in enumerate(names):
            joint = self.model.joints[name]
            origin = transforms[joint.parent] @ joint.origin
            axis = origin[:3, :3] @ joint.axis
            jacobian[:3, i] = np.cross(axis, ee[:3, 3] - origin[:3, 3])
            jacobian[3:, i] = axis
        error = np.r_[target[:3, 3] - ee[:3, 3], rotation_error(target[:3, :3], ee[:3, :3])]
        weights = np.array([1, 1, 1, .2, .2, .2])
        weighted = weights[:, None] * jacobian
        velocity = weighted.T @ np.linalg.solve(weighted @ weighted.T + np.eye(6) * .04**2,
                                                weights * (error * 2.0 + feedforward))
        null_velocity = np.zeros(len(names))
        if nullspace_gain or posture_gain:
            # Use the exact Jacobian nullspace, not the damped inverse's
            # residual projector: the latter leaks the limit-avoidance command
            # into the contact task. Nonlinear finite steps still need feedback
            # and collision validation by the caller.
            _, singular, right = np.linalg.svd(weighted, full_matrices=True)
            rank = int(np.count_nonzero(singular > max(weighted.shape) * singular[0] * np.finfo(float).eps))
            basis = right[rank:].T
            low, high = np.asarray([self.limits[n] for n in names]).T
            q = np.asarray([measured[n] for n in names])
            normalized = np.clip((2 * q - low - high) / (high - low), -.999, .999)
            gradient = np.clip(-normalized / (1 - normalized**2), -3., 3.)
            secondary = nullspace_gain * gradient
            if posture_gain:
                secondary += posture_gain * (np.asarray([posture_target[n] for n in names]) - q)
            null_velocity = basis @ (basis.T @ secondary)
        # Give the primary Cartesian correction first use of actuator bounds.
        # Add only a COMMON scaled nullspace step in its remaining headroom;
        # independently clipping a combined vector destroys J @ dq_null == 0.
        velocity = np.clip(velocity, -self.max_joint_speed, self.max_joint_speed) * speed_scale
        q = np.asarray([measured[n] for n in names])
        old = np.asarray([measured[n] if previous_targets is None else previous_targets[n] for n in names])
        low, high = np.asarray([self.limits[n] for n in names]).T
        tracking = np.asarray([tracking_error_limits.get(n, self.max_tracking_error) for n in names])
        low, high = np.maximum(low, q - tracking), np.minimum(high, q + tracking)
        origin = np.clip(old, low, high)
        primary = np.clip(origin + velocity * dt, low, high)
        step_bound = self.max_joint_speed * speed_scale * dt
        low, high = np.maximum(low, origin - step_bound), np.minimum(high, origin + step_bound)
        bounded_diagnostics = None
        if solver == "bounded_lsq":
            from scipy.optimize import lsq_linear
            # Solve with all actuator bounds present, so a saturated wrist can
            # redistribute the Cartesian motion to remaining joints. Clipping
            # an unconstrained inverse cannot perform that redistribution.
            a = np.vstack((weighted, .04 * np.eye(len(names))))
            b = np.r_[weights * (2. * error + feedforward) * speed_scale,
                      .04 * null_velocity * speed_scale]
            lower, upper = (low - origin) / dt, (high - origin) / dt
            free = upper - lower > 1e-12
            bounded_velocity = lower.copy()
            fit = None
            if np.any(free):
                fit = lsq_linear(a[:, free], b - a[:, ~free] @ lower[~free],
                    bounds=(lower[free], upper[free]), method="bvls", tol=1e-10, max_iter=100)
                if not fit.success or not np.isfinite(fit.x).all():
                    raise RuntimeError("Bounded arm least-squares solve failed")
                bounded_velocity[free] = fit.x
            primary = np.clip(origin + bounded_velocity * dt, low, high)
            bounded_diagnostics = {"active_bounds": {n: ("lower" if abs(v-l) < 1e-8 else "upper")
                for n, v, l, h in zip(names, bounded_velocity, lower, upper)
                if min(abs(v-l), abs(v-h)) < 1e-8},
                "weighted_velocity_residual": float(np.linalg.norm(weighted @ bounded_velocity - b[:6])),
                "optimality": float(fit.optimality) if fit is not None else 0.,
                "posture_priority": "soft_regularization_in_constrained_cartesian_solve"}
            delta = np.zeros(len(names))
        else:
            delta = null_velocity * dt * speed_scale
        step_factor = 1.
        for index, value in enumerate(delta):
            if value > 1e-15:
                step_factor = min(step_factor, max(0., (high[index] - primary[index]) / value))
            elif value < -1e-15:
                step_factor = min(step_factor, max(0., (low[index] - primary[index]) / value))
        commanded = primary + step_factor * delta
        result = dict(zip(names, commanded.astype(float).tolist()))
        return result, {"position_error_m": float(np.linalg.norm(error[:3])),
                        "rotation_error_rad": float(np.linalg.norm(error[3:])),
                        "feedforward_twist_base": feedforward.tolist(),
                        "joint_limit_nullspace_gain_s_inv": float(nullspace_gain),
                        "posture_nullspace_gain_s_inv": float(posture_gain),
                        "nullspace_velocity_norm_rad_s": float(np.linalg.norm(null_velocity)),
                        "nullspace_tcp_twist_norm": float(np.linalg.norm(jacobian @ null_velocity)),
                        "nullspace_common_scaling": float(step_factor * speed_scale) if solver == "dls" else None,
                        "applied_nullspace_tcp_twist_norm": (float(np.linalg.norm(jacobian @ ((commanded - primary) / dt)))
                                                            if solver == "dls" else None),
                        "protective_target_recentering_rad": float(np.max(abs(origin - old))),
                        "applied_command_tcp_twist_norm": float(np.linalg.norm(jacobian @ ((commanded - old) / dt))),
                        "solver": solver, "bounded_lsq": bounded_diagnostics,
                        "collision_checked": False}


def gripper_open_position(gripper, open_position_rad=None):
    """Resolve a task aperture inside the model's closed-to-default-open span.

    The unchanged 0.1 rad measured-open tolerance must not include the closed
    target. A narrower aperture changes neither closure nor pinch acceptance.
    """
    default = float(gripper["open_position_rad"])
    closed = float(gripper["closed_position_rad"])
    value = default if open_position_rad is None else open_position_rad
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("open_gripper_position_rad must be finite")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("open_gripper_position_rad must be finite") from exc
    if (not all(math.isfinite(x) for x in (value, default, closed))
            or not min(default, closed) <= value <= max(default, closed)
            or abs(value - closed) <= .10):
        raise ValueError("open_gripper_position_rad must lie between model closed and default-open targets, "
                         "more than 0.1 rad from closed")
    return value


def gripper_is_open(angle, gripper, open_position_rad=None):
    """Compare a measured angle with the same aperture used for commands."""
    goal = gripper_open_position(gripper, open_position_rad)
    return bool(math.isfinite(float(angle)) and abs(float(angle) - goal) < .10)


def gripper_closed_position(gripper, closed_position_rad=None, open_position_rad=None):
    """Resolve a contact setpoint; measured pinch still uses model closure.

    This changes only the position command/preload, never a joint limit,
    actuator gain, torque cap, or the independent contact acceptance test.
    """
    closed = float(gripper["closed_position_rad"])
    opened = gripper_open_position(gripper, open_position_rad)
    value = closed if closed_position_rad is None else closed_position_rad
    if isinstance(value, bool):
        raise ValueError("closed_gripper_position_rad must be finite")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("closed_gripper_position_rad must be finite") from exc
    if (not math.isfinite(value) or not min(closed, opened) <= value <= max(closed, opened)
            or abs(opened - value) <= .10):
        raise ValueError("closed_gripper_position_rad must lie within the model opening span and clear open tolerance")
    return value


def supported_gripper_preload(gripper, measured_angle, preload_rad, *,
                              open_position_rad=None, closed_position_rad=None):
    """Reduce pinch preload after a separately verified support contact.

    Keep a positive closing error and the original model opening span. The
    caller must gate this on the bounded support hold; contact and retention
    remain measurements, not consequences of this reference.
    """
    closed = gripper_closed_position(gripper, closed_position_rad, open_position_rad)
    opened = gripper_open_position(gripper, open_position_rad)
    if (isinstance(preload_rad, bool) or not math.isfinite(preload_rad)
            or not .005 <= preload_rad <= .05 or not math.isfinite(measured_angle)):
        raise ValueError("Supported preload requires a finite measured jaw and 0.005..0.05 rad preload")
    direction = float(np.sign(opened - closed))
    distance = direction * (measured_angle - closed)
    # Never close more strongly than the original command or enter the open gate.
    distance = float(np.clip(distance - preload_rad, 0., max(0., abs(opened - closed) - .100001)))
    return gripper_closed_position(gripper, closed + direction * distance, open_position_rad)


class LiveCommandSource:
    """The public command-source protocol, without an NPZ or random commands."""
    teleop_joint_names = TELEOPERATION_JOINT_NAMES

    def __init__(self, initial_positions, *, device, command_joint_names=COMMAND_JOINT_NAMES, grippers=None):
        import torch
        self.torch = torch
        self.device = device
        self.command_joint_names = tuple(command_joint_names)
        self.grippers = dict(grippers or {})
        self.targets = {name: float(initial_positions[name]) for name in self.command_joint_names}
        self.base_velocity = np.zeros(3)

    def frame_at(self, timestamp):
        tensor = self.torch.as_tensor
        return SimpleNamespace(frame_index=0,
            base_velocity=tensor(self.base_velocity, dtype=self.torch.float32, device=self.device)[None],
            teleop_joint_names=self.teleop_joint_names,
            teleop_joint_pos=tensor([self.targets[n] for n in self.teleop_joint_names],
                                   dtype=self.torch.float32, device=self.device)[None])

    def apply_hand(self, arm, close, dt, *, closed_positions=None, open_position_rad=None, closed_position_rad=None):
        if arm in self.grippers:
            grip = self.grippers[arm]
            name = grip["joint_name"]
            open_goal = gripper_open_position(grip, open_position_rad)
            closed_goal = gripper_closed_position(grip, closed_position_rad, open_position_rad)
            goal = closed_goal if close else open_goal
            change = float(grip.get("speed_rad_s", .7)) * dt
            self.targets[name] += float(np.clip(goal - self.targets[name], -change, change))
            return
        if open_position_rad is not None or closed_position_rad is not None:
            raise ValueError("A task gripper aperture requires a single-joint gripper model")
        prefix = "l_" if arm == "left" else "r_"
        # Provisional anatomical joint targets; calibrated task config can override.
        defaults = {"0_joint_thumb_proximal": .7, "1_joint_thumb_middle": .55,
                    "3_joint_index_middle": .8, "5_joint_middle_middle": .8,
                    "7_joint_ring_middle": .8, "9_joint_little_middle": .8}
        closed_positions = closed_positions or {prefix + n: v for n, v in defaults.items()}
        for name in self.targets:
            if name.startswith(prefix):
                goal = float(closed_positions.get(name, 0.0)) if close else 0.0
                self.targets[name] += float(np.clip(goal - self.targets[name], -dt * .7, dt * .7))

    def upper_target_array(self):
        return np.asarray([self.targets[n] for n in self.command_joint_names], dtype=np.float32)
