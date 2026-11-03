"""Measured object-relative references for physical CRS manipulation.

This module commands poses only. It never writes a rigid-body pose, attaches a
payload, changes a captured grasp transform, or declares manipulation success.
The task machine must retain its own bilateral-contact, slip and success guards.
References are not collision-checked paths: use them for calibrated local lift,
alignment and insertion phases, after checking their swept volume.
"""
from __future__ import annotations

import math
import numpy as np

from .object_symmetry import (align_spin, pinch_point, pivot_radius, remove_axis_component, spin_aligned_object,
                              symmetric_capture_errors, symmetric_pose_errors, unit_axis)
from .snapshot import matrix_to_pose, pose_to_matrix
from .task_control import rotation_error


def _positive(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be positive and finite") from error
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def _rotation_vector_matrix(vector):
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def goal_object_from_tcp(goal_tcp_pose_w, object_to_tcp):
    """Convert a TCP goal using the control transform that generated that goal."""
    return matrix_to_pose(pose_to_matrix(goal_tcp_pose_w) @ np.linalg.inv(pose_to_matrix(object_to_tcp)))


class _BoundedVectorCorrection:
    """Exactly integrate a critically damped filter of a bounded vector.

    Its impulse response w² t exp(-wt) is positive and integrates to one,
    so a target in the radius-C ball produces an output in that ball.
    The L1 norms of its first and second derivatives give the conservative
    bounds 2*C*w/e for speed and (2+2/e²)*C*w² for acceleration, including
    arbitrary target changes. A protective hold resets velocity immediately;
    resuming at rest is equivalent to a constant bounded input history.
    """

    def __init__(self, limit, speed, acceleration):
        self.limit = limit
        self.omega = min(speed / limit * math.e / 2.,
                         math.sqrt(acceleration / limit / (2. + 2. / math.e**2)))
        if not math.isfinite(self.omega) or self.omega <= 0.:
            raise ValueError("object_feedback rates and limits have an invalid numerical scale")
        self.value = np.zeros(3)
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)

    def update(self, target, dt, *, hold=False):
        if hold:
            self.velocity[:] = 0.
            self.acceleration[:] = 0.
            return self.value.copy()
        target = np.asarray(target, dtype=float)
        # hypot avoids an overflowing squared norm for large finite inputs.
        length = math.hypot(*target)
        if length > self.limit:
            target = target * (self.limit / length)
        w, u = self.omega, self.omega * dt
        if u > 60.:
            self.value = target.copy()
            self.velocity[:] = 0.
        else:
            error, velocity = self.value - target, self.velocity.copy()
            decay = math.exp(-u)
            self.value = target + decay * ((1. + u) * error + dt * velocity)
            self.velocity = decay * ((1. - u) * velocity - w * u * error)
        self.acceleration = w * (w * (target - self.value) - 2. * self.velocity)
        return self.value.copy()


class CrsPoseController:
    """Replace a step pose target with a bounded, measured CRS pose reference.

    ``object_to_tcp`` is the fixed transform captured at physical acquisition.
    Optional ``reference_object_to_tcp`` is a measured phase-entry transform
    for continuous motion references after bounded contact compliance. It never
    replaces ``object_to_tcp`` in the slip guard or task retention checks.
    A quintic reference has zero endpoint velocity/acceleration and respects
    the configured translation/rotation speed and acceleration bounds during
    uninterrupted tracking. Tracking-lead or grasp alarms hold the last target
    immediately; such protective stops are explicitly reported and can have
    an abrupt reference deceleration. The default ``quintic`` profile resumes
    a new quintic at rest. Opt-in ``online`` instead accelerates from rest on
    the same scalar path, using exact accelerate/cruise/brake integration. It
    avoids repeatedly restarting a long quintic after a brief feedback pause.
    These limits are reference limits, not measured robot acceleration bounds.

    Both relative slip and a missing pinch stop reference advancement early;
    they do not replace or extend the task machine's grasp-loss grace period.
    A pause never follows a displaced object or rebases the retention reference.
    Lead is measured along the commanded translation/rotation direction;
    orthogonal errors have separate finite bounds. Small sideways jaw compliance
    must not permanently block an otherwise correctly followed vertical lift.
    Recreate the controller when entering a new local manipulation phase, using
    the SAME acquisition transform until the object has actually been released.

    Optional ``object_feedback`` corrects measured payload drift even when the
    hand accurately follows its nominal reference. It bounds the translation
    and rotation correction separately from the nominal trajectory. The ideal
    hand command is reference_object @ inv(measured_object) @ measured_hand;
    only its bounded, filtered difference from the nominal hand is applied.
    This does not recapture the grasp. Progress uses measured_object @ the fixed
    phase reference transform, and the original acquisition transform remains
    the sole slip reference. Pinch/slip alarms freeze both target components.
    An optional update-time ``hold_reason`` also freezes both components, for
    measured prerequisites such as an auxiliary hand's support load. It is
    reported separately from grasp alarms and never changes grasp evidence.
    Correction limits are additional to nominal-reference limits; the caller
    must retain its joint-speed, tracking, torque, collision and success guards.

    Optional ``object_symmetry_axis`` (a unit vector in the object frame) declares
    the object rotationally symmetric about that axis (the CRS ring normal). The
    slip guard then measures the object in the hand frame so that a spin of the
    ring inside the closed pads is neither slip nor tilt, the tracking pose and
    the object feedback use the object spun back onto the nominal orientation, and
    the rotation correction never turns the hand about the object's axis. The
    spin is reported as ``object_spin_rad``. Absent, every number is unchanged.
    Optional ``object_pivot_radius_m`` (the radius of the circle on which the pads
    pinch the object, the CRS rim) measures the slip at that pinch contact instead
    of at the object centre, so a ring that turned about the pads (its centre
    swung) is no slip either; a radial or axial slip of the rim keeps its size.

    Optional ``two_speed_descent`` (runtime_v26 C4, online profile only) gives a
    non-rotating descending segment a fast zone: the nominal reference runs at
    ``fast_linear_speed`` until its object height above the seat (goal height plus
    ``goal_height_above_seat_m``) reaches ``slow_zone_height_m``, braking at the
    segment's linear acceleration so it enters that slow zone at no more than
    ``linear_speed``, and then runs at ``linear_speed`` to rest at the goal. Pauses,
    holds, lead/cross-track checks and the object feedback are unchanged. A
    rotating, non-descending segment or one that starts inside the slow zone keeps
    the single ``linear_speed`` profile (the evidence names why).
    """

    def __init__(self, object_to_tcp, tcp_pose_w, goal_object_pose_w, *,
                 linear_speed=.03, linear_acceleration=.06,
                 angular_speed=.12, angular_acceleration=.3,
                 position_lead=.004, rotation_lead=.02,
                 cross_track_position=.010, cross_track_rotation=.02,
                 pause_slip_position=.012, pause_slip_rotation=.10,
                 reference_profile="quintic", reference_object_to_tcp=None,
                 object_feedback=None, object_symmetry_axis=None, object_pivot_radius_m=None,
                 two_speed_descent=None):
        if not isinstance(reference_profile, str) or reference_profile not in {"quintic", "online"}:
            raise ValueError("reference_profile must be 'quintic' or 'online'")
        self.reference_profile = reference_profile
        self._symmetry_axis = (None if object_symmetry_axis is None
                               else unit_axis(object_symmetry_axis, "object_symmetry_axis"))
        self._pivot_radius = (None if object_pivot_radius_m is None
                              else pivot_radius(object_pivot_radius_m, "object_pivot_radius_m"))
        if self._pivot_radius is not None and self._symmetry_axis is None:
            raise ValueError("object_pivot_radius_m requires object_symmetry_axis")
        for name, value in {
            "linear_speed": linear_speed, "linear_acceleration": linear_acceleration,
            "angular_speed": angular_speed, "angular_acceleration": angular_acceleration,
            "position_lead": position_lead, "rotation_lead": rotation_lead,
            "cross_track_position": cross_track_position, "cross_track_rotation": cross_track_rotation,
            "pause_slip_position": pause_slip_position, "pause_slip_rotation": pause_slip_rotation,
        }.items():
            setattr(self, name, _positive(value, name))
        self._object_to_tcp = pose_to_matrix(object_to_tcp)
        if self._pivot_radius is not None:
            pinch_point(self._object_to_tcp, self._symmetry_axis, self._pivot_radius)  # the grasp needs a pinch azimuth
        self._reference_object_to_tcp = (self._object_to_tcp.copy() if reference_object_to_tcp is None
                                        else pose_to_matrix(reference_object_to_tcp))
        self._reference = pose_to_matrix(tcp_pose_w)
        self._goal_object = pose_to_matrix(goal_object_pose_w)
        self._goal_tcp = self._goal_object @ self._reference_object_to_tcp
        self._progress = 0.
        self._paused = False
        self._previous_quaternion = np.asarray(matrix_to_pose(self._reference)[3:])
        self._restart_segment()
        distance = float(np.linalg.norm(self._translation))
        angle = float(np.linalg.norm(self._rotation_vector))
        active_limits = [(self.linear_speed / distance, self.linear_acceleration / distance)] if distance > 1e-12 else []
        if angle > 1e-12:
            active_limits.append((self.angular_speed / angle, self.angular_acceleration / angle))
        self._path_speed = min((v for v, _ in active_limits), default=0.)
        self._path_acceleration = min((a for _, a in active_limits), default=0.)
        self._path_velocity = 0.
        self.object_feedback = None
        if object_feedback is not None:
            defaults = {"position_limit": .010, "rotation_limit": .08,
                        "linear_speed": .010, "linear_acceleration": .03,
                        "angular_speed": .05, "angular_acceleration": .15}
            if not isinstance(object_feedback, dict) or set(object_feedback) - defaults.keys():
                raise ValueError("object_feedback must be a mapping of supported correction limits")
            settings = {k: _positive(object_feedback.get(k, v), "object_feedback." + k)
                        for k, v in defaults.items()}
            if settings["rotation_limit"] >= math.pi / 2.:
                raise ValueError("object_feedback.rotation_limit must be less than pi/2")
            self.object_feedback = settings
            self._position_correction = _BoundedVectorCorrection(
                settings["position_limit"], settings["linear_speed"], settings["linear_acceleration"])
            # For R=Exp(r), |angular_velocity| <= |r_dot| and
            # |angular_acceleration| <= |r_ddot| + .5*|r_dot|².
            # Reserve half of the angular acceleration for each term.
            self._rotation_correction = _BoundedVectorCorrection(
                settings["rotation_limit"], min(settings["angular_speed"],
                                                math.sqrt(settings["angular_acceleration"])),
                .5 * settings["angular_acceleration"])
            self._previous_command_quaternion = self._previous_quaternion.copy()
        self._two_speed = None
        if two_speed_descent is not None:
            self._two_speed = self._configure_two_speed(two_speed_descent)

    def _configure_two_speed(self, settings):
        """Validate the opt-in two-speed descent and locate its slow zone on this segment's progress."""
        if self.reference_profile != "online":
            raise ValueError("two_speed_descent requires the online reference profile")
        keys = {"fast_linear_speed", "slow_zone_height_m", "goal_height_above_seat_m"}
        if not isinstance(settings, dict) or set(settings) != keys:
            raise ValueError("two_speed_descent requires exactly fast_linear_speed, slow_zone_height_m and "
                             "goal_height_above_seat_m")
        fast = _positive(settings["fast_linear_speed"], "two_speed_descent.fast_linear_speed")
        zone = _positive(settings["slow_zone_height_m"], "two_speed_descent.slow_zone_height_m")
        offset = settings["goal_height_above_seat_m"]
        if isinstance(offset, bool) or not isinstance(offset, (int, float, np.integer, np.floating)) \
                or not math.isfinite(offset):
            raise ValueError("two_speed_descent.goal_height_above_seat_m must be finite")
        offset = float(offset)
        if fast <= self.linear_speed:
            raise ValueError("two_speed_descent.fast_linear_speed must exceed linear_speed")
        distance = float(np.linalg.norm(self._translation))
        angle = float(np.linalg.norm(self._rotation_vector))
        descent = -float(self._translation[2])
        start_above_goal = zone - offset
        reason = None
        if distance <= 1e-12 or descent <= 1e-12:
            reason = "segment_not_descending"
        elif angle > 1e-9:
            reason = "segment_rotates"
        elif start_above_goal <= 0.:
            reason = "goal_not_below_slow_zone"
        elif descent <= start_above_goal:
            reason = "starts_inside_slow_zone"
        boundary = 0. if reason is not None else 1. - start_above_goal / descent
        return {"fast_linear_speed": fast, "slow_zone_height_m": zone, "goal_height_above_seat_m": offset,
                "descent_height_m": descent if descent > 0. else 0., "boundary": boundary,
                "fast_path_speed": fast / distance if reason is None else None,
                "fast_zone_available": reason is None, "reason": reason}

    def _trapezoid_phases(self, velocity, distance, cap, final):
        """(duration, rate) phases from ``velocity`` to ``final`` over ``distance`` with peak at most ``cap``."""
        acceleration = self._path_acceleration
        peak = min(cap, math.sqrt(acceleration * distance + .5 * (velocity * velocity + final * final)))
        peak = max(velocity, final, peak)  # Only floating-point noise at a brake switch.
        accelerate_time = max(0., (peak - velocity) / acceleration)
        accelerate_distance = max(0., (peak * peak - velocity * velocity) / (2. * acceleration))
        brake_distance = max(0., (peak * peak - final * final) / (2. * acceleration))
        cruise_time = max(0., distance - accelerate_distance - brake_distance) / peak if peak > 0. else 0.
        brake_time = max(0., (peak - final) / acceleration)
        return [(accelerate_time, acceleration), (cruise_time, 0.), (brake_time, -acceleration)]

    def _two_speed_candidate(self, dt):
        """Integrate the two-zone online profile exactly: fast cap, braked entry into the slow zone, slow cap.

        The slow-zone entry speed is the slow cap, or less when even that cannot stop by the goal. A segment
        state inside the slow zone follows the single-cap trapezoid of ``_online_candidate``.
        """
        progress, velocity = self._progress, self._path_velocity
        acceleration, slow = self._path_acceleration, self._path_speed
        remaining = max(0., 1. - progress)
        if remaining <= 1e-14 or slow == 0.:
            return 1., 0.
        boundary = self._two_speed["boundary"]
        if progress >= boundary:
            if velocity * velocity / (2. * acceleration) > remaining + 1e-10:
                raise RuntimeError("Online reference has insufficient stopping distance")
            phases = self._trapezoid_phases(velocity, remaining, slow, 0.)
        else:
            head, tail = boundary - progress, 1. - boundary
            entry = min(slow, math.sqrt(2. * acceleration * tail))
            if velocity > entry and (velocity * velocity - entry * entry) / (2. * acceleration) > head + 1e-10:
                raise RuntimeError("Online reference has insufficient stopping distance")
            reach = math.sqrt(velocity * velocity + 2. * acceleration * head)
            if reach <= entry:
                phases = [((reach - velocity) / acceleration, acceleration)]
                entry = reach
            else:
                phases = self._trapezoid_phases(velocity, head, self._two_speed["fast_path_speed"], entry)
            phases += self._trapezoid_phases(entry, tail, slow, 0.)
        distance = 0.
        time_left = dt
        for duration, rate in phases:
            interval = min(time_left, duration)
            distance += velocity * interval + .5 * rate * interval * interval
            velocity = max(0., velocity + rate * interval)
            time_left -= interval
            if time_left <= 1e-14:
                break
        if dt >= sum(duration for duration, _ in phases) - 1e-12:
            return 1., 0.
        return min(1., progress + distance), velocity

    def _two_speed_evidence(self, obj):
        zone = self._two_speed
        progress = float(self._progress)
        return {"schema_version": "igris_c_two_speed_place_descent_v1",
                "fast_linear_speed_m_s": zone["fast_linear_speed"], "slow_linear_speed_m_s": self.linear_speed,
                "slow_zone_height_m": zone["slow_zone_height_m"],
                "goal_height_above_seat_m": zone["goal_height_above_seat_m"],
                "descent_height_m": zone["descent_height_m"], "fast_zone_available": zone["fast_zone_available"],
                "fallback_reason": zone["reason"], "slow_zone_start_progress": zone["boundary"],
                "zone": "fast" if zone["fast_zone_available"] and progress < zone["boundary"] else "slow",
                "nominal_height_above_seat_m": float((1. - progress) * zone["descent_height_m"]
                                                     + zone["goal_height_above_seat_m"]),
                "measured_height_above_seat_m": float(obj[2, 3] - self._goal_object[2, 3]
                                                      + zone["goal_height_above_seat_m"]),
                "pause_and_resume_logic_changed": False}

    def retarget_collinear_goal(self, new_goal):
        """Continue an online translation without resetting its correction state.

        The new object goal must extend the existing straight, nonrotating path
        forward. Numerical comparisons below allow roundoff only (1 nm for
        collinearity, 1e-10 for rotation matrices); they are not task tolerances.
        A successful call starts a new local segment at the previous *nominal*
        reference, carrying its physical velocity and both feedback filters.
        In particular, the corrected command is never used as the new nominal.

        Return explicit acceptance/evidence; rejection changes no state. The
        caller must separately require the same episode, stage, arm, object,
        acquisition/control transforms, settings and consecutive waypoint index.
        This method accepts no new grasp or settings and never changes either.
        """
        def reject(reason, **evidence):
            return {"accepted": False, "reason": reason, **evidence}

        if self.reference_profile != "online":
            return reject("reference_profile_not_online")
        if self._two_speed is not None:
            return reject("two_speed_descent_segment")
        try:
            goal_object = pose_to_matrix(new_goal)
        except (TypeError, ValueError, OverflowError):
            return reject("invalid_goal_pose")
        goal_tcp = goal_object @ self._reference_object_to_tcp
        rotation_tolerance = 1e-10
        if (not np.allclose(goal_object[:3, :3], self._goal_object[:3, :3],
                            rtol=0., atol=rotation_tolerance)
                or not np.allclose(self._segment_start[:3, :3], self._goal_tcp[:3, :3],
                                   rtol=0., atol=rotation_tolerance)
                or not np.allclose(self._reference[:3, :3], goal_tcp[:3, :3],
                                   rtol=0., atol=rotation_tolerance)):
            return reject("goal_orientation_changed_or_prior_path_rotating")
        old_length = float(np.linalg.norm(self._translation))
        if not math.isfinite(old_length) or old_length <= 1e-12:
            return reject("prior_translation_degenerate")
        axis = self._translation / old_length
        extension = goal_tcp[:3, 3] - self._goal_tcp[:3, 3]
        forward = float(np.dot(extension, axis))
        lateral = float(np.linalg.norm(extension - forward * axis))
        if not math.isfinite(forward) or not math.isfinite(lateral):
            return reject("nonfinite_goal_extension")
        if lateral > 1e-9:
            return reject("goal_not_collinear", lateral_extension_m=lateral)
        if forward <= 1e-12:
            return reject("goal_not_forward", forward_extension_m=forward)

        translation = goal_tcp[:3, 3] - self._reference[:3, 3]
        new_length = float(np.linalg.norm(translation))
        rotation_vector = rotation_error(goal_tcp[:3, :3], self._reference[:3, :3])
        angle = float(np.linalg.norm(rotation_vector))
        velocity = float(self._path_velocity) * old_length
        if (not math.isfinite(new_length) or new_length <= 1e-12
                or not math.isfinite(velocity) or velocity < 0.):
            return reject("invalid_nominal_motion_state")
        limits = [(self.linear_speed / new_length, self.linear_acceleration / new_length)]
        if angle > 1e-12:
            limits.append((self.angular_speed / angle, self.angular_acceleration / angle))
        path_speed = min(v for v, _ in limits)
        path_acceleration = min(a for _, a in limits)
        normalized_velocity = velocity / new_length
        if (not math.isfinite(path_speed) or path_speed <= 0.
                or not math.isfinite(path_acceleration) or path_acceleration <= 0.):
            return reject("invalid_nominal_motion_limits")
        evidence = {"previous_segment_length_m": old_length,
                    "new_segment_length_m": new_length,
                    "forward_extension_m": forward, "lateral_extension_m": lateral,
                    "physical_nominal_speed_m_s": velocity,
                    "previous_normalized_path_velocity_per_s": float(self._path_velocity),
                    "normalized_path_velocity_per_s": normalized_velocity}
        if normalized_velocity > path_speed + 1e-12:
            return reject("nominal_speed_exceeds_limit", **evidence)
        stopping_fraction = normalized_velocity * normalized_velocity / (2. * path_acceleration)
        evidence["stopping_distance_m"] = stopping_fraction * new_length
        if stopping_fraction > 1. + 1e-12:
            return reject("insufficient_stopping_distance", **evidence)
        duration = max(1.875 * new_length / self.linear_speed,
                       math.sqrt((10 / math.sqrt(3)) * new_length / self.linear_acceleration),
                       1.875 * angle / self.angular_speed,
                       math.sqrt((10 / math.sqrt(3)) * angle / self.angular_acceleration))
        if not math.isfinite(duration):
            return reject("invalid_nominal_motion_limits", **evidence)

        # All validation precedes mutation. Filter objects, their value/velocity/
        # acceleration, pause state and quaternion sign references stay intact.
        self._goal_object = goal_object
        self._goal_tcp = goal_tcp
        self._segment_start = self._reference.copy()
        self._segment_start_progress = 0.
        self._translation = translation
        self._rotation_vector = rotation_vector
        self._progress = 0.
        self._elapsed = 0.
        self._duration = duration
        self._path_speed = path_speed
        self._path_acceleration = path_acceleration
        self._path_velocity = normalized_velocity
        return {"accepted": True, "reason": None, **evidence}

    def _restart_segment(self):
        self._segment_start = self._reference.copy()
        self._segment_start_progress = self._progress
        self._translation = self._goal_tcp[:3, 3] - self._reference[:3, 3]
        self._rotation_vector = rotation_error(self._goal_tcp[:3, :3], self._reference[:3, :3])
        distance = float(np.linalg.norm(self._translation))
        angle = float(np.linalg.norm(self._rotation_vector))
        # max d(10u^3-15u^4+6u^5)/du = 15/8;
        # max absolute second derivative = 10/sqrt(3).
        self._duration = max(1.875 * distance / self.linear_speed,
                             math.sqrt((10 / math.sqrt(3)) * distance / self.linear_acceleration),
                             1.875 * angle / self.angular_speed,
                             math.sqrt((10 / math.sqrt(3)) * angle / self.angular_acceleration))
        self._elapsed = 0.

    def _online_candidate(self, dt):
        """Integrate the remaining scalar trapezoid exactly, including switches.

        The peak speed follows d = (vp²-v²)/(2a) + vp²/(2a),
        capped by the common path speed limit. Splitting a timestep at each
        switch avoids an endpoint clamp that would otherwise exceed the
        configured braking acceleration. Protective feedback stops remain the
        explicit exception documented by the controller.
        """
        if self._two_speed is not None and self._two_speed["fast_zone_available"]:
            return self._two_speed_candidate(dt)
        remaining = max(0., 1. - self._progress)
        if remaining <= 1e-14 or self._path_speed == 0.:
            return 1., 0.
        velocity, acceleration = self._path_velocity, self._path_acceleration
        stopping_distance = velocity * velocity / (2. * acceleration)
        if stopping_distance > remaining + 1e-10:
            raise RuntimeError("Online reference has insufficient stopping distance")
        peak = min(self._path_speed, math.sqrt(acceleration * remaining + .5 * velocity * velocity))
        peak = max(velocity, peak)  # Only floating-point noise at the brake switch.
        accelerate_time = max(0., (peak - velocity) / acceleration)
        accelerate_distance = max(0., (peak * peak - velocity * velocity) / (2. * acceleration))
        brake_distance = peak * peak / (2. * acceleration)
        cruise_time = max(0., remaining - accelerate_distance - brake_distance) / peak
        brake_time = peak / acceleration
        distance = 0.
        time_left = dt
        for duration, rate in ((accelerate_time, acceleration), (cruise_time, 0.), (brake_time, -acceleration)):
            interval = min(time_left, duration)
            distance += velocity * interval + .5 * rate * interval * interval
            velocity = max(0., velocity + rate * interval)
            time_left -= interval
            if time_left <= 1e-14:
                break
        if dt >= accelerate_time + cruise_time + brake_time - 1e-12:
            return 1., 0.
        return min(1., self._progress + distance), velocity

    @staticmethod
    def _directional_error(error, direction):
        length = float(np.linalg.norm(direction))
        if length < 1e-10:
            return 0., float(np.linalg.norm(error))
        axis = direction / length
        projected = float(np.dot(error, axis))
        return max(0., projected), float(np.linalg.norm(error - projected * axis))

    def update(self, object_pose_w, tcp_pose_w, dt, *, pinch_confirmed=True, hold_reason=None,
               advance_hold_reason=None, feedback_hold_reason=None):
        dt = _positive(dt, "dt")
        if feedback_hold_reason is not None and (not isinstance(feedback_hold_reason, str)
                                                or not feedback_hold_reason.strip()):
            raise ValueError("feedback_hold_reason must be None or a nonempty string")
        if not isinstance(pinch_confirmed, (bool, np.bool_)):
            raise ValueError("pinch_confirmed must be a measured boolean")
        if hold_reason is not None and (not isinstance(hold_reason, str) or not hold_reason.strip()):
            raise ValueError("hold_reason must be None or a nonempty string")
        if advance_hold_reason is not None and (not isinstance(advance_hold_reason, str)
                                               or not advance_hold_reason.strip()):
            raise ValueError("advance_hold_reason must be None or a nonempty string")
        obj, tcp = pose_to_matrix(object_pose_w), pose_to_matrix(tcp_pose_w)
        relative = np.linalg.inv(obj) @ tcp
        spin = None
        if self._symmetry_axis is None:
            slip_position = float(np.linalg.norm(relative[:3, 3] - self._object_to_tcp[:3, 3]))
            slip_rotation = float(np.linalg.norm(rotation_error(relative[:3, :3], self._object_to_tcp[:3, :3])))
        else:
            # Hand-frame comparison: the rim at the pinch (or, without a pivot radius, the ring centre) and the
            # ring axis relative to the hand; the spin cancels.
            slip_position, slip_rotation, spin = symmetric_capture_errors(
                relative, self._object_to_tcp, self._symmetry_axis, self._pivot_radius)
        reason = None
        if not pinch_confirmed:
            reason = "pinch_missing"
        elif slip_position > self.pause_slip_position or slip_rotation > self.pause_slip_rotation:
            reason = "relative_slip"
        if feedback_hold_reason is not None and self._symmetry_axis is not None:
            # Hand-path transit of a symmetric ring: progress is the hand's, but its rotation about the
            # measured ring axis neither leads nor cross-tracks the reference (as for the object in object
            # mode); the planned route leaves that rotation partly unreachable at its joint command boxes.
            tracking_pose = tcp.copy()
            axis_w = obj[:3, :3] @ self._symmetry_axis
            tracking_pose[:3, :3] = align_spin(tcp[:3, :3], self._reference[:3, :3], tcp[:3, :3].T @ axis_w)[1]
        elif self.object_feedback is None or feedback_hold_reason is not None:
            # No object feedback (or the task machine carries the payload along the hand path): progress is the hand's.
            tracking_pose = tcp
        elif self._symmetry_axis is None:
            tracking_pose = obj @ self._reference_object_to_tcp
        else:
            # Progress is measured through the object spun back onto the current nominal orientation,
            # so a spin inside the pads neither leads nor cross-tracks the reference.
            reference_rotation = (self._reference @ np.linalg.inv(self._reference_object_to_tcp))[:3, :3]
            tracking_pose = (spin_aligned_object(obj, reference_rotation, self._symmetry_axis)
                             @ self._reference_object_to_tcp)
        # An advance-only hold pauses the nominal reference (as a protective stop
        # does) but leaves the bounded object feedback correcting the payload.
        result = self._advance_reference(tracking_pose, dt, reason or hold_reason or advance_hold_reason)
        if self.object_feedback is not None:
            self._apply_object_feedback(result, obj, tcp, dt, reason, hold_reason, suppress_reason=feedback_hold_reason)
        if feedback_hold_reason is not None:  # recorded only while a task suppresses the feedback
            result["feedback_hold_reason"] = feedback_hold_reason
        result.update(slip_position_m=slip_position, slip_rotation_rad=slip_rotation, object_spin_rad=spin,
                      external_hold_reason=hold_reason, advance_hold_reason=advance_hold_reason,
                      feedback_continues_during_advance_hold=bool(
                          advance_hold_reason is not None and reason is None and hold_reason is None
                          and self.object_feedback is not None))
        if self._two_speed is not None:  # declared only: undeclared references keep their schema
            result["two_speed_descent"] = self._two_speed_evidence(obj)
        return result

    def _apply_object_feedback(self, result, obj, tcp, dt, grasp_alarm, external_hold_reason=None, suppress_reason=None):
        nominal = self._reference
        object_reference = nominal @ np.linalg.inv(self._reference_object_to_tcp)
        axis = self._symmetry_axis
        if axis is None:
            ideal = object_reference @ np.linalg.inv(obj) @ tcp
            desired_position = ideal[:3, 3] - nominal[:3, 3]
            desired_rotation = rotation_error(ideal[:3, :3], nominal[:3, :3])
            reference_rotation_error = float(np.linalg.norm(rotation_error(object_reference[:3, :3], obj[:3, :3])))
            goal_rotation_error = float(np.linalg.norm(rotation_error(self._goal_object[:3, :3], obj[:3, :3])))
        else:
            # The object as if it had not spun: the ideal hand never rolls the symmetric object about
            # its own axis, and any residual correction component about that axis is projected out.
            ideal = object_reference @ np.linalg.inv(spin_aligned_object(obj, object_reference[:3, :3], axis)) @ tcp
            desired_position = ideal[:3, 3] - nominal[:3, 3]
            desired_rotation = remove_axis_component(rotation_error(ideal[:3, :3], nominal[:3, :3]),
                                                     obj[:3, :3] @ axis)
            reference_rotation_error = symmetric_pose_errors(obj, object_reference, axis)[1]
            goal_rotation_error = symmetric_pose_errors(obj, self._goal_object, axis)[1]
        if suppress_reason is not None:
            # Hand-path transit: the payload is not dragged back onto its object path. The same bounded filter
            # instead pulls the HAND toward its nominal path, compensating part of the loaded arm's servo sag
            # (run_60: 3-6 mm behind the reference) within the unchanged position limit.
            # Proportional and position-only: the measured hand position error is the desired correction (no
            # integration, so joints pinned at their command boxes cannot wind it up; run_62 wound an integrated
            # rotation correction to its 0.15 rad bound), and no rotation correction is commanded.
            desired_position = nominal[:3, 3] - tcp[:3, 3]
            desired_rotation = np.zeros(3)
        held = grasp_alarm is not None or external_hold_reason is not None
        position = self._position_correction.update(desired_position, dt, hold=held)
        rotation = self._rotation_correction.update(desired_rotation, dt, hold=held)
        command = nominal.copy()
        command[:3, 3] += position
        command[:3, :3] = _rotation_vector_matrix(rotation) @ nominal[:3, :3]
        pose = np.asarray(matrix_to_pose(command))
        if pose[3:] @ self._previous_command_quaternion < 0.:
            pose[3:] *= -1.
        self._previous_command_quaternion = pose[3:].copy()
        result["nominal_tcp_target_pose_w"] = result["tcp_target_pose_w"]
        result["tcp_target_pose_w"] = pose.tolist()
        result["object_feedback"] = {
            "enabled": True, "held_by_grasp_alarm": grasp_alarm is not None, "alarm_reason": grasp_alarm,
            "held_by_external_request": external_hold_reason is not None,
            "external_hold_reason": external_hold_reason,
            "correction_translation_w_m": position.tolist(),
            "correction_rotation_vector_w_rad": rotation.tolist(),
            "correction_linear_velocity_w_m_s": self._position_correction.velocity.tolist(),
            "correction_linear_acceleration_w_m_s2": self._position_correction.acceleration.tolist(),
            "correction_rotation_vector_velocity_w_rad_s": self._rotation_correction.velocity.tolist(),
            "correction_rotation_vector_acceleration_w_rad_s2": self._rotation_correction.acceleration.tolist(),
            "desired_correction_translation_w_m": desired_position.tolist(),
            "desired_correction_rotation_vector_w_rad": desired_rotation.tolist(),
            "desired_position_bound_reached": bool(np.linalg.norm(desired_position) >= self.object_feedback["position_limit"]),
            "desired_rotation_bound_reached": bool(np.linalg.norm(desired_rotation) >= self.object_feedback["rotation_limit"]),
            "applied_position_bound_reached": bool(np.linalg.norm(position) >= (1. - 1e-6) * self.object_feedback["position_limit"]),
            "applied_rotation_bound_reached": bool(np.linalg.norm(rotation) >= (1. - 1e-6) * self.object_feedback["rotation_limit"]),
            "measured_object_position_error_m": float(np.linalg.norm(object_reference[:3, 3] - obj[:3, 3])),
            "measured_object_rotation_error_rad": reference_rotation_error,
            "measured_object_goal_position_error_m": float(np.linalg.norm(self._goal_object[:3, 3] - obj[:3, 3])),
            "measured_object_goal_rotation_error_rad": goal_rotation_error,
            "limits": dict(self.object_feedback),
            "limits_apply_to_correction_only": True,
            "combined_reference_bounds": {
                "linear_speed_m_s": self.linear_speed + self.object_feedback["linear_speed"],
                "linear_acceleration_m_s2": self.linear_acceleration + self.object_feedback["linear_acceleration"],
                "angular_speed_rad_s": self.angular_speed + self.object_feedback["angular_speed"],
                "angular_acceleration_rad_s2": (self.angular_acceleration + self.object_feedback["angular_acceleration"]
                                                + self.angular_speed * self.object_feedback["angular_speed"]),
                "protective_stops_are_explicit_exceptions": True,
                "measured_robot_motion_is_not_bounded_by_this_report": True,
            },
            "protective_hold_may_exceed_correction_acceleration_limit": held,
            "retention_reference_changed": False,
        }
        if suppress_reason is not None:  # recorded only while a task replaces the object feedback by hand feedback
            result["object_feedback"].update(suppressed=True, suppress_reason=suppress_reason, hand_feedback=True,
                                             measured_hand_position_error_m=float(np.linalg.norm(nominal[:3, 3] - tcp[:3, 3])),
                                             measured_hand_rotation_error_rad=float(np.linalg.norm(
                                                 rotation_error(nominal[:3, :3], tcp[:3, :3]))))

    def _advance_reference(self, tcp, dt, reason=None):
        if reason is None:
            if self.reference_profile == "online":
                fraction, path_velocity = self._online_candidate(dt)
                elapsed = self._elapsed + dt
            else:
                if self._paused:
                    self._restart_segment()
                elapsed = min(self._elapsed + dt, self._duration)
                u = elapsed / self._duration if self._duration > 1e-12 else 1.
                fraction = u**3 * (10 + u * (-15 + 6 * u))
            candidate = self._segment_start.copy()
            candidate[:3, 3] += fraction * self._translation
            candidate[:3, :3] = _rotation_vector_matrix(fraction * self._rotation_vector) @ self._segment_start[:3, :3]
            position_ahead, position_cross = self._directional_error(
                candidate[:3, 3] - tcp[:3, 3], self._translation)
            rotation_ahead, rotation_cross = self._directional_error(
                rotation_error(candidate[:3, :3], tcp[:3, :3]), self._rotation_vector)
            if position_cross > self.cross_track_position or rotation_cross > self.cross_track_rotation:
                reason = "cross_track_error"
            elif position_ahead > self.position_lead or rotation_ahead > self.rotation_lead:
                reason = "tracking_lead"
            else:
                self._reference = candidate
                self._elapsed = elapsed
                self._progress = self._segment_start_progress + (1 - self._segment_start_progress) * fraction
                if self.reference_profile == "online":
                    self._path_velocity = path_velocity
        self._paused = reason is not None
        if self._paused:
            self._path_velocity = 0.
        pose = np.asarray(matrix_to_pose(self._reference))
        if np.dot(pose[3:], self._previous_quaternion) < 0:
            pose[3:] *= -1
        self._previous_quaternion = pose[3:].copy()
        position_ahead, position_cross = self._directional_error(
            self._reference[:3, 3] - tcp[:3, 3], self._translation)
        rotation_ahead, rotation_cross = self._directional_error(
            rotation_error(self._reference[:3, :3], tcp[:3, :3]), self._rotation_vector)
        return {
            "tcp_target_pose_w": pose.tolist(),
            "object_reference_pose_w": matrix_to_pose(self._reference @ np.linalg.inv(self._reference_object_to_tcp)),
            "goal_object_pose_w": matrix_to_pose(self._goal_object),
            "progress": float(self._progress), "reference_complete": bool(self._progress >= 1 - 1e-12),
            "paused": self._paused, "reason": reason,
            "reference_profile": self.reference_profile,
            "normalized_path_velocity_per_s": self._path_velocity if self.reference_profile == "online" else None,
            "protective_stop_may_exceed_reference_acceleration_limit": self._paused,
            "position_lead_m": position_ahead, "position_cross_track_m": position_cross,
            "rotation_lead_rad": rotation_ahead, "rotation_cross_track_rad": rotation_cross,
            "collision_checked": False, "physical_success_verified": False,
        }


class TcpPoseController(CrsPoseController):
    """Bound an empty hand's pose reference using measured TCP tracking.

    Shares the speed/acceleration and tracking-lead calculation with payload
    references. It receives no object state and produces no grasp evidence.
    The task machine independently requires an open hand and no object contact.
    """

    def __init__(self, tcp_pose_w, goal_tcp_pose_w, **settings):
        forbidden = {"reference_object_to_tcp", "pause_slip_position", "pause_slip_rotation", "object_feedback",
                     "object_symmetry_axis", "object_pivot_radius_m", "two_speed_descent"} & settings.keys()
        if forbidden:
            raise ValueError("Empty-hand reference cannot configure object grasp/slip: " + str(sorted(forbidden)))
        super().__init__([0., 0., 0., 1., 0., 0., 0.], tcp_pose_w, goal_tcp_pose_w, **settings)

    def update(self, tcp_pose_w, dt):
        result = self._advance_reference(pose_to_matrix(tcp_pose_w), _positive(dt, "dt"))
        result["goal_tcp_pose_w"] = result.pop("goal_object_pose_w")
        result.pop("object_reference_pose_w")
        result["reference_scope"] = "released_hand_tcp"
        return result
