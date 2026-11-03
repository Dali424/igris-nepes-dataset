"""Opt-in C2 joint-waypoint references with an explicit geometric corridor.

The reference is a command, not evidence of collision freedom, grasp retention,
tracking, or dynamic balance. A bent polyline cannot be traversed with nonzero
corner speed and finite acceleration without changing geometry. This module
therefore exposes and bounds that geometry change; it never silently rounds an
existing path. All original waypoints are retained, and no joint overshoots a
segment's endpoint range. Reversals and degenerate segments stop explicitly.
"""
from __future__ import annotations

import math
import numpy as np

from .posture_control import JointPathReference, _finite_joint_value

CONTINUOUS_PROFILE = "bounded_c2"


def _split_half(control):
    """de Casteljau split; leading dimensions batch independent curves."""
    work = np.array(control, dtype=float, copy=True)
    degree = work.shape[-2] - 1
    left, right = [work[..., 0, :]], [work[..., -1, :]]
    for _ in range(degree):
        work = (work[..., :-1, :] + work[..., 1:, :]) * .5
        left.append(work[..., 0, :])
        right.append(work[..., -1, :])
    return np.stack(left, axis=-2), np.stack(right[::-1], axis=-2)


def _peak_bound(control, subdivisions=7):
    """Analytic Bernstein hull bound, tightened by exact curve subdivision.

    Every polynomial value is a convex combination of its Bernstein control
    values. Subdividing preserves that property; the largest absolute control
    value across all children is an upper bound, rather than a sampled maximum.
    """
    curves = np.asarray(control, dtype=float)[None, ...]
    for _ in range(subdivisions):
        left, right = _split_half(curves)
        curves = np.concatenate((left, right), axis=0)
    return float(np.max(np.abs(curves)))


def _evaluate(control, u):
    work = np.array(control, dtype=float, copy=True)
    while len(work) > 1:
        work = (1. - u) * work[:-1] + u * work[1:]
    return work[0]


def _corridor_bound(controls, points):
    """Bound distance to each original segment in the max-joint norm.

    Each control point is paired with its Euclidean projection on the chord.
    A convex combination of those projections is on the chord, and the same
    combination of residuals stays in the declared max-joint-norm ball. This
    proves the bound for the whole curve, not just the control points.
    """
    maximum = 0.
    for polygon, a, b in zip(controls, points[:-1], points[1:]):
        delta = b - a
        squared = float(delta @ delta)
        if squared == 0.:
            residual = polygon - a
        else:
            projection = np.clip((polygon - a) @ delta / squared, 0., 1.)
            residual = polygon - (a + projection[:, None] * delta)
        maximum = max(maximum, float(np.max(np.abs(residual))))
    return maximum


class ContinuousJointPathReference:
    """C2 quintic Bézier segments with bounded waypoint blending.

    The API matches JointPathReference. ``max_joint_deviation_rad`` bounds the
    maximum joint-coordinate distance to the original segment; it is not a TCP
    clearance and cannot certify contact/collision safety. Segment endpoints
    are exact, all joint coordinates stay between their respective endpoints,
    and acceleration is zero on both sides of every waypoint. Shared interior
    velocities eliminate stops where geometry and the corridor permit them.

    Minimum segment durations are honored. Segment time dilation and shared
    waypoint velocity reductions preserve C2 continuity and the corridor while
    satisfying analytic per-joint speed/acceleration bounds. No wall clock is read.
    Optional per-segment caps (runtime_v26, ``segment_max_speeds_rad_s`` and
    ``segment_max_accelerations_rad_s2``, validated as for JointPathReference)
    replace the path limits segment by segment in the acceleration envelope,
    the dilation and the final certification. The initial shared waypoint
    speeds keep the path limit: dilating a capped segment also reduces its two
    knot velocities, so a slow segment does not stretch its neighbours' time
    allocation. Absent, every number is unchanged.
    """

    def __init__(self, start, waypoints, *, joint_names, max_speed_rad_s,
                 max_acceleration_rad_s2, segment_min_durations_s,
                 max_joint_deviation_rad, segment_max_speeds_rad_s=None,
                 segment_max_accelerations_rad_s2=None):
        # Reuse the established shape, finiteness, ownership, and limit checks.
        validated = JointPathReference(
            start, waypoints, joint_names=joint_names,
            max_speed_rad_s=max_speed_rad_s,
            max_acceleration_rad_s2=max_acceleration_rad_s2,
            segment_min_durations_s=segment_min_durations_s,
            segment_max_speeds_rad_s=segment_max_speeds_rad_s,
            segment_max_accelerations_rad_s2=segment_max_accelerations_rad_s2)
        corridor = _finite_joint_value(max_joint_deviation_rad, "Joint path corridor")
        if corridor < 0.:
            raise ValueError("Joint path corridor must be nonnegative")
        self.joint_names = validated.joint_names
        self.max_speed_rad_s = validated.max_speed_rad_s
        self.max_acceleration_rad_s2 = validated.max_acceleration_rad_s2
        self.max_joint_deviation_rad = corridor
        self.segment_max_speeds_rad_s = validated.segment_max_speeds_rad_s
        self.segment_max_accelerations_rad_s2 = validated.segment_max_accelerations_rad_s2
        self._points = validated._points.copy()
        delta = np.diff(self._points, axis=0)
        lengths = np.max(np.abs(delta), axis=1)
        # Per-segment limits (the path limits unless per-segment caps are declared).
        speed_limits = np.asarray(self.segment_max_speeds_rad_s or (self.max_speed_rad_s,) * len(lengths), dtype=float)
        acceleration_limits = np.asarray(self.segment_max_accelerations_rad_s2
                                         or (self.max_acceleration_rad_s2,) * len(lengths), dtype=float)
        self.cusp_waypoint_indices = []
        scalar_speed = np.full(len(self._points), self.max_speed_rad_s)
        scalar_speed[0] = scalar_speed[-1] = 0.
        for index in range(1, len(self._points) - 1):
            # An exact stop is required for reversal, a >=90-degree cusp, or a
            # repeated waypoint. Individual reversing joints also get speed0.
            if float(delta[index - 1] @ delta[index]) <= 0.:
                scalar_speed[index] = 0.
                self.cusp_waypoint_indices.append(index - 1)
        # A forward/backward scalar arc-speed envelope is only an initial time
        # allocation. The Bézier derivative bounds below certify the final
        # multidimensional trajectory, including every rounded corner.
        for index, length in enumerate(lengths):
            scalar_speed[index + 1] = min(scalar_speed[index + 1],
                math.sqrt(scalar_speed[index]**2 + 2. * float(acceleration_limits[index]) * length))
        for index in range(len(lengths) - 1, -1, -1):
            scalar_speed[index] = min(scalar_speed[index],
                math.sqrt(scalar_speed[index + 1]**2 + 2. * float(acceleration_limits[index]) * lengths[index]))
        durations = np.asarray(segment_min_durations_s, dtype=float).copy()
        for index, length in enumerate(lengths):
            speed_sum = scalar_speed[index] + scalar_speed[index + 1]
            travel_time = (2. * length / speed_sum if speed_sum > 0.
                           else 2. * math.sqrt(length / float(acceleration_limits[index])))
            durations[index] = max(durations[index], travel_time)
        directions = np.divide(delta, lengths[:, None], out=np.zeros_like(delta),
                               where=lengths[:, None] > 0.)
        velocities = np.zeros_like(self._points)
        for index in range(1, len(self._points) - 1):
            a, b = directions[index - 1], directions[index]
            same_direction = a * b > 0.
            velocities[index, same_direction] = (scalar_speed[index]
                * 2. * a[same_direction] * b[same_direction]
                / (a[same_direction] + b[same_direction]))

        # Ordered control values imply monotonicity and no per-joint overshoot.
        # P2 <= P3 requires 2*T*(|v0|+|v1|)/5 <= |delta|.
        tangent_scales = np.ones(len(self._points))
        for index, duration in enumerate(durations):
            used = 2. * duration * (np.abs(velocities[index]) + np.abs(velocities[index + 1])) / 5.
            active = used > 0.
            if np.any(active):
                scale = min(1., float(np.min(np.abs(delta[index, active]) / used[active])))
                tangent_scales[index:index + 2] = np.minimum(tangent_scales[index:index + 2], scale)
        velocities *= tangent_scales[:, None]

        def polygons():
            result = []
            for index, duration in enumerate(durations):
                a, b = self._points[index:index + 2]
                va, vb = velocities[index:index + 2]
                result.append(np.asarray((a, a + duration * va / 5., a + 2. * duration * va / 5.,
                                          b - 2. * duration * vb / 5., b - duration * vb / 5., b)))
            return np.asarray(result)

        controls = polygons()
        tangent_scales = np.ones(len(self._points))
        numerical_tolerance = 16. * np.finfo(float).eps * max(1., float(np.max(np.abs(self._points))))
        for index, polygon in enumerate(controls):
            a, b = self._points[index:index + 2]
            chord = b - a
            squared = float(chord @ chord)
            projection = (np.clip((polygon - a) @ chord / squared, 0., 1.)
                          if squared > 0. else np.zeros(6))
            distances = np.max(np.abs(polygon - (a + projection[:, None] * chord)), axis=1)
            for knot, deviation in ((index, max(distances[1:3])),
                                    (index + 1, max(distances[3:5]))):
                if deviation > corridor + numerical_tolerance:
                    tangent_scales[knot] = min(tangent_scales[knot], corridor / deviation)
        velocities *= tangent_scales[:, None]
        controls = polygons()
        initial_durations = durations.copy()
        initial_controls, initial_velocities = controls.copy(), velocities.copy()
        # Dilate only demanding segments. Reducing each shared knot velocity by
        # the larger adjacent factor keeps T*v from growing on either side.
        # Thus every new control point lies between its old value and the
        # segment endpoint: coordinate monotonicity and the convex corridor
        # remain valid. Recompute derivative bounds, since shrinking the end
        # velocities changes the speed/acceleration polynomial inside a segment.
        for _ in range(32):
            controls = polygons()
            speed_bounds = np.asarray([_peak_bound(5. * np.diff(p, axis=0) / t)
                                       for p, t in zip(controls, durations)])
            acceleration_bounds = np.asarray([_peak_bound(20. * np.diff(p, n=2, axis=0) / t**2)
                                              for p, t in zip(controls, durations)])
            factors = np.maximum(1., np.maximum(speed_bounds / speed_limits,
                np.sqrt(acceleration_bounds / acceleration_limits)))
            if float(np.max(factors)) <= 1. + 1e-11:
                break
            factors *= 1. + 1e-12
            durations *= factors
            knot_factors = np.maximum(np.r_[1., factors], np.r_[factors, 1.])
            velocities /= knot_factors[:, None]
        controls = polygons()
        speed_bounds = np.asarray([_peak_bound(5. * np.diff(p, axis=0) / t)
                                   for p, t in zip(controls, durations)])
        acceleration_bounds = np.asarray([_peak_bound(20. * np.diff(p, n=2, axis=0) / t**2)
                                          for p, t in zip(controls, durations)])
        # Final uniform stretch certifies any remaining bound (also if the
        # bounded iteration count was exhausted), without changing geometry.
        dilation = max(1., float(np.max(speed_bounds / speed_limits)),
                       math.sqrt(float(np.max(acceleration_bounds / acceleration_limits))))
        initial_speeds = np.asarray([_peak_bound(5. * np.diff(p, axis=0) / t)
                                     for p, t in zip(initial_controls, initial_durations)])
        initial_accelerations = np.asarray([_peak_bound(20. * np.diff(p, n=2, axis=0) / t**2)
                                            for p, t in zip(initial_controls, initial_durations)])
        initial_speed, initial_acceleration = float(np.max(initial_speeds)), float(np.max(initial_accelerations))
        initial_dilation = max(1., float(np.max(initial_speeds / speed_limits)),
                               math.sqrt(float(np.max(initial_accelerations / acceleration_limits))))
        self.retiming_strategy = "segment_dilation"
        # Local velocity reductions can propagate to neighboring segments.
        # Keep the independently certified uniform solution if it is faster.
        if float(np.sum(initial_durations)) * initial_dilation < float(np.sum(durations)) * dilation:
            controls, durations, velocities = initial_controls, initial_durations, initial_velocities
            speed_bounds = np.asarray([initial_speed])
            acceleration_bounds = np.asarray([initial_acceleration])
            dilation = initial_dilation
            self.retiming_strategy = "uniform_dilation"
        dilation *= 1. + 1e-12
        self._controls = controls
        self.segment_durations = tuple(float(t * dilation) for t in durations)
        self._boundaries = np.concatenate(([0.], np.cumsum(self.segment_durations)))
        self._elapsed = 0.
        self.geometry_deviation_bound_rad = _corridor_bound(controls, self._points)
        self.speed_bound_rad_s = float(np.max(speed_bounds)) / dilation
        self.acceleration_bound_rad_s2 = float(np.max(acceleration_bounds)) / dilation**2
        self.interior_waypoint_velocities = velocities[1:-1] / dilation
        self.segment_time_dilations = tuple(np.asarray(self.segment_durations) / initial_durations)
        self.time_dilation = self.duration / float(np.sum(initial_durations))
        if self.segment_max_speeds_rad_s is not None or self.segment_max_accelerations_rad_s2 is not None:
            # Opt-in caps: the certified (Bernstein hull) bound of every final segment, for its own cap.
            self.segment_speed_bounds_rad_s = tuple(_peak_bound(5. * np.diff(p, axis=0) / t)
                                                    for p, t in zip(controls, self.segment_durations))
            self.segment_acceleration_bounds_rad_s2 = tuple(_peak_bound(20. * np.diff(p, n=2, axis=0) / t**2)
                                                            for p, t in zip(controls, self.segment_durations))

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
        index = int(np.searchsorted(self._boundaries, self._elapsed, side="right")) - 1
        return min(max(index, 0), len(self.segment_durations) - 1)

    @property
    def start(self):
        return dict(zip(self.joint_names, map(float, self._points[0])))

    @property
    def final(self):
        return dict(zip(self.joint_names, map(float, self._points[-1])))

    def sample(self, time_s):
        """Return position, velocity, acceleration dictionaries without advancing."""
        time_s = _finite_joint_value(time_s, "Joint path sample time")
        if time_s < 0. or time_s > self.duration:
            raise ValueError("Joint path sample time must be within the reference")
        if time_s == self.duration:
            zero = dict.fromkeys(self.joint_names, 0.)
            return self.final, zero.copy(), zero.copy()
        index = min(int(np.searchsorted(self._boundaries, time_s, side="right")) - 1,
                    len(self.segment_durations) - 1)
        duration = self.segment_durations[index]
        u = min(1., max(0., (time_s - self._boundaries[index]) / duration))
        control = self._controls[index]
        values = (_evaluate(control, u),
                  _evaluate(5. * np.diff(control, axis=0) / duration, u),
                  _evaluate(20. * np.diff(control, n=2, axis=0) / duration**2, u))
        return tuple(dict(zip(self.joint_names, map(float, value))) for value in values)

    def update(self, dt):
        dt = _finite_joint_value(dt, "Joint path dt")
        if dt <= 0.:
            raise ValueError("Joint path dt must be positive and finite")
        self._elapsed = min(self._elapsed + dt, self.duration)
        return self.final if self.complete else self.sample(self._elapsed)[0]
