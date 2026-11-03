"""Opt-in pressed-key completion of the guarded CRS approach (runtime_v26); measured gates only.

The NEPES CRS pinch keys the fixed-pad tip against Pin_10 along the TCP finger axis (z_TCP). The grasp goal lies
about 0.9 mm beyond that key contact, so a hand pressed on the key stops short of the goal along the finger axis
while its error across it must stay small. The 3-D ``completion_position_m`` bound mixes both (run_125 keyed only
after its x error drifted down), and a force-only rule is refuted by the native traces: the Pin_10 tip force
flickers between 0 and 3 N from one 50 Hz step to the next (longest sustained run 0.10-0.40 s), so ">= 0.3 N for
0.1 s" fired with the TCP 1.49 mm from the goal (speedup_v2/review_pick r07).

Declared as ``guarded_approach.pressed_key_completion`` the approach completes only when, continuously for
``dwell_s`` (at least thresholds.reach_dwell_s), the measured TCP error to the grasp goal (TCP frame) has a
component along ``finger_axis_tcp`` within ``max_axis_error_m`` and a remainder across it within
``max_lateral_error_m``, while every other approach gate holds: hand open, the thresholds TCP pose, live contact
evidence, ``rotation_tolerance_rad`` and the approach_axis_alignment pad-tip axial gate (``point_tcp_m``). With the
optional ``min_mean_tip_force_n`` the MEAN fixed-pad tip force over the trailing ``dwell_s`` window must also
reach it: the summed ``normal_force_n`` of the fixed jaw's (``<Arm>_Hand`` sensor) contact patches whose unit
normal, turned into the measured TCP frame, lies within ``max_normal_angle_rad`` of ``tip_normal_axis_tcp``
(signed: the native tip patches report +z_TCP). Every sample of that window needs complete patch evidence
(available, not truncated); otherwise the run restarts. It replaces ``completion_position_m`` (declaring both is
rejected) and never changes the reference (GuardedHandleApproach accepts and ignores it).

Admissible envelope (runtime_v26 review fix, replayed on the recorded run_116/124/125 approach rows,
tests/v26_native_approach_rows.py): ``dwell_s`` is at least 0.2 s (0.1 s completed run_124's unpressed 0 N state at
24.54 s), and a geometry wider than 1.2 mm on either error (the native keyed states carry a 1.0-1.45 mm lateral
twist) needs the tip-force gate with ``min_mean_tip_force_n`` >= 0.8 N and ``dwell_s`` >= 0.3 s: at 1.5 mm a
geometry-only rule, 0.3 N, or 0.5-0.8 N over 0.2-0.3 s all completed run_124's unkeyed approach at 24.06-24.46 s.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from copy import deepcopy
import math

import numpy as np

PRESSED_KEY_REQUIRED_KEYS = frozenset(("finger_axis_tcp", "max_axis_error_m", "max_lateral_error_m", "dwell_s"))
PRESSED_KEY_FORCE_KEYS = frozenset(("min_mean_tip_force_n", "tip_normal_axis_tcp", "max_normal_angle_rad"))
PRESSED_KEY_MAXIMUM_AXIS_ERROR_M = .0015
PRESSED_KEY_MAXIMUM_LATERAL_ERROR_M = .0015
PRESSED_KEY_DWELL_RANGE_S = (.2, .5)
PRESSED_KEY_MAXIMUM_MEAN_TIP_FORCE_N = 3.
# Wider than this on either error, the tip-force gate is mandatory with at least this mean force and dwell.
PRESSED_KEY_GEOMETRY_ONLY_MAXIMUM_M = .0012
PRESSED_KEY_WIDE_MINIMUM_TIP_FORCE_N = .8
PRESSED_KEY_WIDE_MINIMUM_DWELL_S = .3
PRESSED_KEY_MAXIMUM_NORMAL_ANGLE_RAD = .8
UNIT_TOLERANCE = 1e-6
TIP_FORCE_SOURCE = "physx_contact_patch_buffer: fixed-jaw sensor normal_force_n with the normal near tip_normal_axis_tcp"


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _unit(value, name):
    if isinstance(value, (str, bytes, Mapping)) or not hasattr(value, "__len__") or len(value) != 3:
        raise ValueError(f"{name} must be a unit 3-vector")
    vector = [_number(item, name) for item in value]
    if abs(math.sqrt(sum(item * item for item in vector)) - 1.) > UNIT_TOLERANCE:
        raise ValueError(f"{name} must be a unit 3-vector (length within {UNIT_TOLERANCE:g} of 1)")
    return vector


def validate_pressed_key_completion(settings, name="guarded_approach.pressed_key_completion"):
    """Check the opt-in settings; returns the checked values (the profile itself is never rewritten)."""
    if (not isinstance(settings, Mapping) or not PRESSED_KEY_REQUIRED_KEYS <= set(settings)
            or set(settings) - PRESSED_KEY_REQUIRED_KEYS - PRESSED_KEY_FORCE_KEYS):
        detail = []
        if isinstance(settings, Mapping):
            if set(settings) - PRESSED_KEY_REQUIRED_KEYS - PRESSED_KEY_FORCE_KEYS:
                detail.append("unknown: " + ", ".join(sorted(map(str, set(settings) - PRESSED_KEY_REQUIRED_KEYS
                                                              - PRESSED_KEY_FORCE_KEYS))))
            if PRESSED_KEY_REQUIRED_KEYS - set(settings):
                detail.append("missing: " + ", ".join(sorted(PRESSED_KEY_REQUIRED_KEYS - set(settings))))
        raise ValueError(f"{name} requires exactly finger_axis_tcp, max_axis_error_m, max_lateral_error_m and dwell_s "
                         "(optional together: min_mean_tip_force_n, tip_normal_axis_tcp, max_normal_angle_rad)"
                         + (" (" + "; ".join(detail) + ")" if detail else ""))
    result = {"finger_axis_tcp": _unit(settings["finger_axis_tcp"], name + ".finger_axis_tcp")}
    for key, maximum in (("max_axis_error_m", PRESSED_KEY_MAXIMUM_AXIS_ERROR_M),
                         ("max_lateral_error_m", PRESSED_KEY_MAXIMUM_LATERAL_ERROR_M)):
        value = _number(settings[key], f"{name}.{key}")
        if not 0. < value <= maximum:
            raise ValueError(f"{name}.{key} must be in (0, {maximum:g}] m")
        result[key] = value
    dwell = _number(settings["dwell_s"], name + ".dwell_s")
    if not PRESSED_KEY_DWELL_RANGE_S[0] <= dwell <= PRESSED_KEY_DWELL_RANGE_S[1]:
        raise ValueError(f"{name}.dwell_s must be in [{PRESSED_KEY_DWELL_RANGE_S[0]:g}, "
                         f"{PRESSED_KEY_DWELL_RANGE_S[1]:g}] s")
    result["dwell_s"] = dwell
    force_keys = set(settings) & PRESSED_KEY_FORCE_KEYS
    if force_keys and force_keys != PRESSED_KEY_FORCE_KEYS:
        raise ValueError(f"{name}.min_mean_tip_force_n needs tip_normal_axis_tcp and max_normal_angle_rad "
                         "(declare all three or none)")
    if force_keys:
        force = _number(settings["min_mean_tip_force_n"], name + ".min_mean_tip_force_n")
        if not 0. <= force <= PRESSED_KEY_MAXIMUM_MEAN_TIP_FORCE_N:
            raise ValueError(f"{name}.min_mean_tip_force_n must be in [0, {PRESSED_KEY_MAXIMUM_MEAN_TIP_FORCE_N:g}] N")
        angle = _number(settings["max_normal_angle_rad"], name + ".max_normal_angle_rad")
        if not 0. < angle <= PRESSED_KEY_MAXIMUM_NORMAL_ANGLE_RAD:
            raise ValueError(f"{name}.max_normal_angle_rad must be in (0, {PRESSED_KEY_MAXIMUM_NORMAL_ANGLE_RAD:g}] rad")
        result.update(min_mean_tip_force_n=force,
                      tip_normal_axis_tcp=_unit(settings["tip_normal_axis_tcp"], name + ".tip_normal_axis_tcp"),
                      max_normal_angle_rad=angle)
    wide = [key for key in ("max_axis_error_m", "max_lateral_error_m")
            if result[key] > PRESSED_KEY_GEOMETRY_ONLY_MAXIMUM_M + 1e-12]
    if wide and not (force_keys and result["min_mean_tip_force_n"] >= PRESSED_KEY_WIDE_MINIMUM_TIP_FORCE_N - 1e-12
                     and dwell >= PRESSED_KEY_WIDE_MINIMUM_DWELL_S - 1e-12):
        raise ValueError(f"{name}." + "/".join(wide) + f" above {PRESSED_KEY_GEOMETRY_ONLY_MAXIMUM_M:g} m needs the "
                         f"tip-force gate with min_mean_tip_force_n >= {PRESSED_KEY_WIDE_MINIMUM_TIP_FORCE_N:g} N and "
                         f"dwell_s >= {PRESSED_KEY_WIDE_MINIMUM_DWELL_S:g} s (weaker rules completed run_124's unkeyed "
                         "approach)")
    return result


def tcp_goal_error(tcp, goal, axis_tcp):
    """The measured TCP error to the goal in the measured TCP frame, split along and across the unit axis."""
    error = tcp[:3, :3].T @ (goal[:3, 3] - tcp[:3, 3])
    axis = np.asarray(axis_tcp, dtype=float)
    along = float(error @ axis)
    return along, float(np.linalg.norm(error - along * axis))


def fixed_pad_tip_force(contact, arm, tcp, normal_axis_tcp, max_normal_angle_rad):
    """Summed fixed-jaw contact-patch normal force near the tip normal; returns (force_n, None) or (None, reason).

    Fail closed: the hand-object evidence must be available with complete patch details (not truncated), the fixed
    jaw (``<Arm>_Hand``) must be one of the sensors and every fixed-jaw patch needs a finite nonnegative force and
    a finite nonzero normal. No patch on the fixed jaw is a valid 0 N.
    """
    if not isinstance(contact, Mapping) or contact.get("available") is not True:
        return None, "contact_evidence_unavailable"
    if contact.get("contact_details_available") is not True or contact.get("contact_details_truncated") is not False:
        return None, "contact_details_unavailable_or_truncated"
    details, sensors = contact.get("contact_details"), contact.get("sensor_paths")
    fixed = f"{arm.title()}_Hand"
    if (not isinstance(details, list) or not isinstance(sensors, list)
            or sum(1 for path in sensors if str(path).rsplit("/", 1)[-1] == fixed) != 1):
        return None, "fixed_jaw_sensor_missing"
    axis, threshold = np.asarray(normal_axis_tcp, dtype=float), math.cos(max_normal_angle_rad)
    rotation, total = tcp[:3, :3], 0.
    for detail in details:
        if not isinstance(detail, Mapping) or str(detail.get("sensor_path")).rsplit("/", 1)[-1] != fixed:
            continue
        force, normal = detail.get("normal_force_n"), detail.get("normal_w")
        if (isinstance(force, bool) or not isinstance(force, (int, float)) or not math.isfinite(force) or force < 0
                or isinstance(normal, (str, bytes, Mapping)) or not hasattr(normal, "__len__") or len(normal) != 3):
            return None, "malformed_contact_detail"
        normal = np.asarray(normal, dtype=float)
        length = float(np.linalg.norm(normal))
        if not np.isfinite(normal).all() or length < .5:
            return None, "malformed_contact_detail"
        if float((rotation.T @ normal) @ axis) / length >= threshold:
            total += float(force)
    return total, None


class PressedKeyCompletion:
    """Continuous-run state of the opt-in rule; the task machine calls ``measure`` then ``complete`` per approach step.

    ``measure`` evaluates one sample's geometry (and tip force) and returns whether it may join the run; the
    machine ANDs it with its other approach gates and runs its own dwell over the result. ``complete`` then
    tracks that final condition's continuous run (cleared by any failing sample, like the machine's dwell) and
    adds the mean-force test over the trailing ``dwell_s`` window. ``evidence`` is the last sample's; the step
    that completed the approach keeps it as ``completion``.
    """

    def __init__(self, settings, arm):
        self.settings = validate_pressed_key_completion(settings)
        if arm not in ("left", "right"):
            raise ValueError("pressed_key_completion needs a left or right arm")
        self.arm = arm
        self.dwell_s = self.settings["dwell_s"]
        self.force_gate = "min_mean_tip_force_n" in self.settings
        self.reset()

    def reset(self):
        self._since = None
        self._window = deque()
        self._sample = None
        self.evidence = None
        self.completion = None

    def measure(self, tcp, goal, contact):
        along, across = tcp_goal_error(tcp, goal, self.settings["finger_axis_tcp"])
        geometry = bool(abs(along) <= self.settings["max_axis_error_m"]
                        and across <= self.settings["max_lateral_error_m"])
        force, reason = None, None
        if self.force_gate:
            force, reason = fixed_pad_tip_force(contact, self.arm, tcp, self.settings["tip_normal_axis_tcp"],
                                                self.settings["max_normal_angle_rad"])
        self._sample = {"axis_error_m": along, "lateral_error_m": across, "geometry_satisfied": geometry,
                        "tip_force_n": force, "tip_force_invalid_reason": reason}
        return bool(geometry and (not self.force_gate or force is not None))

    def complete(self, now, condition, dwell_passed):
        if self._sample is None:
            raise ValueError("pressed_key_completion.complete needs a measured sample")
        if condition:
            if self._since is None:
                self._since = now
            self._window.append((now, self._sample["tip_force_n"]))
            while self._window and self._window[0][0] < now - self.dwell_s - 1e-10:
                self._window.popleft()
        else:
            self._since = None
            self._window.clear()
        mean = None
        if self.force_gate and self._window:
            mean = float(sum(force for _, force in self._window) / len(self._window))
        passed = bool(condition and dwell_passed
                      and (not self.force_gate or (mean is not None and mean >= self.settings["min_mean_tip_force_n"])))
        self.evidence = {
            "finger_axis_tcp": list(self.settings["finger_axis_tcp"]),
            "axis_error_m": self._sample["axis_error_m"], "max_axis_error_m": self.settings["max_axis_error_m"],
            "lateral_error_m": self._sample["lateral_error_m"],
            "max_lateral_error_m": self.settings["max_lateral_error_m"],
            "geometry_satisfied": self._sample["geometry_satisfied"], "condition_satisfied": bool(condition),
            "continuous_dwell_s": 0. if self._since is None else float(now - self._since),
            "dwell_s": self.dwell_s, "completed": passed, "measurement_time_s": float(now)}
        if self.force_gate:
            self.evidence.update(tip_force_n=self._sample["tip_force_n"],
                                 tip_force_invalid_reason=self._sample["tip_force_invalid_reason"],
                                 window_mean_tip_force_n=mean, window_sample_count=len(self._window),
                                 min_mean_tip_force_n=self.settings["min_mean_tip_force_n"],
                                 tip_force_source=TIP_FORCE_SOURCE)
        self._sample = None
        if passed:
            self.completion = deepcopy(self.evidence)
        return passed

    def published(self, stage, status, time_s):
        """The command evidence: the live one in a running approach, the completion record on the step that
        completed it (already in close), None otherwise."""
        if stage == "approach" and status == "running":
            return deepcopy(self.evidence)
        if self.completion is not None and self.completion["measurement_time_s"] == time_s:
            return deepcopy(self.completion)
        return None
