"""Measured waist targets synchronized to an object's accepted path progress.

This module commands joints only. It does not anchor the pelvis, move a base,
change grasp evidence, or establish collision clearance or physical success.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real

from .posture_control import ArmPostureGuide


WAIST_JOINTS = ("Joint_Waist_Yaw", "Joint_Waist_Roll", "Joint_Waist_Pitch")


class SynchronizedPosturePath:
    """Follow a three-joint guide, retaining measured load compensation.

    Before advancing the object reference, call ``tracking_status`` at its
    current progress and pass ``hold_reason='waist_tracking'`` to the object
    controller when requested. Then call ``update`` with the object's accepted
    progress. Continue this feedback even while the object reference is held.
    A new instance is required when a stage resets progress to zero.
    """

    def __init__(self, waypoints, joint_limits, measured_positions,
                 previous_commands, *, tracking_tolerance_rad=.01,
                 max_command_speed_rad_s=.08, tracking_envelope_rad=.2,
                 gain_s_inv=.5):
        settings = (tracking_tolerance_rad, max_command_speed_rad_s,
                    tracking_envelope_rad, gain_s_inv)
        if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in settings):
            raise ValueError("Posture path settings must be positive and finite")
        if max_command_speed_rad_s > .08 or tracking_envelope_rad > .2:
            raise ValueError("Posture path speed/envelope cannot exceed .08 rad/s and .2 rad")
        if tracking_tolerance_rad >= tracking_envelope_rad:
            raise ValueError("Posture tracking tolerance must be smaller than the command envelope")
        self.names = WAIST_JOINTS
        self.guide = ArmPostureGuide(waypoints, joint_limits, self.names)
        self.limits = {}
        self.commands = {}
        for name in self.names:
            limits = joint_limits[name]
            if len(limits) != 2:
                raise ValueError("Posture joint needs lower and upper bounds: " + name)
            lo, hi = map(float, limits)
            if not all(map(math.isfinite, (lo, hi))) or lo >= hi:
                raise ValueError("Invalid posture joint limits: " + name)
            self.limits[name] = (lo, hi)
            value = previous_commands.get(name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError("Invalid previous posture command: " + name)
            self.commands[name] = float(value)
        self.speed = float(max_command_speed_rad_s)
        self.envelope = float(tracking_envelope_rad)
        self.tolerance = float(tracking_tolerance_rad)
        self.gain = float(gain_s_inv)
        self.progress = 0.
        self.reference = self.guide.sample(0.)
        measured = self._measurements(measured_positions)
        if any(abs(self.commands[n] - measured[n]) > self.envelope for n in self.names):
            raise ValueError("Initial posture command exceeds the measured tracking envelope")

    def _measurements(self, positions):
        if not isinstance(positions, Mapping):
            raise ValueError("Posture path needs measured joint positions")
        result = {}
        for name in self.names:
            value = positions.get(name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError("Missing or invalid measured posture joint: " + name)
            result[name] = float(value)
        return result

    def _sample(self, progress):
        if isinstance(progress, bool) or not math.isfinite(progress) or not 0 <= progress <= 1:
            raise ValueError("Posture path progress must lie in [0, 1]")
        if progress < self.progress - 1e-12:
            raise ValueError("Posture path progress cannot move backwards; reset the stage controller")
        return self.guide.sample(float(progress))

    def tracking_status(self, progress, measured_positions):
        """Read-only gate against the current, not a future, object reference."""
        reference = self._sample(progress)
        measured = self._measurements(measured_positions)
        errors = {n: reference[n] - measured[n] for n in self.names}
        maximum = max(map(abs, errors.values()))
        hold = maximum > self.tolerance
        return {"progress": float(progress), "reference_positions": reference,
                "measured_errors_rad": errors, "maximum_tracking_error_rad": maximum,
                "tracking_tolerance_rad": self.tolerance,
                "hold_object_reference": hold,
                "hold_reason": "waist_tracking" if hold else None,
                "physical_success_verified": False}

    def update(self, progress, measured_positions, dt):
        """Return bounded targets and diagnostics, committing all joints together.

        Reference-delta feedforward avoids requiring a persistent tracking
        error merely to move. Integral measured-error correction compensates
        load. There is no hidden integral state beyond the previous command.
        Incompatible rate/tracking constraints raise without changing state.
        """
        if isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Posture path dt must be positive and finite")
        status = self.tracking_status(progress, measured_positions)
        reference = status["reference_positions"]
        measured = self._measurements(measured_positions)
        commands, details = {}, {}
        for name in self.names:
            previous = self.commands[name]
            reference_rate = (reference[name] - self.reference[name]) / dt
            feedback_rate = self.gain * (reference[name] - measured[name])
            raw_rate = reference_rate + feedback_rate
            rate = max(-self.speed, min(self.speed, raw_rate))
            lo, hi = self.limits[name]
            lower = max(lo, measured[name] - self.envelope, previous - self.speed * dt)
            upper = min(hi, measured[name] + self.envelope, previous + self.speed * dt)
            if lower > upper:
                raise RuntimeError("posture_tracking_rate_constraints_incompatible:" + name)
            command = max(lower, min(upper, previous + rate * dt))
            commands[name] = command
            details[name] = {"reference_rad": reference[name], "measured_rad": measured[name],
                "reference_rate_rad_s": reference_rate, "feedback_rate_rad_s": feedback_rate,
                "command_rad": command, "command_rate_rad_s": (command - previous) / dt,
                "command_error_rad": command - measured[name],
                "rate_limited": abs(raw_rate) > self.speed,
                "tracking_envelope_limited": abs(command - measured[name]) >= self.envelope - 1e-12,
                "physical_bound_limited": command <= lo + 1e-12 or command >= hi - 1e-12}
        self.commands = commands
        self.reference = reference
        self.progress = float(progress)
        return dict(commands), {**status, "joints": details,
            "reference_complete": self.progress >= 1.,
            "measured_reference_arrived": not status["hold_object_reference"],
            "scope": "waist_joint_feedback_at_shared_object_progress",
            "pelvis_fixed_or_balance_verified": False}
