"""Measured pinch acquisition stability, independent of a physics SDK.

Use one tracker for one arm/object pair and reset it at episode boundaries or
when that pair changes. This qualifies acquisition only: it does not replace
bilateral contact, retention/slip checks, or the FSM's subsequent contact dwell.
Instantaneous solver joint velocity remains useful diagnostic evidence, but is
not this window's measure of whether the loaded jaws have stopped closing.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .snapshot import pose_to_matrix


@dataclass(frozen=True)
class _Sample:
    time_s: float
    angle_rad: float
    relative_pose: np.ndarray
    pinch_confirmed: bool


class PinchSettlingTracker:
    """Require a complete, continuously sampled, stable bilateral pinch window.

    Inputs are measured world poses, xyz + scalar-first normalized quaternion.
    Relative pose is TCP in the object's frame, matching the task slip check.
    Peak-to-peak angle and maximum *pairwise* relative position/rotation changes
    must all satisfy their bounds; net displacement alone cannot hide a wobble.
    """

    def __init__(
        self,
        *,
        window_s: float = 0.2,
        maximum_angle_span_rad: float = 0.002,
        maximum_relative_position_span_m: float = 0.002,
        maximum_relative_rotation_span_rad: float = 0.02,
        maximum_sample_gap_s: float = 0.05,
    ) -> None:
        values = {
            "window_s": window_s,
            "maximum_angle_span_rad": maximum_angle_span_rad,
            "maximum_relative_position_span_m": maximum_relative_position_span_m,
            "maximum_relative_rotation_span_rad": maximum_relative_rotation_span_rad,
            "maximum_sample_gap_s": maximum_sample_gap_s,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, float(value))
        if self.maximum_sample_gap_s > self.window_s:
            raise ValueError("maximum_sample_gap_s must not exceed window_s")
        self._samples: deque[_Sample] = deque()

    def reset(self) -> None:
        """Discard acquisition history before a new episode or arm/object pair."""
        self._samples.clear()

    def update(
        self,
        time_s: float,
        joint_position_rad: float,
        tcp_pose_w: Any,
        object_pose_w: Any,
        *,
        pinch_confirmed: bool,
    ) -> dict[str, Any]:
        """Return JSON-safe diagnostics and ``settled``; invalid data raises.

        Invalid/backwards data clears history before raising. Missing contact or
        a sampling gap clears previous evidence. Repeated timestamps replace
        the last sample, so repeated reads cannot manufacture a stable duration.
        """
        try:
            stamp, angle = float(time_s), float(joint_position_rad)
            if not math.isfinite(stamp) or not math.isfinite(angle):
                raise ValueError("time_s and joint_position_rad must be finite")
            if not isinstance(pinch_confirmed, (bool, np.bool_)):
                raise ValueError("pinch_confirmed must be a measured boolean")
            tcp, obj = pose_to_matrix(tcp_pose_w), pose_to_matrix(object_pose_w)
            relative = np.eye(4)
            relative[:3, :3] = obj[:3, :3].T @ tcp[:3, :3]
            relative[:3, 3] = obj[:3, :3].T @ (tcp[:3, 3] - obj[:3, 3])
            if self._samples and stamp < self._samples[-1].time_s:
                raise ValueError("time_s moved backwards; reset between episodes")
        except (TypeError, ValueError):
            self.reset()
            raise

        gap_reset = bool(self._samples and stamp - self._samples[-1].time_s > self.maximum_sample_gap_s + 1e-9)
        if gap_reset or not pinch_confirmed:
            self.reset()
        if self._samples and stamp == self._samples[-1].time_s:
            self._samples.pop()
        self._samples.append(_Sample(stamp, angle, relative, bool(pinch_confirmed)))
        # Keep the sample at/before the cutoff, ensuring the whole window is
        # covered even when the observation period does not divide window_s.
        cutoff = stamp - self.window_s
        while len(self._samples) > 2 and self._samples[1].time_s <= cutoff + 1e-9:
            self._samples.popleft()

        samples = list(self._samples)
        duration = stamp - samples[0].time_s
        angles = np.array([sample.angle_rad for sample in samples])
        translations = np.array([sample.relative_pose[:3, 3] for sample in samples])
        rotations = np.array([sample.relative_pose[:3, :3] for sample in samples])
        angle_span = float(np.ptp(angles))
        position_span = float(np.linalg.norm(translations[:, None] - translations[None, :], axis=-1).max())
        # Frobenius inner product equals trace(R_i.T R_j).
        traces = np.einsum("aij,bij->ab", rotations, rotations)
        rotation_span = float(np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0)).max())
        covered = duration >= self.window_s - 1e-9
        continuous_pinch = all(sample.pinch_confirmed for sample in samples)
        settled = bool(
            covered and continuous_pinch
            and angle_span <= self.maximum_angle_span_rad
            and position_span <= self.maximum_relative_position_span_m
            and rotation_span <= self.maximum_relative_rotation_span_rad
        )
        return {
            "settled": settled,
            "source": "measured_joint_angle_and_object_relative_pose_window",
            "window_duration_s": float(duration),
            "required_window_s": self.window_s,
            "sample_count": len(samples),
            "window_covered": bool(covered),
            "bilateral_pinch_whole_window": continuous_pinch,
            "angle_span_rad": angle_span,
            "relative_position_span_m": position_span,
            "relative_rotation_span_rad": rotation_span,
            "maximum_angle_span_rad": self.maximum_angle_span_rad,
            "maximum_relative_position_span_m": self.maximum_relative_position_span_m,
            "maximum_relative_rotation_span_rad": self.maximum_relative_rotation_span_rad,
            "maximum_sample_gap_s": self.maximum_sample_gap_s,
            "sampling_gap_reset": gap_reset,
        }
