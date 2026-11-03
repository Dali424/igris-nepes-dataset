"""Opt-in causal filter of the measured JIG support force in the place stage (runtime_v26 C3).

``tasks.<task>.placement_support.filter_s`` (0 < filter_s <= 0.3 s) lets the
place-stage support decision (the opening dwell, the placement hold and the
supported-grip-relief activation) use a filtered JIG normal force instead of
the instantaneous sample. The raw support evidence and its
``jig_support_confirmed`` gate stay in the trace unchanged, and every release,
settle and admission success gate keeps using the raw measurement.

Filter (causal trailing boxcar mean with zero-order hold):

    F_f(t) = (1 / filter_s) * integral over (t - filter_s, t] of F_zoh

where each place observation i holds its measured normal force F_i over the
interval (t_{i-1}, t_i] since the previous place observation. The filter is
reset at the place-entry observation (the one on which the machine entered
``place``); time before that observation counts as zero force, because the ring
is still above the seat there, so a step to F needs a full ``filter_s`` of
loaded samples to reach F. At the 0.02 s control period and filter_s = 0.2 s
this is the mean of the last ten samples. An unavailable or non-finite support
sample has no filtered value (not confirmed) and clears the history: the next
filtered value again starts from zero. The confirmation threshold is the
sample's own ``minimum_force_n`` (placement_support.minimum_load_fraction of the
measured object weight); nothing here lowers it.

Sustained support (runtime_v26 review fix): a mean can be produced by one or two
impulsive samples while the ring is off the seat for the rest of the window.
Confirmation therefore also needs ``sustained_force_n``, the highest force that
the zero-order-held raw samples reached or exceeded during at least
``SUSTAINED_FRACTION`` (30 %) of the window (time outside the observed window
counts as zero force): ``confirmed = filtered >= minimum and sustained >=
minimum``, i.e. raw samples at or above the threshold cover at least 30 % of the
window as well. For any threshold F, ``sustained_force_n >= F`` holds exactly
when raw samples at or above F cover 30 % of the window, so the supported-grip
relief applies the same rule to its own activation load. Replayed on the
recorded raw support of native runs 149/150 (filter_s 0.2 s) the rule changes
none of their place confirmations; one 7 N spike among zeros covers only 10 %.

The opening of the jaw is irreversible, so the place OPENING dwell also needs
continuous seat contact over the window: ``opening_supported = confirmed and
window_minimum_force_n >= CONTACT_FLOOR_FRACTION * minimum`` (25 % of the
threshold, 0.2 m g at 0.8; time outside the observed window counts as zero). A
ring that bounces on and off the seat (2 m g / 0 N every other sample, a 1.0 m g
mean) is confirmed for the hold and the relief but never opens the jaw; the final
opening windows of native runs 149/150 never fell below 0.73 N (0.41 of 1.785 N).
"""
import math
from numbers import Real

import numpy as np

SCHEMA = "igris_c_placement_support_filter_v1"
FILTER_MAXIMUM_S = .3
# Share of the window that raw samples must cover at or above a threshold (sustained_force_n).
SUSTAINED_FRACTION = .3
# The opening dwell's continuous-contact floor, as a fraction of the sample's support threshold.
CONTACT_FLOOR_FRACTION = .25
_EPSILON_S = 1e-12


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)


def validate_filter_s(value):
    """placement_support.filter_s: a finite window in (0, 0.3] seconds."""
    if not _finite(value) or not 0. < value <= FILTER_MAXIMUM_S:
        raise ValueError("placement_support.filter_s must be a finite window in (0, 0.3] seconds")
    return float(value)


class PlacementSupportFilter:
    """Causal boxcar mean of the measured JIG normal force over the place stage (see the module docstring)."""

    def __init__(self, filter_s):
        self.filter_s = validate_filter_s(filter_s)
        self._segments = []
        self._last_time_s = self._reset_time_s = None
        self.evidence = None

    def reset(self, time_s):
        """Restart at the place-entry observation; its own force is not counted (zero-length interval)."""
        if not _finite(time_s):
            raise ValueError("placement_support filter needs a finite reset time")
        self._segments = []
        self._last_time_s = self._reset_time_s = float(time_s)
        self.evidence = None

    def update(self, time_s, support):
        """Add one place observation's support evidence; return the filtered evidence for it."""
        if self._last_time_s is None:
            raise ValueError("placement_support filter must be reset at the place entry")
        if not _finite(time_s) or time_s <= self._last_time_s:
            raise ValueError("placement_support filter needs increasing finite observation times")
        now, previous = float(time_s), self._last_time_s
        self._last_time_s = now
        support = support if isinstance(support, dict) else {}
        force, minimum = support.get("normal_force_n"), support.get("minimum_force_n")
        valid = (support.get("available") is True and _finite(force) and _finite(minimum) and minimum > 0)
        window_start = now - self.filter_s
        filtered = sustained = lowest = None
        if valid:
            self._segments.append((previous, now, float(force)))
            self._segments = [segment for segment in self._segments if segment[1] > window_start + _EPSILON_S]
            integral = sum(value * max(0., end - max(start, window_start)) for start, end, value in self._segments)
            filtered = float(integral / self.filter_s)
            sustained = self._sustained(window_start)
            lowest = self._lowest(window_start)
        else:
            # An unavailable or invalid sample is never bridged: restart from zero after it.
            self._segments = []
        covered = 0. if not self._segments else float(now - max(self._segments[0][0], window_start))
        self.evidence = {
            "schema_version": SCHEMA, "measurement_time_s": now, "filter_s": self.filter_s,
            "filter": "causal_trailing_boxcar_zero_order_hold_zero_before_place_entry",
            "reset_time_s": self._reset_time_s, "window_start_s": float(window_start),
            "observed_window_s": covered, "sample_count": len(self._segments),
            "sample_valid": bool(valid),
            "raw_normal_force_n": float(force) if _finite(force) else None,
            "filtered_normal_force_n": filtered,
            "sustained_fraction": SUSTAINED_FRACTION, "sustained_force_n": sustained,
            "window_minimum_force_n": lowest, "contact_floor_fraction": CONTACT_FLOOR_FRACTION,
            "minimum_force_n": float(minimum) if _finite(minimum) else None,
            "confirmed": bool(filtered is not None and filtered >= minimum and sustained >= minimum),
            "raw_gate_unchanged": True}
        self.evidence["opening_supported"] = bool(self.evidence["confirmed"]
                                                  and lowest >= CONTACT_FLOOR_FRACTION * minimum)
        return dict(self.evidence)

    def _lowest(self, window_start):
        """The lowest raw force over the window (time outside the observed window counts as zero force)."""
        covered = sum(max(0., end - max(start, window_start)) for start, end, _ in self._segments)
        values = [value for start, end, value in self._segments if end - max(start, window_start) > _EPSILON_S]
        if covered < self.filter_s - 1e-9:
            values.append(0.)
        return float(min(values))

    def _sustained(self, window_start):
        """The highest force held for at least SUSTAINED_FRACTION of the window (uncovered time: zero force)."""
        pieces = sorted(((value, max(0., end - max(start, window_start))) for start, end, value in self._segments),
                        key=lambda piece: -piece[0])
        covered = sum(duration for _, duration in pieces)
        pieces.append((0., max(0., self.filter_s - covered)))
        required, total = SUSTAINED_FRACTION * self.filter_s - _EPSILON_S, 0.
        for value, duration in sorted(pieces, key=lambda piece: -piece[0]):
            total += duration
            if total >= required:
                return float(value)
        return float(min(value for value, _ in pieces))
