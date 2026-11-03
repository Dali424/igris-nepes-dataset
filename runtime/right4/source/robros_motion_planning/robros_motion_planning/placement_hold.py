"""Retain a Cartesian stop briefly across a load-only measurement dropout.

This helper owns only a hand command reference. It neither supplies support
evidence nor accumulates opening dwell. Geometry, capture and sensor identity
must remain valid at every update; only a new loaded sample renews the grace.

Opt-in ``latch_dwell_s`` (placement_support.latch_dwell_s, 0.02-0.3 s): the
stop is latched only once valid loaded samples have been seen continuously for
that long, at the measured TCP of the sample that completes the dwell, instead
of at the first loaded sample. Before the latch the hold is inactive (the
caller keeps its nominal descent running); any unloaded or invalid sample
restarts that dwell. After the latch the dropout grace is unchanged.

The pending phase is bounded (runtime_v26 review fix): a pending descent pushes
a ring that already rests on the seat, and a support flicker shorter than the
dwell would otherwise keep it pushing for the whole stage. The hold therefore
falls back to the runtime_v25 first-sample latch, for the rest of the stage, as
soon as ``LATCH_PENDING_BUDGET_S`` has passed since the first valid loaded
sample of the stage, or when the caller reports a ``latch_now`` reason (the
machine does so for a JIG load of at least 1.2 m g, enough to stop pressing, and
for a JIG drift of 1 mm since the place entry; ``latch_now_reason``). A fallback
is sticky until ``reset``: the first loaded sample from then on latches at once.
"""
import math

import numpy as np

LATCH_DWELL_BOUNDS_S = (.02, .3)
# Longest pending phase of one stage, from its first valid loaded sample (cumulative, flicker included).
LATCH_PENDING_BUDGET_S = 1.
LATCH_FALLBACK_REASONS = ("pending_budget", "support_force_reference", "jig_drift")
# Immediate-latch triggers the machine reports while a latch is pending: a raw JIG normal force of at least this
# multiple of the measured object weight (the seat already carries more than the ring: pressing on only adds
# force), and a measured JIG position drift of at least this much since the place-entry JIG pose.
LATCH_FORCE_REFERENCE_FRACTION = 1.2
LATCH_JIG_DRIFT_M = .001
GRAVITY_M_S2 = 9.81


def latch_now_reason(support, jig_pose_w, place_entry_jig_pose_w):
    """The machine's immediate-latch reason for a pending latch (None: keep the dwell rule).

    ``support`` is the sample's raw support evidence (normal_force_n, object_mass_kg); the JIG poses are 4x4.
    """
    if place_entry_jig_pose_w is not None:
        drift = float(np.linalg.norm(np.asarray(jig_pose_w, float)[:3, 3]
                                     - np.asarray(place_entry_jig_pose_w, float)[:3, 3]))
        if not math.isfinite(drift) or drift >= LATCH_JIG_DRIFT_M - 1e-12:
            return "jig_drift"
    force, mass = (support or {}).get("normal_force_n"), (support or {}).get("object_mass_kg")
    if (isinstance(force, (int, float)) and isinstance(mass, (int, float)) and math.isfinite(force)
            and math.isfinite(mass) and mass > 0 and force >= LATCH_FORCE_REFERENCE_FRACTION * mass * GRAVITY_M_S2):
        return "support_force_reference"
    return None


def validate_latch_dwell_s(value):
    """placement_support.latch_dwell_s: a finite sustained-load latch dwell in [0.02, 0.3] seconds."""
    low, high = LATCH_DWELL_BOUNDS_S
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(value) or not low <= value <= high):
        raise ValueError("placement_support.latch_dwell_s must be a finite sustained-load latch dwell in [0.02, 0.3] s")
    return float(value)


class PlacementHold:
    def __init__(self, grace_s, latch_dwell_s=None):
        if isinstance(grace_s, bool) or not math.isfinite(grace_s) or not 0 < grace_s <= .2:
            raise ValueError("Placement hold grace must be in (0, 0.2] seconds")
        self.grace_s = float(grace_s)
        self.latch_dwell_s = None if latch_dwell_s is None else validate_latch_dwell_s(latch_dwell_s)
        self.reset()

    def reset(self):
        self.target = None
        self.last_loaded_time_s = None
        self.last_time_s = None
        self.evidence = None
        # Opt-in sustained-load latch state (unused without latch_dwell_s).
        self.loaded_since_s = None
        self.latch_time_s = None
        self.first_loaded_s = None
        self.latch_fallback = None
        self.latch_reason = None

    @property
    def latch_pending(self):
        """A valid loaded dwell is running but has not latched yet (always False without latch_dwell_s)."""
        return bool(self.latch_dwell_s is not None and self.target is None and self.loaded_since_s is not None)

    def _latch(self, now, measured_tcp, reason=None):
        target = np.asarray(measured_tcp, dtype=float)
        if target.shape != (4, 4) or not np.isfinite(target).all():
            raise ValueError("Placement hold needs a finite measured TCP")
        self.target = target.copy()
        self.latch_time_s = now
        self.latch_reason = reason

    def update(self, now, *, loaded, valid, measured_tcp, latch_now=None):
        if not math.isfinite(now) or (self.last_time_s is not None and now <= self.last_time_s):
            raise ValueError("Placement hold needs increasing finite observation times")
        latch = self.latch_dwell_s
        if latch_now is not None and (latch is None or latch_now not in LATCH_FALLBACK_REASONS[1:]):
            raise ValueError("Placement hold latch_now needs latch_dwell_s and one of "
                             + ", ".join(LATCH_FALLBACK_REASONS[1:]))
        self.last_time_s = now
        age = None if self.last_loaded_time_s is None else now - self.last_loaded_time_s
        if latch is not None and latch_now is not None and self.latch_fallback is None:
            self.latch_fallback = latch_now
        if not valid:
            self.target = self.last_loaded_time_s = None
            self.loaded_since_s = None
        elif loaded:
            if self.target is None:
                if latch is None:
                    self._latch(now, measured_tcp)
                else:
                    if self.first_loaded_s is None:
                        self.first_loaded_s = now
                    if self.loaded_since_s is None:
                        self.loaded_since_s = now
                    if self.latch_fallback is None and now - self.first_loaded_s >= LATCH_PENDING_BUDGET_S - 1e-10:
                        self.latch_fallback = "pending_budget"
                    if self.latch_fallback is not None:
                        self._latch(now, measured_tcp, self.latch_fallback)
                    elif now - self.loaded_since_s >= latch - 1e-10:
                        self._latch(now, measured_tcp, "sustained_load")
            self.last_loaded_time_s = now
            age = 0.
        elif age is None or age > self.grace_s + 1e-10:
            self.target = self.last_loaded_time_s = None
            self.loaded_since_s = None
        elif self.target is None:
            # A pending sustained-load latch needs continuous support: an unloaded sample restarts it.
            self.loaded_since_s = None
        active = self.target is not None
        self.evidence = {"schema_version": "igris_c_placement_hold_v1", "measurement_time_s": now,
            "active": active, "fresh_support": bool(loaded), "valid": bool(valid),
            "in_dropout_grace": bool(active and not loaded), "dropout_grace_s": self.grace_s,
            "last_loaded_time_s": self.last_loaded_time_s, "last_loaded_age_s": age,
            "opening_support_and_dwell_unchanged": True}
        if latch is not None:
            self.evidence.update(
                latch_dwell_s=latch, loaded_since_s=self.loaded_since_s, latch_pending=self.latch_pending,
                continuous_loaded_s=(None if self.loaded_since_s is None else float(now - self.loaded_since_s)),
                latch_time_s=self.latch_time_s if active else None,
                latch_reason=self.latch_reason if active else None,
                pending_budget_s=LATCH_PENDING_BUDGET_S, first_loaded_s=self.first_loaded_s,
                pending_elapsed_s=(None if self.first_loaded_s is None else float(now - self.first_loaded_s)),
                latch_fallback=self.latch_fallback)
        return None if not active else self.target.copy()
