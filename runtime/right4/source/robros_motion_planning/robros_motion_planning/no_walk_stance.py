"""Opt-in no-walk stance for the catalog-v2 upright pickup/load (``navigation_control.no_walk``).

The robot keeps the HugWBC stance of its upright start for the whole episode:
the runtime commands an exactly zero base velocity in every stage (the
navigate stages included), so HugWBC never steps. ``navigate_source`` still
proves the measured upright readiness (its station gate becomes a stance check
that fails fast instead of walking), and ``navigate_target`` completes after
the positive reach dwell (at least two control steps, so the recording keeps a
side_step frame) with the pelvis hold and the grasp retained. The measured pelvis
(``Link_Waist_Pitch``) is reported against the first measured stance and the
planned manipulation pelvis; ``stationary_pelvis_hold`` stays the only drift
gate. Nothing here writes robot or object state or commands a joint.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Real

import numpy as np

NO_WALK_SCHEMA = "igris_c_no_walk_stance_v1"
NO_WALK_KEYS = frozenset(("planned_pelvis_xy_yaw",))
STATION_FAILURE = "no_walk_stance_outside_source_station"
_UPRIGHT_SEQUENCE = "upright_pickup_load_v1"


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def validate_no_walk(task):
    """Validate the opt-in block of one task; return a normalized copy (``None`` when absent)."""
    navigation = task.get("navigation_control", {})
    if not isinstance(navigation, Mapping) or "no_walk" not in navigation:
        return None
    settings = navigation["no_walk"]
    if not isinstance(settings, Mapping) or set(settings) != NO_WALK_KEYS:
        raise ValueError("navigation_control.no_walk needs exactly planned_pelvis_xy_yaw")
    planned = settings["planned_pelvis_xy_yaw"]
    if not isinstance(planned, (list, tuple)) or len(planned) != 3:
        raise ValueError("no_walk.planned_pelvis_xy_yaw needs world x, y (m) and yaw (rad)")
    planned = [_finite(value, "no_walk.planned_pelvis_xy_yaw") for value in planned]
    if abs(planned[2]) > math.pi:
        raise ValueError("no_walk.planned_pelvis_xy_yaw yaw must lie in [-pi, pi]")
    if task.get("stage_sequence") != _UPRIGHT_SEQUENCE:
        raise ValueError("navigation_control.no_walk requires the upright pickup/load stage sequence")
    if "stationary_pelvis_hold" not in navigation:
        raise ValueError("navigation_control.no_walk requires stationary_pelvis_hold as its drift gate")
    if "single_step" in navigation:
        raise ValueError("navigation_control.no_walk and single_step are mutually exclusive (no_walk never steps)")
    if navigation.get("stabilize_carry") is True or navigation.get("compensate_base_motion") is True:
        raise ValueError("navigation_control.no_walk cannot stabilize or compensate a walking carry")
    if task.get("base_routes"):
        raise ValueError("navigation_control.no_walk cannot declare base_routes")
    return {"planned_pelvis_xy_yaw": planned}


def _xy_yaw(pelvis):
    pose = np.asarray(pelvis, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or math.hypot(pose[0, 0], pose[1, 0]) < 1e-8:
        raise ValueError("no_walk needs a finite measured pelvis pose with a defined heading")
    return float(pose[0, 3]), float(pose[1, 3]), math.atan2(pose[1, 0], pose[0, 0])


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class NoWalkStance:
    """Measured pelvis drift of a no-walk episode: a report, never a gate or a command."""

    def __init__(self, settings):
        self.planned = list(map(float, settings["planned_pelvis_xy_yaw"]))
        self._anchor = None
        self._current = None
        self._entries = []
        self._maximum = None
        self._stage_maxima = {}
        self._station_miss_since = None

    def _sample(self, stage, time_s, pelvis):
        x, y, yaw = _xy_yaw(pelvis)
        ax, ay, ayaw = self._anchor["pelvis_xy_yaw"]
        dx, dy = x - ax, y - ay
        c, s = math.cos(ayaw), math.sin(ayaw)
        ex, ey = x - self.planned[0], y - self.planned[1]
        return {"stage_id": stage, "time_s": float(time_s), "pelvis_xy_yaw": [x, y, yaw],
                "drift_from_anchor_xy_m": [dx, dy], "drift_from_anchor_m": math.hypot(dx, dy),
                "drift_from_anchor_heading_frame_m": [c * dx + s * dy, -s * dx + c * dy],
                "yaw_drift_from_anchor_rad": _wrap(yaw - ayaw),
                "error_from_plan_xy_m": [ex, ey], "error_from_plan_m": math.hypot(ex, ey),
                "yaw_error_from_plan_rad": _wrap(yaw - self.planned[2])}

    def anchor(self, stage, time_s, pelvis):
        """The first measured stance (the settled upright start) is the drift origin."""
        x, y, yaw = _xy_yaw(pelvis)
        self._anchor = {"stage_id": stage, "time_s": float(time_s), "pelvis_xy_yaw": [x, y, yaw]}
        self.update(stage, time_s, pelvis)

    def update(self, stage, time_s, pelvis):
        if self._anchor is None:
            raise ValueError("no_walk drift needs its first measured stance")
        sample = self._sample(stage, time_s, pelvis)
        self._current = sample
        if self._maximum is None or sample["drift_from_anchor_m"] > self._maximum["drift_from_anchor_m"]:
            self._maximum = deepcopy(sample)
        stage_maximum = self._stage_maxima.setdefault(stage, {
            "drift_from_anchor_m": 0., "abs_yaw_drift_from_anchor_rad": 0., "error_from_plan_m": 0.})
        stage_maximum["drift_from_anchor_m"] = max(stage_maximum["drift_from_anchor_m"], sample["drift_from_anchor_m"])
        stage_maximum["abs_yaw_drift_from_anchor_rad"] = max(stage_maximum["abs_yaw_drift_from_anchor_rad"],
                                                             abs(sample["yaw_drift_from_anchor_rad"]))
        stage_maximum["error_from_plan_m"] = max(stage_maximum["error_from_plan_m"], sample["error_from_plan_m"])
        return sample

    def enter(self, stage, time_s, pelvis):
        """Stage-entry snapshot (the observation that completed the previous stage)."""
        self._entries.append(self._sample(stage, time_s, pelvis))

    def station_failure(self, readiness, gates, time_s, dwell_s, tolerances):
        """Fail fast when the torso stands outside the source station while every other gate holds.

        Without walking the torso cannot reach a source station it settled
        outside of; waiting for the stage timeout would only hide that. The
        stance error is geometric (station tolerances), not the station gate,
        whose settled-arrival dwell may still be running.
        """
        others = all(readiness[key] for key in gates if key != "station_reached")
        position, yaw = readiness["base_station_position_error_m"], readiness["base_station_yaw_error_rad"]
        outside = position is not None and yaw is not None and (position > tolerances[0] or yaw > tolerances[1])
        if others and outside:
            if self._station_miss_since is None:
                self._station_miss_since = float(time_s)
            if time_s - self._station_miss_since >= dwell_s - 1e-10:
                return STATION_FAILURE
        else:
            self._station_miss_since = None
        return None

    def evidence(self, *, full=True):
        """The drift report; ``full=False`` (per control step) omits the per-stage history."""
        result = {"schema_version": NO_WALK_SCHEMA, "base_command": "exactly_zero_in_every_stage",
                  "navigate_stages": "complete_without_stepping", "planned_pelvis_xy_yaw": list(self.planned),
                  "pelvis_measurement": "Link_Waist_Pitch_world_xy_and_wrapped_yaw",
                  "drift_gate": "stationary_pelvis_hold", "report_only": True,
                  "anchor": deepcopy(self._anchor), "current": deepcopy(self._current),
                  "maximum_drift": deepcopy(self._maximum)}
        if full:
            result.update(stage_entries=deepcopy(self._entries), stage_maxima=deepcopy(self._stage_maxima))
        return result
