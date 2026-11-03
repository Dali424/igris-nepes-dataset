"""Opt-in single short loaded step in ``navigate_target`` (``navigation_control.single_step``).

After the in-place handover the holding hand keeps the ring while the robot takes ONE short step
(for example 0.15 m to its left, toward the JIG); the stationary pelvis hold then protects every
later stage again around the new stance. Absent the block, every profile, machine command,
runtime path and admission rule is unchanged.

Inside ``navigate_target`` (the only stage that may command the base)::

    pre_step_posture -> stepping -> settling -> complete

1. ``pre_step_posture`` (only with ``pre_step_posture_path``): the declared 19-joint path runs as a
   ``joint_posture`` command through the PostureStageExecutor while the base command stays exactly
   zero. It is complete under the posture-stage rules (reference complete, measured arrival within
   ``arrival_tolerance_rad``, still within ``settle_velocity_rad_s``, base quiet, grasp retained,
   continuously for its ``settle_dwell_s``). Without the path the goal is latched at stage entry.
2. Goal latch (the first navigate_target control step after the pre-step posture is complete): the
   measured torso (base_link) and pelvis (Link_Waist_Pitch) poses are latched and
   ``goal = torso_xy + R(pelvis_yaw) @ offset_pelvis_heading_m`` at the latched torso height, with
   the latched torso yaw. It is published as ``base_goal_pose_w``.
3. ``stepping``: the runtime commands the base toward the goal (base_velocity_command, predictive
   braking, pelvis heading frame, BaseCommandLimiter): the only nonzero base commands of the episode.
4. One-shot brake: at the first observation with the torso within ``brake_radius_m`` of the goal, or
   once the commanded travel (sum of the applied planar base command speed times the control
   interval since the latch) reaches ``|offset_pelvis_heading_m|``, braking latches and is never
   cleared in the stage; from that control step the base command is exactly zero (no corrective
   command, no re-command after a torso sway).
5. ``settling``: once the measured torso is slow (``arrival_linear_speed_m_s`` /
   ``arrival_yaw_speed_rad_s`` of navigation_control) continuously for ``settle_dwell_s``, the
   PELVIS displacement in the latched pelvis heading frame must lie in ``window_m`` and the pelvis
   yaw change within ``max_yaw_change_rad``: ``accepted`` (the stage passes once the grasp is also
   retained for the reach dwell), else the episode fails ``single_step_outside_window``.
6. On the navigate_target exit the stationary pelvis hold is re-anchored at the measured pelvis, so
   its drift limits protect the post-step stages. The hold's drift check is suspended only from the
   goal latch to that exit.

Profile schema (``tasks.crs_load_jig.navigation_control.single_step``)::

    {"offset_pelvis_heading_m": [0.0, 0.15],      # [fwd, left] (m) in the latched pelvis heading frame,
                                                  # 0 < |offset| <= 0.25
     "brake_radius_m": 0.08,                       # torso-goal distance that latches braking, 0 < r < |offset|
     "window_m": [[-0.02, 0.04], [0.09, 0.15]],   # settled pelvis displacement [[fwd_min, fwd_max],
                                                  # [left_min, left_max]] (m), each within +-0.3, min < max,
                                                  # the box must exclude the zero displacement
     "max_yaw_change_rad": 0.10,                  # settled |pelvis yaw change|, (0, 0.35]
     "settle_dwell_s": 1.0,                       # torso slow this long before the verdict, [0.1, 5]
     # optional
     "source_hold": true,                        # exactly zero commands in unrecorded preparation
     "pre_step_posture_path": {...}}              # one 19-joint path (posture_paths schema, no hold mode)

``source_hold`` is a strict boolean opt-in. When true, prerequisites/navigate_source
keep an exactly zero base command and initial_station_alignment is false. Existing
upright readiness and dwell remain required; a settled source-station miss fails
fast using the same geometric station test as no_walk. It does not disable the
later loaded step. Absent or false preserves legacy source alignment behavior.

Requirements: the upright pickup/load sequence and ``stationary_pelvis_hold`` (whose identical
source/target station rule is waived: base_stations.target is unused). It cannot be combined with
``no_walk``, ``stationary_feet_hold``, ``turn_in_place``, ``stabilize_carry`` /
``compensate_base_motion``, ``base_routes.target`` or ``stage_postures.navigate_target``. The
predictive-braking envelope keeps using ``target_position_tolerance_m`` / ``target_yaw_tolerance_rad``.

This module validates the block and keeps the step state (shared by the machine and the collection
admission replay); it commands nothing and writes no robot or object state.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

import numpy as np

from .nepes_tasks import UPRIGHT_PICKUP_SEQUENCE, _finite, _validate_posture_path
from .snapshot import matrix_to_pose

SINGLE_STEP_SCHEMA = "igris_c_single_step_v1"
SINGLE_STEP_STAGE = "navigate_target"
REQUIRED_KEYS = frozenset(("offset_pelvis_heading_m", "brake_radius_m", "window_m", "max_yaw_change_rad",
                           "settle_dwell_s"))
PRE_STEP_PATH_KEY = "pre_step_posture_path"
OPTIONAL_KEYS = frozenset((PRE_STEP_PATH_KEY, "source_hold"))
OUTSIDE_WINDOW = "single_step_outside_window"
PHASES = ("inactive", "pre_step_posture", "stepping", "settling", "complete")
ACCEPTED, REJECTED = "accepted", "outside_window"
BRAKE_TRIGGERS = ("brake_radius", "commanded_travel")
GOAL_DEFINITION = ("latched_torso_xy_plus_latched_pelvis_heading_rotation_of_offset;latched_torso_height;"
                   "latched_torso_yaw")
WINDOW_FRAME = "Link_Waist_Pitch_world_xy_displacement_in_the_latched_pelvis_heading_frame_fwd_left"
TRAVEL_DEFINITION = "sum_of_applied_planar_base_command_speed_times_control_interval_since_the_goal_latch"
_MAX_OFFSET_M = .25
_MAX_WINDOW_M = .3
_MAX_YAW_CHANGE_RAD = .35
_SETTLE_DWELL_BOUNDS_S = (.1, 5.)
_EXCLUSIVE = (("no_walk", "no_walk never steps"), ("stationary_feet_hold", "the step moves the feet"),
              ("turn_in_place", "the step keeps its heading"))


def validate_single_step(task):
    """Validate the opt-in block of one task; return a normalized copy (``None`` when absent)."""
    from .posture_hold import is_applied_hold
    navigation = task.get("navigation_control", {})
    if not isinstance(navigation, Mapping) or "single_step" not in navigation:
        return None
    settings = navigation["single_step"]
    if (not isinstance(settings, Mapping) or not REQUIRED_KEYS <= set(settings)
            or set(settings) - REQUIRED_KEYS - OPTIONAL_KEYS):
        raise ValueError("navigation_control.single_step needs exactly " + ", ".join(sorted(REQUIRED_KEYS))
                         + " (optional: " + ", ".join(sorted(OPTIONAL_KEYS)) + ")")
    if "source_hold" in settings and type(settings["source_hold"]) is not bool:
        raise ValueError("single_step.source_hold must be boolean")
    if task.get("stage_sequence") != UPRIGHT_PICKUP_SEQUENCE:
        raise ValueError("navigation_control.single_step requires the upright pickup/load stage sequence")
    if "stationary_pelvis_hold" not in navigation:
        raise ValueError("navigation_control.single_step requires stationary_pelvis_hold (re-anchored after the step)")
    for key, reason in _EXCLUSIVE:
        if key in navigation:
            raise ValueError(f"navigation_control.single_step and {key} are mutually exclusive ({reason})")
    if navigation.get("stabilize_carry") is True or navigation.get("compensate_base_motion") is True:
        raise ValueError("navigation_control.single_step cannot stabilize or compensate a walking carry")
    routes = task.get("base_routes", {})
    if isinstance(routes, Mapping) and routes.get("target"):
        raise ValueError("navigation_control.single_step latches its own goal; base_routes.target must be empty")
    postures = task.get("stage_postures", {})
    if isinstance(postures, Mapping) and SINGLE_STEP_STAGE in postures:
        raise ValueError("navigation_control.single_step.pre_step_posture_path is the only navigate_target posture; "
                         "stage_postures.navigate_target is not available")
    offset = settings["offset_pelvis_heading_m"]
    if not isinstance(offset, list) or len(offset) != 2:
        raise ValueError("single_step.offset_pelvis_heading_m needs [fwd, left] in metres")
    offset = [_finite(value, "single_step.offset_pelvis_heading_m") for value in offset]
    length = math.hypot(*offset)
    if not 0. < length <= _MAX_OFFSET_M:
        raise ValueError(f"single_step.offset_pelvis_heading_m must be a nonzero step of at most {_MAX_OFFSET_M} m")
    radius = _finite(settings["brake_radius_m"], "single_step.brake_radius_m")
    if not 0. < radius < length:
        raise ValueError("single_step.brake_radius_m must be positive and smaller than |offset_pelvis_heading_m| "
                         "(a goal inside the brake radius never steps)")
    window = settings["window_m"]
    if (not isinstance(window, list) or len(window) != 2
            or any(not isinstance(axis, list) or len(axis) != 2 for axis in window)):
        raise ValueError("single_step.window_m needs [[fwd_min, fwd_max], [left_min, left_max]] in metres")
    window = [[_finite(value, "single_step.window_m") for value in axis] for axis in window]
    if any(abs(value) > _MAX_WINDOW_M for axis in window for value in axis) or any(low >= high for low, high in window):
        raise ValueError(f"single_step.window_m bounds must satisfy min < max within +-{_MAX_WINDOW_M} m")
    if all(low <= 0. <= high for low, high in window):
        raise ValueError("single_step.window_m must exclude the zero displacement (a step that never happened "
                         "would pass)")
    yaw = _finite(settings["max_yaw_change_rad"], "single_step.max_yaw_change_rad")
    if not 0. < yaw <= _MAX_YAW_CHANGE_RAD:
        raise ValueError(f"single_step.max_yaw_change_rad must lie in (0, {_MAX_YAW_CHANGE_RAD}]")
    dwell = _finite(settings["settle_dwell_s"], "single_step.settle_dwell_s")
    if not _SETTLE_DWELL_BOUNDS_S[0] <= dwell <= _SETTLE_DWELL_BOUNDS_S[1]:
        raise ValueError("single_step.settle_dwell_s must lie in [{}, {}] s".format(*_SETTLE_DWELL_BOUNDS_S))
    result = {"offset_pelvis_heading_m": offset, "brake_radius_m": radius, "window_m": window,
              "max_yaw_change_rad": yaw, "settle_dwell_s": dwell}
    if "source_hold" in settings:
        result["source_hold"] = settings["source_hold"]
    if PRE_STEP_PATH_KEY in settings:
        path = settings[PRE_STEP_PATH_KEY]
        if is_applied_hold(path):
            raise ValueError("single_step.pre_step_posture_path is a designed joint path; hold_last_applied is not "
                             "available")
        _validate_posture_path(SINGLE_STEP_STAGE, path, prefix=f"navigation_control.single_step.{PRE_STEP_PATH_KEY}",
                               max_waypoints=256 if "transport_constraints" in task else 8)
        result[PRE_STEP_PATH_KEY] = deepcopy(dict(path))
    return result


def arrival_limits(task):
    """The navigation_control arrival speeds the settle uses (the defaults of the settled arrival)."""
    navigation = task.get("navigation_control", {})
    return {"arrival_linear_speed_m_s": float(navigation.get("arrival_linear_speed_m_s", .03)),
            "arrival_yaw_speed_rad_s": float(navigation.get("arrival_yaw_speed_rad_s", .12))}


def single_step_contract(task):
    """The run.json controller_contract.single_step block; the collection admission requires it verbatim."""
    settings = validate_single_step(task)
    contract = {"schema_version": SINGLE_STEP_SCHEMA, "stage_id": SINGLE_STEP_STAGE,
            **{key: deepcopy(settings[key]) for key in sorted(REQUIRED_KEYS)},
            PRE_STEP_PATH_KEY: PRE_STEP_PATH_KEY in settings, **arrival_limits(task),
            "base_command": "exactly_zero_except_navigate_target_from_the_goal_latch_to_the_one_shot_brake",
            "pelvis_hold": "drift_check_suspended_from_the_goal_latch_and_reanchored_at_the_navigate_target_exit",
            "goal_definition": GOAL_DEFINITION, "window_frame": WINDOW_FRAME,
            "commanded_travel_definition": TRAVEL_DEFINITION}
    if "source_hold" in settings:
        contract["source_hold"] = settings["source_hold"]
    return contract


def _xy_yaw(pose, name):
    matrix = np.asarray(pose, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or math.hypot(matrix[0, 0], matrix[1, 0]) < 1e-8:
        raise ValueError(f"single_step needs a finite measured {name} pose with a defined heading")
    return float(matrix[0, 3]), float(matrix[1, 3]), math.atan2(matrix[1, 0], matrix[0, 0])


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def planar_command(value):
    """The applied [vx, vy, yaw_rate] base command reported with an observation (strict finite numbers)."""
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, float, np.integer, np.floating))
                   or not math.isfinite(item) for item in value)):
        raise ValueError("single_step needs commanded_base_velocity: the three base commands applied in the "
                         "previous control step")
    return [float(item) for item in value]


class SingleStep:
    """State of one opt-in single step: pre-step posture, goal latch, one-shot brake, settled window.

    The machine feeds it one observation per control step; the collection admission replays the same
    rows from the raw trace and must reproduce every published evidence record exactly.
    """

    def __init__(self, settings, *, arrival_linear_speed_m_s=.03, arrival_yaw_speed_rad_s=.12):
        self.settings = deepcopy(dict(settings))
        self.offset = np.asarray(self.settings["offset_pelvis_heading_m"], dtype=float)
        self.offset_length = float(np.linalg.norm(self.offset))
        self.limits = {"arrival_linear_speed_m_s": float(arrival_linear_speed_m_s),
                       "arrival_yaw_speed_rad_s": float(arrival_yaw_speed_rad_s)}
        self.phase = PHASES[0]
        self._latch = self._brake = self._current = self._exit = self._verdict = None
        self._travel, self._last_time, self._settle_since = 0., None, None

    @property
    def pre_step_path(self):
        return self.settings.get(PRE_STEP_PATH_KEY)

    @property
    def latched(self):
        return self._latch is not None

    @property
    def braking(self):
        return self._brake is not None

    @property
    def verdict(self):
        return self._verdict

    @property
    def goal_pose_w(self):
        return None if self._latch is None else list(self._latch["goal_pose_w"])

    def enter(self, time_s, torso, pelvis, base_velocity=None):
        """navigate_target entry: the pre-step posture first, else the goal latch from this observation."""
        if self.phase != "inactive":
            raise RuntimeError("single_step: navigate_target was entered twice")
        self.phase = "pre_step_posture"
        if self.pre_step_path is None:
            self.latch(time_s, torso, pelvis, base_velocity)

    def latch(self, time_s, torso, pelvis, base_velocity=None):
        """Latch the measured torso and pelvis poses and the goal of the step."""
        if self.phase != "pre_step_posture":
            raise RuntimeError("single_step: the goal is latched once, after the pre-step posture")
        torso_x, torso_y, torso_yaw = _xy_yaw(torso, "torso")
        _, _, pelvis_yaw = _xy_yaw(pelvis, "pelvis")
        c, s = math.cos(pelvis_yaw), math.sin(pelvis_yaw)
        forward, left = self.offset
        goal = [torso_x + c * forward - s * left, torso_y + s * forward + c * left, float(np.asarray(torso)[2, 3]),
                math.cos(torso_yaw / 2.), 0., 0., math.sin(torso_yaw / 2.)]
        self._latch = {"time_s": float(time_s), "torso_pose_w": matrix_to_pose(torso),
                       "pelvis_pose_w": matrix_to_pose(pelvis), "torso_yaw_rad": torso_yaw,
                       "pelvis_yaw_rad": pelvis_yaw, "goal_pose_w": goal}
        self._last_time = float(time_s)
        self.phase = "stepping"
        self._current = self._measure(time_s, torso, pelvis, base_velocity)
        self._current["settle_continuous_dwell_s"] = 0.

    def _measure(self, time_s, torso, pelvis, base_velocity):
        latch = self._latch
        torso_x, torso_y, _ = _xy_yaw(torso, "torso")
        pelvis_x, pelvis_y, pelvis_yaw = _xy_yaw(pelvis, "pelvis")
        origin = np.asarray(latch["pelvis_pose_w"][:2], dtype=float)
        heading = latch["pelvis_yaw_rad"]
        dx, dy = pelvis_x - float(origin[0]), pelvis_y - float(origin[1])
        forward = math.cos(heading) * dx + math.sin(heading) * dy
        left = -math.sin(heading) * dx + math.cos(heading) * dy
        yaw_change = _wrap(pelvis_yaw - heading)
        goal = latch["goal_pose_w"]
        (forward_low, forward_high), (left_low, left_high) = self.settings["window_m"]
        quiet = False
        if base_velocity is not None:
            velocity = np.asarray(base_velocity, dtype=float)
            quiet = bool(velocity.shape == (6,) and np.isfinite(velocity).all()
                         and float(np.linalg.norm(velocity[:3])) <= self.limits["arrival_linear_speed_m_s"]
                         and abs(float(velocity[5])) <= self.limits["arrival_yaw_speed_rad_s"])
        return {"measurement_time_s": float(time_s),
                "torso_goal_distance_m": math.hypot(torso_x - goal[0], torso_y - goal[1]),
                "pelvis_displacement_heading_m": [forward, left], "pelvis_yaw_change_rad": yaw_change,
                "within_window": bool(forward_low <= forward <= forward_high and left_low <= left <= left_high),
                "within_yaw_limit": bool(abs(yaw_change) <= self.settings["max_yaw_change_rad"]),
                "base_quiet": quiet}

    def observe(self, time_s, torso, pelvis, base_velocity, commanded_base_velocity):
        """One observation judged in navigate_target after the latch; returns the verdict.

        ``commanded_base_velocity`` is the base command applied in the previous control step, i.e.
        the one that produced this observation. The verdict is None until the torso has been slow for
        settle_dwell_s after the brake, then ``accepted`` or ``outside_window``.
        """
        if self._latch is None or self.phase not in ("stepping", "settling"):
            raise RuntimeError("single_step: observations are judged only between the goal latch and the exit")
        command = planar_command(commanded_base_velocity)
        dt = float(time_s) - self._last_time
        if not dt > 0.:
            raise ValueError("single_step observations must advance in time")
        self._travel += math.hypot(command[0], command[1]) * dt
        self._last_time = float(time_s)
        current = self._measure(time_s, torso, pelvis, base_velocity)
        radius = self.settings["brake_radius_m"]
        if self._brake is None and (current["torso_goal_distance_m"] <= radius or self._travel >= self.offset_length):
            self._brake = {"time_s": float(time_s),
                           "trigger": BRAKE_TRIGGERS[0] if current["torso_goal_distance_m"] <= radius else BRAKE_TRIGGERS[1],
                           "torso_goal_distance_m": current["torso_goal_distance_m"],
                           "commanded_travel_m": self._travel}
            self.phase = "settling"
        dwell = 0.
        if self._brake is not None:
            if current["base_quiet"]:
                if self._settle_since is None:
                    self._settle_since = float(time_s)
                dwell = float(time_s) - self._settle_since
            else:
                self._settle_since = None
            settled = current["base_quiet"] and dwell >= self.settings["settle_dwell_s"] - 1e-10
            self._verdict = (None if not settled else
                             ACCEPTED if current["within_window"] and current["within_yaw_limit"] else REJECTED)
        current["settle_continuous_dwell_s"] = dwell
        self._current = current
        return self._verdict

    def complete(self, time_s, pelvis):
        """navigate_target exit: freeze the accepted settled stance of this (the last judged) observation.

        ``pelvis`` is that observation's measured pelvis, where the stationary pelvis hold is re-anchored.
        """
        if (self.phase != "settling" or self._verdict != ACCEPTED or self._current is None
                or self._current["measurement_time_s"] != float(time_s)):
            raise RuntimeError("single_step: navigate_target can end only on an accepted settled window")
        _xy_yaw(pelvis, "pelvis")
        self._exit = {"time_s": float(time_s), "pelvis_pose_w": matrix_to_pose(pelvis),
                      "pelvis_displacement_heading_m": list(self._current["pelvis_displacement_heading_m"]),
                      "pelvis_yaw_change_rad": self._current["pelvis_yaw_change_rad"]}
        self.phase = "complete"

    def evidence(self):
        """The published step record (command.single_step and stationary_metrics.single_step)."""
        settings = {key: deepcopy(self.settings[key]) for key in sorted(REQUIRED_KEYS)}
        settings.update({PRE_STEP_PATH_KEY: self.pre_step_path is not None, **self.limits})
        return {"schema_version": SINGLE_STEP_SCHEMA, "stage_id": SINGLE_STEP_STAGE, "phase": self.phase,
                "settings": settings, "latch": deepcopy(self._latch), "commanded_travel_m": self._travel,
                "brake": deepcopy(self._brake), "current": deepcopy(self._current), "verdict": self._verdict,
                "exit": deepcopy(self._exit), "goal_definition": GOAL_DEFINITION, "window_frame": WINDOW_FRAME,
                "commanded_travel_definition": TRAVEL_DEFINITION}
