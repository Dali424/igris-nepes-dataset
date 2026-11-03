"""Opt-in blended pass-through of intermediate lift/transfer object waypoints (runtime_v26, group B).

Without ``tasks.<task>.transfer_reference_continuity`` (transfer) and the mapping form of
``tasks.<task>.lift_reference_continuity`` (lift) nothing in this module runs: the runtime builds one
CrsPoseController per object waypoint, that controller stops at its waypoint, and the task machine keeps the
strict arrival and ``reach_dwell_s``.

With a key the stage gets ONE continuous online reference through its eligible waypoints:

* Eligibility is static and decided once at stage entry by the task machine (``plan_pass_through``) on the
  nominal TCP polyline (stage-entry TCP, then every waypoint goal through the stage control grasp). A
  waypoint is eligible when it is not one of the last ``final_waypoints``, both adjacent straight segments
  move the TCP at least 1 um, the TCP translation turns by at most ``max_corner_deg`` (<= 60 deg) there,
  both segments use the online profile with identical slip-pause and object-feedback settings, and the
  worst-case angular acceleration of the overlap stays within the larger of the two angular limits.
* The machine counts an eligible waypoint as reached when the measured object is within the unchanged object
  thresholds of it, the TCP tracking goal (the measured grasp carried to the waypoint) is within
  ``pass_position_m``/``pass_rotation_rad`` and the physical grasp is valid: no dwell. Every other waypoint
  keeps the strict arrival and dwell.
* Only after that measured arrival does the runtime offer the next waypoint to the SAME controller
  (``BlendedCrsPoseController.offer_next_segment``): the reference is never ahead of the measured progress by
  more than the segment it is finishing. The next segment's online trapezoid starts from rest at the nominal
  corner while the current one brakes (superposition of the two scalar profiles). It starts at the exact
  instant, inside the control step, at which the current segment's normalized brake speed has fallen to the
  start threshold ``v_s`` (``_start_speed_threshold``: alignment with the next segment's acceleration time,
  both segments' linear/angular speed limits, and the two corridors). Hence

    - the current segment brakes and the next one accelerates during the whole overlap; the scalar speed sums
      are linear in time, so the TCP linear and angular speeds stay within the limits of BOTH segments;
    - the translational acceleration is |a_n L_n u_n - a_k L_k u_k| <= the larger of the two limits for a
      corner <= 60 deg; the angular acceleration |alpha_n e_n - alpha_k e_k| + omega_k omega_n sin(phi_r) / 4 is
      checked statically against the larger of the two angular limits;
    - the reference stays within r0 sin(phi) of the TCP polyline (r0 = remaining length of the current segment
      at the start, <= ``corridor_position_m``) and its rotation within r0_rot (<= ``corridor_rotation_rad``)
      of the next segment's geodesic, for any later progress of either profile.

  When the current segment reaches its waypoint the next one becomes the active segment ("promotion"): its own
  motion limits and tracking guards apply from then on (the stricter guards of the two apply during the
  overlap). The object-feedback filters, the pause handling and the quaternion sign references continue.
* Every existing guard keeps its effect. A tracking-lead/cross-track pause, a slip/pinch pause or an external
  hold stops both profiles at once (a protective stop, as today), and the junction it interrupts is no longer
  blended: a queued next segment, or one frozen inside the overlap, waits at rest until the current segment
  has reached its waypoint (a stop at the corner, without today's dwell and restart). Watchdogs, transport
  constraints and grasp retention are unchanged task/runtime checks. A junction the machine judged strict, or
  one the controller cannot certify for its actual geometry (e.g. a segment restarted from a measured TCP),
  keeps today's behaviour: the strict arrival and dwell, and a new controller from the measured TCP.

This module only shapes references and arrival tolerances. It never writes state, never changes the capture or
retention transforms and is not a collision or balance certificate.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import inspect
import math

import numpy as np

from .crs_manipulation import CrsPoseController, _positive, _rotation_vector_matrix
from .snapshot import matrix_to_pose, pose_to_matrix
from .task_control import rotation_error

MODE = "blended_pass_through"
SCHEMA_VERSION = "igris_c_reference_continuity_v1"
KEYS = ("mode", "max_corner_deg", "corridor_position_m", "corridor_rotation_rad", "pass_position_m",
        "pass_rotation_rad", "final_waypoints")
# Optional (runtime_v26 review fix B1B2): inclusive [first, last] waypoint ranges outside which every waypoint stays
# strict whatever its corner, e.g. the clearance-critical JIG entry and slot approach of the transfer, or the lift
# waypoints still inside the cart slot exit.
OPTIONAL_KEYS = ("pass_ranges",)
PASS_RANGES_MAXIMUM_COUNT = 16
# Each value must be finite and in (0, bound].
BOUNDS = {"max_corner_deg": 60., "corridor_position_m": .003, "corridor_rotation_rad": .02,
          "pass_position_m": .006, "pass_rotation_rad": .06}
STAGE_KEYS = {"lift": "lift_reference_continuity", "transfer": "transfer_reference_continuity"}
# Straight TCP segments shorter than this have no defined translation corner: such a junction stays strict.
MIN_SEGMENT_TRANSLATION_M = 1e-6
# The machine may confirm several short waypoints before the reference has finished one; bound the chain.
MAX_QUEUED_SEGMENTS = 16
# Settings a blended chain may change between consecutive segments: each segment keeps its own motion limits,
# and the tracking guards switch at the promotion (the stricter of two overlapping segments applies meanwhile).
MOTION_KEYS = ("linear_speed", "linear_acceleration", "angular_speed", "angular_acceleration")
GUARD_KEYS = ("position_lead", "rotation_lead", "cross_track_position", "cross_track_rotation")
# Settings that must not change along a blended chain (one slip guard, one object-feedback filter).
FIXED_KEYS = ("reference_profile", "pause_slip_position", "pause_slip_rotation", "object_feedback")
_DEFAULTS = {name: parameter.default for name, parameter in
             inspect.signature(CrsPoseController.__init__).parameters.items()
             if name in (*MOTION_KEYS, *GUARD_KEYS, *FIXED_KEYS)}
# Numerical slack of the controller's re-check of the machine's static plan (roundoff only, not a tolerance).
_RECHECK_ANGLE_RAD = 1e-9
_RECHECK_RELATIVE = 1e-9
_START_SLACK_S = 1e-9


def validate_reference_continuity(settings, name, waypoint_count):
    """Validated copy of one blended pass-through declaration; every key is required and bounded."""
    if not isinstance(settings, Mapping) or set(settings) - set(OPTIONAL_KEYS) != set(KEYS):
        present = set(settings) if isinstance(settings, Mapping) else set()
        detail = []
        if present - set(KEYS) - set(OPTIONAL_KEYS):
            detail.append("unknown " + ", ".join(sorted(map(str, present - set(KEYS) - set(OPTIONAL_KEYS)))))
        if set(KEYS) - present:
            detail.append("missing " + ", ".join(sorted(set(KEYS) - present)))
        raise ValueError(f"{name} requires exactly " + ", ".join(KEYS) + " (optional: pass_ranges)"
                         + (" (" + "; ".join(detail) + ")" if detail else ""))
    if settings["mode"] != MODE:
        raise ValueError(f"{name}.mode must be {MODE}")
    result = {"mode": MODE}
    for key, high in BOUNDS.items():
        value = settings[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or not 0. < value <= high):
            raise ValueError(f"{name}.{key} must be a finite number in (0, {high:g}]")
        result[key] = float(value)
    final = settings["final_waypoints"]
    count = int(waypoint_count)
    if count < 2:
        raise ValueError(f"{name} needs an object path with at least two waypoints")
    if type(final) is not int or not 1 <= final <= count - 1:
        raise ValueError(f"{name}.final_waypoints must be an integer in [1, {count - 1}] "
                         "(at least one intermediate waypoint, the last one stays strict)")
    result["final_waypoints"] = final
    validated = {key: result[key] for key in KEYS}
    if "pass_ranges" in settings:
        validated["pass_ranges"] = _validate_pass_ranges(settings["pass_ranges"], name, count - final - 1)
    return validated


def _validate_pass_ranges(value, name, last_index):
    """1..16 inclusive [first, last] integer ranges inside [0, last_index], strictly increasing, never overlapping."""
    label = f"{name}.pass_ranges"
    if (isinstance(value, (str, bytes, Mapping)) or not isinstance(value, (list, tuple))
            or not 1 <= len(value) <= PASS_RANGES_MAXIMUM_COUNT):
        raise ValueError(f"{label} must be a list of 1..{PASS_RANGES_MAXIMUM_COUNT} [first, last] waypoint ranges")
    ranges = []
    for index, pair in enumerate(value):
        if (isinstance(pair, (str, bytes, Mapping)) or not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(type(item) is not int for item in pair)):
            raise ValueError(f"{label}[{index}] must be an integer [first, last] pair")
        first, last = pair
        if not 0 <= first <= last <= last_index:
            raise ValueError(f"{label}[{index}] must satisfy 0 <= first <= last <= {last_index} (the last "
                             "final_waypoints stay strict)")
        if ranges and first <= ranges[-1][1]:
            raise ValueError(f"{label} ranges must increase without overlapping")
        ranges.append([first, last])
    return ranges


def validate_task_reference_continuity(task_id, task, *, upright_pickup):
    """Validate the declared blended stages of one task in place (absent keys: nothing happens).

    ``lift_reference_continuity`` keeps its runtime_v25 string form ``collinear_online`` (validated by the
    caller); only its mapping form is the blended pass-through.
    """
    for stage, key in STAGE_KEYS.items():
        if key not in task or (stage == "lift" and not isinstance(task[key], Mapping)):
            continue
        label = f"{task_id}.{key}"
        if not upright_pickup:
            raise ValueError(f"{label} requires the independent upright pickup/load task1 (crs_load_jig)")
        paths = task.get("object_waypoint_offsets_w")
        offsets = paths.get(stage) if isinstance(paths, Mapping) else None
        if not isinstance(offsets, list) or len(offsets) < 2:
            raise ValueError(f"{label} requires an explicit {stage} object path with at least two waypoints")
        control = task.get("object_pose_control")
        if not isinstance(control, Mapping) or control.get("reference_profile") != "online":
            raise ValueError(f"{label} requires object_pose_control.reference_profile online")
        overrides = task.get("object_pose_control_overrides", {})
        stage_overrides = overrides.get(stage, []) if isinstance(overrides, Mapping) else []
        for override in stage_overrides if isinstance(stage_overrides, list) else []:
            if isinstance(override, Mapping) and override.get("reference_profile", "online") != "online":
                raise ValueError(f"{label} requires online {stage} object_pose_control_overrides")
        if stage == "transfer" and "transfer_transit_arrival" in task:
            raise ValueError(f"{label} cannot be combined with transfer_transit_arrival")
        postures = task.get("object_waypoint_postures", {})
        if isinstance(postures, Mapping) and stage in postures:
            raise ValueError(f"{label} cannot be combined with object_waypoint_postures.{stage}")
        guidance = task.get("arm_posture_guidance", {})
        guided = guidance.get("stages", {}) if isinstance(guidance, Mapping) else {}
        if isinstance(guided, Mapping) and stage in guided:
            raise ValueError(f"{label} cannot be combined with arm_posture_guidance.stages.{stage}")
        task[key] = validate_reference_continuity(task[key], label, len(offsets))


def continuity_settings(task):
    """{stage: validated settings} of the declared blended stages, or None when none is declared."""
    result = {stage: task[key] for stage, key in STAGE_KEYS.items() if isinstance(task.get(key), Mapping)}
    return result or None


def continuity_contract(config, task):
    """The run.json controller_contract.motion_continuity block, or None (absent keys: no block)."""
    contract = {key: deepcopy(dict(task[key])) for key in STAGE_KEYS.values() if isinstance(task.get(key), Mapping)}
    dwell = config.get("thresholds", {}).get("object_waypoint_intermediate_dwell_s")
    if dwell is not None:
        contract["object_waypoint_intermediate_dwell_s"] = float(dwell)
    if not contract:
        return None
    return {"schema_version": SCHEMA_VERSION, **contract,
            "pass_through_arrival": ("eligible intermediate lift/transfer waypoints: measured object within the "
                                     "object thresholds, tracking goal within the pass tolerances, grasp valid, "
                                     "no dwell; the reference blends through them without stopping"),
            "strict_waypoints": "final_waypoints, sharp corners and incompatible segments keep strict arrival"
                                + (", and so do waypoints outside the declared pass_ranges"
                                   if any("pass_ranges" in task.get(key, {}) for key in STAGE_KEYS.values()
                                          if isinstance(task.get(key), Mapping)) else "")}


def effective_settings(settings):
    """CrsPoseController settings with its defaults filled in (for compatibility comparisons)."""
    return {**_DEFAULTS, **dict(settings)}


def online_step(progress, velocity, path_speed, path_acceleration, dt):
    """CrsPoseController._online_candidate as a pure function of an explicit segment state."""
    remaining = max(0., 1. - progress)
    if remaining <= 1e-14 or path_speed == 0.:
        return 1., 0.
    acceleration = path_acceleration
    stopping_distance = velocity * velocity / (2. * acceleration)
    if stopping_distance > remaining + 1e-10:
        raise RuntimeError("Online reference has insufficient stopping distance")
    peak = min(path_speed, math.sqrt(acceleration * remaining + .5 * velocity * velocity))
    peak = max(velocity, peak)
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
    return min(1., progress + distance), velocity


def _path_limits(length, angle, settings):
    """Normalized path speed and acceleration of one straight segment, exactly as CrsPoseController sets them."""
    limits = [(settings["linear_speed"] / length, settings["linear_acceleration"] / length)] if length > 1e-12 else []
    if angle > 1e-12:
        limits.append((settings["angular_speed"] / angle, settings["angular_acceleration"] / angle))
    return min((v for v, _ in limits), default=0.), min((a for _, a in limits), default=0.)


def _segment_motion(start, end, settings):
    translation = end[:3, 3] - start[:3, 3]
    rotation_vector = rotation_error(end[:3, :3], start[:3, :3])
    length, angle = float(np.linalg.norm(translation)), float(np.linalg.norm(rotation_vector))
    speed, acceleration = _path_limits(length, angle, settings)
    # The largest normalized speed this segment's trapezoid can reach (from rest over the whole segment).
    peak = min(speed, math.sqrt(acceleration)) if speed > 0. and acceleration > 0. else 0.
    return {"translation": translation, "rotation_vector": rotation_vector, "length": length, "angle": angle,
            "path_speed": speed, "path_acceleration": acceleration, "peak": peak}


def _angle(first, second):
    first_norm, second_norm = float(np.linalg.norm(first)), float(np.linalg.norm(second))
    if first_norm <= 0. or second_norm <= 0.:
        return 0.
    return math.acos(float(np.clip(np.dot(first, second) / (first_norm * second_norm), -1., 1.)))


def junction_geometry(start, corner, end, settings_k, settings_n, continuity, *, recheck=False):
    """Static pass-through decision for the junction at ``corner`` (both segments straight in TCP space).

    ``recheck`` adds roundoff slack for the controller's re-check of a junction the machine already accepted.
    Returns {"eligible", "reason", evidence...}; nothing here depends on measured motion.
    """
    start, corner, end = (np.asarray(pose, float) for pose in (start, corner, end))
    first, second = effective_settings(settings_k), effective_settings(settings_n)
    reason = None
    for key in FIXED_KEYS:
        if first.get(key) != second.get(key):
            reason = reason or "settings_changed:" + key
    if first["reference_profile"] != "online" or second["reference_profile"] != "online":
        reason = reason or "reference_profile_not_online"
    k, n = _segment_motion(start, corner, first), _segment_motion(corner, end, second)
    corner_angle = _angle(k["translation"], n["translation"])
    rotation_angle = (_angle(k["rotation_vector"], n["rotation_vector"])
                      if k["angle"] > 1e-12 and n["angle"] > 1e-12 else 0.)
    # Worst-case overlap accelerations: the current profile brakes (-a_k) while the next accelerates (+a_n).
    beta_k, beta_n = k["path_acceleration"] * k["length"], n["path_acceleration"] * n["length"]
    alpha_k, alpha_n = k["path_acceleration"] * k["angle"], n["path_acceleration"] * n["angle"]
    omega_k, omega_n = k["peak"] * k["angle"], n["peak"] * n["angle"]
    translation_bound = math.sqrt(max(0., beta_k ** 2 + beta_n ** 2 - 2. * beta_k * beta_n * math.cos(corner_angle)))
    rotation_bound = (math.sqrt(max(0., alpha_k ** 2 + alpha_n ** 2 - 2. * alpha_k * alpha_n * math.cos(rotation_angle)))
                      + .25 * omega_k * omega_n * math.sin(rotation_angle))
    linear_limit = max(first["linear_acceleration"], second["linear_acceleration"])
    angular_limit = max(first["angular_acceleration"], second["angular_acceleration"])
    slack_angle = _RECHECK_ANGLE_RAD if recheck else 0.
    slack = 1. + (_RECHECK_RELATIVE if recheck else 0.)
    if min(k["length"], n["length"]) < MIN_SEGMENT_TRANSLATION_M:
        reason = reason or "degenerate_translation"
    if corner_angle > math.radians(continuity["max_corner_deg"]) + slack_angle:
        reason = reason or "corner_angle"
    if translation_bound > linear_limit * slack:
        reason = reason or "translation_acceleration"
    if rotation_bound > angular_limit * slack:
        reason = reason or "rotation_acceleration"
    return {"eligible": reason is None, "reason": reason,
            "corner_deg": math.degrees(corner_angle), "rotation_corner_deg": math.degrees(rotation_angle),
            "segment_lengths_m": [k["length"], n["length"]], "segment_rotations_rad": [k["angle"], n["angle"]],
            "overlap_linear_acceleration_bound_m_s2": translation_bound, "linear_acceleration_limit_m_s2": linear_limit,
            "overlap_angular_acceleration_bound_rad_s2": rotation_bound,
            "angular_acceleration_limit_rad_s2": angular_limit,
            # The start rule keeps the overlap within BOTH segments' speed limits.
            "overlap_linear_speed_limit_m_s": min(first["linear_speed"], second["linear_speed"]),
            "overlap_angular_speed_limit_rad_s": min(first["angular_speed"], second["angular_speed"])}


def plan_pass_through(continuity, entry_tcp, tcp_goals, settings):
    """Per-waypoint static decision for one stage: [{"waypoint_index", "eligible", "reason", evidence...}].

    ``entry_tcp`` is the measured stage-entry TCP (the start of the first segment), ``tcp_goals`` the waypoint
    goals through the stage control grasp and ``settings`` each waypoint segment's CrsPoseController settings.
    """
    goals = [np.asarray(goal, float) for goal in tcp_goals]
    if len(settings) != len(goals):
        raise ValueError("pass-through planning needs one settings mapping per waypoint")
    plan = []
    ranges = continuity.get("pass_ranges")
    for index in range(len(goals)):
        if len(goals) - 1 - index < continuity["final_waypoints"]:
            plan.append({"waypoint_index": index, "eligible": False, "reason": "final_waypoint"})
            continue
        if ranges is not None and not any(first <= index <= last for first, last in ranges):
            plan.append({"waypoint_index": index, "eligible": False, "reason": "outside_pass_ranges"})
            continue
        start = np.asarray(entry_tcp, float) if index == 0 else goals[index - 1]
        junction = junction_geometry(start, goals[index], goals[index + 1], settings[index], settings[index + 1],
                                     continuity)
        plan.append({"waypoint_index": index, **junction})
    return plan


def plan_evidence(plan):
    """Compact plan for every command row: eligible waypoints and the reason of each strict one."""
    return {"eligible_waypoints": [entry["waypoint_index"] for entry in plan if entry["eligible"]],
            "strict_reasons": {str(entry["waypoint_index"]): entry["reason"] for entry in plan if not entry["eligible"]}}


class _Segment:
    """One queued or overlapping straight segment of a blended chain (start = the previous waypoint goal)."""

    def __init__(self, index, start_tcp, goal_object, reference_object_to_tcp, settings):
        self.index = int(index)
        self.start = np.array(start_tcp, dtype=float, copy=True)
        self.goal_object = np.array(goal_object, dtype=float, copy=True)
        self.goal_tcp = self.goal_object @ reference_object_to_tcp
        self.settings = settings
        self.translation = self.goal_tcp[:3, 3] - self.start[:3, 3]
        self.rotation_vector = rotation_error(self.goal_tcp[:3, :3], self.start[:3, :3])
        self.length = float(np.linalg.norm(self.translation))
        self.angle = float(np.linalg.norm(self.rotation_vector))
        self.path_speed, self.path_acceleration = _path_limits(self.length, self.angle, settings)
        self.fraction = 0.
        self.velocity = 0.
        self.state = "queued"  # then "blending" (moving in the overlap) or "frozen" (stopped in the overlap)
        self.stop_at_corner = False  # a protective pause flagged its junction: no blend, wait for the waypoint
        self.blend = None
        self.junction = None


class BlendedCrsPoseController(CrsPoseController):
    """CrsPoseController that can continue through eligible waypoints without stopping.

    Without an accepted ``offer_next_segment`` every update is exactly the base controller's (plus the
    ``reference_continuity`` evidence). See the module docstring for the blend, its bounds and the guards.
    """

    def __init__(self, object_to_tcp, tcp_pose_w, goal_object_pose_w, *, continuity, waypoint_index, **settings):
        super().__init__(object_to_tcp, tcp_pose_w, goal_object_pose_w, **settings)
        if self.reference_profile != "online":
            raise ValueError("blended pass-through requires the online reference profile")
        if type(waypoint_index) is not int or waypoint_index < 0:
            raise ValueError("blended pass-through needs a nonnegative integer waypoint index")
        self.continuity = dict(continuity)
        self._segment_index = waypoint_index
        self._settings = effective_settings(settings)
        self._next = None
        self._queue = []
        self._step_events = []
        self._last_blend = None

    @property
    def segment_index(self):
        """Waypoint index of the active segment (the waypoint the reference is currently finishing)."""
        return self._segment_index

    def offer_next_segment(self, goal_object_pose_w, settings, *, waypoint_index):
        """Append the segment toward the next waypoint to the chain; validate-then-mutate.

        Called only after the machine measured the arrival at the chain's last waypoint. Returns explicit
        acceptance evidence; a rejection changes no state (the caller then restarts as today).
        """
        def reject(reason, **evidence):
            return {"accepted": False, "reason": reason, **evidence}

        try:
            goal_object = pose_to_matrix(goal_object_pose_w)
        except (TypeError, ValueError, OverflowError):
            return reject("invalid_goal_pose")
        if not isinstance(settings, Mapping):
            return reject("invalid_settings")
        end = self._queue[-1] if self._queue else self._next
        end_index = self._segment_index if end is None else end.index
        if type(waypoint_index) is not int or waypoint_index != end_index + 1:
            return reject("nonadjacent_waypoint", chain_end_waypoint_index=end_index)
        if len(self._queue) >= MAX_QUEUED_SEGMENTS:
            return reject("queue_full")
        effective = effective_settings(settings)
        try:
            for key in (*MOTION_KEYS, *GUARD_KEYS):
                _positive(effective[key], key)
        except ValueError as error:
            return reject("invalid_settings", detail=str(error))
        if end is None:
            start, corner, previous = self._segment_start, self._goal_tcp, self._settings
        else:
            start, corner, previous = end.start, end.goal_tcp, end.settings
        segment = _Segment(waypoint_index, corner, goal_object, self._reference_object_to_tcp, effective)
        junction = junction_geometry(start, corner, segment.goal_tcp, previous, effective, self.continuity,
                                     recheck=True)
        if not junction["eligible"]:
            return reject(junction["reason"], junction=junction)
        segment.junction = junction
        self._queue.append(segment)
        return {"accepted": True, "reason": None, "junction": junction,
                "queued_waypoint_indices": [item.index for item in self._queue]}

    # --- the blended advance ---------------------------------------------------------------------------------

    def _start_speed_threshold(self, candidate):
        """Normalized brake speed of the active segment at or below which ``candidate`` may start.

        With the active segment braking (normalized speed v, path acceleration a_k, remaining v^2 / (2 a_k)) and
        the candidate accelerating from rest (a_n) for the overlap time T = v / a_k:
          * v <= a_k T_n: the candidate is still accelerating when the active segment stops (alignment);
          * v <= V_n / L_k and v <= W_n / theta_k: the overlap starts within the candidate's speed limits;
          * v <= V_k a_k / (a_n L_n) and v <= W_k a_k / (a_n theta_n): it ends within the active segment's;
          * v^2 <= 2 a_k c_p / (L_k sin phi) and v^2 <= 2 a_k c_r / theta_k: the corridors.
        The scalar speed sums are linear in time over the overlap, so the TCP linear and angular speeds stay
        within the limits of BOTH segments throughout it.
        """
        acceleration, other = self._path_acceleration, candidate.path_acceleration
        length = float(np.linalg.norm(self._translation))
        angle = float(np.linalg.norm(self._rotation_vector))
        peak = min(candidate.path_speed, math.sqrt(other))
        values = [acceleration * peak / other]
        if length > 0.:
            values.append(float(candidate.settings["linear_speed"]) / length)
        if angle > 1e-12:
            values.append(float(candidate.settings["angular_speed"]) / angle)
        if candidate.length > 0.:
            values.append(self.linear_speed * acceleration / (other * candidate.length))
        if candidate.angle > 1e-12:
            values.append(self.angular_speed * acceleration / (other * candidate.angle))
        sine = math.sin(_angle(self._translation, candidate.translation))
        if length > 0. and sine > 1e-12:
            values.append(math.sqrt(2. * acceleration * self.continuity["corridor_position_m"] / (length * sine)))
        if angle > 1e-12:
            values.append(math.sqrt(2. * acceleration * self.continuity["corridor_rotation_rad"] / angle))
        return min(values)

    def _start_offset(self, candidate, dt):
        """(offset within this step at which ``candidate`` starts, or None; the start speed threshold).

        A candidate whose junction a protective pause flagged starts only once the active segment rests at its
        waypoint (a stop at the corner, today's behaviour without the dwell and the restart).
        """
        progress, velocity = self._progress, self._path_velocity
        speed, acceleration = self._path_speed, self._path_acceleration
        if progress >= 1. - 1e-12 or speed == 0.:
            return 0., None  # the active segment rests at its waypoint: start now (a stop at the corner)
        if candidate.stop_at_corner:
            return None, None
        threshold = self._start_speed_threshold(candidate)
        remaining = max(0., 1. - progress)
        peak = max(velocity, min(speed, math.sqrt(acceleration * remaining + .5 * velocity * velocity)))
        accelerate_time = max(0., (peak - velocity) / acceleration)
        accelerate_distance = max(0., (peak * peak - velocity * velocity) / (2. * acceleration))
        brake_distance = peak * peak / (2. * acceleration)
        cruise_time = max(0., remaining - accelerate_distance - brake_distance) / peak
        offset = accelerate_time + cruise_time + max(0., (peak - threshold) / acceleration)
        if offset < dt - _START_SLACK_S:
            return max(0., offset), threshold
        return None, threshold

    def _compose(self, active_fraction, other, other_fraction):
        candidate = self._segment_start.copy()
        candidate[:3, 3] += active_fraction * self._translation
        rotation = _rotation_vector_matrix(active_fraction * self._rotation_vector) @ self._segment_start[:3, :3]
        if other is not None:
            candidate[:3, 3] += other_fraction * other.translation
            rotation = _rotation_vector_matrix(other_fraction * other.rotation_vector) @ rotation
        candidate[:3, :3] = rotation
        return candidate

    def _directions(self, active_fraction, active_velocity, other, other_fraction, other_velocity):
        """Composite reference velocity directions (translation, spatial rotation), with the segment fallback."""
        linear = active_velocity * self._translation
        angular = active_velocity * self._rotation_vector
        if other is not None:
            linear = linear + other_velocity * other.translation
            angular = (other_velocity * other.rotation_vector
                       + _rotation_vector_matrix(other_fraction * other.rotation_vector) @ angular)
        moving_on = other is not None and active_fraction >= 1. - 1e-12
        if float(np.linalg.norm(linear)) < 1e-12:
            linear = other.translation if moving_on else self._translation
        if float(np.linalg.norm(angular)) < 1e-12:
            angular = other.rotation_vector if moving_on else self._rotation_vector
        return linear, angular

    def _guards(self, other):
        values = {key: getattr(self, key) for key in GUARD_KEYS}
        if other is not None:
            values = {key: min(value, float(other.settings[key])) for key, value in values.items()}
        return values

    def _promote(self):
        segment = self._next
        self._segment_start = segment.start.copy()
        self._segment_start_progress = 0.
        self._translation = segment.translation
        self._rotation_vector = segment.rotation_vector
        self._goal_object = segment.goal_object
        self._goal_tcp = segment.goal_tcp
        self._progress = segment.fraction
        self._path_velocity = segment.velocity
        self._path_speed, self._path_acceleration = segment.path_speed, segment.path_acceleration
        for key in (*MOTION_KEYS, *GUARD_KEYS):
            setattr(self, key, float(segment.settings[key]))
        self._settings = segment.settings
        self._duration = max(1.875 * segment.length / self.linear_speed,
                             math.sqrt((10 / math.sqrt(3)) * segment.length / self.linear_acceleration),
                             1.875 * segment.angle / self.angular_speed,
                             math.sqrt((10 / math.sqrt(3)) * segment.angle / self.angular_acceleration))
        self._elapsed = 0.
        self._segment_index = segment.index
        self._next = None
        self._step_events.append({"event": "promoted", "waypoint_index": segment.index})

    def _advance_reference(self, tcp, dt, reason=None):
        self._step_events = []
        if self._next is None and not self._queue:
            result = super()._advance_reference(tcp, dt, reason)
            result["reference_continuity"] = self._evidence()
            return result
        other, other_fraction, other_velocity, starting = self._next, 0., 0., None
        if self._next is not None:
            other_fraction = self._next.fraction
        if reason is None:
            active_fraction, active_velocity = self._online_candidate(dt)
            if self._next is not None and self._next.state == "blending":
                other_fraction, other_velocity = online_step(self._next.fraction, self._next.velocity,
                                                             self._next.path_speed, self._next.path_acceleration, dt)
            elif self._next is None:
                # A frozen next segment keeps its progress, at rest, until the active segment reaches its waypoint.
                waiting = self._queue[0]
                offset, threshold = self._start_offset(waiting, dt)
                if offset is not None:
                    starting = (waiting, offset, threshold)
                    other = waiting
                    other_fraction, other_velocity = (
                        online_step(waiting.fraction, 0., waiting.path_speed, waiting.path_acceleration, dt - offset)
                        if dt - offset > 1e-12 else (waiting.fraction, 0.))
            candidate = self._compose(active_fraction, other, other_fraction)
            linear, angular = self._directions(active_fraction, active_velocity, other, other_fraction, other_velocity)
            guards = self._guards(other)
            position_ahead, position_cross = self._directional_error(candidate[:3, 3] - tcp[:3, 3], linear)
            rotation_ahead, rotation_cross = self._directional_error(
                rotation_error(candidate[:3, :3], tcp[:3, :3]), angular)
            if position_cross > guards["cross_track_position"] or rotation_cross > guards["cross_track_rotation"]:
                reason = "cross_track_error"
            elif position_ahead > guards["position_lead"] or rotation_ahead > guards["rotation_lead"]:
                reason = "tracking_lead"
            else:
                if starting is not None:
                    self._begin(*starting, dt)
                self._reference = candidate
                self._elapsed = self._elapsed + dt
                self._progress = self._segment_start_progress + (1 - self._segment_start_progress) * active_fraction
                self._path_velocity = active_velocity
                if self._next is not None:
                    self._next.fraction, self._next.velocity = other_fraction, other_velocity
                    if self._progress >= 1. - 1e-12:
                        self._promote()
        self._paused = reason is not None
        if self._paused:
            self._path_velocity = 0.
            # Every protective pause keeps its effect: both profiles stop at once, and the junction it flagged is
            # no longer blended (the next segment waits, at rest, for the active segment to reach its waypoint).
            if self._next is not None and self._next.state == "blending":
                self._next.velocity = 0.
                self._next.state = "frozen"
                self._next.stop_at_corner = True
                self._step_events.append({"event": "next_frozen", "waypoint_index": self._next.index,
                                          "reason": reason})
            elif self._next is None and self._queue and not self._queue[0].stop_at_corner:
                self._queue[0].stop_at_corner = True
                self._step_events.append({"event": "junction_stops_at_corner", "waypoint_index": self._queue[0].index,
                                          "reason": reason})
        pose = np.asarray(matrix_to_pose(self._reference))
        if np.dot(pose[3:], self._previous_quaternion) < 0:
            pose[3:] *= -1
        self._previous_quaternion = pose[3:].copy()
        other = self._next
        linear, angular = self._directions(self._progress, self._path_velocity, other,
                                           0. if other is None else other.fraction,
                                           0. if other is None else other.velocity)
        position_ahead, position_cross = self._directional_error(self._reference[:3, 3] - tcp[:3, 3], linear)
        rotation_ahead, rotation_cross = self._directional_error(
            rotation_error(self._reference[:3, :3], tcp[:3, :3]), angular)
        complete = bool(self._progress >= 1 - 1e-12 and self._next is None and not self._queue)
        return {
            "tcp_target_pose_w": pose.tolist(),
            "object_reference_pose_w": matrix_to_pose(self._reference @ np.linalg.inv(self._reference_object_to_tcp)),
            "goal_object_pose_w": matrix_to_pose(self._goal_object),
            "progress": float(self._progress), "reference_complete": complete,
            "paused": self._paused, "reason": reason,
            "reference_profile": self.reference_profile,
            "normalized_path_velocity_per_s": self._path_velocity,
            "protective_stop_may_exceed_reference_acceleration_limit": self._paused,
            "position_lead_m": position_ahead, "position_cross_track_m": position_cross,
            "rotation_lead_rad": rotation_ahead, "rotation_cross_track_rad": rotation_cross,
            "collision_checked": False, "physical_success_verified": False,
            "reference_continuity": self._evidence(),
        }

    def _begin(self, segment, offset, threshold, dt):
        """Commit the start of ``segment`` at ``offset`` inside this step (after the guards passed)."""
        fresh = segment.state == "queued"
        if fresh:
            self._queue.pop(0)
            self._next = segment
            if threshold is None:  # the active segment already rested at its waypoint
                velocity, remaining = 0., 0.
            else:
                velocity = min(threshold, max(self._path_velocity, min(self._path_speed, math.sqrt(
                    self._path_acceleration * max(0., 1. - self._progress)
                    + .5 * self._path_velocity * self._path_velocity))))
                remaining = velocity * velocity / (2. * self._path_acceleration)
            length = float(np.linalg.norm(self._translation))
            angle = float(np.linalg.norm(self._rotation_vector))
            sine = math.sin(_angle(self._translation, segment.translation))
            segment.blend = {
                "from_waypoint_index": self._segment_index, "to_waypoint_index": segment.index,
                "start_offset_s": float(offset), "start_speed_threshold_per_s": threshold,
                "active_normalized_velocity_at_start_per_s": float(velocity),
                "active_remaining_fraction_at_start": float(remaining),
                "active_at_rest": threshold is None, "stopped_at_corner": segment.stop_at_corner,
                "deviation_bound_position_m": float(remaining * length * sine),
                "deviation_bound_rotation_rad": float(remaining * angle),
                **{key: segment.junction[key] for key in (
                    "corner_deg", "rotation_corner_deg", "overlap_linear_acceleration_bound_m_s2",
                    "overlap_angular_acceleration_bound_rad_s2", "overlap_linear_speed_limit_m_s",
                    "overlap_angular_speed_limit_rad_s")}}
            self._last_blend = segment.blend
        segment.state = "blending"
        self._step_events.append({"event": "blend_started", "waypoint_index": segment.index,
                                  "start_offset_s": float(offset)})

    def _evidence(self):
        segment = self._next
        return {
            "schema_version": SCHEMA_VERSION, "mode": MODE,
            "active_waypoint_index": self._segment_index, "active_progress": float(self._progress),
            "next_waypoint_index": None if segment is None else segment.index,
            "next_progress": None if segment is None else float(segment.fraction),
            "next_normalized_velocity_per_s": None if segment is None else float(segment.velocity),
            "next_state": None if segment is None else segment.state,
            "next_stops_at_corner": None if segment is None else segment.stop_at_corner,
            "queued_waypoint_indices": [item.index for item in self._queue],
            "blend": None if segment is None else deepcopy(segment.blend),
            "last_blend": deepcopy(self._last_blend),
            "guards": self._guards(segment),
            "events": deepcopy(self._step_events),
        }


def offer_pass_through(controller, previous, current, goal_object_pose_w, settings, machine_evidence):
    """Runtime decision at an object waypoint change: continue the blended controller or restart (today).

    ``previous``/``current`` are the segment contexts (stage, waypoint_index, arm, object_path, grasp,
    control_grasp); ``machine_evidence`` is the machine's ``reference_continuity`` command evidence.
    """
    def rejected(reason):
        return {"accepted": False, "reason": reason}

    if not isinstance(controller, BlendedCrsPoseController) or previous is None:
        return rejected("initial_segment")
    if previous["stage"] != current["stage"]:
        return rejected("stage_boundary")
    if current["waypoint_index"] != previous["waypoint_index"] + 1:
        return rejected("nonadjacent_waypoint")
    eligible = (machine_evidence or {}).get("eligible_waypoints") if isinstance(machine_evidence, Mapping) else None
    if (not isinstance(machine_evidence, Mapping) or machine_evidence.get("stage_id") != current["stage"]
            or not isinstance(eligible, list) or previous["waypoint_index"] not in eligible):
        return rejected("strict_waypoint")
    for name in ("arm", "object_path"):
        if previous[name] != current[name]:
            return rejected(name + "_changed")
    for name in ("grasp", "control_grasp"):
        if not np.allclose(previous[name], current[name], rtol=0., atol=1e-12):
            return rejected(name + "_changed")
    return controller.offer_next_segment(goal_object_pose_w, settings, waypoint_index=current["waypoint_index"])


def segment_context(stage, waypoint_index, arm, object_path, grasp, control_grasp):
    return {"stage": stage, "waypoint_index": waypoint_index, "arm": arm, "object_path": object_path,
            "grasp": deepcopy(list(grasp)), "control_grasp": deepcopy(list(control_grasp))}
