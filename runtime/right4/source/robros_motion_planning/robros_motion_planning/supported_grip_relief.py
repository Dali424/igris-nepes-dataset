"""Bounded jaw-command relief on measured JIG support; no physics or success edits.

runtime_v26 opt-in keys (each absent: reports, contract and commands are those of runtime_v25):

* ``jump_start_rad`` (0.029..0.04 rad): on each activation the command slews at the gripper's own speed limit from
  the previous applied target to ``jump_start_rad`` beyond the measured jaw toward closed (the pinch saturation edge
  is command - measured = 0.029 rad in run_116), never past the relief goal; the ``max_target_speed_rad_s`` ramp
  continues from there. The loss-of-support re-close mirrors it (runtime_v26 review fix): once the relief has been
  active in the place, a deactivated relief (outside the dropout grace) re-closes a target that lies beyond that
  edge at the same gripper speed back to the edge, then at the ramp rate to the closed command, so a support flicker
  cannot leave the jaw relaxed while the ring is unsupported. ``relief_profile: "fast"`` raises the ramp-rate bound
  from 0.03 to 0.06 rad/s.
* ``activation_minimum_load_fraction`` (0.5..placement_support.minimum_load_fraction): the relief's support guard
  uses this fraction of the measured object weight instead of the placement gate's fraction; the place opening dwell
  and every success gate keep placement_support.minimum_load_fraction.
* ``placement_support.filter_s`` (constructor ``placement_support``): the relief's support guard uses the place
  stage's causal filtered JIG normal force (placement_support_filter.py) instead of the raw sample: the smaller of
  the window mean and the window's sustained force (raw samples at or above it cover 30 % of the window).
* ``place_pinch_reduction`` (constructor ``place_pinch_reduction``, ``slot_pose``): before the first activation in a
  place stage, once the slot geometry and capture guards hold, the measured ring is within ``start_height_m`` (at
  most 4 mm) above the seat and the JIG has measured a contact force (``PINCH_REDUCTION_CONTACT_FORCE_N``, latched
  from the first such sample of the place), the jaw target moves (at the relief rate) to ``delta_rad`` (at least
  0.02 rad: about 30 N of pinch left on run_116's calibration) beyond the measured jaw toward closed: a reduced but
  still closing pinch until the relief takes over. A ring that hangs above the seat without JIG contact keeps the
  full pinch (runtime_v26 review fix).
"""
from collections.abc import Mapping
import math

import numpy as np

from .motion_watchdog import pose_goal_errors
from .object_symmetry import pivot_radius, symmetric_capture_errors, unit_axis
from .snapshot import matrix_to_pose, pose_to_matrix
from .task_control import gripper_closed_position, gripper_open_position

SCHEMA = "igris_c_supported_grip_relief_v1"
KEYS = {"preload_rad", "max_target_speed_rad_s", "activation_dwell_s"}
OPTIONAL = {"support_dropout_grace_s", "jump_start_rad", "relief_profile", "activation_minimum_load_fraction"}
V26_OPTIONAL = {"jump_start_rad", "relief_profile", "activation_minimum_load_fraction"}
SUPPORT_GUARDS = {"jig_support_confirmed", "support_load"}
# runtime_v26 C1: relief_profile "fast" raises the ramp-rate bound; absent, the bound stays 0.03 rad/s.
FAST_PROFILE = "fast"
DEFAULT_MAX_TARGET_SPEED_RAD_S = .03
FAST_MAX_TARGET_SPEED_RAD_S = .06
# runtime_v26 review fix: never below run_116's measured pinch saturation edge (command - measured = 0.029 rad).
JUMP_START_BOUNDS_RAD = (.029, .04)
# runtime_v26 review fix: at least half the object weight (critic R5), at most the placement gate's fraction.
ACTIVATION_LOAD_FRACTION_BOUNDS = (.5, 1.)
PINCH_REDUCTION_KEYS = ("start_height_m", "delta_rad")
# runtime_v26 review fix (critic item 10): only after the first chamfer/seat contact, at most 4 mm above the seat,
# keeping >= ~30 N of pinch (run_116: 44 N saturated, 1296 N/rad below the 0.029 rad saturation edge).
PINCH_REDUCTION_START_HEIGHT_MAXIMUM_M = .004
PINCH_REDUCTION_DELTA_BOUNDS_RAD = (.02, .03)
# The raw JIG normal force that counts as the first JIG contact of a place (the hand-object contact threshold).
PINCH_REDUCTION_CONTACT_FORCE_N = .05
GRAVITY_M_S2 = 9.81


def _finite(value):
    return isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)


def validate_supported_grip_relief(settings, *, minimum_load_fraction=None):
    """Validated relief settings; ``minimum_load_fraction`` bounds activation_minimum_load_fraction from above."""
    if not isinstance(settings, Mapping) or not KEYS <= set(settings) or set(settings) - KEYS - OPTIONAL:
        unknown = sorted(map(str, set(settings) - KEYS - OPTIONAL)) if isinstance(settings, Mapping) else []
        # Exactly the runtime_v25 text unless a runtime_v26 key is declared (then the unknown keys are named).
        declared = isinstance(settings, Mapping) and bool(set(settings) & V26_OPTIONAL)
        raise ValueError("supported_grip_relief requires exactly preload_rad, max_target_speed_rad_s, activation_dwell_s"
                         " and optionally support_dropout_grace_s"
                         + (", jump_start_rad, relief_profile, activation_minimum_load_fraction"
                            + (" (unknown: " + ", ".join(unknown) + ")" if unknown else "") if declared else ""))
    profile = settings.get("relief_profile")
    if "relief_profile" in settings and (not isinstance(profile, str) or profile != FAST_PROFILE):
        raise ValueError("supported_grip_relief.relief_profile must be 'fast' (absent: the 0.03 rad/s ramp bound)")
    speed_bound = FAST_MAX_TARGET_SPEED_RAD_S if profile == FAST_PROFILE else DEFAULT_MAX_TARGET_SPEED_RAD_S
    # The optional load-only dropout grace keeps PlacementHold's (0, 0.2] s bound.
    bounds = {"preload_rad": (.005, .05), "max_target_speed_rad_s": (0., speed_bound), "activation_dwell_s": (.1, 2.),
              "support_dropout_grace_s": (0., .2)}
    result = {}
    for key, (low, high) in bounds.items():
        if key not in settings:
            continue
        value = settings[key]
        if not _finite(value) or not low <= value <= high or value <= 0:
            raise ValueError(f"supported_grip_relief.{key} outside bounded range")
        result[key] = float(value)
    if "jump_start_rad" in settings:
        value, (low, high) = settings["jump_start_rad"], JUMP_START_BOUNDS_RAD
        if not _finite(value) or not low <= value <= high:
            raise ValueError("supported_grip_relief.jump_start_rad must be finite in [0.029, 0.04] rad")
        result["jump_start_rad"] = float(value)
    if "activation_minimum_load_fraction" in settings:
        value, (low, high) = settings["activation_minimum_load_fraction"], ACTIVATION_LOAD_FRACTION_BOUNDS
        if minimum_load_fraction is not None:
            if not _finite(minimum_load_fraction):
                raise ValueError("supported_grip_relief.activation_minimum_load_fraction needs a finite "
                                 "placement_support.minimum_load_fraction")
            high = min(high, float(minimum_load_fraction))
        if not _finite(value) or not low <= value <= high:
            raise ValueError("supported_grip_relief.activation_minimum_load_fraction must be finite in "
                             "[0.5, placement_support.minimum_load_fraction]")
        result["activation_minimum_load_fraction"] = float(value)
    if "relief_profile" in settings:
        result["relief_profile"] = FAST_PROFILE
    return result


def validate_place_pinch_reduction(settings, relief_settings=None):
    """tasks.<task>.place_pinch_reduction: start_height_m in (0, 0.03] m, delta_rad in [0.005, 0.03] rad.

    With the relief settings, delta_rad must also be at least preload_rad: the relief never tightens a reduced pinch.
    """
    if not isinstance(settings, Mapping) or set(settings) != set(PINCH_REDUCTION_KEYS):
        unknown = sorted(map(str, set(settings) - set(PINCH_REDUCTION_KEYS))) if isinstance(settings, Mapping) else []
        raise ValueError("place_pinch_reduction requires exactly start_height_m and delta_rad"
                         + (" (unknown: " + ", ".join(unknown) + ")" if unknown else ""))
    start, delta = settings["start_height_m"], settings["delta_rad"]
    if not _finite(start) or not 0. < start <= PINCH_REDUCTION_START_HEIGHT_MAXIMUM_M:
        raise ValueError("place_pinch_reduction.start_height_m must be finite in (0, 0.004] m")
    low, high = PINCH_REDUCTION_DELTA_BOUNDS_RAD
    if not _finite(delta) or not low <= delta <= high:
        raise ValueError("place_pinch_reduction.delta_rad must be finite in [0.02, 0.03] rad")
    if relief_settings is not None and delta < relief_settings["preload_rad"]:
        raise ValueError("place_pinch_reduction.delta_rad must be at least supported_grip_relief.preload_rad")
    return {"start_height_m": float(start), "delta_rad": float(delta)}


def relief_task_options(task, *, slot_pose=None):
    """Constructor options for the declared runtime_v26 task keys; {} when none is declared (runtime_v25)."""
    options = {}
    support = task.get("placement_support") or {}
    relief = task.get("supported_grip_relief") or {}
    if "filter_s" in support or "activation_minimum_load_fraction" in relief:
        options["placement_support"] = support
    if "place_pinch_reduction" in task:
        options.update(place_pinch_reduction=task["place_pinch_reduction"],
                       slot_pose=task["target"]["pose"] if slot_pose is None else slot_pose)
    return options


def supported_grip_relief_contract(task):
    settings = task.get("supported_grip_relief")
    if settings is None:
        return None
    support = task.get("placement_support") or {}
    settings = validate_supported_grip_relief(settings, minimum_load_fraction=(
        support.get("minimum_load_fraction") if "activation_minimum_load_fraction" in settings else None))
    contract = {"schema_version": SCHEMA, "settings": settings,
                "stage": "place", "arm": "current_measured_holding_arm",
                "target": "measured_jaw_minus_positive_closing_preload_bounded_by_first_activation",
                "rate_reference": "previous_applied_jaw_command",
                "loss_of_support": "rate_limited_original_close",
                "success_gates_changed": False, "physics_parameters_changed": False}
    if "support_dropout_grace_s" in settings:
        contract["support_dropout_grace"] = ("engaged_relief_holds_previous_applied_target_while_only_measured_support"
                                             "_guards_fail_within_grace_of_its_last_fully_valid_sample")
    if "jump_start_rad" in settings:
        contract["activation_jump_start"] = ("each_activation_slews_the_previous_applied_target_at_the_gripper_speed_limit"
                                             "_to_jump_start_rad_beyond_the_measured_jaw_toward_closed_never_past_the"
                                             "_relief_goal_then_ramps_at_max_target_speed_rad_s")
        contract["loss_of_support_reclose"] = ("after_an_activation_a_deactivated_relief_outside_the_grace_reclose"
                                               "s_a_target_beyond_jump_start_rad_of_the_measured_jaw_at_the_gripper"
                                               "_speed_limit_to_that_edge_then_at_max_target_speed_rad_s")
    if "activation_minimum_load_fraction" in settings:
        contract["activation_load"] = ("relief_support_guard_uses_activation_minimum_load_fraction_of_the_measured_object"
                                       "_weight;_the_opening_dwell_and_success_gates_keep_placement_support_minimum_load"
                                       "_fraction")
    if "filter_s" in support:
        from .placement_support_filter import validate_filter_s
        contract["support_force"] = {"source": "placement_support_filter", "filter_s": validate_filter_s(support["filter_s"]),
                                     "filter": "causal_trailing_boxcar_zero_order_hold_zero_before_place_entry",
                                     "raw_jig_support_confirmed_unchanged": True}
    if "place_pinch_reduction" in task:
        contract["place_pinch_reduction"] = {
            "settings": validate_place_pinch_reduction(task["place_pinch_reduction"], settings),
            "law": ("before_the_first_activation_with_slot_geometry_and_capture_guards_and_the_measured_ring_within"
                    "_start_height_m_above_the_seat_the_jaw_target_ramps_to_delta_rad_beyond_the_measured_jaw_toward"
                    "_closed"),
            "contact": "only_after_the_first_measured_jig_contact_of_the_place_raw_normal_force_at_least_0.05_n"}
    return contract


class SupportedGripRelief:
    def __init__(self, settings, *, object_path, support_path, capture_position_m, capture_rotation_rad,
                 object_symmetry_axis=None, object_pivot_radius_m=None,
                 placement_support=None, place_pinch_reduction=None, slot_pose=None):
        if placement_support is not None and not isinstance(placement_support, Mapping):
            raise ValueError("supported_grip_relief placement_support must be the task placement_support mapping")
        minimum_fraction = None if placement_support is None else placement_support.get("minimum_load_fraction")
        self.settings = validate_supported_grip_relief(settings, minimum_load_fraction=minimum_fraction)
        self.object_path, self.support_path = object_path, support_path
        self.capture_position_m, self.capture_rotation_rad = capture_position_m, capture_rotation_rad
        # Declared object symmetry axis (object frame): the capture retention guard then ignores the spin.
        self.object_symmetry_axis = (None if object_symmetry_axis is None
                                     else unit_axis(object_symmetry_axis, "object_symmetry_axis"))
        # Optional pinch-circle radius: the retention slip is then measured at the pinch, not at the centre.
        self.object_pivot_radius_m = (None if object_pivot_radius_m is None
                                      else pivot_radius(object_pivot_radius_m, "object_pivot_radius_m"))
        if self.object_pivot_radius_m is not None and self.object_symmetry_axis is None:
            raise ValueError("object_pivot_radius_m requires object_symmetry_axis")
        self._key = self._last_time = self._valid_since = self._bound = self._engaged_at = None
        # runtime_v26 opt-in state (unused without the corresponding key).
        self._activation_fraction = self.settings.get("activation_minimum_load_fraction")
        if self._activation_fraction is not None and minimum_fraction is None:
            raise ValueError("supported_grip_relief.activation_minimum_load_fraction requires placement_support")
        self._filtered_support = placement_support is not None and "filter_s" in placement_support
        if self._filtered_support:
            from .placement_support_filter import validate_filter_s
            validate_filter_s(placement_support["filter_s"])
        self._load_basis = self._filtered_support or self._activation_fraction is not None
        self._jump_start = self.settings.get("jump_start_rad")
        self._jump_target = None
        self._was_active = False
        self._pinch_contact_s = None
        self._pinch = (None if place_pinch_reduction is None
                       else validate_place_pinch_reduction(place_pinch_reduction, self.settings))
        self._slot_pose = None
        if self._pinch is not None:
            if slot_pose is None:
                raise ValueError("place_pinch_reduction requires the slot target pose")
            slot = np.asarray(slot_pose, dtype=float)
            self._slot_pose = slot.copy() if slot.shape == (4, 4) else pose_to_matrix(slot_pose)
            if not np.isfinite(self._slot_pose).all():
                raise ValueError("place_pinch_reduction requires a finite slot target pose")

    def _declared_placeholders(self, report):
        """Opt-in report keys exist on every report of a declared profile (None while not evaluated)."""
        if self._load_basis:
            report["support_load_basis"] = None
        if self._jump_start is not None:
            report["jump_start"] = None
        if self._pinch is not None:
            report["place_pinch_reduction"] = None

    def _support_basis(self, now, metrics, support):
        """(force, threshold, evidence) of the declared support basis (filtered force and/or activation fraction).

        With the filter the guard force is the smaller of the window mean and the window's sustained force
        (placement_support_filter.py): one or two impulses cannot confirm support for the relief either.
        """
        raw = support.get("normal_force_n")
        sustained = None
        if self._filtered_support:
            filtered = metrics.get("placement_support_filter")
            current = (isinstance(filtered, Mapping) and _finite(filtered.get("measurement_time_s"))
                       and abs(filtered["measurement_time_s"] - now) <= 1e-9)
            mean = filtered.get("filtered_normal_force_n") if current else None
            sustained = filtered.get("sustained_force_n") if current else None
            force = min(mean, sustained) if _finite(mean) and _finite(sustained) else None
            source = "placement_support_filter"
        else:
            force, source = raw, "raw_support_sample"
        gate_minimum = support.get("minimum_force_n")
        if self._activation_fraction is not None:
            mass = support.get("object_mass_kg")
            minimum = self._activation_fraction * mass * GRAVITY_M_S2 if _finite(mass) and mass > 0 else None
            threshold = "activation_minimum_load_fraction"
        else:
            minimum, threshold = gate_minimum, "placement_support.minimum_load_fraction"
        evidence = {"force_source": source, "threshold": threshold,
                    "normal_force_n": float(force) if _finite(force) else None,
                    "raw_normal_force_n": float(raw) if _finite(raw) else None,
                    "threshold_n": float(minimum) if _finite(minimum) else None,
                    "activation_minimum_load_fraction": self._activation_fraction,
                    "success_gate_minimum_force_n": float(gate_minimum) if _finite(gate_minimum) else None,
                    "success_gates_changed": False}
        if self._filtered_support:
            # Declared filter only: the window mean and sustained force behind the guard force (their minimum).
            evidence.update(filtered_mean_normal_force_n=float(mean) if _finite(mean) else None,
                            sustained_force_n=float(sustained) if _finite(sustained) else None)
        return force, minimum, evidence

    def _pinch_reduction(self, observation, guards, measured, closed, direction, active, grace, support, now):
        """Opt-in pre-seat reduced pinch: its checks, availability and jaw goal (None unless engaged)."""
        force = support.get("normal_force_n")
        if (self._pinch_contact_s is None and support.get("available") is True and _finite(force)
                and force >= PINCH_REDUCTION_CONTACT_FORCE_N):
            self._pinch_contact_s = float(now)  # the first measured JIG contact of this place (latched)
        height = None
        try:
            poses = observation["object_poses_w"]
            relative = np.linalg.inv(pose_to_matrix(poses[self.support_path])) @ pose_to_matrix(poses[self.object_path])
            height = float(relative[2, 3] - self._slot_pose[2, 3])
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
            height = None
        checks = {name: passed for name, passed in guards.items() if name not in SUPPORT_GUARDS}
        checks["within_start_height"] = bool(height is not None and math.isfinite(height)
                                             and height <= self._pinch["start_height_m"] + 1e-12)
        checks["jig_contact_measured"] = self._pinch_contact_s is not None
        available = self._bound is None
        engaged = bool(available and not active and not grace and all(checks.values()))
        goal = (closed + direction * max(0., direction * (measured - closed) - self._pinch["delta_rad"])
                if engaged else None)
        return {"start_height_m": self._pinch["start_height_m"], "delta_rad": self._pinch["delta_rad"],
                "remaining_height_m": height, "available_before_first_activation": available,
                "jig_contact_since_s": self._pinch_contact_s, "contact_force_n": PINCH_REDUCTION_CONTACT_FORCE_N,
                "checks": checks, "active": engaged, "goal_rad": None if goal is None else float(goal)}

    @staticmethod
    def _gripper_speed(gripper):
        speed = float(gripper.get("speed_rad_s", .7))
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("supported_grip_relief jump start needs a positive finite gripper speed")
        return speed

    def _jump(self, active, previous, goal, measured, closed, direction, gripper, dt, *, reclose_allowed=False):
        """Opt-in activation jump start: the slew target and its step limit (gripper speed) while slewing; when
        inactive after an activation (``reclose_allowed``), the mirrored fast re-close back to the edge."""
        evidence = {"jump_start_rad": self._jump_start, "activation_edge": False, "slewing": False,
                    "slew_target_rad": None, "step_limit_rad": None, "completed": False,
                    "reclosing": False, "reclose_target_rad": None}
        if not active:
            self._jump_target = None
            if reclose_allowed and _finite(measured):
                d_previous = direction * (previous - closed)
                d_edge = max(0., direction * (measured - closed) - self._jump_start)
                if d_previous > d_edge + 1e-12:
                    evidence.update(reclosing=True, reclose_target_rad=float(closed + direction * d_edge),
                                    step_limit_rad=self._gripper_speed(gripper) * dt)
            return evidence
        d_previous, d_goal = direction * (previous - closed), direction * (goal - closed)
        if not self._was_active:
            # Only an activation latches a slew target: jump_start beyond the measured jaw toward closed.
            evidence["activation_edge"] = True
            d_edge = max(0., direction * (measured - closed) - self._jump_start)
            self._jump_target = closed + direction * d_edge if d_edge > d_previous + 1e-12 else None
        if self._jump_target is not None:
            # Capped by the current relief goal every sample, and never toward closed.
            d_target = min(direction * (self._jump_target - closed), d_goal)
            if d_target <= d_previous + 1e-12:
                self._jump_target = None
            else:
                evidence.update(slewing=True, slew_target_rad=float(closed + direction * d_target),
                                step_limit_rad=self._gripper_speed(gripper) * dt)
        return evidence

    def update(self, command, observation, gripper, *, open_position_rad, closed_position_rad, dt):
        now = observation.get("time_s")
        if not _finite(now) or not _finite(dt) or dt <= 0:
            raise ValueError("supported_grip_relief requires finite observation time and positive dt")
        stage, arm = command.get("stage_id"), command.get("arm")
        key = (stage, arm)
        if key != self._key:
            self._valid_since = self._bound = self._engaged_at = None
            self._key = key
            self._jump_target, self._was_active = None, False
            self._pinch_contact_s = None
        previous_time = self._last_time
        if previous_time is not None and now <= previous_time:
            raise ValueError("supported_grip_relief requires increasing measured time")
        self._last_time = now
        controlled = stage == "place" and command.get("hand_command") in ("close", "hold")
        report = {"schema_version": SCHEMA, "measurement_time_s": float(now), "stage_id": stage,
                  "active_arm": arm, "controlled": controlled, "active": False,
                  "guard_valid": False, "guard_results": {}, "activation_bound_rad": self._bound,
                  "continuous_guard_dwell_s": 0., "measured_joint_position_rad": None,
                  "previous_applied_target_rad": None, "unlimited_target_rad": None, "target_rad": None,
                  "applied_step_rad": 0., "maximum_step_rad": self.settings["max_target_speed_rad_s"] * dt,
                  "dropout_reason": None, "settings": dict(self.settings)}
        grace_s = self.settings.get("support_dropout_grace_s")
        if grace_s is not None:
            report.update(support_dropout_grace_active=False, support_dropout_age_s=None)
        self._declared_placeholders(report)
        if not controlled:
            self._valid_since = self._engaged_at = None
            self._jump_target, self._was_active = None, False
            return report
        closed = gripper_closed_position(gripper, closed_position_rad, open_position_rad)
        opened = gripper_open_position(gripper, open_position_rad)
        direction = float(np.sign(opened - closed))
        name = gripper["joint_name"]
        previous = observation.get("commanded_upper_joint_positions", {}).get(name)
        if not _finite(previous):
            raise ValueError("supported_grip_relief missing finite previous APPLIED jaw target")
        # A corrupt applied reference cannot be safely repaired with a clipped jump.
        gripper_closed_position(gripper, previous, open_position_rad)
        measured = observation.get("joint_positions", {}).get(name)
        metrics = command.get("stationary_metrics") or {}
        gates = metrics.get("gates") or {}
        contact = observation.get("contact_evidence", {}).get(arm, {}).get(self.object_path, {})
        support = metrics.get("support_evidence") or {}
        basis = None
        if self._load_basis:
            force, minimum, basis = self._support_basis(now, metrics, support)
        else:
            force, minimum = support.get("normal_force_n"), support.get("minimum_force_n")
        load_measured = _finite(force) and _finite(minimum) and minimum > 0
        # A declared support basis replaces the raw 0.8 gate in the relief's own support guard (support_load).
        names = ("slot_pose_reached", "slot_normal_reached", "jig_anchor_retained") + (
            () if self._load_basis else ("jig_support_confirmed",))
        guards = {n: gates.get(n) is True for n in names}
        guards.update(holding_arm=arm in ("left", "right") and command.get("holding_arm", arm) == arm,
                      metrics_current=_finite(metrics.get("measurement_time_s"))
                          and abs(metrics["measurement_time_s"] - now) <= 1e-9,
                      object_identity=command.get("object_path") == self.object_path,
                      contact_available=contact.get("available") is True,
                      bilateral_contact=contact.get("bilateral_contact_confirmed") is True,
                      measured_hand_closed=observation.get("hand_closed", {}).get(arm) is True,
                      measured_jaw_finite=_finite(measured),
                      support_identity=support.get("available") is True and support.get("object_path") == self.object_path
                          and support.get("support_path") == self.support_path,
                      support_load=load_measured and force >= minimum,
                      fresh_measurement=previous_time is None or now - previous_time <= dt * 1.5 + 1e-9)
        retention = None
        try:
            measured_g = np.linalg.inv(pose_to_matrix(observation["object_poses_w"][self.object_path])) @ pose_to_matrix(
                observation["tcp_poses_w"][arm])
            if self.object_symmetry_axis is None:
                retention = pose_goal_errors(matrix_to_pose(measured_g), command["retention_object_to_tcp"])
            else:
                position, rotation, spin = symmetric_capture_errors(
                    measured_g, pose_to_matrix(command["retention_object_to_tcp"]), self.object_symmetry_axis,
                    self.object_pivot_radius_m)
                retention = {"position_error_m": position, "rotation_error_rad": rotation, "object_spin_rad": spin}
            guards["immutable_capture_retained"] = (retention["position_error_m"] <= self.capture_position_m
                and retention["rotation_error_rad"] <= self.capture_rotation_rad)
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
            guards["immutable_capture_retained"] = False
        guards["measured_jaw_within_opening_span"] = bool(_finite(measured)
            and 0 <= direction * (measured - closed) < abs(opened - closed) - .1)
        valid = all(guards.values())
        # Opt-in grace: a relief active at its last fully valid sample bridges a measured (finite) support-only
        # under-load until grace_s after that sample: dwell kept, applied target held. Activations need a real dwell.
        age = (now - self._engaged_at if grace_s is not None and self._engaged_at is not None and not valid
               and load_measured and {n for n, passed in guards.items() if not passed} <= SUPPORT_GUARDS else None)
        grace = bool(age is not None and age <= grace_s + 1e-9)
        if valid and self._valid_since is None:
            self._valid_since = now
        elif not valid and not grace:
            self._valid_since = self._engaged_at = None
        dwell = 0. if self._valid_since is None else now - self._valid_since
        active = valid and dwell >= self.settings["activation_dwell_s"] - 1e-9
        if valid:
            self._engaged_at = now if active else None
        goal = previous if grace else closed
        reduction = None
        if self._pinch is not None:
            reduction = self._pinch_reduction(observation, guards, measured, closed, direction, active, grace,
                                              support, now)
            if reduction["active"]:
                goal = reduction["goal_rad"]
        if active:
            distance = max(0., direction * (measured - closed) - self.settings["preload_rad"])
            if self._bound is None:
                self._bound = closed + direction * distance
            distance = min(distance, direction * (self._bound - closed))
            goal = closed + direction * distance
        jump = None
        if self._jump_start is not None:
            # The mirrored re-close needs a previous activation in this place, no dropout grace and no engaged
            # pre-seat reduction (which is only available before the first activation anyway).
            jump = self._jump(active, previous, goal, measured, closed, direction, gripper, dt,
                              reclose_allowed=bool(not active and not grace and self._bound is not None
                                                   and not (reduction is not None and reduction["active"])))
        if jump is not None and jump["slewing"]:
            # The activation jump start slews at the gripper's own speed limit; the relief rate applies after it.
            limit = jump["step_limit_rad"]
            change = float(np.clip(jump["slew_target_rad"] - previous, -limit, limit))
        elif jump is not None and jump["reclosing"]:
            # Loss of support after an activation: back to the pinch saturation edge at the same gripper speed.
            limit = jump["step_limit_rad"]
            change = float(np.clip(jump["reclose_target_rad"] - previous, -limit, limit))
        else:
            change = float(np.clip(goal - previous, -report["maximum_step_rad"], report["maximum_step_rad"]))
        target = gripper_closed_position(gripper, previous + change, open_position_rad)
        if jump is not None and jump["slewing"] and abs(target - jump["slew_target_rad"]) <= 1e-12:
            self._jump_target = None
            jump["completed"] = True
        report.update(active=bool(active), guard_valid=valid, guard_results=guards,
                      continuous_guard_dwell_s=float(dwell), activation_bound_rad=self._bound,
                      measured_joint_position_rad=float(measured) if _finite(measured) else None,
                      previous_applied_target_rad=float(previous), unlimited_target_rad=float(goal),
                      target_rad=target, applied_step_rad=change, retention=retention,
                      dropout_reason=None if valid else ",".join(n for n, passed in guards.items() if not passed))
        if grace_s is not None:
            report.update(support_dropout_grace_active=grace, support_dropout_age_s=None if age is None else float(age))
        if basis is not None:
            report["support_load_basis"] = basis
        if jump is not None:
            report["jump_start"] = jump
        if reduction is not None:
            report["place_pinch_reduction"] = reduction
        self._was_active = bool(active)
        return report
