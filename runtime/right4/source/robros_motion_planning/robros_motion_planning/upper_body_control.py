"""Optional measured-pelvis Cartesian feedback for waist3 plus one arm7.

This module emits joint targets only. It has no simulator API, state writes,
leg outputs, collision avoidance or balance guarantee. The measured pelvis is
a kinematic boundary condition, never a root pose command. Runtime admission
must separately enforce the opted-in stages and joint-controller ownership.
"""
from collections.abc import Mapping
import math
from copy import deepcopy
from numbers import Real

import numpy as np
from scipy.optimize import lsq_linear

from .constants import TELEOPERATION_JOINT_NAMES, TELEOPERATION_RANGES
from .robot_model import arm_joint_names
from .snapshot import pose_to_matrix, matrix_to_pose
from .task_control import ArmServo, rotation_error

WAIST = ("Joint_Waist_Yaw", "Joint_Waist_Roll", "Joint_Waist_Pitch")
PELVIS = "Link_Waist_Pitch"
UPPER_BODY_IK_STAGES = frozenset(("pregrasp", "approach", "lift", "transfer", "place"))
_SHARED_LIMITS = dict(zip(TELEOPERATION_JOINT_NAMES, TELEOPERATION_RANGES))
# Waist command speed caps: the measured posture controller's 0.08 rad/s, or the opt-in (runtime_v26)
# ``waist_speed_profile: simulation_fast`` simulation cap. Command caps only, not actuator ratings.
# The runtime_v26 review bounded the profile at 0.25 rad/s (critic/spec: 0.16-0.25 rad/s) and keeps the contact
# stages (the guarded approach and the place) at the 0.08 rad/s cap: the faster cap serves free-space IK stages.
WAIST_SPEED_LIMIT_RAD_S = .08
WAIST_SPEED_PROFILES = {"simulation_fast": .25}
WAIST_SPEED_CONTACT_STAGES = ("approach", "place")


def waist_speed_limit(profile):
    """The max_waist_speed bound of a waist speed profile (None: the default 0.08 rad/s)."""
    if profile is None:
        return WAIST_SPEED_LIMIT_RAD_S
    if not isinstance(profile, str) or profile not in WAIST_SPEED_PROFILES:
        raise ValueError("waist_speed_profile must be " + ", ".join(sorted(WAIST_SPEED_PROFILES)))
    return WAIST_SPEED_PROFILES[profile]
_ARM_JOINTS = frozenset(arm_joint_names("left") + arm_joint_names("right"))


def _number(value, name, *, low=0., high=math.inf, inclusive_low=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(name + " must be finite numeric")
    value = float(value)
    if not math.isfinite(value) or (value < low if inclusive_low else value <= low) or value > high:
        raise ValueError(name + " is outside permitted bounds")
    return value


def _waist_limits(value, shared_limits):
    if not isinstance(value, Mapping) or set(value) != set(WAIST):
        raise ValueError("waist_command_limits_rad must name exactly the three waist joints")
    result = {}
    for name in WAIST:
        pair = value[name]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"waist_command_limits_rad.{name} needs [lower, upper]")
        low, high = [_number(v, name, low=-math.inf, inclusive_low=True) for v in pair]
        if not shared_limits[name][0] <= low < high <= shared_limits[name][1]:
            raise ValueError(f"waist_command_limits_rad.{name} must narrow the shared command range")
        result[name] = [low, high]
    return result


def _arm_limits(value, command_limits, allowed_names=_ARM_JOINTS):
    """Copy explicit arm bounds, rejecting expansion of every known envelope.

    The simulator-free shared table contains shoulder/elbow ranges only.
    Wrist physical limits and all model-specific margins are checked again
    by the constructor against its complete effective command envelope.
    """
    if (not isinstance(value, Mapping) or not value
            or set(value) - set(allowed_names)):
        raise ValueError("arm_command_limits_rad must name a nonempty subset of arm joints")
    result = {}
    for name, pair in value.items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"arm_command_limits_rad.{name} needs [lower, upper]")
        low, high = [_number(v, "arm_command_limits_rad." + name,
                            low=-math.inf, inclusive_low=True) for v in pair]
        limits = command_limits.get(name, (-math.inf, math.inf))
        if not limits[0] <= low < high <= limits[1]:
            raise ValueError(f"arm_command_limits_rad.{name} must narrow the existing command range")
        result[name] = [low, high]
    return result


def _stage_limits(value, owned_stages, command_limits, allowed_names=None):
    """Validate only narrowing of the already established global envelope."""
    allowed_names = set(WAIST) | _ARM_JOINTS if allowed_names is None else set(allowed_names)
    if (not isinstance(value, Mapping) or not value
            or any(not isinstance(stage, str) or stage not in owned_stages for stage in value)):
        raise ValueError("stage_command_limits_rad must name a nonempty subset of owned stages")
    result = {}
    for stage, overrides in value.items():
        label = "stage_command_limits_rad." + stage
        if (not isinstance(overrides, Mapping) or not overrides
                or set(overrides) - allowed_names):
            raise ValueError(label + " must name only waist/active-arm joints")
        result[stage] = {}
        for name, pair in overrides.items():
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(label + "." + name + " needs [lower, upper]")
            low, high = [_number(v, label + "." + name,
                                low=-math.inf, inclusive_low=True) for v in pair]
            limits = command_limits.get(name, (-math.inf, math.inf))
            if not limits[0] <= low < high <= limits[1]:
                raise ValueError(label + "." + name + " must narrow the global command range")
            result[stage][name] = [low, high]
    return result


def validate_upper_body_ik(settings):
    """Validate the explicit opt-in settings without loading a robot or simulator.

    Constructor validation additionally intersects these shared teleoperation
    limits with the loaded model's authored limits. The optional arm command
    margin defaults to zero; stage ownership and stationary-pelvis requirements
    belong to the task validator and runtime.
    """
    required = {"stages", "waist_command_limits_rad", "max_waist_speed_rad_s"}
    optional = {"arm_command_margin_rad", "arm_command_limits_rad", "pelvis_motion_compensation",
                "target_motion_feedforward_stages", "stage_command_limits_rad", "posture_saturation_aware",
                "transfer_early_command_limits", "transfer_secondary_arm_schedule", "stage_waist_hold",
                "waist_speed_profile"}
    if (not isinstance(settings, Mapping) or not required <= set(settings)
            or set(settings) - required - optional):
        # Exactly the runtime_v25 text unless a runtime_v26 key (stage_waist_hold, waist_speed_profile) is declared.
        declared = isinstance(settings, Mapping) and bool(set(settings) & {"stage_waist_hold", "waist_speed_profile"})
        raise ValueError("upper_body_ik requires stages, waist_command_limits_rad and max_waist_speed_rad_s; optional fields are arm_command_margin_rad, arm_command_limits_rad, pelvis_motion_compensation, target_motion_feedforward_stages, stage_command_limits_rad, posture_saturation_aware, transfer_early_command_limits and transfer_secondary_arm_schedule"
                         + (" (runtime_v26 opt-in: stage_waist_hold and waist_speed_profile)" if declared else ""))
    profile = settings.get("waist_speed_profile")
    if "waist_speed_profile" in settings and (not isinstance(profile, str) or profile not in WAIST_SPEED_PROFILES):
        raise ValueError("upper_body_ik.waist_speed_profile must be " + ", ".join(sorted(WAIST_SPEED_PROFILES)))
    stages = settings["stages"]
    if (not isinstance(stages, list) or not stages
            or any(not isinstance(stage, str) or stage not in UPPER_BODY_IK_STAGES for stage in stages)
            or len(set(stages)) != len(stages)):
        raise ValueError("upper_body_ik.stages must be a unique nonempty subset of pregrasp/approach/lift/transfer/place")
    result = {"stages": list(stages),
            "waist_command_limits_rad": _waist_limits(settings["waist_command_limits_rad"], _SHARED_LIMITS),
            "max_waist_speed_rad_s": _number(settings["max_waist_speed_rad_s"],
                                           "max_waist_speed_rad_s", high=waist_speed_limit(profile))}
    if "waist_speed_profile" in settings:
        result["waist_speed_profile"] = profile
    if "arm_command_margin_rad" in settings:
        result["arm_command_margin_rad"] = _number(settings["arm_command_margin_rad"],
            "arm_command_margin_rad", high=.1, inclusive_low=True)
    if "arm_command_limits_rad" in settings:
        margin = result.get("arm_command_margin_rad", 0.)
        shared = {name: (low + margin, high - margin)
                  for name, (low, high) in _SHARED_LIMITS.items() if name in _ARM_JOINTS}
        result["arm_command_limits_rad"] = _arm_limits(settings["arm_command_limits_rad"], shared)
    if "stage_command_limits_rad" in settings:
        margin = result.get("arm_command_margin_rad", 0.)
        known = {name: (low + margin, high - margin)
                 for name, (low, high) in _SHARED_LIMITS.items() if name in _ARM_JOINTS}
        known.update(result["waist_command_limits_rad"])
        known.update(result.get("arm_command_limits_rad", {}))
        result["stage_command_limits_rad"] = _stage_limits(
            settings["stage_command_limits_rad"], stages, known)
    if "pelvis_motion_compensation" in settings:
        if type(settings["pelvis_motion_compensation"]) is not bool:
            raise ValueError("pelvis_motion_compensation must be a boolean")
        result["pelvis_motion_compensation"] = settings["pelvis_motion_compensation"]
    if "transfer_early_command_limits" in settings:
        # Opt-in (runtime_v22): narrower transfer command limits until the object waypoint index first reaches
        # until_waypoint (then the transfer stage limits apply for the rest of the stage, latched).
        early = settings["transfer_early_command_limits"]
        if (not isinstance(early, Mapping) or set(early) != {"until_waypoint", "limits"}
                or type(early["until_waypoint"]) is not int or early["until_waypoint"] < 1):
            raise ValueError("transfer_early_command_limits needs an integer until_waypoint >= 1 and limits")
        if "transfer" not in result.get("stage_command_limits_rad", {}):
            raise ValueError("transfer_early_command_limits requires stage_command_limits_rad.transfer")
        margin = result.get("arm_command_margin_rad", 0.)
        known = {name: (low + margin, high - margin)
                 for name, (low, high) in _SHARED_LIMITS.items() if name in _ARM_JOINTS}
        known.update(result["waist_command_limits_rad"])
        known.update(result.get("arm_command_limits_rad", {}))
        known.update({name: tuple(pair) for name, pair in result["stage_command_limits_rad"]["transfer"].items()})
        try:
            limits = _stage_limits({"transfer": early["limits"]}, stages, known)["transfer"]
        except ValueError as error:
            raise ValueError("transfer_early_command_limits: " + str(error)) from None
        result["transfer_early_command_limits"] = {"until_waypoint": early["until_waypoint"], "limits": limits}
    if "transfer_secondary_arm_schedule" in settings:
        # Opt-in (runtime_v22): during transfer the INACTIVE arm moves from its transfer-entry command to
        # joint_positions as the object waypoint index goes from_waypoint -> to_waypoint, rate limited.
        # The task validator and the servo factory check that every joint belongs to the arm that is NOT
        # servoed in transfer and lies inside the model command limits; collision checks are offline.
        schedule = settings["transfer_secondary_arm_schedule"]
        keys = {"from_waypoint", "to_waypoint", "joint_positions", "max_speed_rad_s"}
        if (not isinstance(schedule, Mapping) or set(schedule) != keys
                or type(schedule["from_waypoint"]) is not int or type(schedule["to_waypoint"]) is not int
                or not 0 <= schedule["from_waypoint"] < schedule["to_waypoint"]):
            raise ValueError("transfer_secondary_arm_schedule needs integer 0 <= from_waypoint < to_waypoint, "
                             "joint_positions and max_speed_rad_s")
        if "transfer" not in stages:
            raise ValueError("transfer_secondary_arm_schedule requires the transfer stage")
        positions = schedule["joint_positions"]
        if not isinstance(positions, Mapping) or not positions or set(positions) - _ARM_JOINTS:
            raise ValueError("transfer_secondary_arm_schedule.joint_positions must name arm joints")
        checked = {}
        for name, value in positions.items():
            value = _number(value, "transfer_secondary_arm_schedule." + name, low=-math.inf, inclusive_low=True)
            if name in _SHARED_LIMITS:
                low, high = _SHARED_LIMITS[name]
                if not low <= value <= high:
                    raise ValueError("transfer_secondary_arm_schedule." + name + " outside the shared command range")
            checked[name] = value
        result["transfer_secondary_arm_schedule"] = {
            "from_waypoint": schedule["from_waypoint"], "to_waypoint": schedule["to_waypoint"],
            "joint_positions": checked,
            "max_speed_rad_s": _number(schedule["max_speed_rad_s"], "transfer_secondary_arm_schedule.max_speed_rad_s", high=.3)}
    if "posture_saturation_aware" in settings:
        if type(settings["posture_saturation_aware"]) is not bool:
            raise ValueError("posture_saturation_aware must be a boolean")
        result["posture_saturation_aware"] = settings["posture_saturation_aware"]
    if "target_motion_feedforward_stages" in settings:
        selected = settings["target_motion_feedforward_stages"]
        if (not isinstance(selected, list)
                or any(not isinstance(stage, str) or stage not in {"lift", "transfer", "place"}
                       or stage not in stages for stage in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError("upper_body_ik.target_motion_feedforward_stages must be a unique list "
                             "of owned lift/transfer/place stages")
        result["target_motion_feedforward_stages"] = list(selected)
    if "stage_waist_hold" in settings:
        # Opt-in (runtime_v26): in each listed stage the waist command stays exactly at the value applied when the
        # stage starts (latched at its first control step, zero-width IK bounds); see latch_stage_waist_hold.
        held = settings["stage_waist_hold"]
        if (not isinstance(held, list) or not held
                or any(not isinstance(stage, str) or stage not in UPPER_BODY_IK_STAGES or stage not in stages
                       for stage in held)
                or len(set(held)) != len(held)):
            raise ValueError("upper_body_ik.stage_waist_hold must be a unique nonempty list of owned "
                             "pregrasp/approach/lift/transfer/place stages")
        for stage in held:
            declared = sorted(set(result.get("stage_command_limits_rad", {}).get(stage, {})) & set(WAIST))
            if declared:
                raise ValueError(f"upper_body_ik.stage_waist_hold: {stage} also declares waist entries in "
                                 "stage_command_limits_rad (" + ", ".join(declared) + "); a held stage keeps its "
                                 "entry waist command instead")
        early = sorted(set(result.get("transfer_early_command_limits", {}).get("limits", {})) & set(WAIST))
        if "transfer" in held and early:
            raise ValueError("upper_body_ik.stage_waist_hold: transfer also declares waist entries in "
                             "transfer_early_command_limits (" + ", ".join(early) + ")")
        result["stage_waist_hold"] = list(held)
    return result


def build_upper_body_servos(model, settings, arm, *, max_joint_speed=.6, max_tracking_error=.15,
                            stage_arms=None):
    """Return the unchanged global servo and independent narrowed stage servos.

    The runtime selects exactly one of these per solve. Global posture-guide
    validation continues to use the global servo, so bounds on transfer/place
    cannot modify or invalidate acquisition preferences in other stages.
    """
    settings = validate_upper_body_ik(settings)
    if stage_arms is None:
        stage_arms = {}
    if (not isinstance(stage_arms, Mapping) or set(stage_arms) - set(settings["stages"])
            or any(value not in ("left", "right") for value in stage_arms.values())):
        raise ValueError("stage_arms must map owned stages to left/right")
    kwargs = dict(waist_command_limits=settings["waist_command_limits_rad"],
                  max_joint_speed=max_joint_speed, max_tracking_error=max_tracking_error,
                  max_waist_speed=settings["max_waist_speed_rad_s"],
                  arm_command_margin_rad=settings.get("arm_command_margin_rad", 0.),
                  arm_command_limits_rad=settings.get("arm_command_limits_rad"),
                  posture_saturation_aware=settings.get("posture_saturation_aware", False),
                  waist_speed_profile=settings.get("waist_speed_profile"))
    base = UpperBodyServo(model, **kwargs)
    profile = settings.get("waist_speed_profile")

    def stage_kwargs(stage):
        """Opt-in waist_speed_profile: the contact stages keep the 0.08 rad/s cap (runtime_v26 review fix)."""
        if profile is None or stage not in WAIST_SPEED_CONTACT_STAGES:
            return kwargs
        return {**kwargs, "max_waist_speed": min(settings["max_waist_speed_rad_s"], WAIST_SPEED_LIMIT_RAD_S)}

    if profile is not None:
        base.waist_speed_scope = "free_space_stages"
    stages = {}
    if "stage_command_limits_rad" in settings:
        for stage, value in settings["stage_command_limits_rad"].items():
            limits = _stage_limits({stage: value}, settings["stages"], base.limits,
                                   base.names(stage_arms.get(stage, arm)))[stage]
            waist = {name: list(pair) for name, pair in base.waist_command_limits.items()}
            waist.update({name: pair for name, pair in limits.items() if name in WAIST})
            arms = {name: list(pair) for name, pair in base.arm_command_limits_rad.items()}
            arms.update({name: pair for name, pair in limits.items() if name not in WAIST})
            servo = UpperBodyServo(model, **{**stage_kwargs(stage), "waist_command_limits": waist,
                                            "arm_command_limits_rad": arms or None})
            servo.stage_id = stage
            servo.stage_command_limits_rad = {name: list(pair) for name, pair in limits.items()}
            stages[stage] = servo
    if profile is not None:
        for stage in WAIST_SPEED_CONTACT_STAGES:
            if stage not in settings["stages"]:
                continue
            if stage not in stages:  # the base servo's limits with the contact-stage waist cap
                stages[stage] = UpperBodyServo(model, **stage_kwargs(stage))
            stages[stage].waist_speed_scope = "contact_stage_default_cap"
        for stage, servo in stages.items():
            if stage not in WAIST_SPEED_CONTACT_STAGES:
                servo.waist_speed_scope = "free_space_stages"
    transfer_arm = stage_arms.get("transfer", arm)
    early = settings.get("transfer_early_command_limits")
    if early is not None:
        transfer = stages["transfer"]
        try:
            checked = _stage_limits({"transfer": early["limits"]}, settings["stages"], transfer.limits,
                                    base.names(transfer_arm))["transfer"]
        except ValueError as error:
            raise ValueError("transfer_early_command_limits: " + str(error)) from None
        limits = {**transfer.stage_command_limits_rad, **checked}
        waist = {name: list(pair) for name, pair in transfer.waist_command_limits.items()}
        waist.update({name: list(pair) for name, pair in limits.items() if name in WAIST})
        arms = {name: list(pair) for name, pair in transfer.arm_command_limits_rad.items()}
        arms.update({name: list(pair) for name, pair in limits.items() if name not in WAIST})
        servo = UpperBodyServo(model, **{**kwargs, "waist_command_limits": waist, "arm_command_limits_rad": arms or None})
        servo.stage_id = "transfer"
        servo.stage_command_limits_rad = {name: list(pair) for name, pair in limits.items()}
        servo.until_waypoint = early["until_waypoint"]
        transfer.early_servo = servo
    schedule = settings.get("transfer_secondary_arm_schedule")
    if schedule is not None:
        owned = set(base.names(transfer_arm))
        for name, value in schedule["joint_positions"].items():
            if name in owned:
                raise ValueError("transfer_secondary_arm_schedule may name only the arm that is not servoed in transfer: "
                                 + name)
            low, high = base.limits[name]
            if not low <= value <= high:
                raise ValueError("transfer_secondary_arm_schedule." + name + " outside the model command limits")
    return base, stages


def select_transfer_servo(servo, waypoint_index, released):
    """Opt-in transfer_early_command_limits selection (runtime_v22).

    Returns (selected servo, released, evidence). The narrower early servo is used until the object
    waypoint index first reaches ``until_waypoint``; from then on (latched) the transfer stage servo is used,
    so a later or repeated index never narrows the box again. Without an early servo nothing changes and
    the evidence is None. A non-integer index keeps the narrower box (conservative) without releasing.
    """
    early = getattr(servo, "early_servo", None)
    if early is None:
        return servo, released, None
    if not released and type(waypoint_index) is int and waypoint_index >= early.until_waypoint:
        released = True
    selected = servo if released else early
    return selected, released, {"active": selected is early, "until_waypoint": early.until_waypoint,
                                "object_waypoint_index": waypoint_index,
                                "limits_rad": deepcopy(early.stage_command_limits_rad)}


def apply_transfer_secondary_arm_schedule(schedule, state, targets, waypoint_index, dt, owned_joints=()):
    """Opt-in inactive-arm transfer schedule (runtime_v22); mutates ``targets`` for the scheduled joints.

    ``state`` is None on the first transfer call; it then holds the transfer-entry command of the scheduled
    joints and the goal fraction reached so far. The fraction follows the object waypoint index over
    [from_waypoint, to_waypoint] and never decreases (a missing or non-integer index holds it); each joint
    moves toward start + fraction * (goal - start) by at most max_speed_rad_s * dt. A scheduled joint that the
    servo owns in this step (``owned_joints``) is a configuration error. Returns (state, evidence).
    """
    names = list(schedule["joint_positions"])
    overlap = sorted(set(names) & set(owned_joints))
    if overlap:
        raise ValueError("transfer_secondary_arm_schedule names joints owned by the transfer servo: " + ", ".join(overlap))
    if state is None:
        state = {"start": {name: float(targets[name]) for name in names}, "fraction": 0.}
    span = schedule["to_waypoint"] - schedule["from_waypoint"]
    fraction = state["fraction"]
    if type(waypoint_index) is int:
        fraction = max(fraction, float(np.clip((waypoint_index - schedule["from_waypoint"]) / span, 0., 1.)))
    state = {"start": state["start"], "fraction": fraction}
    step = float(schedule["max_speed_rad_s"]) * float(dt)
    for name in names:
        start = state["start"][name]
        desired = start + fraction * (schedule["joint_positions"][name] - start)
        targets[name] = float(targets[name]) + float(np.clip(desired - float(targets[name]), -step, step))
    return state, {"fraction": fraction, "object_waypoint_index": waypoint_index,
                   "targets_rad": {name: float(targets[name]) for name in names}}


STAGE_WAIST_HOLD_SOURCE = "observation.commanded_upper_joint_positions_before_stage_command"


def latch_stage_waist_hold(stage, applied_targets, servo, time_s):
    """Opt-in upper_body_ik.stage_waist_hold latch (runtime_v26), taken at the first control step of a held stage.

    ``applied_targets`` are the commands applied in the previous control step (the runtime's live targets before
    this stage computes anything; the observation reports the same values as commanded_upper_joint_positions).
    The three waist values are latched as they are, never re-measured or clipped: a value outside the selected
    servo's waist command limits cannot be held exactly and fails closed. The returned evidence is published with
    every command of the stage; ``latched_waist_command_rad`` is the solve_step ``waist_hold``.
    """
    if not isinstance(stage, str) or stage not in UPPER_BODY_IK_STAGES:
        raise ValueError("upper_body_ik.stage_waist_hold needs an upper-body IK stage")
    if not isinstance(applied_targets, Mapping):
        raise ValueError("upper_body_ik.stage_waist_hold needs the previously applied commands")
    latched = {}
    for name in WAIST:
        value = applied_targets.get(name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"upper_body_ik.stage_waist_hold: the applied {name} command at {stage} entry is "
                             "missing or not a finite number")
        value = float(value)
        low, high = servo.limits[name]
        if not low <= value <= high:
            raise ValueError(f"upper_body_ik.stage_waist_hold: the applied {name} command {value:g} at {stage} entry "
                             f"lies outside the stage waist command limits [{low:g}, {high:g}]")
        latched[name] = value
    return {"stage_id": stage, "latched_waist_command_rad": latched,
            "latched_at_time_s": _number(time_s, "upper_body_ik.stage_waist_hold time", low=-math.inf,
                                         inclusive_low=True),
            "source": STAGE_WAIST_HOLD_SOURCE}


class UpperBodyServo:
    """Bounded waist3+arm7 LSQ, with an optional soft posture preference.

    Arm limits, gain 2/s, rotation weighting .2, damping .04, previous-command
    initialization and protective recentering match ArmServo.bounded_lsq.
    Waist command speed is additionally capped at the existing measured
    posture controller's .08 rad/s. Default tracking envelope stays .15 rad.
    Optional arm margins narrow command bounds, never measured kinematics.
    Opt-in ``waist_speed_profile="simulation_fast"`` (runtime_v26) raises the
    bound of ``max_waist_speed`` to 0.4 rad/s; the configured value is the cap.
    """
    def __init__(self, model, *, waist_command_limits, max_joint_speed=.6,
                 max_tracking_error=.15, max_waist_speed=.08, arm_command_margin_rad=0.,
                 arm_command_limits_rad=None, posture_saturation_aware=False, waist_speed_profile=None):
        self.model = model
        self.waist_speed_profile = waist_speed_profile
        waist_speed_bound = waist_speed_limit(waist_speed_profile)
        if type(posture_saturation_aware) is not bool:
            raise ValueError("posture_saturation_aware must be a boolean")
        # Opt-in (runtime_v22): project the posture preference only onto the null space of the joints that are
        # not pinned on a POSITION bound (command limit or tracking envelope; speed bounds do not pin), so a joint
        # held on its command limit cannot turn its share of the null motion into a TCP error (run_81: 9.7 mm
        # held with the waist yaw at 1.47).
        self.posture_saturation_aware = posture_saturation_aware
        self.max_joint_speed = _number(max_joint_speed, "max_joint_speed", high=.6)
        self.max_tracking_error = _number(max_tracking_error, "max_tracking_error", high=.3)
        self.max_waist_speed = _number(max_waist_speed, "max_waist_speed", high=waist_speed_bound)
        self.arm_command_margin_rad = _number(arm_command_margin_rad,
            "arm_command_margin_rad", high=.1, inclusive_low=True)
        self.authored_limits = dict(model.joint_limits)
        self.shared_limits = dict(ArmServo(model).limits)
        self.limits = dict(self.shared_limits)
        if PELVIS not in model.link_names or not set(WAIST) <= set(model.joints):
            raise ValueError("Model must include all three waist joints and measured pelvis link")
        self.waist_command_limits = _waist_limits(waist_command_limits, self.limits)
        self.limits.update({name: tuple(pair) for name, pair in self.waist_command_limits.items()})
        for arm in ("left", "right"):
            for name in model.arm_joint_names(arm):
                low, high = self.shared_limits[name]
                narrowed = (low + self.arm_command_margin_rad, high - self.arm_command_margin_rad)
                if narrowed[0] >= narrowed[1]:
                    raise ValueError("arm_command_margin_rad leaves no command range: " + name)
                self.limits[name] = narrowed
        self.arm_command_limits_rad = {}
        if arm_command_limits_rad is not None:
            self.arm_command_limits_rad = _arm_limits(arm_command_limits_rad, self.limits,
                set(model.arm_joint_names("left") + model.arm_joint_names("right")))
            self.limits.update({name: tuple(pair) for name, pair in self.arm_command_limits_rad.items()})
        self.stage_id = None
        self.stage_command_limits_rad = {}
        self._ancestors = {}
        by_child = {joint.child: joint for joint in model.joints.values()}
        for link in model.link_names:
            ancestors, current = set(), link
            while current != model.root_link:
                joint = by_child[current]
                ancestors.add(joint.name)
                current = joint.parent
            self._ancestors[link] = ancestors

    def names(self, arm):
        return WAIST + self.model.arm_joint_names(arm)

    def measured(self, joints):
        if not isinstance(joints, Mapping):
            raise ValueError("Measured joints must be a mapping")
        result = dict(joints)
        for name, (low, high) in self.model.joint_limits.items():
            value = result.get(name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError("Missing or nonfinite measured joint: " + name)
            if value < low - .05 or value > high + .05:
                raise ValueError("Measured joint outside physical limits: " + name)
            result[name] = float(np.clip(value, low, high))
        return result

    def pose_and_jacobian(self, joint_positions, pelvis_pose_w, arm):
        """World TCP and analytic spatial Jacobian at a fixed measured pelvis.

        With A=FK(pelvis), B=FK(TCP), R=R_pelvis R_A^T and d=p_B-p_A:
        Jv_world = R (Jv_B - Jv_A - Jw_A x d)
        Jw_world = R (Jw_B - Jw_A).
        Waist joints belong to the pelvis branch of a torso-rooted URDF;
        omitting the inverse-pelvis derivative gives the wrong waist Jacobian.
        """
        measured = self.measured(joint_positions)
        transforms = self.model.forward_kinematics(measured)
        pelvis = pose_to_matrix(pelvis_pose_w)
        tcp_link = self.model.default_ee_link(arm)
        a, b = transforms[PELVIS], transforms[tcp_link]
        root_w = pelvis @ np.linalg.inv(a)
        tcp = root_w @ b
        rotation, delta = root_w[:3, :3], b[:3, 3] - a[:3, 3]
        jacobian = np.zeros((6, len(self.names(arm))))

        def column(name, link):
            if name not in self._ancestors[link]:
                return np.zeros(3), np.zeros(3)
            joint = self.model.joints[name]
            origin = transforms[joint.parent] @ joint.origin
            axis = origin[:3, :3] @ joint.axis
            if joint.kind == "prismatic":
                return axis, np.zeros(3)
            return np.cross(axis, transforms[link][:3, 3] - origin[:3, 3]), axis

        for index, name in enumerate(self.names(arm)):
            va, wa = column(name, PELVIS)
            vb, wb = column(name, tcp_link)
            jacobian[:3, index] = rotation @ (vb - va - np.cross(wa, delta))
            jacobian[3:, index] = rotation @ (wb - wa)
        return tcp, jacobian

    def solve_step(self, joint_positions, pelvis_pose_w, target_pose_w, arm, dt,
                   previous_targets=None, *, speed_scale=1., tracking_error_limits=None,
                   waist_posture_target=None, waist_posture_gain=0.,
                   posture_target=None, posture_gain=0., feedforward_twist_world=None,
                   waist_hold=None):
        """Return one bounded ten-joint command, never a posture replay.

        ``posture_target`` is an optional desired *measured* waist3+arm7
        posture within authored physical limits. It can lie outside narrowed
        command bounds; the emitted commands still obey those bounds. Its
        projected velocity is a soft LSQ preference, not a collision constraint
        or a guarantee of zero realized Cartesian interference under bounds.
        The older waist-only preference remains mutually exclusive with it.
        An optional world-frame twist enters the same constrained solve; it
        grants no additional joint speed, position, or tracking authority.
        Opt-in ``waist_hold`` (runtime_v26 stage_waist_hold): the three waist
        commands get zero-width bounds at these values (each inside this
        servo's command limits), so only the arm solves, and a posture
        preference is projected onto the arm columns' null space (the held
        waist has no null motion). The measured tracking envelope still
        applies: a held value farther than it from the measured waist fails
        like any other empty envelope.
        """
        dt = _number(dt, "dt")
        speed_scale = _number(speed_scale, "speed_scale", high=1.)
        waist_gain = _number(waist_posture_gain, "waist_posture_gain", high=2., inclusive_low=True)
        full_gain = _number(posture_gain, "posture_gain", high=2., inclusive_low=True)
        if ((waist_posture_target is not None or waist_gain)
                and (posture_target is not None or full_gain)):
            raise ValueError("Waist-only and ten-joint posture preferences are mutually exclusive")
        gain = full_gain if posture_target is not None else waist_gain
        names = self.names(arm)
        feedforward = np.zeros(6) if feedforward_twist_world is None else np.asarray(feedforward_twist_world, dtype=float)
        if feedforward.shape != (6,) or not np.isfinite(feedforward).all():
            raise ValueError("feedforward_twist_world must be a finite six-vector")
        measured = self.measured(joint_positions)
        tcp, jacobian = self.pose_and_jacobian(measured, pelvis_pose_w, arm)
        target = pose_to_matrix(target_pose_w)
        error = np.r_[target[:3, 3] - tcp[:3, 3], rotation_error(target[:3, :3], tcp[:3, :3])]
        if waist_posture_target is not None:
            if not isinstance(waist_posture_target, Mapping) or set(waist_posture_target) != set(WAIST):
                raise ValueError("waist_posture_target must name exactly the three waist joints")
            for name, value in waist_posture_target.items():
                value = _number(value, name, low=-math.inf, inclusive_low=True)
                if not self.limits[name][0] <= value <= self.limits[name][1]:
                    raise ValueError("Waist posture outside shared limits: " + name)
        elif waist_gain:
            raise ValueError("waist_posture_gain requires waist_posture_target")
        if posture_target is not None:
            if not isinstance(posture_target, Mapping) or set(posture_target) != set(names):
                raise ValueError("posture_target must name exactly the ten owned waist/active-arm joints")
            for name, value in posture_target.items():
                value = _number(value, name + " posture target", low=-math.inf, inclusive_low=True)
                if not self.authored_limits[name][0] <= value <= self.authored_limits[name][1]:
                    raise ValueError("Posture target outside authored physical limits: " + name)
        elif full_gain:
            raise ValueError("posture_gain requires posture_target")
        if tracking_error_limits is not None and not isinstance(tracking_error_limits, Mapping):
            raise ValueError("Tracking limits must be a mapping")
        limits = dict(tracking_error_limits or {})
        if set(limits) - set(names):
            raise ValueError("Tracking limits may name only owned upper-body joints")
        limits = {name: _number(value, name + " tracking limit", high=.3) for name, value in limits.items()}
        q = np.asarray([measured[name] for name in names])
        if previous_targets is not None and (not isinstance(previous_targets, Mapping)
                or not set(names) <= set(previous_targets)):
            raise ValueError("Previous commands must include all ten owned joints")
        old = np.asarray([_number(measured[name] if previous_targets is None else previous_targets[name],
                                 name + " previous command", low=-math.inf, inclusive_low=True)
                          for name in names])
        if self.stage_command_limits_rad:
            for name, (lo, hi) in self.stage_command_limits_rad.items():
                value = old[names.index(name)]
                if value < lo - 1e-9 or value > hi + 1e-9:
                    raise ValueError("Previous command outside stage command limits; "
                                     "align before entering " + self.stage_id + ": " + name)
        hard_low, hard_high = np.asarray([self.limits[name] for name in names]).T
        hold = None
        if waist_hold is not None:
            if not isinstance(waist_hold, Mapping) or set(waist_hold) != set(WAIST):
                raise ValueError("waist_hold must name exactly the three waist joints")
            hold = {}
            for name in WAIST:
                value = waist_hold[name]
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
                    raise ValueError("Held waist command must be a finite number: " + name)
                value = float(value)
                if not self.limits[name][0] <= value <= self.limits[name][1]:
                    raise ValueError("Held waist command outside the command limits: " + name)
                hold[name] = value
                index = names.index(name)
                hard_low[index] = hard_high[index] = value
        tracking = np.asarray([limits.get(name, self.max_tracking_error) for name in names])
        low, high = np.maximum(hard_low, q - tracking), np.minimum(hard_high, q + tracking)
        if np.any(low > high):
            raise ValueError("Measured tracking envelope does not intersect shared command limits")
        position_low, position_high = low, high
        origin = np.clip(old, low, high)
        speed = np.asarray([min(self.max_joint_speed, self.max_waist_speed) if name in WAIST
                            else self.max_joint_speed for name in names]) * speed_scale
        low, high = np.maximum(low, origin - speed * dt), np.minimum(high, origin + speed * dt)
        weights = np.array([1., 1., 1., .2, .2, .2])
        weighted = weights[:, None] * jacobian
        null_velocity = np.zeros(len(names))
        if gain:
            _, singular, right = np.linalg.svd(weighted, full_matrices=True)
            rank = int(np.count_nonzero(singular > max(weighted.shape) * singular[0] * np.finfo(float).eps))
            basis = right[rank:].T
            preferred = np.zeros(len(names))
            if posture_target is not None:
                preferred = gain * (np.asarray([posture_target[n] for n in names]) - q)
            else:
                preferred[:len(WAIST)] = gain * (np.asarray([waist_posture_target[n] for n in WAIST]) - q[:len(WAIST)])
            null_velocity = basis @ (basis.T @ preferred)
            if hold is not None and not self.posture_saturation_aware:
                # Opt-in stage_waist_hold: the held waist has no null motion, so the preference is projected onto the
                # null space of the arm columns alone (posture_saturation_aware pins it the same way from the start).
                held = np.array([name in WAIST for name in names])
                masked = weighted * (~held)[None, :]
                _, masked_singular, masked_right = np.linalg.svd(masked, full_matrices=True)
                masked_rank = (int(np.count_nonzero(masked_singular > max(masked.shape) * masked_singular[0]
                                                    * np.finfo(float).eps)) if masked_singular[0] > 0 else 0)
                masked_basis = masked_right[masked_rank:].T
                null_velocity = masked_basis @ (masked_basis.T @ np.where(held, 0., preferred))
        a = np.vstack((weighted, .04 * np.eye(len(names))))
        lower, upper = (low - origin) / dt, (high - origin) / dt
        free = upper - lower > 1e-12

        def bounded(null):
            rhs = np.r_[weights * (2. * error + feedforward) * speed_scale, .04 * null * speed_scale]
            solution, result_fit = lower.copy(), None
            if np.any(free):
                result_fit = lsq_linear(a[:, free], rhs - a[:, ~free] @ lower[~free],
                    bounds=(lower[free], upper[free]), method="bvls", tol=1e-10, max_iter=100)
                if not result_fit.success or not np.isfinite(result_fit.x).all():
                    raise RuntimeError("Bounded upper-body least-squares solve failed")
                solution[free] = result_fit.x
            return solution, result_fit, rhs

        pinned_joints = []
        if gain and self.posture_saturation_aware:
            # Only position bounds (command limits or the tracking envelope) pin a joint for good; a joint at
            # its speed bound still moves, so its preference is kept.
            position_lower, position_upper = (position_low - origin) / dt, (position_high - origin) / dt

            def at_bound(solution):
                return np.array([(not is_free) or (abs(v - lo) < 1e-8 and abs(lo - plo) < 1e-8)
                                 or (abs(v - hi) < 1e-8 and abs(hi - phi) < 1e-8)
                                 for v, lo, hi, plo, phi, is_free in zip(solution, lower, upper, position_lower,
                                                                          position_upper, free)])

            def masked_null(mask):
                masked = weighted * (~mask)[None, :]
                _, masked_singular, masked_right = np.linalg.svd(masked, full_matrices=True)
                masked_rank = (int(np.count_nonzero(masked_singular > max(masked.shape) * masked_singular[0]
                                                    * np.finfo(float).eps)) if masked_singular[0] > 0 else 0)
                masked_basis = masked_right[masked_rank:].T
                return masked_basis @ (masked_basis.T @ np.where(mask, 0., preferred))

            # Start from the joints with no command range this step (zero-width bounds); every joint the ordinary
            # solve then leaves on a POSITION bound loses its preference and its Jacobian column in the
            # projection, and the solve is repeated. The set only grows, so this ends; a joint resting on a bound
            # whose preference points away from it is never pinned (it moves like in the plain solve).
            pinned = ~free
            candidate = masked_null(pinned) if np.any(pinned) else null_velocity
            for _ in range(len(names)):
                velocity, fit, b = bounded(candidate)
                newly = at_bound(velocity) & ~pinned
                if not np.any(newly):
                    break
                pinned = pinned | newly
                candidate = masked_null(pinned)
            else:
                velocity, fit, b = bounded(candidate)
            null_velocity = candidate
            pinned_joints = [name for name, flag in zip(names, pinned) if flag]
        else:
            velocity, fit, b = bounded(null_velocity)
        commanded = np.clip(origin + velocity * dt, low, high)
        result = dict(zip(names, commanded.astype(float).tolist()))
        active = {name: ("lower" if abs(v - lo) < 1e-8 else "upper")
                  for name, v, lo, hi in zip(names, velocity, lower, upper)
                  if min(abs(v - lo), abs(v - hi)) < 1e-8}
        def margins(values, bounds):
            return {name: float(min(value - bounds[name][0], bounds[name][1] - value))
                    for name, value in zip(names, values)}

        diagnostics = {
            "solver": "pelvis_relative_upper_body_bounded_lsq", "owned_joint_names": list(names),
            "pelvis_pose_w": list(pelvis_pose_w), "model_tcp_pose_w": matrix_to_pose(tcp),
            "position_error_m": float(np.linalg.norm(error[:3])), "rotation_error_rad": float(np.linalg.norm(error[3:])),
            "effective_command_bounds_rad": {n: [float(l), float(h)] for n, l, h in zip(names, low, high)},
            "tracking_envelope_rad": dict(zip(names, tracking.tolist())),
            "waist_command_limits_rad": {name: list(self.limits[name]) for name in WAIST},
            "arm_command_margin_rad": self.arm_command_margin_rad,
            "command_speed_limits_rad_s": dict(zip(names, speed.tolist())),
            "active_bounds": active,
            "measured_joint_margin_rad": margins(q, self.limits),
            "command_joint_margin_rad": margins(commanded, self.limits),
            "measured_authored_joint_margin_rad": margins(q, self.authored_limits),
            "measured_shared_command_margin_rad": margins(q, self.shared_limits),
            "measured_narrowed_command_margin_rad": margins(q, self.limits),
            "command_authored_joint_margin_rad": margins(commanded, self.authored_limits),
            "command_shared_command_margin_rad": margins(commanded, self.shared_limits),
            "command_narrowed_command_margin_rad": margins(commanded, self.limits),
            "weighted_velocity_residual": float(np.linalg.norm(weighted @ velocity - b[:6])),
            "optimality": float(fit.optimality) if fit is not None else 0.,
            "waist_posture_gain_s_inv": waist_gain,
            "posture_gain_s_inv": full_gain,
            "posture_reference_kind": ("waist_and_active_arm" if posture_target is not None
                else "waist_only" if waist_posture_target is not None else "none"),
            "posture_target_joint_positions": (None if posture_target is None else
                {name: float(posture_target[name]) for name in names}),
            "measured_posture_error_rad": (None if posture_target is None else
                {name: float(posture_target[name] - measured[name]) for name in names}),
            "posture_priority": "soft_regularization_in_constrained_cartesian_solve",
            "nullspace_tcp_twist_norm": float(np.linalg.norm(jacobian @ null_velocity)),
            "protective_target_recentering_rad": float(np.max(np.abs(origin - old))),
            "maximum_applied_target_step_rad": float(np.max(np.abs(commanded - old))),
            "leg_commands_emitted": False, "root_pose_written": False,
            "collision_checked": False, "balance_verified": False,
        }
        if self.waist_speed_profile is not None:
            # Declared only: the waist cap in command_speed_limits_rad_s comes from this simulation profile
            # (free-space stages) or stays at the 0.08 rad/s default cap (the contact stages approach/place).
            diagnostics["waist_speed_profile"] = self.waist_speed_profile
            diagnostics["waist_speed_scope"] = getattr(self, "waist_speed_scope", "free_space_stages")
        if self.posture_saturation_aware:
            # Only opted-in profiles record the new fields; undeclared profiles keep identical diagnostics.
            diagnostics["posture_saturation_aware"] = True
            diagnostics["posture_pinned_joints"] = list(pinned_joints)
        if feedforward_twist_world is not None:
            diagnostics["feedforward_twist_world"] = feedforward.tolist()
        if hold is not None:
            # Opt-in stage_waist_hold only: the zero-width waist bounds of this solve.
            diagnostics["stage_waist_hold_rad"] = dict(hold)
        if self.arm_command_limits_rad:
            diagnostics["arm_command_limits_rad"] = {
                name: list(pair) for name, pair in self.arm_command_limits_rad.items() if name in names}
        if self.stage_command_limits_rad:
            diagnostics["stage_command_limits_rad"] = {
                name: list(pair) for name, pair in self.stage_command_limits_rad.items()}
            diagnostics["command_limits_stage_id"] = self.stage_id
        return result, diagnostics


# --- NEPES upper-body IK backend (Pink release) ---
# Everything above this marker is the unchanged native solver. The Pink worker
# executes exactly that part in its isolated Python; see upper_body_ik_backend.py.
# NEPES_IK_BACKEND selects pink (package default) or native for this process.
from .upper_body_ik_backend import select_upper_body_servo as _select_upper_body_servo  # noqa: E402

NativeUpperBodyServo = UpperBodyServo
UpperBodyServo = _select_upper_body_servo(NativeUpperBodyServo, __file__)
