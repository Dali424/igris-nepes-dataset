"""Segment-local posture preferences for the single upper-body IK controller.

These values are soft kinematic preferences, never independently applied joint
commands. Collision clearance and physical grasp retention require separate
verification at the actual pose.

Each guided stage names waist3 and the arm7 the solver owns in that stage
(``guidance_arm``): ``task["arm"]``, except that with the opt-in handover block
the stages after the handover (transfer, place) run on the receiving arm.
"""
from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Real

import numpy as np

from .upper_body_control import WAIST

CONSTANT_GUIDANCE_STAGES = frozenset(("pregrasp", "approach"))
# Opt-in (runtime_v26): a constant stage may instead declare {"from_entry": true}; its preference is then the
# measured waist3+arm7 latched at the stage's first control step (UpperBodyPostureGuide.latch_entry_posture).
FROM_ENTRY_SOURCE = "observation.joint_positions_at_stage_entry"


def joint_names(arm):
    if arm not in ("left", "right"):
        raise ValueError("Upper-body guidance requires a left or right arm")
    return WAIST + tuple("Joint_" + name + "_" + arm.title() for name in (
        "Shoulder_Pitch", "Shoulder_Roll", "Shoulder_Yaw", "Elbow_Pitch",
        "Wrist_Yaw", "Wrist_Roll", "Wrist_Pitch"))


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(name + " must be finite numeric")
    return float(value)


def guidance_arm(task, stage):
    """The arm whose seven joints the single solver owns with the waist in ``stage``.

    Without ``task["handover"]`` this is ``task["arm"]`` in every stage. With the
    opt-in block the machine's command arm becomes ``handover.receiving_arm`` at
    handover_pregrasp (handover.observation_arm_for_stage), so transfer and place
    run on the receiving (holding) arm while lift keeps the giving task arm.
    """
    if "handover" not in task:
        return task.get("arm")
    settings = task["handover"]
    receiving = settings.get("receiving_arm") if isinstance(settings, Mapping) else None
    if task.get("arm") not in ("left", "right") or receiving not in ("left", "right") or receiving == task["arm"]:
        raise ValueError("handover.receiving_arm must be the arm opposite the task arm")
    from .handover import observation_arm_for_stage
    return observation_arm_for_stage(task, stage)


def validate_upper_body_guidance(task):
    if not isinstance(task, Mapping):
        raise ValueError("Upper-body guidance requires a task mapping")
    if "upper_body_posture_guidance" not in task:
        return None
    settings = task["upper_body_posture_guidance"]
    if not isinstance(settings, Mapping) or set(settings) != {"gain_s_inv", "stages"}:
        raise ValueError("upper_body_posture_guidance requires gain_s_inv and stages")
    gain = _number(settings["gain_s_inv"], "upper-body posture gain")
    if not 0 < gain <= 2:
        raise ValueError("Upper-body posture gain must be in (0, 2]")
    stages = settings["stages"]
    controller = task.get("upper_body_ik", {})
    if not isinstance(controller, Mapping):
        raise ValueError("Upper-body guidance requires a controller mapping")
    owned = controller.get("stages", [])
    if not isinstance(owned, list) or any(not isinstance(s, str) for s in owned):
        raise ValueError("Upper-body guidance requires owned stage names")
    if (not isinstance(stages, Mapping) or not stages
            or not set(stages) <= CONSTANT_GUIDANCE_STAGES | {"lift", "transfer", "place"}
            or not set(stages) <= set(owned)
            or (set(stages) - CONSTANT_GUIDANCE_STAGES and "object_pose_control" not in task)):
        raise ValueError("Upper-body guidance requires owned stages and object control for lift/transfer/place")
    arms = {stage: guidance_arm(task, stage) for stage in stages}
    names = {stage: joint_names(arm) for stage, arm in arms.items()}
    paths = task.get("object_waypoint_offsets_w", {})
    if not isinstance(paths, Mapping):
        raise ValueError("Upper-body guidance object waypoints must be a mapping")

    def pose(values, stage):
        if not isinstance(values, Mapping) or set(values) != set(names[stage]):
            raise ValueError("Upper-body guidance must name exactly waist3 and the active arm7" + (
                "" if arms[stage] == task["arm"] else f" ({stage} runs on the receiving {arms[stage]} arm)"))
        return {n: _number(values[n], n) for n in names[stage]}

    result = {"gain_s_inv": gain, "stages": {}}
    for stage, path in stages.items():
        if isinstance(path, Mapping) and "from_entry" in path:
            # Opt-in (runtime_v26): the constant preference is the measured posture latched at stage entry.
            name = f"upper_body_posture_guidance.stages.{stage}.from_entry"
            if stage not in CONSTANT_GUIDANCE_STAGES:
                raise ValueError(name + " is only valid for the constant pregrasp/approach stages")
            if set(path) != {"from_entry"} or path["from_entry"] is not True:
                extra = sorted(map(str, set(path) - {"from_entry"}))
                raise ValueError(name + " must be true and the stage's only key (it replaces "
                                 "entry_joint_positions/waypoint_joint_positions)"
                                 + (" (unknown: " + ", ".join(extra) + ")" if extra else ""))
            result["stages"][stage] = {"from_entry": True}
            continue
        if not isinstance(path, Mapping) or set(path) != {"entry_joint_positions", "waypoint_joint_positions"}:
            raise ValueError("Upper-body guidance needs an entry and per-waypoint postures")
        if stage in CONSTANT_GUIDANCE_STAGES:
            # These world-TCP stages do not have an object-segment clock. A
            # constant preference avoids wall-time drift or waypoint resets;
            # the existing bounded solver remains the only joint writer.
            waypoints = path["waypoint_joint_positions"]
            if not isinstance(waypoints, list) or len(waypoints) != 1:
                raise ValueError("Pregrasp/approach guidance requires exactly one constant posture")
            entry, goal = pose(path["entry_joint_positions"], stage), pose(waypoints[0], stage)
            if entry != goal:
                raise ValueError("Pregrasp/approach guidance entry must equal its single waypoint")
            from .upper_body_control import validate_upper_body_ik
            from .constants import TELEOPERATION_JOINT_NAMES, TELEOPERATION_RANGES
            upper = validate_upper_body_ik(controller)
            margin = upper.get("arm_command_margin_rad", 0.)
            known = {n: (lo + margin, hi - margin)
                     for n, (lo, hi) in zip(TELEOPERATION_JOINT_NAMES, TELEOPERATION_RANGES)
                     if n in names[stage][3:]}
            known.update(upper["waist_command_limits_rad"])
            known.update(upper.get("arm_command_limits_rad", {}))
            for name, value in entry.items():
                if name in known and not known[name][0] <= value <= known[name][1]:
                    raise ValueError("Constant upper-body preference outside effective command limits: " + name)
            result["stages"][stage] = {"entry_joint_positions": entry,
                "waypoint_joint_positions": [goal]}
            continue
        offsets = paths.get(stage)
        if stage in paths and (not isinstance(offsets, list) or not offsets):
            raise ValueError("Upper-body guidance object offsets must be a nonempty list")
        count = len(offsets) if offsets is not None else 1
        waypoints = path["waypoint_joint_positions"]
        if not isinstance(waypoints, list) or len(waypoints) != count or not 1 <= count <= 128:
            raise ValueError("Upper-body guidance must match each object waypoint (place has one)")
        result["stages"][stage] = {"entry_joint_positions": pose(path["entry_joint_positions"], stage),
            "waypoint_joint_positions": [pose(q, stage) for q in waypoints]}
    return result


class UpperBodyPostureGuide:
    def __init__(self, task, model, *, command_limits=None):
        self.settings = validate_upper_body_guidance(task)
        if self.settings is None:
            raise ValueError("An explicit upper-body posture guide is required")
        self.names = joint_names(task["arm"])
        if tuple(model.arm_joint_names(task["arm"])) != self.names[3:]:
            raise ValueError("Guidance joint order differs from the loaded robot")
        self.stage_arms = {stage: guidance_arm(task, stage) for stage in self.settings["stages"]}
        self.stage_names = {stage: joint_names(arm) for stage, arm in self.stage_arms.items()}
        for arm in set(self.stage_arms.values()) - {task["arm"]}:
            if tuple(model.arm_joint_names(arm)) != joint_names(arm)[3:]:
                raise ValueError("Guidance joint order differs from the loaded robot")
        if set(self.settings["stages"]) & CONSTANT_GUIDANCE_STAGES and command_limits is None:
            # Standalone construction must enforce the same model-specific
            # bounds as runtime, including narrowed wrists and arm margins.
            from .upper_body_control import UpperBodyServo
            upper = task["upper_body_ik"]
            command_limits = UpperBodyServo(model,
                waist_command_limits=upper["waist_command_limits_rad"],
                max_waist_speed=upper["max_waist_speed_rad_s"],
                arm_command_margin_rad=upper.get("arm_command_margin_rad", 0.),
                arm_command_limits_rad=upper.get("arm_command_limits_rad"),
                waist_speed_profile=upper.get("waist_speed_profile")).limits
        self._authored_limits = dict(model.joint_limits)
        self._entry_postures = {}
        for stage, path in self.settings["stages"].items():
            if path.get("from_entry") is True:
                continue  # latched at runtime (latch_entry_posture) and checked there
            for pose in [path["entry_joint_positions"], *path["waypoint_joint_positions"]]:
                for name, value in pose.items():
                    low, high = model.joint_limits[name]
                    if not low <= value <= high:
                        raise ValueError("Upper-body posture preference outside authored limits: " + name)
                    if stage in CONSTANT_GUIDANCE_STAGES:
                        low, high = command_limits[name]
                        if not low <= value <= high:
                            raise ValueError("Constant upper-body preference outside effective command limits: " + name)

    def from_entry(self, stage):
        """Whether ``stage`` declares the opt-in entry-latched constant preference (runtime_v26)."""
        return self.settings["stages"].get(stage, {}).get("from_entry") is True

    def entry_posture(self, stage):
        """The latch evidence of an entry-latched stage (a copy; the preference itself is ``sample``), or None
        before the stage's first step latched it."""
        latched = self._entry_postures.get(stage)
        return None if latched is None else deepcopy(latched["evidence"])

    def reset_entry_postures(self):
        """Forget every entry latch; the runtime calls this on each stage change (no effect without from_entry)."""
        self._entry_postures = {}

    def latch_entry_posture(self, stage, measured_joint_positions, command_limits, time_s):
        """Opt-in from_entry (runtime_v26): latch the measured waist3+arm7 of the stage's first control step.

        Each measured value must be finite and within 0.05 rad of the authored limits (the servo's measured-joint
        rule); the preference is that value clipped into ``command_limits`` (the stage servo's effective command
        limits, inside the authored ones), so it never pulls a joint against its command bound. Clipped joints are
        named in the evidence. A stage is latched once until reset_entry_postures.
        """
        if not self.from_entry(stage):
            raise ValueError("No entry-latched upper-body posture preference for requested stage")
        if stage in self._entry_postures:
            raise ValueError("The entry posture of this stage is already latched")
        if not isinstance(measured_joint_positions, Mapping) or not isinstance(command_limits, Mapping):
            raise ValueError("Entry posture latch needs measured joints and command limits")
        time_s = _number(time_s, "entry posture time")
        preference, clipped = {}, {}
        for name in self.stage_names[stage]:
            value = measured_joint_positions.get(name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError("Missing or nonfinite measured joint for the entry posture: " + name)
            value = float(value)
            low, high = self._authored_limits[name]
            if value < low - .05 or value > high + .05:
                raise ValueError("Measured joint outside physical limits for the entry posture: " + name)
            low, high = (float(bound) for bound in command_limits[name])
            preference[name] = min(max(value, low), high)
            if preference[name] != value:
                clipped[name] = value
        self._entry_postures[stage] = {"joint_positions": preference, "evidence": {
            "stage_id": stage, "latched_at_time_s": time_s, "source": FROM_ENTRY_SOURCE,
            "clipped_joint_names": list(clipped), "clipped_measured_rad": clipped}}
        return self.entry_posture(stage)

    def sample(self, stage, waypoint_index, progress):
        if not isinstance(stage, str) or stage not in self.settings["stages"]:
            raise ValueError("No upper-body posture guide for requested stage")
        path = self.settings["stages"][stage]
        if path.get("from_entry") is True:
            # Opt-in from_entry: the one constant posture is the latched measured entry posture.
            if type(waypoint_index) is not int or waypoint_index != 0:
                raise ValueError("Upper-body posture waypoint index is invalid")
            progress = _number(progress, "segment progress")
            if not 0 <= progress <= 1:
                raise ValueError("Upper-body posture progress must be in [0, 1]")
            if stage not in self._entry_postures:
                raise ValueError("The entry posture of this stage has not been latched")
            return dict(self._entry_postures[stage]["joint_positions"])
        goals = path["waypoint_joint_positions"]
        if type(waypoint_index) is not int or not 0 <= waypoint_index < len(goals):
            raise ValueError("Upper-body posture waypoint index is invalid")
        progress = _number(progress, "segment progress")
        if not 0 <= progress <= 1:
            raise ValueError("Upper-body posture progress must be in [0, 1]")
        if stage in CONSTANT_GUIDANCE_STAGES:
            return dict(goals[0])
        start = path["entry_joint_positions"] if waypoint_index == 0 else goals[waypoint_index - 1]
        goal = goals[waypoint_index]
        return {n: (1. - progress) * start[n] + progress * goal[n] for n in self.stage_names[stage]}

    def sample_blended(self, stage, waypoint_index, progress, next_progress):
        """Preference along a blended pass-through reference (runtime_v26, reference_continuity).

        The reference finishes segment ``waypoint_index`` (``progress``) while it already moves along the next
        segment (``next_progress``), so the preference superposes the same two interpolations:
        q = sample(index, progress) + next_progress * (goal[index + 1] - goal[index]). It equals
        ``sample(index + 1, next_progress)`` once the first segment is complete, so the preference is continuous
        when the reference promotes the next segment.
        """
        preferred = self.sample(stage, waypoint_index, progress)
        if stage in CONSTANT_GUIDANCE_STAGES:
            raise ValueError("Blended upper-body guidance needs an object-segment stage")
        goals = self.settings["stages"][stage]["waypoint_joint_positions"]
        if waypoint_index + 1 >= len(goals):
            raise ValueError("Blended upper-body guidance needs a next object waypoint")
        next_progress = _number(next_progress, "next segment progress")
        if not 0 <= next_progress <= 1:
            raise ValueError("Upper-body posture next progress must be in [0, 1]")
        start, goal = goals[waypoint_index], goals[waypoint_index + 1]
        return {n: preferred[n] + next_progress * (goal[n] - start[n]) for n in self.stage_names[stage]}
