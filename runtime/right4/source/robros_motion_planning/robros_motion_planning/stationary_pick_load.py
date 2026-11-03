"""Catalog-v2 task 1: recorded CRS pickup from an upright start and released JIG loading.

Only the unrecorded preparation (``prerequisites``, ``navigate_source``) precedes
the demonstration: it proves a quiet upright stance at the source station with
the commanded upright joints actually reached and both hands free of the CRS.
The upper command of the first step (the declared upright start) must stay
unchanged until recording, and the CRS must stay at its first measured pose
until the jaws close on it, so neither a pre-posed body nor a shoved ring can
pass as the canonical cart pickup.
Everything from the first joint-space reach onward is recorded: physical
acquisition, the slot-clearing lift, the carry posture, the side-step, the
outside-JIG alignment, the insertion under the handle bar, supported placement,
release, retreat and the rest posture. Placement acceptance reuses the
validated task-2 measurements unchanged. The machine only emits commands; it
never writes robot or object state and never creates an attachment.
"""
from collections.abc import Mapping
from copy import deepcopy
import math

import numpy as np

from .nepes_tasks import (HANDOVER_ROTATE_STAGE, NepesTaskMachine, PICKUP_LOAD_STAGES, POSTURE_STAGES, STAGES,
                          UPRIGHT_PICKUP_SEQUENCE, _errors, _shift, handover_stages, pick_gate_conditions, task_stages,
                          TRANSFER_TRANSIT_DEVIATION_DEFAULT_M)
from .handover import (HANDOVER_CARTESIAN_STAGES, HANDOVER_JOINT_PATH_STAGES, HANDOVER_PHASES, HANDOVER_SCHEMA,
                       HANDOVER_STAGE_SET, HAND_TCP_DISTANCE_MEASUREMENT, RELEASE_VERIFYING_PHASE,
                       hand_commands_for_stage, handover_contract, rotation_measurement, validate_handover)
from .recording_boundary import upright_readiness_limits
from .payload_transport import evaluate_payload_transport, validate_transport_constraints
from .waypoint_postures import stage_posture
from .posture_hold import HOLD_MODE, hold_anchor, is_applied_hold
from .single_step import ACCEPTED as SINGLE_STEP_ACCEPTED, OUTSIDE_WINDOW as SINGLE_STEP_OUTSIDE_WINDOW
from .single_step import REJECTED as SINGLE_STEP_REJECTED, SINGLE_STEP_STAGE, SingleStep, arrival_limits
from .single_step import single_step_contract, validate_single_step
from .snapshot import matrix_to_pose, pose_to_matrix
from .stationary_lift import _bilateral_contact
from .stationary_place import PLACEMENT_CONTRACT_VERSION, StationaryPlaceMachine, _contact_absent

GOAL_ID = "crs_pickup_load_jig"
LEGACY_TASK_ID = "crs_load_jig"
RECORDING_START_STAGE = "reach_posture"
PREPARATION_STAGES = ("prerequisites", "navigate_source")
RECORDED_STAGES = PICKUP_LOAD_STAGES[len(PREPARATION_STAGES):]
# Dataset phase names (frame phase_name); phase ids stay nepes_tasks.STAGES.index(stage).
STATIONARY_PHASES = {
    "prerequisites": "preparation", "navigate_source": "upright_ready", "reach_posture": "reach",
    "pregrasp": "pregrasp", "approach": "approach", "close": "grasp", "lift": "lift",
    "carry_posture": "carry", "navigate_target": "side_step", "place_posture": "align_outside_jig",
    "transfer": "insert_under_handle", "place": "lower_into_slot", "release": "release",
    "retreat": "retreat", "rest_posture": "rest", "settle": "settle",
}
RECORDED_PHASES = tuple(STATIONARY_PHASES[stage] for stage in RECORDED_STAGES)
# Labels of every stage the machine can run; the opt-in handover stages (handover.py) are recorded too.
STAGE_PHASES = {**STATIONARY_PHASES, **HANDOVER_PHASES}
# Gates of the unrecorded upright readiness; each must hold for the preparation
# dwell (the longer of preparation_dwell_s and the recording readiness dwell).
READINESS_GATES = ("station_reached", "upright_posture_reached", "upright_command_retained", "pelvis_upright",
                   "hand_open", "same_object_contacts_absent", "sensors_available", "object_quiet", "base_quiet",
                   "source_object_retained")
# Exact published key sets, for admission to pin.
READINESS_KEYS = frozenset((
    "valid", *READINESS_GATES, "upright_reference", "upright_joint_names", "max_upright_joint_error_rad",
    "max_upright_error_joint", "upright_command_targets", "max_upright_command_change_rad", "pelvis_tilt_rad",
    "hand_contact_force_n", "base_station_position_error_m", "base_station_yaw_error_rad",
    "source_object_anchor_pose_w", "source_object_position_error_m", "source_object_rotation_error_rad",
    "object_com_linear_speed_m_s", "object_com_angular_speed_rad_s",
    "object_pose_linear_speed_m_s", "object_pose_angular_speed_rad_s", "base_linear_speed_m_s",
    "base_yaw_speed_rad_s", "required_dwell_s", "continuous_dwell_s", "measurement_time_s", "limits"))
PICKUP_CONTRACT_KEYS = frozenset(("capture_time_s", "lift_m", "minimum_lift_m"))
# The CRS must keep its first measured (initialized cart-slot) pose until the jaws
# close on it; the bounds are upright_readiness_limits' source-object limits.
SOURCE_ANCHORED_STAGES = ("prerequisites", "navigate_source", "reach_posture", "pregrasp", "approach")
# The posture-joint command may not change at all before recording (no pre-posing).
UPRIGHT_COMMAND_TOLERANCE_RAD = 1e-9
_CARRYING = frozenset(("lift", "carry_posture", "navigate_target", "place_posture", "transfer", "place",
                       # opt-in handover: the machine's observation arm holds the ring in these stages
                       "handover_posture", "handover_release", "handover_retreat"))
# Handover stages observed through the RECEIVING hand while the GIVING hand still holds the ring:
# the giving pinch and capture retention are checked from the raw evidence of the other hand.
_GIVING_HELD = frozenset(("handover_pregrasp", "handover_approach", "handover_close"))
_RELEASED = frozenset(("retreat", "rest_posture", "settle"))
_SUCCESS_CONDITIONS = {
    "prerequisites": "none: an independent episode; the catalog forbids prerequisites for this goal",
    "navigate_source": ("unrecorded: torso at the upright source station (settled arrival), every posture joint "
                        "within upright_joint_tolerance_rad of its commanded upright target, the posture-joint "
                        "command unchanged since the upright start, pelvis tilt bound, task hand open, no hand "
                        "contact on the CRS, sensors available, the CRS at its first measured pose, object and base "
                        "quiet for the preparation dwell (the longer of preparation_dwell_s and the recording "
                        "readiness dwell)"),
    "reach_posture": ("recording starts: joint-space reach path complete, posture joints within the arrival "
                      "tolerance and nearly still, base quiet; any hand contact on the CRS fails, and the CRS must "
                      "keep its first measured pose until the jaws close"),
    "pregrasp": "measured TCP at clearance pose and measured hand open; the CRS keeps its first measured pose",
    "approach": "measured TCP at grasp pose and measured hand open; the CRS keeps its first measured pose",
    "close": "same-object bilateral pinch, settled jaws, CRS contact and object-relative acquisition for the contact dwell",
    "lift": ("measured CRS reaches each vertical lift waypoint with the original capture retained; the final "
             "waypoint requires the stationary minimum lift that clears the cart POM slots"),
    "carry_posture": "joint-space carry path complete with measured posture arrival and the physical grasp retained",
    "navigate_target": "measured torso settled at the target station (side-step) with the physical grasp retained",
    "place_posture": ("joint-space path to the outside-JIG alignment complete with measured posture arrival and the "
                      "physical grasp retained"),
    "transfer": ("measured CRS reaches each slot-relative object waypoint through the +X gate under the handle bar, "
                 "ending at the validated pre-place height, with the original capture retained"),
    "place": "measured JIG-relative slot pose, actual JIG support and retained capture, stationary for the support dwell",
    "release": "measured jaw opening followed by bounded clearance/separation, actual JIG support and no hand contact",
    "retreat": ("measured TCP reaches each retreat waypoint (the first also clears the released CRS by the minimum "
                "retreat distance) with the hand open, no contact and the placement retained"),
    "rest_posture": "joint-space rest path complete with the hand open, no CRS contact and the placement retained",
    "settle": ("JIG-relative placement, JIG support, open/contact-free hand, measured retreat clearance and "
               "object/base stability for settle_dwell_s"),
    # Opt-in handover (task["handover"]): the pick arm gives the ring to the other arm in place.
    "handover_posture": ("joint-space two-arm pre-handover path complete with measured posture arrival and "
                         "stillness, base quiet and the giving hand's physical grasp retained"),
    "handover_pregrasp": ("measured receiving TCP at the clearance pose above the receiving grasp of the MEASURED "
                          "ring, receiving hand open and free of the ring, giving pinch and capture retained, "
                          "hand TCP distance above the kinematic bound"),
    "handover_approach": ("measured receiving TCP at the receiving grasp pose of the measured ring, receiving hand "
                          "open, giving pinch and capture retained"),
    "handover_close": ("receiving same-object bilateral pinch with settled jaws, receiving grasp within the "
                       "acquisition bounds of the designed receiving grasp, giving pinch retained, for the contact "
                       "dwell; the measured receiving grasp becomes the retention reference"),
    "handover_rotate": ("opt-in handover.rotation_path: joint-space two-arm rotation path complete with measured "
                        "posture arrival and stillness, base quiet, both hands' bilateral pinch and capture retention "
                        "(each against its own grasp reference), ring tilt within max_tilt_rad, and the receiving "
                        "grasp point within max_release_offset_deg of the top of the rim"),
    "handover_release": ("giving jaw measured open, giving hand separated along release_arm_offset_w and free of "
                         "the ring, receiving pinch (bilateral contact and capture retention) held for dwell_s; with "
                         "pre_release_dwell_s the giving jaw first stays closed while the receiving pinch is "
                         "re-verified for that long"),
    "handover_retreat": ("measured giving TCP reaches each release_retreat waypoint with the giving hand open and "
                         "contact-free and the receiving grasp retained"),
}
_FAILURES = ("stage_timeout", "task_timeout", "invalid_observation", "observation_gap", "target_unreachable",
             "runtime_fault", "jig_displaced", "source_object_displaced", "upright_command_changed",
             "grasp_lost", "reach_posture:unexpected_object_contact",
             "reach_posture:contact_sensor_unavailable", "posture_tracking_error", "posture_reference_invalid",
             "posture_reference_mismatch", "contact_sensor_unavailable", "placement_displaced_during_release",
             "release_lost", "placement_displaced_after_release", "diagnostic_stop_after",
             "unsafe_payload_transport", "stationary_pelvis_drift", "no_walk_stance_outside_source_station",
             "handover:contact_sensor_unavailable", "handover_hand_clearance",
             "handover_pregrasp:unexpected_object_contact", "handover_release_lost",
             "handover_rotate_grasp_lost", "handover_rotate_ring_tilt", "handover_rotate_receiving_off_top",
             "posture_tracking_error:handover_rotate")


# runtime_v26 C3: the place success-condition suffix of a declared placement_support.filter_s.
PLACE_FILTER_CONDITION = ("; placement_support.filter_s: the support of the place dwell, the placement hold and the "
                          "descent is the causal filter_s boxcar mean of the measured JIG normal force (zero before "
                          "the place entry) against the unchanged minimum_load_fraction, with raw samples at or above "
                          "it covering 30 % of the window, and the opening dwell also needs every raw sample of the "
                          "window at or above 25 % of it (continuous seat contact); the raw jig_support_confirmed "
                          "gate and every release/settle gate stay raw")


def _joint_values(values, names):
    """Finite measured/commanded values for every name, else None: absent evidence never passes."""
    if not isinstance(values, Mapping):
        return None
    result = {}
    for name in names:
        value = values.get(name)
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
                or not math.isfinite(value)):
            return None
        result[name] = float(value)
    return result


def _pelvis_tilt(value):
    """Lower-body tilt from the measured pelvis (Link_Waist_Pitch) z axis, as robot_stability measures it."""
    try:
        pelvis = pose_to_matrix(value)
    except (ValueError, TypeError):
        return None
    return float(math.acos(np.clip(pelvis[2, 2], -1., 1.)))


def _contact_force(raw, arm, object_path):
    """Summed same-object hand force under _contact_absent's evidence rules; None if unavailable."""
    item = raw.get("contact_evidence", {}).get(arm, {}).get(object_path, {})
    try:
        force = np.asarray(item["forces_w"], dtype=float)
        paths = item["sensor_paths"]
        valid = (item.get("available") is True and item.get("arm") == arm
                 and item.get("object_path") == object_path
                 and item.get("source") == "physx_filtered_hand_object_contacts"
                 and force.ndim == 3 and force.shape[0] == len(paths) and len(paths) > 0
                 and force.shape[1] > 0 and force.shape[2] == 3 and np.isfinite(force).all())
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    return float(np.linalg.norm(force, axis=-1).sum()) if valid else None


class StationaryPickLoadMachine(StationaryPlaceMachine):
    """Upright readiness, joint-space postures, physical pickup, side-step and released JIG loading.

    ``task_id`` stays the legacy ``crs_load_jig`` for calibration lookups; the
    dataset identity is ``crs_pickup_load_jig``. The held-start placement
    settings of task 2 are never applied: the catalog-v2 spec is validated
    against this whole profile, and the placement tail reuses the validated
    StationaryPlaceMachine measurements (``_measure``, ``_support``,
    ``_retreat_goal``, bilateral ``_read_observation`` / ``_grasp_valid``).
    """

    def __init__(self, config, legacy_task_id=LEGACY_TASK_ID, *, seed=0, completed_tasks=(), goal_settings=None):
        if legacy_task_id != LEGACY_TASK_ID:
            raise ValueError("StationaryPickLoadMachine implements only the upright CRS pickup and JIG loading")
        NepesTaskMachine.__init__(self, config, legacy_task_id, seed=seed, completed_tasks=completed_tasks)
        if self.task.get("stage_sequence") != UPRIGHT_PICKUP_SEQUENCE:
            raise ValueError("StationaryPickLoadMachine requires the upright pickup/load stage sequence")
        from .stationary_catalog import validate_stationary_spec
        spec = self.config.get("stationary_task") if goal_settings is None else goal_settings
        if not isinstance(spec, Mapping):
            raise ValueError("stationary_task must be a mapping")
        # Catalog validation pins the v2 identity, the no-relaxation bounds,
        # the lift/retreat fit and the upright-stance recording boundary.
        self.goal_settings = validate_stationary_spec({**self.config, "stationary_task": deepcopy(dict(spec))})
        if self.goal_settings["goal"] != "pickup_load_jig":
            raise ValueError("StationaryPickLoadMachine requires the catalog-v2 pickup_load_jig goal")
        for key in ("required_object_relations", "support_settling_hold_s"):
            if self.task.get(key):
                raise ValueError(f"The upright pickup/load machine does not evaluate {key}")
        # Opt-in in-place handover (task["handover"], handover.py): six stages after carry_posture.
        self._handover = validate_handover(self.task, self.config["thresholds"]) if "handover" in self.task else None
        # With handover.rotation_path the two-hand rotation stage sits between handover_close and handover_release.
        self.stages = PICKUP_LOAD_STAGES if self._handover is None else task_stages(self.task)
        self.stage_id, self._stage_index = self.stages[0], 0
        self._rotation = None if self._handover is None or "rotation_path" not in self._handover else self._handover
        self._rotate_entry_ring = self._rotate_receiving_capture = None
        self._rotation_completed = None
        self._rotate_off_top_since = self._lost_rotate_grasp_since = None
        self._handover_verify_since = None
        self._giving_arm = self.task["arm"]
        self._receiving_arm = None if self._handover is None else self._handover["receiving_arm"]
        self._holding_arm = self._giving_arm
        self._receiving_grasp = None if self._handover is None else pose_to_matrix(self._handover["grasp_pose_object"])
        self._giving_object_to_tcp = self._receiving_object_to_tcp = None
        self._giving_hold_in_base = self._receiving_hold_in_base = None
        self._secondary_target = self._giving_release_origin = None
        self._handover_release_phase = None
        self._handover_retreat_index = 0
        self._handover_capture_time_s = self._handover_switch_time_s = None
        self._handover_events = []
        self._handover_evidence = None
        self._lost_giving_grasp_since = None
        self.jig_path = self.config["objects"]["jig"]
        self._slot_pose = pose_to_matrix(self.task["target"]["pose"])
        self._slot_pose[:3, 3] += self._jitter
        self._jig_anchor = self._release_origin = self._preparation_anchor = None
        self._placement_alignment_settings = self.task.get("placement_alignment")
        self._placement_jig_reference = None
        self._placement_hold = None
        if self._placement_alignment_settings and "support_hold_dropout_grace_s" in self._placement_alignment_settings:
            from .placement_hold import PlacementHold
            # Opt-in (runtime_v26 C2): latch only after placement_support.latch_dwell_s of continuous support.
            latch = self.task["placement_support"].get("latch_dwell_s")
            self._placement_hold = (
                PlacementHold(self._placement_alignment_settings["support_hold_dropout_grace_s"]) if latch is None
                else PlacementHold(self._placement_alignment_settings["support_hold_dropout_grace_s"], latch_dwell_s=latch))
        self._support_latch_pending = False
        # Opt-in (runtime_v26 C3): the place-stage support decision uses a causal filtered JIG normal force.
        self._support_filter = None
        if "filter_s" in self.task["placement_support"]:
            from .placement_support_filter import PlacementSupportFilter
            self._support_filter = PlacementSupportFilter(self.task["placement_support"]["filter_s"])
        # Opt-in lateral alignment gating of the transfer exit and the nominal descent (place_aligned_descent.py).
        self._aligned_descent = None
        if "place_aligned_descent" in self.task:
            from .place_aligned_descent import PlaceAlignedDescent
            self._aligned_descent = PlaceAlignedDescent(self.task["place_aligned_descent"], self.goal_settings)
        self._object_advance_hold_reason = None
        # Opt-in transit policy for intermediate transfer waypoints of a pivoting symmetric payload (nepes_tasks);
        # the hand-path mode suppresses the object feedback through this command field.
        self._object_feedback_hold_reason = None
        self._support_hold_active = False
        self._preparation_anchor_time_s = None
        self._capture_time_s = None
        self._release_phase = "opening"
        self._release_subphase_events = []
        self._clearance_completion = None
        self._readiness = None
        self._metrics = None
        self._raw = None
        self._stage_times = {}
        self._recording_ready = False
        self._max_lift_m = None
        self._posture_evidence = {}
        self._posture_hold_references = {}
        # Previous (time, measured posture joints) per stage for the position-derived settle speed.
        self._posture_settle_samples = {}
        self._transport_constraints = (validate_transport_constraints(self.task["transport_constraints"])
                                       if "transport_constraints" in self.task else None)
        self._transport_evidence = None
        self._walk_entry_transport_evidence = None
        self._stationary_pelvis_settings = deepcopy(self.task.get("navigation_control", {}).get("stationary_pelvis_hold"))
        self._stationary_pelvis_anchor = None
        self._stationary_pelvis_anchor_time_s = None
        self._stationary_pelvis_evidence = None
        feet_settings = self.task.get("navigation_control", {}).get("stationary_feet_hold")
        self._stationary_feet = None
        self._stationary_feet_evidence = None
        if feet_settings is not None:
            from .stationary_feet import StationaryFeetHold
            self._stationary_feet = StationaryFeetHold(feet_settings)
        # Opt-in single stance (navigation_control.no_walk): measured pelvis drift report.
        self._no_walk = None
        if "no_walk" in self.task.get("navigation_control", {}):
            from .no_walk_stance import NoWalkStance, validate_no_walk
            self._no_walk = NoWalkStance(validate_no_walk(self.task))
        # Opt-in single short loaded step in navigate_target (navigation_control.single_step; single_step.py).
        single_step = validate_single_step(self.task)
        self._single_step = None if single_step is None else SingleStep(single_step, **arrival_limits(self.task))
        # Source-only stance check: the later loaded step stays enabled. Reuse
        # no-walk's settled geometric station miss logic without a no-walk mode.
        self._source_hold_station = None
        if single_step is not None and single_step.get("source_hold") is True:
            from .no_walk_stance import NoWalkStance
            source_pose = pose_to_matrix(self.task["base_stations"]["source"])
            self._source_hold_station = NoWalkStance({"planned_pelvis_xy_yaw": [
                float(source_pose[0, 3]), float(source_pose[1, 3]),
                float(np.arctan2(source_pose[1, 0], source_pose[0, 0]))]})
        self._single_step_posture_since = None
        self._retreat_offsets = [np.asarray(offset, float) for offset in
                                 self.task.get("retreat_waypoints_w", [self.task["retreat_offset_w"]])]
        self._retreat_index = 0
        # Opt-in (runtime_v26, reference_continuity.py): blended pass-through of eligible intermediate lift/transfer
        # waypoints, planned once per stage entry, and a shorter strict dwell for intermediate waypoints.
        # Absent both keys these stay empty/None and no evidence or arrival rule changes.
        self._continuity = {}
        if any(isinstance(self.task.get(key), Mapping) for key in ("lift_reference_continuity",
                                                                   "transfer_reference_continuity")):
            from .reference_continuity import continuity_settings
            self._continuity = continuity_settings(self.task) or {}
        self._continuity_plan = self._continuity_plan_stage = None
        self._intermediate_dwell_s = self.config["thresholds"].get("object_waypoint_intermediate_dwell_s")
        self._posture_names = tuple(self.task["posture_paths"]["reach_posture"]["joint_names"])
        self._limits = upright_readiness_limits(self.goal_settings)
        # First-step evidence: the upright-start command and the initialized CRS pose.
        self._upright_command = self._source_anchor = None
        # The boundary demands at least its own readiness dwell of continuous machine
        # readiness at the transition; leaving after a shorter dwell never records.
        self._preparation_dwell_s = max(self.goal_settings["preparation_dwell_s"],
                                        float(self.config["recording_boundary"]["readiness_dwell_s"]))

    def _read_observation(self, raw):
        observed = super()._read_observation(raw)
        # Both hands and the JIG support are measured every step; malformed
        # per-arm or per-object containers are invalid observations, not crashes.
        evidence = raw.get("contact_evidence", {})
        if not isinstance(evidence, Mapping) or any(not isinstance(evidence.get(arm, {}), Mapping)
                                                    for arm in ("left", "right")):
            raise ValueError("contact_evidence must contain per-arm sensor evidence")
        support = raw.get("support_evidence", {})
        if not isinstance(support, Mapping) or not isinstance(support.get(self.object_path, {}), Mapping):
            raise ValueError("support_evidence must contain per-object support evidence")
        if self._stationary_pelvis_settings is not None:
            observed["pelvis"] = pose_to_matrix(raw.get("pelvis_pose_w"))
            if math.hypot(observed["pelvis"][0, 0], observed["pelvis"][1, 0]) < 1e-8:
                raise ValueError("stationary pelvis heading is undefined")
        if self._handover is not None:
            observed["hands"] = self._read_hands(raw)
        return observed

    def _read_hands(self, raw):
        """Both hands' measured TCP, jaw state, contacts and pinch evidence (opt-in handover only)."""
        hands = {}
        for arm in ("left", "right"):
            opened, closed = raw["hand_open"][arm], raw["hand_closed"][arm]
            if (not isinstance(opened, (bool, np.bool_)) or not isinstance(closed, (bool, np.bool_))
                    or (opened and closed)):
                raise ValueError(f"hand_open/hand_closed must be measured booleans: {arm}")
            contacts = raw["contacts"][arm]
            if isinstance(contacts, (str, bytes)) or not isinstance(contacts, (list, tuple, set, frozenset)):
                raise ValueError("contacts must contain full object prim paths")
            hands[arm] = {"tcp": pose_to_matrix(raw["tcp_poses_w"][arm]), "open": bool(opened), "closed": bool(closed),
                          "contacts": set(contacts), "bilateral": _bilateral_contact(raw, arm, self.object_path),
                          "contact_force_n": _contact_force(raw, arm, self.object_path),
                          "contact_absent": _contact_absent(raw, arm, self.object_path)}
        return hands

    def _posture_path(self, stage):
        """The joint path of a posture stage: posture_paths, else the handover block's handover_posture."""
        path = self.task["posture_paths"].get(stage)
        if path is None and self._handover is not None:
            path = self._handover["posture_paths"].get(stage)
        if path is None and self._rotation is not None and stage == HANDOVER_ROTATE_STAGE:
            path = self._rotation["rotation_path"]
        if path is None and self._single_step is not None and stage == SINGLE_STEP_STAGE:
            path = self._single_step.pre_step_path  # opt-in single step: None without a pre-step path
        return path

    def _retreat_waypoint_goal(self, index):
        """Retreat waypoint = cleared release origin + offset; the first is the validated retreat goal."""
        goal = self._retreat_goal()
        if index:
            goal[:3, 3] += self._retreat_offsets[index] - np.asarray(self.task["retreat_offset_w"], float)
        return goal

    def _enter(self, index, observation):
        if self._placement_hold is not None:
            self._placement_hold.reset()
            self._support_hold_active = False
            self._support_latch_pending = False
        if self._support_filter is not None and self.stages[index] == "place":
            # The filter starts at the place-entry observation; earlier time counts as zero support force.
            self._support_filter.reset(observation["time_s"])
        if self._aligned_descent is not None:
            self._aligned_descent.reset()
            self._object_advance_hold_reason = None
        # Hand-path transit starts suppressing the object feedback from the first transfer command.
        self._object_feedback_hold_reason = ("transfer_transit_hand_path"
                                             if self.stages[index] == "transfer" and self._transit_hand_path(0) else None)
        if self.stages[index] == "place" and self._placement_alignment_settings is not None:
            # Latch the command frame from this measured observation only. The
            # original _jig_anchor remains the independent displacement guard.
            from .placement_alignment import placement_command_slot_pose
            jig = observation["objects"][self.jig_path]
            command_slot = placement_command_slot_pose(self._placement_alignment_settings, self._slot_pose)
            self._target_object = jig @ command_slot
            self._placement_jig_reference = {"measurement_time_s": observation["time_s"],
                "jig_pose_w": matrix_to_pose(jig), "target_object_pose_w": matrix_to_pose(self._target_object),
                "source": "measured_jig_at_place_entry", "original_jig_anchor_retained": True}
            if "command_height_offset_m" in self._placement_alignment_settings:
                self._placement_jig_reference.update(
                    command_height_offset_m=self._placement_alignment_settings["command_height_offset_m"],
                    command_offset_frame="jig_local_z", acceptance_slot_pose_unchanged=True)
        NepesTaskMachine._enter(self, index, observation)
        stage, now = self.stage_id, observation["time_s"]
        self._stage_times[stage] = now
        if self._no_walk is not None:
            self._no_walk.enter(stage, now, observation["pelvis"])
        if self._single_step is not None and index > 0 and self.stages[index - 1] == SINGLE_STEP_STAGE:
            # Opt-in single step: freeze the accepted stance and re-anchor the stationary pelvis hold at the
            # measured pelvis of this exit, so its drift limits protect every post-step stage.
            self._single_step.complete(now, observation["pelvis"])
            self._stationary_pelvis_anchor = observation["pelvis"].copy()
            self._stationary_pelvis_anchor_time_s = now
            self._check_stationary_pelvis(observation)
        path = self._posture_path(stage)
        if is_applied_hold(path):
            # _raw is the observation used for this transition, before this
            # new stage computes any actuator command. Never seed measured q.
            self._posture_hold_references[stage] = hold_anchor(
                self._raw.get("commanded_upper_joint_positions"), path["joint_names"], now)
        if stage == RECORDING_START_STAGE and self._stationary_pelvis_settings is not None:
            self._stationary_pelvis_anchor = observation["pelvis"].copy()
            self._stationary_pelvis_anchor_time_s = now
            self._check_stationary_pelvis(observation)
        if stage == RECORDING_START_STAGE and self._stationary_feet is not None:
            self._check_stationary_feet(anchor=True)
        if stage == "pregrasp":
            # The slot target stays the first-step JIG anchor; acceptance re-measures the JIG.
            self._target_object = self._jig_anchor @ self._slot_pose
        elif stage == "lift":
            self._capture_time_s = now
            self._max_lift_m = 0.
        elif stage == "navigate_target" and self._single_step is not None:
            # Opt-in single step: never brake before the goal latch. Without a pre-step path the measured
            # torso and pelvis of this transition latch the goal of the first navigate_target command.
            self._single_step.enter(now, observation["base"], observation["pelvis"], observation["base_velocity"])
            self._single_step_posture_since = None
        elif stage == "navigate_target":
            # A torso already inside the target station (e.g. a profile without a
            # side-step) brakes from the first command instead of creeping on.
            self._base_near(observation, "target")
        elif stage == "release":
            self._release_origin = observation["tcp"].copy()
            self._supported_release_tcp = self._release_origin.copy()
            self._tcp_target = self._release_origin.copy()
            self._release_phase = "opening"
            self._release_subphase_events.append({"phase": "opening", "time_s": now})
        elif stage == "retreat":
            self._retreat_index = 0
            self._tcp_target = self._retreat_waypoint_goal(0)
        elif stage == "settle":
            # The rest posture owns the arm; a Cartesian goal would drive it back to the JIG.
            self._tcp_target = None
        if self._handover is not None:
            self._enter_handover(stage, observation)
        if self._continuity:
            self._plan_continuity(stage, observation)

    def _plan_continuity(self, stage, observation):
        """Opt-in blended pass-through: the static per-waypoint plan of a declared stage, at its entry.

        The plan uses the nominal TCP polyline the runtime's first controller starts on: the measured entry TCP
        of this transition, then each object waypoint goal through the stage control grasp.
        """
        self._continuity_plan = self._continuity_plan_stage = None
        settings = self._continuity.get(stage)
        if settings is None or not self._object_path_goals:
            return
        from .reference_continuity import plan_pass_through
        from .upright_runtime import object_pose_settings
        goals = [goal @ self._control_object_to_tcp for goal in self._object_path_goals]
        self._continuity_plan = plan_pass_through(settings, observation["tcp"], goals,
            [object_pose_settings(self.task, stage, index) for index in range(len(goals))])
        self._continuity_plan_stage = stage

    def _waypoint_arrival(self, stage):
        """(arrival kind, required dwell, waypoint index, waypoint count) of the current lift/transfer/retreat
        waypoint under the opt-in keys, else None.

        Eligible blended waypoints pass without dwell; every other intermediate waypoint (not the last one, nor
        one of a declared continuity's final_waypoints) uses object_waypoint_intermediate_dwell_s when declared;
        final waypoints, and everything without the keys, keep reach_dwell_s.
        """
        reach = self.config["thresholds"]["reach_dwell_s"]
        if stage in ("lift", "transfer") and self._object_path_goals:
            index, count = self._object_path_index, len(self._object_path_goals)
            settings = self._continuity.get(stage)
            if (settings is not None and self._continuity_plan_stage == stage
                    and self._continuity_plan[index]["eligible"]):
                return "pass_through", 0., index, count
            intermediate = index < count - (settings["final_waypoints"] if settings is not None else 1)
        elif stage == "retreat":
            index, count = self._retreat_index, len(self._retreat_offsets)
            intermediate = index < count - 1
        else:
            return None
        dwell = self._intermediate_dwell_s if intermediate and self._intermediate_dwell_s is not None else reach
        return "strict", float(dwell), index, count

    # --- opt-in handover (task["handover"]; handover.py) ---------------------------------------------

    def _enter_handover(self, stage, observation):
        """Handover stage entries: the observation-arm switch, receiving goals, torso-frame holds, grasp switch."""
        now, settings = observation["time_s"], self._handover
        giving, receiving, hands = self._giving_arm, self._receiving_arm, observation["hands"]
        ring = observation["objects"][self.object_path]
        if stage == "handover_posture":
            self._tcp_target = self._secondary_target = None
        elif stage == "handover_pregrasp":
            # From here the machine observes the RECEIVING hand (pregrasp/approach/close semantics) while
            # the giving hand's pinch and capture retention are checked from the raw evidence (_GIVING_HELD).
            self._giving_object_to_tcp = self._object_to_tcp.copy()
            self.arm = receiving
            self._giving_hold_in_base = np.linalg.inv(observation["base"]) @ hands[giving]["tcp"]
            self._secondary_target = hands[giving]["tcp"].copy()
            self._tcp_target = _shift(ring @ self._receiving_grasp, settings["pregrasp_offset_w"])
            self._handover_events.append({"event": "observation_arm_switched", "time_s": now, "arm": receiving})
        elif stage in ("handover_approach", "handover_close"):
            self._tcp_target = ring @ self._receiving_grasp
        elif stage == HANDOVER_ROTATE_STAGE:
            # Opt-in two-hand rotation: an open-loop two-arm joint path with both jaws closed. Both grasps are
            # monitored against their own references: the giving pick capture and the receiving grasp as
            # measured here (verified in handover_close). No Cartesian target: the joint path owns both arms.
            self._tcp_target = self._secondary_target = None
            self._rotate_entry_ring = ring.copy()
            self._rotate_receiving_capture = np.linalg.inv(ring) @ hands[receiving]["tcp"]
            self._rotate_off_top_since = self._lost_rotate_grasp_since = None
            self._holding_arm = "both"
            entry = rotation_measurement(self._rotation, ring, hands[receiving]["tcp"])
            self._handover_events.append({"event": "two_hand_rotation_started", "time_s": now,
                "receiving_capture_object_to_tcp": matrix_to_pose(self._rotate_receiving_capture),
                "receiving_offset_from_top_rad": entry["receiving_offset_from_top_rad"],
                "ring_tilt_rad": entry["ring_tilt_rad"]})
        elif stage == "handover_release":
            if self._rotation is not None:
                # The rotation ended on this observation: record the measured offset and the rotation achieved,
                # and re-anchor the giving hand's torso-frame hold where the rotation left it.
                final = rotation_measurement(self._rotation, ring, hands[receiving]["tcp"], self._rotate_entry_ring)
                self._rotation_completed = {"time_s": now, **{key: final[key] for key in (
                    "receiving_offset_from_top_rad", "receiving_offset_from_top_deg", "receiving_offset_from_top_signed_rad",
                    "at_top", "ring_rotation_rad", "ring_rotation_deg", "ring_tilt_rad")}}
                self._giving_hold_in_base = np.linalg.inv(observation["base"]) @ hands[giving]["tcp"]
                self._handover_events.append({"event": "two_hand_rotation_complete", **self._rotation_completed})
            # The receiving pinch was verified in handover_close: its measured grasp is now the retention
            # reference. The pick capture stays in the pickup contract and the giving-hand evidence.
            self._receiving_object_to_tcp = np.linalg.inv(ring) @ observation["tcp"]
            self._object_to_tcp = self._receiving_object_to_tcp.copy()
            self._control_object_to_tcp = self._receiving_object_to_tcp.copy()
            self._handover_capture_time_s = self._handover_switch_time_s = now
            self._receiving_hold_in_base = np.linalg.inv(observation["base"]) @ observation["tcp"]
            self._tcp_target = observation["tcp"].copy()
            self._giving_release_origin = hands[giving]["tcp"].copy()
            self._secondary_target = hands[giving]["tcp"].copy()
            # Opt-in pre_release_dwell_s: close second, dwell, open first. The giving jaw stays closed while
            # the receiving pinch is re-verified; absent the option the giving jaw opens at entry (unchanged).
            self._handover_release_phase = (RELEASE_VERIFYING_PHASE if "pre_release_dwell_s" in settings else "opening")
            self._handover_verify_since = None
            self._holding_arm = "both"
            self._handover_events.append({"event": "grasp_reference_switched", "time_s": now, "arm": receiving,
                "receiving_capture_object_to_tcp": matrix_to_pose(self._receiving_object_to_tcp)})
        elif stage == "handover_retreat":
            self._holding_arm = receiving
            self._handover_retreat_index = 0
            self._secondary_target = self._handover_retreat_goal(0)
            self._handover_events.append({"event": "giving_hand_released", "time_s": now, "arm": giving})
        elif stage == "navigate_target":
            self._secondary_target = None
            self._handover_events.append({"event": "handover_complete", "time_s": now, "holding_arm": self._holding_arm})

    def _handover_retreat_goal(self, index):
        """Giving-hand retreat waypoint = its pose at the release entry + the declared world offset."""
        goal = self._giving_release_origin.copy()
        goal[:3, 3] += np.asarray(self._handover["release_retreat_waypoints_w"][index], float)
        return goal

    def _refresh_handover_targets(self, obs):
        """Per step: hold targets follow the measured torso (base_link); receiving goals follow the measured ring."""
        stage, base = self.stage_id, obs["base"]
        if stage in _GIVING_HELD:
            self._secondary_target = base @ self._giving_hold_in_base
            if stage != "handover_close":  # object-pose feedback: the ring may drift in the giving hand
                goal = obs["objects"][self.object_path] @ self._receiving_grasp
                self._tcp_target = (_shift(goal, self._handover["pregrasp_offset_w"])
                                    if stage == "handover_pregrasp" else goal)
        elif stage in ("handover_release", "handover_retreat"):
            self._tcp_target = base @ self._receiving_hold_in_base
            if stage == "handover_release" and self._handover_release_phase in (RELEASE_VERIFYING_PHASE, "opening"):
                self._secondary_target = base @ self._giving_hold_in_base

    def _handover_measure(self, obs, stage):
        """Both hands' evidence and the kinematic hand-hand proxy, published every step (metrics and command)."""
        settings, hands, ring = self._handover, obs["hands"], obs["objects"][self.object_path]
        giving, receiving = self._giving_arm, self._receiving_arm
        giving_capture = (self._giving_object_to_tcp if self._giving_object_to_tcp is not None
                          else self._object_to_tcp if self.arm == giving else None)

        def side(arm, capture):
            hand, retention = hands[arm], None
            if capture is not None:
                measured = np.linalg.inv(ring) @ hand["tcp"]
                position, rotation = _errors(measured, capture)
                retention = {"position_error_m": position, "rotation_error_rad": rotation,
                             "valid": bool(hand["bilateral"] and hand["closed"] and self.object_path in hand["contacts"]
                                           and self._near(measured, capture, "capture"))}
            return {"arm": arm, "open": hand["open"], "closed": hand["closed"], "bilateral_pinch": hand["bilateral"],
                    "object_contact": self.object_path in hand["contacts"], "contact_absent": hand["contact_absent"],
                    "contact_force_n": hand["contact_force_n"], "sensor_available": hand["contact_force_n"] is not None,
                    "capture_object_to_tcp": None if capture is None else matrix_to_pose(capture),
                    "retention": retention}

        distance = float(np.linalg.norm(hands[giving]["tcp"][:3, 3] - hands[receiving]["tcp"][:3, 3]))
        # During the opt-in two-hand rotation the receiving hand is judged against its rotate-entry grasp.
        receiving_capture = (self._rotate_receiving_capture if stage == HANDOVER_ROTATE_STAGE
                             else self._receiving_object_to_tcp)
        evidence = {"schema_version": HANDOVER_SCHEMA, "stage_id": stage, "measurement_time_s": obs["time_s"],
                "giving_arm": giving, "receiving_arm": receiving, "holding_arm": self._holding_arm,
                "observation_arm": self.arm, "in_handover": stage in HANDOVER_STAGE_SET,
                "hand_tcp_distance_m": distance, "minimum_hand_tcp_distance_m": settings["minimum_hand_tcp_distance_m"],
                "hand_tcp_distance_measurement": HAND_TCP_DISTANCE_MEASUREMENT,
                "giving": side(giving, giving_capture), "receiving": side(receiving, receiving_capture),
                "release_phase": self._handover_release_phase if stage == "handover_release" else None,
                "retreat_waypoint_index": self._handover_retreat_index if stage == "handover_retreat" else None,
                "retreat_waypoint_count": len(settings["release_retreat_waypoints_w"]),
                "receiving_capture_object_to_tcp": (None if self._receiving_object_to_tcp is None
                                                    else matrix_to_pose(self._receiving_object_to_tcp)),
                "receiving_capture_time_s": self._handover_capture_time_s,
                "grasp_reference_switch_time_s": self._handover_switch_time_s,
                "events": deepcopy(self._handover_events)}
        if self._rotation is not None:
            # Opt-in two-hand rotation: the live ring tilt, the receiving grasp point's offset from the top of the
            # rim and (from the rotate entry) the ring spin achieved; ``completed`` is frozen at the release entry.
            rotation = rotation_measurement(self._rotation, ring, hands[receiving]["tcp"], self._rotate_entry_ring)
            rotation.update(stage_active=stage == HANDOVER_ROTATE_STAGE,
                            entry_ring_pose_w=(None if self._rotate_entry_ring is None
                                               else matrix_to_pose(self._rotate_entry_ring)),
                            receiving_capture_object_to_tcp=(None if self._rotate_receiving_capture is None
                                                             else matrix_to_pose(self._rotate_receiving_capture)),
                            giving_capture_object_to_tcp=(None if giving_capture is None
                                                          else matrix_to_pose(giving_capture)),
                            completed=deepcopy(self._rotation_completed))
            evidence["rotation"] = rotation
        return evidence

    def _handover_guard(self, obs, stage):
        """Fail-closed guards of the handover stages; None when nothing is wrong."""
        if stage not in HANDOVER_STAGE_SET:
            return None
        evidence, hands, now = self._handover_evidence, obs["hands"], obs["time_s"]
        if not evidence["giving"]["sensor_available"] or not evidence["receiving"]["sensor_available"]:
            return "handover:contact_sensor_unavailable"
        if evidence["hand_tcp_distance_m"] < self._handover["minimum_hand_tcp_distance_m"]:
            return "handover_hand_clearance"
        if stage == HANDOVER_ROTATE_STAGE:
            # Both hands pinch the ring and keep their own grasp references (lost_grasp_grace_s applies to a
            # flicker); the ring plane may not tilt beyond max_tilt_rad.
            if not evidence["rotation"]["ring_tilt_valid"]:
                return "handover_rotate_ring_tilt"
            retained = all(evidence[hand]["retention"] is not None and evidence[hand]["retention"]["valid"]
                           for hand in ("giving", "receiving"))
            if retained:
                self._lost_rotate_grasp_since = None
            elif self._lost_rotate_grasp_since is None:
                self._lost_rotate_grasp_since = now
            elif now - self._lost_rotate_grasp_since >= self.config["thresholds"]["lost_grasp_grace_s"]:
                return "handover_rotate_grasp_lost"
        if stage == "handover_pregrasp" and not hands[self._receiving_arm]["contact_absent"]:
            return "handover_pregrasp:unexpected_object_contact"
        if stage in _GIVING_HELD:
            retention = evidence["giving"]["retention"]
            if retention is not None and retention["valid"]:
                self._lost_giving_grasp_since = None
            elif self._lost_giving_grasp_since is None:
                self._lost_giving_grasp_since = now
            elif now - self._lost_giving_grasp_since >= self.config["thresholds"]["lost_grasp_grace_s"]:
                return "grasp_lost"
        if stage == "handover_retreat" and not (hands[self._giving_arm]["open"] and hands[self._giving_arm]["contact_absent"]):
            return "handover_release_lost"
        return None

    def _handover_close_pose_valid(self, obs):
        """The receiving grasp within the acquisition bounds of its designed grasp, hand at its frozen goal."""
        threshold = self.config["thresholds"]
        measured = np.linalg.inv(obs["objects"][self.object_path]) @ obs["tcp"]
        position, rotation = _errors(measured, self._receiving_grasp)
        return bool(position <= self._handover.get("close_acquisition_position_m", threshold["capture_position_m"])
                    and rotation <= self._handover.get("close_acquisition_rotation_rad", threshold["capture_rotation_rad"])
                    and self._near(obs["tcp"], self._tcp_target))

    def _handover_command(self, command):
        """Command fields of a handover profile: both jaws, the second servoed arm and the holding-arm provenance."""
        stage, running = self.stage_id, self.status == "running"
        giving, receiving = self._giving_arm, self._receiving_arm
        if running and stage in HANDOVER_STAGE_SET:
            command["allowed_contact_paths"] = [self.object_path]
            if stage in HANDOVER_JOINT_PATH_STAGES:
                # handover_posture and the opt-in handover_rotate: the two-arm joint path owns both arms,
                # the base command is exactly zero and the holding jaw(s) stay closed.
                command.update(motion_mode="joint_posture", posture_stage=stage, hand_command="close")
            else:
                command.update(motion_mode="contact_servo",
                               hand_command="open" if stage in ("handover_pregrasp", "handover_approach") else "close")
        secondary = bool(running and stage in HANDOVER_CARTESIAN_STAGES and self._secondary_target is not None)
        if self._last_observation is not None and "hands" in self._last_observation:
            # A stage entry changes the observation arm, the holding arm, the captures and the events
            # after the step's evidence was measured: publish the evidence of the stage this command is for.
            self._handover_evidence = self._handover_measure(self._last_observation, stage)
            if self._metrics is not None:
                self._metrics["handover"] = deepcopy(self._handover_evidence)
            if isinstance(command.get("stationary_metrics"), dict):
                command["stationary_metrics"]["handover"] = deepcopy(self._handover_evidence)
        command.update(
            holding_arm=self._holding_arm,
            contact_arms=[giving, receiving] if stage in HANDOVER_STAGE_SET else [self.arm],
            hand_commands=hand_commands_for_stage(
                self.task, stage, command["hand_command"], running=running,
                release_phase=self._handover_release_phase if stage == "handover_release" else None),
            secondary_arm=giving if secondary else None,
            secondary_tcp_target_pose_w=matrix_to_pose(self._secondary_target) if secondary else None,
            handover=deepcopy(self._handover_evidence))

    def _check_stationary_feet(self, *, anchor=False):
        """Judge native sole/ground evidence; never infer feet from pelvis or FK."""
        if self._stationary_feet is None or (self._stationary_feet_evidence is None and not anchor):
            return None
        evidence = self._stationary_feet.check(self._raw, anchor=anchor)
        self._stationary_feet_evidence = evidence
        if self._metrics is not None:
            self._metrics["stationary_feet_hold"] = deepcopy(evidence)
        return None if evidence["valid"] else "stationary_feet_hold:" + evidence["reason"]

    def _check_stationary_pelvis(self, observation):
        """Pin the measured pelvis at recording start; torso motion is not pelvis drift.

        Opt-in single step: from its goal latch to the navigate_target exit the drift is still measured
        but cannot fail (the step moves the pelvis on purpose); the exit re-anchors the hold.
        """
        if self._stationary_pelvis_anchor is None:
            return None
        anchor, current = self._stationary_pelvis_anchor, observation["pelvis"]
        delta = float(np.linalg.norm(current[:2, 3] - anchor[:2, 3]))
        angle = math.atan2(current[1, 0], current[0, 0]) - math.atan2(anchor[1, 0], anchor[0, 0])
        yaw = abs(math.atan2(math.sin(angle), math.cos(angle)))
        limits = self._stationary_pelvis_settings
        valid = delta <= limits["maximum_position_drift_m"] and yaw <= limits["maximum_yaw_drift_rad"]
        self._stationary_pelvis_evidence = {"valid": valid, "measurement_time_s": observation["time_s"],
            "anchor_time_s": self._stationary_pelvis_anchor_time_s,
            "anchor_pose_w": matrix_to_pose(anchor), "measured_pose_w": matrix_to_pose(current),
            "position_drift_m": delta, "yaw_drift_rad": yaw, "limits": deepcopy(limits),
            "position_measurement": "world_xy", "orientation_measurement": "wrapped_world_yaw",
            "zero_base_command_required": True, "actual_base_motion_zero_claimed": False}
        stepping_stage = self._single_step is not None and self.stage_id == SINGLE_STEP_STAGE
        suspended = stepping_stage and self._single_step.latched
        if self._single_step is not None:
            self._stationary_pelvis_evidence["drift_check_suspended"] = suspended
        if self._metrics is not None:
            self._metrics["stationary_pelvis_hold"] = deepcopy(self._stationary_pelvis_evidence)
        if not stepping_stage:
            # (The opt-in single step's one-shot brake owns the arrival braking of navigate_target.)
            self._base_braking = True
        return None if valid or suspended else "stationary_pelvis_drift"

    def _base_near(self, observation, station):
        if self._stationary_pelvis_settings is None or station != "target":
            return super()._base_near(observation, station)
        self._base_braking = True
        if self._stationary_pelvis_anchor is None:
            return False
        valid = self._check_stationary_pelvis(observation) is None
        navigation = self.task["navigation_control"]
        velocity = observation["base_velocity"]
        slow = bool(velocity is not None
                    and np.linalg.norm(velocity[:3]) <= navigation.get("arrival_linear_speed_m_s", .03)
                    and abs(velocity[5]) <= navigation.get("arrival_yaw_speed_rad_s", .12))
        if valid and slow:
            if self._base_settle_since is None:
                self._base_settle_since = observation["time_s"]
        else:
            self._base_settle_since = None
        dwell = (0. if self._base_settle_since is None else observation["time_s"] - self._base_settle_since)
        self._stationary_pelvis_evidence.update(base_quiet=slow, arrival_continuous_dwell_s=dwell,
            arrival_required_dwell_s=navigation.get("arrival_dwell_s", .4))
        if self._metrics is not None:
            self._metrics["stationary_pelvis_hold"] = deepcopy(self._stationary_pelvis_evidence)
        return valid and slow and dwell >= navigation.get("arrival_dwell_s", .4)

    def _single_step_condition(self, raw, obs, gates):
        """navigate_target of the opt-in single step; returns (condition, failure).

        Before the goal latch the pre-step posture path runs under the posture-stage rules with the grasp
        retained (the runtime holds the base command at exactly zero); its completion latches the goal from
        this observation, so the next command steps. After the latch the step state latches the one-shot
        brake and judges the settled window; never a corrective command. The stage passes once the window
        is accepted and the grasp retained (for the reach dwell).
        """
        step, now = self._single_step, obs["time_s"]
        if not step.latched:
            condition, failure = self._posture_condition(raw, obs, SINGLE_STEP_STAGE, gates)
            if failure is not None:
                return False, failure
            evidence = self._posture_evidence[SINGLE_STEP_STAGE]
            evidence["grasp_retained"] = retained = self._grasp_valid(obs)
            ready = bool(condition and retained)
            if not ready:
                self._single_step_posture_since = None
            elif self._single_step_posture_since is None:
                self._single_step_posture_since = now
            dwell = 0. if self._single_step_posture_since is None else now - self._single_step_posture_since
            evidence["continuous_dwell_s"] = dwell
            if ready and dwell >= evidence["required_dwell_s"] - 1e-10:
                step.latch(now, obs["base"], obs["pelvis"], obs["base_velocity"])
                self._base_braking = False
            return False, None
        try:
            verdict = step.observe(now, obs["base"], obs["pelvis"], obs["base_velocity"],
                                   raw.get("commanded_base_velocity"))
        except ValueError as exc:
            return False, f"invalid_observation:{exc}"
        self._base_braking = step.braking
        if verdict == SINGLE_STEP_REJECTED:
            return False, SINGLE_STEP_OUTSIDE_WINDOW
        return bool(verdict == SINGLE_STEP_ACCEPTED and self._grasp_valid(obs)), None

    def _pickup_contract(self):
        return {"capture_time_s": self._capture_time_s, "lift_m": self._max_lift_m,
                "minimum_lift_m": self.goal_settings["minimum_lift_m"]}

    def _upright_readiness(self, raw, obs, gates):
        """Unrecorded upright stance evidence; the commanded upright targets come from the runtime.

        The measured posture joints are compared with the command applied in the
        previous step, and that command with the first step's (the upright start):
        reaching a changed command would certify a pre-pose as upright.
        """
        cfg, limits, now = self.goal_settings, self._limits, obs["time_s"]
        names = self._posture_names
        measured = _joint_values(raw.get("joint_positions"), names)
        commanded = _joint_values(raw.get("commanded_upper_joint_positions"), names)
        worst_joint = worst = change = None
        if measured is not None and commanded is not None:
            errors = {name: abs(measured[name] - commanded[name]) for name in names}
            worst_joint = max(errors, key=errors.get)
            worst = errors[worst_joint]
        if commanded is not None and self._upright_command is not None:
            change = max(abs(commanded[name] - self._upright_command[name]) for name in names)
        source_position, source_rotation = _errors(obs["objects"][self.object_path], self._source_anchor)
        tilt = _pelvis_tilt(raw.get("pelvis_pose_w"))
        forces = {arm: _contact_force(raw, arm, self.object_path) for arm in ("left", "right")}
        station = _pose_or_none(self.task["base_stations"]["source"])
        position_error = yaw_error = None
        if station is not None and obs["base"] is not None:
            position_error = float(np.linalg.norm(obs["base"][:2, 3] - station[:2, 3]))
            yaw = (math.atan2(obs["base"][1, 0], obs["base"][0, 0]) - math.atan2(station[1, 0], station[0, 0]))
            yaw_error = abs(math.atan2(math.sin(yaw), math.cos(yaw)))
        readiness = {
            "valid": False,
            "station_reached": bool(self._base_near(obs, "source")),
            "upright_posture_reached": worst is not None and worst <= cfg["upright_joint_tolerance_rad"],
            "upright_command_retained": change is not None and change <= UPRIGHT_COMMAND_TOLERANCE_RAD,
            "pelvis_upright": tilt is not None and tilt <= cfg["maximum_pelvis_tilt_rad"],
            "hand_open": gates["hand_open"],
            "same_object_contacts_absent": gates["same_object_contacts_absent"],
            "sensors_available": all(force is not None for force in forces.values())
                                 and obs["contact_sensor_available"] is True,
            "object_quiet": gates["object_com_stationary"] and gates["object_pose_stationary"],
            "base_quiet": gates["base_stationary"],
            "source_object_retained": bool(source_position <= limits["max_source_object_position_error_m"]
                                           and source_rotation <= limits["max_source_object_rotation_error_rad"]),
            "upright_reference": "commanded_upper_joint_positions",
            "upright_joint_names": list(names),
            "max_upright_joint_error_rad": worst, "max_upright_error_joint": worst_joint,
            "upright_command_targets": None if self._upright_command is None else dict(self._upright_command),
            "max_upright_command_change_rad": change,
            "pelvis_tilt_rad": tilt, "hand_contact_force_n": forces,
            "base_station_position_error_m": position_error, "base_station_yaw_error_rad": yaw_error,
            "source_object_anchor_pose_w": matrix_to_pose(self._source_anchor),
            "source_object_position_error_m": source_position, "source_object_rotation_error_rad": source_rotation,
            **{key: self._metrics[key] for key in (
                "object_com_linear_speed_m_s", "object_com_angular_speed_rad_s", "object_pose_linear_speed_m_s",
                "object_pose_angular_speed_rad_s", "base_linear_speed_m_s", "base_yaw_speed_rad_s")},
            "required_dwell_s": self._preparation_dwell_s, "continuous_dwell_s": 0., "measurement_time_s": now,
            "limits": deepcopy(limits)}
        readiness["valid"] = all(readiness[key] for key in READINESS_GATES)
        return readiness

    def _posture_condition(self, raw, obs, stage, gates):
        """Joint-space reference completion plus measured arrival, stillness and a quiet base.

        Returns (condition, failure). Tracking compares the measured joints with
        the targets actually applied in the previous control step.
        """
        path = self._posture_path(stage)
        names = path["joint_names"]
        hold = is_applied_hold(path)
        anchor = self._posture_hold_references.get(stage)
        if hold and anchor is None:
            return False, f"posture_hold_anchor_missing:{stage}"
        final = (anchor["joint_positions"] if hold
                 else dict(zip(names, map(float, path["waypoints"][-1]))))
        reference = raw.get("posture_reference")
        if reference is not None and not isinstance(reference, Mapping):
            return False, f"posture_reference_invalid:{stage}"
        current = isinstance(reference, Mapping) and reference.get("stage_id") == stage
        if hold and current and reference.get("reference_mode") != HOLD_MODE:
            return False, f"posture_reference_mismatch:{stage}"
        reported = reference.get("final_joint_positions") if current else None
        if current and reported is not None and (
                _joint_values(reported, names) is None or set(reported) != set(names)
                or any(abs(float(reported[name]) - final[name]) > 1e-9 for name in names)):
            return False, f"posture_reference_mismatch:{stage}"
        complete = bool(current and reported is not None and reference.get("status") == "running"
                        and reference.get("complete") is True)
        measured = _joint_values(raw.get("joint_positions"), names)
        commanded = _joint_values(raw.get("commanded_upper_joint_positions"), names)
        velocities = _joint_values(raw.get("joint_velocities"), names)
        if hold and (commanded is None or any(abs(commanded[n] - final[n]) > 1e-9 for n in names)):
            return False, f"posture_hold_command_changed:{stage}"
        tracking = (None if measured is None or commanded is None
                    else max(abs(measured[name] - commanded[name]) for name in names))
        arrival_joint = arrival = speed = None
        if measured is not None:
            errors = {name: abs(measured[name] - final[name]) for name in names}
            arrival_joint = max(errors, key=errors.get)
            arrival = errors[arrival_joint]
        readback = None if velocities is None else max(abs(value) for value in velocities.values())
        # Opt-in (posture path settle_speed_source): stillness from consecutive measured positions
        # instead of the joint-velocity readback; absent, the evidence and gate are unchanged.
        source = path.get("settle_speed_source")
        speed = (self._measured_position_speed(stage, measured, obs["time_s"])
                 if source == "measured_positions" else readback)
        self._posture_evidence[stage] = {
            "stage_id": stage, "reference_complete": complete,
            "reference_elapsed_s": reference.get("elapsed_s") if current else None,
            "reference_duration_s": reference.get("duration_s") if current else None,
            "max_arrival_error_rad": arrival, "max_arrival_error_joint": arrival_joint,
            "max_joint_speed_rad_s": speed, "max_tracking_error_rad": tracking,
            "arrival_tolerance_rad": path["arrival_tolerance_rad"],
            "settle_velocity_rad_s": path["settle_velocity_rad_s"],
            "tracking_error_limit_rad": path["max_tracking_error_rad"],
            "base_quiet": gates["base_stationary"], "grasp_retained": None,
            "required_dwell_s": path["settle_dwell_s"], "continuous_dwell_s": 0.,
            "measurement_time_s": obs["time_s"]}
        if hold:
            self._posture_evidence[stage]["hold_reference"] = deepcopy(anchor)
        if source is not None:
            self._posture_evidence[stage]["settle_speed_source"] = source
            self._posture_evidence[stage]["max_joint_velocity_readback_rad_s"] = readback
        if tracking is not None and tracking > path["max_tracking_error_rad"]:
            return False, f"posture_tracking_error:{stage}"
        return bool(complete and arrival is not None and arrival <= path["arrival_tolerance_rad"]
                    and speed is not None and speed <= path["settle_velocity_rad_s"]
                    and gates["base_stationary"]), None

    def _measured_position_speed(self, stage, measured, time_s):
        """Posture-joint speed implied by consecutive measured positions, max |dq| / dt (rad/s).

        Opt-in alternative (posture path ``settle_speed_source: measured_positions``) to
        the joint-velocity readback: a simulated articulation can report a steady
        nonzero joint velocity on a joint locked by a stiff grasp contact while its
        measured position does not change, which would never pass the stillness gate.
        None until two finite samples of the stage exist (absent evidence never passes).
        """
        previous = self._posture_settle_samples.get(stage)
        self._posture_settle_samples[stage] = (float(time_s), measured)
        if measured is None or previous is None or previous[1] is None:
            return None
        dt = float(time_s) - previous[0]
        if dt <= 0.:
            return None
        return max(abs(measured[name] - previous[1][name]) for name in measured) / dt

    def _check_payload_transport(self, obs, stage, *, at_walk_entry=False):
        """Optional measured envelope guard, including the command-producing walk transition."""
        if self._transport_constraints is None:
            return None
        evidence = evaluate_payload_transport(self._transport_constraints,
            matrix_to_pose(obs["objects"][self.object_path]), stage, at_walk_entry=at_walk_entry,
            slot_target_pose_w=matrix_to_pose(self._target_object) if self._target_object is not None else None)
        evidence["measurement_time_s"] = obs["time_s"]
        evidence["object_path"] = self.object_path
        if evidence["checked"]:
            self._transport_evidence = evidence
            if at_walk_entry:
                self._walk_entry_transport_evidence = deepcopy(evidence)
        if self._metrics is not None:
            self._metrics["payload_transport"] = deepcopy(self._transport_evidence)
            self._metrics["walk_entry_payload_transport"] = deepcopy(self._walk_entry_transport_evidence)
        if not evidence["valid"]:
            return "unsafe_payload_transport:" + ";".join(evidence["violations"])
        return None

    def step(self, raw):
        if self.status != "running":
            return self._command()
        if raw.get("faults"):
            return self.fail("runtime_fault:" + ";".join(map(str, raw["faults"])))
        if raw.get("unreachable") is True:
            return self.fail("target_unreachable")
        try:
            obs = self._read_observation(raw)
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            return self.fail(f"invalid_observation:{exc}")
        now, obj = obs["time_s"], obs["objects"][self.object_path]
        if self._previous_time is not None:
            dt = now - self._previous_time
            if dt > self.config["thresholds"]["max_observation_gap_s"]:
                return self.fail("observation_gap")
            distance, angle = _errors(obj, self._previous_object)
            self._linear_speed, self._angular_speed = distance / dt, angle / dt
        self._previous_time, self._previous_object = now, obj.copy()
        self._last_observation, self._raw = obs, raw
        if self._start_time is None:
            self._start_time = self._stage_start = now
            # The JIG anchor is fixed before any motion: the robot may never shove it.
            self._jig_anchor = obs["objects"][self.jig_path].copy()
            self._target_object = self._jig_anchor @ self._slot_pose
            # Likewise the initialized CRS, until the jaws close on it, and the
            # upright-start command, until recording (no pre-posing).
            self._source_anchor = obj.copy()
            self._upright_command = _joint_values(raw.get("commanded_upper_joint_positions"), self._posture_names)
            self._stage_times[self.stage_id] = now
            self.history.append({"time_s": now, "stage_id": self.stage_id, "status": "running"})
            if self._no_walk is not None:
                # The first measured (settled upright-start) pelvis is the no-walk drift origin.
                self._no_walk.anchor(self.stage_id, now, obs["pelvis"])
        if now - self._start_time >= self.task["timeout_s"]:
            return self.fail("task_timeout")
        if now - self._stage_start >= self.task["stage_timeouts_s"][self.stage_id]:
            return self.fail("stage_timeout:" + self.stage_id)
        stage, threshold, cfg = self.stage_id, self.config["thresholds"], self.goal_settings
        if stage == "lift":
            self._max_lift_m = max(self._max_lift_m, float(obj[2, 3] - self._source_object[2, 3]))
        self._metrics = self._measure(raw, obs)
        self._metrics.update(pickup_demonstrated=True, pickup_contract=self._pickup_contract())
        support_confirmed = self._metrics["gates"]["jig_support_confirmed"]
        opening_supported = True
        if self._support_filter is not None:
            # Declared only: the raw gate and support evidence stay as measured; the place-stage support
            # decision (dwell, hold, descent) uses the causal filtered force (placement_support_filter.py).
            filtered = (self._support_filter.update(now, self._metrics["support_evidence"])
                        if stage == "place" else None)
            self._metrics["placement_support_filter"] = filtered
            if filtered is not None:
                support_confirmed = filtered["confirmed"]
                # runtime_v26 review fix: the irreversible opening also needs continuous seat contact.
                opening_supported = filtered["opening_supported"]
        alignment_ready = True
        if self._placement_alignment_settings is not None:
            from .placement_alignment import measure_placement_alignment
            alignment = measure_placement_alignment(self._placement_alignment_settings, obj,
                obs["objects"][self.jig_path], self._slot_pose, axis_object=self._symmetry_axis)
            alignment.update(measurement_time_s=now, command_reference=deepcopy(self._placement_jig_reference))
            self._metrics["placement_alignment"] = alignment
            alignment_ready = alignment["ready"]
        if self._handover is not None:
            # Opt-in handover: torso-frame holds, ring-following receiving goals, both hands' evidence, guards.
            self._refresh_handover_targets(obs)
            self._handover_evidence = self._handover_measure(obs, stage)
            self._metrics["handover"] = deepcopy(self._handover_evidence)
            handover_failure = self._handover_guard(obs, stage)
            if handover_failure:
                return self.fail(handover_failure)
        if self._no_walk is not None:
            self._no_walk.update(stage, now, obs["pelvis"])
        gates = self._metrics["gates"]
        if not gates["jig_anchor_retained"]:
            return self.fail("jig_displaced")
        feet_failure = self._check_stationary_feet()
        if feet_failure:
            return self.fail(feet_failure)
        stationary_failure = self._check_stationary_pelvis(obs)
        if stationary_failure:
            return self.fail(stationary_failure)
        transport_failure = self._check_payload_transport(obs, stage)
        if transport_failure:
            return self.fail(transport_failure)
        if stage in SOURCE_ANCHORED_STAGES:
            # A body or forearm bump moves the ring without any hand contact.
            position, rotation = _errors(obj, self._source_anchor)
            if (position > self._limits["max_source_object_position_error_m"]
                    or rotation > self._limits["max_source_object_rotation_error_rad"]):
                return self.fail("source_object_displaced")
        if stage in PREPARATION_STAGES:
            commanded = _joint_values(raw.get("commanded_upper_joint_positions"), self._posture_names)
            if (commanded is not None and self._upright_command is not None
                    and any(abs(commanded[name] - self._upright_command[name]) > UPRIGHT_COMMAND_TOLERANCE_RAD
                            for name in self._posture_names)):
                return self.fail("upright_command_changed")
        self._support_active = supported_retained = False
        lateral_errors = None
        if self._aligned_descent is not None and stage in ("transfer", "place"):
            from .place_aligned_descent import measure_lateral_alignment
            lateral_errors = measure_lateral_alignment(obj, obs["objects"][self.jig_path], self._slot_pose,
                                                      axis_object=self._symmetry_axis)
        grasp_loss_supported = False
        if stage == "place":
            self._support_active = bool(gates["slot_pose_reached"] and gates["slot_normal_reached"]
                                        and support_confirmed and alignment_ready)
            measured_capture = np.linalg.inv(obj) @ obs["tcp"]
            supported_retained = bool(self._support_active and not obs["open"]
                                      and self._near(measured_capture, self._object_to_tcp, "capture"))
            grasp_loss_supported = supported_retained
            if self._support_filter is not None:
                # The grasp-loss exemption stays on the RAW support sample: a filtered value that still lags
                # above the threshold must not delay grasp_lost (only the dwell/hold/descent use the filter).
                grasp_loss_supported = bool(gates["slot_pose_reached"] and gates["slot_normal_reached"]
                    and gates["jig_support_confirmed"] and alignment_ready and not obs["open"]
                    and self._near(measured_capture, self._object_to_tcp, "capture"))
        if stage in _CARRYING:
            if self._grasp_valid(obs) or grasp_loss_supported:
                self._lost_grasp_since = None
            elif self._lost_grasp_since is None:
                self._lost_grasp_since = now
            elif now - self._lost_grasp_since >= threshold["lost_grasp_grace_s"]:
                return self.fail("grasp_lost")
        if stage in _RELEASED:
            if not gates["hand_open"] or not gates["same_object_contacts_absent"]:
                return self.fail("release_lost")
            if not gates["slot_pose_reached"] or not gates["slot_normal_reached"]:
                return self.fail("placement_displaced_after_release")
        condition, duration = False, threshold["reach_dwell_s"]
        if stage == "prerequisites":
            condition = True
        elif stage == "navigate_source":
            self._readiness = self._upright_readiness(raw, obs, gates)
            condition, duration = self._readiness["valid"], self._preparation_dwell_s
            if self._no_walk is not None:
                # No step can reach a station the stance settled outside of: fail fast.
                failure = self._no_walk.station_failure(self._readiness, READINESS_GATES, now, duration,
                                                        self.navigation_tolerances("source"))
                if failure:
                    return self.fail(failure)
            if self._source_hold_station is not None:
                failure = self._source_hold_station.station_failure(
                    self._readiness, READINESS_GATES, now, duration, self.navigation_tolerances("source"))
                if failure:
                    return self.fail("single_step_source_hold_outside_source_station")
        elif stage in POSTURE_STAGES:
            if stage == "reach_posture":
                # The empty reach must not touch the ring with either hand.
                for arm in ("left", "right"):
                    if _contact_force(raw, arm, self.object_path) is None:
                        return self.fail("reach_posture:contact_sensor_unavailable")
                    if not _contact_absent(raw, arm, self.object_path):
                        return self.fail(f"reach_posture:unexpected_object_contact:{arm}")
            condition, failure = self._posture_condition(raw, obs, stage, gates)
            if failure is not None:
                return self.fail(failure)
            if stage in ("carry_posture", "place_posture"):
                retained = self._grasp_valid(obs)
                self._posture_evidence[stage]["grasp_retained"] = retained
                condition = condition and retained
            elif stage == "rest_posture":
                condition = bool(condition and gates["hand_open"] and gates["same_object_contacts_absent"]
                                 and gates["slot_pose_reached"] and gates["slot_normal_reached"])
            duration = self._posture_path(stage)["settle_dwell_s"]
        elif stage == "handover_posture":
            # Both arms on one designed joint path; the giving hand (the observation arm) keeps the ring.
            condition, failure = self._posture_condition(raw, obs, stage, gates)
            if failure is not None:
                return self.fail(failure)
            retained = self._grasp_valid(obs)
            self._posture_evidence[stage]["grasp_retained"] = retained
            condition = condition and retained
            duration = self._posture_path(stage)["settle_dwell_s"]
        elif stage in ("handover_pregrasp", "handover_approach"):
            # The receiving hand (observation arm) at its goal of the measured ring, open; the giving
            # hand's retention and the hand-hand bound are guarded above.
            condition = obs["open"] and self._near(obs["tcp"], self._tcp_target)
        elif stage == "handover_close":
            condition = (obs["closed"] and obs["hand_settled"] and self.object_path in obs["contacts"]
                         and self._handover_close_pose_valid(obs))
            duration = threshold["contact_dwell_s"]
        elif stage == HANDOVER_ROTATE_STAGE:
            # Opt-in two-hand rotation: the joint path complete, arrived and still with a quiet base, both
            # pinches retained (guarded above with the grace) and the receiving grasp point at the top of the
            # rim. An open-loop path that completed and settled off the top cannot recover: fail closed.
            condition, failure = self._posture_condition(raw, obs, stage, gates)
            if failure is not None:
                return self.fail(failure)
            rotation = self._handover_evidence["rotation"]
            retained = all(self._handover_evidence[hand]["retention"] is not None
                           and self._handover_evidence[hand]["retention"]["valid"] for hand in ("giving", "receiving"))
            self._posture_evidence[stage]["grasp_retained"] = retained
            self._posture_evidence[stage]["receiving_at_top"] = rotation["at_top"]
            self._posture_evidence[stage]["receiving_offset_from_top_rad"] = rotation["receiving_offset_from_top_rad"]
            self._posture_evidence[stage]["ring_rotation_rad"] = rotation["ring_rotation_rad"]
            duration = self._posture_path(stage)["settle_dwell_s"]
            if condition and retained and not rotation["at_top"]:
                if self._rotate_off_top_since is None:
                    self._rotate_off_top_since = now
                elif now - self._rotate_off_top_since >= duration - 1e-10:
                    return self.fail("handover_rotate_receiving_off_top")
            else:
                self._rotate_off_top_since = None
            condition = bool(condition and retained and rotation["at_top"])
        elif stage == "handover_release":
            if obs["contact_sensor_available"] is not True:
                return self.fail("contact_sensor_unavailable")
            giving = obs["hands"][self._giving_arm]
            if self._handover_release_phase == RELEASE_VERIFYING_PHASE:
                # Opt-in pre_release_dwell_s (close second, dwell, open first): the giving jaw stays closed until
                # the receiving bilateral pinch and its capture retention have held continuously for the dwell.
                verified = bool(self._grasp_valid(obs) and obs["hands"][self._receiving_arm]["bilateral"]
                                and giving["bilateral"] and giving["closed"])
                if not verified:
                    self._handover_verify_since = None
                elif self._handover_verify_since is None:
                    self._handover_verify_since = now
                if (self._handover_verify_since is not None
                        and now - self._handover_verify_since >= self._handover["pre_release_dwell_s"] - 1e-10):
                    self._handover_release_phase = "opening"
                    self._handover_events.append({"event": "receiving_pinch_verified", "time_s": now,
                                                  "pre_release_dwell_s": self._handover["pre_release_dwell_s"]})
            if giving["open"] and self._handover_release_phase == "opening":
                self._condition_since = None
                self._handover_release_phase = "separating"
                self._secondary_target = self._giving_release_origin.copy()
                self._secondary_target[:3, 3] += np.asarray(self._handover["release_arm_offset_w"], float)
                self._handover_events.append({"event": "giving_jaw_open", "time_s": now})
            condition = bool(self._handover_release_phase == "separating" and giving["open"]
                             and giving["contact_absent"] and self._near(giving["tcp"], self._secondary_target)
                             and self._grasp_valid(obs))
            duration = self._handover["dwell_s"]
        elif stage == "handover_retreat":
            condition = self._near(obs["hands"][self._giving_arm]["tcp"], self._secondary_target)
        elif stage in ("pregrasp", "approach"):
            # NepesTaskMachine pick semantics, including its optional guarded paths.
            condition = obs["open"] and self._near(obs["tcp"], self._tcp_target)
            if stage == "pregrasp" and "pregrasp_pose_control" in self.task:
                from .pregrasp import pregrasp_contact_failure
                contact = raw.get("contact_evidence", {}).get(self.arm, {}).get(self.object_path, {})
                failure = pregrasp_contact_failure(contact, self.arm, self.object_path)
                if failure:
                    return self.fail("pregrasp:" + failure)
                path = raw.get("pregrasp_reference", {})
                if not isinstance(path, Mapping):
                    return self.fail("pregrasp:invalid_reference_evidence")
                condition = (condition and path.get("status") == "running" and path.get("stage_id") == "pregrasp"
                             and path.get("arm") == self.arm and path.get("object_path") == self.object_path
                             and path.get("waypoints_complete") is True)
            if stage == "pregrasp" and "pregrasp_position_m" in self.task.get("approach_axis_alignment", {}):
                # Opt-in: the (pad-tip) error along the pinch axis from the final pregrasp pose, never a waypoint.
                condition = condition and self._approach_axis_alignment(obs, "pregrasp")["satisfied"]
            if stage == "approach" and "guarded_approach" in self.task:
                settings = self.task["guarded_approach"]
                distance, angle = _errors(obs["tcp"], self._tcp_target)
                contact = raw.get("contact_evidence", {}).get(self.arm, {}).get(self.object_path, {})
                if self._pressed_key is not None:
                    # Opt-in (runtime_v26), as NepesTaskMachine: the finger-axis/lateral split replaces the distance
                    # bound; the dwell and mean tip force are checked with the dwell below.
                    keyed = self._pressed_key.measure(obs["tcp"], self._tcp_target, contact)
                    condition = (condition and contact.get("available") is True and keyed
                                 and angle <= settings.get("rotation_tolerance_rad", .025))
                    duration = max(duration, self._pressed_key.dwell_s)
                else:
                    # Opt-in completion_position_m replaces lateral_tolerance_m here (a pin key may stop the hand short).
                    condition = (condition and contact.get("available") is True
                                 and distance <= settings.get("completion_position_m",
                                                              settings.get("lateral_tolerance_m", .003))
                                 and angle <= settings.get("rotation_tolerance_rad", .025))
            if stage == "approach" and "approach_axis_alignment" in self.task:
                # Opt-in: the TCP error along the pinch axis too (axis turned by the pregrasp-entry CRS pose
                # that set the target, itself held within the source-anchor limits).
                condition = condition and self._approach_axis_alignment(obs)["satisfied"]
        elif stage == "close":
            condition = (obs["closed"] and obs["hand_settled"] and self.object_path in obs["contacts"]
                         and self._close_pose_valid(obs))
            duration = threshold["contact_dwell_s"]
        elif stage == "lift":
            lifted = obj[2, 3] - self._source_object[2, 3] >= cfg["minimum_lift_m"]
            condition = lifted and self._grasp_valid(obs) and self._tracking_goal_near(obs)
        elif stage == SINGLE_STEP_STAGE and self._single_step is not None:
            # Opt-in single step: pre-step posture, goal latch, one-shot brake and the settled window.
            condition, failure = self._single_step_condition(raw, obs, gates)
            if failure is not None:
                return self.fail(failure)
        elif stage == "navigate_target" and self._no_walk is not None:
            # No side-step: the zero-command stance completes once the pelvis hold and the grasp are
            # retained for the positive reach dwell (the hold evidence is kept). The dwell keeps the
            # stage active for at least two control steps (10 physics ticks), longer than the 8-9 tick
            # 30 fps frame spacing, so the recording always holds a side_step frame.
            self._base_near(obs, "target")
            condition = bool(self._stationary_pelvis_evidence["valid"]) and self._grasp_valid(obs)
        elif stage == "navigate_target":
            condition = self._base_near(obs, "target") and self._grasp_valid(obs)
        elif stage == "transfer":
            transit = self._transit_tolerance()
            hand_path = transit is not None and self._transit_hand_path(self._object_path_index)
            self._object_feedback_hold_reason = "transfer_transit_hand_path" if hand_path else None
            if self._transit_arrival is not None:  # undeclared profiles keep their recorded metrics byte-identical
                self._metrics["transfer_transit_arrival"] = {
                "declared": True, "active": transit is not None,
                "mode": (self._transit_arrival or {}).get("mode", "object"), "hand_path": hand_path,
                "object_feedback_suppressed": hand_path,
                "waypoint_index": self._object_path_index, "waypoint_count": len(self._object_path_goals),
                "position_tolerance_m": transit[0] if transit else self.config["thresholds"]["tcp_position_m"],
                "rotation_tolerance_rad": transit[1] if transit else self.config["thresholds"]["tcp_rotation_rad"]}
            if hand_path:
                # The ring may pivot in the pads while the hand follows its planned path, but only within a
                # declared bound on its centre's displacement relative to the hand (hand frame, acquisition).
                deviation = float(np.linalg.norm((np.linalg.inv(obs["tcp"]) @ obj)[:3, 3]
                                                 - np.linalg.inv(self._object_to_tcp)[:3, 3]))
                bound = self._transit_arrival.get("max_object_deviation_m", TRANSFER_TRANSIT_DEVIATION_DEFAULT_M)
                self._metrics["transfer_transit_arrival"].update(object_deviation_m=deviation, max_object_deviation_m=bound)
                if deviation > bound:
                    return self.fail("transfer_transit_object_deviation")
                condition = self._hand_path_near(obs, obj, transit) and self._grasp_valid(obs)
            else:
                condition = self._tracking_goal_near(obs, tolerance=transit) and self._grasp_valid(obs)
        elif stage == "place":
            condition = (supported_retained and gates["object_com_stationary"] and gates["object_pose_stationary"]
                         and opening_supported)
            duration = self.task["placement_support"]["dwell_s"]
            if self._placement_hold is not None:
                # A load-only dropout may preserve the hand's stop reference.
                # It does not satisfy supported_retained or the opening dwell.
                valid = bool(gates["slot_pose_reached"] and gates["slot_normal_reached"]
                    and gates["jig_anchor_retained"] and alignment_ready
                    and self._metrics["support_evidence"]["available"]
                    and obs["contact_sensor_available"] is True and not obs["open"]
                    and self._near(np.linalg.inv(obj) @ obs["tcp"], self._object_to_tcp, "capture")
                    and (self._support_active or self._grasp_valid(obs)))
                latch_now = None
                if self._placement_hold.latch_dwell_s is not None:
                    # Declared C2 only (runtime_v26 review fix): a pending latch stops pressing at once on a raw JIG
                    # load of 1.2 m g or a 1 mm JIG drift since the place entry (placement_hold.latch_now_reason).
                    from .placement_hold import latch_now_reason
                    entry = self._placement_jig_reference
                    latch_now = latch_now_reason(self._metrics["support_evidence"], obs["objects"][self.jig_path],
                                                 None if entry is None else pose_to_matrix(entry["jig_pose_w"]))
                held = self._placement_hold.update(now, loaded=self._support_active,
                    valid=valid, measured_tcp=obs["tcp"], latch_now=latch_now)
                self._support_hold_active = held is not None
                self._support_latch_pending = self._placement_hold.latch_pending
                self._supported_release_tcp = held
                self._tcp_target = (held.copy() if held is not None
                                    else self._target_object @ self._control_object_to_tcp)
                self._metrics["placement_cartesian_hold"] = deepcopy(self._placement_hold.evidence)
            elif self._support_active:
                if self._supported_release_tcp is None:
                    self._supported_release_tcp = obs["tcp"].copy()
                self._tcp_target = self._supported_release_tcp.copy()
            elif self._supported_release_tcp is not None:
                self._supported_release_tcp = None
                self._tcp_target = self._target_object @ self._control_object_to_tcp
        elif stage == "release":
            if not gates["slot_pose_reached"] or not gates["slot_normal_reached"]:
                return self.fail("placement_displaced_during_release")
            if obs["contact_sensor_available"] is not True:
                return self.fail("contact_sensor_unavailable")
            if obs["open"] and self._release_phase == "opening":
                self._condition_since = None
                if "release_clearance_offset_w" in self.task:
                    self._release_phase = "clearing"
                    self._tcp_target = self._release_origin.copy()
                    self._tcp_target[:3, 3] += self.task["release_clearance_offset_w"]
                else:
                    self._release_phase = "separating"
                    fraction = cfg["release_separation_m"] / np.linalg.norm(self.task["retreat_offset_w"])
                    self._tcp_target = self._retreat_goal(fraction)
                self._release_subphase_events.append({"phase": self._release_phase, "time_s": now})
            if self._release_phase == "clearing":
                clearance = self._metrics["release_clearance"]
                clearance["gates"] = {key: gates[key] for key in ("hand_open", "same_object_contacts_absent",
                    "jig_support_confirmed", "slot_pose_reached", "slot_normal_reached")}
                clearance["gates"].update(
                    projected_clearance_reached=clearance["projected_distance_m"] >= clearance["required_projected_m"],
                    tcp_goal_reached=self._near(obs["tcp"], self._tcp_target))
                condition = all(clearance["gates"].values())
            else:
                cleared_origin = self._release_origin[:3, 3] + np.asarray(
                    self.task.get("release_clearance_offset_w", [0., 0., 0.]))
                separation_distance = float(np.linalg.norm(obs["tcp"][:3, 3] - cleared_origin))
                condition = (self._release_phase == "separating" and gates["hand_open"]
                    and gates["same_object_contacts_absent"] and gates["jig_support_confirmed"]
                    and separation_distance >= cfg["release_separation_m"] - .003
                    and self._near(obs["tcp"], self._tcp_target))
            duration = threshold["contact_dwell_s"]
        elif stage == "retreat":
            condition = self._near(obs["tcp"], self._tcp_target)
            if self._retreat_index == 0:
                condition = condition and gates["retreat_clearance"]
            if self._intermediate_dwell_s is not None:
                duration = self._waypoint_arrival(stage)[1]
        elif stage == "settle":
            condition, duration = all(gates.values()), cfg["settle_dwell_s"]
        if self._object_path_goals:
            # Intermediate points require measured payload arrival, not only a
            # commanded tool pose; the final lift point also clears the cart slots.
            # An intermediate transfer point of a pivoting symmetric payload may use the declared transit tolerance;
            # in hand_path mode the HAND arrives (the ring hangs where it pivots to) and the object check waits
            # for the final waypoints.
            transit = self._transit_tolerance() if stage == "transfer" else None
            arrival = (self._waypoint_arrival(stage)
                       if self._continuity or self._intermediate_dwell_s is not None else None)
            if transit is not None and self._transit_hand_path(self._object_path_index):
                condition = self._hand_path_near(obs, obj, transit) and self._grasp_valid(obs)
            elif arrival is not None and arrival[0] == "pass_through":
                # Opt-in blended pass-through: the measured payload passes the waypoint within the pass tolerances
                # (object thresholds unchanged), with the physical grasp retained, and the stage moves on at once.
                settings = self._continuity[stage]
                condition = (self._near(obj, self._object_path_goals[self._object_path_index], "object")
                             and self._tracking_goal_near(
                                 obs, tolerance=(settings["pass_position_m"], settings["pass_rotation_rad"]))
                             and self._grasp_valid(obs))
            else:
                condition = (self._near(obj, self._object_path_goals[self._object_path_index], "object")
                             and self._tracking_goal_near(obs, tolerance=transit) and self._grasp_valid(obs))
            if arrival is not None:
                duration = arrival[1]
            if stage == "lift" and self._object_path_index == len(self._object_path_goals) - 1:
                condition = condition and obj[2, 3] - self._source_object[2, 3] >= cfg["minimum_lift_m"]
        if lateral_errors is not None:
            # Opt-in: the final transfer point also needs measured lateral alignment with the
            # acceptance slot pose for its own dwell; in place, a hysteretic advance-only hold.
            # A pending sustained-load latch (runtime_v26 C2) does not own the hand goal yet: the descent pause stays.
            supported = bool((self._support_active and not self._support_latch_pending) or self._support_hold_active)
            hold = self._aligned_descent.update(stage, now, lateral_errors, supported=supported)
            self._object_advance_hold_reason = hold
            self._metrics["place_aligned_descent"] = deepcopy(self._aligned_descent.evidence)
            if stage == "transfer" and (not self._object_path_goals
                                        or self._object_path_index == len(self._object_path_goals) - 1):
                condition = condition and self._aligned_descent.transfer_aligned(lateral_errors)
                duration = max(duration, self._aligned_descent.settings["transfer_exit_dwell_s"])
        else:
            self._object_advance_hold_reason = None
        if stage != "transfer":
            self._object_feedback_hold_reason = None
        posture = stage_posture(self.task, stage, self._object_path_index)
        if posture is not None:
            # Generic measured stage-posture arrival, as NepesTaskMachine gates it.
            goal = posture["joint_positions"]
            joints = _joint_values(raw.get("joint_positions"), goal)
            condition = condition and joints is not None and all(
                abs(joints[name] - value) <= posture.get("tolerance_rad", .03) for name, value in goal.items())
        passed = self._dwell(bool(condition), now, duration)
        if self._pressed_key is not None and stage == "approach" and "guarded_approach" in self.task:
            # Opt-in (runtime_v26): the same continuous run, plus the mean tip force over its trailing window.
            passed = self._pressed_key.complete(now, bool(condition), passed)
        dwell = 0. if self._condition_since is None else now - self._condition_since
        if stage == "navigate_source":
            self._readiness["continuous_dwell_s"] = dwell
        elif stage in POSTURE_STAGES or stage in HANDOVER_JOINT_PATH_STAGES:
            self._posture_evidence[stage]["continuous_dwell_s"] = dwell
        elif stage == "settle":
            self._metrics["continuous_dwell_s"] = dwell
        elif stage == "release" and self._release_phase == "clearing":
            self._metrics["release_clearance"]["continuous_dwell_s"] = dwell
        if passed and stage == "release" and self._release_phase == "clearing":
            self._clearance_completion = {"time_s": now, "continuous_dwell_s": dwell,
                "projected_distance_m": self._metrics["release_clearance"]["projected_distance_m"],
                "measured_tcp_pose_w": matrix_to_pose(obs["tcp"]),
                "hand_open": gates["hand_open"], "same_object_contacts_absent": gates["same_object_contacts_absent"],
                "jig_support_confirmed": gates["jig_support_confirmed"],
                "slot_pose_reached": gates["slot_pose_reached"], "slot_normal_reached": gates["slot_normal_reached"]}
            self._release_phase = "separating"
            self._release_subphase_events.append({"phase": "separating", "time_s": now})
            self._condition_since = None
            fraction = cfg["release_separation_m"] / np.linalg.norm(self.task["retreat_offset_w"])
            self._tcp_target = self._retreat_goal(fraction)
            self._metrics["release_clearance"]["completion"] = deepcopy(self._clearance_completion)
            self._metrics["release_clearance"]["current_phase"] = "separating"
            self._metrics["release_subphase_events"] = deepcopy(self._release_subphase_events)
            return self._command()
        if passed:
            self.reason = None
            if self._object_path_index + 1 < len(self._object_path_goals):
                was_hand_path = stage == "transfer" and self._transit_hand_path(self._object_path_index)
                self._object_path_index += 1
                self._condition_since = None
                if was_hand_path and not self._transit_hand_path(self._object_path_index):
                    # First final waypoint: re-capture the measured (spin-aligned) grasp so the object-centred
                    # approach starts from where the ring actually hangs in the pads.
                    self._set_control_grasp(obj, obs["tcp"])
                    self._object_feedback_hold_reason = None
                    self.history.append({"time_s": now, "stage_id": stage, "status": "transit_hand_path_ended",
                                         "control_object_to_tcp": matrix_to_pose(self._control_object_to_tcp)})
                self._tcp_target = self._object_path_goals[self._object_path_index] @ self._control_object_to_tcp
                self.history.append({"time_s": now, "stage_id": stage, "status": "waypoint_reached",
                                     "next_waypoint_index": self._object_path_index})
                if self._continuity_plan_stage == stage and self._continuity_plan[self._object_path_index - 1]["eligible"]:
                    self.history[-1]["arrival"] = "pass_through"
                return self._command()
            if stage == "retreat" and self._retreat_index + 1 < len(self._retreat_offsets):
                self._retreat_index += 1
                self._condition_since = None
                self._tcp_target = self._retreat_waypoint_goal(self._retreat_index)
                self.history.append({"time_s": now, "stage_id": stage, "status": "waypoint_reached",
                                     "next_waypoint_index": self._retreat_index})
                return self._command()
            if (stage == "handover_retreat"
                    and self._handover_retreat_index + 1 < len(self._handover["release_retreat_waypoints_w"])):
                self._handover_retreat_index += 1
                self._condition_since = None
                self._secondary_target = self._handover_retreat_goal(self._handover_retreat_index)
                self.history.append({"time_s": now, "stage_id": stage, "status": "waypoint_reached",
                                     "next_waypoint_index": self._handover_retreat_index})
                return self._command()
            if self._stage_index == len(self.stages) - 1:
                self.status = "succeeded"
                self.history.append({"time_s": now, "stage_id": stage, "status": self.status})
            else:
                if stage == "navigate_source":
                    self._recording_ready = True
                next_stage = self.stages[self._stage_index + 1]
                transport_failure = self._check_payload_transport(
                    obs, next_stage, at_walk_entry=next_stage == "navigate_target")
                if transport_failure:
                    return self.fail(transport_failure)
                self._enter(self._stage_index + 1, obs)
                # The recording-boundary observation must pass before the first
                # reach command can be applied. The anchor check happens once
                # in _enter; every later recorded row is checked above.
                if self._stationary_feet_evidence is not None and not self._stationary_feet_evidence["valid"]:
                    return self.fail("stationary_feet_hold:" + self._stationary_feet_evidence["reason"])
        return self._command()

    def _hand_path_near(self, obs, obj, tolerance):
        """Hand arrival at its planned target during hand-path transit; with a declared ring symmetry the hand's
        rotation about the MEASURED ring axis is not an error (the route leaves it partly unreachable at its joint
        command boxes, and it only moves the pinch along the rim of a ring that hangs freely in the pads)."""
        from .task_control import rotation_error
        from .object_symmetry import remove_axis_component
        target, tcp = self._tcp_target, obs["tcp"]
        position = float(np.linalg.norm(tcp[:3, 3] - target[:3, 3]))
        rotation_vector = rotation_error(target[:3, :3], tcp[:3, :3])
        if self._symmetry_axis is not None:
            rotation_vector = remove_axis_component(rotation_vector, obj[:3, :3] @ self._symmetry_axis)
        rotation = float(np.linalg.norm(rotation_vector))
        if "transfer_transit_arrival" in self._metrics:
            self._metrics["transfer_transit_arrival"].update(hand_position_error_m=position, hand_rotation_error_rad=rotation)
        return position <= tolerance[0] and rotation <= tolerance[1]

    def _transit_tolerance(self):
        """(position_m, rotation_rad) for an intermediate transfer waypoint under transfer_transit_arrival, else None.

        The last ``final_waypoints`` object waypoints (the insertion approach) keep the strict tcp thresholds.
        """
        settings = self._transit_arrival
        if settings is None or not self._object_path_goals:
            return None
        remaining = len(self._object_path_goals) - 1 - self._object_path_index
        if remaining < settings["final_waypoints"]:
            return None
        return (settings["position_m"], settings["rotation_rad"])

    def _continuity_evidence(self):
        """The machine side of the opt-in blended pass-through for the current stage, or None."""
        stage = self.stage_id
        settings = self._continuity.get(stage)
        if self.status != "running" or settings is None or self._continuity_plan_stage != stage:
            return None
        from .reference_continuity import MODE, SCHEMA_VERSION, plan_evidence
        index = self._object_path_index
        return {"schema_version": SCHEMA_VERSION, "mode": MODE, "stage_id": stage,
                "waypoint_index": index, "waypoint_count": len(self._continuity_plan),
                "arrival": "pass_through" if self._continuity_plan[index]["eligible"] else "strict",
                "final_waypoints": settings["final_waypoints"],
                "pass_position_m": settings["pass_position_m"], "pass_rotation_rad": settings["pass_rotation_rad"],
                **plan_evidence(self._continuity_plan)}

    def _command(self):
        command = NepesTaskMachine._command(self)
        if self._no_walk is not None and self._metrics is not None:
            # Running rows carry the compact report; the terminal one the per-stage history.
            self._metrics["no_walk_stance"] = self._no_walk.evidence(full=self.status != "running")
        if self._single_step is not None and self._metrics is not None:
            self._metrics["single_step"] = self._single_step.evidence()
        if self.status == "running" and self.stage_id == "prerequisites":
            command["hand_command"] = "open"
        command.update(stationary_phase=STAGE_PHASES[self.stage_id], goal_id=GOAL_ID,
            dataset_task_id=GOAL_ID, legacy_task_id=self.task_id, episode_scope="stationary_task",
            full_task_completed=self.status == "succeeded", legacy_full_task_completed=False,
            pickup_demonstrated=True, recording_start_stage=RECORDING_START_STAGE,
            stationary_metrics=deepcopy(self._metrics), preparation_readiness=deepcopy(self._readiness),
            posture_evidence=deepcopy(self._posture_evidence),
            release_phase=self._release_phase if self.stage_id == "release" else None,
            retreat_waypoint_index=self._retreat_index if self.stage_id == "retreat" else None,
            retreat_waypoint_count=len(self._retreat_offsets))
        if self._aligned_descent is not None:
            command["object_advance_hold_reason"] = (self._object_advance_hold_reason
                if self.status == "running" and self.stage_id == "place" else None)
        if self._placement_hold is not None and self._placement_hold.latch_dwell_s is not None:
            # Declared only (runtime_v26 C2): support is confirmed but the sustained-load latch has not yet
            # latched, so the runtime keeps the nominal descent reference running.
            command["placement_support_latch_pending"] = bool(
                self._support_latch_pending and self.status == "running" and self.stage_id == "place")
        if self._transit_arrival is not None:  # declared transit policy only: undeclared profiles keep their command schema
            command["object_feedback_hold_reason"] = (self._object_feedback_hold_reason
                                                      if self.status == "running" and self.stage_id == "transfer" else None)
        if self._continuity:  # declared blended pass-through only (runtime_v26)
            command["reference_continuity"] = self._continuity_evidence()
        if self._intermediate_dwell_s is not None:  # declared intermediate dwell only (runtime_v26)
            arrival = self._waypoint_arrival(self.stage_id) if self.status == "running" else None
            command["waypoint_arrival_dwell"] = None if arrival is None else {
                "stage_id": self.stage_id, "waypoint_index": arrival[2], "waypoint_count": arrival[3],
                "arrival": arrival[0], "required_dwell_s": arrival[1],
                "intermediate_dwell_s": float(self._intermediate_dwell_s),
                "reach_dwell_s": float(self.config["thresholds"]["reach_dwell_s"])}
        if any(is_applied_hold(path) for path in self.task["posture_paths"].values()):
            command["posture_hold_references"] = deepcopy(self._posture_hold_references)
        if self._transport_constraints is not None:
            command["payload_transport"] = deepcopy(self._transport_evidence)
            command["walk_entry_payload_transport"] = deepcopy(self._walk_entry_transport_evidence)
        if self._stationary_pelvis_settings is not None:
            command["stationary_pelvis_hold"] = deepcopy(self._stationary_pelvis_evidence)
            if self._stationary_pelvis_anchor is not None:
                command["base_arrival_braking"] = True
        if self._no_walk is not None:
            # Every stage of a no-walk episode, the unrecorded preparation included.
            command["base_arrival_braking"] = True
        if self._source_hold_station is not None and self.stage_id in PREPARATION_STAGES:
            command["base_arrival_braking"] = True
        if self._stationary_feet is not None:
            command["stationary_feet_hold"] = deepcopy(self._stationary_feet_evidence)
        if self._single_step is not None:
            self._single_step_command(command)
        if self._handover is not None:
            self._handover_command(command)
        return command

    def _single_step_command(self, command):
        """Command fields of the opt-in single step: its evidence and, in navigate_target, the base goal.

        Before the goal latch navigate_target is a joint-posture stage (the pre-step path, no base goal);
        from the latch it navigates to the latched goal, with arrival braking only once the one-shot brake
        has latched. Every other stage keeps the stationary pelvis hold's command fields.
        """
        step = self._single_step
        command["single_step"] = step.evidence()
        if self.status != "running" or self.stage_id != SINGLE_STEP_STAGE:
            return
        if step.latched:
            command.update(motion_mode="navigate_base", base_goal_pose_w=step.goal_pose_w,
                           base_arrival_braking=step.braking)
        else:
            command.update(motion_mode="joint_posture", posture_stage=SINGLE_STEP_STAGE, base_goal_pose_w=None,
                           base_arrival_braking=False)

    def graph(self):
        graph = NepesTaskMachine.graph(self)
        graph.update(episode_scope="stationary_task", stationary_task=deepcopy(self.goal_settings),
            goal_id=GOAL_ID, dataset_task_id=GOAL_ID, legacy_task_id=self.task_id,
            legacy_full_task_completed=False, pickup_demonstrated=True,
            initial_condition=("upright HugWBC stance at the source station; no pre-posed arm, held object, "
                               "attachment or prepared snapshot"),
            placement_contract_version=PLACEMENT_CONTRACT_VERSION,
            recording_start_stage=RECORDING_START_STAGE, unrecorded_stages=list(PREPARATION_STAGES),
            recorded_phases=[STAGE_PHASES[stage] for stage in self.stages[len(PREPARATION_STAGES):]])
        if self._handover is not None:
            # Opt-in handover: the stage insertion, the arm switch and the kinematic hand-hand proxy.
            graph["handover"] = {**handover_contract(self.task),
                                 "designed_hand_tcp_distance_m": self._handover["designed_hand_tcp_distance_m"],
                                 "inserted_after": "carry_posture", "dataset_phase_ids": {
                                     stage: STAGES.index(stage) for stage in handover_stages(self.task)}}
        gates = pick_gate_conditions(self.task)
        if self._support_filter is not None:
            gates["place"] = gates.get("place", "") + PLACE_FILTER_CONDITION
        for stage in graph["stages"]:
            stage.update(success_condition=_SUCCESS_CONDITIONS[stage["id"]] + gates.get(stage["id"], ""),
                         stationary_phase=STAGE_PHASES[stage["id"]],
                         dataset_phase_id=STAGES.index(stage["id"]),
                         recorded=stage["id"] not in PREPARATION_STAGES)
        for stage in graph["stages"]:
            if is_applied_hold(self._posture_path(stage["id"])):
                stage["success_condition"] = ("entry's last applied 19-joint targets held unchanged; "
                    "existing measured arrival, joint velocity, command tracking, base quiet and "
                    "physical grasp conditions continuously satisfied for the existing posture dwell")
        if self._no_walk is not None:
            # Opt-in single stance (no_walk_stance): no stage steps, so no station is walked to.
            for stage in graph["stages"]:
                if stage["id"] == "navigate_target":
                    stage["success_condition"] = (
                        "no_walk: exactly zero base command (no side-step); the stationary pelvis hold valid and "
                        "the physical grasp retained continuously for the reach dwell")
                elif stage["id"] == "navigate_source":
                    stage["success_condition"] += (
                        "; no_walk: exactly zero base command, and a stance outside the source station tolerances "
                        "while every other readiness gate holds fails after the preparation dwell "
                        "(no_walk_stance_outside_source_station)")
        if self._single_step is not None:
            # Opt-in single step (single_step.py): one short step to a goal latched from the measured stance.
            graph["single_step"] = single_step_contract(self.task)
            for stage in graph["stages"]:
                if stage["id"] == SINGLE_STEP_STAGE:
                    stage["success_condition"] = (
                        "single_step: the pre-step posture path (when declared) complete under the posture-stage "
                        "rules with the grasp retained and an exactly zero base command; the goal latched from the "
                        "measured torso and pelvis (torso + pelvis-heading offset, torso yaw); one-shot braking at "
                        "the first torso entry within brake_radius_m or once the commanded travel reaches the "
                        "offset length, then an exactly zero base command; the torso slow for settle_dwell_s with "
                        "the pelvis displacement (latched pelvis heading frame) inside window_m, the pelvis yaw "
                        "change within max_yaw_change_rad and the physical grasp retained for the reach dwell "
                        "(settled outside: single_step_outside_window); the stationary pelvis hold re-anchors at "
                        "the exit")
        for edge in graph["failure_edges"]:
            edge["reasons"] = list(_FAILURES) + ([SINGLE_STEP_OUTSIDE_WINDOW] if self._single_step is not None else [])
        return graph


def _pose_or_none(value):
    try:
        return pose_to_matrix(value)
    except (ValueError, TypeError):
        return None
