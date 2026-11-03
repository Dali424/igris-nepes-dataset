"""Bounded collection of independently initialized, measured stationary goals.

Only committed recordings from a normally completed HugWBC run are admitted.
Historical placement tasks and transport primitives never count toward these
quotas. Raw mode preserves native run directories. Opt-in compact mode verifies
an official video archive before removing PNGs and applying trace retention.
"""
from __future__ import annotations

import errno
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np

from .collection import digest, read_json, run_child, write_json
from .dataset import validate_camera_sizes, validate_declared_camera_layout, validate_physics_clock
from .latest_robot import (DEFAULT_GRIPPER_VARIANT, require_gripper_package, validate_gripper_variant,
                           verify_gripper_manifest)
from .handover import (HANDOVER_PHASES, HANDOVER_SCHEMA, HANDOVER_STAGE_SET, RELEASE_VERIFYING_PHASE, handover_contract,
                       handover_events, release_phases, rotation_measurement, rotation_settings)
from .nepes_tasks import HANDOVER_ROTATE_STAGE, STAGES, handover_stages
from .single_step import (ACCEPTED as SINGLE_STEP_ACCEPTED, REJECTED as SINGLE_STEP_REJECTED, SINGLE_STEP_SCHEMA,
                          SINGLE_STEP_STAGE, SingleStep, arrival_limits, planar_command, single_step_contract,
                          validate_single_step)
from .stationary_catalog import (
    DEFAULT_CATALOG, STATIONARY_SCOPE, config_digest, default_stationary_config,
    load_stationary_config, recorded_inter_station_transport, resolve_stationary_task, stationary_max_task_seconds,
    stationary_spec_catalog, task_catalog, validate_stationary_spec,
)


COLLECTION_SCHEMA = "igris_c_nepes_stationary_collection_v1"
ROOT = Path(__file__).resolve().parents[3]
_CONTROLLER = {"hugwbc_enabled": True, "pelvis_fixed": False,
               "robot_gravity_enabled": True, "grasp_attachment": "none",
               "inter_station_transport": False}
_HOLD_BOUNDS = {"max_object_pose_linear_m_s": .015, "max_hold_position_drift_m": .005,
                "max_hold_rotation_drift_rad": .05}
_PLACEMENT_PHASES = ("transfer", "place", "release", "retreat", "settle")
_PLACEMENT_GATES = frozenset(("slot_pose_reached", "slot_normal_reached", "jig_anchor_retained",
    "jig_support_confirmed", "hand_open", "same_object_contacts_absent", "retreat_clearance",
    "object_com_stationary", "object_pose_stationary", "base_stationary"))
_PLACEMENT_BOUNDS = {"max_object_pose_linear_m_s": .015, "max_hand_contact_force_n": .05}
# Catalog-v2 task 1 (goal pickup_load_jig), pinned here independently of the
# machine. Frames carry the dataset label as phase_name and the stage's
# nepes_tasks.STAGES index as phase_id; the boundary names the stage id.
_PICKUP_UNRECORDED = (("prerequisites", "preparation"), ("navigate_source", "upright_ready"))
_PICKUP_RECORDED = (
    ("reach_posture", "reach"), ("pregrasp", "pregrasp"), ("approach", "approach"), ("close", "grasp"),
    ("lift", "lift"), ("carry_posture", "carry"), ("navigate_target", "side_step"),
    ("place_posture", "align_outside_jig"), ("transfer", "insert_under_handle"), ("place", "lower_into_slot"),
    ("release", "release"), ("retreat", "retreat"), ("rest_posture", "rest"), ("settle", "settle"))
_PICKUP_LABELS = dict(_PICKUP_UNRECORDED + _PICKUP_RECORDED)
_PICKUP_STAGES = tuple(_PICKUP_LABELS)
_PICKUP_RECORDED_STAGES = tuple(stage for stage, _ in _PICKUP_RECORDED)
_PICKUP_RECORDED_PHASES = tuple(label for _, label in _PICKUP_RECORDED)
_PICKUP_PREPARATION_NAMES = frozenset(("prerequisites", "navigate_source", "station_alignment",
                                       "preparation", "upright_ready"))
_PICKUP_CONTRACT_KEYS = frozenset(("capture_time_s", "lift_m", "minimum_lift_m"))
_UPRIGHT_SPEEDS = (("object_com_linear_speed_m_s", "max_object_com_linear_m_s"),
                   ("object_com_angular_speed_rad_s", "max_object_com_angular_rad_s"),
                   ("object_pose_linear_speed_m_s", "max_object_pose_linear_m_s"),
                   ("object_pose_angular_speed_rad_s", "max_object_pose_angular_rad_s"),
                   ("base_linear_speed_m_s", "max_base_linear_m_s"),
                   ("base_yaw_speed_rad_s", "max_base_yaw_rad_s"))
_UPRIGHT_GATES = ("station_reached", "upright_posture_reached", "upright_command_retained", "pelvis_upright",
                  "hand_open", "same_object_contacts_absent", "sensors_available", "object_quiet", "base_quiet",
                  "source_object_retained")
_UPRIGHT_MACHINE_KEYS = frozenset((
    "valid", *_UPRIGHT_GATES, "upright_reference", "upright_joint_names", "max_upright_joint_error_rad",
    "max_upright_error_joint", "upright_command_targets", "max_upright_command_change_rad", "pelvis_tilt_rad",
    "hand_contact_force_n", "base_station_position_error_m", "base_station_yaw_error_rad",
    "source_object_anchor_pose_w", "source_object_position_error_m", "source_object_rotation_error_rad",
    *(key for key, _ in _UPRIGHT_SPEEDS), "required_dwell_s", "continuous_dwell_s", "measurement_time_s", "limits"))
_UPRIGHT_BOUNDARY_KEYS = frozenset((
    "valid", "machine_readiness", "machine_readiness_valid", "continuous_dwell_s", "required_dwell_s",
    "hand_open", "same_object_contacts_absent", "sensors_available", *(key for key, _ in _UPRIGHT_SPEEDS),
    "contact_evidence", "limits"))
_UPRIGHT_RECORD_KEYS = frozenset(("robot_root_pose_w", "joint_overrides_applied", "head_gaze", "scope",
                                  "demonstration_pose_write", "upright_joint_targets"))
# The upright-start posture command is never changed before recording.
_UPRIGHT_COMMAND_TOLERANCE_RAD = 1e-9
# The CRS keeps its initialized (first measured) pose until the jaws close on it.
_PICKUP_SOURCE_ANCHORED = frozenset(("prerequisites", "navigate_source", "reach_posture", "pregrasp", "approach"))
_SOURCE_ANCHOR_BOUNDS = {"max_source_object_position_error_m": .005, "max_source_object_rotation_error_rad": .05}
_UPRIGHT_EPISODE_SCOPE = "episode_boundary_before_physics_settle_and_recording"
_UPRIGHT_STARTUP_SCOPE = "before_startup_physics_settle"
_NECK_JOINTS = frozenset(("Joint_Neck_Yaw", "Joint_Neck_Pitch"))
# Opt-in handover (handover.py): six recorded stages inserted after carry_posture (seven with the two-hand
# rotation handover.rotation_path, which sits between handover_close and handover_release), pinned here
# independently of the machine; a profile without the block keeps the sixteen-stage contract.
def _handover_recorded(task):
    return tuple((stage, HANDOVER_PHASES[stage]) for stage in handover_stages(task))


def _pickup_recorded(task):
    """The recorded (stage, label) pairs of one profile; the handover stages only when it opts in."""
    if "handover" not in task:
        return _PICKUP_RECORDED
    index = _PICKUP_RECORDED_STAGES.index("carry_posture") + 1
    return (*_PICKUP_RECORDED[:index], *_handover_recorded(task), *_PICKUP_RECORDED[index:])


def _pickup_labels(task):
    return dict(_PICKUP_UNRECORDED + _pickup_recorded(task))


def _algorithm_fingerprints():
    package = Path(__file__).resolve().parent
    paths = [*sorted(package.glob("*.py")), ROOT / "scripts/nepes/run_tasks.py",
             ROOT / "scripts/nepes/run_stationary.py"]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def _capture_options(*, capture_render_mode="legacy", image_writer_workers=0,
                     image_writer_queue=8, profile_throughput=False):
    if capture_render_mode not in ("legacy", "capture_only"):
        raise ValueError("capture_render_mode must be legacy or capture_only")
    if type(image_writer_workers) is not int or image_writer_workers < 0:
        raise ValueError("image_writer_workers must be a nonnegative integer")
    if type(image_writer_queue) is not int or image_writer_queue < 1:
        raise ValueError("image_writer_queue must be a positive integer")
    if type(profile_throughput) is not bool:
        raise ValueError("profile_throughput must be a boolean")
    return {"capture_render_mode": capture_render_mode, "image_writer_workers": image_writer_workers,
            "image_writer_queue": image_writer_queue, "profile_throughput": profile_throughput}


def build_plan(*, config=None, isaac_python, task=1, successes=1,
               max_attempts=3, seed=0, max_wall_seconds=3600., catalog=None, gripper_variant=None,
               capture_render_mode="legacy", image_writer_workers=0, image_writer_queue=8,
               profile_throughput=False, storage_mode="raw", physics_trace="gzip",
               failed_physics_trace="gzip", min_free_gib=20., max_consecutive_failures=3):
    """Resolve a bounded collection contract; this function never writes files.

    ``catalog`` names the task numbering (v1 or v2). When omitted, an explicit
    profile's own schema selects it, and the default profile uses catalog v2.
    ``gripper_variant`` is an assertion only: datasets use the jaw model the
    profile pins (absent = authored_v0), so the config digest covers it.
    Capture settings are part of the collection plan digest. These attempts
    always record headlessly, including the capture_only rendering option.
    """
    capture_options = _capture_options(capture_render_mode=capture_render_mode,
        image_writer_workers=image_writer_workers, image_writer_queue=image_writer_queue,
        profile_throughput=profile_throughput)
    from .compact_collection import storage_options
    storage = storage_options(storage_mode=storage_mode, physics_trace=physics_trace,
                              failed_physics_trace=failed_physics_trace, min_free_gib=min_free_gib,
                              max_consecutive_failures=max_consecutive_failures)
    if gripper_variant is not None:
        validate_gripper_variant(gripper_variant)
    for key, value in (("successes", successes), ("max_attempts", max_attempts)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if max_attempts < successes:
        raise ValueError("max_attempts cannot be less than successes")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(max_wall_seconds, bool) or not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0:
        raise ValueError("max_wall_seconds must be positive and finite")
    if catalog is not None:
        task_catalog(catalog)
    path = None if config is None else Path(config).resolve()
    document = None
    if catalog is None:
        if path is None:
            catalog = DEFAULT_CATALOG
        else:
            document = load_stationary_config(path)
            catalog = stationary_spec_catalog(document.get("stationary_task"))
    entry = resolve_stationary_task(task, catalog)
    if not entry["configured"]:
        raise ValueError(f"Stationary task {entry['number']} is not configured yet")
    if path is None:
        path = Path(default_stationary_config(task, catalog)).resolve()
    if document is None:
        document = load_stationary_config(path)
    spec = validate_stationary_spec(document)
    if "required_diagnostic_stop_after_waypoint" in document["tasks"][spec["legacy_task_id"]]:
        raise ValueError("A required diagnostic waypoint profile cannot be used for collection")
    if spec["id"] != entry["id"] or stationary_spec_catalog(spec) != catalog:
        raise ValueError(f"Requested task and stationary configuration differ: catalog-{catalog} task "
                         f"{entry['number']} is {entry['id']}, the profile configures catalog-"
                         f"{stationary_spec_catalog(spec)} {spec['id']}")
    declared_gripper = require_gripper_package(document.get("gripper_variant", DEFAULT_GRIPPER_VARIANT))
    if gripper_variant is not None and gripper_variant != declared_gripper:
        raise ValueError(f"The profile pins gripper_variant {declared_gripper}; --gripper-variant {gripper_variant} "
                         "differs (collect with a profile that declares the jaw model)")
    plan = {"task": dict(entry), "catalog": catalog, "stationary_task": spec, "config": str(path),
            "config_sha256": config_digest(document), "config_source_sha256": digest(path),
            "runner": [str(Path(isaac_python).expanduser().absolute()), str(ROOT / "scripts/nepes/run_stationary.py")],
            "successes": successes, "max_attempts": max_attempts, "seed": seed,
            "max_wall_seconds": float(max_wall_seconds),
            "capture_options": capture_options,
            "algorithm_sha256": _algorithm_fingerprints()}
    # A v2 attempt is capped by its own profile timeout; v1 plans keep the fixed cap.
    max_task_seconds = stationary_max_task_seconds(document)
    if max_task_seconds is not None:
        plan["max_task_seconds"] = max_task_seconds
    if "gripper_variant" in document:  # Plans of profiles without a declaration keep their old shape.
        plan["gripper_variant"] = declared_gripper
    if storage_mode == "compact":
        plan["storage_options"] = storage
    return plan


def _plan_catalog(plan):
    # Plans written before catalog v2 carry no catalog and number tasks in v1.
    return plan.get("catalog", "v1")


def attempt_command(plan, output, seed):
    cap = str(plan["max_task_seconds"]) if "max_task_seconds" in plan else "300"
    # Old saved plans retain the exact legacy invocation. New options are only
    # passed when nondefault, and remain fingerprinted in the plan either way.
    options = _capture_options(**plan.get("capture_options", {}))
    command = [*plan["runner"], "--task", str(plan["task"]["number"]), "--catalog", _plan_catalog(plan),
            "--mode", "execute", "--headless", "--record", "--device", "cpu", "--cycles", "1",
            "--max_task_seconds", cap, "--seed", str(seed), "--config", plan["config"],
            "--output", str(output)]
    for key, default in (("capture_render_mode", "legacy"), ("image_writer_workers", 0),
                         ("image_writer_queue", 8)):
        if options[key] != default:
            command.extend(("--" + key.replace("_", "-"), str(options[key])))
    if options["profile_throughput"]:
        command.append("--profile-throughput")
    return command


def _safe_file(path, root):
    path, root = Path(path), Path(root).resolve()
    if (not path.is_file() or not path.resolve().is_relative_to(root)
            or any(part.is_symlink() for part in (path, *path.parents) if part != root and root in part.parents)):
        raise ValueError(f"Missing or external evidence file: {path.name}")
    return path


def _controller_contract(value, contract=_CONTROLLER):
    return (isinstance(value, dict)
            and all(type(value.get(key)) is type(expected) and value.get(key) == expected
                    for key, expected in contract.items()))


def _goal_controller_contract(config):
    """The pinned controller contract; only a v2 side-step between distinct stations is transport.

    A profile with the opt-in handover must also record the arm-switch contract verbatim, and one with
    the opt-in single step (navigation_control.single_step) its step contract.
    """
    contract = {**_CONTROLLER, "inter_station_transport": recorded_inter_station_transport(config)}
    spec = config.get("stationary_task")
    task = config.get("tasks", {}).get(spec.get("legacy_task_id")) if isinstance(spec, dict) else None
    if isinstance(task, dict) and "handover" in task:
        contract["handover"] = handover_contract(task)
    if isinstance(task, dict) and "single_step" in task.get("navigation_control", {}):
        contract["single_step"] = single_step_contract(task)
        if contract["single_step"].get("source_hold") is True:
            contract["initial_station_alignment"] = False
    if isinstance(task, dict) and "supported_grip_relief" in task:
        from .supported_grip_relief import supported_grip_relief_contract
        contract["supported_grip_relief"] = supported_grip_relief_contract(task)
    if isinstance(task, dict) and "press_in_mode" in task.get("placement_alignment", {}):
        from .placement_alignment import placement_overtravel_contract
        contract["placement_command_overtravel"] = placement_overtravel_contract(task)
    if isinstance(task, dict) and "transfer_completed_reference_continuity" in task:
        from .completed_transfer_continuity import contract as completed_transfer_contract
        contract["completed_transfer_reference_continuity"] = completed_transfer_contract(task)
    if isinstance(task, dict) and "transfer_cartesian_gain_s_inv" in task.get("upper_body_ik", {}):
        from .upper_body_control import transfer_cartesian_gain_contract
        contract["transfer_cartesian_gain"] = transfer_cartesian_gain_contract(task)
    if isinstance(task, dict) and (
            any(isinstance(task.get(key), dict) for key in ("lift_reference_continuity", "transfer_reference_continuity"))
            or "object_waypoint_intermediate_dwell_s" in config.get("thresholds", {})):
        # Opt-in motion continuity (runtime_v26): a runtime that ignored these keys records no such block.
        from .reference_continuity import continuity_contract
        contract["motion_continuity"] = continuity_contract(config, task)
    if isinstance(task, dict) and "place_entry_height_m" in task:
        from .place_entry_height import place_entry_height_contract
        contract["place_entry_height"] = place_entry_height_contract(task)
    if isinstance(task, dict) and "transfer_arrival_precision" in task:
        from .transfer_arrival_precision import transfer_arrival_precision_contract
        contract["transfer_arrival_precision"] = transfer_arrival_precision_contract(task)
    return contract


def _check_hold(metrics, spec, config):
    if (not isinstance(metrics, dict) or metrics.get("all_conditions_met") is not True
            or not isinstance(metrics.get("gates"), dict)
            or set(metrics["gates"]) != {"minimum_lift", "physical_grasp_retained", "object_com_stationary",
                                         "object_pose_stationary", "base_stationary", "hold_anchor_retained"}
            or any(value is not True for value in metrics["gates"].values())):
        raise ValueError("measured_hold_gates_not_verified")
    if (metrics.get("hold_contract_version") != "igris_c_crs_lift_hold_v2"
            or metrics.get("fixed_acceptance_bounds") != _HOLD_BOUNDS):
        raise ValueError("measured_hold_contract_mismatch")
    for measured, threshold, minimum in (
        ("lift_m", "minimum_lift_m", True), ("continuous_dwell_s", "hold_dwell_s", True),
        ("object_com_linear_speed_m_s", "max_object_linear_m_s", False),
        ("object_com_angular_speed_rad_s", "max_object_angular_rad_s", False),
        ("object_pose_linear_speed_m_s", "max_object_linear_m_s", False),
        ("object_pose_angular_speed_rad_s", "max_object_angular_rad_s", False),
        ("base_linear_speed_m_s", "max_base_linear_m_s", False),
        ("base_yaw_speed_rad_s", "max_base_yaw_rad_s", False),
    ):
        value = metrics.get(measured)
        limit = min(spec[threshold], .015) if measured == "object_pose_linear_speed_m_s" else spec[threshold]
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or value < 0
                or (value + 1e-9 < limit if minimum else value > limit)):
            raise ValueError(f"measured_hold_threshold_failed:{measured}")
    if metrics.get("required_dwell_s") != spec["hold_dwell_s"]:
        raise ValueError("measured_hold_requirement_changed")
    for field, bound in (("hold_anchor_position_error_m", .005), ("hold_anchor_rotation_error_rad", .05)):
        value = metrics.get(field)
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or not 0 <= value <= bound):
            raise ValueError(f"measured_hold_anchor_drift_failed:{field}")
    pose = metrics.get("hold_object_anchor_pose_w", ())
    times = [metrics.get("measurement_time_s"), metrics.get("hold_anchor_time_s")]
    if (not isinstance(pose, list) or len(pose) != 7
            or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in pose)
            or abs(sum(v * v for v in pose[3:]) - 1.) > 1e-5
            or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0 for v in times)
            or times[0] - times[1] + 1e-9 < metrics["continuous_dwell_s"]):
        raise ValueError("measured_hold_anchor_evidence_invalid")
    # The hold anchor ignores a spin about a declared symmetry axis and then reports it; an undeclared
    # profile must not carry one (its recorded rotation error is the full rotation).
    from .object_symmetry import symmetry_axis
    spin = metrics.get("hold_anchor_spin_rad")
    if symmetry_axis(config["tasks"][spec["legacy_task_id"]]) is None:
        if spin is not None:
            raise ValueError("measured_hold_spin_reported_without_declared_symmetry")
    elif not _finite_number(spin):
        raise ValueError("measured_hold_spin_invalid")


def _finite_nonnegative(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 0)


def _measured_pose(value, name):
    from .snapshot import pose_to_matrix
    if (not isinstance(value, list) or len(value) != 7
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value)
            or abs(sum(v * v for v in value[3:]) - 1.) > 1e-5):
        raise ValueError(f"placement_invalid_measured_pose:{name}")
    return pose_to_matrix(value)


def _pose_errors(first, second):
    position = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
    rotation = float(math.acos(np.clip((np.trace(first[:3, :3].T @ second[:3, :3]) - 1) / 2, -1., 1.)))
    return position, rotation


def _check_placement(metrics, spec, config):
    """Independently validate released slot placement; a lift success cannot qualify.

    The catalog-v2 pickup/load goal reuses every placement bound unchanged and
    must also prove its recorded pickup: a lift of at least minimum_lift_m that
    began at acquisition and ended before the carry posture.
    """
    pickup = spec["goal"] == "pickup_load_jig"
    if (not isinstance(metrics, dict) or metrics.get("all_conditions_met") is not True
            or metrics.get("placement_contract_version") != "igris_c_crs_jig_placement_v1"
            or metrics.get("pickup_demonstrated") is not pickup
            or not isinstance(metrics.get("gates"), dict)
            or set(metrics["gates"]) != _PLACEMENT_GATES
            or any(value is not True for value in metrics["gates"].values())
            or metrics.get("fixed_acceptance_bounds") != _PLACEMENT_BOUNDS):
        raise ValueError("measured_placement_contract_or_gates_not_verified")
    bounds = {
        "slot_position_error_m": spec["slot_position_tolerance_m"],
        "slot_normal_position_error_m": spec["slot_normal_position_tolerance_m"],
        "slot_rotation_error_rad": spec["slot_rotation_tolerance_rad"],
        "jig_position_drift_m": spec["maximum_jig_position_drift_m"],
        "jig_rotation_drift_rad": spec["maximum_jig_rotation_drift_rad"],
        "object_com_linear_speed_m_s": spec["max_object_linear_m_s"],
        "object_com_angular_speed_rad_s": spec["max_object_angular_rad_s"],
        "object_pose_linear_speed_m_s": .015,
        "object_pose_angular_speed_rad_s": spec["max_object_angular_rad_s"],
        "base_linear_speed_m_s": spec["max_base_linear_m_s"],
        "base_yaw_speed_rad_s": spec["max_base_yaw_rad_s"],
    }
    for key, limit in bounds.items():
        if not _finite_nonnegative(metrics.get(key)) or metrics[key] > limit:
            raise ValueError(f"measured_placement_threshold_failed:{key}")
    for key, minimum in (("continuous_dwell_s", spec["settle_dwell_s"]),
                         ("retreat_distance_m", spec["minimum_retreat_m"])):
        if not _finite_nonnegative(metrics.get(key)) or metrics[key] + 1e-9 < minimum:
            raise ValueError(f"measured_placement_threshold_failed:{key}")
    if metrics.get("required_dwell_s") != spec["settle_dwell_s"]:
        raise ValueError("measured_placement_settle_requirement_changed")
    task = config["tasks"][spec["legacy_task_id"]]
    measured = _measured_pose(metrics.get("slot_relative_pose"), "slot_relative_pose")
    target = _measured_pose(metrics.get("target_slot_relative_pose"), "target_slot_relative_pose")
    nominal = _measured_pose(task["target"]["pose"], "configured_slot_pose")
    jitter = np.asarray(task.get("target_position_jitter_m", [0., 0., 0.]), float)
    if (np.any(np.abs(target[:3, 3] - nominal[:3, 3]) > jitter + 1e-9)
            or _pose_errors(target, nominal)[1] > 1e-6):
        raise ValueError("measured_placement_target_differs_from_configured_slot")
    from .object_symmetry import symmetric_pose_errors, symmetry_axis
    axis = symmetry_axis(task)
    if axis is None:
        recomputed = _pose_errors(measured, target)
        if metrics.get("object_spin_rad") is not None:
            raise ValueError("measured_placement_spin_reported_without_declared_symmetry")
    else:
        # The profile declares the ring symmetric about its axis: the recorded slot rotation error is the
        # ring-axis error after the best spin, and the spin itself must be the one re-measured here.
        position, rotation, spin = symmetric_pose_errors(measured, target, axis)
        recomputed = (position, rotation)
        if not _finite_number(metrics.get("object_spin_rad")) or abs(metrics["object_spin_rad"] - spin) > 1e-7:
            raise ValueError("measured_placement_spin_inconsistent")
    for key, actual in zip(("slot_position_error_m", "slot_rotation_error_rad"), recomputed):
        if abs(metrics[key] - actual) > 1e-7:
            raise ValueError(f"measured_placement_pose_error_inconsistent:{key}")
    if abs(metrics["slot_normal_position_error_m"] - abs(measured[1, 3] - target[1, 3])) > 1e-8:
        raise ValueError("measured_placement_slot_normal_error_inconsistent")
    _measured_pose(metrics.get("jig_anchor_pose_w"), "jig_anchor_pose_w")
    _measured_pose(metrics.get("capture_object_to_tcp"), "capture_object_to_tcp")
    support = metrics.get("support_evidence", {})
    if (not isinstance(support, dict) or support.get("available") is not True
            or support.get("load_supported") is not True
            or support.get("source") != "physx_filtered_object_support_contacts"
            or support.get("object_path") != config["objects"]["crs"]
            or support.get("support_path") != config["objects"]["jig"]):
        raise ValueError("measured_placement_jig_support_not_verified")
    if (not _finite_nonnegative(support.get("object_mass_kg")) or support["object_mass_kg"] <= 0
            or not _finite_nonnegative(support.get("normal_force_n"))
            or not _finite_nonnegative(support.get("minimum_force_n"))):
        raise ValueError("measured_placement_support_force_invalid")
    minimum_force = task["placement_support"]["minimum_load_fraction"] * support["object_mass_kg"] * 9.81
    if abs(support["minimum_force_n"] - minimum_force) > 1e-8 or support["normal_force_n"] < minimum_force:
        raise ValueError("measured_placement_support_load_insufficient")
    times = metrics.get("stage_entry_times_s", {})
    expected = tuple(_pickup_labels(task)) if pickup else ("navigate_target", *_PLACEMENT_PHASES)
    if (not isinstance(times, dict) or set(times) != set(expected)
            or any(not _finite_nonnegative(times[key]) for key in expected)
            or any(times[later] <= times[earlier] for earlier, later in zip(expected, expected[1:]))
            or not _finite_nonnegative(metrics.get("capture_time_s"))
            or (not times["lift"] - 1e-9 <= metrics["capture_time_s"] <= times["carry_posture"] if pickup
                else not times["navigate_target"] <= metrics["capture_time_s"] <= times["transfer"])
            or not _finite_nonnegative(metrics.get("measurement_time_s"))
            or metrics["measurement_time_s"] - times["settle"] + 1e-9 < metrics["continuous_dwell_s"]):
        raise ValueError("measured_placement_stage_evidence_invalid")
    if pickup:
        contract = metrics.get("pickup_contract")
        if (not isinstance(contract, dict) or set(contract) != _PICKUP_CONTRACT_KEYS
                or contract["minimum_lift_m"] != spec["minimum_lift_m"]
                or contract["capture_time_s"] != metrics["capture_time_s"]
                or not _finite_nonnegative(contract["lift_m"]) or contract["lift_m"] < spec["minimum_lift_m"]):
            raise ValueError("measured_pickup_contract_not_verified")
        if "handover" in task:
            _check_handover_metrics(metrics.get("handover"), task, times)
        if "single_step" in task.get("navigation_control", {}):
            _check_single_step_metrics(metrics.get("single_step"), task, times)


def _check_single_step_metrics(evidence, task, times):
    """Terminal single-step evidence: latched, braked once and accepted inside the window at the step's exit."""
    settings = validate_single_step(task)
    labels = tuple(_pickup_labels(task))
    exit_stage = labels[labels.index(SINGLE_STEP_STAGE) + 1]
    latch, brake, exit_ = (evidence.get(key) if isinstance(evidence, dict) else None
                           for key in ("latch", "brake", "exit"))
    if (not isinstance(evidence, dict) or evidence.get("schema_version") != SINGLE_STEP_SCHEMA
            or evidence.get("phase") != "complete" or evidence.get("verdict") != SINGLE_STEP_ACCEPTED
            or not all(isinstance(item, dict) for item in (latch, brake, exit_))
            or not all(_finite_nonnegative(item.get("time_s")) for item in (latch, brake, exit_))
            or exit_["time_s"] != times.get(exit_stage)
            or not times[SINGLE_STEP_STAGE] <= latch["time_s"] < brake["time_s"] < exit_["time_s"]):
        raise ValueError("measured_single_step_evidence_not_verified")
    displacement, yaw_change = exit_.get("pelvis_displacement_heading_m"), exit_.get("pelvis_yaw_change_rad")
    (forward_low, forward_high), (left_low, left_high) = settings["window_m"]
    if (not isinstance(displacement, list) or len(displacement) != 2 or not all(map(_finite_number, displacement))
            or not _finite_number(yaw_change)
            or not (forward_low <= displacement[0] <= forward_high and left_low <= displacement[1] <= left_high)
            or abs(yaw_change) > settings["max_yaw_change_rad"]):
        raise ValueError("measured_single_step_outside_window")


def _check_handover_metrics(evidence, task, times):
    """Terminal handover evidence: the receiving hand holds, the reference switched at handover_release."""
    receiving = task["handover"]["receiving_arm"]
    if (not isinstance(evidence, dict) or evidence.get("schema_version") != HANDOVER_SCHEMA
            or evidence.get("giving_arm") != task["arm"] or evidence.get("receiving_arm") != receiving
            or evidence.get("holding_arm") != receiving or evidence.get("observation_arm") != receiving
            or not _finite_nonnegative(evidence.get("grasp_reference_switch_time_s"))
            or evidence.get("receiving_capture_time_s") != evidence["grasp_reference_switch_time_s"]
            or evidence["grasp_reference_switch_time_s"] != times.get("handover_release")
            or not _guard_evidence_equal(evidence.get("minimum_hand_tcp_distance_m"),
                                         float(task["handover"]["minimum_hand_tcp_distance_m"]))
            or [event.get("event") for event in evidence.get("events", [])] != list(handover_events(task))):
        raise ValueError("measured_handover_evidence_not_verified")
    _measured_pose(evidence.get("receiving_capture_object_to_tcp"), "receiving_capture_object_to_tcp")
    if "rotation_path" in task["handover"]:
        # Opt-in two-hand rotation: the terminal record of the measured offset from the top and the rotation
        # achieved, frozen at the handover_release entry, inside the declared bound.
        rotation = evidence.get("rotation")
        completed = rotation.get("completed") if isinstance(rotation, dict) else None
        bound = rotation_settings(task["handover"])["max_release_offset_rad"]
        if (not isinstance(completed, dict) or completed.get("at_top") is not True
                or not _finite_nonnegative(completed.get("receiving_offset_from_top_rad"))
                or completed["receiving_offset_from_top_rad"] > bound
                or not _finite_number(completed.get("ring_rotation_rad"))
                or not _finite_nonnegative(completed.get("ring_tilt_rad"))
                or completed.get("time_s") != times.get("handover_release")):
            raise ValueError("measured_handover_rotation_not_verified")


def _check_placement_trace(path, boundary, result, config):
    """Task 2 keeps the full preparation trace and proves the recorded stage path.

    The nominal control clock is compared only with the boundary's nominal
    observation time, never directly with Isaac's accumulated simulation clock.
    A terminal observation has no newly applied base command; every running
    recorded control sample must retain an exactly zero base command.
    """
    start = boundary["start_observation_time_s"]
    phases, previous_time, first_time, terminal = [], None, None, None
    phase_times, release_tcp, initial_jig = {}, None, None
    arm = config["tasks"]["crs_load_jig"]["arm"]
    object_path, jig_path = config["objects"]["crs"], config["objects"]["jig"]
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            now = row.get("observation", {}).get("time_s")
            if not _finite_nonnegative(now) or previous_time is not None and now <= previous_time:
                raise ValueError("placement_trace_observation_clock_invalid")
            previous_time = now
            observation = row["observation"]
            if initial_jig is None:
                initial_jig = _measured_pose(observation.get("object_poses_w", {}).get(jig_path), "initial_jig")
            if now < start - 1e-9:
                continue
            if first_time is None:
                first_time = now
                contact = observation.get("contact_evidence", {}).get(arm, {}).get(object_path)
                if (contact != boundary["readiness_evidence"]["contact_evidence"]
                        or observation.get("hand_closed", {}).get(arm) is not True
                        or observation.get("hand_open", {}).get(arm) is not False
                        or observation.get("object_poses_w", {}).get(object_path) != boundary["object_pose_w"]
                        or observation.get("tcp_poses_w", {}).get(arm) != boundary["measured_tcp_pose_w"]):
                    raise ValueError("held_recording_boundary_differs_from_actual_trace")
            command = row.get("command", {})
            stage = command.get("stage_id")
            if stage not in _PLACEMENT_PHASES:
                raise ValueError("placement_recording_contains_preparation_or_navigation")
            if not phases or phases[-1] != stage:
                phases.append(stage)
                phase_times[stage] = now
                if stage == "release":
                    release_tcp = _measured_pose(observation.get("tcp_poses_w", {}).get(arm), "release_tcp")
            if command.get("status") == "running":
                velocity = command.get("base_velocity_applied")
                if (not isinstance(velocity, list) or len(velocity) != 3
                        or any(isinstance(v, bool) or not isinstance(v, (int, float))
                               or not math.isfinite(v) or v != 0 for v in velocity)):
                    raise ValueError("placement_recorded_base_command_not_zero")
            elif command.get("status") != "succeeded":
                raise ValueError("placement_trace_contains_failed_control")
            if terminal is not None and terminal["command"].get("status") != "running":
                raise ValueError("placement_trace_continues_after_completion")
            terminal = row
    if (first_time is None or abs(first_time - start) > 1e-9
            or tuple(phases) != _PLACEMENT_PHASES or terminal is None
            or terminal["command"].get("status") != "succeeded"
            or terminal["command"].get("stationary_metrics") != result.get("stationary_metrics")):
        raise ValueError("placement_trace_completion_or_phase_evidence_mismatch")
    metrics = result["stationary_metrics"]
    if any(metrics["stage_entry_times_s"].get(stage) != now for stage, now in phase_times.items()):
        raise ValueError("placement_trace_stage_times_disagree")
    _check_released_terminal(terminal["observation"], metrics, release_tcp, initial_jig, config)


def _check_released_terminal(observed, metrics, release_tcp, initial_jig, config, *, arm=None):
    """Re-measure the released-placement claims from the terminal trace observation.

    ``arm`` is the hand that placed the ring: the task arm, or the receiving arm of an opt-in handover.
    """
    arm = config["tasks"]["crs_load_jig"]["arm"] if arm is None else arm
    object_path, jig_path = config["objects"]["crs"], config["objects"]["jig"]
    if observed["time_s"] != metrics["measurement_time_s"]:
        raise ValueError("placement_trace_terminal_measurement_time_disagrees")
    object_pose = _measured_pose(observed.get("object_poses_w", {}).get(object_path), "terminal_crs")
    jig_pose = _measured_pose(observed.get("object_poses_w", {}).get(jig_path), "terminal_jig")
    relative = _measured_pose(metrics["slot_relative_pose"], "terminal_slot")
    anchor = _measured_pose(metrics["jig_anchor_pose_w"], "jig_anchor")
    if (np.max(np.abs(np.linalg.inv(jig_pose) @ object_pose - relative)) > 1e-7
            or np.max(np.abs(initial_jig - anchor)) > 1e-7):
        raise ValueError("placement_terminal_or_anchor_pose_differs_from_trace")
    for key, actual in zip(("jig_position_drift_m", "jig_rotation_drift_rad"), _pose_errors(jig_pose, anchor)):
        if abs(metrics[key] - actual) > 1e-7:
            raise ValueError("placement_jig_drift_differs_from_trace")
    tcp = _measured_pose(observed.get("tcp_poses_w", {}).get(arm), "terminal_tcp")
    if release_tcp is None or abs(np.linalg.norm(tcp[:3, 3] - release_tcp[:3, 3]) - metrics["retreat_distance_m"]) > 1e-8:
        raise ValueError("placement_retreat_distance_differs_from_trace")
    from .stationary_place import _contact_absent
    if (observed.get("hand_open", {}).get(arm) is not True
            or observed.get("hand_closed", {}).get(arm) is not False
            or not all(_contact_absent(observed, side, object_path) for side in ("left", "right"))):
        raise ValueError("placement_terminal_actual_hand_release_not_verified")
    support = observed.get("support_evidence", {}).get(object_path, {})
    force = np.asarray(support.get("force_w", []), float)
    if (support.get("available") is not True or support.get("object_path") != object_path
            or support.get("support_path") != jig_path
            or support.get("source") != "physx_filtered_object_support_contacts"
            or force.shape != (3,) or not np.isfinite(force).all()
            or support.get("object_mass_kg") != metrics["support_evidence"]["object_mass_kg"]):
        raise ValueError("placement_terminal_actual_support_not_verified")
    normal_force = float(np.dot(force, config["tasks"]["crs_load_jig"]["placement_support"]["normal_w"]))
    if abs(normal_force - metrics["support_evidence"]["normal_force_n"]) > 1e-8:
        raise ValueError("placement_terminal_support_force_differs_from_trace")


def _check_held_readiness(boundary, settings, limits, spec, config):
    from .stationary_lift import _bilateral_contact
    readiness = boundary.get("readiness_evidence", {})
    expected_limits = {**limits, "minimum_initial_clearance_m": spec["minimum_initial_clearance_m"],
        "max_anchor_position_error_m": .005, "max_anchor_rotation_error_rad": .05,
        "maximum_jig_support_force_n": .05}
    if (boundary.get("pickup_demonstrated") is not False or not isinstance(readiness, dict)
            or readiness.get("limits") != expected_limits
            or any(readiness.get(key) is not True for key in ("valid", "physical_grasp_retained",
                "bilateral_contact_confirmed", "hand_closed", "sensors_available", "no_jig_support"))
            or readiness.get("required_dwell_s") != spec["preparation_dwell_s"]
            or not _finite_nonnegative(readiness.get("continuous_dwell_s"))
            or readiness["continuous_dwell_s"] + 1e-10 < settings["readiness_dwell_s"]
            or not _finite_nonnegative(readiness.get("clearance_above_slot_m"))
            or readiness["clearance_above_slot_m"] < spec["minimum_initial_clearance_m"]):
        raise ValueError("held_recording_measured_grasp_readiness_not_verified")
    pairs = (("anchor_position_error_m", .005), ("anchor_rotation_error_rad", .05),
        ("object_com_linear_speed_m_s", spec["max_object_linear_m_s"]),
        ("object_com_angular_speed_rad_s", spec["max_object_angular_rad_s"]),
        ("object_pose_linear_speed_m_s", .015),
        ("object_pose_angular_speed_rad_s", spec["max_object_angular_rad_s"]),
        ("base_linear_speed_m_s", spec["max_base_linear_m_s"]),
        ("base_yaw_speed_rad_s", spec["max_base_yaw_rad_s"]))
    for key, maximum in pairs:
        if not _finite_nonnegative(readiness.get(key)) or readiness[key] > maximum:
            raise ValueError(f"held_recording_readiness_threshold_failed:{key}")
    arm, object_path = config["tasks"][spec["legacy_task_id"]]["arm"], config["objects"]["crs"]
    contact = readiness.get("contact_evidence", {})
    if (not isinstance(contact, dict) or contact.get("force_threshold_n") != .05
            or not _bilateral_contact({"contact_evidence": {arm: {object_path: contact}}}, arm, object_path)):
        raise ValueError("held_recording_bilateral_force_evidence_missing")
    support = readiness.get("jig_support_evidence", {})
    if (not isinstance(support, dict) or support.get("available") is not True
            or support.get("load_supported") is not False
            or support.get("source") != "physx_filtered_object_support_contacts"
            or support.get("object_path") != object_path or support.get("support_path") != config["objects"]["jig"]
            or isinstance(support.get("normal_force_n"), bool)
            or not isinstance(support.get("normal_force_n"), (int, float))
            or not math.isfinite(support["normal_force_n"]) or abs(support["normal_force_n"]) > .05):
        raise ValueError("held_recording_initial_object_still_supported_or_sensor_missing")
    if (not _finite_nonnegative(readiness.get("capture_time_s"))
            or readiness.get("measurement_time_s") != boundary["start_observation_time_s"]
            or not boundary["attempt_start_observation_time_s"] <= readiness["capture_time_s"] <= readiness["measurement_time_s"]
            or readiness["measurement_time_s"] - readiness["capture_time_s"] + 1e-9 < readiness["continuous_dwell_s"]
            or boundary.get("capture_object_to_tcp") != readiness.get("capture_object_to_tcp")):
        raise ValueError("held_recording_capture_evidence_invalid")
    capture = _measured_pose(readiness["capture_object_to_tcp"], "held_capture")
    object_pose = _measured_pose(boundary.get("object_pose_w"), "held_object_pose")
    tcp = _measured_pose(boundary.get("measured_tcp_pose_w"), "held_tcp_pose")
    from .object_symmetry import symmetric_capture_errors, symmetry_axis, symmetry_pivot_radius
    held_task = config["tasks"][spec["legacy_task_id"]]
    axis = symmetry_axis(held_task)
    if axis is None:
        position, rotation = _pose_errors(np.linalg.inv(object_pose) @ tcp, capture)
    else:
        position, rotation, _ = symmetric_capture_errors(np.linalg.inv(object_pose) @ tcp, capture, axis,
                                                         symmetry_pivot_radius(held_task))
    if (position > config["thresholds"]["capture_position_m"]
            or rotation > config["thresholds"]["capture_rotation_rad"]):
        raise ValueError("held_recording_grasp_capture_not_retained")


def _upright_limits(spec):
    return {"max_object_com_linear_m_s": spec["max_object_linear_m_s"],
            "max_object_com_angular_rad_s": spec["max_object_angular_rad_s"],
            "max_object_pose_linear_m_s": .015,
            "max_object_pose_angular_rad_s": spec["max_object_angular_rad_s"],
            "max_base_linear_m_s": spec["max_base_linear_m_s"],
            "max_base_yaw_rad_s": spec["max_base_yaw_rad_s"], "max_hand_contact_force_n": .05,
            "upright_joint_tolerance_rad": spec["upright_joint_tolerance_rad"],
            "maximum_pelvis_tilt_rad": spec["maximum_pelvis_tilt_rad"], **_SOURCE_ANCHOR_BOUNDS}


def _check_upright_readiness(boundary, settings, spec, config):
    """The pickup recording starts only from a measured, quiet upright stance.

    The gate's own measurements and the machine's nested readiness are both
    pinned: exact keys and limits, every gate true, and every measured number
    inside its bound, so a valid flag can never stand in for the evidence.
    """
    from .stationary_place import _contact_absent
    task, object_path = config["tasks"][spec["legacy_task_id"]], config["objects"]["crs"]
    limits = _upright_limits(spec)
    readiness = boundary.get("readiness_evidence", {})
    if (boundary.get("pickup_demonstrated") is not True or not isinstance(readiness, dict)
            or set(readiness) != _UPRIGHT_BOUNDARY_KEYS or readiness["limits"] != limits
            or any(readiness[key] is not True for key in ("valid", "machine_readiness_valid", "hand_open",
                                                          "same_object_contacts_absent", "sensors_available"))
            or readiness["required_dwell_s"] != settings["readiness_dwell_s"]
            or not _finite_nonnegative(readiness["continuous_dwell_s"])
            or readiness["continuous_dwell_s"] + 1e-10 < settings["readiness_dwell_s"]):
        raise ValueError("upright_recording_readiness_not_verified")
    for key, limit in _UPRIGHT_SPEEDS:
        if not _finite_nonnegative(readiness[key]) or readiness[key] > limits[limit]:
            raise ValueError(f"upright_recording_readiness_threshold_failed:{key}")
    contacts = readiness["contact_evidence"]
    if (not isinstance(contacts, dict) or set(contacts) != {"left", "right"}
            or not all(_contact_absent({"contact_evidence": {arm: {object_path: contacts[arm]}}}, arm, object_path)
                       for arm in ("left", "right"))):
        raise ValueError("upright_recording_hand_contact_evidence_invalid")
    machine, names = readiness["machine_readiness"], task["posture_paths"]["reach_posture"]["joint_names"]
    # The machine dwells the longer of its preparation and this boundary's readiness dwell.
    if (not isinstance(machine, dict) or set(machine) != _UPRIGHT_MACHINE_KEYS or machine["limits"] != limits
            or any(machine[key] is not True for key in ("valid", *_UPRIGHT_GATES))
            or machine["upright_reference"] != "commanded_upper_joint_positions"
            or machine["upright_joint_names"] != names or machine["max_upright_error_joint"] not in names
            or machine["required_dwell_s"] != max(spec["preparation_dwell_s"], settings["readiness_dwell_s"])
            or machine["continuous_dwell_s"] != readiness["continuous_dwell_s"]
            or machine["measurement_time_s"] != boundary["start_observation_time_s"]
            or not isinstance(machine["upright_command_targets"], dict)
            or set(machine["upright_command_targets"]) != set(names)
            or not all(map(_finite_number, machine["upright_command_targets"].values()))):
        raise ValueError("upright_recording_machine_readiness_not_verified")
    try:
        _measured_pose(machine["source_object_anchor_pose_w"], "source_object_anchor")
    except ValueError as exc:
        raise ValueError("upright_recording_machine_readiness_not_verified") from exc
    # Station bounds are the machine's source-station stopping region.
    navigation = task.get("navigation_control", {})
    bounds = (("max_upright_joint_error_rad", spec["upright_joint_tolerance_rad"]),
              ("max_upright_command_change_rad", _UPRIGHT_COMMAND_TOLERANCE_RAD),
              ("pelvis_tilt_rad", spec["maximum_pelvis_tilt_rad"]),
              ("base_station_position_error_m",
               navigation.get("station_position_tolerance_m", config["thresholds"]["base_position_m"])),
              ("base_station_yaw_error_rad",
               navigation.get("station_yaw_tolerance_rad", config["thresholds"]["base_yaw_rad"])),
              ("source_object_position_error_m", limits["max_source_object_position_error_m"]),
              ("source_object_rotation_error_rad", limits["max_source_object_rotation_error_rad"]),
              *((key, limits[limit]) for key, limit in _UPRIGHT_SPEEDS))
    for key, bound in bounds:
        if not _finite_nonnegative(machine[key]) or machine[key] > bound:
            raise ValueError(f"upright_recording_machine_readiness_threshold_failed:{key}")
    forces = machine["hand_contact_force_n"]
    if (not isinstance(forces, dict) or set(forces) != {"left", "right"}
            or any(not _finite_nonnegative(force) or force > .05 for force in forces.values())):
        raise ValueError("upright_recording_machine_readiness_threshold_failed:hand_contact_force_n")


def _check_recording_boundary(config, result, provenance):
    """Old whole-attempt recordings remain valid; opt-in prepared recordings require evidence."""
    settings = config.get("recording_boundary")
    if settings is None:
        return None
    held = settings["kind"] == "after_measured_grasp_hold"
    upright = settings["kind"] == "after_measured_upright_stance"
    boundary = result.get("recording_boundary")
    if (not isinstance(boundary, dict) or provenance.get("recording_boundary") != boundary
            or boundary.get("schema_version") != "igris_c_prepared_recording_boundary_v1"
            or boundary.get("kind") != settings["kind"] or boundary.get("started") is not True
            or boundary.get("start_stage") != ("transfer" if held else "reach_posture" if upright else "approach")
            or boundary.get("preparation_trace") != "trace.jsonl"
            or boundary.get("world_or_joint_state_written_at_boundary") is not False
            or boundary.get("object_attachment_created_at_boundary") is not False):
        raise ValueError("prepared_recording_boundary_contract_mismatch")
    for key in ("start_observation_time_s", "start_sim_time_s", "preparation_duration_s",
                "attempt_start_observation_time_s"):
        if not _finite_nonnegative(boundary.get(key)):
            raise ValueError("prepared_recording_boundary_time_invalid")
    if abs(boundary["start_observation_time_s"] - boundary["attempt_start_observation_time_s"]
           - boundary["preparation_duration_s"]) > 1e-8:
        raise ValueError("prepared_recording_preparation_duration_mismatch")
    spec = config["stationary_task"]
    limits = {"tcp_position_m": config["thresholds"]["tcp_position_m"],
              "tcp_rotation_rad": config["thresholds"]["tcp_rotation_rad"],
              "max_object_com_linear_m_s": spec["max_object_linear_m_s"],
              "max_object_com_angular_rad_s": spec["max_object_angular_rad_s"],
              "max_object_pose_linear_m_s": .015,
              "max_object_pose_angular_rad_s": spec["max_object_angular_rad_s"],
              "max_base_linear_m_s": spec["max_base_linear_m_s"],
              "max_base_yaw_rad_s": spec["max_base_yaw_rad_s"],
              "max_hand_contact_force_n": .05}
    readiness = boundary.get("readiness_evidence", {})
    if held:
        _check_held_readiness(boundary, settings, limits, spec, config)
    elif upright:
        _check_upright_readiness(boundary, settings, spec, config)
    elif (not isinstance(readiness, dict) or readiness.get("limits") != limits
            or any(readiness.get(key) is not True for key in
                   ("valid", "hand_open", "same_object_contacts_absent", "sensors_available"))
            or readiness.get("required_dwell_s") != settings["readiness_dwell_s"]
            or not _finite_nonnegative(readiness.get("continuous_dwell_s"))
            or readiness["continuous_dwell_s"] + 1e-10 < settings["readiness_dwell_s"]):
        raise ValueError("prepared_recording_readiness_not_verified")
    pairs = () if held or upright else (("tcp_position_error_m", "tcp_position_m"),
                           ("tcp_rotation_error_rad", "tcp_rotation_rad"),
                           ("object_com_linear_speed_m_s", "max_object_com_linear_m_s"),
                           ("object_com_angular_speed_rad_s", "max_object_com_angular_rad_s"),
                           ("object_pose_linear_speed_m_s", "max_object_pose_linear_m_s"),
                           ("object_pose_angular_speed_rad_s", "max_object_pose_angular_rad_s"),
                           ("base_linear_speed_m_s", "max_base_linear_m_s"),
                           ("base_yaw_speed_rad_s", "max_base_yaw_rad_s"))
    for measured, limit in pairs:
        if not _finite_nonnegative(readiness.get(measured)) or readiness[measured] > limits[limit]:
            raise ValueError(f"prepared_recording_readiness_threshold_failed:{measured}")
    summary = boundary.get("preparation_contact_summary", {})
    if (not isinstance(summary, dict) or summary.get("target_object_path") != config["objects"]["crs"]
            or any(type(summary.get(key)) is not int or summary[key] < 0 for key in
                   ("samples", "hand_contact_samples", "sensors_unavailable_samples"))
            or summary["samples"] < 1 or summary["hand_contact_samples"] > summary["samples"]
            or summary["sensors_unavailable_samples"] > summary["samples"]
            or not _finite_nonnegative(summary.get("max_hand_contact_force_n"))):
        raise ValueError("prepared_recording_contact_provenance_missing")
    if summary["max_hand_contact_force_n"] > 0:
        peak = summary.get("peak_contact", {})
        if (not isinstance(peak, dict) or peak.get("arm") not in ("left", "right")
                or peak.get("force_magnitude_n") != summary["max_hand_contact_force_n"]
                or not _finite_nonnegative(peak.get("time_s"))
                or not boundary["attempt_start_observation_time_s"] <= peak["time_s"] <= boundary["start_observation_time_s"]
                or not isinstance(peak.get("sensor_paths"), list) or not peak["sensor_paths"]):
            raise ValueError("prepared_recording_peak_contact_provenance_missing")
    return boundary


def _check_upright_boundary_row(row, previous_stage, boundary, task, object_path):
    """The boundary row is the fresh reach entry whose observation the readiness measured.

    The upright joint error and pelvis tilt are re-measured from that raw
    observation: measured joints against the targets applied in the previous
    control step, and the pelvis (Link_Waist_Pitch) z axis against vertical.
    """
    observation, command = row["observation"], row["command"]
    readiness, arm = boundary["readiness_evidence"], task["arm"]
    machine = readiness["machine_readiness"]
    try:
        measured, commanded = observation["joint_positions"], observation["commanded_upper_joint_positions"]
        errors = {name: abs(float(measured[name]) - float(commanded[name]))
                  for name in task["posture_paths"]["reach_posture"]["joint_names"]}
        pelvis = _measured_pose(observation.get("pelvis_pose_w"), "boundary_pelvis")
        tilt = math.acos(float(np.clip(pelvis[2, 2], -1., 1.)))
    except (KeyError, TypeError, ValueError, AttributeError):
        errors, tilt = {}, math.nan
    worst = max(errors.values(), default=math.nan)
    if (not abs(observation.get("time_s") - boundary["start_observation_time_s"]) <= 1e-9
            or previous_stage != "navigate_source" or command.get("stage_id") != "reach_posture"
            or command.get("status") != "running" or command.get("preparation_readiness") != machine
            or any(observation.get("contact_evidence", {}).get(side, {}).get(object_path)
                   != readiness["contact_evidence"][side] for side in ("left", "right"))
            or observation.get("hand_open", {}).get(arm) is not True
            or observation.get("hand_closed", {}).get(arm) is not False
            or observation.get("object_poses_w", {}).get(object_path) != boundary["object_pose_w"]
            or observation.get("tcp_poses_w", {}).get(arm) != boundary["measured_tcp_pose_w"]
            or not abs(worst - machine["max_upright_joint_error_rad"]) <= 1e-9
            or not abs(errors.get(machine["max_upright_error_joint"], math.nan) - worst) <= 1e-9
            or not abs(tilt - machine["pelvis_tilt_rad"]) <= 1e-9):
        raise ValueError("upright_recording_boundary_differs_from_actual_trace")


def _guard_evidence_equal(actual, expected):
    """Compare a fully specified guard record without treating True as numeric 1."""
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and set(actual) == set(expected)
                and all(_guard_evidence_equal(actual[key], value) for key, value in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(_guard_evidence_equal(a, b) for a, b in zip(actual, expected)))
    if isinstance(expected, bool) or expected is None:
        return actual is expected
    if isinstance(expected, (int, float)):
        return _finite_number(actual) and abs(actual - expected) <= 1e-9
    return type(actual) is type(expected) and actual == expected


class _PickupTransportAdmission:
    """Recompute opt-in transport evidence from recorded measurements, not success flags.

    The trace row carries the observation just judged in the previous stage and
    the *next* command. Check both sides of a transition before accepting that
    command. The geometry calculation remains a declared conservative proxy,
    not a native/full-mesh collision certificate.
    """

    def __init__(self, task, object_path):
        from .payload_transport import validate_transport_constraints
        self.constraints = (validate_transport_constraints(task["transport_constraints"])
                            if "transport_constraints" in task else None)
        self.navigation = task.get("navigation_control", {})
        self.hold = self.navigation.get("stationary_pelvis_hold")
        # Opt-in single stance: navigate_target has no arrival to settle (no_walk_stance).
        self.no_walk = "no_walk" in self.navigation
        self.feet_hold = None
        self.feet_anchored = False
        if "stationary_feet_hold" in self.navigation:
            from .stationary_feet import StationaryFeetHold
            self.feet_hold = StationaryFeetHold(self.navigation["stationary_feet_hold"])
        self.object_path = object_path
        self.recorded_stages = tuple(stage for stage, _ in _pickup_recorded(task))
        self.payload_evidence = self.entry_evidence = None
        self.pelvis_anchor = self.pelvis_anchor_time = self.base_settle_since = None
        # Opt-in single step (single_step.py): replayed from the raw rows with the machine's own step state.
        settings = validate_single_step(task)
        self.single_step = None if settings is None else SingleStep(settings, **arrival_limits(task))
        self.source_hold = settings is not None and settings.get("source_hold") is True
        self.single_step_stepping = False
        self.previous_base_command = None

    def check_source_hold_preparation(self, row, previous_stage):
        """Verify raw prior/applied commands before the recorded boundary independently.

        Legacy single-step admission starts at reach_posture. This opt-in also
        checks both sides of every preparation observation, including the row
        that transitions from navigate_source to the recorded reach.
        """
        observation, command = row["observation"], row["command"]
        stage = command.get("stage_id")
        preparation = {"prerequisites", "navigate_source"}
        if not self.source_hold or not (stage in preparation or previous_stage in preparation):
            return
        try:
            prior = planar_command(observation.get("commanded_base_velocity"))
            applied = planar_command(command.get("base_velocity_applied"))
        except ValueError as exc:
            raise ValueError("pickup_source_hold_preparation_command_missing") from exc
        if any(prior) or any(applied):
            raise ValueError("pickup_source_hold_preparation_command_not_zero")
        if stage in preparation and command.get("base_arrival_braking") is not True:
            raise ValueError("pickup_source_hold_preparation_braking_not_enabled")

    def _check_single_step(self, row, previous_stage):
        """Replay the opt-in single step; returns whether this observation's hold drift check is suspended.

        As in the machine, the observation was judged in previous_stage and the command was produced by that
        step: the goal is latched from the navigate_target entry observation (no pre-step path) or from the
        observation where the machine reports its pre-step posture complete (that observation's executor
        status must be complete and the published posture evidence must have held its settle dwell; like the
        other joint-path stages, the arrival itself is not replayed); the latched poses and goal, the brake,
        the commanded travel (from the causal applied base commands), the settle and the window are
        recomputed, the published evidence must match exactly, and navigate_target may command the base only
        between the latch and the one-shot brake (exactly zero before and after, never re-commanded).
        """
        step = self.single_step
        observation, command = row["observation"], row["command"]
        now, stage = observation["time_s"], command["stage_id"]
        try:
            applied = planar_command(observation.get("commanded_base_velocity"))
        except ValueError as exc:
            raise ValueError("pickup_trace_single_step_command_missing") from exc
        if self.previous_base_command is not None and applied != self.previous_base_command:
            raise ValueError("pickup_trace_single_step_command_not_causal")
        judged = previous_stage == SINGLE_STEP_STAGE
        suspended = judged and stage == SINGLE_STEP_STAGE and step.latched
        if judged or stage == SINGLE_STEP_STAGE:
            torso = _measured_pose(observation.get("base_pose_w"), "single_step_torso")
            pelvis = _measured_pose(observation.get("pelvis_pose_w"), "single_step_pelvis")
            velocity = observation.get("base_velocity_w")
            if (not isinstance(velocity, list) or len(velocity) != 6
                    or not all(_finite_number(value) for value in velocity)):
                raise ValueError("pickup_trace_single_step_velocity_missing")
            if judged and step.latched and step.observe(now, torso, pelvis, velocity, applied) == SINGLE_STEP_REJECTED:
                raise ValueError("pickup_trace_single_step_outside_window")
            if stage == SINGLE_STEP_STAGE and not judged:
                step.enter(now, torso, pelvis, velocity)
            elif (stage == SINGLE_STEP_STAGE and not step.latched
                    and (command.get("single_step") or {}).get("phase") == "stepping"):
                reference = observation.get("posture_reference")
                evidence = (command.get("posture_evidence") or {}).get(SINGLE_STEP_STAGE)
                if (not isinstance(reference, dict) or reference.get("stage_id") != SINGLE_STEP_STAGE
                        or reference.get("status") != "running" or reference.get("complete") is not True
                        or not isinstance(evidence, dict) or evidence.get("reference_complete") is not True
                        or evidence.get("grasp_retained") is not True
                        or not _finite_nonnegative(evidence.get("continuous_dwell_s"))
                        or evidence["continuous_dwell_s"] < step.pre_step_path["settle_dwell_s"] - 1e-9):
                    raise ValueError("pickup_trace_single_step_latch_before_pre_step_posture")
                step.latch(now, torso, pelvis, velocity)
            if judged and stage != SINGLE_STEP_STAGE:
                if step.verdict != SINGLE_STEP_ACCEPTED:
                    raise ValueError("pickup_trace_single_step_exit_without_accepted_window")
                step.complete(now, pelvis)
        self._compare(command, "single_step", step.evidence())
        if stage == SINGLE_STEP_STAGE and command.get("status") == "running":
            velocity, braking = command.get("base_velocity_applied"), command.get("base_arrival_braking")
            if not isinstance(velocity, list) or len(velocity) != 3 or not all(map(_finite_number, velocity)):
                raise ValueError("pickup_trace_single_step_base_command_invalid")
            goal = _guard_evidence_equal(command.get("base_goal_pose_w"), step.goal_pose_w)
            if step.phase == "pre_step_posture":
                valid = (braking is False and not any(velocity) and command.get("motion_mode") == "joint_posture"
                         and command.get("base_goal_pose_w") is None)
            elif step.phase == "stepping":
                valid = (braking is False and goal and command.get("motion_mode") == "navigate_base"
                         and math.hypot(velocity[0], velocity[1]) <= self.navigation.get("max_speed_m_s", .12) + 1e-9)
                self.single_step_stepping = True
            else:
                valid = braking is True and goal and not any(velocity) and command.get("motion_mode") == "navigate_base"
            if not valid:
                raise ValueError("pickup_trace_single_step_base_command_" + step.phase)
        self.previous_base_command = command.get("base_velocity_applied")
        return suspended

    @staticmethod
    def _compare(command, key, expected):
        metrics = command.get("stationary_metrics")
        if (key not in command or not isinstance(metrics, dict) or key not in metrics
                or not _guard_evidence_equal(command[key], expected)
                or not _guard_evidence_equal(metrics[key], expected)):
            raise ValueError("pickup_trace_" + key + "_evidence_mismatch")

    def check(self, row, previous_stage, *, slot_target_pose_w=None):
        from .payload_transport import evaluate_payload_transport
        from .snapshot import matrix_to_pose
        observation, command = row["observation"], row["command"]
        now, stage = observation["time_s"], command["stage_id"]
        self.single_step_stepping = False
        suspended = self.single_step is not None and self._check_single_step(row, previous_stage)
        if self.feet_hold is not None:
            anchor = not self.feet_anchored
            if anchor and (stage != "reach_posture" or previous_stage != "navigate_source"):
                raise ValueError("pickup_trace_stationary_feet_boundary_missing")
            evidence = self.feet_hold.check(observation, anchor=anchor)
            if not evidence["valid"]:
                raise ValueError("pickup_trace_stationary_feet_hold:" + evidence["reason"])
            self.feet_anchored = True
            self._compare(command, "stationary_feet_hold", evidence)
        if self.constraints is not None:
            crs = _measured_pose(observation.get("object_poses_w", {}).get(self.object_path), "transport_crs")
            checked_stages = [previous_stage] if previous_stage in self.recorded_stages else []
            if not checked_stages or previous_stage != stage:
                checked_stages.append(stage)
            for checked_stage in checked_stages:
                entry = checked_stage == "navigate_target" and stage != previous_stage and checked_stage == stage
                evidence = evaluate_payload_transport(self.constraints, matrix_to_pose(crs), checked_stage,
                                                      at_walk_entry=entry, slot_target_pose_w=slot_target_pose_w)
                evidence.update(measurement_time_s=now, object_path=self.object_path)
                if not evidence["valid"]:
                    raise ValueError("pickup_trace_unsafe_payload_transport:" + ";".join(evidence["violations"]))
                if evidence["checked"]:
                    self.payload_evidence = evidence
                    if entry:
                        self.entry_evidence = evidence
            self._compare(command, "payload_transport", self.payload_evidence)
            self._compare(command, "walk_entry_payload_transport", self.entry_evidence)
        if self.hold is not None:
            pelvis = _measured_pose(observation.get("pelvis_pose_w"), "stationary_pelvis")
            if math.hypot(pelvis[0, 0], pelvis[1, 0]) < 1e-8:
                raise ValueError("pickup_trace_stationary_pelvis_heading_invalid")
            if self.pelvis_anchor is None:
                if stage != "reach_posture" or previous_stage != "navigate_source":
                    raise ValueError("pickup_trace_stationary_pelvis_boundary_missing")
                self.pelvis_anchor, self.pelvis_anchor_time = pelvis.copy(), now
            if self.single_step is not None and previous_stage == SINGLE_STEP_STAGE and stage != SINGLE_STEP_STAGE:
                # The machine re-anchors the hold at the single step's navigate_target exit (this observation).
                self.pelvis_anchor, self.pelvis_anchor_time = pelvis.copy(), now
            delta = float(np.linalg.norm(pelvis[:2, 3] - self.pelvis_anchor[:2, 3]))
            angle = (math.atan2(pelvis[1, 0], pelvis[0, 0])
                     - math.atan2(self.pelvis_anchor[1, 0], self.pelvis_anchor[0, 0]))
            yaw = abs(math.atan2(math.sin(angle), math.cos(angle)))
            drifted = delta > self.hold["maximum_position_drift_m"] or yaw > self.hold["maximum_yaw_drift_rad"]
            if drifted and not suspended:
                raise ValueError("pickup_trace_stationary_pelvis_drift")
            evidence = {"valid": not drifted, "measurement_time_s": now, "anchor_time_s": self.pelvis_anchor_time,
                "anchor_pose_w": matrix_to_pose(self.pelvis_anchor), "measured_pose_w": matrix_to_pose(pelvis),
                "position_drift_m": delta, "yaw_drift_rad": yaw, "limits": self.hold,
                "position_measurement": "world_xy", "orientation_measurement": "wrapped_world_yaw",
                "zero_base_command_required": True, "actual_base_motion_zero_claimed": False}
            if self.single_step is not None:
                # The single step's own settle and window replace the settled-arrival evidence (replayed above).
                evidence["drift_check_suspended"] = suspended
            elif stage == "navigate_target" or previous_stage == "navigate_target":
                if previous_stage != "navigate_target":
                    self.base_settle_since = None
                velocity = observation.get("base_velocity_w")
                if (not isinstance(velocity, list) or len(velocity) != 6
                        or not all(_finite_number(value) for value in velocity)):
                    raise ValueError("pickup_trace_stationary_pelvis_velocity_missing")
                slow = (np.linalg.norm(velocity[:3]) <= self.navigation.get("arrival_linear_speed_m_s", .03)
                        and abs(velocity[5]) <= self.navigation.get("arrival_yaw_speed_rad_s", .12))
                if slow:
                    if self.base_settle_since is None:
                        self.base_settle_since = now
                else:
                    self.base_settle_since = None
                dwell = 0. if self.base_settle_since is None else now - self.base_settle_since
                required = self.navigation.get("arrival_dwell_s", .4)
                evidence.update(base_quiet=bool(slow), arrival_continuous_dwell_s=dwell,
                                arrival_required_dwell_s=required)
                if previous_stage == "navigate_target" and stage != "navigate_target" and not self.no_walk and (
                        not slow or dwell < required - 1e-9):
                    raise ValueError("pickup_trace_stationary_pelvis_arrival_not_settled")
            # A running single-step navigate_target row follows the step's own command rules (_check_single_step).
            if self.single_step is None or stage != SINGLE_STEP_STAGE or command.get("status") != "running":
                if command.get("base_arrival_braking") is not True:
                    raise ValueError("pickup_trace_stationary_pelvis_braking_missing")
                if command.get("status") == "running":
                    velocity = command.get("base_velocity_applied")
                    if (not isinstance(velocity, list) or len(velocity) != 3
                            or any(not _finite_number(value) or value != 0 for value in velocity)):
                        raise ValueError("pickup_trace_stationary_pelvis_base_command_not_zero")
            self._compare(command, "stationary_pelvis_hold", evidence)


class _HandoverAdmission:
    """Replay the opt-in handover from raw rows: both hands' evidence, the hand-hand proxy, the arm switch.

    A row carries the observation judged in ``previous_stage`` and the next command. Every claim is
    re-measured from that observation (both hands' filtered contacts, the two TCP poses, the ring
    pose) and the command's provenance fields (observation arm, holding arm, per-arm jaw commands,
    recorded receiving capture) must agree. Absent the block nothing is checked.
    """

    def __init__(self, task, object_path, thresholds):
        self.enabled = "handover" in task
        if not self.enabled:
            return
        from .handover import validate_handover
        self.settings = validate_handover(task, thresholds)
        self.task, self.object_path, self.thresholds = task, object_path, thresholds
        self.giving, self.receiving = task["arm"], self.settings["receiving_arm"]
        self.order = handover_stages(task)
        self.seen = []
        self.capture = self.switch_time = None
        # Opt-in two-hand rotation (handover_rotate) and pre-release dwell (handover_release ``verifying`` phase).
        self.rotation = self.settings if "rotation_path" in self.settings else None
        self.rotate_capture = None
        self.release_entry_time = None
        self.release_phase_seen = []

    def _relative(self, observation, arm):
        ring = _measured_pose(observation.get("object_poses_w", {}).get(self.object_path), "handover_ring")
        tcp = _measured_pose(observation.get("tcp_poses_w", {}).get(arm), f"handover_{arm}_tcp")
        return np.linalg.inv(ring) @ tcp

    def _retained(self, observation, arm, capture):
        from .stationary_lift import _bilateral_contact
        position, rotation = _pose_errors(self._relative(observation, arm), capture)
        return (_bilateral_contact(observation, arm, self.object_path)
                and position <= self.thresholds["capture_position_m"]
                and rotation <= self.thresholds["capture_rotation_rad"])

    def check(self, row, previous_stage):
        if not self.enabled:
            return
        from .handover import (hand_commands_for_stage, hand_tcp_distance, holding_arm_for_stage,
                               observation_arm_for_stage)
        from .stationary_lift import _bilateral_contact
        from .stationary_pick_load import _contact_force
        from .stationary_place import _contact_absent
        observation, command = row["observation"], row["command"]
        stage, status, now = command.get("stage_id"), command.get("status"), observation.get("time_s")
        evidence, metrics = command.get("handover"), command.get("stationary_metrics")
        if (not isinstance(evidence, dict) or not isinstance(metrics, dict)
                or not _guard_evidence_equal(metrics.get("handover"), evidence)
                or evidence.get("schema_version") != HANDOVER_SCHEMA
                or evidence.get("giving_arm") != self.giving or evidence.get("receiving_arm") != self.receiving):
            raise ValueError("pickup_trace_handover_evidence_missing")
        if (command.get("arm") != observation_arm_for_stage(self.task, stage)
                or evidence.get("observation_arm") != command["arm"]):
            raise ValueError("pickup_trace_handover_observation_arm_mismatch")
        holding = holding_arm_for_stage(self.task, stage)
        if command.get("holding_arm") != holding or evidence.get("holding_arm") != holding:
            raise ValueError("pickup_trace_handover_holding_arm_mismatch")
        if stage in HANDOVER_STAGE_SET and stage != previous_stage:
            # The profile's own handover order (six stages, seven with rotation_path), each entered once.
            expected_previous = "carry_posture" if not self.seen else self.seen[-1]
            if (stage in self.seen or previous_stage != expected_previous or stage not in self.order
                    or self.order.index(stage) != len(self.seen)):
                raise ValueError("pickup_trace_handover_stage_order_invalid")
            self.seen.append(stage)
        if stage == "navigate_target" and previous_stage == "handover_retreat" and len(self.seen) != len(self.order):
            raise ValueError("pickup_trace_handover_stage_order_invalid")
        phase = evidence.get("release_phase")
        if stage == "handover_release" and status == "running" and phase not in release_phases(self.task):
            raise ValueError("pickup_trace_handover_release_phase_invalid")
        if status == "running":
            if command.get("hand_commands") != hand_commands_for_stage(
                    self.task, stage, command.get("hand_command"),
                    release_phase=phase if stage == "handover_release" else None):
                raise ValueError("pickup_trace_handover_hand_commands_mismatch")
            if command.get("contact_arms") != ([self.giving, self.receiving] if stage in HANDOVER_STAGE_SET
                                               else [command["arm"]]):
                raise ValueError("pickup_trace_handover_contact_arms_mismatch")
        if previous_stage not in HANDOVER_STAGE_SET and stage not in HANDOVER_STAGE_SET:
            return
        # Both hands' filtered contact sensors and the kinematic hand-hand proxy, re-measured.
        for arm in (self.giving, self.receiving):
            if _contact_force(observation, arm, self.object_path) is None:
                raise ValueError("pickup_trace_handover_contact_evidence_missing")
        distance = hand_tcp_distance(observation.get("tcp_poses_w", {}), self.giving, self.receiving)
        if not _guard_evidence_equal(evidence.get("hand_tcp_distance_m"), distance):
            raise ValueError("pickup_trace_handover_hand_distance_mismatch")
        if previous_stage in HANDOVER_STAGE_SET and distance < self.settings["minimum_hand_tcp_distance_m"]:
            raise ValueError("pickup_trace_handover_hand_clearance")
        entry = stage != previous_stage
        if entry and stage in ("handover_pregrasp", "handover_approach", "handover_close", "handover_release"):
            # The giving hand still holds the ring at every transition up to the switch.
            giving_capture = _measured_pose(evidence.get("giving", {}).get("capture_object_to_tcp"), "giving_capture")
            if not self._retained(observation, self.giving, giving_capture):
                raise ValueError("pickup_trace_handover_giving_pinch_missing")
        if previous_stage == "handover_pregrasp" and not _contact_absent(observation, self.receiving, self.object_path):
            raise ValueError("pickup_trace_handover_receiving_contact_before_grasp")
        if entry and stage == "handover_release":
            # The switch: the receiving pinch is physical and its measured grasp is the recorded capture.
            if not _bilateral_contact(observation, self.receiving, self.object_path):
                raise ValueError("pickup_trace_handover_receiving_pinch_missing")
            capture = self._relative(observation, self.receiving)
            recorded = _measured_pose(evidence.get("receiving_capture_object_to_tcp"), "receiving_capture")
            if (np.max(np.abs(capture - recorded)) > 1e-7 or evidence.get("grasp_reference_switch_time_s") != now
                    or evidence.get("receiving_capture_time_s") != now):
                raise ValueError("pickup_trace_handover_capture_differs_from_trace")
            self.capture, self.switch_time = capture, now
            self.release_entry_time = now
        if previous_stage in ("handover_release", "handover_retreat"):
            if self.capture is None:
                raise ValueError("pickup_trace_handover_capture_missing")
            if entry and not self._retained(observation, self.receiving, self.capture):
                raise ValueError("pickup_trace_handover_receiving_pinch_missing")
            position, rotation = _pose_errors(self._relative(observation, self.receiving), self.capture)
            if position > self.thresholds["capture_position_m"] or rotation > self.thresholds["capture_rotation_rad"]:
                raise ValueError("pickup_trace_handover_receiving_capture_not_retained")
        if previous_stage == "handover_retreat" and not _contact_absent(observation, self.giving, self.object_path):
            raise ValueError("pickup_trace_handover_giving_contact_after_release")
        if stage == "handover_release" and status == "running":
            self._check_release_phase(observation, phase, now)
        if self.rotation is not None:
            self._check_rotation(observation, evidence, stage, previous_stage, entry, now)

    def _check_release_phase(self, observation, phase, now):
        """Opt-in pre_release_dwell_s: the giving jaw waits (``verifying``) for the whole dwell before it opens."""
        from .stationary_lift import _bilateral_contact
        if phase != (self.release_phase_seen[-1] if self.release_phase_seen else None):
            allowed = release_phases(self.task)
            if phase not in allowed or allowed.index(phase) != len(self.release_phase_seen):
                raise ValueError("pickup_trace_handover_release_phase_invalid")
            self.release_phase_seen.append(phase)
            if (len(self.release_phase_seen) == 2 and allowed[0] == RELEASE_VERIFYING_PHASE
                    and (self.release_entry_time is None
                         or now - self.release_entry_time < self.settings["pre_release_dwell_s"] - 1e-9)):
                raise ValueError("pickup_trace_handover_pre_release_dwell_short")
        if phase == RELEASE_VERIFYING_PHASE and not _bilateral_contact(observation, self.giving, self.object_path):
            # Close second, dwell, open first: the giving pinch is physical while the giving jaw is told to wait.
            raise ValueError("pickup_trace_handover_giving_pinch_missing")

    def _check_rotation(self, observation, evidence, stage, previous_stage, entry, now):
        """Replay the opt-in two-hand rotation: the recorded tilt/top-offset/rotation claims re-measured from the
        raw ring and receiving TCP poses, both pinches in every rotate row, and the at-top offset at the
        handover_release entry."""
        from .stationary_lift import _bilateral_contact
        rotation = evidence.get("rotation")
        if not isinstance(rotation, dict):
            raise ValueError("pickup_trace_handover_rotation_evidence_missing")
        ring = _measured_pose(observation.get("object_poses_w", {}).get(self.object_path), "handover_ring")
        tcp = _measured_pose(observation.get("tcp_poses_w", {}).get(self.receiving), "handover_receiving_tcp")
        entry_ring = (None if rotation.get("entry_ring_pose_w") is None
                      else _measured_pose(rotation["entry_ring_pose_w"], "rotate_entry_ring"))
        measured = rotation_measurement(self.rotation, ring, tcp, entry_ring)
        for key in ("ring_tilt_rad", "ring_tilt_valid", "receiving_offset_from_top_rad",
                    "receiving_offset_from_top_signed_rad", "at_top", "ring_rotation_rad", "receiving_grasp_point_tcp"):
            if not _guard_evidence_equal(rotation.get(key), measured[key]):
                raise ValueError("pickup_trace_handover_rotation_claim_mismatch")
        if entry and stage == HANDOVER_ROTATE_STAGE:
            # The rotate entry: the recorded receiving reference is this observation's measured grasp and the
            # recorded entry ring pose is this ring; both hands pinch.
            capture = self._relative(observation, self.receiving)
            recorded = _measured_pose(rotation.get("receiving_capture_object_to_tcp"), "rotate_receiving_capture")
            if (np.max(np.abs(capture - recorded)) > 1e-7 or entry_ring is None
                    or np.max(np.abs(entry_ring - ring)) > 1e-7):
                raise ValueError("pickup_trace_handover_rotate_capture_differs_from_trace")
            self.rotate_capture = capture
        if previous_stage == HANDOVER_ROTATE_STAGE or (entry and stage == HANDOVER_ROTATE_STAGE):
            if self.rotate_capture is None:
                raise ValueError("pickup_trace_handover_rotate_capture_missing")
            for arm in (self.giving, self.receiving):
                if not _bilateral_contact(observation, arm, self.object_path):
                    raise ValueError("pickup_trace_handover_rotate_pinch_missing")
            giving_capture = _measured_pose(evidence.get("giving", {}).get("capture_object_to_tcp"), "giving_capture")
            if (not self._retained(observation, self.giving, giving_capture)
                    or not self._retained(observation, self.receiving, self.rotate_capture)):
                raise ValueError("pickup_trace_handover_rotate_grasp_not_retained")
            if measured["ring_tilt_rad"] > self.rotation["max_tilt_rad"]:
                raise ValueError("pickup_trace_handover_rotate_ring_tilt")
        if entry and stage == "handover_release":
            # The rotation ended here: the receiving grasp point must be at the top of the rim, and the frozen
            # record must be this observation's measurement.
            if previous_stage != HANDOVER_ROTATE_STAGE:
                raise ValueError("pickup_trace_handover_stage_order_invalid")
            if not measured["at_top"]:
                raise ValueError("pickup_trace_handover_receiving_not_at_top")
            completed = rotation.get("completed")
            if (not isinstance(completed, dict) or completed.get("time_s") != now or not all(
                    _guard_evidence_equal(completed.get(key), measured[key]) for key in (
                        "receiving_offset_from_top_rad", "at_top", "ring_rotation_rad", "ring_tilt_rad"))):
                raise ValueError("pickup_trace_handover_rotation_claim_mismatch")

    def finish(self):
        if self.enabled and (len(self.seen) != len(self.order) or self.capture is None):
            raise ValueError("pickup_trace_handover_stages_missing")


def _raw_support_evidence(task, observation, object_path, jig_path):
    """The machine's support evidence (StationaryPlaceMachine._support), re-derived from one raw observation."""
    raw = observation.get("support_evidence", {}).get(object_path, {})
    support = {"available": False, "object_path": object_path, "support_path": jig_path,
        "source": raw.get("source"), "normal_force_n": None, "object_mass_kg": None,
        "minimum_force_n": None, "load_supported": False}
    force, mass = raw.get("force_w"), raw.get("object_mass_kg")
    if (raw.get("available") is True and raw.get("object_path") == object_path
            and raw.get("support_path") == jig_path
            and raw.get("source") == "physx_filtered_object_support_contacts"
            and isinstance(force, list) and len(force) == 3 and all(map(_finite_number, force))
            and _finite_number(mass) and mass > 0):
        normal = float(np.dot(force, task["placement_support"]["normal_w"]))
        minimum = task["placement_support"]["minimum_load_fraction"] * mass * 9.81
        support.update(available=True, normal_force_n=normal, object_mass_kg=mass,
            minimum_force_n=minimum, load_supported=normal >= minimum)
    return support


class _PlacementSupportFilterAdmission:
    """Replay the opt-in causal place support filter (placement_support.filter_s) from the raw support rows.

    The trace row carries the observation judged in the previous stage: a row judged in ``place`` adds its raw
    support sample to the replayed filter and must publish exactly the replayed evidence; every other row
    publishes None, and the row on which the machine entered ``place`` resets the filter at its observation
    time. A profile without the key may not publish the evidence at all.
    """
    def __init__(self, config):
        spec = config["stationary_task"]
        self.task = config["tasks"][spec["legacy_task_id"]]
        settings = self.task.get("placement_support", {})
        self.enabled = isinstance(settings, dict) and "filter_s" in settings
        self.expected = None
        if not self.enabled:
            return
        from .placement_support_filter import PlacementSupportFilter
        self.filter = PlacementSupportFilter(settings["filter_s"])
        self.object_path, self.jig_path = config["objects"]["crs"], config["objects"]["jig"]

    def check(self, row, previous_stage):
        observation, command = row.get("observation", {}), row.get("command", {})
        metrics = command.get("stationary_metrics")
        if not self.enabled:
            if isinstance(metrics, dict) and "placement_support_filter" in metrics:
                raise ValueError("pickup_support_filter_undeclared")
            return None
        now, stage = observation.get("time_s"), command.get("stage_id")
        if previous_stage == "place":
            try:
                expected = self.filter.update(now, _raw_support_evidence(
                    self.task, observation, self.object_path, self.jig_path))
            except ValueError as exc:
                raise ValueError("pickup_support_filter_replay_invalid") from exc
        else:
            expected = None
            if stage == "place":
                self.filter.reset(now)
        if (not isinstance(metrics, dict) or "placement_support_filter" not in metrics
                or not _guard_evidence_equal(metrics["placement_support_filter"], expected)):
            raise ValueError("pickup_support_filter_replay_mismatch")
        self.expected = expected
        return expected


class _SupportedGripReliefAdmission:
    """Replay the opt-in jaw command from raw support, poses and prior applied targets.

    This adds no placement acceptance exception. In particular, false support
    flags, a replaced capture or an unrecorded command jump cannot justify relief.
    With the runtime_v26 opt-in keys the replay also re-derives the filtered support
    (placement_support.filter_s) and the pre-seat pinch reduction from raw rows.
    """
    def __init__(self, config, grippers):
        from .supported_grip_relief import SupportedGripRelief, relief_task_options
        self.spec = config["stationary_task"]
        self.task = config["tasks"][self.spec["legacy_task_id"]]
        self.enabled = "supported_grip_relief" in self.task
        self.previous_time = self.previous_targets = self.jig_anchor = self.capture = None
        self.seen_place = False
        self.support_filter = None
        if not self.enabled:
            return
        self.object_path, self.jig_path = config["objects"]["crs"], config["objects"]["jig"]
        if not isinstance(grippers, dict) or set(grippers) != {"left", "right"}:
            raise ValueError("pickup_grip_relief_model_grippers_missing")
        self.grippers = grippers
        for arm, grip in grippers.items():
            if (not isinstance(grip, dict) or grip.get("joint_name") != f"{arm[0]}_1_dof_gripper_moving"
                    or not _finite_number(grip.get("speed_rad_s")) or grip["speed_rad_s"] <= 0):
                raise ValueError("pickup_grip_relief_model_grippers_invalid")
        from .object_symmetry import symmetry_axis, symmetry_pivot_radius
        self.symmetry_axis = symmetry_axis(self.task)
        self.replay = SupportedGripRelief(self.task["supported_grip_relief"],
            object_path=self.object_path, support_path=self.jig_path,
            capture_position_m=config["thresholds"]["capture_position_m"],
            capture_rotation_rad=config["thresholds"]["capture_rotation_rad"],
            object_symmetry_axis=self.symmetry_axis, object_pivot_radius_m=symmetry_pivot_radius(self.task),
            **relief_task_options(self.task))
        if "filter_s" in self.task.get("placement_support", {}):
            self.support_filter = _PlacementSupportFilterAdmission(config)

    @staticmethod
    def _same(actual, expected, label):
        if not _guard_evidence_equal(actual, expected):
            raise ValueError("pickup_grip_relief_" + label + "_mismatch")

    def _raw_metrics(self, observation):
        from .nepes_tasks import _errors
        from .snapshot import matrix_to_pose
        poses = observation.get("object_poses_w", {})
        jig = _measured_pose(poses.get(self.jig_path), "grip_relief_jig")
        obj = _measured_pose(poses.get(self.object_path), "grip_relief_object")
        relative = np.linalg.inv(jig) @ obj
        target = _measured_pose(self.task["target"]["pose"], "grip_relief_slot")
        if self.symmetry_axis is None:
            position, rotation = _errors(relative, target)
        else:
            from .object_symmetry import symmetric_pose_errors
            position, rotation, _ = symmetric_pose_errors(relative, target, self.symmetry_axis)
        jig_position, jig_rotation = _errors(jig, self.jig_anchor)
        support = _raw_support_evidence(self.task, observation, self.object_path, self.jig_path)
        cfg = self.spec
        gates = {
            "slot_pose_reached": position <= cfg["slot_position_tolerance_m"] and rotation <= cfg["slot_rotation_tolerance_rad"],
            "slot_normal_reached": abs(float(relative[1, 3] - target[1, 3])) <= cfg["slot_normal_position_tolerance_m"],
            "jig_anchor_retained": jig_position <= cfg["maximum_jig_position_drift_m"] and jig_rotation <= cfg["maximum_jig_rotation_drift_rad"],
            "jig_support_confirmed": support["load_supported"],
        }
        return {"measurement_time_s": observation["time_s"], "gates": gates,
                "support_evidence": support, "jig_anchor_pose_w": matrix_to_pose(self.jig_anchor)}

    def check(self, row, previous_stage):
        from copy import deepcopy
        from .handover import jaw_positions, observation_arm_for_stage, holding_arm_for_stage
        from .stationary_lift import _bilateral_contact
        from .task_control import gripper_closed_position, gripper_open_position
        observation, command = row["observation"], row["command"]
        if not self.enabled:
            if "supported_grip_relief" in command:
                raise ValueError("pickup_grip_relief_undeclared")
            return
        now, stage = observation.get("time_s"), command.get("stage_id")
        if (not _finite_number(now) or self.previous_time is not None
                and abs(now - self.previous_time - .02) > 1e-9):
            raise ValueError("pickup_grip_relief_control_clock_invalid")
        # Opt-in filtered support: replayed (and its published evidence checked) on every row.
        filtered = self.support_filter.check(row, previous_stage) if self.support_filter is not None else None
        applied = observation.get("commanded_upper_joint_positions", {})
        for grip in self.grippers.values():
            name = grip["joint_name"]
            if not _finite_number(applied.get(name)):
                raise ValueError("pickup_grip_relief_previous_applied_missing")
            if self.previous_targets is not None:
                self._same(applied[name], self.previous_targets.get(name), "causal_previous_applied")
        self.previous_time = now
        poses = observation.get("object_poses_w", {})
        if self.jig_anchor is None:
            self.jig_anchor = _measured_pose(poses.get(self.jig_path), "grip_relief_initial_jig")
        arm = (observation_arm_for_stage(self.task, stage) if "handover" in self.task else self.task["arm"])
        if command.get("arm") != arm:
            raise ValueError("pickup_grip_relief_wrong_controlled_arm")
        if (stage == "lift" and previous_stage == "close"
                or "handover" in self.task and stage == "handover_release" and previous_stage != stage):
            self.capture = (np.linalg.inv(_measured_pose(poses.get(self.object_path), "grip_relief_capture_object"))
                            @ _measured_pose(observation.get("tcp_poses_w", {}).get(arm), "grip_relief_capture_tcp"))
        if command.get("status") != "running":
            if "supported_grip_relief" in command:
                raise ValueError("pickup_grip_relief_terminal_command_present")
            return
        replay_command = deepcopy(command)
        if stage == "place":
            self.seen_place = True
            expected_holding = holding_arm_for_stage(self.task, stage) if "handover" in self.task else arm
            if command.get("holding_arm", arm) != expected_holding or expected_holding != arm:
                raise ValueError("pickup_grip_relief_wrong_holding_arm")
            if self.capture is None:
                raise ValueError("pickup_grip_relief_physical_capture_missing")
            captured = _measured_pose(command.get("retention_object_to_tcp"), "grip_relief_capture")
            if not np.allclose(captured, self.capture, rtol=0, atol=1e-7):
                raise ValueError("pickup_grip_relief_immutable_capture_changed")
            expected = self._raw_metrics(observation)
            if self.support_filter is not None:
                expected["placement_support_filter"] = filtered
            metrics = command.get("stationary_metrics", {})
            for key in ("measurement_time_s", "support_evidence", "jig_anchor_pose_w"):
                self._same(metrics.get(key), expected[key], "raw_" + key)
            for key, value in expected["gates"].items():
                self._same(metrics.get("gates", {}).get(key), value, "raw_" + key)
            replay_command["stationary_metrics"] = expected
            contact = observation.get("contact_evidence", {}).get(arm, {}).get(self.object_path, {})
            if contact.get("bilateral_contact_confirmed") is True and not _bilateral_contact(observation, arm, self.object_path):
                raise ValueError("pickup_grip_relief_bilateral_force_not_verified")
        open_goal, closed_goal = jaw_positions(self.task, arm)
        expected = self.replay.update(replay_command, observation, self.grippers[arm],
            open_position_rad=open_goal, closed_position_rad=closed_goal, dt=.02)
        self._same(command.get("supported_grip_relief"), expected, "replayed_evidence")
        emitted = row.get("commanded_upper_joint_positions", {})
        hand = "close" if expected["controlled"] else command.get("hand_command", "hold")
        hands = {**(command.get("hand_commands") or {}), arm: hand}
        for side, grip in self.grippers.items():
            name, goal = grip["joint_name"], applied[grip["joint_name"]]
            action = hands.get(side, "hold")
            if side == arm and expected["controlled"]:
                goal = expected["target_rad"]
            elif action in ("open", "close"):
                opening, closing = jaw_positions(self.task, side)
                target = (gripper_open_position(grip, opening) if action == "open" else
                          gripper_closed_position(grip, closing, opening))
                goal += float(np.clip(target - goal, -grip["speed_rad_s"] * .02, grip["speed_rad_s"] * .02))
            self._same(emitted.get(name), goal, "issued_" + side + "_jaw")
        self.previous_targets = deepcopy(emitted)

    def finish(self):
        if self.enabled and not self.seen_place:
            raise ValueError("pickup_grip_relief_place_evidence_missing")


def _check_pickup_trace(path, boundary, result, config, spec, upright_targets, initialized_crs, *, grippers=None):
    """Catalog-v2 task 1 keeps one row per control step and proves its recorded claims.

    Rows before the boundary may only prepare the upright stance: through the
    boundary row (the fresh navigate_source -> reach_posture transition) every
    observation reports exactly the recorded upright-start posture command, so
    nothing pre-posed the body before recording. The CRS keeps its first
    measured pose, itself the initialized cart pose, until the jaws close. Inside
    the recording only a declared side-step between distinct stations
    (navigate_target) may command the base; every other running sample keeps an
    exactly zero base command. The JIG never leaves its first measured pose,
    every stage entry time is re-read from the trace, and the claimed lift is
    re-measured from the CRS poses the machine consumed while lifting (each
    observation after the lift entry, through the one that completed the lift).
    """
    from .snapshot import matrix_to_pose
    task = config["tasks"][spec["legacy_task_id"]]
    arm, object_path, jig_path = task["arm"], config["objects"]["crs"], config["objects"]["jig"]
    # The hand that places the ring: the task arm, or the receiving arm of an opt-in handover.
    place_arm = task["handover"]["receiving_arm"] if "handover" in task else arm
    labels = _pickup_labels(task)
    recorded_stages = tuple(stage for stage, _ in _pickup_recorded(task))
    start = boundary["start_observation_time_s"]
    machine = boundary["readiness_evidence"]["machine_readiness"]
    transport = recorded_inter_station_transport(config)
    transport_admission = _PickupTransportAdmission(task, object_path)
    handover_admission = _HandoverAdmission(task, object_path, config["thresholds"])
    from .posture_hold_admission import AppliedHoldAdmission
    posture_hold_admission = AppliedHoldAdmission(task, spec, object_path, config["thresholds"],
                                                 _guard_evidence_equal, jig_path=jig_path)
    grip_relief_admission = _SupportedGripReliefAdmission(config, grippers)
    support_filter_admission = _PlacementSupportFilterAdmission(config)
    from .completed_transfer_continuity import CompletedTransferAdmission
    completed_transfer_admission = CompletedTransferAdmission(task, _guard_evidence_equal)
    from .upper_body_control import check_transfer_cartesian_gain_evidence
    stages, times, lift = [], {}, 0.
    previous_time = previous_stage = initial_jig = initial_crs = release_tcp = lift_origin = first = terminal = None
    anchored_slot_target = None
    with path.open() as stream:
        for line in stream:
            if terminal is not None:
                raise ValueError("pickup_trace_continues_after_completion")
            row = json.loads(line)
            observation, command = row.get("observation", {}), row.get("command", {})
            now, stage, status = observation.get("time_s"), command.get("stage_id"), command.get("status")
            if not _finite_nonnegative(now) or previous_time is not None and now <= previous_time:
                raise ValueError("pickup_trace_observation_clock_invalid")
            if (not isinstance(stage, str) or stage not in labels
                    or command.get("stationary_phase") != labels[stage]):
                raise ValueError("pickup_trace_stage_or_label_invalid")
            poses = observation.get("object_poses_w", {})
            jig, crs = _measured_pose(poses.get(jig_path), "trace_jig"), _measured_pose(poses.get(object_path), "trace_crs")
            if initial_jig is None:
                initial_jig = jig
                anchored_slot_target = matrix_to_pose(initial_jig @ _measured_pose(task["target"]["pose"], "configured_slot"))
            drift = _pose_errors(jig, initial_jig)
            if drift[0] > spec["maximum_jig_position_drift_m"] or drift[1] > spec["maximum_jig_rotation_drift_rad"]:
                raise ValueError("pickup_trace_jig_displaced")
            if initial_crs is None:
                initial_crs = crs
                anchor = _pose_errors(crs, initialized_crs)
                if (anchor[0] > _SOURCE_ANCHOR_BOUNDS["max_source_object_position_error_m"]
                        or anchor[1] > _SOURCE_ANCHOR_BOUNDS["max_source_object_rotation_error_rad"]):
                    raise ValueError("pickup_trace_source_object_not_at_initialized_pose")
                if not np.allclose(crs, _measured_pose(machine["source_object_anchor_pose_w"], "source_anchor"),
                                   atol=1e-9, rtol=0):
                    raise ValueError("pickup_trace_source_anchor_differs_from_readiness")
            # The machine judged this observation in previous_stage (the first one in prerequisites).
            if stage in _PICKUP_SOURCE_ANCHORED or previous_stage in _PICKUP_SOURCE_ANCHORED:
                shift = _pose_errors(crs, initial_crs)
                if (shift[0] > _SOURCE_ANCHOR_BOUNDS["max_source_object_position_error_m"]
                        or shift[1] > _SOURCE_ANCHOR_BOUNDS["max_source_object_rotation_error_rad"]):
                    raise ValueError("pickup_trace_source_object_displaced")
            if previous_stage == "lift":
                lift = max(lift, float(crs[2, 3] - lift_origin))
            if not stages or stages[-1] != stage:
                stages.append(stage)
                times[stage] = now
                if stage == "lift":
                    lift_origin = crs[2, 3]
                elif stage == "release":
                    release_tcp = _measured_pose(observation.get("tcp_poses_w", {}).get(place_arm), "release_tcp")
            recorded = now >= start - 1e-9
            if first is None:
                # Every preparation observation, and the boundary row's, reports the
                # posture command applied in the previous step: the upright start.
                commanded = observation.get("commanded_upper_joint_positions")
                if (not isinstance(commanded, dict) or any(
                        not _finite_number(commanded.get(name))
                        or abs(commanded[name] - target) > _UPRIGHT_COMMAND_TOLERANCE_RAD
                        for name, target in upright_targets.items())):
                    raise ValueError("pickup_trace_preparation_changed_the_upright_command")
            if recorded and first is None:
                first = row
                _check_upright_boundary_row(row, previous_stage, boundary, task, object_path)
            if recorded and stage not in recorded_stages:
                raise ValueError("pickup_recording_contains_preparation_stage")
            if recorded:
                transport_admission.check(row, previous_stage, slot_target_pose_w=anchored_slot_target)
            transport_admission.check_source_hold_preparation(row, previous_stage)
            if status == "running":
                velocity = command.get("base_velocity_applied")
                # Opt-in single step: only the replayed stepping rows of navigate_target may command the base.
                stepping = recorded and transport_admission.single_step_stepping
                # A no-walk profile keeps the zero base command through the preparation too.
                if (recorded or transport_admission.no_walk or transport_admission.source_hold) and (not isinstance(velocity, list) or len(velocity) != 3
                                 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                                        or v != 0 and not (transport and stage == "navigate_target") and not stepping
                                        for v in velocity)):
                    raise ValueError("pickup_recorded_base_command_not_zero")
            elif status == "succeeded" and recorded:
                terminal = row
            else:
                raise ValueError("pickup_trace_contains_failed_control")
            posture_hold_admission.check(row, previous_stage)
            handover_admission.check(row, previous_stage)
            support_filter_admission.check(row, previous_stage)
            grip_relief_admission.check(row, previous_stage)
            completed_transfer_admission.check(row, previous_stage)
            check_transfer_cartesian_gain_evidence(task, command)
            previous_time, previous_stage = now, stage
    posture_hold_admission.finish()
    handover_admission.finish()
    grip_relief_admission.finish()
    completed_transfer_admission.finish()
    if (first is None or terminal is None or tuple(stages) != tuple(labels) or times["reach_posture"] != start
            or terminal["command"].get("stationary_metrics") != result.get("stationary_metrics")):
        raise ValueError("pickup_trace_completion_or_phase_evidence_mismatch")
    metrics = result["stationary_metrics"]
    if metrics["stage_entry_times_s"] != times:
        raise ValueError("pickup_trace_stage_times_disagree")
    if not abs(metrics["pickup_contract"]["lift_m"] - lift) <= 1e-9:
        raise ValueError("pickup_trace_lift_differs_from_metrics")
    _check_released_terminal(terminal["observation"], metrics, release_tcp, initial_jig, config, arm=place_arm)


def _finite_number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _upright_record_matches(record, declared, scope, names):
    """One upright-start write record: exactly the declared root pose and overrides, plus the declared gaze.

    The gaze neck targets are computed at the boundary, so only their shape is
    pinned; a stance without a declared gaze records none. The recorded posture
    command covers exactly the posture-path joints, and its neck equals the
    recorded gaze or the declared neck override.
    """
    if not isinstance(record, dict) or set(record) != _UPRIGHT_RECORD_KEYS:
        return False
    pose, applied, gaze = record["robot_root_pose_w"], record["joint_overrides_applied"], record["head_gaze"]
    targets, overrides = record["upright_joint_targets"], declared["joint_overrides"]
    neck = gaze.get("joint_targets_rad") if isinstance(gaze, dict) else None
    commanded_neck = neck if declared["head_gaze"] else {name: value for name, value in overrides.items()
                                                         if name in _NECK_JOINTS}
    return (record["scope"] == scope and record["demonstration_pose_write"] is False
            and isinstance(pose, list) and len(pose) == 7 and all(map(_finite_number, pose))
            and all(abs(value - target) <= 1e-6 for value, target in zip(pose, declared["robot_root_pose_w"]))
            and isinstance(applied, dict) and set(applied) == set(overrides)
            and all(_finite_number(value) and abs(value - overrides[name]) <= 1e-6 for name, value in applied.items())
            and (isinstance(neck, dict) and set(neck) == _NECK_JOINTS and all(map(_finite_number, neck.values()))
                 if declared["head_gaze"] else gaze is None)
            and isinstance(targets, dict) and set(targets) == set(names) and all(map(_finite_number, targets.values()))
            and all(abs(targets[name] - value) <= 1e-6 for name, value in commanded_neck.items()))


def _check_upright_initialization(config, spec, report, result, provenance, path):
    """Catalog-v2 task 1 begins at the declared upright stance; nothing may pre-pose it.

    The startup and the episode boundary must both record the declared root
    pose and joint overrides, and neither may carry a prepared held snapshot,
    an active-arm initialization or an upper-posture write. Both reset to the
    same env-default joints, so their posture commands agree off the gazing
    neck, and the machine's readiness captured exactly the episode's command.
    Returns the recorded posture command and the initialized CRS pose.
    """
    task = config["tasks"][spec["legacy_task_id"]]
    declared, names = task["upright_start"], task["posture_paths"]["reach_posture"]["joint_names"]
    initialization = read_json(path)
    episode = initialization.get("upright_start") if isinstance(initialization, dict) else None
    startup = report.get("startup_upright_start")
    if (not isinstance(initialization, dict)
            or not _upright_record_matches(episode, declared, _UPRIGHT_EPISODE_SCOPE, names)
            or not _upright_record_matches(startup, declared, _UPRIGHT_STARTUP_SCOPE, names)
            or any(abs(episode["upright_joint_targets"][name] - startup["upright_joint_targets"][name])
                   > _UPRIGHT_COMMAND_TOLERANCE_RAD
                   for name in names if not (declared["head_gaze"] and name in _NECK_JOINTS))
            or any(key in initialization for key in ("prepared_grasp_initialization", "arm_initialization"))
            or initialization.get("upper_posture_initialization") is not None
            or initialization.get("initial_upper_joint_positions")
            or any(key in report for key in ("startup_prepared_grasp_initialization",
                                             "startup_active_arm_initialization"))
            or report.get("startup_upper_joint_positions")
            or result.get("initialization") != initialization
            or provenance.get("initialization") != initialization):
        raise ValueError("pickup_upright_initialization_provenance_mismatch")
    captured = result["recording_boundary"]["readiness_evidence"]["machine_readiness"]["upright_command_targets"]
    if any(abs(captured[name] - episode["upright_joint_targets"][name]) > _UPRIGHT_COMMAND_TOLERANCE_RAD
           for name in names):
        raise ValueError("pickup_upright_command_differs_from_the_recorded_start")
    try:
        crs = _measured_pose(initialization["object_poses_w"][config["objects"]["crs"]], "initialized_crs")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("pickup_upright_initialization_provenance_mismatch") from exc
    return dict(episode["upright_joint_targets"]), crs


def validate_episode(run_dir, config_sha256):
    """Verify goal/controller identity, completed physics evidence and every RGB frame."""
    from PIL import Image

    run = Path(run_dir).resolve()
    config = load_stationary_config(_safe_file(run / "config.json", run), expected_sha256=config_sha256)
    spec = validate_stationary_spec(config)
    contract = _goal_controller_contract(config)
    report = read_json(_safe_file(run / "run.json", run))
    # Stage/waypoint calibration and partial-path profiles are never demonstrations.
    if (report.get("complete") is not True or report.get("mode") != "execute"
            or report.get("episode_scope") != STATIONARY_SCOPE
            or report.get("execution_mode") != "independent"
            or report.get("config_sha256") != config_sha256
            or report.get("stationary_task") != spec
            or report.get("diagnostic_stop_after") is not None
            or report.get("diagnostic_stop_after_waypoint") is not None
            or "required_diagnostic_stop_after_waypoint" in config["tasks"][spec["legacy_task_id"]]
            or not _controller_contract(report.get("controller_contract"), contract)):
        raise ValueError("incomplete_or_wrong_stationary_run_contract")
    results = report.get("episodes", [])
    if len(results) != 1:
        raise ValueError("one_independent_episode_per_attempt_required")
    result = results[0]
    if (result.get("task_id") != spec["id"] or result.get("legacy_task_id") != spec["legacy_task_id"]
            or result.get("success") is not True or result.get("full_task_completed") is not True
            or result.get("legacy_full_task_completed") is not False
            or result.get("episode_scope") != STATIONARY_SCOPE or result.get("stationary_task") != spec
            or not _controller_contract(result.get("controller_contract"), contract)):
        raise ValueError("stationary_goal_success_not_verified")
    if spec["goal"] == "lift_hold":
        _check_hold(result.get("stationary_metrics"), spec, config)
    else:
        _check_placement(result.get("stationary_metrics"), spec, config)
    name = result.get("episode_id", "")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("unsafe_episode_id")
    runtime = run / name
    if read_json(_safe_file(runtime / "result.json", run)) != result:
        raise ValueError("run_and_episode_result_disagree")
    episode = run / "staging/episodes" / name
    metadata = read_json(_safe_file(episode / "episode.json", run))
    provenance = metadata.get("provenance", {})
    if (metadata.get("episode_id") != name or metadata.get("success") is not True
            or metadata.get("status") != "success" or metadata.get("task_id") != spec["number"]
            or metadata.get("metrics") != result or provenance.get("episode_scope") != STATIONARY_SCOPE
            or provenance.get("stationary_task") != spec
            or not _controller_contract(provenance.get("controller_contract"), contract)):
        raise ValueError("committed_recording_goal_contract_mismatch")
    boundary = _check_recording_boundary(config, result, provenance)
    schema = metadata.get("schema", {})
    try:
        cameras = validate_camera_sizes(schema.get("camera_sizes"), require_standard_sizes=True)
        if "camera_layout" in config:
            validate_declared_camera_layout(cameras, config["camera_layout"])
    except ValueError as error:
        raise ValueError("recording_schema_mismatch") from error
    if (schema.get("schema_version") != "igris_c_nepes_episode_staging_v1"
            or schema.get("fps") != 30 or schema.get("physics_hz") != 250 or schema.get("control_hz") != 50
            or set(metadata.get("camera_configuration", {})) != set(cameras)):
        raise ValueError("recording_schema_mismatch")
    for field in ("joint_names", "policy_action_names", "upper_target_names"):
        names = schema.get(field)
        if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
            raise ValueError("recording_joint_or_action_schema_invalid")
    if len(schema["policy_action_names"]) != 13:
        raise ValueError("hugwbc_policy_action_schema_invalid")
    model_manifest = Path(report.get("robot_model_manifest", ""))
    _safe_file(model_manifest, run)
    # The jaw model is part of the admitted contract: the profile's declaration (absent = authored_v0),
    # never a CLI-only selection or a development pad-material override.
    declared_gripper = config.get("gripper_variant", DEFAULT_GRIPPER_VARIANT)
    gripper = report.get("robot_gripper")
    manifest = read_json(model_manifest)
    if (not isinstance(manifest, dict) or (gripper is not None and not isinstance(gripper, dict))
            or (gripper is None and declared_gripper != DEFAULT_GRIPPER_VARIANT)
            or (gripper is not None and (gripper.get("variant") != declared_gripper
                                         or gripper.get("dev_pad_material_override") is not None))
            or manifest.get("gripper_variant", DEFAULT_GRIPPER_VARIANT) != declared_gripper):
        raise ValueError("robot_gripper_variant_not_the_profile_declaration")
    if declared_gripper != DEFAULT_GRIPPER_VARIANT:
        # A pad model must be the clean spec build (actuator, pad material, pinned package, no override)
        # whose fingerprint the run recorded.
        try:
            verify_gripper_manifest(manifest, declared_gripper, config.get("gripper_pad_contact_patch"),
                                    config.get("gripper_pad_compliant_contact"),
                                    config.get("gripper_pad_contact_spheres"))
        except ValueError as error:
            raise ValueError(f"robot_gripper_model_not_the_variant_spec: {error}") from None
        if gripper.get("robot_model_fingerprint") != manifest["fingerprint"]:
            raise ValueError("robot_gripper_model_not_the_variant_spec: run.json and manifest fingerprints differ")
    frames = _safe_file(episode / "frames.jsonl", run)
    if digest(frames) != metadata.get("frames_sha256"):
        raise ValueError("frame_hash_mismatch")
    count, previous, previous_step, first_time, first_step = 0, None, None, None, None
    recorded_phases = set()
    image_hash = sha256()
    # Pickup frames are labelled (phase_name) while the boundary names the
    # stage id, so the first frame must carry the label of the start stage.
    pickup, phase_order = spec["goal"] == "pickup_load_jig", 0
    pickup_task = config["tasks"][spec["legacy_task_id"]]
    # The profile's recorded stages and labels: the opt-in handover inserts six labelled phases.
    pickup_recorded = _pickup_recorded(pickup_task)
    pickup_recorded_stages = tuple(stage for stage, _ in pickup_recorded)
    pickup_recorded_phases = tuple(label for _, label in pickup_recorded)
    handover_frames = pickup and "handover" in pickup_task
    if handover_frames:
        from .handover import holding_arm_for_stage
    with frames.open() as stream:
        for count, line in enumerate(stream, 1):
            row = json.loads(line)
            recorded_phases.add(row.get("phase_name"))
            if row["frame_index"] != count - 1 or abs(row["timestamp"] - (count - 1) / 30) > 1e-8:
                raise ValueError("recording_frame_sequence_mismatch")
            actual, step = row["sim_time_s"], row["physics_step"]
            if type(step) is not int or step < 0:
                raise ValueError("invalid_recording_physics_step")
            if first_time is None:
                first_step, first_time = step, actual
                if boundary is not None and (step != 0 or row.get("phase_name") != (
                        _PICKUP_LABELS[boundary["start_stage"]] if pickup else boundary["start_stage"])
                        or not math.isfinite(actual) or abs(actual - boundary["start_sim_time_s"]) > 1e-5):
                    raise ValueError("prepared_recording_first_frame_mismatch")
            if boundary is not None and not pickup and row.get("phase_name") in {"prerequisites", "navigate_source", "station_alignment", "pregrasp"}:
                raise ValueError("prepared_recording_contains_preparation_phase")
            if spec["goal"] == "load_jig" and row.get("phase_name") not in _PLACEMENT_PHASES:
                raise ValueError("placement_recording_contains_preparation_or_navigation")
            if pickup:
                phase = row.get("phase_name")
                if phase in _PICKUP_PREPARATION_NAMES:
                    raise ValueError("prepared_recording_contains_preparation_phase")
                if phase not in pickup_recorded_phases or row.get("phase_id") != STAGES.index(
                        pickup_recorded_stages[pickup_recorded_phases.index(phase)]):
                    raise ValueError("pickup_recording_phase_label_or_id_invalid")
                if pickup_recorded_phases.index(phase) < phase_order:
                    raise ValueError("pickup_recording_phase_order_reversed")
                phase_order = pickup_recorded_phases.index(phase)
                if handover_frames and row.get("holding_arm") != holding_arm_for_stage(
                        pickup_task, pickup_recorded_stages[phase_order]):
                    # Per-frame provenance of the arm switch: which hand holds the ring in this frame.
                    raise ValueError("pickup_recording_holding_arm_provenance_invalid")
            expected_step = first_step + ((count - 1) * 250 + 29) // 30
            if step != expected_step:
                raise ValueError("recording_physics_tick_mismatch")
            validate_physics_clock(sample_time=actual, first_time=first_time,
                elapsed_ticks=step - first_step, physics_hz=250, previous_time=previous,
                interval_ticks=None if previous_step is None else step - previous_step)
            previous, previous_step = actual, step
            age = actual - row["command_time_s"]
            if not math.isfinite(age) or not -1e-5 <= age <= .02001:
                raise ValueError("recording_command_time_mismatch")
            for field, names in (("joint_positions", "joint_names"), ("joint_velocities", "joint_names"),
                                 ("policy_action", "policy_action_names"), ("upper_targets", "upper_target_names")):
                values = row[field]
                if len(values) != len(schema[names]) or not all(math.isfinite(v) for v in values):
                    raise ValueError("recording_state_or_action_mismatch")
            if set(row["images"]) != set(cameras) or set(row["cameras"]) != set(cameras):
                raise ValueError("recording_camera_set_mismatch")
            for camera, size in cameras.items():
                relative = f"images/{camera}/{count - 1:06d}.png"
                if row["images"][camera] != relative:
                    raise ValueError("unexpected_rgb_path")
                path = _safe_file(episode / relative, run)
                with Image.open(path) as picture:
                    if picture.mode != "RGB" or picture.size != size:
                        raise ValueError("incorrect_rgb_mode_or_resolution")
                    picture.verify()
                image_hash.update((relative + ":" + digest(path) + "\n").encode())
                camera_info = row["cameras"][camera]
                rendered = camera_info["render_time_s"]
                if not math.isfinite(rendered) or abs(rendered - actual) > 1e-5:
                    raise ValueError("stale_camera")
                pose, intrinsic = camera_info["pose_w_xyz_wxyz"], camera_info["intrinsics"]
                if (len(pose) != 7 or not all(math.isfinite(v) for v in pose)
                        or len(intrinsic) != 3 or any(len(r) != 3 or not all(math.isfinite(v) for v in r) for r in intrinsic)):
                    raise ValueError("invalid_camera_pose_or_intrinsics")
    if count < 1 or count != metadata.get("frame_count"):
        raise ValueError("empty_or_incomplete_recording")
    if spec["goal"] == "load_jig":
        if recorded_phases != set(_PLACEMENT_PHASES):
            raise ValueError("placement_recording_omits_required_physical_phase")
        required_span = result["stationary_metrics"]["measurement_time_s"] - boundary["start_observation_time_s"]
        if previous - first_time + 1 / 30 + 2 / 250 < required_span:
            raise ValueError("placement_recording_ends_before_measured_completion")
    if pickup:
        if recorded_phases != set(pickup_recorded_phases):
            raise ValueError("pickup_recording_omits_required_physical_phase")
        # The exporter takes phase names from this map and the instruction from the profile title.
        if metadata.get("phases") != {str(STAGES.index(stage)): label for stage, label in pickup_recorded}:
            raise ValueError("pickup_recording_phase_metadata_mismatch")
        if metadata.get("task") != config["tasks"][spec["legacy_task_id"]].get("title", spec["legacy_task_id"]):
            raise ValueError("pickup_recording_language_instruction_mismatch")
        required_span = result["stationary_metrics"]["measurement_time_s"] - boundary["start_observation_time_s"]
        if previous - first_time + 1 / 30 + 2 / 250 < required_span:
            raise ValueError("pickup_recording_ends_before_measured_completion")
    if boundary is not None:
        # The capture clock is the simulator's float32-accumulated time (validate_physics_clock budgets
        # its roundoff the same way), while attempt/preparation durations are tick-exact: a 300 s
        # episode drifts about 1.5e-5 s (run_51). Budget the comparison by the prepared span.
        prepared_span = (boundary["preparation_duration_s"] + result["recorded_duration_s"]
                         if _finite_nonnegative(result.get("recorded_duration_s")) else math.inf)
        span_budget = max(1e-5, 4. * abs(prepared_span) * float(np.finfo(np.float32).eps)) if math.isfinite(prepared_span) else 0.
        if (type(result.get("recorded_frame_count")) is not int or result["recorded_frame_count"] != count
                or not _finite_nonnegative(result.get("recorded_duration_s"))
                or abs(result["recorded_duration_s"] - (previous - first_time)) > 1e-6
                or not _finite_nonnegative(result.get("attempt_duration_s"))
                or result["attempt_duration_s"] != result.get("duration_s")
                or result["attempt_duration_s"] + span_budget < prepared_span):
            raise ValueError("prepared_recording_duration_or_count_mismatch")
    files = {"result": runtime / "result.json", "trace": runtime / "trace.jsonl",
             "initialization": runtime / "initialization.json", "metadata": episode / "episode.json",
             "robot_model_manifest": model_manifest, "run": run / "run.json"}
    for path in files.values():
        if not _safe_file(path, run).stat().st_size:
            raise ValueError("empty_physical_evidence")
    if spec["goal"] == "load_jig":
        from .stationary_initialization import validate_prepared_grasp_initialization, prepared_grasp_provenance
        declared = prepared_grasp_provenance(validate_prepared_grasp_initialization(config))
        initialization = read_json(files["initialization"])
        if (not isinstance(initialization, dict)
                or initialization.get("prepared_grasp_initialization") != declared
                or report.get("startup_prepared_grasp_initialization") != declared
                or result.get("initialization") != initialization
                or provenance.get("initialization") != initialization):
            raise ValueError("placement_held_initialization_provenance_mismatch")
        _check_placement_trace(files["trace"], boundary, result, config)
    elif pickup:
        upright_targets, initialized_crs = _check_upright_initialization(
            config, spec, report, result, provenance, files["initialization"])
        _check_pickup_trace(files["trace"], boundary, result, config, spec, upright_targets, initialized_crs,
                            grippers=manifest.get("grippers"))
    return {"run_dir": str(run), "episode_id": name, "task_id": spec["id"],
            "episode_dir": str(episode), "frame_count": count, "config_sha256": config_sha256,
            "episode_scope": STATIONARY_SCOPE, "frames_sha256": digest(frames),
            "images_sha256": image_hash.hexdigest(), "robot_model_manifest": str(model_manifest),
            "artifact_sha256": {key: digest(path) for key, path in files.items()}}


def summarize(plan, attempts):
    successes = sum(attempt.get("accepted") is True for attempt in attempts)
    task_id = plan["task"]["id"]
    return {"task_id": task_id, "episode_scope": STATIONARY_SCOPE,
            "successes": successes, "requested_successes": plan["successes"],
            "attempts": len(attempts), "max_attempts": plan["max_attempts"],
            "quota_complete": successes >= plan["successes"], "five_task_coverage": False,
            "tasks": [{**dict(task), "status": ("measured_recorded_success" if successes else "candidate_unverified")
                       if task["id"] == task_id else
                       "configured_not_evaluated_in_this_collection" if task["configured"] else
                       "not_configured"} for task in task_catalog(_plan_catalog(plan))]}


def collect(plan, output, *, execute=False, launcher=run_child, dataset_python=None, resume=False,
            retry_after_failure_limit=False):
    """Dry-run by default; execute sequential fresh native attempts when requested."""
    if plan.get("storage_options", {}).get("storage_mode") == "compact":
        from .compact_collection import collect_compact
        return collect_compact(plan, output, execute=execute, launcher=launcher,
                               dataset_python=dataset_python, resume=resume,
                               retry_after_failure_limit=retry_after_failure_limit)
    if retry_after_failure_limit:
        raise ValueError("--retry-after-failure-limit is supported for compact collections only")
    if resume:
        raise ValueError("--resume is supported for compact collections only")
    output = Path(output).resolve()
    if not execute:
        return {**summarize(plan, []), "dry_run": True,
                "capture_options": _capture_options(**plan.get("capture_options", {})),
                "command": attempt_command(plan, output / "runs/attempt_000000", plan["seed"])}
    if output.exists():
        raise FileExistsError("Collection output exists; choose a new output to preserve its evidence")
    output.mkdir(parents=True)
    state = {"schema_version": COLLECTION_SCHEMA, "plan": plan, "plan_sha256": config_digest(plan),
             "attempts": [], "complete": False}
    manifest = output / "collection.json"
    write_json(manifest, state)
    while (len(state["attempts"]) < plan["max_attempts"]
           and not summarize(plan, state["attempts"])["quota_complete"]
           and not (output / "STOP").exists()):
        load_stationary_config(plan["config"], expected_sha256=plan["config_sha256"])
        if digest(plan["config"]) != plan["config_source_sha256"]:
            raise ValueError("Source configuration changed during collection")
        if _algorithm_fingerprints() != plan["algorithm_sha256"]:
            raise ValueError("Runtime source changed during collection; use a new plan/output")
        index = len(state["attempts"])
        run = output / "runs" / f"attempt_{index:06d}"
        attempt = {"index": index, "run_dir": str(run), "seed": plan["seed"] + index,
                   "accepted": False, "status": "running"}
        attempt["command"] = attempt_command(plan, run, attempt["seed"])
        state["attempts"].append(attempt)
        write_json(manifest, state)
        def started(pid):
            attempt["pid"] = pid
            write_json(manifest, state)
        try:
            outcome = launcher(attempt["command"], log_path=output / "logs" / f"attempt_{index:06d}.log",
                               stop_path=output / "STOP", run_dir=run,
                               max_wall_seconds=plan["max_wall_seconds"], on_start=started)
            attempt.update(outcome)
            if outcome["returncode"] != 0 or outcome.get("timed_out") or outcome.get("forced_termination"):
                raise ValueError(f"native_process_failed:{outcome['returncode']}")
            if _algorithm_fingerprints() != plan["algorithm_sha256"]:
                raise ValueError("Runtime source changed during attempt")
            attempt["evidence"] = validate_episode(run, plan["config_sha256"])
            attempt["accepted"] = True
        except (ValueError, TypeError, KeyError, OSError) as exc:
            attempt["failure_reason"] = str(exc)
        attempt["status"] = "complete"
        state["summary"] = summarize(plan, state["attempts"])
        write_json(manifest, state)
    state.update(complete=True, operator_stopped=(output / "STOP").exists(),
                 summary=summarize(plan, state["attempts"]))
    write_json(manifest, state)
    return state


_IMAGE_LINK_COPY_FALLBACK = frozenset((
    errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP, errno.ENOSYS, errno.EMLINK,
))


def _copy_checked(source, target, *, link_image=False):
    """Copy evidence, or link immutable RGB for the temporary read-only exporter."""
    before = digest(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "copy"
    if link_image:
        try:
            # Never chmod a link: that would change the original image inode.
            os.link(source, target, follow_symlinks=False)
            mode = "hardlink"
        except OSError as error:
            if error.errno not in _IMAGE_LINK_COPY_FALLBACK:
                raise
            shutil.copy2(source, target)
    else:
        shutil.copy2(source, target)
    if digest(target) != before or digest(source) != before:
        raise ValueError("Evidence changed while staging")
    return mode


def merge_successes(collection_root, destination, *, image_link_mode=False):
    """Copy accepted episodes; optionally link PNGs for temporary read-only export.

    Public merges retain independent image copies by default. The internal
    exporter may share only immutable RGB inodes with completed native runs;
    metadata, frames and evidence always remain independently copied.
    """
    if type(image_link_mode) is not bool:
        raise ValueError("image_link_mode must be a boolean")
    collection_root, destination = Path(collection_root).resolve(), Path(destination).resolve()
    state = read_json(collection_root / "collection.json")
    if (state.get("schema_version") != COLLECTION_SCHEMA or state.get("complete") is not True
            or state.get("plan_sha256") != config_digest(state["plan"])
            or not summarize(state["plan"], state["attempts"])["quota_complete"]):
        raise ValueError("Stationary collection quota is incomplete or its contract changed")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": "igris_c_nepes_stationary_merge_v1", "episode_scope": STATIONARY_SCOPE,
                "source_collection_sha256": digest(collection_root / "collection.json"),
                "stationary_task": state["plan"]["stationary_task"],
                "algorithm_sha256": state["plan"]["algorithm_sha256"], "episodes": []}
    if image_link_mode:
        manifest["temporary_image_staging"] = {
            "requested_mode": "hardlink_with_checked_copy_fallback",
            "read_only_consumer": "robros_motion_planning.dataset",
            "hardlinked_images": 0, "copied_images": 0,
        }
    schema = None
    with tempfile.TemporaryDirectory(prefix=".stationary-merge-", dir=destination.parent) as temporary:
        merged = Path(temporary) / "staging"
        for attempt in state["attempts"]:
            if not attempt.get("accepted"):
                continue
            if attempt.get("returncode") != 0 or attempt.get("timed_out") or attempt.get("forced_termination"):
                raise ValueError("Accepted episode lacks a normal native process exit")
            evidence = validate_episode(attempt["run_dir"], state["plan"]["config_sha256"])
            if evidence != attempt.get("evidence"):
                raise ValueError("Accepted evidence changed since admission")
            original = Path(evidence["episode_dir"])
            report = read_json(original / "episode.json")
            if schema is not None and schema != report["schema"]:
                raise ValueError("Incompatible recording schemas")
            schema = report["schema"]
            name = f"stationary{state['plan']['task']['number']}_attempt{attempt['index']:06d}_{evidence['frames_sha256'][:12]}"
            target = merged / "episodes" / name
            for path in sorted(original.rglob("*")):
                if path.is_symlink():
                    raise ValueError("Source episode symlink rejected")
                if path.is_file():
                    relative = path.relative_to(original)
                    image = (len(relative.parts) == 3 and relative.parts[0] == "images"
                             and relative.parts[1] in report["schema"]["camera_sizes"]
                             and relative.suffix == ".png")
                    mode = _copy_checked(path, target / relative, link_image=image_link_mode and image)
                    if image_link_mode and image:
                        key = "hardlinked_images" if mode == "hardlink" else "copied_images"
                        manifest["temporary_image_staging"][key] += 1
            raw = merged / "evidence" / name
            runtime = Path(evidence["run_dir"]) / evidence["episode_id"]
            sources = {"original_episode.json": original / "episode.json",
                       "result.json": runtime / "result.json", "trace.jsonl": runtime / "trace.jsonl",
                       "initialization.json": runtime / "initialization.json",
                       "config.json": Path(evidence["run_dir"]) / "config.json",
                       "run.json": Path(evidence["run_dir"]) / "run.json",
                       "robot_model_manifest.json": Path(evidence["robot_model_manifest"])}
            for filename, path in sources.items():
                _copy_checked(path, raw / filename)
            report["episode_id"] = name
            report["provenance"]["collection_source"] = {
                "run_dir": evidence["run_dir"], "original_episode_id": evidence["episode_id"],
                "original_metadata_sha256": evidence["artifact_sha256"]["metadata"]}
            write_json(target / "episode.json", report)
            if validate_episode(evidence["run_dir"], state["plan"]["config_sha256"]) != evidence:
                raise ValueError("Accepted evidence changed during merge")
            manifest["episodes"].append({"merged_episode_id": name, "source": evidence,
                "derived_metadata_sha256": digest(target / "episode.json")})
        write_json(merged / "stationary_collection_merge.json", manifest)
        merged.replace(destination)
    return manifest


def export_collection(collection_root, output, dataset_python, *, repo_id=None, allow_partial=False):
    """Export admitted successes through the existing isolated official v3 writer."""
    source, output = Path(collection_root).resolve(), Path(output).resolve()
    state = read_json(source / "collection.json")
    if state.get("plan", {}).get("storage_options", {}).get("storage_mode") == "compact":
        from .compact_collection import export_compact_collection
        return export_compact_collection(source, output, dataset_python, repo_id=repo_id,
                                         allow_partial=allow_partial)
    if allow_partial:
        raise ValueError("Partial export is supported for compact collections only")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".stationary-export-", dir=output.parent) as temporary:
        staging, official = Path(temporary) / "staging", Path(temporary) / "official"
        manifest = merge_successes(source, staging, image_link_mode=True)
        if repo_id is None:
            repo_id = "local/igris-c-stationary-" + manifest["stationary_task"]["id"].replace("_", "-")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
        try:
            subprocess.run([str(dataset_python), "-m", "robros_motion_planning.dataset",
                            "--staging-root", str(staging), "--output-root", str(official),
                            "--repo-id", repo_id], env=env, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Official dataset exporter failed with exit code {exc.returncode}") from exc
        # The official writer opens staging PNGs read-only. Revalidate native
        # hashes after that consumer before accepting an export using links.
        for item in manifest["episodes"]:
            evidence = item["source"]
            if validate_episode(evidence["run_dir"], evidence["config_sha256"]) != evidence:
                raise ValueError("Accepted evidence changed during export")
        exported = read_json(official / "meta/nepes_collection.json")["episodes"]
        index = {item["source_episode_id"]: item["episode_index"] for item in exported}
        if len(index) != len(exported) or set(index) != {item["merged_episode_id"] for item in manifest["episodes"]}:
            raise ValueError("Official exporter episode identity mismatch")
        for item in manifest["episodes"]:
            item["official_episode_index"] = index[item["merged_episode_id"]]
        shutil.copytree(staging / "evidence", official / "auxiliary/stationary_physics")
        write_json(official / "meta/nepes_stationary_collection.json", manifest)
        official.replace(output)
    return output
