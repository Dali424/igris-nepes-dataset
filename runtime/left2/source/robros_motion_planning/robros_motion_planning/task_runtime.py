"""Isaac/HugWBC integration for the five measured-state NEPES task machines.

This module is loaded only after SimulationApp. Draft workcell calibration is
retained in every report. Local servo and station navigation are experimental:
they do not certify collision-free transport or dynamic payload feasibility.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import numpy as np

from .task_control import (ArmServo, BaseCommandLimiter, CarryStabilizer, CarryTrackingGate,
                           LiveCommandSource, base_velocity_command, rotation_error,
                           gripper_open_position, gripper_is_open, arrival_limited_base_velocity,
                           base_command_in_heading_frame, turn_in_place_base_velocity)
from .snapshot import matrix_to_pose, pose_to_matrix
from .grasp_settling import PinchSettlingTracker
from .crs_manipulation import CrsPoseController, TcpPoseController, goal_object_from_tcp
from .posture_control import (PostureReference, MeasuredPostureController, pelvis_preserving_boundary_pose,
                              resolve_auxiliary_commands)
from .motion_watchdog import MotionProgressWatchdog, pose_goal_errors, posture_goal_errors
from .object_symmetry import spin_aligned_object, symmetric_pose_errors, symmetry_axis, symmetry_pivot_radius
from .auxiliary_support import AuxiliarySupportController
from .waypoint_postures import stage_posture
from .upright_runtime import (EPISODE_UPRIGHT_START_SCOPE, STARTUP_UPRIGHT_START_SCOPE, PostureStageExecutor,
                              diagnostic_stop_reason, gaze_target_w, motion_waypoint_index, object_pose_settings,
                              retreat_pose_settings, retreat_segment_key, upright_joint_writes, upright_overrides, upright_spawn_pose,
                              upright_start_provenance, validate_diagnostic_stop, validate_upright_task_runtime)
from .upright_runtime import (diagnostic_waypoint_stop_reason, validate_diagnostic_waypoint_stop,
                              validate_required_diagnostic_waypoint_stop)
from .upright_runtime import JOINT_PATH_STAGES, RetreatMotionFeedforward, place_two_speed_descent
from .handover import HANDOVER_STAGE_SET, handover_contract, posture_path_for_stage
from .handover import jaw_positions as handover_jaw_positions
from .supported_grip_relief import SupportedGripRelief, relief_task_options, supported_grip_relief_contract


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _with_ik_backend(report):
    """Pink release: refresh run.json's upper-body IK record (live status, failure, worker exit code)."""
    if "upper_body_ik_backend" in report:
        try:
            from .upper_body_ik_backend import backend_report
            report["upper_body_ik_backend"] = backend_report()
        except Exception as exc:  # A report refresh must never replace the task outcome.
            report["upper_body_ik_backend_refresh_error"] = f"{type(exc).__name__}: {exc}"
    return report


def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _tracking_progress_goal(machine, observation, arm):
    """Final TCP goal used by the FSM, including measured-grasp compensation.

    A looser object-position tolerance must not keep refreshing the watchdog
    while the FSM still waits for TCP arrival. Use the final waypoint rather
    than the moving servo reference; this is pose progress, not task success.
    """
    goal = machine._tcp_target
    if goal is None:
        return None
    if getattr(machine, "_object_feedback_hold_reason", None) is not None:
        # Hand-path transit: the FSM judges the hand against its planned target; so does the watchdog.
        return matrix_to_pose(goal)
    if (machine.task.get("object_pose_control", {}).get("object_feedback") is not None
            and machine.stage_id in machine.task.get(
                "object_feedback_stages", ["lift", "recover_carry", "transfer", "place"])):
        measured_object = pose_to_matrix(observation["object_poses_w"][machine.object_path])
        measured_tcp = pose_to_matrix(observation["tcp_poses_w"][arm])
        desired_object = goal @ np.linalg.inv(machine._control_object_to_tcp)
        axis = getattr(machine, "_symmetry_axis", None)
        if axis is not None:
            # Declared object symmetry: the goal never rolls the hand to undo a spin of the object.
            measured_object = spin_aligned_object(measured_object, desired_object[:3, :3], axis)
        goal = desired_object @ np.linalg.inv(measured_object) @ measured_tcp
    return matrix_to_pose(goal)


def _reset_prepared_policy_history(raw, terms, source, *, return_observations=False):
    """Seed history after boundary writes; optionally return the exact cached input.

    The default metadata-only return preserves existing callers. Runtime callers
    consume the returned observations directly, without evaluating stateful
    observation terms again at the same simulation time.
    """
    before = float(raw.sim.current_time)
    for term in terms:
        term.set_source(source)
    raw.observation_manager.reset()
    observations = raw.observation_manager.compute(update_history=True)
    values = _array(observations["policy"])
    if values.ndim != 2 or values.shape[1] != 700 or not np.isfinite(values).all():
        raise RuntimeError("Prepared boundary did not produce finite 700-dimensional HugWBC observations")
    if float(raw.sim.current_time) != before:
        raise RuntimeError("Preparing policy observation history must not advance physics")
    evidence = {"method": "actual_boundary_state_replicated_no_physics_elapsed",
        "simulation_time_s": before, "observation_dim": int(values.shape[1]),
        "physics_elapsed_s": 0.}
    return (observations, evidence) if return_observations else evidence


def _configured_initial_arm_targets(task, model, limits, positions):
    configured = task.get("initial_arm_joint_positions")
    if configured is None:
        return None
    if set(configured) != set(model.arm_joint_names(task["arm"])):
        raise ValueError("initial_arm_joint_positions must name exactly the active seven joints")
    targets = {n: float(q) for n, q in configured.items()}
    if any(not math.isfinite(q) or not limits[n][0] <= q <= limits[n][1] for n, q in targets.items()):
        raise ValueError("initial_arm_joint_positions exceeds shared control limits")
    model.validate_snapshot({**positions, **targets})
    return targets


class PlannerBridge:
    def __init__(self, args, stage, robot, live, urdf_path):
        self.args, self.stage, self.robot, self.live = args, stage, robot, live
        self.urdf_path = urdf_path
        self.sequence = 0

    def plan(self, arm, pose, directory):
        from .live_scene import capture_live_snapshot
        self.sequence += 1
        directory = Path(directory) / f"plan_{self.sequence:03d}"
        directory.mkdir(parents=True)
        snapshot_path = directory / "snapshot.json"
        capture_live_snapshot(self.stage, self.robot, self.live, snapshot_path,
                              scene_usd=str(self.args.scene_usd.resolve()))
        output = directory / "approach.npz"
        command = [str(self.args.planner_python), "-m", "robros_motion_planning.cli",
                   "--snapshot", str(snapshot_path), "--arm", arm,
                   "--urdf", str(self.urdf_path), "--arm-path-only", "--cpu-goal-seed",
                   "--collision-fit-intermediates", "3",
                   # The supplied wrist CAD has a ~3.86 mm internal gap.
                   # Keep a positive 1 mm self margin per surface and 2 mm
                   # world inflation; the former hand's 2+2 mm self margin
                   # cannot represent this neutral mechanism configuration.
                   "--sphere-buffer", "0.001", "--world-extra-buffer", "0.001",
                   "--goal-pose-world", *[format(float(v), ".12f") for v in pose], "--output", str(output)]
        print(f"[NEPES] cuRobo planning from live state: {directory}", flush=True)
        with (directory / "planner.log").open("w") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        if process.returncode:
            raise RuntimeError(f"cuRobo rejected approach (exit {process.returncode}); see {directory / 'planner.log'}")
        with np.load(output, allow_pickle=False) as data:
            if str(data["schema"]) != "igris_c_named_arm_path_v1":
                raise ValueError("Unexpected live arm trajectory schema")
            names, positions = data["joint_names"].tolist(), data["joint_pos"].copy()
        manifest = json.loads(output.with_suffix(".manifest.json").read_text())
        return names, positions, manifest


def run_tasks(args, config, selected, app):
    import gymnasium as gym
    import torch
    import omni.usd
    from isaaclab_tasks.utils import parse_env_cfg
    import robros_lab.tasks  # noqa: F401
    from robros_lab.tasks.base.usd_scene import configure_usd_scene, apply_usd_scene_render_settings
    from robros_lab.tasks.hug_wbc.controllers import HugWBCPolicy
    from .latest_robot import (DEFAULT_GRIPPER_VARIANT, GRIPPER_JOINT_NAMES, build_latest_robot_variant,
                               configure_latest_robot, resolve_gripper_selection)
    from .robot_model import load_robot_model
    from .nepes_tasks import NepesTaskMachine, POSTURE_STAGES, STAGES, TASK_IDS
    from .grasp_carry import GraspCarryMachine
    from .live_scene import (LiveScene, discover_nepes_objects, read_usd_object_poses,
                             validate_render_poses, two_finger_pinch_evidence, enable_support_contact_reports)
    from .throughput import ThroughputProfile, CaptureRenderControl
    from .actuator_delay import (configure_actuator_delay_profile, read_actuator_delay_state,
                                 require_actuator_delay_match, validate_actuator_delay_profile,
                                 prepare_actuator_delay_reset)
    from .single_arm_layout import (CART_PATH, resolve_layout_scene, discover_layout_objects,
                                    measured_layout_provenance)

    # Recheck immutable layout selection for direct callers and immediately
    # before scene creation; no scene/object transform is written at runtime.
    args.scene_usd, layout_contract = resolve_layout_scene(config, args.scene_usd)

    collection_scope = getattr(args, "collection_scope", "task")
    if layout_contract is not None and (collection_scope != "stationary_task" or args.sequence
                                       or selected != ["crs_load_jig"]):
        raise ValueError("single_arm_layout requires independent stationary crs_load_jig episodes")
    from .camera_capture import validate_camera_config
    camera_layout = validate_camera_config(config)
    delay_profile = (validate_actuator_delay_profile(config["actuator_delay_profile"])
                     if "actuator_delay_profile" in config else None)
    primitive = collection_scope == "grasp_carry"
    stationary = collection_scope == "stationary_task"
    stationary_spec = None
    recording_boundary_settings = config.get("recording_boundary")
    if recording_boundary_settings is not None and not stationary:
        raise ValueError("Prepared recording boundaries require independent stationary task scope")
    if stationary:
        from .stationary_catalog import validate_stationary_spec
        from .stationary_lift import StationaryLiftMachine
        from .stationary_place import StationaryPlaceMachine
        stationary_spec = validate_stationary_spec(config)
        if stationary_spec["goal"] == "pickup_load_jig":
            from .stationary_pick_load import StationaryPickLoadMachine
    # Held start (v1 task 2) requires the prepared grasp. Both JIG-loading goals
    # share the placement tail: CRS->JIG support reporting, the dynamic JIG
    # support path, CRS keep-awake and the release/retreat TCP reference.
    held_start = stationary and stationary_spec["goal"] == "load_jig"
    pickup_load = stationary and stationary_spec["goal"] == "pickup_load_jig"
    placement_tail = held_start or pickup_load
    upright = any("upright_start" in config["tasks"][task_id] for task_id in selected)
    if stationary:
        if args.sequence or selected != [stationary_spec["legacy_task_id"]]:
            raise ValueError("Stationary task scope requires its own independent task configuration")
    if upright != pickup_load:
        raise ValueError("The upright start is exactly the independent catalog-v2 pickup/load initial condition")
    if upright and args.initialize_arm_at_pregrasp:
        raise ValueError("The upright start records the whole reach; it forbids --initialize_arm_at_pregrasp")
    if upright and args.no_head_gaze and config["tasks"][selected[0]]["upright_start"]["head_gaze"]:
        raise ValueError("The upright start declares its head gaze; --no_head_gaze would change that initial condition")
    diagnostic_stop_after = getattr(args, "diagnostic_stop_after", None)
    if diagnostic_stop_after is not None and not stationary:
        raise ValueError("--diagnostic_stop_after is a stationary calibration stop")
    machine_type = (StationaryPickLoadMachine if pickup_load else StationaryPlaceMachine if held_start
                    else StationaryLiftMachine if stationary
                    else GraspCarryMachine if primitive else NepesTaskMachine)
    diagnostic_stop_after_waypoint = getattr(args, "diagnostic_stop_after_waypoint", None)
    diagnostic_waypoint_stop = None
    if diagnostic_stop_after_waypoint is not None:
        if diagnostic_stop_after is not None:
            raise ValueError("--diagnostic_stop_after_waypoint and --diagnostic_stop_after are mutually exclusive")
        if not stationary or args.mode != "execute" or args.sequence:
            raise ValueError("--diagnostic_stop_after_waypoint requires independent stationary_task scope in execute mode")
        diagnostic_stages = machine_type(
            config, selected[0], goal_settings=stationary_spec).stages
        diagnostic_waypoint_stop = validate_diagnostic_waypoint_stop(
            diagnostic_stop_after_waypoint, config["tasks"][selected[0]], diagnostic_stages)
    for selected_task in selected:
        validate_required_diagnostic_waypoint_stop(config["tasks"][selected_task],
            diagnostic_stop_after_waypoint, execute=args.mode == "execute")
    def dataset_task_id(task_id):
        return stationary_spec["id"] if stationary else task_id
    def dataset_task_number(task_id):
        return stationary_spec["number"] if stationary else TASK_IDS.index(task_id) + 1
    if any("auxiliary_support_arm" in config["tasks"][task_id] for task_id in selected):
        if primitive or args.approach != "servo":
            raise ValueError("Auxiliary support is a servo-only unsuccessful lift diagnostic, not a grasp-carry episode")
    if args.approach == "curobo" and any(
            "pregrasp" in config["tasks"][task_id].get("stage_postures", {}) for task_id in selected):
        raise ValueError("A cuRobo frozen-joint approach cannot run with a changing pregrasp posture")
    if args.approach == "curobo" and any("pregrasp_pose_control" in config["tasks"][task_id] for task_id in selected):
        raise ValueError("Bounded primary pregrasp requires --approach servo")
    if args.approach != "servo" and any("upper_body_ik" in config["tasks"][task_id] for task_id in selected):
        raise ValueError("Measured-pelvis upper_body_ik requires --approach servo")
    def episode_title(task_id):
        title = config["tasks"][task_id].get("title", task_id)
        return f"Physical grasp, lift and short transport only (parent: {title})" if primitive else title

    args.output.mkdir(parents=True)
    _json(args.output / "config.json", config)
    task_name = "Robros-IGRIS-C-Flat-HugWBC-LowerBody-Symmetry"
    env_cfg = parse_env_cfg(task_name, device=args.device, num_envs=1, use_fabric=False)
    env_cfg.seed = args.seed
    if "enable_scene_ccd" in config:
        env_cfg.sim.physx.enable_ccd = config["enable_scene_ccd"]
    # run_tasks.py resolves (and rejects) the selection before Isaac starts; direct callers fall back here.
    gripper = getattr(args, "gripper_selection", None) or resolve_gripper_selection(config)
    variant = build_latest_robot_variant(args.output / "robot_model", model_dir=args.robot_model_dir,
                                         gripper_variant=gripper["variant"],
                                         pad_material_override=gripper["dev_pad_material_override"],
                                         pad_contact_patch=gripper.get("pad_contact_patch"),
                                         pad_compliant_contact=gripper.get("pad_compliant_contact"),
                                         pad_contact_spheres=gripper.get("pad_contact_spheres"))
    if variant.metadata.get("gripper_variant", DEFAULT_GRIPPER_VARIANT) != gripper["variant"]:
        raise RuntimeError("The derived robot model does not carry the selected gripper variant")
    configure_latest_robot(env_cfg, variant)
    actuator_delay_configuration = configure_actuator_delay_profile(env_cfg, delay_profile)
    placement = configure_usd_scene(env_cfg, args.scene_usd, "/World/igris_c")
    if config.get("lid_fit_clearance_profile"):
        from isaaclab.managers import EventTermCfg
        from .lid_fit_clearance import prepare_lid_fit_clearance
        # One environment: per-prim mesh edits must precede PhysX parsing,
        # which IsaacLab disallows with replicated physics enabled.
        env_cfg.scene.replicate_physics = False
        env_cfg.events.nepes_lid_fit_clearance = EventTermCfg(
            func=prepare_lid_fit_clearance, mode="prestartup")
    if config.get("floor_support_profile"):
        from isaaclab.managers import EventTermCfg
        from .floor_support import prepare_floor_support
        env_cfg.scene.replicate_physics = False
        env_cfg.events.nepes_floor_support = EventTermCfg(
            func=prepare_floor_support, mode="prestartup")
    reporters = sorted({config["objects"][task["object"]] for task in config["tasks"].values()
                        if "placement_support" in task})
    if reporters:
        from isaaclab.managers import EventTermCfg
        from .live_scene import prepare_support_contact_reporting
        env_cfg.scene.replicate_physics = False
        env_cfg.events.nepes_support_contact_reporting = EventTermCfg(
            func=prepare_support_contact_reporting, mode="prestartup",
            params={"object_paths": reporters,
                "keep_awake_object_paths": [config["objects"]["crs"]]
                    if placement_tail and config.get("support_contact_awake_object") == "crs" else []})
    record_foot_contacts = bool(stationary and config.get("floor_support_profile"))
    if record_foot_contacts:
        from isaaclab.managers import EventTermCfg
        from .foot_contact_observation import prepare_foot_contact_reporting
        env_cfg.scene.replicate_physics = False
        env_cfg.events.nepes_foot_contact_reporting = EventTermCfg(
            func=prepare_foot_contact_reporting, mode="prestartup")
    env = gym.make(task_name, cfg=env_cfg)
    raw = env.unwrapped
    # Headless IsaacLab kits disable USD write-back independently of Fabric.
    # Rendering and camera attachment require live transforms for every actor.
    raw.sim.get_physics_context().set_physx_update_transformations_settings(
        update_to_usd=True, update_velocities_to_usd=True, output_velocities_local_space=False)
    live = camera = writer = foot_contacts = None
    original_scene_update = None
    original_scene_update_had_override = False
    throughput = ThroughputProfile(getattr(args, "profile_throughput", False))
    render_control = CaptureRenderControl(raw,
        mode=getattr(args, "capture_render_mode", "legacy"),
        headless=getattr(args, "headless", False), recording=args.record, profile=throughput)
    report = {"schema": "nepes_task_run_v1", "mode": args.mode, "seed": args.seed,
              "episode_scope": collection_scope,
              "execution_mode": "sequence" if args.sequence else "independent",
              "scene_usd": str(args.scene_usd.resolve()), "tasks": selected,
              "robot_model_manifest": str(variant.manifest_path),
              "robot_gripper": {**gripper, "robot_model_fingerprint": variant.metadata["fingerprint"],
                  "runtime_implicit_actuator": variant.metadata["gripper_actuator_profile"]["runtime_implicit_actuator"],
                  "pad_physics_material": (None if "gripper_pads" not in variant.metadata else {
                      key: variant.metadata["gripper_pads"]["physics_material"][key] for key in
                      ("static_friction", "dynamic_friction", "restitution", "friction_combine_mode")})},
              "planner_clearance_profile_m": {"self_per_surface": .001, "world_per_surface": .002,
                  "basis": "260910_wrist_CAD_gap_about_3.857mm; conservative_full_geometry_retained"},
              "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
              "calibration_status": config.get("calibration_status", "draft_unvalidated"),
              "approach_backend": args.approach, "contact_controller": "bounded_damped_least_squares",
              "collision_free_execution_verified": False, "episodes": [], "complete": False}
    if layout_contract is not None:
        report["single_arm_layout"] = layout_contract
    report["capture_options"] = {
        "capture_render_mode": render_control.mode,
        "image_writer_workers": getattr(args, "image_writer_workers", 0),
        "image_writer_queue": getattr(args, "image_writer_queue", 8),
        "profile_throughput": throughput.enabled,
    }
    report["actuator_delay"] = {"configuration": actuator_delay_configuration,
                                "reset_preparations": [], "reset_readbacks": []}

    def prepare_actuator_delay_boundary(boundary):
        evidence = prepare_actuator_delay_reset(
            raw.scene["robot"], actuator_delay_configuration, boundary=boundary)
        report["actuator_delay"]["reset_preparations"].append(evidence)

    def record_actuator_delay_reset(boundary):
        # env.reset() already chose the lags. Reading them must not reset or
        # re-sample any buffer, policy history, command, or physical state.
        evidence = read_actuator_delay_state(
            raw.scene["robot"], actuator_delay_configuration, physics_dt_s=float(raw.physics_dt),
            boundary=boundary, num_envs=int(getattr(raw, "num_envs", 1)))
        report["actuator_delay"]["reset_readbacks"].append(evidence)
        return evidence
    if stationary:
        from .stationary_catalog import config_digest, recorded_inter_station_transport
        # Only the v2 pickup/load records a walk between two declared stations
        # (its side-step); every v1 goal keeps False.
        report.update(stationary_task=stationary_spec, tasks=[stationary_spec["id"]],
                      config_sha256=config_digest(config),
                      legacy_task_ids=selected, legacy_full_task_completed=False,
                      controller_contract={"hugwbc_enabled": True, "pelvis_fixed": False,
                          "robot_gravity_enabled": True, "grasp_attachment": "none",
                          "initialization": "independent_episode_boundary",
                          "initial_station_alignment": True,
                          "inter_station_transport": recorded_inter_station_transport(config)})
        no_walk_settings = config["tasks"][selected[0]].get("navigation_control", {}).get("no_walk")
        if no_walk_settings is not None:
            # Opt-in single stance (no_walk_stance): exactly zero base command in every stage.
            report["controller_contract"].update(initial_station_alignment=False, no_walk={
                "base_command": "exactly_zero_in_every_stage", "navigate_stages": "complete_without_stepping",
                "planned_pelvis_xy_yaw": [float(v) for v in no_walk_settings["planned_pelvis_xy_yaw"]]})
        if config["tasks"][selected[0]].get("handover") is not None:
            # Opt-in in-place handover (handover.py): the arm switch is part of the admitted contract.
            report["controller_contract"]["handover"] = handover_contract(config["tasks"][selected[0]])
        if config["tasks"][selected[0]].get("navigation_control", {}).get("single_step") is not None:
            # Opt-in single step (single_step.py): its base command window, one-shot brake and hold re-anchor.
            from .single_step import single_step_contract
            report["controller_contract"]["single_step"] = single_step_contract(config["tasks"][selected[0]])
            if report["controller_contract"]["single_step"].get("source_hold") is True:
                report["controller_contract"]["initial_station_alignment"] = False
        relief_contract = supported_grip_relief_contract(config["tasks"][selected[0]])
        if relief_contract is not None:
            report["controller_contract"]["supported_grip_relief"] = relief_contract
        selected_task = config["tasks"][selected[0]]
        if "press_in_mode" in selected_task.get("placement_alignment", {}):
            from .placement_alignment import placement_overtravel_contract
            report["controller_contract"]["placement_command_overtravel"] = placement_overtravel_contract(selected_task)
        if "transfer_completed_reference_continuity" in selected_task:
            from .completed_transfer_continuity import contract as completed_transfer_contract
            report["controller_contract"]["completed_transfer_reference_continuity"] = completed_transfer_contract(selected_task)
        if "transfer_cartesian_gain_s_inv" in config["tasks"][selected[0]].get("upper_body_ik", {}):
            from .upper_body_control import transfer_cartesian_gain_contract
            report["controller_contract"]["transfer_cartesian_gain"] = transfer_cartesian_gain_contract(config["tasks"][selected[0]])
        if "place_entry_height_m" in selected_task:
            from .place_entry_height import place_entry_height_contract
            report["controller_contract"]["place_entry_height"] = place_entry_height_contract(selected_task)
        if "transfer_arrival_precision" in selected_task:
            from .transfer_arrival_precision import transfer_arrival_precision_contract
            report["controller_contract"]["transfer_arrival_precision"] = transfer_arrival_precision_contract(
                selected_task)
        if (any(isinstance(selected_task.get(key), dict)
                for key in ("lift_reference_continuity", "transfer_reference_continuity"))
                or "object_waypoint_intermediate_dwell_s" in config.get("thresholds", {})):
            # Opt-in motion continuity (reference_continuity.py, runtime_v26): admission requires it verbatim.
            from .reference_continuity import continuity_contract
            report["controller_contract"]["motion_continuity"] = continuity_contract(config, selected_task)
    if diagnostic_stop_after is not None:
        # A calibration run: every episode fails on leaving this stage.
        report["diagnostic_stop_after"] = diagnostic_stop_after
    if diagnostic_stop_after_waypoint is not None:
        report["diagnostic_stop_after_waypoint"] = diagnostic_stop_after_waypoint
    if config.get("lid_fit_clearance_profile"):
        # Fail if the prestartup event was omitted; never silently run another
        # physical model than the one declared by the calibration config.
        report["lid_fit_clearance"] = raw.nepes_lid_fit_clearance_provenance
    if config.get("floor_support_profile"):
        report["floor_support"] = raw.nepes_floor_support_provenance
    if reporters:
        report["support_contact_reporting"] = raw.nepes_support_contact_reporting_provenance
        if config.get("support_contact_diagnostics", False):
            report["support_contact_diagnostics"] = raw.nepes_support_contact_reporting_provenance
    if record_foot_contacts:
        report["foot_contact_reporting"] = getattr(raw, "nepes_foot_contact_reporting_provenance", {
            "available": False, "foot_paths": {}, "reason": "Foot reporter prestartup event did not run"})
    try:
        apply_usd_scene_render_settings(placement.usd_path)
        stage = omni.usd.get_context().get_stage()
        physics_scene = stage.GetPrimAtPath(env_cfg.sim.physics_prim_path)
        actual_ccd = physics_scene.GetAttribute("physxScene:enableCCD").Get()
        report["physics_scene_ccd"] = {"requested": bool(env_cfg.sim.physx.enable_ccd),
            "authored_enabled": actual_ccd, "physics_prim_path": env_cfg.sim.physics_prim_path,
            "physics_dt_s": float(raw.physics_dt), "control_dt_s": float(raw.step_dt)}
        if config.get("enable_scene_ccd") and actual_ccd is not True:
            raise RuntimeError("Requested scene CCD was not enabled")
        if config.get("hinged_door"):
            from .hinged_door import author_hinged_door
            report["hinged_door"] = author_hinged_door(stage, config)
        from .jig_slots import author_jig_slot_collisions
        report["jig_slot_collisions"] = author_jig_slot_collisions(stage, config)
        if config.get("contact_distance_profile"):
            from .contact_distance import author_contact_distance
            report["contact_distance_calibration"] = author_contact_distance(stage, config["contact_distance_profile"])
        if not args.no_cover_parking:
            # Task-only session geometry; authored Scene.usda is never changed.
            from pxr import Gf, UsdGeom, UsdPhysics
            report["added_simulation_fixtures"] = []
            for fixture in config.get("fixtures", []):
                if fixture["type"] != "static_collision_box":
                    raise ValueError(f"Unsupported task fixture: {fixture['type']}")
                path, pose, size = fixture["prim_path"], fixture["pose_w"], fixture["size_m"]
                if stage.GetPrimAtPath(path).IsValid():
                    raise ValueError(f"Task fixture already exists in the authored scene: {path}")
                cube = UsdGeom.Cube.Define(stage, path)
                cube.CreateSizeAttr(1.0)
                cube.AddTranslateOp().Set(Gf.Vec3d(*pose[:3]))
                cube.AddOrientOp().Set(Gf.Quatf(pose[3], Gf.Vec3f(*pose[4:])))
                cube.AddScaleOp().Set(Gf.Vec3d(*size))
                cube.CreateDisplayColorAttr([Gf.Vec3f(.25, .35, .4)])
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                report["added_simulation_fixtures"].append(fixture)
        support_tasks = ([config["tasks"][selected[0]]] if placement_tail else config["tasks"].values())
        support_pairs = {config["objects"][task["object"]]: config["objects"][task["placement_support"]["object"]]
                         for task in support_tasks if "placement_support" in task}
        dynamic_support_paths = [config["objects"]["jig"]] if placement_tail else []
        report["dynamic_support_paths"] = dynamic_support_paths
        enable_support_contact_reports(stage, support_pairs, dynamic_support_paths=dynamic_support_paths)
        prepare_actuator_delay_boundary("startup_env_reset")
        obs, _ = env.reset()
        latest_actuator_delay_readback = record_actuator_delay_reset("startup_env_reset")
        require_actuator_delay_match(latest_actuator_delay_readback)
        robot = raw.scene["robot"]
        if stationary:
            report["controller_contract"]["pelvis_fixed"] = bool(robot.is_fixed_base)
            if robot.is_fixed_base:
                raise RuntimeError("Stationary datasets require the live floating-base HugWBC robot")
        initial = dict(zip(robot.joint_names, _array(robot.data.joint_pos)[0]))
        command_names, auxiliary_names = variant.command_joint_names, variant.auxiliary_joint_names
        auxiliary_names = resolve_auxiliary_commands(
            command_names, auxiliary_names,
            raw.action_manager.get_term("joint_pos")._joint_names,
            raw.action_manager.get_term("teleop")._joint_names)
        report["auxiliary_actuator_command_names"] = list(auxiliary_names)
        source = LiveCommandSource(initial, device=raw.device, command_joint_names=command_names,
                                   grippers=variant.grippers)
        terms = [raw.command_manager.get_term(n) for n in ("base_velocity", "teleoperation_command")]
        for term in terms:
            term.set_source(source)
        auxiliary_ids = [robot.joint_names.index(n) for n in auxiliary_names]
        # The upright start writes its declared root pose (height included) below.
        if not args.sequence and not held_start and not upright:
            initial_task = config["tasks"][selected[0]]
            station = initial_task.get("initial_robot_pose_w", initial_task.get("base_stations", {}).get("source"))
            if station is not None:
                spawn = list(station)
                spawn[2] = float(robot.data.default_root_state[0, 2])
                robot.write_root_pose_to_sim(torch.tensor([spawn], dtype=torch.float32, device=raw.device))
                robot.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=raw.device))
                report["initial_robot_station_pose_w"] = spawn
        policy = HugWBCPolicy(args.checkpoint, device=raw.device)
        report["hugwbc_policy"] = {
            "checkpoint_path": str(policy.checkpoint_path),
            "checkpoint_sha256": hashlib.sha256(policy.checkpoint_path.read_bytes()).hexdigest(),
            "observation_dim": policy.observation_dim,
            "action_dim": policy.action_dim,
            "control_hz": 1. / float(raw.step_dt),
            "observation_timing": "one_history_update_per_env_step;cached_policy_input",
            "boundary_history": "reseed_actual_state_after_prepared_or_upright_writes_before_physics",
        }
        if policy.action_dim != 13 or policy.observation_dim != obs["policy"].shape[1] or abs(raw.step_dt - .02) > 1e-8:
            raise RuntimeError("NEPES runner requires the 700-observation / 13-action / 50 Hz HugWBC contract")
        dynamic_objects = (discover_layout_objects(stage, layout_contract) if layout_contract is not None
                           else discover_nepes_objects(stage))
        live = LiveScene(robot._physics_sim_view, dynamic_objects,
                         physics_dt=float(raw.physics_dt), stage=stage, support_pairs=support_pairs,
                         support_contact_diagnostics=config.get("support_contact_diagnostics", False),
                         dynamic_support_paths=dynamic_support_paths)
        if record_foot_contacts:
            from .foot_contact_observation import FootContactObservation
            foot_contacts = FootContactObservation(
                robot._physics_sim_view, physics_dt=float(raw.physics_dt), stage=stage,
                foot_paths=report["foot_contact_reporting"].get("foot_paths", {}))
        original_object_states = live.capture_initial_state()
        static_paths = [p for p in config["objects"].values() if p not in live.object_paths.values()
                        and stage.GetPrimAtPath(p).IsValid()]
        if layout_contract is not None and CART_PATH not in static_paths:
            static_paths.append(CART_PATH)
        static_poses = read_usd_object_poses(stage, static_paths)
        servo = ArmServo(load_robot_model(variant.urdf_path))
        from .stationary_initialization import validate_prepared_grasp_initialization, prepared_grasp_provenance
        prepared_grasp = validate_prepared_grasp_initialization(config, model=servo.model,
            command_names=command_names, control_limits=servo.limits)
        if held_start and prepared_grasp is None:
            raise ValueError("Held-start task2 requires an explicit prepared grasp initial condition")

        def apply_prepared_grasp():
            # Only called before startup or independent episode physics settle.
            nonlocal obs
            if set(prepared_grasp["joint_positions"]) != set(robot.joint_names):
                raise ValueError("Prepared state must name every actual robot joint")
            robot.write_root_pose_to_sim(torch.tensor([prepared_grasp["robot_root_pose_w"]],
                dtype=torch.float32, device=raw.device))
            robot.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=raw.device))
            robot.write_joint_state_to_sim(torch.tensor([[prepared_grasp["joint_positions"][name]
                for name in robot.joint_names]], dtype=torch.float32, device=raw.device),
                torch.zeros((1, len(robot.joint_names)), dtype=torch.float32, device=raw.device))
            source.targets.update(prepared_grasp["command_joint_positions"])
            source.base_velocity[:] = 0
            live.set_initialization_poses({config["objects"]["crs"]: prepared_grasp["object_pose_w"]},
                episode_boundary=True)
            # IsaacLab pose/joint write APIs update their measured buffers and
            # invalidate dependent body caches. Seed the history AFTER all
            # writes and command copies, before the first WBC inference.
            obs, report["prepared_policy_history"] = _reset_prepared_policy_history(
                raw, terms, source, return_observations=True)
            return dict(prepared_grasp["joint_positions"])

        def apply_upright_start(task_id, object_poses_w, scope):
            """Write the declared upright stance before a physics settle; never during an episode.

            Only the root pose (declared height kept), the declared jaw/neck
            overrides and the optional gaze toward the source grasp point are
            written. Every other joint keeps its env reset default: no upper
            posture, arm pose, held object or attachment is prepared here.
            """
            nonlocal obs
            task_settings = config["tasks"][task_id]
            start = task_settings["upright_start"]
            spawn = upright_spawn_pose(start)
            robot.write_root_pose_to_sim(torch.tensor([spawn], dtype=torch.float32, device=raw.device))
            robot.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=raw.device))
            positions = dict(zip(robot.joint_names, _array(robot.data.joint_pos)[0]))
            source.targets.update({n: float(positions[n]) for n in command_names})
            overrides = upright_overrides(start, positions, command_names=command_names,
                                          joint_limits=servo.limits, model=servo.model)
            gaze = None
            if start["head_gaze"]:
                from .head_gaze import recommend_head_joint_targets
                object_pose_w = object_poses_w[config["objects"][task_settings["object"]]]
                gaze = recommend_head_joint_targets(servo.model, {**positions, **overrides}, spawn,
                                                   gaze_target_w(task_settings, object_pose_w),
                                                   variant.camera_attachments)
            writes = upright_joint_writes(overrides, gaze, joint_limits=servo.limits)
            if writes:
                ids = [robot.joint_names.index(n) for n in writes]
                robot.write_joint_state_to_sim(
                    torch.tensor([list(writes.values())], dtype=torch.float32, device=raw.device),
                    torch.zeros((1, len(ids)), dtype=torch.float32, device=raw.device), joint_ids=ids)
                source.targets.update(writes)
            source.base_velocity[:] = 0
            # env.reset() seeded history before these root/jaw/neck writes.
            # Replace it with the actual upright state before the first WBC step.
            obs, report["upright_policy_history"] = _reset_prepared_policy_history(
                raw, terms, source, return_observations=True)
            # The resulting posture-joint command; the unrecorded preparation must keep exactly it.
            upright_targets = {n: float(source.targets[n])
                               for n in task_settings["posture_paths"]["reach_posture"]["joint_names"]}
            return {**positions, **writes}, upright_start_provenance(
                root_pose_w=spawn, overrides=overrides, gaze=gaze, scope=scope, joint_targets=upright_targets)

        planner = PlannerBridge(args, stage, robot, live, variant.urdf_path)
        dt = float(raw.step_dt)
        body_ids = {name: robot.body_names.index(name) for name in ("base_link", "Link_Waist_Pitch", "Left_Hand", "Right_Hand")}
        from .physics_readback import read_joint_drives, read_rigid_body_properties
        property_bodies = {"Left_Hand", "Right_Hand", "l_1_dof_gripper_moving", "r_1_dof_gripper_moving"}
        property_paths = [p for p in robot.root_physx_view.link_paths[0]
                          if p.rsplit("/", 1)[-1] in property_bodies] + list(live.object_paths.values())
        _json(args.output / "robot_physics.json", {
            "rigid_body_properties": read_rigid_body_properties(robot._physics_sim_view, property_paths),
            "body_masses_kg": dict(zip(robot.body_names,
                _array(robot.root_physx_view.get_masses())[0].astype(float).tolist())),
            "joint_names": list(robot.joint_names), "body_names": list(robot.body_names),
            "policy_observation_dim": policy.observation_dim, "policy_action_dim": policy.action_dim,
            "physics_self_collision_enabled": bool(env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions),
            # Applied jaw drive, read back (run.json robot_gripper carries the requested manifest actuator).
            "gripper_joint_drives": read_joint_drives(robot, GRIPPER_JOINT_NAMES),
            "qualification": "measured_loaded_model; dynamics_transfer_requires_validation"})
        tick = 0
        last_action = np.zeros(13, dtype=np.float32)
        last_action_time = float(raw.sim.current_time)
        current_command = None
        current_task_id = selected[0]
        posture_reset_records = {}

        def prepare_upper_posture(task_id, positions):
            """Apply a declared inactive-arm/torso posture only at reset."""
            task = config["tasks"][task_id]
            targets = {n: float(q) for n, q in task.get("initial_upper_joint_positions", {}).items()}
            if "open_gripper_position_rad" in task:
                grip = variant.grippers[task["arm"]]
                targets[grip["joint_name"]] = gripper_open_position(grip, task["open_gripper_position_rad"])
            if not targets:
                return positions
            if not set(targets) <= set(command_names) or set(targets) & set(servo.model.arm_joint_names(task["arm"])):
                raise ValueError("initial_upper_joint_positions must name only inactive-arm, torso, neck or gripper commands")
            servo.model.validate_snapshot({**positions, **targets})
            if any(not servo.limits[n][0] <= q <= servo.limits[n][1] for n, q in targets.items()):
                raise ValueError("initial_upper_joint_positions exceeds shared control limits")
            ids = [robot.joint_names.index(n) for n in targets]
            if any(n.startswith("Joint_Waist_") for n in targets):
                base_pose = [*_array(robot.data.root_pos_w)[0].tolist(),
                             *_array(robot.data.root_quat_w)[0].tolist()]
                adjusted = pelvis_preserving_boundary_pose(
                    servo.model, base_pose, positions, {**positions, **targets})
                robot.write_root_pose_to_sim(torch.tensor([adjusted], dtype=torch.float32, device=raw.device))
                posture_reset_records[task_id] = {
                    "method": "episode_boundary_pelvis_preserving_root_transform",
                    "root_before_pose_w": base_pose, "root_after_pose_w": adjusted,
                    "joint_positions_before": {n: float(q) for n, q in positions.items()},
                    "joint_targets": targets, "demonstration_pose_write": False}
            robot.write_joint_state_to_sim(
                torch.tensor([list(targets.values())], dtype=torch.float32, device=raw.device),
                torch.zeros((1, len(ids)), dtype=torch.float32, device=raw.device), joint_ids=ids)
            source.targets.update(targets)
            return {**positions, **targets}
        pinch_settling = {arm: PinchSettlingTracker() for arm in ("left", "right")}
        settling_object_paths = {arm: None for arm in ("left", "right")}

        @throughput.track("measured_state_and_contacts")
        def measured():
            states = live.read_states()
            poses = _array(robot.data.body_pos_w)[0]
            quats = _array(robot.data.body_quat_w)[0]
            joints = dict(zip(robot.joint_names, _array(robot.data.joint_pos)[0]))
            joint_velocities = dict(zip(robot.joint_names, _array(robot.data.joint_vel)[0]))
            tcp = {arm: [*poses[body_ids[arm.title() + "_Hand"]].tolist(),
                          *quats[body_ids[arm.title() + "_Hand"]].tolist()] for arm in ("left", "right")}
            contacts, evidence, opened, closed, settled = {}, {}, {}, {}, {}
            measurement_time_s = float(raw.sim.current_time)
            aperture_task = config["tasks"][current_task_id]
            for arm in ("left", "right"):
                grip = variant.grippers[arm]
                angle = joints[grip["joint_name"]]
                jaw_velocity = float(joint_velocities[grip["joint_name"]])
                raw_velocity_small = bool(math.isfinite(jaw_velocity) and abs(jaw_velocity) < .02)
                settled[arm] = False
                # The task arm's declared aperture; with an opt-in handover also the receiving jaw's.
                aperture = handover_jaw_positions(aperture_task, arm)[0]
                opened[arm] = gripper_is_open(angle, grip, aperture)
                contacts[arm], evidence[arm] = [], {}
                for path in states:
                    active = current_command or {}
                    active_settings = config["tasks"].get(active.get("task_id"), {})
                    # Opt-in handover: both hands' pinch on the ring is measured (command.contact_arms).
                    grasp_arms = {active.get("arm"), *(active.get("contact_arms") or ())}
                    measured_arm = (arm in grasp_arms
                        or active_settings.get("auxiliary_support_arm", {}).get("arm") == arm)
                    detail_limit = (active_settings.get("contact_detail_limit", 0)
                                    if measured_arm and active.get("object_path") == path else 0)
                    item = live.contact_evidence(arm, path, detail_limit=detail_limit)
                    pinch_bound = (active_settings.get("maximum_pinch_angle_rad", .15)
                                   if arm in grasp_arms and active.get("object_path") == path else .15)
                    item.update(two_finger_pinch_evidence(
                        item, arm, angle, closed_position_rad=grip["closed_position_rad"],
                        maximum_closed_angle_rad=pinch_bound))
                    item.update({"joint_velocity_rad_s": jaw_velocity,
                                 "jaw_settled": False,
                                 "raw_velocity_below_diagnostic_threshold": raw_velocity_small,
                                 "raw_velocity_diagnostic_threshold_rad_s": .02})
                    evidence[arm][path] = item
                    if item.get("contact_confirmed", False):
                        contacts[arm].append(path)
                # A half-open rotating jaw and palm contact cannot confirm a
                # pinch. Scope both-jaw evidence to the active task's object;
                # retain any-hand contacts above so release remains strict.
                active = current_command or {}
                target_path = (active.get("object_path")
                               if arm in {active.get("arm"), *(active.get("contact_arms") or ())} else None)
                closed[arm] = bool(evidence[arm].get(target_path, {}).get("pinch_confirmed", False))
                if settling_object_paths[arm] != target_path:
                    pinch_settling[arm].reset()
                    settling_object_paths[arm] = target_path
                if target_path in states:
                    window = pinch_settling[arm].update(
                        measurement_time_s, angle, tcp[arm], states[target_path]["pose_w"],
                        pinch_confirmed=closed[arm])
                    window["measurement_time_s"] = measurement_time_s
                    settled[arm] = window["settled"]
                    evidence[arm][target_path].update(jaw_settled=settled[arm], jaw_settling_window=window)
            base = [*poses[body_ids["base_link"]].tolist(), *quats[body_ids["base_link"]].tolist()]
            pelvis = [*poses[body_ids["Link_Waist_Pitch"]].tolist(), *quats[body_ids["Link_Waist_Pitch"]].tolist()]
            from .robot_stability import measured_stability
            from .foot_observation import native_foot_observation
            stability = measured_stability(base, pelvis)
            feet = native_foot_observation(
                robot.body_names, poses, quats,
                _array(robot.data.body_link_lin_vel_w)[0],
                _array(robot.data.body_link_ang_vel_w)[0])
            feet["measurement_time_s"] = tick * dt
            foot_contact_evidence = None
            if foot_contacts is not None:
                foot_contact_evidence = foot_contacts.read(measurement_time_s=measurement_time_s)
                # Native physics and canonical control clocks are separately
                # recorded; their small offset is not a stale-contact signal.
                foot_contact_evidence["trace_time_s"] = tick * dt
                if "stationary_feet_hold" in config["tasks"][current_task_id].get("navigation_control", {}):
                    # The producer does not infer support or balance. This
                    # opt-in consumer now uses its measurements to reject
                    # episodes whose sampled feet leave the declared stance.
                    foot_contact_evidence["used_for_stationary_feet_guard"] = True
                    foot_contact_evidence["used_for_support_or_success"] = True
                    for foot in foot_contact_evidence.get("feet", {}).values():
                        native_report = foot.get("native_contact_report")
                        if isinstance(native_report, dict):
                            native_report["used_for_stationary_feet_guard"] = True
                            native_report["used_for_support_or_success"] = True
            faults = stability["faults"]
            return {"time_s": tick * dt, "object_poses_w": {**static_poses, **{p: s["pose_w"] for p, s in states.items()}},
                    "object_states": states, "tcp_poses_w": tcp, "base_pose_w": base,
                    "pelvis_pose_w": pelvis, "stability_evidence": stability,
                    "foot_observation": feet,
                    "foot_contact_observation": foot_contact_evidence,
                    "base_velocity_w": [*_array(robot.data.body_lin_vel_w)[0, body_ids["base_link"]].astype(float).tolist(),
                                        *_array(robot.data.body_ang_vel_w)[0, body_ids["base_link"]].astype(float).tolist()],
                    "hand_open": opened, "hand_closed": closed, "hand_settled": settled, "contacts": contacts,
                    "contact_evidence": evidence, "joint_positions": {n: float(v) for n, v in joints.items()},
                    "joint_velocities": {n: float(v) for n, v in joint_velocities.items()},
                    "support_evidence": {path: live.support_evidence(path) for path in support_pairs},
                    "applied_actuator_effort_Nm": dict(zip(robot.joint_names, _array(robot.data.applied_torque)[0].astype(float).tolist())),
                    "events": [], "faults": faults}

        @throughput.track("control_step_including_capture")
        def step():
            nonlocal tick, obs, last_action, last_action_time
            started = time.monotonic()
            for term in terms:
                term.set_source(source)  # Copy this step's current command before inference.
            with throughput.measure("policy_observation"):
                # env.step() already evaluated and appended each observation term.
                # A second compute(False) returns the existing history but still
                # advances stateful gait terms, doubling phase during walking.
                policy_obs = obs["policy"]
            with torch.inference_mode():
                with throughput.measure("hugwbc_inference"):
                    action = policy.infer(policy_obs)
                last_action = _array(action)[0].copy()
                last_action_time = float(raw.sim.current_time)
                robot.set_joint_position_target(
                    torch.tensor([[source.targets[n] for n in auxiliary_names]],
                                 dtype=torch.float32, device=raw.device), joint_ids=auxiliary_ids)
                with throughput.measure("environment_step_including_capture"):
                    obs, _, terminated, truncated, _ = env.step(action)
            tick += 1
            if bool(torch.any(terminated | truncated).item()):
                raise RuntimeError("robot_terminated_or_reset: post-reset state is not an episode sample")
            if not app.is_running():
                raise RuntimeError("simulation_closed")
            if args.real_time:
                time.sleep(max(0., dt - (time.monotonic() - started)))

        # A recorded initial frame must contain an actually applied policy action.
        # Let newly created passive constraints settle before an independent
        # fixture applies a large hinge-angle reset. A one-step startup showed
        # an inconsistent first-episode lid reset; settle traces retain the
        # evidence needed to verify the warmed-up initialization.
        # This startup is outside every recorded demonstration.
        startup_steps = 1 if prepared_grasp is not None else max(1, math.ceil(args.settle_seconds / dt))
        if prepared_grasp is not None:
            report["prepared_settling_method"] = "one_observation_step_then_cartesian_hold_and_measured_readiness"
        report["startup_settle_seconds"] = startup_steps * dt
        if prepared_grasp is not None:
            initial = apply_prepared_grasp()
            report["startup_prepared_grasp_initialization"] = prepared_grasp_provenance(prepared_grasp)
        elif upright:
            # Fail before any write if a posture waypoint or override leaves the loaded model's limits.
            validate_upright_task_runtime(config["tasks"][selected[0]], command_names=command_names,
                                          joint_limits=servo.limits, model=servo.model, positions=initial)
            initial, report["startup_upright_start"] = apply_upright_start(
                selected[0], {path: state["pose_w"] for path, state in original_object_states.items()},
                STARTUP_UPRIGHT_START_SCOPE)
            report["startup_upright_policy_history"] = dict(report["upright_policy_history"])
        elif not args.sequence:
            initial = prepare_upper_posture(selected[0], initial)
            report["startup_upper_joint_positions"] = config["tasks"][selected[0]].get("initial_upper_joint_positions", {})
            startup_task = config["tasks"][selected[0]]
            if startup_task.get("startup_apply_configured_arm", False):
                if not args.initialize_arm_at_pregrasp:
                    raise ValueError("startup_apply_configured_arm requires --initialize_arm_at_pregrasp")
                targets = _configured_initial_arm_targets(startup_task, servo.model, servo.limits, initial)
                ids = [robot.joint_names.index(n) for n in targets]
                # Both sides of a configured balancing posture must exist
                # before the first WBC step. This is a preparation boundary,
                # outside the episode and all demonstration recording.
                robot.write_joint_state_to_sim(
                    torch.tensor([list(targets.values())], dtype=torch.float32, device=raw.device),
                    torch.zeros((1, len(ids)), dtype=torch.float32, device=raw.device), joint_ids=ids)
                source.targets.update(targets)
                initial.update(targets)
                report["startup_active_arm_initialization"] = {
                    "joint_positions": targets, "demonstration_pose_write": False,
                    "scope": "before_startup_physics_settle", "collision_checked": False}
        with (args.output / "startup_trace.jsonl").open("w") as startup_trace:
            for _ in range(startup_steps):
                sample = measured()
                startup_trace.write(json.dumps({key: sample[key] for key in
                    ("time_s", "base_pose_w", "joint_positions", "faults")}, allow_nan=False) + "\n")
                step()
        initial_measurement = measured()
        initial_measurement["robot_model_urdf"] = str(variant.urdf_path.resolve())
        initial_measurement["measurement_phase"] = "settled_probe" if args.mode == "probe" else "startup_before_task_settle"
        initial_measurement["elapsed_physics_s"] = tick * dt
        initial_measurement["body_masses_kg"] = dict(zip(robot.body_names,
            _array(robot.root_physx_view.get_masses())[0].astype(float).tolist()))
        initial_measurement["object_masses_kg"] = live.read_masses()
        if layout_contract is not None:
            initial_measurement["single_arm_layout"] = measured_layout_provenance(
                layout_contract, initial_measurement["object_poses_w"],
                phase=initial_measurement["measurement_phase"])
        baseline_poses = initial_measurement["object_poses_w"]
        if prepared_grasp is not None:
            # A startup collision must not redefine the independently prepared
            # workcell. Restore every original tracked actor before the held
            # CRS override and retain all measured startup displacement.
            original_poses = {path: state["pose_w"] for path, state in original_object_states.items()}
            report["prepared_startup_object_displacement"] = {
                path: {"original_pose_w": pose,
                    "after_startup_pose_w": initial_measurement["object_poses_w"][path],
                    "position_displacement_m": float(np.linalg.norm(
                        np.asarray(pose[:3]) - np.asarray(initial_measurement["object_poses_w"][path][:3])))}
                for path, pose in original_poses.items()}
            baseline_poses = {**baseline_poses, **original_poses}
        _json(args.output / "probe.json", initial_measurement)
        if args.mode == "probe":
            report["complete"] = True
            report["probe_policy_steps"] = tick
            report["probe_stable"] = not bool(initial_measurement["faults"])
            _json(args.output / "run.json", _with_ik_backend(report))
            print(f"[NEPES] Live-state probe saved: {args.output / 'probe.json'}", flush=True)
            return 0 if report["probe_stable"] else 2

        # The per-physics-step camera hook is installed only when recording.
        capture = {"active": False, "physics_step": 0, "origin": 0.,
                   "frame_count": 0, "first_sim_time_s": None, "last_sim_time_s": None}
        if args.record:
            from .camera_capture import (CameraCapture, CaptureScheduler, latest_camera_specs,
                                         render_current_frame, with_third_person_pose,
                                         with_wrist_camera_overrides, with_head_camera_override,
                                         with_head_camera_calibration)
            from .dataset import DatasetWriter
            camera_arm = config["tasks"][selected[0]]["arm"]
            camera_specs = latest_camera_specs(
                variant.camera_attachments,
                hand_arm="both" if camera_layout == "head_both_wrists" else camera_arm)
            camera_specs = with_wrist_camera_overrides(
                camera_specs, config.get("wrist_camera_overrides"),
                hand_arm=camera_arm if camera_layout == "head_hand" else None)
            camera_specs = with_head_camera_calibration(camera_specs, config.get("head_camera_calibration"))
            camera_specs = with_head_camera_override(camera_specs, config.get("head_camera_override"))
            camera_specs = with_third_person_pose(
                camera_specs,
                translation=config.get("third_person_translation"),
                orientation_wxyz=config.get("third_person_orientation_wxyz"),
                focal_length_usd_tenths=config.get("third_person_focal_length_usd_tenths"))
            # Each wrist has its own persistent render product in the four-view
            # layout. Handovers never swap the meanings of these image channels.
            hand_specs = ({arm: with_wrist_camera_overrides(
                            latest_camera_specs(variant.camera_attachments, hand_arm=arm),
                            config.get("wrist_camera_overrides"), hand_arm=arm)[1]
                           for arm in ("left", "right")} if camera_layout == "head_hand" else None)
            camera = CameraCapture(camera_specs, hand_specs=hand_specs)
            camera.initialize()
            report["camera_layout"] = camera_layout
            report["camera_configuration"] = [asdict(spec) for spec in camera_specs]
            render_control.activate()
            lower = list(raw.action_manager.get_term("joint_pos")._joint_names)
            writer = DatasetWriter(args.output / "staging", repo_id="local/igris-c-nepes",
                                   joint_names=list(robot.joint_names), policy_action_names=["gait_frequency", *lower],
                                   upper_target_names=command_names,
                                   camera_sizes={spec.name: (spec.width, spec.height) for spec in camera_specs},
                                   keep_failed_frames=args.keep_failed_frames,
                                   image_workers=getattr(args, "image_writer_workers", 0),
                                   max_pending_frames=getattr(args, "image_writer_queue", 8))
            scheduler = CaptureScheduler(physics_hz=round(1 / raw.physics_dt), fps=30)
            original_scene_update_had_override = "update" in vars(raw.scene)
            original_scene_update = raw.scene.update

            def record_physics_frame(pstep):
                event = scheduler.poll(pstep)
                if event is None:
                    return
                # Isaac's physics callback accumulates a float32 step size.
                # The nominal tick product drifts by 10 us after ~210 seconds
                # at 250 Hz. Keep the measured clock for RGB and state alike.
                sample_time = float(raw.sim.current_time)
                positions = _array(robot.data.body_pos_w)[0]
                quaternions = _array(robot.data.body_quat_w)[0]
                camera.update_attachment_poses({name: [*p.tolist(), *q.tolist()]
                    for name, p, q in zip(robot.body_names, positions, quaternions)})
                with throughput.measure("capture_render_and_read"):
                    bundle = render_current_frame(camera, sample_time, raw.sim.render,
                                                  lambda: raw.sim.current_time)
                measured_poses = {p: s["pose_w"] for p, s in live.read_states().items()}
                measured_poses.update({f"/World/envs/env_0/Robot/{name}": [*positions[index], *quaternions[index]]
                                       for name, index in body_ids.items()})
                validate_render_poses(stage, measured_poses)
                # Opt-in handover: every frame names the hand that holds the ring (provenance of the arm switch).
                annotations = ({"holding_arm": current_command["holding_arm"]}
                               if "holding_arm" in current_command else None)
                with throughput.measure("writer_submit"):
                    writer.add_frame(sim_time_s=sample_time,
                        joint_positions=_array(robot.data.joint_pos)[0],
                        joint_velocities=_array(robot.data.joint_vel)[0], policy_action=last_action,
                        upper_targets=source.upper_target_array(), images=bundle.images,
                        phase_id=STAGES.index(current_command["stage_id"]),
                        phase_name=current_command.get("stationary_phase", current_command["stage_id"]),
                        physics_step=pstep, command_time_s=last_action_time,
                        camera_metadata=bundle.metadata,
                        **({"annotations": annotations} if annotations is not None else {}))
                if capture["first_sim_time_s"] is None:
                    capture["first_sim_time_s"] = sample_time
                capture["last_sim_time_s"] = sample_time
                capture["frame_count"] += 1

            def update_with_capture(dt, *a, **kw):
                result = original_scene_update(dt, *a, **kw)
                if capture["active"]:
                    capture["physics_step"] += 1
                    pstep = capture["physics_step"]
                    record_physics_frame(pstep)
                return result
            raw.scene.update = update_with_capture

        for cycle in range(args.cycles):
            if (args.output / "STOP").exists():
                report["operator_stopped"] = True
                break
            if cycle and args.sequence:
                live.end_episode()
                live.reset_objects(episode_boundary=True)
                prepare_actuator_delay_boundary(f"sequence_cycle_{cycle:04d}_env_reset")
                obs, _ = env.reset()
                latest_actuator_delay_readback = record_actuator_delay_reset(f"sequence_cycle_{cycle:04d}_env_reset")
                require_actuator_delay_match(latest_actuator_delay_readback)
                initial = dict(zip(robot.joint_names, _array(robot.data.joint_pos)[0]))
                source.targets.update({n: float(initial[n]) for n in command_names})
                source.base_velocity[:] = 0
                for _ in range(max(int(args.record), math.ceil(args.settle_seconds / dt))):
                    step()
            if args.sequence:
                live.begin_episode()
            completed = []
            for task_id in selected:
                if (args.output / "STOP").exists():
                    report["operator_stopped"] = True
                    break
                # Acquisition evidence cannot cross an episode/object boundary.
                current_command = None
                current_task_id = task_id
                for arm in pinch_settling:
                    pinch_settling[arm].reset()
                    settling_object_paths[arm] = None
                episode_id = f"cycle_{cycle:04d}_{dataset_task_id(task_id)}"
                directory = args.output / episode_id
                directory.mkdir()
                initialization = None
                if args.sequence:
                    initialization = {"kind": "sequence_existing_scene",
                                      "actuator_delay": latest_actuator_delay_readback}
                initial_events = []
                task_prerequisites = completed
                if not args.sequence:
                    initialization_start = tick * dt
                    try:
                        from .nepes_initialization import prepare_independent_task
                        initialization = prepare_independent_task(config, task_id, baseline_poses)
                        live.end_episode()
                        live.reset_objects(episode_boundary=True)
                        prepare_actuator_delay_boundary(f"{episode_id}_env_reset")
                        obs, _ = env.reset()
                        latest_actuator_delay_readback = record_actuator_delay_reset(f"{episode_id}_env_reset")
                        initialization["actuator_delay"] = latest_actuator_delay_readback
                        require_actuator_delay_match(latest_actuator_delay_readback)
                        live.set_initialization_poses(
                            {p: pose for p, pose in initialization["object_poses_w"].items() if p in live.object_paths.values()},
                            episode_boundary=True)
                        initialization["settle_object_measurements"] = [
                            {"control_step": 0, "time_s": tick * dt, "states": live.read_states()}]
                        if prepared_grasp is not None:
                            initial = apply_prepared_grasp()
                            initialization["prepared_grasp_initialization"] = prepared_grasp_provenance(prepared_grasp)
                            initialization["initial_policy_history"] = dict(report["prepared_policy_history"])
                            initialization["settling_method"] = report["prepared_settling_method"]
                            initialization["object_poses_w"][config["objects"]["crs"]] = list(prepared_grasp["object_pose_w"])
                            initialization["robot_reset_pose_w"] = list(prepared_grasp["robot_root_pose_w"])
                            initialization["settle_object_measurements"] = [
                                {"control_step": 0, "time_s": tick * dt, "states": live.read_states()}]
                        elif upright:
                            # Upright start: root, overrides and gaze only; the reach,
                            # pickup and every arm motion are recorded demonstration.
                            initial, initialization["upright_start"] = apply_upright_start(
                                task_id, initialization["object_poses_w"], EPISODE_UPRIGHT_START_SCOPE)
                            initialization["initial_policy_history"] = dict(report["upright_policy_history"])
                            initialization["robot_reset_pose_w"] = list(initialization["upright_start"]["robot_root_pose_w"])
                            initialization["upper_posture_initialization"] = None
                        else:
                            spawn = initialization["robot_source_base_pose_w"]
                            if spawn is not None:
                                # Stations specify planar placement. Preserve the asset's
                                # supported reset height instead of lowering feet into the floor.
                                spawn = list(spawn)
                                spawn[2] = float(robot.data.default_root_state[0, 2])
                                initialization["robot_reset_pose_w"] = spawn
                                robot.write_root_pose_to_sim(torch.tensor([spawn], dtype=torch.float32, device=raw.device))
                                robot.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=raw.device))
                            initial = dict(zip(robot.joint_names, _array(robot.data.joint_pos)[0]))
                            source.targets.update({n: float(initial[n]) for n in command_names})
                            initial = prepare_upper_posture(task_id, initial)
                            initialization["initial_upper_joint_positions"] = config["tasks"][task_id].get("initial_upper_joint_positions", {})
                            initialization["upper_posture_initialization"] = posture_reset_records.get(task_id)
                            if args.initialize_arm_at_pregrasp:
                                from .goal_ik import solve_goal_candidate
                                task = config["tasks"][task_id]
                                object_pose = initialization["object_poses_w"][config["objects"][task["object"]]]
                                pregrasp = pose_to_matrix(object_pose) @ pose_to_matrix(task["grasp_pose_object"])
                                pregrasp[:3, 3] += task["pregrasp_offset_w"]
                                # Joint reset is only an episode-boundary fixture. A
                                # separately recorded physical settle follows. Use
                                # the calibrated standing height, not the higher
                                # unsupported asset reset height, for initial IK.
                                reference_base = list(task.get("initial_robot_pose_w", task["base_stations"]["source"]))
                                posture_reset = initialization["upper_posture_initialization"]
                                if posture_reset is not None:
                                    reference_base = pelvis_preserving_boundary_pose(
                                        servo.model, reference_base, posture_reset["joint_positions_before"], initial)
                                initial_candidate = None
                                configured = task.get("initial_arm_joint_positions")
                                if configured is not None:
                                    arm_targets = _configured_initial_arm_targets(task, servo.model, servo.limits, initial)
                                else:
                                    initial_candidate = solve_goal_candidate(
                                        servo.model, initial, task["arm"],
                                        matrix_to_pose(np.linalg.inv(pose_to_matrix(reference_base)) @ pregrasp))
                                    if not initial_candidate["success"]:
                                        raise RuntimeError("pregrasp_initialization_ik_failed")
                                    arm_targets = initial_candidate["active_joint_positions"]
                                arm_ids = [robot.joint_names.index(n) for n in arm_targets]
                                ramp_seconds = float(task.get("initial_arm_ramp_seconds", 0.0))
                                if not math.isfinite(ramp_seconds) or not 0 <= ramp_seconds <= 30:
                                    raise ValueError("initial_arm_ramp_seconds must be in [0, 30]")
                                ramp_measurements = []
                                if ramp_seconds:
                                    # Boundary preparation uses the real actuators
                                    # and WBC, allowing balance to respond while the
                                    # arm is raised. This motion is not a recorded
                                    # or collision-certified approach demonstration.
                                    source.base_velocity[:] = 0
                                    ramp_steps = math.ceil(ramp_seconds / dt)
                                    for ramp_index in range(1, ramp_steps + 1):
                                        alpha = ramp_index / ramp_steps
                                        blend = alpha * alpha * (3 - 2 * alpha)
                                        source.targets.update({n: initial[n] + blend * (q - initial[n])
                                                               for n, q in arm_targets.items()})
                                        step()
                                        sample = measured()
                                        if ramp_index == 1 or ramp_index % 25 == 0 or ramp_index == ramp_steps:
                                            ramp_measurements.append({"time_s": sample["time_s"],
                                                "base_pose_w": sample["base_pose_w"],
                                                "joint_positions": sample["joint_positions"], "faults": sample["faults"]})
                                        if sample["faults"]:
                                            initialization["arm_ramp_measurements"] = ramp_measurements
                                            raise RuntimeError("arm_initialization_ramp:" + ";".join(sample["faults"]))
                                else:
                                    robot.write_joint_state_to_sim(
                                        torch.tensor([list(arm_targets.values())], dtype=torch.float32, device=raw.device),
                                        torch.zeros((1, len(arm_ids)), dtype=torch.float32, device=raw.device), joint_ids=arm_ids)
                                source.targets.update(arm_targets)
                                initial.update(arm_targets)
                                initialization["arm_initialization"] = {
                                    "scope": "episode_boundary_before_physics_settle_and_recording",
                                    "method": "configured_joint_pose" if configured else "bounded_cpu_pregrasp_ik",
                                    "joint_positions": arm_targets, "pregrasp_pose_w": matrix_to_pose(pregrasp),
                                    "reference_base_pose_w": reference_base, "ik_candidate": initial_candidate,
                                    "ramp_seconds": ramp_seconds, "ramp_measurements": ramp_measurements,
                                    "approach_trajectory_generated": False, "object_attached": False,
                                    "collision_checked": False}
                            if not args.no_head_gaze:
                                from .head_gaze import recommend_head_joint_targets
                                task = config["tasks"][task_id]
                                object_pose = initialization["object_poses_w"][config["objects"][task["object"]]]
                                gaze_target = (pose_to_matrix(object_pose) @ pose_to_matrix(task["grasp_pose_object"]))[:3, 3]
                                base_pose = (posture_reset_records[task_id]["root_after_pose_w"]
                                             if task_id in posture_reset_records
                                             else spawn if spawn is not None else measured()["base_pose_w"])
                                gaze = recommend_head_joint_targets(servo.model, initial, base_pose, gaze_target,
                                                                   variant.camera_attachments)
                                neck_targets = gaze["joint_targets_rad"]
                                neck_ids = [robot.joint_names.index(n) for n in neck_targets]
                                robot.write_joint_state_to_sim(
                                    torch.tensor([list(neck_targets.values())], dtype=torch.float32, device=raw.device),
                                    torch.zeros((1, len(neck_ids)), dtype=torch.float32, device=raw.device),
                                    joint_ids=neck_ids)
                                source.targets.update(neck_targets)
                                initialization["head_gaze"] = gaze
                        source.base_velocity[:] = 0
                        settle_steps = 1 if prepared_grasp is not None else max(1, math.ceil(args.settle_seconds / dt))
                        for settle_index in range(settle_steps):
                            step()
                            if settle_index + 1 in {1, 2, 5, 10, 25, 50, 100, settle_steps}:
                                settle_sample = measured()
                                initialization["settle_object_measurements"].append({
                                    "control_step": settle_index + 1, "time_s": tick * dt,
                                    "states": live.read_states(), "robot_base_pose_w": settle_sample["base_pose_w"],
                                    "robot_joint_positions": settle_sample["joint_positions"],
                                    "robot_faults": settle_sample["faults"]})
                        task_prerequisites = initialization["satisfied_prerequisites"]
                        initial_events = initialization["events"]
                        live.begin_episode()
                        initial_sample = measured()
                        initial_sample.update(robot_model_urdf=str(variant.urdf_path.resolve()),
                                              measurement_phase="settled_task_initialization")
                        if layout_contract is not None:
                            initialization["single_arm_layout"] = measured_layout_provenance(
                                layout_contract, initial_sample["object_poses_w"],
                                phase=initial_sample["measurement_phase"])
                            initial_sample["single_arm_layout"] = deepcopy(initialization["single_arm_layout"])
                        _json(directory / "initialization.json", initialization)
                        _json(directory / "initial_measurement.json", initial_sample)
                    except Exception as exc:
                        # An independent fixture/stance failure invalidates this
                        # attempt only. No camera frame has been recorded yet.
                        capture["active"] = False
                        source.base_velocity[:] = 0
                        live.end_episode()
                        reason = f"initialization:{type(exc).__name__}: {exc}"
                        result = {"episode_id": episode_id, "task_id": dataset_task_id(task_id), "success": False,
                                  "episode_scope": collection_scope, "full_task_completed": False,
                                  "failure_stage": "initialization", "reason": reason,
                                  "last_stage": "initialization", "duration_s": tick * dt - initialization_start,
                                  "seed": args.seed + cycle, "calibration_status": report["calibration_status"],
                                  "execution_mode": report["execution_mode"], "initialization": initialization,
                                  "collision_free_execution_verified": False,
                                  "simulated_wash_seconds": args.simulate_wash_seconds,
                                  "recorded_frame_count": 0}
                        if initialization is not None:
                            _json(directory / "initialization.json", initialization)
                        _json(directory / "result.json", result)
                        report["episodes"].append(result)
                        _json(args.output / "run.json", _with_ik_backend(report))
                        if writer:
                            writer.begin_episode(episode_id, dataset_task_number(task_id),
                                episode_title(task_id),
                                provenance={"kind": "original", "cycle": cycle, "initialization": initialization,
                                            "episode_scope": collection_scope,
                                            "failure_stage": "initialization",
                                            **({"single_arm_layout": layout_contract}
                                               if layout_contract is not None else {})})
                            writer.abort(reason, metrics=result)
                        print(f"[NEPES] Initialization failed for {episode_id}: {reason}; continuing next independent task.", flush=True)
                        continue
                machine = machine_type(config, task_id, seed=args.seed + cycle, completed_tasks=task_prerequisites,
                                       **({"goal_settings": stationary_spec} if stationary else
                                          {"offset_w": args.carry_offset_w} if primitive else {}))
                validate_diagnostic_stop(diagnostic_stop_after, machine.stages)
                validate_diagnostic_waypoint_stop(diagnostic_stop_after_waypoint,
                                                 config["tasks"][task_id], machine.stages)
                diagnostic_transition = None
                elapsed_start = tick * dt
                plan_key, plan_frames, plan_names, plan_index = None, None, None, 0
                last_stage, reason = None, ""
                task_settings = config["tasks"][task_id]
                upper_body_settings = task_settings.get("upper_body_ik")
                upper_body_servo = None
                upper_body_stage_servos = {}
                transfer_early_released = False  # opt-in transfer_early_command_limits latch (runtime_v22)
                secondary_schedule_state = None  # opt-in transfer_secondary_arm_schedule state (runtime_v22)
                transfer_schedule = None  # validated transfer_secondary_arm_schedule (runtime_v22)
                stage_waist_hold = None  # opt-in upper_body_ik.stage_waist_hold latch of the current stage (runtime_v26)
                pelvis_motion_compensation = None
                upper_target_feedforward = None
                if upper_body_settings is not None:
                    from .upper_body_control import build_upper_body_servos
                    from .upper_body_guidance import guidance_arm
                    from .upper_body_ik_backend import backend_report
                    try:
                        upper_body_servo, upper_body_stage_servos = build_upper_body_servos(
                            servo.model, upper_body_settings, task_settings["arm"],
                            max_joint_speed=servo.max_joint_speed, max_tracking_error=servo.max_tracking_error,
                            stage_arms={stage: guidance_arm(task_settings, stage)
                                        for stage in upper_body_settings["stages"]})
                    finally:
                        # Pink release: the selected IK backend, worker interpreter, package versions and
                        # source hashes, or the startup failure that stops this attempt (no fallback).
                        report["upper_body_ik_backend"] = backend_report()
                    from .upper_body_control import validate_upper_body_ik
                    transfer_schedule = validate_upper_body_ik(upper_body_settings).get("transfer_secondary_arm_schedule")
                    if upper_body_settings.get("target_motion_feedforward_stages"):
                        from .upper_body_feedforward import UpperBodyTargetMotionFeedforward
                        upper_target_feedforward = UpperBodyTargetMotionFeedforward(
                            upper_body_settings["target_motion_feedforward_stages"])
                auxiliary_support = (AuxiliarySupportController(task_settings["auxiliary_support_arm"],
                    task_settings["arm"], machine.object_path) if "auxiliary_support_arm" in task_settings else None)
                if auxiliary_support is not None:
                    jaw = source.grippers[auxiliary_support.arm]["joint_name"]
                    goal = auxiliary_support.settings["joint_position_rad"]
                    if jaw != auxiliary_support.joint_name or not servo.limits[jaw][0] <= goal <= servo.limits[jaw][1]:
                        raise ValueError("Auxiliary support jaw differs from the loaded model geometry or limits")
                navigation = task_settings.get("navigation_control", {})
                single_step_settings = navigation.get("single_step")  # opt-in (single_step.py); None: unchanged
                base_limiter = BaseCommandLimiter(
                    linear_acceleration=navigation.get("linear_acceleration_m_s2", .05),
                    yaw_acceleration=navigation.get("yaw_acceleration_rad_s2", .08))
                carry_stabilizer, carry_stabilizer_stage, carry_tracking_gate = None, None, None
                carry_feedforward, carry_feedforward_stage = None, None
                torso_feedforward, torso_feedforward_stage = None, None
                retreat_feedforward = (RetreatMotionFeedforward()
                    if task_settings.get("retreat_motion_feedforward", False) else None)
                object_controller, object_controller_stage = None, None
                object_controller_context, object_continuity_event = None, None
                completed_transfer_settings = task_settings.get("transfer_completed_reference_continuity")
                completed_transfer_context, completed_transfer_event = None, None
                # Opt-in blended pass-through (reference_continuity.py, runtime_v26): {stage: settings} of the
                # declared lift/transfer stages, else None and every object segment keeps its own controller.
                blended_stages = None
                if any(isinstance(task_settings.get(key), dict)
                       for key in ("lift_reference_continuity", "transfer_reference_continuity")):
                    from .reference_continuity import continuity_settings
                    blended_stages = continuity_settings(task_settings)
                blended_context, blended_event, blended_run = None, None, 0

                def feedforward_index(command):
                    """Feedforward segment key: the waypoint, or the blended reference run spanning waypoints."""
                    run = (command.get("object_reference_continuity") or {}).get("run_id")
                    return command.get("object_waypoint_index", 0) if run is None else ("blended_reference_run", run)
                retreat_controller = None
                retreat_controller_stage = None
                pregrasp_controller = None
                # Opt-in handover (task["handover"]): the receiving hand's bounded approach reference and the
                # giving hand's (second servoed arm) bounded separation/retreat reference.
                handover_settings = task_settings.get("handover")
                handover_controller, handover_controller_stage, handover_goal, handover_reference = None, None, None, None
                secondary_controller, secondary_controller_stage = None, None

                def arm_tracking_limits(limits_arm):
                    """The task's per-joint tracking limits restricted to one arm (handover profiles servo both)."""
                    limits = task_settings.get("arm_tracking_error_limits_rad")
                    if handover_settings is None or limits is None:
                        return limits
                    return {name: value for name, value in limits.items()
                            if name in servo.model.arm_joint_names(limits_arm)}
                posture_controller, posture_stage = None, None
                posture_executor = None
                hinge_follower = None
                guarded_approach = None
                from .posture_control import ArmPostureGuide
                posture_guidance = task_settings.get("arm_posture_guidance", {})
                arm_guides = {stage_name: ArmPostureGuide(waypoints, servo.limits,
                                servo.model.arm_joint_names(task_settings["arm"]))
                              for stage_name, waypoints in posture_guidance.get("stages", {}).items()}
                upper_posture_guide = None
                if "upper_body_posture_guidance" in task_settings:
                    from .upper_body_guidance import UpperBodyPostureGuide
                    upper_posture_guide = UpperBodyPostureGuide(task_settings, servo.model,
                        command_limits=upper_body_servo.limits if upper_body_servo is not None else None)
                progress_watchdog = (MotionProgressWatchdog(**task_settings["motion_watchdog"])
                                     if "motion_watchdog" in task_settings else None)
                success = False
                support_unloading_was_active = False
                object_symmetry_axis = symmetry_axis(task_settings)
                object_pivot_radius_m = symmetry_pivot_radius(task_settings)
                grip_relief = (SupportedGripRelief(task_settings["supported_grip_relief"],
                    object_path=machine.object_path, support_path=config["objects"]["jig"],
                    capture_position_m=config["thresholds"]["capture_position_m"],
                    capture_rotation_rad=config["thresholds"]["capture_rotation_rad"],
                    object_symmetry_axis=object_symmetry_axis, object_pivot_radius_m=object_pivot_radius_m,
                    # runtime_v26 opt-in keys only ({} otherwise): filtered/activation support, pinch reduction.
                    **relief_task_options(task_settings, slot_pose=getattr(machine, "_slot_pose", None)))
                    if "supported_grip_relief" in task_settings else None)
                writer_started = False
                prepared_recording = None
                capture.update(frame_count=0, first_sim_time_s=None, last_sim_time_s=None)
                if writer and recording_boundary_settings is not None:
                    from .recording_boundary import recording_boundary_type
                    # The catalog pins each goal's kind: pregrasp (v1 lift),
                    # grasp hold (v1 placement) or upright stance (v2 pickup).
                    boundary_type = recording_boundary_type(recording_boundary_settings)
                    prepared_recording = boundary_type(recording_boundary_settings,
                        arm=machine.arm, object_path=machine.object_path,
                        thresholds=config["thresholds"], stationary_spec=stationary_spec,
                        attempt_start_time_s=elapsed_start)

                def begin_recording(boundary=None):
                    nonlocal writer_started
                    provenance = {**report, "cycle": cycle, "kind": "original", "initialization": initialization,
                                  "episode_scope": collection_scope}
                    if boundary is not None:
                        provenance["recording_boundary"] = boundary
                    writer.begin_episode(episode_id, dataset_task_number(task_id), episode_title(task_id),
                                         provenance=provenance)
                    writer_started = True
                    scheduler.reset()
                    camera.reset_timestamps()
                    capture.update(active=True, physics_step=0, origin=float(raw.sim.current_time))
                    for _ in range(3):
                        raw.sim.render()
                    record_physics_frame(0)

                if writer:
                    if camera_layout == "head_hand" and camera_arm != config["tasks"][task_id]["arm"]:
                        camera_arm = config["tasks"][task_id]["arm"]
                        camera.set_hand_arm(camera_arm)
                    if prepared_recording is None:
                        current_command = {"stage_id": "prerequisites"}
                        begin_recording()
                with (directory / "trace.jsonl").open("w", encoding="utf-8") as trace:
                    try:
                        while tick * dt - elapsed_start < args.max_task_seconds:
                            if (args.output / "STOP").exists():
                                report["operator_stopped"] = True
                                reason = "operator_stopped"
                                machine.fail(reason)
                                break
                            observation = measured()
                            observation["events"] = list(initial_events)
                            if args.events_file and args.events_file.exists():
                                observation["events"].extend(json.loads(args.events_file.read_text())["events"])
                            if (task_id == "jig_extract_cleaner" and args.simulate_wash_seconds is not None
                                    and "jig_insert_cleaner" in completed
                                    and tick * dt - elapsed_start >= args.simulate_wash_seconds):
                                observation["events"].append("wash_complete")
                            if auxiliary_support is not None:
                                observation["auxiliary_support"] = auxiliary_support.observe(machine.stage_id, observation)
                            if pregrasp_controller is not None and machine.stage_id == "pregrasp":
                                observation["pregrasp_reference"] = pregrasp_controller.last_result
                            # The upper command applied in the previous control step, i.e.
                            # the one that produced this measurement (upright readiness and
                            # posture tracking compare measured joints with it).
                            observation["commanded_upper_joint_positions"] = dict(source.targets)
                            if single_step_settings is not None:
                                # Opt-in single step: the base command applied in the previous control step
                                # (its commanded travel can latch the one-shot brake).
                                observation["commanded_base_velocity"] = [float(v) for v in source.base_velocity]
                            if posture_executor is not None and (machine.stage_id in JOINT_PATH_STAGES or (
                                    single_step_settings is not None and machine.stage_id == "navigate_target"
                                    and posture_executor.stage_id == "navigate_target")):
                                observation["posture_reference"] = posture_executor.status
                            previous_stage_id = machine.stage_id
                            previous_waypoint_command = (machine._command()
                                if diagnostic_waypoint_stop is not None else None)
                            completed_pregrasp_goal = (matrix_to_pose(machine._tcp_target)
                                if previous_stage_id == "pregrasp" and machine._tcp_target is not None else None)
                            current_command = machine.step(observation)
                            stage_id = current_command["stage_id"]
                            if stage_id != previous_stage_id:
                                pelvis_motion_compensation = None
                                if upper_target_feedforward is not None:
                                    upper_target_feedforward.reset()
                                # Opt-in latches (runtime_v26) are taken again at the first step of every stage.
                                stage_waist_hold = None
                                if upper_posture_guide is not None:
                                    upper_posture_guide.reset_entry_postures()
                            stop_reason = diagnostic_stop_reason(diagnostic_stop_after, previous_stage_id, current_command)
                            if stop_reason is None and diagnostic_waypoint_stop is not None:
                                stop_reason = diagnostic_waypoint_stop_reason(
                                    diagnostic_waypoint_stop, previous_waypoint_command, current_command)
                                if stop_reason is not None:
                                    diagnostic_transition = {
                                        "trigger": "measured_fsm_waypoint_advance",
                                        "stage_id": stage_id,
                                        "completed_waypoint_index": diagnostic_waypoint_stop[1],
                                        "next_waypoint_index": current_command["object_waypoint_index"],
                                        "measurement_time_s": observation["time_s"],
                                        "next_waypoint_commands_applied": False,
                                        "completed_fsm_tcp_goal_pose_w": deepcopy(
                                            previous_waypoint_command.get("tcp_target_pose_w")),
                                        "next_unapplied_fsm_tcp_goal_pose_w": deepcopy(
                                            current_command.get("tcp_target_pose_w")),
                                        "last_applied_upper_joint_positions": dict(source.targets)}
                            if stop_reason is not None:
                                # Calibration stop on entering the next stage: before its
                                # recording boundary or any of its commands; never a success.
                                current_command = machine.fail(stop_reason)
                                if diagnostic_transition is not None:
                                    current_command["diagnostic_stop_transition"] = deepcopy(diagnostic_transition)
                            if prepared_recording is not None and current_command["status"] == "running":
                                boundary = prepared_recording.update(observation, current_command,
                                    previous_stage_id=previous_stage_id,
                                    pregrasp_goal_pose_w=completed_pregrasp_goal)
                                current_command["recording_boundary"] = boundary
                                if boundary["status"] == "failed":
                                    current_command = machine.fail(boundary["reason"])
                                    current_command["recording_boundary"] = boundary
                                elif boundary["start_recording"]:
                                    # Frame zero must precede new approach targets:
                                    # source.targets and last_action still describe
                                    # the commands actually applied to this state.
                                    begin_recording(prepared_recording.mark_started(float(raw.sim.current_time)))
                            if auxiliary_support is not None and current_command["status"] == "running":
                                auxiliary_state = auxiliary_support.observe(stage_id, observation)
                                current_command["auxiliary_support"] = auxiliary_state
                                observation["auxiliary_support"] = auxiliary_state
                                auxiliary_reason = (auxiliary_state["reason"] if auxiliary_state["status"] == "failed"
                                    else "diagnostic_stop:auxiliary_support_lift_prefix" if auxiliary_state.get("diagnostic_stop") else None)
                                if auxiliary_reason:
                                    current_command = machine.fail(auxiliary_reason)
                                    current_command["auxiliary_support"] = auxiliary_state
                            if stage_id != last_stage:
                                print(f"[NEPES] {episode_id}: {stage_id}", flush=True)
                                last_stage = stage_id
                            if current_command["status"] != "running":
                                trace.write(json.dumps({"observation": observation, "command": current_command,
                                                        "applied_policy_action_previous_step": last_action.tolist()},
                                                       ensure_ascii=False, allow_nan=False) + "\n")
                                success = current_command["status"] == "succeeded"
                                reason = current_command.get("reason") or ""
                                break
                            arm = current_command["arm"]
                            motion = current_command.get("motion_mode", "contact_servo")
                            desired_base_velocity = np.zeros(3)
                            turn_yaw_acceleration = None
                            if stage_id == "open_hinge" and "hinge_following_path" in task_settings:
                                from .hinge_controller import HingeFollowingPath, HingeFollower
                                from .hinged_door import hinge_pose_w
                                if hinge_follower is None:
                                    path = HingeFollowingPath(config, task_settings["hinge_following_path"], servo.limits)
                                    if not np.allclose(path.grasp, pose_to_matrix(task_settings["grasp_pose_object"])):
                                        raise ValueError("Hinge path grasp differs from the task grasp")
                                    hinge_follower = HingeFollower(
                                        path, lost_grasp_grace_s=config["thresholds"]["lost_grasp_grace_s"])
                                following = hinge_follower.step(
                                    cover_pose_w=observation["object_poses_w"][machine.object_path],
                                    base_pose_w=observation["base_pose_w"],
                                    joint_positions=observation["joint_positions"],
                                    hand_captured=machine._grasp_valid(machine._last_observation), dt=dt)
                                current_command["hinge_follower"] = following
                                if following["status"] == "failed":
                                    raise RuntimeError("hinge_follower:" + following["reason"])
                                current_command["tcp_target_pose_w"] = matrix_to_pose(
                                    pose_to_matrix(hinge_pose_w(config, following["hinge_angle_rad"])) @ machine._object_to_tcp)
                                desired_base_velocity = base_velocity_command(
                                    observation["base_pose_w"], following["base_goal_pose_w"],
                                    max_speed=navigation.get("max_speed_m_s", .12),
                                    max_yaw_rate=navigation.get("max_yaw_rate_rad_s", .12),
                                    position_gain=navigation.get("position_gain_s_inv", 1.8),
                                    yaw_gain=navigation.get("yaw_gain_s_inv", 1.8))
                                waist_targets = following["waist_joint_targets"]
                                if not set(waist_targets) <= {"Joint_Waist_Yaw", "Joint_Waist_Roll", "Joint_Waist_Pitch"}:
                                    raise ValueError("Hinge waist targets may name only waist joints")
                                if following["paused_for_contact"]:
                                    desired_base_velocity[:] = 0
                                else:
                                    for name, q in waist_targets.items():
                                        source.targets[name] += float(np.clip(q - source.targets[name], -.08 * dt, .08 * dt))
                            if motion == "navigate_base":
                                desired_base_velocity = base_velocity_command(
                                    observation["base_pose_w"], current_command["base_goal_pose_w"],
                                    max_speed=navigation.get("max_speed_m_s", .12),
                                    max_yaw_rate=navigation.get("max_yaw_rate_rad_s", .25),
                                    position_gain=navigation.get("position_gain_s_inv", .6),
                                    yaw_gain=navigation.get("yaw_gain_s_inv", .6))
                                if (navigation.get("predictive_braking", False) and stage_id in navigation.get(
                                        "predictive_braking_stages", ["navigate_source", "navigate_target", "recover_base"])):
                                    station = "recovery" if stage_id == "recover_base" else stage_id.removeprefix("navigate_")
                                    position_tolerance, yaw_tolerance = machine.navigation_tolerances(station)
                                    desired_base_velocity = arrival_limited_base_velocity(
                                        desired_base_velocity, observation["base_pose_w"], current_command["base_goal_pose_w"],
                                        position_tolerance=position_tolerance, yaw_tolerance=yaw_tolerance,
                                        linear_acceleration=navigation.get("linear_acceleration_m_s2", .05),
                                        yaw_acceleration=navigation.get("yaw_acceleration_rad_s2", .08))
                                    current_command["base_arrival_limited_velocity"] = desired_base_velocity.tolist()
                                if "turn_in_place" in navigation:
                                    # Opt-in: a goal at the current planar position with a new heading
                                    # gets its own yaw cap, yaw acceleration and deadband floor; XY and
                                    # the arrival test (measured torso within the station tolerances) are
                                    # unchanged. The policy's actual yaw response is not certified here.
                                    station = "recovery" if stage_id == "recover_base" else stage_id.removeprefix("navigate_")
                                    position_tolerance, yaw_tolerance = machine.navigation_tolerances(station)
                                    desired_base_velocity, turn = turn_in_place_base_velocity(
                                        desired_base_velocity, observation["base_pose_w"], current_command["base_goal_pose_w"],
                                        settings=navigation["turn_in_place"], position_tolerance=position_tolerance,
                                        yaw_tolerance=yaw_tolerance, yaw_gain=navigation.get("yaw_gain_s_inv", .6))
                                    current_command["turn_in_place"] = turn
                                    turn_yaw_acceleration = turn["yaw_acceleration_rad_s2"]
                                if navigation.get("command_frame", "torso") == "pelvis":
                                    # Opt-in: the checkpoint realizes XY in the leg heading frame
                                    # when the waist is twisted (task_control docstring).
                                    desired_base_velocity = base_command_in_heading_frame(
                                        desired_base_velocity, observation["base_pose_w"], observation["pelvis_pose_w"])
                                    current_command["base_command_frame"] = "pelvis"
                                if current_command.get("base_arrival_braking", False):
                                    desired_base_velocity[:] = 0
                                if stage_id in {"navigate_target", "recover_base"} and navigation.get("stabilize_carry", False):
                                    carry_base_key = ("pelvis_pose_w" if navigation.get("command_frame") == "pelvis"
                                                      else "base_pose_w")
                                    if carry_stabilizer_stage != stage_id:
                                        carry_stabilizer = CarryStabilizer(
                                            observation[carry_base_key], observation["tcp_poses_w"][arm],
                                            max_linear_speed=navigation.get("carry_tcp_speed_m_s", .12),
                                            max_yaw_rate=navigation.get("carry_tcp_yaw_rate_rad_s", .10))
                                        carry_tracking_gate = (CarryTrackingGate(**navigation["carry_tracking_gate"])
                                                               if "carry_tracking_gate" in navigation else None)
                                        carry_stabilizer_stage = stage_id
                                    current_command["tcp_target_pose_w"] = carry_stabilizer.update(
                                        observation[carry_base_key], dt).tolist()
                                    current_command["carry_reference"] = (
                                        "measured_planar_pelvis_world_height_stabilized" if carry_base_key == "pelvis_pose_w"
                                        else "measured_planar_base_world_height_stabilized")
                                    if carry_tracking_gate is not None:
                                        diagnostics = carry_stabilizer.diagnostics(observation["tcp_poses_w"][arm])
                                        tracking = carry_tracking_gate.update(
                                            max(diagnostics["position_error_m"], diagnostics["base_xy_lag_m"]),
                                            max(diagnostics["rotation_error_rad"], diagnostics["base_yaw_lag_rad"]))
                                        current_command["carry_tracking"] = {
                                            **diagnostics, **tracking,
                                            "tcp_position_error_m": diagnostics["position_error_m"],
                                            "tcp_rotation_error_rad": diagnostics["rotation_error_rad"],
                                            "gate_error_source": "maximum_tcp_tracking_and_planar_base_filter_lag"}
                                        desired_base_velocity *= tracking["speed_scale"]
                            waypoint_posture = stage_id in task_settings.get("object_waypoint_postures", {})
                            posture_index = current_command.get("object_waypoint_index", 0)
                            posture = stage_posture(task_settings, stage_id, posture_index)
                            posture_key = (stage_id, posture_index) if waypoint_posture else stage_id
                            if posture is not None:
                                if posture_stage != posture_key:
                                    goal = posture["joint_positions"]
                                    if (not set(goal) <= set(command_names)
                                            or set(goal) & set(servo.model.arm_joint_names(arm))):
                                        raise ValueError("Stage posture must use inactive-arm, torso or neck command joints")
                                    if any(not servo.limits[n][0] <= q <= servo.limits[n][1] for n, q in goal.items()):
                                        raise ValueError("Stage posture exceeds shared control limits")
                                    if posture.get("feedback", False):
                                        posture_controller = MeasuredPostureController(
                                            observation["joint_positions"], source.targets, goal, servo.limits,
                                            minimum_duration_s=posture.get("duration_s", 3.))
                                    else:
                                        posture_controller = PostureReference(
                                            source.targets, goal, minimum_duration_s=posture.get("duration_s", 3.))
                                    posture_stage = posture_key
                                if posture.get("feedback", False):
                                    posture_targets, posture_diagnostics = posture_controller.update(observation["joint_positions"], dt)
                                    source.targets.update(posture_targets)
                                    current_command["posture_feedback"] = posture_diagnostics
                                else:
                                    source.targets.update(posture_controller.update(dt))
                                current_command["posture_reference_duration_s"] = posture_controller.duration
                                if waypoint_posture:
                                    current_command["object_waypoint_posture"] = {
                                        "stage_id": stage_id, "waypoint_index": posture_index,
                                        "joint_positions": dict(posture["joint_positions"]),
                                        "feedback": True}
                                posture_reached = (posture_controller.elapsed >= posture_controller.duration
                                    and all(abs(observation["joint_positions"][n] - q) <= posture.get("tolerance_rad", .03)
                                            for n, q in posture["joint_positions"].items()))
                                if motion == "navigate_base" and not posture_reached:
                                    desired_base_velocity[:] = 0
                                    current_command["base_paused_for_posture"] = True
                            if motion == "joint_posture":
                                # Upright pickup/load: the posture path alone owns all 19
                                # waist/arm/wrist/neck commands (jaws stay with the hand).
                                if current_command.get("tcp_target_pose_w") is not None:
                                    raise RuntimeError("joint_posture_stage_has_cartesian_target:" + stage_id)
                                if posture_executor is None or posture_executor.stage_id != stage_id:
                                    posture_executor = PostureStageExecutor(
                                        stage_id, posture_path_for_stage(task_settings, stage_id), start_targets=source.targets,
                                        measured_positions=observation["joint_positions"], command_names=command_names,
                                        joint_limits=servo.limits, model=servo.model,
                                        excluded_joint_names=[grip["joint_name"] for grip in source.grippers.values()])
                                joint_path_targets, current_command["posture_reference"] = posture_executor.update(
                                    observation["joint_positions"], source.targets, dt)
                                if joint_path_targets is None:
                                    trace.write(json.dumps({"observation": observation, "command": current_command,
                                        "commanded_upper_joint_positions": dict(source.targets)}, allow_nan=False) + "\n")
                                    raise RuntimeError(current_command["posture_reference"]["reason"])
                                source.targets.update(joint_path_targets)
                            # Opt-in single step: only navigate_target from its goal latch to its one-shot brake may
                            # command the base; before (pre-step posture) and after it the hold's exact zero applies.
                            single_step_stepping = (single_step_settings is not None and stage_id == "navigate_target"
                                                    and (current_command.get("single_step") or {}).get("phase")
                                                    == "stepping")
                            if ("no_walk" in navigation  # opt-in single stance: every stage, preparation included
                                    or single_step_settings is not None
                                    and single_step_settings.get("source_hold") is True
                                    and stage_id in {"prerequisites", "navigate_source"}
                                    or "stationary_pelvis_hold" in navigation
                                    and stage_id not in {"prerequisites", "navigate_source"}
                                    and not single_step_stepping):
                                # Waist articulation changes torso yaw at a stationary pelvis.
                                # A declared no-walk episode never uses that change as a
                                # navigation error to correct with a footstep command.
                                base_limiter.reset()
                                source.base_velocity = np.zeros(3)
                                current_command["stationary_base_command"] = True
                            elif current_command.get("carry_tracking", {}).get("paused", False):
                                # Stop command output while the arm catches up; HugWBC still
                                # controls the floating base and no robot state is written.
                                base_limiter.reset()
                                source.base_velocity = np.zeros(3)
                                current_command["base_protective_stop_overrides_acceleration_limit"] = True
                            elif current_command.get("hinge_follower", {}).get("paused_for_contact", False):
                                base_limiter.reset()
                                source.base_velocity = np.zeros(3)
                                current_command["base_protective_stop_overrides_acceleration_limit"] = True
                            elif motion == "joint_posture":
                                # Posture stages never step: an exact zero base command,
                                # not a decaying limiter output (recorded stages require it).
                                base_limiter.reset()
                                source.base_velocity = np.zeros(3)
                            else:
                                source.base_velocity = base_limiter.update(desired_base_velocity, dt,
                                                                           yaw_acceleration=turn_yaw_acceleration)
                            current_command["base_velocity_applied"] = source.base_velocity.tolist()
                            if (upper_target_feedforward is not None
                                    and (stage_id not in upper_target_feedforward.stages
                                         or current_command.get("tcp_target_pose_w") is None
                                         or current_command.get("placement_support_active", False)
                                         or current_command.get("placement_support_hold_active", False))):
                                upper_target_feedforward.reset()
                            if current_command.get("tcp_target_pose_w") is not None:
                                target = current_command["tcp_target_pose_w"]
                                arm_posture_target, arm_posture_gain = None, 0.
                                if stage_id == "pregrasp" and "pregrasp_pose_control" in task_settings:
                                    from .pregrasp import BoundedPregraspController
                                    if args.approach != "servo":
                                        raise ValueError("Bounded primary pregrasp requires --approach servo")
                                    if pregrasp_controller is None:
                                        pregrasp_controller = BoundedPregraspController(
                                            observation["tcp_poses_w"][arm], target, arm=arm,
                                            object_path=machine.object_path,
                                            settings=task_settings["pregrasp_pose_control"],
                                            waypoints=task_settings.get("pregrasp_waypoints", ()),
                                            position_tolerance_m=config["thresholds"]["tcp_position_m"],
                                            rotation_tolerance_rad=config["thresholds"]["tcp_rotation_rad"])
                                    reference = pregrasp_controller.update(
                                        observation["tcp_poses_w"][arm], dt,
                                        observation.get("contact_evidence", {}).get(arm, {}).get(machine.object_path, {}))
                                    current_command["pregrasp_pose_reference"] = reference
                                    current_command["fsm_tcp_goal_pose_w"] = target
                                    if reference["status"] == "failed":
                                        trace.write(json.dumps({"observation": observation, "command": current_command,
                                            "commanded_upper_joint_positions": dict(source.targets)}, allow_nan=False) + "\n")
                                        raise RuntimeError("pregrasp:" + reference["reason"])
                                    target = reference["tcp_target_pose_w"]
                                    current_command["tcp_target_pose_w"] = target
                                if hinge_follower is not None and stage_id == "open_hinge" and "hinge_posture_gain_s_inv" in task_settings:
                                    arm_posture_target = current_command["hinge_follower"]["arm_joint_seed"]
                                    arm_posture_gain = task_settings["hinge_posture_gain_s_inv"]
                                if stage_id == "approach" and "guarded_approach" in task_settings:
                                    from .hinge_controller import GuardedHandleApproach
                                    if guarded_approach is None:
                                        guarded_approach = GuardedHandleApproach(
                                            observation["tcp_poses_w"][arm], target, **task_settings["guarded_approach"])
                                    contact = observation["contact_evidence"][arm][machine.object_path]
                                    approach = guarded_approach.step(
                                        tcp_pose_w=observation["tcp_poses_w"][arm], dt=dt,
                                        contact_force_w=np.asarray(contact.get("forces_w", np.zeros((1, 1, 3)))).sum(axis=(0, 1)),
                                        contact_available=contact.get("available", False),
                                        contact_force_magnitude_n=contact.get("force_magnitude_sum_n"))
                                    current_command["guarded_approach"] = approach
                                    current_command["fsm_tcp_goal_pose_w"] = target
                                    if approach["status"] == "failed":
                                        trace.write(json.dumps({"observation": observation, "command": current_command,
                                            "commanded_upper_joint_positions": dict(source.targets)}, allow_nan=False) + "\n")
                                        raise RuntimeError("guarded_approach:" + approach["reason"])
                                    target = approach["tcp_goal_pose_w"]
                                    current_command["tcp_target_pose_w"] = target
                                reference_settings = task_settings.get("object_pose_control")
                                # A declared sustained-load latch (placement_support.latch_dwell_s) keeps the
                                # nominal descent running while confirmed support has not yet latched the hold.
                                if (reference_settings is not None and stage_id in {"lift", "recover_carry", "transfer", "place"}
                                        and not (current_command.get("placement_support_active", False)
                                                 and not current_command.get("placement_support_latch_pending", False))
                                        and not current_command.get("placement_support_hold_active", False)):
                                    object_segment = (stage_id, current_command.get("object_waypoint_index", 0))
                                    if object_controller_stage != object_segment:
                                        grasp = matrix_to_pose(machine._object_to_tcp)
                                        control_grasp = matrix_to_pose(machine._control_object_to_tcp)
                                        # Task settings refined per segment by any declared override;
                                        # object feedback still only in the object-feedback stages.
                                        stage_reference_settings = object_pose_settings(
                                            task_settings, stage_id, object_segment[1])
                                        goal_object = goal_object_from_tcp(target, control_grasp)
                                        preserve = False
                                        blended = None if blended_stages is None else blended_stages.get(stage_id)
                                        completed_scope = bool(completed_transfer_settings is not None and stage_id == "transfer"
                                            and completed_transfer_settings["from_waypoint"] <= object_segment[1]
                                            <= completed_transfer_settings["through_waypoint"])
                                        completed_transfer_event = None
                                        if completed_scope:
                                            from .completed_transfer_continuity import context as completed_context, offer as completed_offer, SCHEMA as COMPLETED_SCHEMA
                                            handoff_context = completed_context(stage_id, object_segment[1], arm,
                                                machine.object_path, grasp, control_grasp, stage_reference_settings,
                                                observation["time_s"])
                                            candidate, decision = completed_offer(object_controller,
                                                completed_transfer_context, handoff_context, goal_object,
                                                completed_transfer_settings, controller_factory=CrsPoseController)
                                            preserve = decision["accepted"]
                                            if preserve:
                                                object_controller = candidate
                                            object_continuity_event = None
                                            completed_transfer_event = {"schema_version": COMPLETED_SCHEMA,
                                                "measurement_time_s": observation["time_s"],
                                                "previous_segment": None if object_controller_stage is None else list(object_controller_stage),
                                                "current_segment": list(object_segment),
                                                "controller_state_preserved": preserve,
                                                "decision": decision, "measured_arrival_and_dwell_changed": False}
                                        elif blended is not None:
                                            # Opt-in blended pass-through: after the machine measured the arrival at
                                            # an eligible waypoint, the same reference continues into the next one.
                                            from .reference_continuity import (SCHEMA_VERSION as BLENDED_SCHEMA,
                                                offer_pass_through, segment_context as blended_segment_context)
                                            context = blended_segment_context(stage_id, object_segment[1], arm,
                                                machine.object_path, grasp, control_grasp)
                                            decision = offer_pass_through(object_controller, blended_context, context,
                                                goal_object, stage_reference_settings,
                                                current_command.get("reference_continuity"))
                                            preserve = decision["accepted"]
                                            blended_context = context
                                            if not preserve:
                                                blended_run += 1
                                            blended_event = {"schema_version": BLENDED_SCHEMA,
                                                "measurement_time_s": observation["time_s"], "stage_id": stage_id,
                                                "previous_segment": object_controller_stage,
                                                "current_segment": object_segment, "run_id": blended_run,
                                                "controller_preserved": preserve, "decision": decision,
                                                "acquisition_transform_rewritten_by_continuity": False}
                                        elif task_settings.get("lift_reference_continuity") == "collinear_online":
                                            from .lift_reference_continuity import segment_context, preserve_lift_reference
                                            context = segment_context(stage_id, object_segment[1], arm,
                                                machine.object_path, grasp, control_grasp, stage_reference_settings)
                                            decision = preserve_lift_reference(object_controller,
                                                object_controller_context, context, goal_object)
                                            preserve = decision["accepted"]
                                            object_continuity_event = {"schema_version": "igris_c_lift_reference_continuity_v1",
                                                "measurement_time_s": observation["time_s"],
                                                "previous_segment": object_controller_stage,
                                                "current_segment": object_segment,
                                                "controller_preserved": preserve, "decision": decision,
                                                "acquisition_transform_rewritten_by_continuity": False}
                                            object_controller_context = context
                                        if not preserve:
                                            reference_start = observation["tcp_poses_w"][arm]
                                            if stage_id == "place" and task_settings.get("place_aligned_descent") is not None:
                                                # Opt-in: a vertical nominal descent from above the goal; the
                                                # bounded feedback removes the transfer-end lateral error at once.
                                                from .place_aligned_descent import vertical_descent_start
                                                reference_start = vertical_descent_start(
                                                    goal_object, control_grasp, observation["object_poses_w"][machine.object_path])
                                                current_command["place_vertical_reference_start"] = {
                                                    "tcp_pose_w": list(reference_start),
                                                    "measured_tcp_pose_w": list(observation["tcp_poses_w"][arm]),
                                                    "goal_object_pose_w": list(goal_object),
                                                    "goal_or_gates_changed": False}
                                            two_speed = (place_two_speed_descent(task_settings)
                                                         if stage_id == "place" else None)
                                            if two_speed is not None:
                                                # runtime_v26 C4: the seat is the command goal shifted back by the
                                                # declared JIG-local command height offset (vertical component).
                                                offset = float(task_settings.get("placement_alignment", {}).get(
                                                    "command_height_offset_m", 0.))
                                                jig_reference = getattr(machine, "_placement_jig_reference", None)
                                                axis_z = (1. if jig_reference is None
                                                          else float(pose_to_matrix(jig_reference["jig_pose_w"])[2, 2]))
                                                two_speed = {"two_speed_descent": {
                                                    **two_speed, "goal_height_above_seat_m": offset * axis_z}}
                                            # blended (lift/transfer only) and two_speed (place only) never coexist.
                                            if blended is not None:
                                                from .reference_continuity import BlendedCrsPoseController
                                                object_controller = BlendedCrsPoseController(
                                                    grasp, reference_start, goal_object, continuity=blended,
                                                    waypoint_index=object_segment[1],
                                                    reference_object_to_tcp=control_grasp,
                                                    object_symmetry_axis=object_symmetry_axis,
                                                    object_pivot_radius_m=object_pivot_radius_m,
                                                    **stage_reference_settings)
                                            else:
                                                object_controller = CrsPoseController(
                                                    grasp, reference_start, goal_object,
                                                    reference_object_to_tcp=control_grasp,
                                                    object_symmetry_axis=object_symmetry_axis,
                                                    object_pivot_radius_m=object_pivot_radius_m, **stage_reference_settings,
                                                    **(two_speed or {}))
                                        object_controller_stage = object_segment
                                    reference = object_controller.update(
                                        observation["object_poses_w"][machine.object_path],
                                        observation["tcp_poses_w"][arm], dt,
                                        pinch_confirmed=bool(observation["hand_closed"][arm]
                                            and machine.object_path in observation["contacts"][arm]),
                                        hold_reason=(auxiliary_state.get("hold_reason")
                                            if auxiliary_support is not None else None),
                                        advance_hold_reason=current_command.get("object_advance_hold_reason"),
                                        feedback_hold_reason=current_command.get("object_feedback_hold_reason"))
                                    current_command["fsm_tcp_goal_pose_w"] = target
                                    current_command["object_pose_reference"] = reference
                                    if completed_transfer_settings is not None and stage_id == "transfer":
                                        from .completed_transfer_continuity import context as completed_context
                                        completed_transfer_context = completed_context(stage_id, object_segment[1], arm,
                                            machine.object_path, matrix_to_pose(machine._object_to_tcp),
                                            matrix_to_pose(machine._control_object_to_tcp), stage_reference_settings,
                                            observation["time_s"])
                                        if completed_transfer_event is not None:
                                            current_command["completed_transfer_reference_handoff"] = deepcopy(completed_transfer_event)
                                    if object_continuity_event is not None:
                                        current_command["object_pose_continuity"] = deepcopy(object_continuity_event)
                                    if blended_stages is not None and stage_id in blended_stages and blended_event is not None:
                                        current_command["object_reference_continuity"] = deepcopy(blended_event)
                                    target = reference["tcp_target_pose_w"]
                                    current_command["tcp_target_pose_w"] = target
                                    if stage_id in arm_guides:
                                        path_progress = reference["progress"]
                                        if stage_id == "recover_carry":
                                            path_progress = (current_command["recovery_prior_path_length_m"]
                                                + path_progress * current_command["recovery_segment_length_m"]
                                                ) / current_command["recovery_path_length_m"]
                                        elif current_command["object_path_length_m"] > 1e-8:
                                            path_progress = (current_command["object_prior_path_length_m"]
                                                + path_progress * current_command["object_segment_length_m"]
                                                ) / current_command["object_path_length_m"]
                                        arm_posture_target = arm_guides[stage_id].sample(float(np.clip(path_progress, 0., 1.)))
                                        arm_posture_gain = posture_guidance["gain_s_inv"]
                                if (stage_id == "retreat" and "retreat_pose_control" in task_settings
                                        or placement_tail and stage_id in {"release", "retreat"}):
                                    retreat_segment = retreat_segment_key(stage_id, current_command)
                                    if retreat_controller is None or retreat_controller_stage != retreat_segment:
                                        retreat_settings = retreat_pose_settings(
                                            task_settings, stage_id, current_command.get("release_phase"))
                                        retreat_controller = TcpPoseController(
                                            observation["tcp_poses_w"][arm], target, **retreat_settings)
                                        retreat_controller_stage = retreat_segment
                                    reference = retreat_controller.update(observation["tcp_poses_w"][arm], dt)
                                    current_command["fsm_tcp_goal_pose_w"] = target
                                    current_command["retreat_pose_reference"] = reference
                                    if (stage_id == "release" and current_command.get("release_phase") == "separating"
                                            and "release_separating" in task_settings.get(
                                                "retreat_pose_control_overrides", {})):
                                        # Declared only (runtime_v26 C5): the separating subphase's own settings.
                                        current_command["release_separating_pose_control"] = {
                                            "schema_version": "igris_c_release_separating_pose_control_v1",
                                            "applied": True, "release_phase": "separating",
                                            "settings": dict(retreat_settings),
                                            "override": dict(task_settings["retreat_pose_control_overrides"][
                                                "release_separating"]),
                                            "opening_and_clearing_keep_retreat_pose_control": True}
                                    target = reference["tcp_target_pose_w"]
                                    current_command["tcp_target_pose_w"] = target
                                if (handover_settings is not None and "approach_pose_control" in handover_settings
                                        and stage_id in ("handover_pregrasp", "handover_approach")):
                                    # Opt-in handover: a bounded empty-hand reference for the receiving arm. The
                                    # machine's goal follows the measured ring; when it moves, the reference restarts
                                    # from its last target toward the new goal (object-pose feedback, no jump).
                                    goal = pose_to_matrix(target)
                                    if handover_controller is None or handover_controller_stage != stage_id:
                                        handover_controller = TcpPoseController(
                                            observation["tcp_poses_w"][arm], target, **handover_settings["approach_pose_control"])
                                        handover_controller_stage, handover_goal = stage_id, goal
                                    elif (np.linalg.norm(goal[:3, 3] - handover_goal[:3, 3]) > 1e-3
                                          or np.linalg.norm(rotation_error(goal[:3, :3], handover_goal[:3, :3])) > 1e-2):
                                        handover_controller = TcpPoseController(
                                            handover_reference["tcp_target_pose_w"], target,
                                            **handover_settings["approach_pose_control"])
                                        handover_goal = goal
                                    handover_reference = handover_controller.update(observation["tcp_poses_w"][arm], dt)
                                    current_command["fsm_tcp_goal_pose_w"] = target
                                    current_command["handover_pose_reference"] = handover_reference
                                    target = handover_reference["tcp_target_pose_w"]
                                    current_command["tcp_target_pose_w"] = target
                                if motion == "plan" and args.approach == "curobo":
                                    if plan_key != stage_id:
                                        plan_names, plan_frames, _ = planner.plan(arm, target, directory)
                                        plan_index, plan_key = 0, stage_id
                                        plan_base = pose_to_matrix(observation["base_pose_w"])
                                    actual_base = pose_to_matrix(observation["base_pose_w"])
                                    if (np.linalg.norm(actual_base[:3, 3] - plan_base[:3, 3]) > .08
                                            or np.linalg.norm(rotation_error(actual_base[:3, :3], plan_base[:3, :3])) > .12):
                                        raise RuntimeError("base_drift_invalidates_approach_plan")
                                    active = servo.model.arm_joint_names(arm)
                                    for name in active:
                                        source.targets[name] = float(plan_frames[min(plan_index, len(plan_frames)-1), plan_names.index(name)])
                                    plan_index += 1
                                else:
                                    # Slower physical manipulation reduces inertial
                                    # loads on the small jaw contact patch. This
                                    # changes commands, never grasp predicates.
                                    speed_scale = config["tasks"][task_id].get(
                                        "contact_servo_speed_scale", 1.0) if stage_id in {
                                            "close", "lift", "recover_carry", "transfer", "place", "open_hinge"} else 1.0
                                    if handover_settings is not None and stage_id in HANDOVER_STAGE_SET:
                                        speed_scale = handover_settings.get("servo_speed_scale", 1.)
                                    feedforward = None
                                    if (stage_id in task_settings.get("base_motion_feedforward_stages", ["navigate_target", "recover_base"])
                                            and navigation.get("compensate_base_motion", False)):
                                        from .carry_feedforward import RelativePoseFeedforward
                                        feedforward_segment = (stage_id, feedforward_index(current_command))
                                        if carry_feedforward_stage != feedforward_segment:
                                            carry_feedforward = RelativePoseFeedforward()
                                            carry_feedforward_stage = feedforward_segment
                                        feedforward = carry_feedforward.update(observation["base_pose_w"], target, dt)
                                    if stage_id in task_settings.get("torso_motion_feedforward_stages", ()):
                                        # Opt-in: the measured torso (base_link) moves under the arm during a
                                        # waist sweep; its derivative relative to the target removes the
                                        # proportional servo lag. Restarts with every object segment.
                                        from .carry_feedforward import RelativePoseFeedforward
                                        torso_segment = (stage_id, feedforward_index(current_command))
                                        if torso_feedforward_stage != torso_segment:
                                            torso_feedforward = RelativePoseFeedforward()
                                            torso_feedforward_stage = torso_segment
                                        feedforward = torso_feedforward.update(observation["base_pose_w"], target, dt)
                                        current_command["torso_motion_feedforward_twist_base"] = feedforward.tolist()
                                    if retreat_feedforward is not None:
                                        empty_hand_twist = retreat_feedforward.update(
                                            stage_id, current_command, observation["base_pose_w"], target, dt)
                                        if empty_hand_twist is not None:
                                            feedforward = empty_hand_twist
                                            current_command["retreat_motion_feedforward_twist_base"] = feedforward.tolist()
                                    if upper_body_servo is not None and stage_id in upper_body_settings["stages"]:
                                        # The measured floating pelvis is a boundary condition,
                                        # never a root command. This controller alone owns the
                                        # waist and active arm in its opted-in stages. Lift
                                        # uses the existing measured-object reference above.
                                        if posture is not None or arm_posture_target is not None:
                                            raise RuntimeError("upper_body_ik_joint_ownership_conflict")
                                        upper_preference = {}
                                        if (upper_posture_guide is not None
                                                and stage_id in upper_posture_guide.settings["stages"]
                                                and (stage_id in {"pregrasp", "approach"}
                                                     or "object_pose_reference" in current_command)):
                                            constant_preference = stage_id in {"pregrasp", "approach"}
                                            progress = (0. if constant_preference else
                                                current_command["object_pose_reference"]["progress"])
                                            index = (0 if constant_preference else
                                                current_command.get("object_waypoint_index", 0))
                                            entry_posture = constant_preference and upper_posture_guide.from_entry(stage_id)
                                            if entry_posture and upper_posture_guide.entry_posture(stage_id) is None:
                                                # Opt-in from_entry (runtime_v26): the measured waist3+arm7 of this
                                                # stage's first step, inside the stage servo's command limits.
                                                upper_posture_guide.latch_entry_posture(
                                                    stage_id, observation["joint_positions"],
                                                    upper_body_stage_servos.get(stage_id, upper_body_servo).limits,
                                                    observation["time_s"])
                                            blended_reference = (None if constant_preference else
                                                current_command["object_pose_reference"].get("reference_continuity"))
                                            if blended_reference is not None:
                                                # Opt-in blended pass-through: the preference follows the REFERENCE's
                                                # segment(s), which may still finish the waypoint the machine passed.
                                                index = blended_reference["active_waypoint_index"]
                                                next_index = blended_reference["next_waypoint_index"]
                                                preferred = (upper_posture_guide.sample(stage_id, index, progress)
                                                    if next_index is None else upper_posture_guide.sample_blended(
                                                        stage_id, index, progress, blended_reference["next_progress"]))
                                            else:
                                                preferred = upper_posture_guide.sample(stage_id, index, progress)
                                            upper_preference = {"posture_target": preferred,
                                                "posture_gain": upper_posture_guide.settings["gain_s_inv"]}
                                            current_command["upper_body_posture_reference"] = {
                                                "joint_positions": preferred, "waypoint_index": index,
                                                "segment_progress": progress,
                                                "gain_s_inv": upper_preference["posture_gain"],
                                                "independent_joint_commands": False}
                                            if constant_preference:
                                                current_command["upper_body_posture_reference"]["reference_kind"] = "constant_stage_posture"
                                            if entry_posture:
                                                current_command["upper_body_posture_reference"].update(
                                                    reference_kind="entry_latched_stage_posture",
                                                    entry_latch=upper_posture_guide.entry_posture(stage_id))
                                            if blended_reference is not None:
                                                current_command["upper_body_posture_reference"].update(
                                                    reference_kind=("blended_reference_segments" if next_index is not None
                                                                    else "reference_segment"),
                                                    fsm_waypoint_index=current_command.get("object_waypoint_index", 0),
                                                    next_waypoint_index=next_index,
                                                    next_segment_progress=blended_reference["next_progress"])
                                        if upper_body_settings.get("pelvis_motion_compensation", False):
                                            reset = pelvis_motion_compensation is None
                                            if reset:
                                                from .pelvis_motion import PelvisMotionCompensation
                                                pelvis_motion_compensation = PelvisMotionCompensation()
                                            twist = pelvis_motion_compensation.update(
                                                observation["pelvis_pose_w"], observation["tcp_poses_w"][arm], dt)
                                            upper_preference["feedforward_twist_world"] = twist
                                            current_command["pelvis_motion_compensation"] = {
                                                "coordinate_frame": "world", "stage_id": stage_id,
                                                "measurement_time_s": observation["time_s"], "dt_s": dt,
                                                "history_reset": reset, "twist_world": twist.tolist(),
                                                "source": "consecutive_measured_pelvis_poses_at_current_tcp",
                                                "target_motion_included": False,
                                                "pelvis_pose_written": False,
                                            }
                                        if upper_target_feedforward is not None:
                                            target_twist = upper_target_feedforward.update(
                                                stage_id, current_command, arm, target, dt)
                                            if target_twist is not None:
                                                pelvis_twist = upper_preference.get("feedforward_twist_world")
                                                upper_preference["feedforward_twist_world"] = (
                                                    target_twist if pelvis_twist is None
                                                    else pelvis_twist + target_twist)
                                                current_command["upper_body_target_motion_feedforward"] = {
                                                    **upper_target_feedforward.last_result,
                                                    "measurement_time_s": observation["time_s"],
                                                }
                                        selected_upper_servo = upper_body_stage_servos.get(stage_id, upper_body_servo)
                                        from .upper_body_control import select_transfer_servo
                                        selected_upper_servo, transfer_early_released, early_evidence = select_transfer_servo(
                                            selected_upper_servo, current_command.get("object_waypoint_index"),
                                            transfer_early_released)
                                        if early_evidence is not None:
                                            current_command["upper_body_transfer_early_limits"] = early_evidence
                                        if stage_id in upper_body_settings.get("stage_waist_hold", ()):
                                            # Opt-in (runtime_v26): the waist command applied when this stage started
                                            # is held exactly (zero-width IK bounds) for the whole stage.
                                            if stage_waist_hold is None or stage_waist_hold["stage_id"] != stage_id:
                                                from .upper_body_control import latch_stage_waist_hold
                                                stage_waist_hold = latch_stage_waist_hold(
                                                    stage_id, source.targets, selected_upper_servo, observation["time_s"])
                                            upper_preference["waist_hold"] = stage_waist_hold["latched_waist_command_rad"]
                                            current_command["upper_body_stage_waist_hold"] = deepcopy(stage_waist_hold)
                                        targets, servo_diagnostics = selected_upper_servo.solve_step(
                                            observation["joint_positions"], observation["pelvis_pose_w"],
                                            target, arm, dt, source.targets, speed_scale=speed_scale,
                                            tracking_error_limits=task_settings.get("arm_tracking_error_limits_rad"),
                                            **upper_preference)
                                        current_command["upper_body_control"] = servo_diagnostics
                                    else:
                                        targets, servo_diagnostics = servo.solve_step(observation["joint_positions"], observation["base_pose_w"],
                                                                 target, arm, dt, source.targets,
                                                                 speed_scale=speed_scale,
                                                                 tracking_error_limits=arm_tracking_limits(arm),
                                                                 nullspace_gain=task_settings.get("arm_nullspace_gain_s_inv", 0.),
                                                                 posture_target=arm_posture_target, posture_gain=arm_posture_gain,
                                                                 solver=(task_settings.get("arm_solver", "dls")
                                                                    if stage_id in task_settings.get("arm_solver_stages", machine.stages) else "dls"),
                                                                 feedforward_twist_base=feedforward)
                                    source.targets.update(targets)
                                    schedule = transfer_schedule
                                    if schedule is not None and stage_id == "transfer":
                                        # Inactive arm: entry command -> joint_positions over the declared
                                        # waypoint window, rate limited; holds the goal afterwards.
                                        from .upper_body_control import apply_transfer_secondary_arm_schedule
                                        secondary_schedule_state, schedule_evidence = apply_transfer_secondary_arm_schedule(
                                            schedule, secondary_schedule_state, source.targets,
                                            current_command.get("object_waypoint_index"), dt, owned_joints=tuple(targets))
                                        current_command["upper_body_transfer_secondary_arm_schedule"] = schedule_evidence
                                    current_command["servo_diagnostics"] = servo_diagnostics
                                    if arm_posture_target is not None:
                                        current_command["arm_posture_target"] = arm_posture_target
                            secondary_arm = current_command.get("secondary_arm")
                            secondary_target = current_command.get("secondary_tcp_target_pose_w")
                            if handover_settings is not None and secondary_arm is not None and secondary_target is not None:
                                # Opt-in handover: the second arm is servoed in the same control step. The giving
                                # hand holds its torso-frame pose (the ring stays still) while the receiving hand
                                # regrasps, then follows a bounded reference for its separation and each retreat
                                # waypoint. Commands only; no state is written.
                                evidence = current_command.get("handover") or {}
                                segment = (stage_id, evidence.get("release_phase"), evidence.get("retreat_waypoint_index"))
                                bounded = (stage_id == "handover_retreat"
                                           or (stage_id == "handover_release" and evidence.get("release_phase") == "separating"))
                                if bounded:
                                    if secondary_controller is None or secondary_controller_stage != segment:
                                        secondary_settings = handover_settings.get("retreat_pose_control", task_settings.get(
                                            "retreat_pose_control", {"linear_speed": .002, "linear_acceleration": .004,
                                                                     "angular_speed": .12, "angular_acceleration": .3,
                                                                     "position_lead": .004, "rotation_lead": .02}))
                                        secondary_controller = TcpPoseController(
                                            observation["tcp_poses_w"][secondary_arm], secondary_target, **secondary_settings)
                                        secondary_controller_stage = segment
                                    reference = secondary_controller.update(observation["tcp_poses_w"][secondary_arm], dt)
                                    current_command["secondary_fsm_tcp_goal_pose_w"] = secondary_target
                                    current_command["secondary_pose_reference"] = reference
                                    secondary_target = reference["tcp_target_pose_w"]
                                    current_command["secondary_tcp_target_pose_w"] = secondary_target
                                targets, diagnostics = servo.solve_step(
                                    observation["joint_positions"], observation["base_pose_w"], secondary_target,
                                    secondary_arm, dt, source.targets, speed_scale=handover_settings.get("servo_speed_scale", 1.),
                                    tracking_error_limits=arm_tracking_limits(secondary_arm),
                                    solver=(task_settings.get("arm_solver", "dls")
                                            if stage_id in task_settings.get("arm_solver_stages", machine.stages) else "dls"))
                                source.targets.update(targets)
                                current_command["secondary_servo_diagnostics"] = diagnostics
                            if auxiliary_support is not None:
                                nominal_reference = current_command.get("object_pose_reference", {}).get("object_reference_pose_w")
                                secondary_target = auxiliary_support.target(observation, dt,
                                    nominal_object_pose_w=nominal_reference)
                                if secondary_target is not None:
                                    targets, diagnostics = servo.solve_step(observation["joint_positions"],
                                        observation["base_pose_w"], secondary_target, auxiliary_support.arm, dt,
                                        source.targets, speed_scale=.5,
                                        tracking_error_limits={n: value for n, value in task_settings.get(
                                            "arm_tracking_error_limits_rad", {}).items()
                                            if n in servo.model.arm_joint_names(auxiliary_support.arm)})
                                    source.targets.update(targets)
                                    jaw = auxiliary_support.joint_name
                                    source.targets[jaw] += float(np.clip(
                                        auxiliary_support.settings["joint_position_rad"] - source.targets[jaw], -.3 * dt, .3 * dt))
                                    current_command["auxiliary_support"].update(
                                        tcp_target_pose_w=secondary_target, servo_diagnostics=diagnostics,
                                        joint_command_rad=source.targets[jaw],
                                        nominal_object_reference_pose_w=nominal_reference)
                            if progress_watchdog is not None:
                                metric, goal, actual = "tcp", None, observation["tcp_poses_w"][arm]
                                if stage_id in {"navigate_source", "navigate_target", "recover_base"}:
                                    metric, goal, actual = "base", current_command.get("base_goal_pose_w"), observation["base_pose_w"]
                                elif stage_id == "open_hinge":
                                    metric, goal, actual = "object", matrix_to_pose(machine._target_object), observation["object_poses_w"][machine.object_path]
                                elif stage_id in {"transfer", "place"}:
                                    goal = _tracking_progress_goal(machine, observation, arm)
                                elif stage_id in {"lift", "recover_carry"}:
                                    metric, actual = "object", observation["object_poses_w"][machine.object_path]
                                    goal = goal_object_from_tcp(matrix_to_pose(machine._tcp_target), matrix_to_pose(machine._control_object_to_tcp))
                                elif stage_id == "pregrasp" and "pregrasp_pose_reference" in current_command:
                                    goal = current_command["pregrasp_pose_reference"]["waypoint_goal_pose_w"]
                                elif machine._tcp_target is not None:
                                    goal = matrix_to_pose(machine._tcp_target)
                                if goal is None:
                                    errors = {}
                                elif metric == "object" and object_symmetry_axis is not None:
                                    # Declared symmetry: the object's spin about its axis is not a goal error.
                                    position_error, rotation_error_rad, _ = symmetric_pose_errors(
                                        pose_to_matrix(actual), pose_to_matrix(goal), object_symmetry_axis)
                                    errors = {"position_error_m": position_error, "rotation_error_rad": rotation_error_rad}
                                else:
                                    errors = pose_goal_errors(actual, goal, metric=metric)
                                reference = current_command.get("object_pose_reference", {})
                                if stage_id == "pregrasp" and "pregrasp_pose_reference" in current_command:
                                    reference = current_command["pregrasp_pose_reference"].get("reference") or {}
                                servo_errors = current_command.get("servo_diagnostics", {})
                                thresholds = config["thresholds"]
                                if metric == "base":
                                    station = "recovery" if stage_id == "recover_base" else stage_id.removeprefix("navigate_")
                                    position_tolerance, rotation_tolerance = machine.navigation_tolerances(station)
                                else:
                                    position_tolerance = thresholds[f"{metric}_position_m"]
                                    rotation_tolerance = thresholds[f"{metric}_rotation_rad"]
                                reached = bool(errors and errors["position_error_m"] <= position_tolerance
                                    and errors["rotation_error_rad"] <= rotation_tolerance)
                                if posture is not None:
                                    reached = reached and all(abs(observation["joint_positions"][n] - q) <= posture.get("tolerance_rad", .03)
                                                               for n, q in posture["joint_positions"].items())
                                if current_command.get("base_paused_for_posture", False):
                                    metric = "posture"
                                    errors = posture_goal_errors(observation["joint_positions"], posture["joint_positions"])
                                    reached = posture_reached
                                progress = progress_watchdog.update(
                                    observation["time_s"], stage_id, metric=metric, **errors,
                                    waypoint_index=motion_waypoint_index(stage_id, current_command),
                                    goal_reached=reached, enabled=goal is not None,
                                    diagnostics={"pause_reason": "body_posture" if metric == "posture" else reference.get("reason"),
                                        "reference_progress": reference.get("progress"),
                                        "servo_position_error_m": servo_errors.get("position_error_m"),
                                        "servo_rotation_error_rad": servo_errors.get("rotation_error_rad")})
                                current_command["motion_progress"] = progress
                                if progress["stalled"] or not progress["valid"]:
                                    trace.write(json.dumps({"observation": observation, "command": current_command,
                                        "commanded_upper_joint_positions": dict(source.targets),
                                        "applied_policy_action_previous_step": last_action.tolist()}, allow_nan=False) + "\n")
                                    raise RuntimeError(progress["reason"])
                            hand = current_command.get("hand_command", "hold")
                            supported_unloading = (hand in ("close", "hold")
                                and current_command["stage_id"] == "place"
                                and current_command.get("placement_support_hold_active", False)
                                and "support_gripper_preload_rad" in task_settings)
                            if supported_unloading or (support_unloading_was_active and hand == "hold"):
                                hand = "close"
                            support_unloading_was_active = supported_unloading
                            relief = None
                            if grip_relief is not None:
                                relief_open, relief_closed = handover_jaw_positions(task_settings, arm)
                                relief = grip_relief.update(current_command, observation, source.grippers[arm],
                                    open_position_rad=relief_open, closed_position_rad=relief_closed, dt=dt)
                                current_command["supported_grip_relief"] = relief
                                if relief["controlled"]:
                                    hand = "close"
                            # One jaw per arm: the opt-in handover publishes hand_commands for both arms (each
                            # with its own declared aperture); otherwise only the task arm's hand_command applies.
                            hand_commands = current_command.get("hand_commands")
                            jaw_commands = {**hand_commands, arm: hand} if isinstance(hand_commands, dict) else {arm: hand}
                            for jaw_arm, jaw_command in jaw_commands.items():
                                if jaw_command not in ("open", "close"):
                                    continue
                                open_goal, closed_goal = handover_jaw_positions(task_settings, jaw_arm)
                                if jaw_arm == arm and relief is not None and relief["controlled"]:
                                    name = source.grippers[arm]["joint_name"]
                                    if abs(source.targets[name] - relief["previous_applied_target_rad"]) > 1e-9:
                                        raise RuntimeError("supported_grip_relief previous APPLIED target mismatch")
                                    closed_goal = relief["target_rad"]
                                if jaw_arm == arm and supported_unloading:
                                    from .task_control import supported_gripper_preload
                                    grip = source.grippers[arm]
                                    closed_goal = supported_gripper_preload(grip,
                                        observation["joint_positions"][grip["joint_name"]],
                                        task_settings["support_gripper_preload_rad"],
                                        open_position_rad=open_goal,
                                        closed_position_rad=closed_goal)
                                    current_command["supported_gripper_target_rad"] = closed_goal
                                source.apply_hand(jaw_arm, jaw_command == "close", dt,
                                    open_position_rad=open_goal, closed_position_rad=closed_goal)
                                if jaw_arm == arm and relief is not None and relief["controlled"]:
                                    if abs(source.targets[source.grippers[arm]["joint_name"]] - relief["target_rad"]) > 1e-9:
                                        raise RuntimeError("supported_grip_relief emitted target mismatch")
                            with throughput.measure("trace_serialize_write"):
                                trace.write(json.dumps({"observation": observation, "command": current_command,
                                                        "commanded_upper_joint_positions": dict(source.targets),
                                                        "applied_policy_action_previous_step": last_action.tolist()},
                                                       ensure_ascii=False, allow_nan=False) + "\n")
                            step()
                        else:
                            reason = "task_wall_simulation_timeout"
                            machine.fail(reason)
                    except Exception as exc:
                        reason = f"{type(exc).__name__}: {exc}"
                        machine.fail(reason)
                    finally:
                        capture["active"] = False
                        source.base_velocity[:] = 0
                result = {"episode_id": episode_id, "task_id": dataset_task_id(task_id), "success": success,
                          "episode_scope": collection_scope, "full_task_completed": success and not primitive,
                          "auxiliary_support_diagnostic": (dict(auxiliary_support._last)
                              if auxiliary_support is not None else None),
                          "primitive_metrics": {"transport_projection_m": machine.transport_m,
                                                "lift_m": machine.lift_m,
                                                "required_hold_seconds": machine.hold_seconds} if primitive else None,
                          "reason": reason, "last_stage": last_stage,
                          "duration_s": tick * dt - elapsed_start, "seed": args.seed + cycle,
                          "attempt_duration_s": tick * dt - elapsed_start,
                          "recorded_duration_s": (capture["last_sim_time_s"] - capture["first_sim_time_s"]
                              if capture["frame_count"] else 0.),
                          "recorded_frame_count": capture["frame_count"],
                          "calibration_status": report["calibration_status"],
                          "execution_mode": report["execution_mode"],
                          "initialization": initialization,
                          "collision_free_execution_verified": False,
                          "simulated_wash_seconds": args.simulate_wash_seconds}
                if layout_contract is not None:
                    result["single_arm_layout"] = deepcopy(initialization["single_arm_layout"])
                if stationary:
                    result.update(legacy_task_id=task_id, legacy_full_task_completed=False,
                                  stationary_task=stationary_spec,
                                  stationary_metrics=machine._command().get("stationary_metrics"),
                                  controller_contract=report["controller_contract"])
                    if handover_settings is not None:
                        # Opt-in handover: the arm-switch provenance (holding arm, receiving capture, events).
                        result["handover"] = machine._command().get("handover")
                if diagnostic_transition is not None:
                    result["diagnostic_stop_transition"] = deepcopy(diagnostic_transition)
                if prepared_recording is not None:
                    result["recording_boundary"] = prepared_recording.metadata
                    if success and not writer_started:
                        success = False
                        reason = "recording_boundary_never_started"
                        result.update(success=False, full_task_completed=False, reason=reason)
                _json(directory / "result.json", result)
                if writer:
                    if not writer_started:
                        # Preserve preparation failures without creating a
                        # successful empty episode or starting camera capture.
                        writer.begin_episode(episode_id, dataset_task_number(task_id), episode_title(task_id),
                            provenance={**report, "cycle": cycle, "kind": "original",
                                "initialization": initialization, "episode_scope": collection_scope,
                                "recording_boundary": prepared_recording.metadata if prepared_recording else None})
                    with throughput.measure("writer_finish_drain"):
                        writer.finish_episode(success=success, metrics=result, reason=reason)
                report["episodes"].append(result)
                _json(args.output / "run.json", _with_ik_backend(report))
                if not success:
                    print(f"[NEPES] Task failed: {reason}.", flush=True)
                    if args.sequence:
                        print("[NEPES] Remaining dependent tasks skipped.", flush=True)
                        break
                    live.end_episode()
                    continue
                if not primitive and not stationary:
                    completed.append(task_id)
                if not args.sequence:
                    live.end_episode()
            live.end_episode()
        report["complete"] = True
        report["all_episodes_succeeded"] = len(report["episodes"]) == args.cycles * len(selected) and all(e["success"] for e in report["episodes"])
        report["all_tasks_succeeded"] = report["all_episodes_succeeded"] and not primitive
        report["episode_scope"] = collection_scope
        _json(args.output / "run.json", _with_ik_backend(report))
        if writer:
            writer.finalize()
        return 0 if report["all_episodes_succeeded"] else 2
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        try:
            _json(args.output / "run.json", _with_ik_backend(report))
        except Exception as report_error:
            # A report-writing error must not replace the original task fault.
            print(f"[NEPES] Could not save task error report: {type(report_error).__name__}: {report_error}", flush=True)
        raise
    finally:
        import sys
        if "capture" in locals():
            capture["active"] = False
        try:
            render_summary = render_control.summary()
        except Exception as profile_error:
            render_summary = {"error": f"{type(profile_error).__name__}: {profile_error}"}
        try:
            _finalize_task_run(
                report=report, report_path=args.output / "run.json", original_error=sys.exc_info()[1],
                writer=writer, camera=camera, live=live, env=env, raw=raw,
                foot_contacts=locals().get("foot_contacts"),
                original_scene_update=original_scene_update,
                original_scene_update_had_override=original_scene_update_had_override,
                robot=locals().get("robot"), render_control=render_control)
        finally:
            if throughput.enabled:
                try:
                    _json(args.output / "throughput.json", {
                        "schema": "nepes_runtime_throughput_v1", **throughput.summary(),
                        "render_control": render_summary,
                        "capture_options": report["capture_options"],
                        "image_writer": getattr(writer, "image_writer_metrics", None),
                        "recorded_frame_count": sum(e.get("recorded_frame_count", 0)
                                                    for e in report["episodes"]),
                    })
                except Exception as profile_error:
                    print(f"[NEPES] Could not save throughput profile: {profile_error}", flush=True)


def _finalize_task_run(*, report, report_path, original_error=None, writer=None,
                       camera=None, live=None, env=None, raw=None, foot_contacts=None,
                       original_scene_update=None, original_scene_update_had_override=True, robot=None,
                       render_control=None):
    """Release every runtime owner, preserving an already-propagating task error.

    This function imports no simulation runtime and can be fault-injection
    tested with CPU-only owners. A cleanup error raises only after all owners
    have been visited and only if no original task exception is propagating.
    """
    from .shutdown import release_articulation_views
    errors = []

    def attempt(resource, operation, callback):
        try:
            return callback()
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            errors.append({"resource": resource, "operation": operation, "error": message})
            print(f"[NEPES] Cleanup {resource}.{operation} failed: {message}", flush=True)
            return None

    print("[NEPES] Releasing task recording and physics resources.", flush=True)
    if render_control is not None:
        attempt("render_control", "restore", render_control.restore)
    if original_scene_update is not None and raw is not None:
        def restore_scene_update():
            if original_scene_update_had_override:
                raw.scene.update = original_scene_update
            elif "update" in vars(raw.scene):
                # Restore class-method lookup. Storing its bound method on the
                # instance creates scene -> bound_method -> scene, delaying
                # native resource release beyond env.close until cyclic GC.
                del raw.scene.update
        attempt("scene", "restore_update", restore_scene_update)
    if writer is not None:
        attempt("writer", "finalize", writer.finalize)
    if camera is not None:
        print("[NEPES] Closing cameras.", flush=True)
        camera_errors = attempt("camera", "close", camera.close)
        if camera_errors:
            report["camera_cleanup_errors"] = list(camera_errors)
            errors.extend({"resource": "camera", "operation": "close", "error": str(message)}
                          for message in camera_errors)
    if foot_contacts is not None:
        attempt("foot_contacts", "close", foot_contacts.close)
    if live is not None:
        attempt("live_scene", "end_episode", live.end_episode)
        if hasattr(live, "close"):
            attempt("live_scene", "close", live.close)
    if env is not None:
        print("[NEPES] Closing environment.", flush=True)
        attempt("environment", "close", env.close)
    if robot is not None:
        print("[NEPES] Releasing articulation tensor views.", flush=True)
        attempt("articulation", "release_views", lambda: release_articulation_views(robot))
    from .upper_body_ik_backend import shutdown_worker
    attempt("upper_body_ik_worker", "shutdown", shutdown_worker)
    if "upper_body_ik_backend" in report:
        # Final IK record after shutdown: closed with the worker exit code, or the failure that ended the attempt.
        _with_ik_backend(report)
        if not errors:
            attempt("report", "write_ik_backend", lambda: _json(report_path, report))
    print("[NEPES] Task resources released.", flush=True)
    if errors:
        report["cleanup_errors"] = errors
        if original_error is not None:
            report.setdefault("error", f"{type(original_error).__name__}: {original_error}")
        attempt("report", "write_cleanup_errors", lambda: _json(report_path, report))
        if original_error is None:
            raise RuntimeError("Task resource cleanup failed: " + "; ".join(
                f"{entry['resource']}.{entry['operation']}: {entry['error']}" for entry in errors))
    return errors
