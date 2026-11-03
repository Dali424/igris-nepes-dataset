#!/usr/bin/env python3
"""Inspect, probe, or physically execute the five NEPES tasks with HugWBC."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT.parents[1] / "assets"
sys.path[:0] = [str(ROOT / "source/robros_motion_planning"), str(ROOT / "source/robros_lab")]


def main(argv=None):
    from robros_motion_planning.latest_robot import FRICTION_COMBINE_MODES, GRIPPER_VARIANTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inspect", "probe", "execute"), default="inspect")
    parser.add_argument("--config", type=Path,
                        help="Task candidates by default; grasp_carry selects its separate calibrated source/transport config")
    parser.add_argument("--tasks", default="all", help="all or comma-separated task IDs / numbers 1..5")
    parser.add_argument("--sequence", action="store_true", help="Chain tasks in one world; default prepares each task independently")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/nepes/run")
    parser.add_argument("--scene_usd", type=Path, default=ASSETS / "workcell/layout.usda")
    parser.add_argument("--robot_model_dir", type=Path, default=ASSETS / "robot",
                        help="Latest 1-DOF two-finger robot assets, including authored cameras")
    parser.add_argument("--gripper_variant", choices=GRIPPER_VARIANTS, default=None,
                        help="Jaw model of the derived robot. Default: the profile's gripper_variant, else "
                             "authored_v0; a profile that declares one cannot be overridden")
    parser.add_argument("--dev_pad_friction", nargs=2, type=float, metavar=("STATIC", "DYNAMIC"), default=None,
                        help="DEVELOPMENT sensitivity only (pad variants): pad static/dynamic friction. "
                             "Fingerprinted and recorded; forbids --record; never admissible for collection")
    parser.add_argument("--dev_pad_friction_combine", choices=FRICTION_COMBINE_MODES, default=None,
                        help="DEVELOPMENT sensitivity only (pad variants): pad friction combine mode")
    parser.add_argument("--checkpoint", type=Path, default=ASSETS / "policies/hugwbc.pt")
    parser.add_argument("--planner_python", type=Path, default=Path(sys.executable))
    parser.add_argument("--approach", choices=("curobo", "servo"), default="curobo",
                        help="Free-space pregrasp planner; servo is an explicitly unvalidated diagnostic mode")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_task_seconds", type=float, default=180.)
    parser.add_argument("--settle_seconds", type=float, default=3.)
    parser.add_argument("--simulate_wash_seconds", type=float, default=None,
                        help="Emit a labelled simulated wash_complete event after task 3; no cleaning physics")
    parser.add_argument("--events_file", type=Path, help='External JSON {"events": ["wash_complete"]}; re-read each step')
    parser.add_argument("--record", action="store_true", help="Record synchronized RGB/state/action staging for LeRobot export")
    parser.add_argument("--capture-render-mode", choices=("legacy", "capture_only"), default="legacy",
                        help="capture_only: headless RGB renders only at capture deadlines; physics/control rates stay unchanged")
    parser.add_argument("--image-writer-workers", type=int, default=0,
                        help="Bounded asynchronous lossless PNG writer threads; 0 preserves synchronous writing")
    parser.add_argument("--image-writer-queue", type=int, default=8,
                        help="Maximum queued/in-progress image frames with asynchronous writing")
    parser.add_argument("--profile-throughput", action="store_true",
                        help="Write wall-clock capture/render/writer measurements to throughput.json")
    parser.add_argument("--collection_scope", choices=("task", "grasp_carry", "stationary_task"), default="task",
                        help="grasp_carry collects a separately labelled lift/short transport/hold primitive; not full placement")
    parser.add_argument("--carry_offset_w", nargs=2, type=float, default=[.25, 0.], metavar=("X", "Y"),
                        help="World XY transport goal offset in meters for grasp_carry")
    parser.add_argument("--keep_failed_frames", action="store_true", help="Keep failed camera episodes for calibration; they are never exported")
    parser.add_argument("--no_cover_parking", action="store_true", help="Do not add the explicitly labelled simulation lid parking support")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--real_time", action="store_true")
    parser.add_argument("--no_head_gaze", action="store_true", help="Keep reset neck angles instead of aiming at the source grasp during independent initialization")
    parser.add_argument("--initialize_arm_at_pregrasp", action="store_true",
                        help="Initialize the active arm above the source at the independent episode boundary; physical approach/grasp/transport still run")
    diagnostics = parser.add_mutually_exclusive_group()
    diagnostics.add_argument("--diagnostic_stop_after", metavar="STAGE",
                        help="Stationary calibration only: fail the episode as diagnostic_stop_after:STAGE as soon as the "
                             "task machine leaves STAGE; never a success and never admitted by collection")
    diagnostics.add_argument("--diagnostic_stop_after_waypoint", metavar="STAGE:INDEX",
                        help="Independent stationary calibration only: stop after measured arrival at a zero-based "
                             "non-final lift/transfer waypoint, before applying the next waypoint; never success")
    args, remaining = parser.parse_known_args(argv)
    if args.config is None:
        filename = ("nepes_grasp_carry_twofinger.json" if args.collection_scope == "grasp_carry" else
                    "nepes_tasks_twofinger_lift_off_candidates.json")
        args.config = ROOT / "source/robros_motion_planning/config" / filename
    if args.cycles < 1 or args.settle_seconds < 0 or args.max_task_seconds <= 0:
        parser.error("cycles and max_task_seconds must be positive; settle_seconds must be nonnegative")
    if args.simulate_wash_seconds is not None and args.simulate_wash_seconds < 0:
        parser.error("simulate_wash_seconds must be nonnegative")
    if args.record and args.mode != "execute":
        parser.error("--record is available in execute mode; probe only measures robot/object state")
    if args.capture_render_mode == "capture_only" and not (args.record and args.headless):
        parser.error("--capture-render-mode capture_only requires --record --headless")
    if args.image_writer_workers < 0 or args.image_writer_queue < 1:
        parser.error("image-writer-workers must be nonnegative and image-writer-queue must be positive")
    if args.image_writer_workers and not args.record:
        parser.error("--image-writer-workers requires --record")
    if args.initialize_arm_at_pregrasp and (args.sequence or args.mode != "execute"):
        parser.error("--initialize_arm_at_pregrasp requires independent execute mode")
    if args.diagnostic_stop_after is not None and (args.mode != "execute" or args.collection_scope != "stationary_task"):
        parser.error("--diagnostic_stop_after requires stationary_task scope in execute mode")
    if args.diagnostic_stop_after_waypoint is not None and (
            args.mode != "execute" or args.collection_scope != "stationary_task" or args.sequence):
        parser.error("--diagnostic_stop_after_waypoint requires independent stationary_task scope in execute mode")
    from robros_motion_planning.nepes_tasks import load_task_config, NepesTaskMachine
    config = load_task_config(args.config)
    from robros_motion_planning.single_arm_layout import resolve_layout_scene
    try:
        args.scene_usd, layout_contract = resolve_layout_scene(config, args.scene_usd)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.collection_scope not in config.get("allowed_collection_scopes", ["task", "grasp_carry"]):
        parser.error("This configuration is restricted to " + ", ".join(config["allowed_collection_scopes"]))
    from robros_motion_planning.latest_robot import resolve_gripper_selection
    if args.record and (args.dev_pad_friction is not None or args.dev_pad_friction_combine is not None):
        parser.error("--dev_pad_friction/--dev_pad_friction_combine are sensitivity diagnostics and cannot be recorded")
    try:
        args.gripper_selection = resolve_gripper_selection(
            config, args.gripper_variant, dev_pad_friction=args.dev_pad_friction,
            dev_pad_friction_combine=args.dev_pad_friction_combine)
    except ValueError as exc:
        parser.error(str(exc))
    all_ids = [task["id"] for task in config["tasks"]] if isinstance(config["tasks"], list) else list(config["tasks"])
    selected = all_ids if args.tasks == "all" else [all_ids[int(v) - 1] if v in ("1", "2", "3", "4", "5") else v
                                                  for v in args.tasks.split(",")]
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(all_ids):
        parser.error("tasks must be unique task IDs or numbers 1..5")
    if set(selected) - set(config.get("allowed_task_ids", all_ids)):
        parser.error("This configuration supports only: " + ", ".join(config["allowed_task_ids"]))
    if layout_contract is not None and (args.collection_scope != "stationary_task" or args.sequence
                                       or selected != ["crs_load_jig"]):
        parser.error("single_arm_layout requires independent stationary crs_load_jig episodes")
    from robros_motion_planning.upright_runtime import (
        validate_diagnostic_waypoint_stop, validate_required_diagnostic_waypoint_stop)
    try:
        for task_id in selected:
            validate_required_diagnostic_waypoint_stop(config["tasks"][task_id],
                args.diagnostic_stop_after_waypoint, execute=args.mode == "execute")
    except ValueError as exc:
        parser.error(str(exc))
    if args.collection_scope == "stationary_task":
        from robros_motion_planning.stationary_catalog import validate_stationary_spec
        from robros_motion_planning.stationary_lift import StationaryLiftMachine
        from robros_motion_planning.stationary_place import StationaryPlaceMachine
        try:
            spec = validate_stationary_spec(config)
            if args.sequence or selected != [spec["legacy_task_id"]]:
                raise ValueError("Stationary collection requires its one independently initialized task")
            if spec["goal"] == "pickup_load_jig":
                # The upright start is recorded from the first reach; any
                # episode-boundary arm pose would silently pre-pose the robot.
                if args.initialize_arm_at_pregrasp:
                    raise ValueError("Upright pickup/load forbids --initialize_arm_at_pregrasp")
                # The profile declares the boundary gaze; the runtime records it as the initial condition.
                if args.no_head_gaze and config["tasks"][selected[0]]["upright_start"]["head_gaze"]:
                    raise ValueError("The upright profile declares head_gaze; --no_head_gaze would change its start")
                from robros_motion_planning.stationary_pick_load import StationaryPickLoadMachine as stationary_machine
            else:
                stationary_machine = StationaryPlaceMachine if spec["goal"] == "load_jig" else StationaryLiftMachine
            if args.mode == "execute" and spec["goal"] == "lift_hold" and not args.initialize_arm_at_pregrasp:
                raise ValueError("Stationary collection requires the configured episode-boundary arm initialization")
            stages = stationary_machine(config, selected[0], goal_settings=spec).stages
            validate_diagnostic_waypoint_stop(args.diagnostic_stop_after_waypoint,
                                              config["tasks"][selected[0]], stages)
            # Leaving the final stage is completion, so only earlier stages can stop.
            if args.diagnostic_stop_after is not None and args.diagnostic_stop_after not in stages[:-1]:
                raise ValueError("--diagnostic_stop_after must name a non-final stage of this task: "
                                 + ", ".join(stages[:-1]))
        except ValueError as exc:
            parser.error(str(exc))
    if args.collection_scope == "grasp_carry":
        if args.sequence or "open_cleaner_cover" in selected:
            parser.error("grasp_carry requires independent non-door tasks")
        from robros_motion_planning.grasp_carry import GraspCarryMachine
        try:
            for task_id in selected:
                GraspCarryMachine(config, task_id, offset_w=args.carry_offset_w)
        except ValueError as exc:
            parser.error(str(exc))
    if args.mode == "inspect":
        if remaining:
            parser.error(f"Unknown arguments: {remaining}")
        document = {"task_ids": selected, "calibration_status": config.get("calibration_status"),
                    "graphs": [(stationary_machine(config, t, seed=args.seed, goal_settings=spec)
                                if args.collection_scope == "stationary_task" else
                                GraspCarryMachine(config, t, seed=args.seed, offset_w=args.carry_offset_w)
                                if args.collection_scope == "grasp_carry" else
                                NepesTaskMachine(config, t, seed=args.seed)).graph() for t in selected]}
        if args.collection_scope == "stationary_task":
            document.update(stationary_task=spec, task_ids=[spec["id"]])
        document["robot_gripper"] = args.gripper_selection
        if layout_contract is not None:
            document["single_arm_layout"] = layout_contract
        print(json.dumps(document, indent=2, ensure_ascii=False))
        return 0
    if args.output.exists():
        parser.error(f"Output already exists; choose a new --output: {args.output}")
    from robros_motion_planning.camera_capture import prepare_calibrated_camera_plugins
    camera_plugin_paths = prepare_calibrated_camera_plugins(config) if args.record else ()
    from isaaclab.app import AppLauncher
    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launch_args = launcher_parser.parse_args(remaining)
    launch_args.headless = args.headless
    launch_args.device = args.device
    launch_args.enable_cameras = args.record
    launch_args.kit_args = " ".join((launch_args.kit_args or "", "--/UJITSO/geometry=true",
                                     "--/renderer/multiGpu/enabled=false", "--/renderer/multiGpu/maxGpuCount=1",
                                     "--enable omni.usd.schema.omni_lens_distortion" if camera_plugin_paths else ""))
    launcher = AppLauncher(launch_args, fast_shutdown=False)
    app = launcher.app
    try:
        from robros_motion_planning.task_runtime import run_tasks
        import torch
        # IsaacLab caches tensors first materialized by policy inference. Keep
        # subsequent episode reset writes in the same inference context.
        with torch.inference_mode():
            return run_tasks(args, config, selected, app)
    finally:
        from robros_motion_planning.shutdown import close_simulation_app
        close_simulation_app(app)


if __name__ == "__main__":
    raise SystemExit(main())
