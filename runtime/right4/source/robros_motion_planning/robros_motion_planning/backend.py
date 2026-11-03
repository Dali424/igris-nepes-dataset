"""Snapshot-specific cuRobo 0.7.8 single-arm planning, with no simulator imports."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path

import numpy as np

from .robot_model import build_robot_config, load_robot_model
from .snapshot import load_snapshot, matrix_to_pose, pose_to_matrix, snapshot_digest
from .trajectory import export_hugwbc_npz, export_arm_npz

LOGGER = logging.getLogger(__name__)


class PlanningError(RuntimeError):
    """A rejected plan must never become a replay command."""


def verify_snapshot_kinematics(model, snapshot, arm, *, position_tolerance=0.002,
                               rotation_tolerance=0.01):
    """Reject different USD/URDF joint frames before using planned joint angles."""
    transforms = model.forward_kinematics(snapshot["joint_positions"])
    world_to_base = np.linalg.inv(pose_to_matrix(snapshot["planning_base_pose_w"]))
    bodies = snapshot["body_poses_w"]
    required = {model.joints[name].child for name in model.arm_joint_names(arm)}
    missing = required - bodies.keys()
    if missing:
        raise ValueError(f"Snapshot lacks articulated arm links: {sorted(missing)}")
    errors = {}
    for link in bodies.keys() & transforms.keys():
        measured = world_to_base @ pose_to_matrix(bodies[link])
        expected = transforms[link]
        distance = float(np.linalg.norm(measured[:3, 3] - expected[:3, 3]))
        relative = measured[:3, :3].T @ expected[:3, :3]
        angle = float(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
        errors[link] = {"position_m": distance, "rotation_rad": angle}
        if distance > position_tolerance or angle > rotation_tolerance:
            raise ValueError(f"USD/URDF FK mismatch at {link}: {distance:.6f} m, {angle:.6f} rad")
    return errors


def moving_links(model, arm):
    """Include fingers locked relative to the moving palm, exclude the fixed torso."""
    links = {model.joints[model.arm_joint_names(arm)[0]].child}
    changed = True
    while changed:
        changed = False
        for joint in model.joints.values():
            if joint.parent in links and joint.child not in links:
                links.add(joint.child)
                changed = True
    return links


def world_config(snapshot, snapshot_path):
    records = snapshot["world"]["mesh"]
    if not records:
        raise ValueError("An empty workcell collision snapshot cannot be planned")
    return {"mesh": {
        record["name"]: {
            "file_path": str((Path(snapshot_path).resolve().parent / record["file_path"]).resolve()),
            "pose": record["pose"],
        } for record in records
    }}


def goal_pose_in_base(snapshot, *, goal_pose_world=None, tcp_offset_base=None,
                      object_path=None, object_offset=None, orientation_world=None,
                      model=None, arm="right"):
    """One explicit target; object offsets are pregrasp candidates, not grasp solutions."""
    choices = (goal_pose_world is not None, tcp_offset_base is not None, object_path is not None)
    if sum(choices) != 1:
        raise ValueError("Specify exactly one world pose, base TCP offset, or object target")
    base_to_world = pose_to_matrix(snapshot["planning_base_pose_w"])
    world_to_base = np.linalg.inv(base_to_world)
    if goal_pose_world is not None:
        return matrix_to_pose(world_to_base @ pose_to_matrix(goal_pose_world))
    if model is None:
        model = load_robot_model()
    tcp = model.forward_kinematics(snapshot["joint_positions"])[model.default_ee_link(arm)].copy()
    if tcp_offset_base is not None:
        offset = np.asarray(tcp_offset_base, dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("TCP offset must contain three finite meters")
        tcp[:3, 3] += offset
        return matrix_to_pose(tcp)
    if object_offset is None:
        raise ValueError("An object goal requires an explicit object-local offset in meters")
    obj = snapshot["objects"].get(object_path)
    if obj is None:
        raise ValueError(f"Object is not present in snapshot: {object_path}")
    offset = np.asarray(object_offset, dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("Object offset must contain three finite meters")
    object_to_world = pose_to_matrix(obj["pose_w"])
    tcp[:3, 3] = (world_to_base @ object_to_world @ np.r_[offset, 1])[:3]
    if orientation_world is not None:
        orientation = pose_to_matrix([0, 0, 0, *orientation_world])
        tcp[:3, :3] = world_to_base[:3, :3] @ orientation[:3, :3]
    return matrix_to_pose(tcp)


def validate_goal_endpoint(model, initial_positions, arm, path_joint_names, final_positions,
                           goal_pose_base, *, position_tolerance_m=0.005, rotation_tolerance_rad=0.05):
    """Independent CPU FK validation before any path is exported for execution."""
    if set(path_joint_names) != set(model.arm_joint_names(arm)):
        raise PlanningError("Endpoint joint names do not match the selected arm")
    values = np.asarray(final_positions, dtype=float)
    if values.shape != (len(path_joint_names),) or not np.isfinite(values).all():
        raise PlanningError("Invalid endpoint joint values")
    actual = model.forward_kinematics({**initial_positions, **dict(zip(path_joint_names, values))})[
        model.default_ee_link(arm)]
    goal = pose_to_matrix(goal_pose_base)
    distance = float(np.linalg.norm(actual[:3, 3] - goal[:3, 3]))
    relative = actual[:3, :3].T @ goal[:3, :3]
    rotation = float(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
    if distance > position_tolerance_m or rotation > rotation_tolerance_rad:
        raise PlanningError(f"Planned endpoint misses TCP goal: {distance:.6f} m, {rotation:.6f} rad")
    return {"position_error_m": distance, "rotation_error_rad": rotation,
            "position_tolerance_m": position_tolerance_m, "rotation_tolerance_rad": rotation_tolerance_rad,
            "verified_by": "independent_cpu_urdf_forward_kinematics"}


def plan_snapshot(snapshot_path, output_path, *, arm="right", goal_pose_base,
                  urdf_path=None, max_attempts=8, timeout_s=60.0, device="cuda:0", arm_path_only=False,
                  sphere_buffer=0.002, world_extra_buffer=0.0, cpu_goal_seed=False,
                  collision_fit_intermediates=0, waypoint_poses_base=None,
                  path_method="curobo"):
    """Plan one goal or an explicit ordered route from the measured snapshot.

    Waypoints are TCP poses in the frozen planning-base frame and exclude the
    final goal. Every segment is independently collision checked. The combined
    execution artifact is created only after every segment passes validation.
    """
    arguments = dict(arm=arm, goal_pose_base=goal_pose_base, urdf_path=urdf_path,
                     max_attempts=max_attempts, timeout_s=timeout_s, device=device,
                     arm_path_only=arm_path_only, sphere_buffer=sphere_buffer,
                     world_extra_buffer=world_extra_buffer, cpu_goal_seed=cpu_goal_seed,
                     collision_fit_intermediates=collision_fit_intermediates, path_method=path_method)
    if waypoint_poses_base is not None:
        from .waypoint_planning import plan_waypoint_route
        return plan_waypoint_route(snapshot_path, output_path, waypoint_poses_base,
                                   segment_planner=_plan_snapshot_segment, **arguments)
    return _plan_snapshot_segment(snapshot_path, output_path, **arguments)


def _plan_snapshot_segment(snapshot_path, output_path, *, arm="right", goal_pose_base,
                           urdf_path=None, max_attempts=8, timeout_s=60.0,
                           device="cuda:0", arm_path_only=False, sphere_buffer=0.002,
                           world_extra_buffer=0.0, cpu_goal_seed=False,
                           collision_fit_intermediates=0, continuation_joint_positions=None,
                           path_method="curobo"):
    """Plan an unloaded arm at a frozen base, validate, then export at 50 Hz.

    This is not a full-body dynamics or grasp planner. Objects, base, opposite arm
    and fingers remain frozen. Capture/replan after any of those states change.
    cpu_goal_seed explicitly selects bounded CPU IK plus public cuRobo joint-goal
    planning. Its candidate also refines the collision cover; all start/goal/path
    self and world constraints remain enabled. No fallback bypasses a rejection.
    """
    if max_attempts < 1 or not np.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("max_attempts and timeout_s must be positive")
    if path_method not in ("curobo", "checked_joint_interpolation"):
        raise ValueError("Unknown path_method")
    if path_method == "checked_joint_interpolation" and not cpu_goal_seed:
        raise ValueError("checked_joint_interpolation requires cpu_goal_seed")
    if not np.isfinite([sphere_buffer, world_extra_buffer]).all() or min(sphere_buffer, world_extra_buffer) < 0:
        raise ValueError("Collision sphere buffers must be finite and nonnegative")
    if (not isinstance(collision_fit_intermediates, int) or isinstance(collision_fit_intermediates, bool)
            or not 0 <= collision_fit_intermediates <= 32):
        raise ValueError("collision_fit_intermediates must be an integer from 0 to 32")
    if collision_fit_intermediates and not cpu_goal_seed:
        raise ValueError("Intermediate collision fitting requires cpu_goal_seed")
    output_path = Path(output_path).resolve()
    if output_path.suffix.lower() != ".npz":
        raise ValueError("output_path must have an .npz suffix")
    if output_path.exists() or output_path.with_suffix(".manifest.json").exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    snapshot = load_snapshot(snapshot_path)
    model = load_robot_model(urdf_path)
    fk_errors = verify_snapshot_kinematics(model, snapshot, arm)
    initial_positions = dict(snapshot["joint_positions"])
    if continuation_joint_positions is not None:
        continuation = model.validate_snapshot(continuation_joint_positions)
        active = set(model.arm_joint_names(arm))
        changed_locked = [name for name in initial_positions if name not in active
                          and abs(continuation[name] - initial_positions[name]) > 1e-12]
        if changed_locked:
            raise PlanningError(f"Route continuation changes frozen joints: {changed_locked}")
        initial_positions = continuation
    goal_pose_base = matrix_to_pose(pose_to_matrix(goal_pose_base))
    generated_dir = output_path.parent / (output_path.stem + "_model")
    candidate = None
    fit_positions = []
    fit_provenance = {"intermediate_count": collision_fit_intermediates,
                      "interpolation": "hypothetical_linear_active_joint_positions",
                      "path_certified": False,
                      "limitation": "Fitting failures at hypothetical interpolation poses do not prove that a curved collision-free path is absent."}
    if cpu_goal_seed:
        from .goal_ik import solve_goal_candidate
        candidate = solve_goal_candidate(model, initial_positions, arm, goal_pose_base)
        generated_dir.mkdir(parents=True, exist_ok=True)
        (generated_dir / "cpu_goal_candidate.json").write_text(
            json.dumps(candidate, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if not candidate["success"]:
            raise PlanningError("Bounded CPU goal IK found no acceptable candidate; no collision or reachability conclusion")
        LOGGER.info("CPU IK candidate: %.6f m / %.6f rad; fitting start and candidate geometry",
                    candidate["position_error_m"], candidate["rotation_error_rad"])
        fractions = np.linspace(0, 1, collision_fit_intermediates + 2)[1:]
        for fraction in fractions:
            positions = dict(initial_positions)
            for name in model.arm_joint_names(arm):
                positions[name] = float((1 - fraction) * positions[name] + fraction * candidate["joint_positions"][name])
            fit_positions.append(positions)
        fit_provenance["fractions_including_measured_start_and_candidate"] = [0.0, *fractions.tolist()]
        (generated_dir / "collision_fit_request.json").write_text(
            json.dumps(fit_provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    version = importlib.metadata.version("nvidia-curobo")
    if version != "0.7.8":
        raise RuntimeError(f"Expected cuRobo 0.7.8, found {version}; use scripts/planning/install_curobo.sh")
    import torch
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import RobotConfig
    from curobo.types.state import JointState
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
    from .world_collision import make_world_checker
    from .world_geometry import prepare_world

    if not torch.cuda.is_available():
        raise RuntimeError("cuRobo requires CUDA")
    tensor_args = TensorDeviceType(device=torch.device(device))
    try:
        config = build_robot_config(model, initial_positions, arm, generated_dir,
                                    sphere_buffer=sphere_buffer, validation_joint_positions=fit_positions)
    except ValueError as exc:
        if fit_positions:
            raise PlanningError(f"Collision fitting rejected the requested hypothetical joint poses; "
                                f"this does not prove that no curved collision-free path exists: {exc}") from exc
        raise
    LOGGER.info("Robot collision model: %d spheres", config.collision_model_metadata["sphere_count"])
    config_path = generated_dir / "robot_config.json"
    config_path.write_text(json.dumps(config.robot_cfg, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    robot_cfg = RobotConfig.from_dict(copy.deepcopy(config.robot_cfg), tensor_args)
    prepared_world, world_metadata = prepare_world(snapshot, snapshot_path, generated_dir)
    LOGGER.info("Prepared collision world: %d meshes", len(prepared_world["mesh"]))
    checker = make_world_checker(
        prepared_world, tensor_args,
        robot_cfg.kinematics.kinematics_config, moving_links(model, arm),
        extra_radius=world_extra_buffer,
    )
    mg = MotionGen(MotionGenConfig.load_from_robot_config(
        robot_cfg, world_coll_checker=checker, tensor_args=tensor_args,
        interpolation_dt=0.02, interpolation_steps=5000, use_cuda_graph=False,
        self_collision_check=True, self_collision_opt=True,
        evaluate_interpolated_trajectory=True, collision_activation_distance=0.01,
        num_ik_seeds=32, num_trajopt_seeds=4,
    ))
    names = list(mg.joint_names)
    if set(names) != set(config.active_joint_names):
        raise PlanningError(f"Unexpected cuRobo active joints: {names}")
    start = JointState.from_position(tensor_args.to_device(
        [[initial_positions[name] for name in names]]), joint_names=names)
    # check_start_state in 0.7.8 can change cost toggles on failure; abort immediately.
    valid, status = mg.check_start_state(start)
    if not valid:
        raise PlanningError(f"Planning start rejected: {status}")
    LOGGER.info("Start state passed; planning TCP goal %s", goal_pose_base[:3])
    plan_config = MotionGenPlanConfig(
        max_attempts=max_attempts, timeout=timeout_s, enable_graph=True,
        enable_finetune_trajopt=True,
    )
    timing = None
    if candidate is None:
        goal = Pose(position=tensor_args.to_device([goal_pose_base[:3]]),
                    quaternion=tensor_args.to_device([goal_pose_base[3:]]))
        result = mg.plan_single(start, goal, plan_config)
    else:
        goal_state = JointState.from_position(tensor_args.to_device(
            [[candidate["joint_positions"][name] for name in names]]), joint_names=names)
        metrics = mg.rollout_fn.rollout_constraint(goal_state.position[:, None, :].contiguous(), use_batch_env=False)
        if not bool(torch.all(metrics.feasible).item()):
            raise PlanningError("CPU IK candidate rejected by GPU self/world/joint constraints")
        LOGGER.info("CPU IK candidate passed GPU constraints; selected path method: %s", path_method)
        if path_method == "curobo":
            result = mg.plan_single_js(start, goal_state, plan_config)
    if path_method == "checked_joint_interpolation":
        from .trajectory import bounded_quintic_joint_path
        cspace = config.robot_cfg["kinematics"]["cspace"]
        timing = bounded_quintic_joint_path(
            [initial_positions[name] for name in names],
            [candidate["joint_positions"][name] for name in names],
            [min(1., model.joints[name].velocity) for name in names],
            [cspace["max_acceleration"]] * len(names), [cspace["max_jerk"]] * len(names),
        )
        positions = timing["positions"]
        validation_positions = timing["validation_positions"]
        sample_dt, path_names = timing["sample_dt"], names
        status = "CHECKED_JOINT_INTERPOLATION_NO_OPTIMIZATION"
        LOGGER.info("Checking %d physics-interval poses before exporting %d command frames",
                    len(validation_positions), len(positions))
    else:
        if result.success is None or not bool(result.success.item()):
            raise PlanningError(f"MotionGen did not find a valid path: {result.status}")
        LOGGER.info("MotionGen succeeded; validating output frames")
        plan = result.get_interpolated_plan()
        positions = plan.position.detach().cpu().numpy()
        if positions.ndim == 3 and positions.shape[0] == 1:
            positions = positions[0]
        validation_positions = positions
        sample_dt = float(result.interpolation_dt)
        path_names = list(plan.joint_names or names)
        status = str(result.status)
    if abs(sample_dt - 0.02) > 1e-8:
        raise PlanningError(f"Unexpected interpolation dt: {sample_dt}")
    if set(path_names) != set(names):
        raise PlanningError("cuRobo path joint names differ from active joint set")
    # Recheck every output frame, including the endpoint, against both constraints.
    ordered = validation_positions[:, [path_names.index(name) for name in names]]
    validated_count = 0
    for block in np.array_split(ordered, max(1, (len(ordered) + 127) // 128)):
        queries = tensor_args.to_device(block)[:, None, :].contiguous()
        metrics = mg.rollout_fn.rollout_constraint(queries, use_batch_env=False)
        if not bool(torch.all(metrics.feasible).item()):
            feasible = metrics.feasible.detach().cpu().numpy().reshape(len(block), -1).all(axis=1)
            rejected = np.flatnonzero(~feasible)
            failure = {"path_method": path_method, "joint_names": names,
                       "first_failed_validation_index": int(validated_count + rejected[0]),
                       "validation_dt_s": sample_dt if timing is None else timing["validation_dt"],
                       "failed_joint_positions": block[rejected[:16]].tolist(),
                       "failure_kind": "self_world_or_joint_constraint",
                       "execution_artifact_written": False}
            try:
                state = mg.rollout_fn.dynamics_model.forward(mg.rollout_fn.start_state, queries)
                components = {
                    "joint_bounds": mg.rollout_fn.bound_constraint.forward(state.state_seq),
                    "world_collision": mg.rollout_fn.primitive_collision_constraint.forward(state.robot_spheres, env_query_idx=None),
                    "self_collision": mg.rollout_fn.robot_self_collision_constraint.forward(state.robot_spheres),
                }
                failure["constraint_components_at_failed_samples"] = {
                    name: value.detach().cpu().numpy().reshape(len(block), -1)[rejected[:16]].tolist()
                    for name, value in components.items()}
            except Exception as diagnostic_error:
                failure["component_diagnostic_error"] = f"{type(diagnostic_error).__name__}: {diagnostic_error}"
            (generated_dir / "validation_failure.json").write_text(json.dumps(failure, indent=2, allow_nan=False) + "\n")
            raise PlanningError("Interpolated output or finer physics sample fails collision/joint constraints")
        validated_count += len(block)
    endpoint = validate_goal_endpoint(model, initial_positions, arm, path_names,
                                     positions[-1], goal_pose_base)
    info = {
        "backend": "curobo" if timing is None else "explicit_joint_interpolation_with_curobo_collision_validation",
        "version": version, "arm": arm,
        "warp_version": importlib.metadata.version("warp-lang"),
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(tensor_args.device),
        "dependency_patch": "scripts/planning/curobo_warp_1_15.patch",
        "scope": "single_unloaded_arm_frozen_base_reset_snapshot",
        "segment_start_kind": "measured_snapshot" if continuation_joint_positions is None else "validated_previous_segment_endpoint",
        "segment_initial_joint_positions": initial_positions,
        "source_snapshot_state_is_unmodified": True,
        "goal_pose_base": goal_pose_base, "tcp_link": config.ee_link,
        "goal_planning_mode": ("curobo_pose_goal" if candidate is None else
                               "bounded_cpu_ik_then_curobo_plan_single_js" if timing is None else
                               "bounded_cpu_ik_then_explicit_checked_joint_interpolation"),
        "timing": None if timing is None else timing["metadata"],
        "cpu_goal_candidate": candidate,
        "collision_fit_request": fit_provenance,
        "endpoint_validation": endpoint,
        "collision_validation": {"start_passed": True, "goal_and_all_output_frames_passed": True,
                                  "checked_output_frames": len(positions),
                                  "checked_validation_frames": len(validation_positions),
                                  "validation_dt_s": sample_dt if timing is None else timing["validation_dt"],
                                  "continuous_collision_proof": False,
                                  "physical_execution_validated": False},
        "active_joints": names, "locked_joints": config.locked_joint_positions,
        "world_collision_links": sorted(moving_links(model, arm)),
        "world_collision_filter": checker.world_collision_manifest,
        "world_geometry": world_metadata,
        "self_collision": "full_body_with_locked_and_adjacent_pair_exclusions",
        "fk_errors": fk_errors, "status": status,
        "source_urdf_sha256": hashlib.sha256(model.urdf_path.read_bytes()).hexdigest(),
        "collision_geometry_sources": config.geometry_sources,
        "collision_model": config.collision_model_metadata,
        "collision_margins": {
            "self_per_sphere_m": sphere_buffer,
            "world_base_per_sphere_m": sphere_buffer,
            "world_extra_per_sphere_m": world_extra_buffer,
            "world_total_per_sphere_m": sphere_buffer + world_extra_buffer,
            "self_buffer_semantics": "Explicit per-link cuRobo self_collision_buffer; CPU and GPU use the same self margin.",
        },
        "robot_config_path": str(config_path),
        "robot_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "snapshot_path": str(Path(snapshot_path).resolve()),
        "physics_execution_validated": False,
    }
    limits = dict(model.joint_limits)
    limits.update(config.joint_limits)
    exporter = export_arm_npz if arm_path_only else export_hugwbc_npz
    return exporter(
        output_path, joint_names=path_names, positions=positions, sample_dt=sample_dt,
        initial_positions=initial_positions, position_limits=limits,
        velocity_limits={name: min(1.0, model.joints[name].velocity) for name in names},
        planner_success=True, snapshot_hash=snapshot_digest(snapshot), planner_info=info,
    )
