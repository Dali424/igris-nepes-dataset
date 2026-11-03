"""Explicit unloaded-arm routes sharing one immutable measured world snapshot."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import numpy as np

from .robot_model import load_robot_model, planning_joint_limits
from .snapshot import load_snapshot, matrix_to_pose, pose_to_matrix, snapshot_digest
from .trajectory import export_arm_npz, export_hugwbc_npz


def propose_cart_lift_route(snapshot, model, arm, goal_pose_base, *,
                            transit_offset_goal_world=(-.12, -.287, .078),
                            hover_offset_goal_world=(.04, 0., .12), seeds=16):
    """Propose the NEPES cart-front route using current joints and base pose.

    Offsets are explicit WORLD meters relative to the requested pregrasp TCP,
    not object-frame offsets. Defaults describe the right-arm cart station;
    another station/arm must supply appropriate offsets. First transit IK holds
    the measured wrist angles while allowing TCP orientation to change. Its
    actual FK orientation becomes an explicit pose waypoint. The hover retains
    the requested final orientation. Returned candidates have NOT passed any
    collision/path check and must be sent to ``plan_snapshot`` as waypoints.
    """
    from .goal_ik import solve_goal_candidate

    offsets = np.asarray([transit_offset_goal_world, hover_offset_goal_world], dtype=float)
    if offsets.shape != (2, 3) or not np.isfinite(offsets).all():
        raise ValueError("Transit and hover offsets must be three finite world meters")
    base = pose_to_matrix(snapshot["planning_base_pose_w"])
    world_to_base = np.linalg.inv(base)
    goal_world = base @ pose_to_matrix(goal_pose_base)
    transit = goal_world.copy()
    transit[:3, 3] += offsets[0]
    lifted = solve_goal_candidate(model, snapshot["joint_positions"], arm,
                                  matrix_to_pose(world_to_base @ transit), seeds=seeds,
                                  position_only=True, hold_wrist=True)
    report = {"method": "cart_front_lift_then_hover_pose_candidate",
              "source_base_pose_w": list(snapshot["planning_base_pose_w"]),
              "source_joint_positions": dict(snapshot["joint_positions"]),
              "transit_offset_goal_world_m": offsets[0].tolist(),
              "hover_offset_goal_world_m": offsets[1].tolist(),
              "collision_checked": False, "path_validated": False,
              "physics_execution_validated": False, "transit_ik": lifted,
              "success": bool(lifted["success"])}
    if not lifted["success"]:
        return report
    hover = goal_world.copy()
    hover[:3, 3] += offsets[1]
    poses_base = [lifted["candidate_pose_base"], matrix_to_pose(world_to_base @ hover)]
    report["waypoint_poses_base"] = poses_base
    report["waypoint_poses_world"] = [matrix_to_pose(base @ pose_to_matrix(pose)) for pose in poses_base]
    # Check pose reachability sequentially with unchanged measured locked joints.
    # The planner will still recompute each candidate from its actual segment
    # endpoint, rather than treating these hypothetical states as measurements.
    current = dict(snapshot["joint_positions"])
    report["pose_ik_candidates"] = []
    for pose in [*poses_base, goal_pose_base]:
        candidate = solve_goal_candidate(model, current, arm, pose, seeds=seeds)
        report["pose_ik_candidates"].append(candidate)
        if not candidate["success"]:
            report["success"] = False
            break
        current = candidate["joint_positions"]
    return report


def world_waypoints_to_base(snapshot, waypoints_world):
    """Normalize explicit world TCP poses using this snapshot's measured base."""
    if not isinstance(waypoints_world, (list, tuple)) or not 1 <= len(waypoints_world) <= 8:
        raise ValueError("Provide one to eight ordered TCP waypoints")
    world_to_base = np.linalg.inv(pose_to_matrix(snapshot["planning_base_pose_w"]))
    return [matrix_to_pose(world_to_base @ pose_to_matrix(pose)) for pose in waypoints_world]


def join_checked_segments(segments, *, join_tolerance_rad=1e-6):
    """Join equal-rate, already validated paths; never bridge a discontinuity."""
    if not segments:
        raise ValueError("At least one checked segment is required")
    names = tuple(segments[0][0])
    parts = []
    for index, (segment_names, positions) in enumerate(segments):
        if len(set(segment_names)) != len(names) or set(segment_names) != set(names):
            raise ValueError("Segment joint names do not match")
        values = np.asarray(positions, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(names) or not len(values) or not np.isfinite(values).all():
            raise ValueError("Invalid segment positions")
        values = values[:, [list(segment_names).index(name) for name in names]]
        if index:
            error = float(np.max(np.abs(parts[-1][-1] - values[0])))
            if error > join_tolerance_rad:
                raise ValueError(f"Discontinuous route join: {error:.9f} rad")
            # Keep both boundary samples: the equal state supplies a 20 ms hold,
            # so concatenation introduces no newly interpolated joint edge.
        parts.append(values)
    return names, np.concatenate(parts)


def plan_waypoint_route(snapshot_path, output_path, waypoint_poses_base, *, segment_planner,
                        arm, goal_pose_base, urdf_path, max_attempts, timeout_s, device,
                        arm_path_only, sphere_buffer, world_extra_buffer, cpu_goal_seed,
                        collision_fit_intermediates, path_method="curobo"):
    """Plan each segment from its predecessor's actual validated last frame.

    Source snapshot, world meshes, base, fingers, neck and opposite arm stay
    unchanged. Intermediate starts are predictions, never relabeled sensor
    observations. Partial segment artifacts remain diagnostic if a later
    segment fails; no combined execution artifact is then written.
    """
    from .backend import PlanningError, validate_goal_endpoint

    if not isinstance(waypoint_poses_base, (list, tuple)) or not 1 <= len(waypoint_poses_base) <= 8:
        raise ValueError("Provide one to eight ordered TCP waypoints excluding the final goal")
    goals = [matrix_to_pose(pose_to_matrix(pose)) for pose in [*waypoint_poses_base, goal_pose_base]]
    output_path = Path(output_path).resolve()
    if output_path.suffix.lower() != ".npz":
        raise ValueError("output_path must have an .npz suffix")
    if output_path.exists() or output_path.with_suffix(".manifest.json").exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    snapshot = load_snapshot(snapshot_path)
    model = load_robot_model(urdf_path)
    current = dict(snapshot["joint_positions"])
    directory = output_path.parent / (output_path.stem + "_segments")
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "igris_c_frozen_snapshot_waypoint_route_v1",
        "source_snapshot_sha256": snapshot_digest(snapshot),
        "goal_poses_base_including_final": goals,
        "scope": "unloaded_arm_only_all_nonselected_joints_and_world_frozen",
        "intermediate_starts": "predicted_from_validated_previous_segment_endpoint_not_new_observations",
        "all_segments_passed": False, "physics_execution_validated": False, "segments": [],
        "path_method": path_method,
    }
    report_path = directory / "route_report.json"

    def save_report():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save_report()
    segments = []
    try:
        for index, goal in enumerate(goals):
            target = directory / f"segment_{index:02d}.npz"
            # Named-arm artifacts are independent of legacy 51-joint schemas.
            result = segment_planner(
                snapshot_path, target, arm=arm, goal_pose_base=goal, urdf_path=urdf_path,
                max_attempts=max_attempts, timeout_s=timeout_s, device=device,
                arm_path_only=True, sphere_buffer=sphere_buffer,
                world_extra_buffer=world_extra_buffer, cpu_goal_seed=cpu_goal_seed,
                collision_fit_intermediates=collision_fit_intermediates,
                continuation_joint_positions=None if index == 0 else current,
                path_method=path_method,
            )
            manifest = json.loads(Path(result.manifest_path).read_text())
            if manifest["snapshot_sha256"] != snapshot_digest(snapshot):
                raise PlanningError("Route segment changed the source snapshot")
            info = manifest["planner"]
            checks = info.get("collision_validation", {})
            if not (checks.get("start_passed") and checks.get("goal_and_all_output_frames_passed")):
                raise PlanningError("Route segment lacks complete collision validation")
            with np.load(result.npz_path, allow_pickle=False) as archive:
                names = archive["joint_names"].tolist()
                positions = archive["joint_pos"].copy()
                if float(archive["fps"]) != 50:
                    raise PlanningError("Route segment is not sampled at 50 Hz")
            if set(names) != set(model.arm_joint_names(arm)):
                raise PlanningError("Route segment changes the active joint contract")
            if np.max(np.abs(positions[0] - [current[name] for name in names])) > 1e-6:
                raise PlanningError("Route segment start does not match its predecessor")
            current.update(dict(zip(names, map(float, positions[-1]))))
            endpoint = validate_goal_endpoint(model, snapshot["joint_positions"], arm, names,
                                              positions[-1], goal)
            segments.append((names, positions))
            report["segments"].append({
                "index": index, "npz_path": str(result.npz_path),
                "npz_sha256": hashlib.sha256(Path(result.npz_path).read_bytes()).hexdigest(),
                "manifest_path": str(result.manifest_path), "frame_count": len(positions),
                "endpoint_validation": endpoint,
                "collision_model": info.get("collision_model"),
                "collision_margins": info.get("collision_margins"),
                "collision_validation": checks, "timing": info.get("timing"),
            })
            save_report()
            gc.collect()
        names, positions = join_checked_segments(segments)
        report["all_segments_passed"] = True
        report["frame_count"] = len(positions)
        report["join_behavior"] = "one_20ms_identical_position_hold_at_each_join_no_new_geometric_edges"
        save_report()
        limits = dict(model.joint_limits)
        limits.update(planning_joint_limits(model, arm))
        exporter = export_arm_npz if arm_path_only else export_hugwbc_npz
        result = exporter(
            output_path, joint_names=names, positions=positions, sample_dt=.02,
            initial_positions=snapshot["joint_positions"], position_limits=limits,
            velocity_limits={name: min(1., model.joints[name].velocity) for name in names},
            planner_success=True, snapshot_hash=snapshot_digest(snapshot), planner_info={
                "backend": "curobo_segmented_waypoint_route" if path_method == "curobo" else
                           "explicit_waypoint_interpolation_with_curobo_collision_validation",
                "route": report,
                "collision_validation": {"start_passed": True,
                    "goal_and_all_output_frames_passed": True, "checked_output_frames": len(positions),
                    "physical_execution_validated": False},
                "snapshot_path": str(Path(snapshot_path).resolve()),
                "physics_execution_validated": False,
            },
        )
        return result
    except BaseException as error:
        report["all_segments_passed"] = False
        report["failure"] = f"{type(error).__name__}: {error}"
        save_report()
        raise
