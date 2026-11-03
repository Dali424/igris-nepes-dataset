"""Command line entry point for a workcell snapshot to HugWBC NPZ."""

import argparse
import json
import logging
from pathlib import Path

from .backend import goal_pose_in_base, plan_snapshot, verify_snapshot_kinematics
from .robot_model import load_robot_model
from .snapshot import load_snapshot


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    logging.getLogger("robros_motion_planning").setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--urdf")
    parser.add_argument("--arm-path-only", action="store_true",
                        help="Export named 7-joint path for model-specific live controllers")
    parser.add_argument("--output", help="New .npz file (existing files are refused)")
    parser.add_argument("--inspect", action="store_true", help="Validate snapshot and FK without CUDA")
    targets = parser.add_mutually_exclusive_group()
    targets.add_argument("--goal-pose-world", nargs=7, type=float, metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"))
    targets.add_argument("--tcp-offset-base", nargs=3, type=float, metavar=("DX", "DY", "DZ"))
    targets.add_argument("--goal-object", help="Full captured prim path, e.g. /World/Scene/Ring_01")
    parser.add_argument("--object-offset", nargs=3, type=float, help="Object-local target offset in meters")
    parser.add_argument("--orientation-world", nargs=4, type=float, help="QW QX QY QZ; default holds TCP orientation")
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sphere-buffer", type=float, default=0.002,
                        help="Self and base world margin per robot sphere in meters (default 0.002)")
    parser.add_argument("--world-extra-buffer", type=float, default=0.0,
                        help="Additional world-only sphere margin in meters; self geometry is unchanged")
    parser.add_argument("--cpu-goal-seed", action="store_true",
                        help="Fit conservative geometry at start plus bounded CPU IK candidate, then use cuRobo joint-goal planning with full collision checks")
    parser.add_argument("--collision-fit-intermediates", type=int, default=0,
                        help="Additional hypothetical joint-interpolation fitting poses (0..32, requires --cpu-goal-seed); not a path certificate")
    parser.add_argument("--waypoints-world", help="JSON array of 1..8 ordered world TCP poses, excluding final goal; all cuRobo segments must pass before one path is exported")
    parser.add_argument("--path-method", choices=("curobo", "checked_joint_interpolation"), default="curobo",
                        help="Explicit interpolation requires CPU goal IK, bounded joint rates, and collision checks at 250 Hz; it is not trajectory optimization")
    args = parser.parse_args()
    try:
        snapshot = load_snapshot(args.snapshot)
        model = load_robot_model(args.urdf)
        fk = verify_snapshot_kinematics(model, snapshot, args.arm)
        if args.inspect:
            print(json.dumps({"joints": len(snapshot["joint_positions"]),
                              "meshes": len(snapshot["world"]["mesh"]),
                              "objects": sorted(snapshot["objects"]),
                              "fk_max_position_m": max(v["position_m"] for v in fk.values())}, indent=2))
            return
        if not args.output:
            parser.error("--output is required for planning")
        if not args.goal_object and (args.object_offset is not None or args.orientation_world is not None):
            parser.error("--object-offset/--orientation-world require --goal-object")
        goal = goal_pose_in_base(
            snapshot, goal_pose_world=args.goal_pose_world, tcp_offset_base=args.tcp_offset_base,
            object_path=args.goal_object, object_offset=args.object_offset,
            orientation_world=args.orientation_world, model=model, arm=args.arm,
        )
        waypoints = None
        if args.waypoints_world:
            from .waypoint_planning import world_waypoints_to_base
            waypoints = world_waypoints_to_base(snapshot, json.loads(Path(args.waypoints_world).read_text()))
        result = plan_snapshot(args.snapshot, args.output, arm=args.arm, goal_pose_base=goal,
                               urdf_path=args.urdf, max_attempts=args.max_attempts,
                               timeout_s=args.timeout, device=args.device, arm_path_only=args.arm_path_only,
                               sphere_buffer=args.sphere_buffer, world_extra_buffer=args.world_extra_buffer,
                               cpu_goal_seed=args.cpu_goal_seed,
                               collision_fit_intermediates=args.collision_fit_intermediates,
                               waypoint_poses_base=waypoints, path_method=args.path_method)
        print(f"Wrote {result.frame_count} frames / {result.duration_s:.3f}s: {result.npz_path}")
        print(f"Manifest: {result.manifest_path}")
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        parser.exit(1, f"Planning rejected: {error}\n")


if __name__ == "__main__":
    main()
