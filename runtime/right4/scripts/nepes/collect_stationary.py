#!/usr/bin/env python3
"""Collect independent HugWBC successes; default prints the command without running Isaac."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/robros_motion_planning"))


def main(argv=None):
    from robros_motion_planning.latest_robot import GRIPPER_VARIANTS
    from robros_motion_planning.stationary_catalog import CATALOGS, DEFAULT_CATALOG
    from robros_motion_planning.stationary_collection import build_plan, collect, export_collection
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="1",
                        help="Task number or ID in the selected catalog (v2: task 1 configured; v1: tasks 1 and 2)")
    parser.add_argument("--catalog", choices=tuple(CATALOGS), default=None,
                        help="v2: corrected five tasks; v1: legacy component profiles. Default: the --config "
                             f"profile's own catalog, else {DEFAULT_CATALOG}")
    parser.add_argument("--config", type=Path, help="Override the selected task's independent scene profile")
    parser.add_argument("--gripper-variant", choices=GRIPPER_VARIANTS, default=None,
                        help="Assert the jaw model the profile pins (absent in the profile = authored_v0); "
                             "a mismatch stops before any attempt. The profile, not this flag, selects it")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--successes", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-wall-seconds", type=float, default=3600.)
    parser.add_argument("--capture-render-mode", choices=("legacy", "capture_only"), default="legacy",
                        help="legacy keeps automatic rendering; capture_only renders at the recorded 30 Hz deadlines")
    parser.add_argument("--image-writer-workers", type=int, default=0,
                        help="PNG writer workers; 0 keeps synchronous image writes")
    parser.add_argument("--image-writer-queue", type=int, default=8,
                        help="Maximum queued RGB frame bundles for asynchronous image writing")
    parser.add_argument("--profile-throughput", action="store_true",
                        help="Record wall-time counters for rendering, control, and recording")
    parser.add_argument("--isaac-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--execute", action="store_true", help="Start bounded native attempts; existing output is never overwritten")
    parser.add_argument("--storage-mode", choices=("raw", "compact"), default="raw",
                        help="compact: verify one official MP4/Parquet episode before cleaning its temporary PNGs")
    parser.add_argument("--physics-trace", choices=("gzip", "keep", "discard"), default="gzip",
                        help="Compact mode: detailed successful physics traces (default: verified gzip)")
    parser.add_argument("--failed-physics-trace", choices=("gzip", "keep", "discard"), default="gzip",
                        help="Compact mode: detailed failed-attempt traces; summaries are always retained")
    parser.add_argument("--min-free-gib", type=float, default=20.,
                        help="Compact mode: minimum available GiB before each new native attempt")
    parser.add_argument("--max-consecutive-failures", type=int, default=3,
                        help="Compact mode: stop after this many consecutive failed attempts (default: 3)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume the same compact output with identical configuration and collection options")
    parser.add_argument("--retry-after-failure-limit", action="store_true",
                        help="With --resume, explicitly start one new bounded retry window after inspecting failures")
    parser.add_argument("--export", type=Path, metavar="DATASET", help="Export accepted successes as official LeRobot v3 after reaching the quota")
    parser.add_argument("--export-partial", action="store_true",
                        help="Explicitly export verified compact successes even if the collection quota is unmet")
    parser.add_argument("--export-only", action="store_true",
                        help="Export an existing compact collection; do not create a plan or start simulation")
    parser.add_argument("--dataset-python", type=Path, default=Path(sys.executable))
    args = parser.parse_args(argv)
    if args.export and not (args.execute or args.export_only):
        parser.error("--export requires --execute or --export-only")
    if args.export_only and (not args.export or args.execute or args.resume):
        parser.error("--export-only requires --export and cannot be combined with --execute or --resume")
    if args.export_partial and (not args.export or (not args.export_only and args.storage_mode != "compact")):
        parser.error("--export-partial requires --export and a compact collection")
    if args.resume and (not args.execute or args.storage_mode != "compact"):
        parser.error("--resume requires --execute --storage-mode compact")
    if args.retry_after_failure_limit and not args.resume:
        parser.error("--retry-after-failure-limit requires --resume")
    try:
        if args.export_only:
            from robros_motion_planning.collection import read_json
            from robros_motion_planning.compact_collection import export_compact_collection
            state = read_json(args.output / "collection.json")
            if state.get("plan", {}).get("storage_options", {}).get("storage_mode") != "compact":
                raise ValueError("--export-only currently requires an existing compact collection")
            path = export_compact_collection(args.output, args.export, args.dataset_python,
                                             allow_partial=args.export_partial)
            summary = dict(state.get("summary", {}))
            summary.update(dataset_path=str(path), export_only=True, partial_export=args.export_partial)
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return 0
        plan = build_plan(config=args.config, isaac_python=args.isaac_python, task=args.task,
                          catalog=args.catalog, successes=args.successes, max_attempts=args.max_attempts,
                          seed=args.seed, max_wall_seconds=args.max_wall_seconds,
                          gripper_variant=args.gripper_variant,
                          capture_render_mode=args.capture_render_mode,
                          image_writer_workers=args.image_writer_workers,
                          image_writer_queue=args.image_writer_queue,
                          profile_throughput=args.profile_throughput,
                          storage_mode=args.storage_mode, physics_trace=args.physics_trace,
                          failed_physics_trace=args.failed_physics_trace, min_free_gib=args.min_free_gib,
                          max_consecutive_failures=args.max_consecutive_failures)
        report = collect(plan, args.output, execute=args.execute,
                         dataset_python=args.dataset_python, resume=args.resume,
                         retry_after_failure_limit=args.retry_after_failure_limit)
        summary = report.get("summary", report)
        if args.export and (summary["quota_complete"] or args.export_partial):
            summary["dataset_path"] = str(export_collection(
                args.output, args.export, args.dataset_python, allow_partial=args.export_partial))
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if not args.execute or summary["quota_complete"] else 2
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Stationary collection stopped: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
