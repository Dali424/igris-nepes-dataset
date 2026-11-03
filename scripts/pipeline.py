#!/usr/bin/env python3
"""Collect a seeded batch, then generate its requested augmentations."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
# Resource gate of scripts/seeded/collect_parallel.py; forwarded to collect.py only when given.
GATE_OPTIONS = (("--min-free-vram-gib", float), ("--kit-vram-gib", float), ("--min-available-ram-gib", float),
                ("--kit-ram-gib", float), ("--disk-reserve-gib-per-attempt", float),
                ("--launch-spacing-seconds", float), ("--warmup-seconds", float), ("--gpu-index", int),
                ("--max-idle-wait-seconds", float), ("--foreign-kit", str))


def run_stage(command):
    """Forward a stop to the owned stage, which preserves its own children/data."""
    child = subprocess.Popen(command, start_new_session=True)
    stopped = [None]

    def stop(signum, _frame):
        stopped[0] = stopped[0] or signum
        if child.poll() is None:
            try:
                child.send_signal(signum)
            except ProcessLookupError:
                pass

    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        code = child.wait()
        return 128 + stopped[0] if stopped[0] else code
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def commands(args):
    output = args.output.expanduser().absolute()
    collect = [sys.executable, "-B", str(ROOT / "scripts/collect.py"),
               "--per-pair", str(args.per_pair), "--first-seed", str(args.first_seed),
               "--output", str(output / "collection")]
    if args.max_attempts_per_pair is not None:
        collect += ["--max-attempts-per-pair", str(args.max_attempts_per_pair)]
    if getattr(args, "parallel", None) is not None:
        collect += ["--parallel", str(args.parallel)]
    for flag, _ in GATE_OPTIONS:
        if (value := getattr(args, flag[2:].replace("-", "_"), None)) is not None:
            collect += [flag, str(value)]
    augment = [sys.executable, "-B", str(ROOT / "scripts/augment.py"),
               "--source-batch", str(output / "collection"),
               "--output", str(output / "augmentation"), "--mode", args.augmentation_mode,
               "--workers", str(args.workers),
               "--max-attempts-per-parent", str(args.max_attempts_per_parent),
               "--video-presets", *args.video_presets]
    if args.augmentation_episodes is not None:
        augment += ["--episodes", str(args.augmentation_episodes)]
    for option in ("isaac_python", "dataset_python"):
        if value := getattr(args, option):
            flag = "--" + option.replace("_", "-")
            collect += [flag, value]
            augment += [flag, value]
    if args.resume:
        if (output / "collection/batch.json").exists():
            collect.append("--resume")
            if getattr(args, "interrupt_unfinished", False):
                collect.append("--interrupt-unfinished")
        if (output / "augmentation").exists():
            augment.append("--resume")
    return collect, augment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New run directory outside the release, e.g. ../runs/pc01")
    parser.add_argument("--per-pair", type=int, default=25)
    parser.add_argument("--first-seed", type=int, default=1000)
    parser.add_argument("--max-attempts-per-pair", type=int)
    parser.add_argument("--augmentation-mode", choices=("video", "render", "both"), default="video")
    parser.add_argument("--augmentation-episodes", type=int,
                        help="Parent count per video preset and/or render branch; defaults to all parents")
    parser.add_argument("--video-presets", nargs="+", choices=("darker", "brighter"), default=["darker"])
    parser.add_argument("--max-attempts-per-parent", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2, help="CPU video augmentation workers")
    parser.add_argument("--isaac-python", default=os.environ.get("NEPES_SIM_PYTHON"))
    parser.add_argument("--dataset-python", default=os.environ.get("NEPES_DATASET_PYTHON"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print both commands without executing")
    parser.add_argument("--parallel", type=int,
                        help="Concurrent Isaac attempts during collection: 1 (default), 2 or 3; a resumed batch keeps its value")
    for flag, kind in GATE_OPTIONS:
        if flag == "--foreign-kit":
            parser.add_argument(flag, choices=("wait", "fail"), help="--parallel only; forwarded to collect.py")
        else:
            parser.add_argument(flag, type=kind, help="--parallel only; forwarded to collect.py")
    parser.add_argument("--interrupt-unfinished", action="store_true",
                        help="--parallel batch with --resume only; forwarded to collect.py --resume")
    args = parser.parse_args()
    if args.per_pair < 1 or args.first_seed < 0 or args.workers < 1 or args.max_attempts_per_parent < 1:
        parser.error("Counts must be positive and first seed must be nonnegative")
    if args.max_attempts_per_pair is not None and args.max_attempts_per_pair < args.per_pair:
        parser.error("Attempt budget must be at least the per-pair quota")
    if args.augmentation_episodes is not None and not 1 <= args.augmentation_episodes <= 2 * args.per_pair:
        parser.error("Augmentation parent count must be between 1 and the original episode count")
    if len(set(args.video_presets)) != len(args.video_presets):
        parser.error("Video presets must be unique")
    if args.parallel is not None and not 1 <= args.parallel <= 3:
        parser.error("--parallel must be 1, 2 or 3")
    if args.interrupt_unfinished and not args.resume:
        parser.error("--interrupt-unfinished requires --resume")
    ik_python = os.environ.get("NEPES_IK_PYTHON", "").strip()
    if ik_python and not os.path.isabs(os.path.expanduser(ik_python)):  # see scripts/collect.py relative_ik_python()
        parser.error(f"NEPES_IK_PYTHON must be an absolute path (env.sh exports one), not {ik_python!r}: Isaac attempts "
                     "start in the release folder, where a relative path can name another interpreter than the one "
                     "checked and frozen")
    output = args.output.expanduser().resolve()
    if output.is_relative_to(ROOT) or ROOT.is_relative_to(output):
        parser.error("Choose an output outside the release and not an ancestor of it")
    collect, augment = commands(args)
    parent_count = args.augmentation_episodes or args.per_pair * 2
    video_count = parent_count * len(args.video_presets) if args.augmentation_mode in ("video", "both") else 0
    render_count = parent_count if args.augmentation_mode in ("render", "both") else 0
    print(json.dumps({"original_episodes": 2 * args.per_pair,
                      "video_augmented_episodes": video_count,
                      "render_augmented_episodes": render_count,
                      "commands": {"collect": shlex.join(collect), "augment": shlex.join(augment)}},
                     indent=2), flush=True)
    if args.dry_run:
        return 0
    if output.exists() and not args.resume:
        parser.error("Output exists; use --resume or a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    batch = output / "collection/batch.json"
    if batch.exists() and json.loads(batch.read_text()).get("status") == "complete":
        plan = json.loads(batch.read_text())["plan"]
        expected = {"quota_each": args.per_pair, "first_seed": args.first_seed,
                    "max_attempts_each": args.max_attempts_per_pair or 3 * args.per_pair}
        if any(plan.get(key) != value for key, value in expected.items()):
            parser.error("Completed collection has different quota/seed/attempt settings")
        if Path(plan["bundle"]).resolve() != ROOT:
            parser.error("Completed collection belongs to another release location")
        print("Collection complete; augmentation will verify the source before processing.", flush=True)
    else:
        code = run_stage(collect)
        if code:
            return code
    return run_stage(augment)


if __name__ == "__main__":
    raise SystemExit(main())
