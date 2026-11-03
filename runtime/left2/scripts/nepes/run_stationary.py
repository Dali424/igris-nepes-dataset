#!/usr/bin/env python3
"""Run the independently initialized NEPES tasks using the user's numbering.

Catalog v2 (default) is the corrected five-task list; catalog v1 keeps the
earlier component profiles (task 1 lift-and-hold, task 2 held-start loading).
An explicit --config profile selects its own catalog unless --catalog is given.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source/robros_motion_planning"), str(Path(__file__).parent)]


def main(argv=None):
    from robros_motion_planning.stationary_catalog import (
        CATALOGS, DEFAULT_CATALOG, default_stationary_config, load_stationary_config,
        resolve_stationary_task, stationary_max_task_seconds, stationary_spec_catalog, task_catalog,
        validate_stationary_spec)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="1", help="Task number 1..5 or its stable task ID in the selected catalog")
    parser.add_argument("--catalog", choices=tuple(CATALOGS), default=None,
                        help="v2: corrected five tasks; v1: legacy component profiles. Default: the --config "
                             f"profile's own catalog, else {DEFAULT_CATALOG}")
    parser.add_argument("--list", action="store_true", help="List the five independent task definitions of the catalog")
    parser.add_argument("--config", type=Path, help="Override the selected task's independent scene profile")
    parser.add_argument("--mode", choices=("inspect", "probe", "execute"), default="inspect")
    diagnostics = parser.add_mutually_exclusive_group()
    diagnostics.add_argument("--diagnostic_stop_after", metavar="STAGE")
    diagnostics.add_argument("--diagnostic_stop_after_waypoint", metavar="STAGE:INDEX",
                            help="Stop after measured arrival at a non-final lift/transfer waypoint; diagnostic only")
    args, forwarded = parser.parse_known_args(argv)
    if (args.diagnostic_stop_after is not None or args.diagnostic_stop_after_waypoint is not None) and args.mode != "execute":
        parser.error("Diagnostic stops require execute mode")
    config = None
    try:
        catalog = args.catalog
        if catalog is None and args.config is not None:
            # A named profile carries its catalog: old v1 command lines keep their meaning.
            config = load_stationary_config(args.config)
            catalog = stationary_spec_catalog(config.get("stationary_task"))
        catalog = catalog or DEFAULT_CATALOG
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    if args.list:
        print(json.dumps([dict(item) for item in task_catalog(catalog)], ensure_ascii=False, indent=2))
        return 0
    protected = {"--tasks", "--collection_scope", "--sequence", "--initialize_arm_at_pregrasp"}
    if any(token.split("=", 1)[0] in protected for token in forwarded):
        parser.error("Use --task with this independent task runner; scope and boundary initialization are fixed")
    try:
        task = resolve_stationary_task(args.task, catalog)
        if not task["configured"]:
            raise ValueError(f"Task {task['number']} ({task['id']}) is catalogued but has no calibrated execution configuration yet")
        if args.config is None:
            args.config = default_stationary_config(task["number"], catalog)
        if config is None:
            config = load_stationary_config(args.config)
        spec = validate_stationary_spec(config)
        if stationary_spec_catalog(spec) != catalog:
            raise ValueError(f"The profile is a catalog-{stationary_spec_catalog(spec)} configuration, "
                             f"but --catalog {catalog} was selected")
        if spec["id"] != task["id"]:
            raise ValueError(f"Catalog-{catalog} task {task['number']} is {task['id']}, but the profile configures "
                             f"{spec['id']} (task {spec['number']})")
        from robros_motion_planning.upright_runtime import (
            validate_diagnostic_waypoint_stop, validate_required_diagnostic_waypoint_stop)
        settings = config["tasks"][spec["legacy_task_id"]]
        validate_diagnostic_waypoint_stop(args.diagnostic_stop_after_waypoint, settings)
        validate_required_diagnostic_waypoint_stop(settings, args.diagnostic_stop_after_waypoint,
                                                  execute=args.mode == "execute")
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    launch = ["--mode", args.mode, "--config", str(args.config),
              "--tasks", spec["legacy_task_id"], "--collection_scope", "stationary_task",
              "--approach", "servo", *forwarded]
    for flag in ("diagnostic_stop_after", "diagnostic_stop_after_waypoint"):
        if getattr(args, flag) is not None:
            launch += ["--" + flag, getattr(args, flag)]
    # Only catalog-v1 profiles pre-pose the active arm at the episode boundary;
    # the v2 upright start records the whole reach and must never be pre-posed.
    if args.mode == "execute" and catalog == "v1":
        launch.append("--initialize_arm_at_pregrasp")
    # A v2 attempt runs until its own profile timeout, not run_tasks' default cap.
    cap = stationary_max_task_seconds(config)
    if (args.mode == "execute" and cap is not None
            and not any(token.split("=", 1)[0] == "--max_task_seconds" for token in forwarded)):
        launch += ["--max_task_seconds", str(cap)]
    from run_tasks import main as run_main
    return run_main(launch)


if __name__ == "__main__":
    raise SystemExit(main())
