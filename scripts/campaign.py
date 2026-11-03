#!/usr/bin/env python3
"""Resume verified originals to a total quota, then produce two visual datasets.

Historical evidence keeps its original schema and verifier. Only missing base
episodes are collected. Augmentation parents come from the completed new batch,
with one photometric and one freshly simulated render derivative per parent.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/seeded"))
import collect_seeded as helpers

PROFILES = ("left2", "right4")
# Resource gate of scripts/seeded/collect_parallel.py: runtime options (never frozen), parallel campaigns only.
GATE_OPTIONS = (("--min-free-vram-gib", float), ("--kit-vram-gib", float), ("--min-available-ram-gib", float),
                ("--kit-ram-gib", float), ("--disk-reserve-gib-per-attempt", float),
                ("--launch-spacing-seconds", float), ("--warmup-seconds", float), ("--gpu-index", int),
                ("--max-idle-wait-seconds", float), ("--foreign-kit", str))


def imported_counts(document):
    rows = document["imports"]
    helpers.require(all(row["profile"] in PROFILES for row in rows), "Unknown imported profile")
    paths = [str(Path(row["attempt"]).resolve()) for row in rows]
    helpers.require(len(paths) == len(set(paths)), "Duplicate imported attempt")
    return {name: sum(row["profile"] == name for row in rows) for name in PROFILES}


def build_plan(args):
    imports = args.imports.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    helpers.require(not output.is_relative_to(ROOT) and not ROOT.is_relative_to(output),
                    "Output must be outside the immutable release")
    helpers.require(not imports.is_relative_to(output), "Import ledger must be outside run output")
    ledger = helpers.read(imports)
    old = imported_counts(ledger)
    for row in ledger["imports"]:
        # All historical datasets and their provenance remain immutable, even
        # when a caller chooses an otherwise nonexistent child output folder.
        for key in ("attempt", "archive", "bundle", "batch"):
            if key not in row:
                continue  # The preflight verifier checks the full ledger schema.
            protected = Path(row[key]).expanduser().resolve()
            if key == "batch" and protected.suffix == ".json":
                protected = protected.parent
            helpers.require(not output.is_relative_to(protected) and not protected.is_relative_to(output),
                            "Output overlaps imported source: " + str(protected))
    missing = {name: args.base_per_pair - old[name] for name in PROFILES}
    helpers.require(args.base_per_pair > 0 and all(value > 0 for value in missing.values()),
                    "This campaign requires a positive collection deficit for both profiles")
    helpers.require(args.augmentation_episodes > 0 and args.augmentation_episodes % 2 == 0,
                    "Use an even augmentation quota for equal arm counts")
    helpers.require(args.augmentation_episodes // 2 <= min(missing.values()),
                    "Insufficient new parents to balance augmentation arms")
    budget = args.max_attempts_per_pair or 5 * max(missing.values())
    helpers.require(budget >= max(missing.values()), "Attempt budget is below collection deficit")
    helpers.require(0 <= args.first_seed <= 2_147_483_647 - budget + 1, "Invalid seed range")
    helpers.require(1 <= args.workers <= 4 and args.max_attempts_per_parent > 0, "Invalid worker/attempt budget")
    interpreters = {}
    for name in ("isaac_python", "dataset_python"):
        value = getattr(args, name)
        helpers.require(bool(value), "Set --" + name.replace("_", "-"))
        path = Path(value).expanduser().absolute()
        helpers.require(path.is_file() and os.access(path, os.X_OK), "Python executable missing: " + str(path))
        interpreters[name] = str(path)
    slots = getattr(args, "parallel", None)
    slots = 1 if slots is None else slots
    helpers.require(type(slots) is int and 1 <= slots <= 3, "Use --parallel 1, 2 or 3")
    plan = {"schema": "nepes_quota_campaign_plan_v1", "bundle": str(ROOT),
            "bundle_manifest_sha256": helpers.verify_bundle(ROOT),
            "imports": str(imports), "imports_sha256": helpers.sha(imports), "output": str(output),
            "imported_by_profile": old, "new_base_by_profile": missing,
            "target_base_by_profile": {name: args.base_per_pair for name in PROFILES},
            "target_base": 2 * args.base_per_pair, "target_brightness": args.augmentation_episodes,
            "target_render": args.augmentation_episodes, "first_seed": args.first_seed,
            "max_attempts_per_pair": budget, "max_attempts_per_parent": args.max_attempts_per_parent,
            "video_preset": args.video_preset, "workers": args.workers, **interpreters,
            "augmentation_parent_policy": "balanced distinct parents from new_base; same parent split_group across base and derivatives",
            "existing_augmentation_policy": "prior smoke derivatives remain separate; requested quotas use the selected new campaign parents"}
    if slots > 1:
        plan["parallel"] = slots  # serial campaigns keep exactly the previous plan
        # The same profile and seed resample the same start, so new seeds must not repeat an imported one.
        last = args.first_seed + budget - 1
        for name in PROFILES:
            reused = sorted({row["seed"] for row in ledger["imports"] if row.get("profile") == name
                             and type(row.get("seed")) is int and args.first_seed <= row["seed"] <= last})
            helpers.require(not reused, f"New seeds {args.first_seed}..{last} repeat imported {name} seeds "
                            f"{reused[:6]}{'...' if len(reused) > 6 else ''}; choose --first-seed above every imported seed "
                            "(import_candidates.py prints suggested_first_seed). Only imported seeds are checked here: "
                            "also keep clear of seeds that a paused source campaign may still use")
    return plan


def commands(plan):
    root = Path(plan["output"])
    python = plan["dataset_python"]
    helper = str(ROOT / "scripts/campaign_sources.py")
    common = ["--bundle", str(ROOT), "--imports", plan["imports"]]
    interpreters = ["--isaac-python", plan["isaac_python"], "--dataset-python", python]
    collect_base = [python, "-B", str(ROOT / "scripts/collect.py"),
                    "--left-successes", str(plan["new_base_by_profile"]["left2"]),
                    "--right-successes", str(plan["new_base_by_profile"]["right4"]),
                    "--first-seed", str(plan["first_seed"]),
                    "--max-attempts-per-pair", str(plan["max_attempts_per_pair"]),
                    "--output", str(root / "new_base"), *interpreters]
    if plan.get("parallel", 1) > 1:
        collect_base += ["--parallel", str(plan["parallel"])]
    return [
        ("verify_imports", [python, "-B", helper, "preflight-import", *common,
                            "--output", str(root / "import_verification.json")]),
        ("collect_base", collect_base),
        ("merge_base", [python, "-B", helper, "merge", *common,
                       "--new-batch", str(root / "new_base"), "--output", str(root / ("basic_" + str(plan["target_base"]))),
                       "--expected-per-pair", str(plan["target_base_by_profile"]["left2"]),
                       "--audit", str(root / "merge_verification.json")]),
        ("augment", [python, "-B", str(ROOT / "scripts/augment.py"),
                     "--source-batch", str(root / "new_base"), "--output", str(root / "augmentation"),
                     "--mode", "both", "--episodes", str(plan["target_render"]),
                     "--video-presets", plan["video_preset"], "--workers", str(plan["workers"]),
                     "--max-attempts-per-parent", str(plan["max_attempts_per_parent"]), *interpreters]),
    ]


def progress(plan):
    root = Path(plan["output"])
    accepted = {name: 0 for name in PROFILES}
    attempts = {name: 0 for name in PROFILES}
    batch_path = root / "new_base/batch.json"
    active = None
    if batch_path.exists():
        batch = helpers.read(batch_path)
        for record in batch.get("attempts", []):
            attempts[record["profile"]] += 1
            accepted[record["profile"]] += record.get("status") == "accepted"
        active = batch.get("active")
    video = 0
    video_state = root / "augmentation/video/state.json"
    if video_state.exists():
        document = helpers.read(video_state)
        if document.get("status") == "complete":
            video = document.get("total_augmented_episodes", 0)
    render_state = root / "augmentation/render/state.json"
    render = 0
    if render_state.exists():
        render = sum(job.get("status") == "accepted" for job in helpers.read(render_state).get("jobs", []))
    totals = {name: plan["imported_by_profile"][name] + accepted[name] for name in PROFILES}
    return {"base": sum(totals.values()), "base_by_profile": totals,
            "new_successes": accepted, "new_attempts": attempts, "brightness": video, "render": render,
            "targets": {"base": plan["target_base"], "brightness": plan["target_brightness"], "render": plan["target_render"]},
            "active_attempt": active}


def validate_new_batch_plan(plan, batch):
    source = batch["plan"]
    quotas = source.get("quota_by_profile", {name: source.get("quota_each") for name in PROFILES})
    budgets = source.get("max_attempts_by_profile", {name: source.get("max_attempts_each") for name in PROFILES})
    helpers.require(quotas == plan["new_base_by_profile"]
                    and budgets == {name: plan["max_attempts_per_pair"] for name in PROFILES}
                    and source.get("first_seed") == plan["first_seed"],
                    "New base batch quota/seed plan differs from campaign")
    helpers.require(Path(source.get("bundle", "")).resolve() == Path(plan["bundle"])
                    and Path(source.get("output", "")).resolve() == Path(plan["output"]) / "new_base"
                    and source.get("bundle_manifest_sha256") == plan["bundle_manifest_sha256"],
                    "New base batch source/output binding differs")
    for key in ("isaac_python", "dataset_python"):
        helpers.require(source.get(key) == plan[key], "New base interpreter differs: " + key)
    if plan.get("parallel", 1) > 1:
        parallel = source.get("parallel")
        helpers.require(source.get("ordering") == "round_robin_parallel_v1" and isinstance(parallel, dict)
                        and parallel.get("slots") == plan["parallel"], "New base batch concurrency differs from campaign")


def validate_completion(plan):
    """Require published, verified artifacts and exact arm quotas after workers.

    Workers perform full payload decoding and admission checks. This controller
    additionally binds their terminal states and receipts to the expected output
    paths, so an exit code or a partially populated progress counter is never
    sufficient to publish campaign completion.
    """
    root = Path(plan["output"])
    totals = progress(plan)
    helpers.require(totals["base_by_profile"] == plan["target_base_by_profile"]
                    and totals["new_successes"] == plan["new_base_by_profile"],
                    "Final base profile quotas are incomplete")
    helpers.require(all(totals[key] == plan["target_" + key] for key in ("base", "brightness", "render")),
                    "Final campaign quotas are incomplete")
    batch = helpers.read(root / "new_base/batch.json")
    validate_new_batch_plan(plan, batch)
    helpers.require(batch.get("schema") == "nepes_seeded_batch_v1" and batch.get("status") == "complete"
                    and batch.get("counts") == plan["new_base_by_profile"], "New base batch is not complete")
    expected_paths = {
        "basic": root / ("basic_" + str(plan["target_base"])),
        "brightness": root / "augmentation/video" / ("visible_" + plan["video_preset"] + "_v2"),
        "lighting_material": root / "augmentation/render" / ("dataset_" + str(plan["target_render"]))}
    augmented = helpers.read(root / "augmentation/state.json")
    helpers.require(augmented.get("schema") == "nepes_portable_augmentation_batch_v1"
                    and augmented.get("status") == "complete", "Augmentation supervisor is not complete")
    video = helpers.read(root / "augmentation/video/state.json")
    helpers.require(video.get("schema") == "nepes_seeded_video_batch_v1" and video.get("status") == "complete"
                    and video.get("total_augmented_episodes") == plan["target_brightness"],
                    "Video batch is not complete")
    results = video.get("results", [])
    helpers.require(len(results) == 1 and augmented.get("results", {}).get("video") == results,
                    "Video result binding differs")
    result = results[0]
    helpers.require(result.get("verified") is True and result.get("episodes") == plan["target_brightness"]
                    and result.get("preset") == "visible_" + plan["video_preset"] + "_v2"
                    and Path(result.get("dataset", "")).resolve() == expected_paths["brightness"],
                    "Video result is not the requested verified dataset")
    audit_path = expected_paths["brightness"] / "meta/video_augmentation_audit.json"
    audit = helpers.read(audit_path)
    helpers.require(helpers.sha(audit_path) == result.get("audit_sha256")
                    and audit.get("schema") == "nepes_seeded_video_dataset_audit_v1"
                    and audit.get("verified") is True and audit.get("episodes") == plan["target_brightness"],
                    "Video verification receipt differs")
    render = helpers.read(root / "augmentation/render/state.json")
    balanced = {profile: plan["target_render"] // 2 for profile in PROFILES}
    jobs = render.get("jobs", [])
    job_counts = {profile: sum(job.get("status") == "accepted" and job.get("profile") == profile
                               for job in jobs) for profile in PROFILES}
    helpers.require(render.get("schema") == "nepes_native_render_batch_v1" and render.get("status") == "complete"
                    and render.get("accepted") == plan["target_render"] and render.get("counts") == balanced
                    and len(jobs) == plan["target_render"] and job_counts == balanced,
                    "Native render profile quotas are incomplete")
    helpers.require(Path(render.get("dataset", "")).resolve() == expected_paths["lighting_material"],
                    "Native render dataset path differs")
    render_result = augmented.get("results", {}).get("render", {})
    helpers.require(render_result.get("episodes") == plan["target_render"]
                    and render_result.get("new_physics_execution") is True
                    and Path(render_result.get("dataset", "")).resolve() == expected_paths["lighting_material"],
                    "Native render result binding differs")
    for name, path in expected_paths.items():
        expected = plan["target_base" if name == "basic" else "target_brightness" if name == "brightness" else "target_render"]
        helpers.require(helpers.read(path / "meta/info.json")["total_episodes"] == expected,
                        "Published dataset quota differs: " + name)
        if name != "brightness":
            receipt = helpers.read(path / "meta/nepes_storage_verification.json")
            helpers.require(receipt.get("schema") == "nepes_verified_lerobot_storage_v1"
                            and receipt.get("verified") is True and receipt.get("kind") == "merged_dataset"
                            and receipt.get("total_episodes") == expected,
                            "Published dataset is not verified: " + name)
    return totals, {name: str(path) for name, path in expected_paths.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imports", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-per-pair", type=int, default=100)
    parser.add_argument("--augmentation-episodes", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=10000)
    parser.add_argument("--max-attempts-per-pair", type=int)
    parser.add_argument("--max-attempts-per-parent", type=int, default=5)
    parser.add_argument("--video-preset", choices=("darker", "brighter"), default="darker")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--isaac-python", default=os.environ.get("NEPES_SIM_PYTHON"))
    parser.add_argument("--dataset-python", default=os.environ.get("NEPES_DATASET_PYTHON", sys.executable))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--stop-after-new-successes", type=int,
                        help="Pause for inspection before merge/augmentation; omitted from frozen plan")
    parser.add_argument("--parallel", type=int,
                        help="Concurrent Isaac attempts while collecting new_base: 1 (default), 2 or 3; frozen in the plan")
    for flag, kind in GATE_OPTIONS:
        if flag == "--foreign-kit":
            parser.add_argument(flag, choices=("wait", "fail"), help="--parallel only; forwarded to collect_base")
        else:
            parser.add_argument(flag, type=kind, help="--parallel only; forwarded to collect_base, never frozen")
    parser.add_argument("--interrupt-unfinished", action="store_true",
                        help="--parallel --resume only: collect_base records attempts whose native never reported an exit "
                             "(e.g. a PC restart) as interrupted; never frozen")
    args = parser.parse_args(argv)
    root = args.output.expanduser().resolve()
    state_path = root / "campaign.json"
    if args.status:
        state = helpers.read(state_path)
        print(json.dumps({"status": state["status"], "updated_utc": state["updated_utc"],
                          "stage": state.get("active"), "error": state.get("error"),
                          "progress": progress(state["plan"]), "datasets": state.get("datasets")}, indent=2))
        return 0
    if args.imports is None:
        parser.error("--imports is required except for --status")
    ik_python = os.environ.get("NEPES_IK_PYTHON", "").strip()
    if ik_python and not os.path.isabs(os.path.expanduser(ik_python)):  # see scripts/collect.py relative_ik_python()
        parser.error(f"NEPES_IK_PYTHON must be an absolute path (env.sh exports one), not {ik_python!r}: Isaac attempts "
                     "start in the release folder, where a relative path can name another interpreter than the one "
                     "checked and frozen")
    if args.stop_after_new_successes is not None and args.stop_after_new_successes < 1:
        parser.error("Checkpoint must be positive")
    if args.parallel is not None and not 1 <= args.parallel <= 3:
        parser.error("--parallel must be 1, 2 or 3")
    if args.parallel is None and args.resume and state_path.is_file():
        try:  # a resumed campaign keeps its frozen concurrency
            args.parallel = helpers.read(state_path)["plan"].get("parallel")
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass
    gate = [value for flag, _ in GATE_OPTIONS if getattr(args, flag[2:].replace("-", "_")) is not None
            for value in (flag, str(getattr(args, flag[2:].replace("-", "_"))))]
    if gate and (args.parallel or 1) == 1:
        parser.error("Resource gate options apply only to --parallel 2 or 3")
    if args.interrupt_unfinished and ((args.parallel or 1) == 1 or not args.resume):
        parser.error("--interrupt-unfinished applies only to --resume of a --parallel campaign")
    plan = build_plan(args)
    stages = commands(plan)
    parallel = plan.get("parallel", 1) > 1
    if args.dry_run:
        print(json.dumps({"writes_performed": False, "plan": plan, "commands": dict(stages)}, indent=2))
        return 0
    helpers.require(args.resume or not root.exists(), "Output exists; use --resume")
    helpers.require(not args.resume or state_path.is_file(), "No campaign to resume")
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / ".campaign.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            state = helpers.read(state_path)
            helpers.require(state["plan"] == plan, "Campaign plan/source/code changed")
            helpers.require(not helpers.alive(state.get("owned_processes", {})), "Prior campaign child remains active")
        else:
            state = {"schema": "nepes_quota_campaign_v1", "plan": plan, "status": "prepared",
                     "created_utc": helpers.now(), "completed_stages": []}
        if args.resume and (root / "STOP").exists():
            (root / "STOP").rename(root / ("STOP_resumed_" + str(time.time_ns())))
        if parallel and args.resume and (root / "DRAIN").exists():
            (root / "DRAIN").rename(root / ("DRAIN_resumed_" + str(time.time_ns())))
    except BaseException:
        lock.close()
        raise
    stop, owned = [False], {}
    handlers = {sig: signal.signal(sig, lambda *_: stop.__setitem__(0, True))
                for sig in (signal.SIGINT, signal.SIGTERM)}

    def save():
        state.update(updated_utc=helpers.now(), pid=os.getpid(), owned_processes=helpers.alive(owned), progress=progress(plan))
        helpers.write(state_path, state)

    def frozen():
        helpers.require(helpers.sha(plan["imports"]) == plan["imports_sha256"], "Import ledger changed")
        helpers.require(helpers.verify_bundle(ROOT) == plan["bundle_manifest_sha256"], "Campaign release changed")

    def run(label, command):
        frozen()
        if stop[0] or (root / "STOP").exists():
            raise InterruptedError("Campaign STOP requested")
        if parallel and (root / "DRAIN").exists():
            raise InterruptedError("Campaign DRAIN requested; completed stages preserved")
        state.update(status="running", active=label)
        state.pop("error", None)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        env.pop("PYTHONPATH", None)
        with (root / (label + ".log")).open("a") as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            owned.update(helpers.descendants(proc.pid))
            save()
            last, stopping = time.monotonic(), None
            while proc.poll() is None:
                owned.update(helpers.descendants(proc.pid))
                stop[0] |= (root / "STOP").exists()
                if (parallel and label == "collect_base" and (root / "DRAIN").exists()
                        and (root / "new_base").is_dir() and not (root / "new_base/DRAIN").exists()):
                    # Boundary stop: the collector finishes in-flight attempts and starts no new one. Re-created
                    # while missing, because a resuming collector renames the DRAIN that ended its last session.
                    (root / "new_base/DRAIN").touch(exist_ok=True)
                if stop[0]:
                    if stopping is None:
                        stopping = time.monotonic()
                        proc.send_signal(signal.SIGINT)
                    if time.monotonic() - stopping > 200:
                        helpers.send(owned, signal.SIGKILL)
                    elif time.monotonic() - stopping > 185:
                        helpers.send(owned, signal.SIGTERM)
                if time.monotonic() - last >= 30:
                    save(); last = time.monotonic()
                    print(json.dumps({"stage": label, **state["progress"]}), flush=True)
                try:
                    proc.wait(timeout=.5)
                except subprocess.TimeoutExpired:
                    pass
        helpers.require(not helpers.alive(owned), "Owned child survived stage completion")
        state["active"] = None
        save()
        if stop[0] or proc.returncode == 130:
            raise InterruptedError("Campaign stopped; completed records preserved")
        helpers.require(proc.returncode == 0, f"{label} failed; inspect {root / (label + '.log')}")

    try:
        save()
        for label, command in stages:
            if label == "collect_base":
                existing = root / "new_base/batch.json"
                if existing.exists():
                    batch = helpers.read(existing)
                    validate_new_batch_plan(plan, batch)
                    if batch.get("status") == "complete":
                        # The following merge and augmentation reverify all source payloads.
                        continue
                    command.append("--resume")
                    if args.interrupt_unfinished:
                        command.append("--interrupt-unfinished")
                if args.stop_after_new_successes is not None:
                    command += ["--stop-after-total-successes", str(args.stop_after_new_successes)]
                command += gate
            if label == "augment" and (root / "augmentation/state.json").exists():
                command.append("--resume")
            if label == "merge_base" and (root / ("basic_" + str(plan["target_base"]))).exists():
                command.append("--verify-existing")
            run(label, command)
            if label == "collect_base" and helpers.read(root / "new_base/batch.json")["status"] == "checkpoint":
                state.update(status="checkpoint", active=None); save()
                return 0
            if label not in state["completed_stages"]:
                state["completed_stages"].append(label)
            save()
        frozen()
        totals, datasets = validate_completion(plan)
        state.update(status="complete", active=None, finished_utc=helpers.now(),
                     datasets=datasets)
        state.pop("error", None)
        helpers.write(root / "datasets.json", {"schema": "nepes_campaign_datasets_v1", "counts": totals,
                                               "datasets": state["datasets"], "split_policy": plan["augmentation_parent_policy"]})
        save(); print(json.dumps(state["datasets"], indent=2)); return 0
    except BaseException as exc:
        for pid in list(owned):
            if helpers.process(pid) == owned[pid]:
                owned.update(helpers.descendants(int(pid)))
        helpers.send(owned, signal.SIGINT)
        started = time.monotonic()
        while helpers.alive(owned) and time.monotonic() - started < 205:
            if time.monotonic() - started > 200: helpers.send(owned, signal.SIGKILL)
            elif time.monotonic() - started > 185: helpers.send(owned, signal.SIGTERM)
            time.sleep(.5)
        state.update(status="stopped" if isinstance(exc, InterruptedError) else "needs_review",
                     active=None, error=f"{type(exc).__name__}: {exc}")
        save()
        if isinstance(exc, InterruptedError): return 130
        raise
    finally:
        for sig, handler in handlers.items(): signal.signal(sig, handler)
        lock.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        raise SystemExit(f"Campaign stopped: {exc}") from exc
