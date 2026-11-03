"""Bounded, resumable full-task collection. Importing this module never loads Isaac.

Each attempt runs in a fresh child process. Original run artifacts are immutable;
only copied episode metadata receives a collection-unique identifier at export.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time
from typing import Callable

from .nepes_tasks import load_task_config
from .dataset import DEFAULT_CAMERA_SIZES, validate_camera_sizes, validate_declared_camera_layout

TASKS = ("crs_load_jig", "open_cleaner_cover", "jig_insert_cleaner",
         "jig_extract_cleaner", "crs_final_place")
PLAN_SCHEMA = "igris_c_nepes_collection_plan_v1"
STATE_SCHEMA = "igris_c_nepes_collection_v1"
# Retained for callers describing historical three-view recordings.
CAMERAS = {name: list(size) for name, size in DEFAULT_CAMERA_SIZES.items()}
VALUE_ARGS = {"approach", "scene_usd", "robot_model_dir", "checkpoint", "planner_python",
              "max_task_seconds", "settle_seconds", "simulate_wash_seconds", "events_file", "device"}
FLAG_ARGS = {"headless", "initialize_arm_at_pregrasp", "keep_failed_frames",
             "no_cover_parking", "no_head_gaze"}
PATH_ARGS = {"scene_usd", "robot_model_dir", "checkpoint", "planner_python", "events_file"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_digest(value):
    # Match task_runtime.py's configuration fingerprint exactly.
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def algorithm_fingerprints(runner):
    """Hash bounded Python source only; do not scan assets, outputs or checkpoints."""
    package = Path(__file__).resolve().parent
    hashes = {"package/" + str(p.relative_to(package)): digest(p)
              for p in sorted(package.rglob("*.py")) if "__pycache__" not in p.parts}
    hashes["scripts/nepes/run_tasks.py"] = digest(Path(runner[-1]))
    return hashes


def load_plan(path):
    """Read a strict five-task plan, resolving data paths against its directory."""
    path = Path(path).resolve()
    plan = read_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA or set(plan.get("tasks", {})) != set(TASKS):
        raise ValueError("Plan must name all five NEPES tasks with the supported schema")
    plan.setdefault("seed", 0)
    if type(plan["seed"]) is not int or plan["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    for key, default in (("repeated_failure_limit", 2), ("calibration_attempt_limit", 2)):
        _positive_int(plan.setdefault(key, default), key)
    plan.setdefault("max_wall_seconds", 3600.)
    if not math.isfinite(plan["max_wall_seconds"]) or plan["max_wall_seconds"] <= 0:
        raise ValueError("max_wall_seconds must be finite and positive")
    runner = plan.get("runner")
    if not isinstance(runner, list) or not runner or not all(isinstance(s, str) and s for s in runner):
        raise ValueError("runner must be an argv prefix ending in run_tasks.py")
    if not runner[-1].endswith("run_tasks.py"):
        raise ValueError("runner must end with the existing run_tasks.py script")
    plan["runner"] = [str((path.parent / s).resolve()) if "/" in s or s.endswith(".py") else s for s in runner]
    plan["algorithm_sha256"] = algorithm_fingerprints(plan["runner"])
    options = plan.setdefault("runner_options", {})
    for key, value in {"headless": True, "approach": "servo", "initialize_arm_at_pregrasp": True, "device": "cpu"}.items():
        options.setdefault(key, value)
    options.setdefault("max_task_seconds", 420.)
    if set(options) - VALUE_ARGS - FLAG_ARGS:
        raise ValueError(f"Unsupported runner_options: {sorted(set(options) - VALUE_ARGS - FLAG_ARGS)}")
    if any(type(options[k]) is not bool for k in FLAG_ARGS & options.keys()):
        raise ValueError("Runner flags must be explicit booleans")
    if options.get("approach", "curobo") not in ("curobo", "servo"):
        raise ValueError("approach must be curobo or servo")
    for key in PATH_ARGS & options.keys():
        options[key] = str((path.parent / options[key]).resolve())
    for task_id in TASKS:
        task = plan["tasks"][task_id]
        _positive_int(task["quota"], f"{task_id}.quota")
        _positive_int(task["max_attempts"], f"{task_id}.max_attempts")
        if task["max_attempts"] < task["quota"]:
            raise ValueError("max_attempts cannot be smaller than quota")
        task["config"] = str((path.parent / task["config"]).resolve())
        # The runner validates/normalizes before writing config.json and its
        # digest (for example integer aperture 1 becomes 1.0). Fingerprint that
        # same runtime configuration, retaining source bytes separately.
        config = load_task_config(task["config"])
        if task_id not in config.get("tasks", {}) or "task" not in config.get("allowed_collection_scopes", ["task"]):
            raise ValueError(f"Configuration does not support full task {task_id}")
        if task_id not in config.get("allowed_task_ids", TASKS):
            raise ValueError(f"Configuration excludes {task_id}")
        task["config_sha256"] = json_digest(config)
        task["config_source_sha256"] = digest(task["config"])
        readiness = task.setdefault("readiness", {"status": "unverified"})
        if readiness.get("status") not in ("unverified", "geometry_only", "validated", "blocked"):
            raise ValueError(f"Invalid readiness status for {task_id}")
        if readiness.get("evidence_run"):
            readiness["evidence_run"] = str((path.parent / readiness["evidence_run"]).resolve())
    return plan


def execution_digest(plan, task_id):
    return json_digest({"runner": plan["runner"], "runner_options": plan["runner_options"],
                        "algorithm_sha256": plan["algorithm_sha256"],
                        "config_sha256": plan["tasks"][task_id]["config_sha256"],
                        "task_id": task_id, "collection_scope": "task"})


def attempt_command(plan, task_id, output, seed):
    command = [*plan["runner"], "--mode", "execute", "--collection_scope", "task", "--record",
               "--tasks", task_id, "--cycles", "1", "--seed", str(seed),
               "--config", plan["tasks"][task_id]["config"], "--output", str(output)]
    for key, value in plan["runner_options"].items():
        if key in FLAG_ARGS:
            if value:
                command.append("--" + key)
        else:
            command.extend(("--" + key, str(value)))
    return command


def _full_result(value, task_id):
    return (value.get("success") is True and value.get("episode_scope") == "task"
            and value.get("full_task_completed") is True and value.get("task_id") == task_id)


def validate_episode(run_dir, task_id, config_sha256, episode_id=None):
    """Fail closed on partial primitives, stale hashes, missing RGB or outcomes.

    Returns source paths/fingerprints. It never changes or promotes source status.
    Native process exit is checked separately by the collector.
    """
    run_dir = Path(run_dir).resolve()
    report = read_json(run_dir / "run.json")
    if report.get("complete") is not True or report.get("episode_scope") != "task" or report.get("mode") != "execute":
        raise ValueError("incomplete_run_or_non_task_scope")
    config = read_json(run_dir / "config.json")
    if report.get("config_sha256") != config_sha256 or json_digest(config) != config_sha256:
        raise ValueError("config_hash_mismatch")
    results = [r for r in report.get("episodes", []) if r.get("task_id") == task_id
               and (episode_id is None or r.get("episode_id") == episode_id)]
    if len(results) != 1 or not _full_result(results[0], task_id):
        raise ValueError("full_task_success_not_verified")
    result = results[0]
    name = result["episode_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("unsafe_episode_id")
    runtime = run_dir / name
    if read_json(runtime / "result.json") != result:
        raise ValueError("run_and_episode_result_disagree")
    episode = run_dir / "staging" / "episodes" / name
    metadata = read_json(episode / "episode.json")
    if (metadata.get("episode_id") != name or metadata.get("status") != "success"
            or metadata.get("success") is not True or metadata.get("metrics") != result
            or metadata.get("provenance", {}).get("episode_scope") != "task"):
        raise ValueError("staging_is_not_explicit_full_task_success")
    schema = metadata["schema"]
    try:
        cameras = validate_camera_sizes(schema.get("camera_sizes"), require_standard_sizes=True)
        if "camera_layout" in config:
            validate_declared_camera_layout(cameras, config["camera_layout"])
    except ValueError as error:
        raise ValueError("recording_schema_mismatch") from error
    if (schema.get("schema_version") != "igris_c_nepes_episode_staging_v1"
            or schema.get("fps") != 30 or schema.get("physics_hz") != 250
            or schema.get("control_hz") != 50):
        raise ValueError("recording_schema_mismatch")
    if set(metadata.get("camera_configuration", {})) != set(cameras):
        raise ValueError("missing_camera_configuration")
    model_manifest = Path(report.get("robot_model_manifest", ""))
    if not model_manifest.is_file():
        raise ValueError("missing_robot_model_manifest")
    frames = episode / "frames.jsonl"
    if digest(frames) != metadata.get("frames_sha256"):
        raise ValueError("frame_hash_mismatch")
    count = 0
    previous = None
    image_digest = sha256()
    first_step = first_time = None
    for count, line in enumerate(frames.open(), 1):
        row = json.loads(line)
        if row["frame_index"] != count - 1 or abs(row["timestamp"] - (count - 1) / 30) > 1e-8:
            raise ValueError("recording_frame_sequence_mismatch")
        actual = row["sim_time_s"]
        if not math.isfinite(actual) or (previous is not None and actual <= previous):
            raise ValueError("recording_time_not_monotonic")
        previous = actual
        if first_step is None:
            first_step, first_time = row["physics_step"], actual
        expected_step = first_step + ((count - 1) * 250 + 29) // 30
        if row["physics_step"] != expected_step or abs(actual - first_time - (expected_step - first_step) / 250) > 1e-6:
            raise ValueError("recording_physics_tick_mismatch")
        age = actual - row["command_time_s"]
        if not math.isfinite(age) or not -1e-5 <= age <= .02001:
            raise ValueError("recording_command_time_mismatch")
        for field, names in (("joint_positions", "joint_names"), ("joint_velocities", "joint_names"),
                             ("policy_action", "policy_action_names"), ("upper_targets", "upper_target_names")):
            if len(row[field]) != len(schema[names]) or not all(math.isfinite(v) for v in row[field]):
                raise ValueError("recording_state_or_action_mismatch")
        if set(row["images"]) != set(cameras) or set(row["cameras"]) != set(cameras):
            raise ValueError("recording_camera_set_mismatch")
        for camera in cameras:
            relative = f"images/{camera}/{count - 1:06d}.png"
            image = episode / relative
            if (row["images"].get(camera) != relative or not image.is_file() or image.is_symlink()
                    or not image.resolve().is_relative_to(episode.resolve())):
                raise ValueError("missing_or_external_rgb")
            image_digest.update((relative + ":" + digest(image) + "\n").encode())
            rendered = row["cameras"][camera]["render_time_s"]
            if not math.isfinite(rendered) or abs(rendered - actual) > 1e-5:
                raise ValueError("stale_camera")
    if count < 1 or count != metadata.get("frame_count"):
        raise ValueError("empty_or_incomplete_recording")
    for name in ("trace.jsonl", "initialization.json"):
        if not (runtime / name).is_file() or not (runtime / name).stat().st_size:
            raise ValueError("missing_physical_evidence")
    return {"run_dir": str(run_dir), "episode_id": result["episode_id"], "task_id": task_id,
            "episode_dir": str(episode), "frame_count": count, "config_sha256": config_sha256,
            "frames_sha256": digest(frames), "result_sha256": digest(runtime / "result.json"),
            "trace_sha256": digest(runtime / "trace.jsonl"), "images_sha256": image_digest.hexdigest(),
            "episode_metadata_sha256": digest(episode / "episode.json"),
            "initialization_sha256": digest(runtime / "initialization.json"), "episode_scope": "task",
            "robot_model_manifest": str(model_manifest.resolve()),
            "robot_model_manifest_sha256": digest(model_manifest),
            "full_task_completed": True}


def readiness(plan, task_id, attempts):
    """A measured full completion of the same collection contract bootstraps readiness."""
    key = execution_digest(plan, task_id)
    for a in attempts:
        if a["task_id"] == task_id and a.get("accepted") and a.get("execution_sha256") == key:
            try:
                checked = validate_episode(a["run_dir"], task_id, plan["tasks"][task_id]["config_sha256"])
                if checked != a["evidence"]:
                    return False, "prior_success_evidence_changed", None
                return True, "measured_full_task_success", a["run_dir"]
            except (ValueError, KeyError, OSError):
                return False, "prior_success_evidence_changed", None
    spec = plan["tasks"][task_id]["readiness"]
    if spec.get("status") != "validated":
        return False, spec.get("status", "unverified"), None
    if spec.get("execution_sha256") != key:
        return False, "readiness_execution_hash_mismatch", None
    try:
        validate_episode(spec["evidence_run"], task_id, plan["tasks"][task_id]["config_sha256"],
                         spec.get("evidence_episode"))
    except (ValueError, KeyError, OSError) as exc:
        return False, f"readiness_evidence_invalid:{exc}", None
    return True, "measured_full_task_success", spec["evidence_run"]


def failure_signature(reason, stage=""):
    # Camera times, solver timings, or addresses must not defeat a stagnation cut.
    value = re.sub(r"0x[0-9a-fA-F]+|[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", "#", str(reason))
    return f"{stage}:{value}"[:500]


def summarize(plan, attempts, *, calibrate=False):
    tasks = {}
    for task_id, spec in plan["tasks"].items():
        history = [a for a in attempts if a["task_id"] == task_id]
        successes = sum(bool(a.get("accepted")) for a in history)
        ready, reason, evidence = readiness(plan, task_id, attempts)
        failures = Counter(a.get("failure_reason", "") for a in history if a.get("status") == "complete" and not a.get("accepted"))
        tail = []
        for a in reversed(history):
            if a.get("accepted") or a.get("status") != "complete":
                break
            tail.append(a.get("failure_signature"))
        repeated = len(tail) >= plan["repeated_failure_limit"] and len(set(tail[:plan["repeated_failure_limit"]])) == 1
        calibration_used = sum(a.get("calibration_attempt", False) for a in history)
        blocked = None
        if reason == "prior_success_evidence_changed":
            raise ValueError(f"Accepted source evidence changed: {task_id}")
        if successes >= spec["quota"]:
            blocked = "quota_reached"
        elif repeated:
            blocked = "repeated_identical_failure_requires_calibration_change"
        elif len(history) >= spec["max_attempts"]:
            blocked = "maximum_attempts_reached"
        elif not ready and not (calibrate and spec["readiness"]["status"] != "blocked"
                                and calibration_used < plan["calibration_attempt_limit"]):
            blocked = "readiness:" + reason
        tasks[task_id] = {"quota": spec["quota"], "successes": successes, "attempts": len(history),
                          "max_attempts": spec["max_attempts"], "ready": ready, "readiness_reason": reason,
                          "readiness_evidence_run": evidence, "calibration_attempts": calibration_used,
                          "blocked": blocked, "failure_reasons": dict(failures),
                          "execution_sha256": execution_digest(plan, task_id)}
    return {"tasks": tasks, "five_task_coverage": all(t["successes"] > 0 for t in tasks.values()),
            "quotas_complete": all(t["successes"] >= t["quota"] for t in tasks.values()),
            "full_task_only": True, "total_successes": sum(t["successes"] for t in tasks.values())}


def run_child(command, *, log_path, stop_path, run_dir, max_wall_seconds, on_start):
    """Forward STOP cooperatively, then bound a hung child/process group on timeout."""
    started = time.monotonic()
    stopping = None
    forced = False
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        on_start(child.pid)
        while child.poll() is None:
            try:
                if stop_path.exists() or time.monotonic() - started >= max_wall_seconds:
                    stopping = stopping or time.monotonic()
                    if run_dir.exists():
                        (run_dir / "STOP").touch(exist_ok=True)
                    if time.monotonic() - stopping > 60:
                        os.killpg(child.pid, signal.SIGTERM)
                        forced = True
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                        break
                child.wait(timeout=.25)
            except subprocess.TimeoutExpired:
                pass
            except KeyboardInterrupt:
                stop_path.touch(exist_ok=True)
        code = child.wait()
    return {"returncode": code, "wall_seconds": time.monotonic() - started,
            "timed_out": time.monotonic() - started >= max_wall_seconds,
            "forced_termination": forced, "stop_requested": stop_path.exists()}


def collect(plan, output, *, execute=False, resume=False, calibrate=False, launcher: Callable = run_child):
    """Plan without side effects, or sequentially collect bounded full-task attempts."""
    output = Path(output).resolve()
    state_path = output / "collection.json"
    state = read_json(state_path) if state_path.exists() else {
        "schema_version": STATE_SCHEMA, "plan_sha256": json_digest(plan), "plan": plan, "attempts": []}
    if state["plan_sha256"] != json_digest(plan):
        raise ValueError("Plan/config changed; use a new collection output to preserve provenance")
    if not execute:
        return {**summarize(plan, state["attempts"], calibrate=calibrate), "dry_run": True,
                "commands": {t: attempt_command(plan, t, output / "runs" / f"{t}_attempt_000000", plan["seed"])
                             for t in TASKS}}
    if output.exists() and not resume:
        raise FileExistsError("Collection exists; use --resume with the identical plan")
    output.mkdir(parents=True, exist_ok=True)
    for a in state["attempts"]:
        if a.get("status") == "running":
            pid = a.get("pid")
            if pid:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError(f"Prior child PID {pid} is still running; request STOP first")
            # Without a captured normal process exit, retain artifacts but fail closed.
            a.update(status="complete", accepted=False, failure_reason="interrupted_collector_process_outcome_unknown",
                     failure_signature="interrupted_collector_process_outcome_unknown")
    write_json(state_path, state)
    while not (output / "STOP").exists():
        summary = summarize(plan, state["attempts"], calibrate=calibrate)
        candidates = [t for t in TASKS if summary["tasks"][t]["blocked"] is None]
        if not candidates:
            break
        # Round robin by attempts so a difficult first task cannot starve the others.
        task_id = min(candidates, key=lambda t: summary["tasks"][t]["attempts"])
        index = len(state["attempts"])
        run_dir = output / "runs" / f"{task_id}_attempt_{index:06d}"
        if run_dir.exists():
            raise FileExistsError(f"Untracked run directory exists: {run_dir}")
        task_spec = plan["tasks"][task_id]
        if (json_digest(load_task_config(task_spec["config"])) != task_spec["config_sha256"]
                or (task_spec.get("config_source_sha256") is not None
                    and digest(task_spec["config"]) != task_spec["config_source_sha256"])):
            raise ValueError("Task configuration changed during collection")
        if algorithm_fingerprints(plan["runner"]) != plan["algorithm_sha256"]:
            raise ValueError("Runtime algorithm source changed during collection; start a new plan/output")
        attempt = {"index": index, "task_id": task_id, "seed": plan["seed"] + index,
                   "run_dir": str(run_dir), "status": "running", "accepted": False,
                   "calibration_attempt": not summary["tasks"][task_id]["ready"],
                   "execution_sha256": execution_digest(plan, task_id)}
        command = attempt_command(plan, task_id, run_dir, attempt["seed"])
        attempt["command"] = command
        state["attempts"].append(attempt)
        write_json(state_path, state)
        def started(pid):
            attempt["pid"] = pid
            write_json(state_path, state)
        try:
            outcome = launcher(command, log_path=output / "logs" / f"attempt_{index:06d}.log",
                               stop_path=output / "STOP", run_dir=run_dir,
                               max_wall_seconds=plan["max_wall_seconds"], on_start=started)
            attempt.update(outcome)
            if outcome["returncode"] != 0 or outcome.get("timed_out") or outcome.get("forced_termination"):
                raise ValueError(f"process_exit:{outcome['returncode']}" if not outcome.get("timed_out") else "process_wall_timeout")
            if algorithm_fingerprints(plan["runner"]) != plan["algorithm_sha256"]:
                raise ValueError("runtime_algorithm_changed_during_attempt")
            attempt["evidence"] = validate_episode(run_dir, task_id, plan["tasks"][task_id]["config_sha256"])
            attempt["accepted"] = True
        except (ValueError, KeyError, OSError) as exc:
            reason, stage = str(exc), ""
            try:
                report = read_json(run_dir / "run.json")
                failures = [e for e in report.get("episodes", []) if not e.get("success")]
                if failures:
                    reason, stage = failures[-1].get("reason") or reason, failures[-1].get("last_stage", "")
            except (OSError, ValueError):
                pass
            attempt.update(failure_reason=reason, failure_signature=failure_signature(reason, stage))
        attempt["status"] = "complete"
        state["summary"] = summarize(plan, state["attempts"], calibrate=calibrate)
        write_json(state_path, state)
    state["operator_stopped"] = (output / "STOP").exists()
    state["summary"] = summarize(plan, state["attempts"], calibrate=calibrate)
    write_json(state_path, state)
    return state


def merge_successes(collection_root, destination):
    """Copy validated successes into unique staging IDs; require all five quotas."""
    from PIL import Image

    source, destination = Path(collection_root).resolve(), Path(destination).resolve()
    state = read_json(source / "collection.json")
    summary = summarize(state["plan"], state["attempts"])
    if not summary["five_task_coverage"] or not summary["quotas_complete"]:
        raise ValueError("Five-task full-completion coverage and all quotas are required for export")
    if destination.exists():
        raise FileExistsError(destination)
    manifest = {"schema_version": "igris_c_nepes_collection_merge_v1", "collection_scope": "task",
                "five_task_coverage": True, "quotas_complete": True, "source_collection": str(source),
                "collection_manifest_sha256": digest(source / "collection.json"), "episodes": []}
    destination.parent.mkdir(parents=True, exist_ok=True)
    schema = None
    with tempfile.TemporaryDirectory(prefix=".nepes-merge-", dir=destination.parent) as temporary:
        merged = Path(temporary) / "staging"
        for a in state["attempts"]:
            if not a.get("accepted"):
                continue
            t = a["task_id"]
            evidence = validate_episode(a["run_dir"], t, state["plan"]["tasks"][t]["config_sha256"])
            if evidence != a["evidence"]:
                raise ValueError("Accepted physical/recording evidence changed after admission")
            original = Path(evidence["episode_dir"])
            report = read_json(original / "episode.json")
            if schema is not None and schema != report["schema"]:
                raise ValueError("Different robot/camera schemas cannot share a LeRobot dataset")
            schema = report["schema"]
            cameras = validate_camera_sizes(schema["camera_sizes"], require_standard_sizes=True)
            name = f"task{TASKS.index(t) + 1}_attempt{a['index']:06d}_{evidence['frames_sha256'][:12]}"
            target = merged / "episodes" / name
            hashes = {}
            for path in sorted(original.rglob("*")):
                if path.is_symlink():
                    raise ValueError("Source episode symlinks are not accepted")
                if not path.is_file():
                    continue
                relative = path.relative_to(original)
                target_file = target / relative
                target_file.parent.mkdir(parents=True, exist_ok=True)
                if path.suffix == ".png":
                    with Image.open(path) as image:
                        image.load()
                        if image.mode != "RGB" or image.size != cameras.get(path.parent.name):
                            raise ValueError("Incorrect source image mode/resolution")
                expected = digest(path)
                shutil.copy2(path, target_file)
                if digest(target_file) != expected or digest(path) != expected:
                    raise ValueError("Source changed while copying")
                hashes[str(relative)] = expected
            auxiliary = merged / "evidence" / name
            auxiliary.mkdir(parents=True)
            shutil.copy2(original / "episode.json", auxiliary / "original_episode.json")
            raw_hashes = {}
            runtime = Path(a["run_dir"]) / evidence["episode_id"]
            for key, path in (("result.json", runtime / "result.json"), ("trace.jsonl", runtime / "trace.jsonl"),
                              ("initialization.json", runtime / "initialization.json"),
                              ("robot_model_manifest.json", Path(evidence["robot_model_manifest"])),
                              ("config.json", Path(a["run_dir"]) / "config.json"),
                              ("run.json", Path(a["run_dir"]) / "run.json")):
                raw_hashes[key] = digest(path)
                shutil.copy2(path, auxiliary / key)
                if digest(auxiliary / key) != raw_hashes[key]:
                    raise ValueError("Physical evidence copy mismatch")
            report["episode_id"] = name
            report["provenance"]["collection_source"] = {"run_dir": a["run_dir"],
                "original_episode_id": evidence["episode_id"], "original_episode_sha256": hashes["episode.json"]}
            write_json(target / "episode.json", report)
            if validate_episode(a["run_dir"], t, state["plan"]["tasks"][t]["config_sha256"]) != evidence:
                raise ValueError("Source evidence changed while merging")
            manifest["episodes"].append({"merged_episode_id": name, "source": evidence,
                "source_artifact_sha256": hashes, "physical_evidence_sha256": raw_hashes,
                "derived_episode_metadata_sha256": digest(target / "episode.json")})
        write_json(merged / "collection_merge.json", manifest)
        merged.replace(destination)
    return manifest


def export_collection(collection_root, output, dataset_python, *, repo_id="local/igris-c-nepes-five-tasks"):
    """Run the existing official exporter and append a separate provenance manifest."""
    collection_root, output = Path(collection_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(prefix=".collection-export-", dir=collection_root) as temporary:
        staging = Path(temporary) / "staging"
        official = Path(temporary) / "official"
        manifest = merge_successes(collection_root, staging)
        manifest["algorithm_sha256"] = read_json(collection_root / "collection.json")["plan"]["algorithm_sha256"]
        env = dict(os.environ)
        package_root = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = package_root + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run([str(dataset_python), "-m", "robros_motion_planning.dataset", "--staging-root", str(staging),
                        "--output-root", str(official), "--repo-id", repo_id], env=env, check=True)
        exported = read_json(official / "meta" / "nepes_collection.json")["episodes"]
        index = {e["source_episode_id"]: e["episode_index"] for e in exported}
        if set(index) != {e["merged_episode_id"] for e in manifest["episodes"]}:
            raise ValueError("Official export episode mapping differs from admitted collection")
        for episode in manifest["episodes"]:
            episode["official_episode_index"] = index[episode["merged_episode_id"]]
        shutil.copytree(staging / "evidence", official / "auxiliary" / "collection_physics")
        write_json(official / "meta" / "nepes_collection_merge.json", manifest)
        output.parent.mkdir(parents=True, exist_ok=True)
        official.replace(output)
    return output
