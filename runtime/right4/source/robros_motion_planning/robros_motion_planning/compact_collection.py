"""Sequential compact collection; this module never imports Isaac or LeRobot.

The native run is admitted before archiving. Cleanup is delegated to
compact_storage, which requires a verified official dataset and durable
source hashes. Interrupted archive/cleanup operations are retried before any
new native attempt. Raw-mode collections retain their historical behavior.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .collection import digest, read_json


def storage_options(*, storage_mode="raw", physics_trace="gzip",
                    failed_physics_trace="gzip", min_free_gib=20., max_consecutive_failures=3):
    if storage_mode not in ("raw", "compact"):
        raise ValueError("storage_mode must be raw or compact")
    for name, value in (("physics_trace", physics_trace), ("failed_physics_trace", failed_physics_trace)):
        if value not in ("gzip", "keep", "discard"):
            raise ValueError(f"{name} must be gzip, keep or discard")
    if (isinstance(min_free_gib, bool) or not isinstance(min_free_gib, (int, float))
            or not math.isfinite(min_free_gib) or min_free_gib <= 0):
        raise ValueError("min_free_gib must be positive and finite")
    if type(max_consecutive_failures) is not int or max_consecutive_failures < 1:
        raise ValueError("max_consecutive_failures must be a positive integer")
    return dict(storage_mode=storage_mode, physics_trace=physics_trace,
                failed_physics_trace=failed_physics_trace, min_free_gib=float(min_free_gib),
                max_consecutive_failures=max_consecutive_failures)


def _write(path, value):
    """Persist a transaction checkpoint before a destructive retention step."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _lock(root):
    path = root / ".compact_collection.lock"
    if path.is_symlink():
        raise ValueError("Collection lock must not be a symlink")
    with path.open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another collector/exporter is using this collection") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _check_sources(plan):
    from . import stationary_collection as stationary
    stationary.load_stationary_config(plan["config"], expected_sha256=plan["config_sha256"])
    if digest(plan["config"]) != plan["config_source_sha256"]:
        raise ValueError("Source configuration changed during collection")
    if stationary._algorithm_fingerprints() != plan["algorithm_sha256"]:
        raise ValueError("Runtime source changed; resume with the original frozen runtime")


def _summary(state):
    from .stationary_collection import summarize
    summary = summarize(state["plan"], state["attempts"])
    summary["storage_mode"] = "compact"
    summary["archived_successes"] = sum(
        attempt.get("accepted") is True and attempt.get("storage_complete") is True
        for attempt in state["attempts"])
    summary["consecutive_failures"] = _consecutive_failures(state)
    summary["max_consecutive_failures"] = state["plan"]["storage_options"].get("max_consecutive_failures")
    if state.get("stop_reason"):
        summary["stop_reason"] = state["stop_reason"]
    return summary


def _consecutive_failures(state):
    # A retry event starts one new bounded window without changing the frozen plan.
    events = state.get("failure_retry_events", [])
    start = events[-1]["attempt_count"] if events else 0
    count = 0
    for attempt in reversed(state["attempts"][start:]):
        if attempt.get("accepted") is True or attempt.get("status") != "complete":
            break
        count += 1
    return count


def _failure_limit_reached(state):
    # Older frozen plans omitted this policy. Never silently change their contract.
    limit = state["plan"]["storage_options"].get("max_consecutive_failures")
    return limit is not None and _consecutive_failures(state) >= limit


def failure_diagnostics(run_dir, attempt, admission_error=None):
    """Read bounded run/result summaries; diagnostics never replace admission."""
    root = Path(run_dir).resolve()
    documents, errors = [], []
    paths = [root / "run.json", *sorted(root.glob("*/result.json"))[:16]]
    for path in paths:
        try:
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError("unsafe diagnostic report path")
            if path.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("diagnostic report exceeds 16 MiB")
            report = read_json(path)
            if not isinstance(report, dict):
                raise ValueError("diagnostic report must be an object")
            documents.append((path, report))
        except (OSError, ValueError, TypeError) as error:
            errors.append({"path": str(path), "error": f"{type(error).__name__}: {error}"})
    candidates = []
    for path, report in documents:
        episodes = report.get("episodes", []) if path.name == "run.json" else [report]
        if not isinstance(episodes, list):
            episodes = []
        for result in episodes:
            if isinstance(result, dict) and result.get("success") is False:
                reason = result.get("reason") or result.get("failure_reason") or result.get("error")
                if isinstance(reason, str) and reason:
                    candidates.append({"reason": reason, "stage": result.get("failure_stage") or result.get("last_stage"),
                                       "episode_id": result.get("episode_id"), "source": str(path)})
        if isinstance(report.get("error"), str) and report["error"]:
            candidates.append({"reason": report["error"], "stage": report.get("failure_stage"),
                               "source": str(path)})
    selected = candidates[0] if candidates else {
        "reason": admission_error or attempt.get("failure_reason") or
                  f"native_process_failed_or_exit_unrecorded:{attempt.get('returncode')}",
        "stage": None, "source": None}
    return {**selected, "returncode": attempt.get("returncode"),
            "admission_error": admission_error or attempt.get("failure_reason"),
            "timed_out": bool(attempt.get("timed_out")),
            "forced_termination": bool(attempt.get("forced_termination")), "report_errors": errors}


def _save(root, state):
    state["summary"] = _summary(state)
    _write(root / "collection.json", state)


def _check_state(root, state, plan):
    from .stationary_collection import COLLECTION_SCHEMA, config_digest
    if (state.get("schema_version") != COLLECTION_SCHEMA or state.get("plan") != plan
            or state.get("plan_sha256") != config_digest(plan)):
        raise ValueError("Resume requires the identical compact collection plan/config/runtime/options")
    for index, attempt in enumerate(state["attempts"]):
        expected = root / "runs" / f"attempt_{index:06d}"
        if (attempt.get("index") != index or Path(attempt.get("run_dir", "")) != expected
                or expected.is_symlink() or attempt.get("seed") != plan["seed"] + index):
            raise ValueError("Invalid attempt identity in compact collection")


def _ensure_not_running(attempt):
    if attempt.get("status") != "running":
        return
    pid = attempt.get("pid")
    if type(pid) is not int or pid <= 0:
        raise RuntimeError("Interrupted native launch has no recorded PID; inspect it before resuming")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    except PermissionError as error:
        raise RuntimeError(f"Cannot establish that recorded native PID {pid} has stopped") from error
    raise RuntimeError(f"Native PID {pid} is still running; resume after it exits")


def _finish_attempt(root, state, attempt, dataset_python):
    """Recover admission and storage using the last durable native outcome."""
    from .stationary_collection import validate_episode
    from .compact_storage import archive_success, compact_failed_attempt, _fresh_admission_token
    options, plan = state["plan"]["storage_options"], state["plan"]
    fresh_admission = None  # Never inferred from persisted accepted/evidence on resume.
    _ensure_not_running(attempt)
    if not attempt.get("accepted") and "failure_reason" not in attempt:
        try:
            if (attempt.get("returncode") != 0 or attempt.get("timed_out")
                    or attempt.get("forced_termination")):
                raise ValueError(f"native_process_failed_or_exit_unrecorded:{attempt.get('returncode')}")
            _check_sources(plan)
            attempt["evidence"] = validate_episode(attempt["run_dir"], plan["config_sha256"])
            attempt["accepted"] = True
            fresh_admission = _fresh_admission_token(plan, attempt["evidence"])
        except (ValueError, TypeError, KeyError, OSError) as error:
            attempt["failure_reason"] = str(error)
    if not attempt.get("accepted"):
        attempt["failure_diagnostics"] = failure_diagnostics(
            attempt["run_dir"], attempt, attempt.get("failure_reason"))
    attempt["status"] = "complete"
    _save(root, state)  # Admission must be durable before archive/cleanup.
    try:
        if attempt.get("accepted"):
            receipt = archive_success(root, attempt, plan, dataset_python,
                                      trace_policy=options["physics_trace"], _fresh_admission=fresh_admission)
        else:
            receipt = compact_failed_attempt(root, attempt, trace_policy=options["failed_physics_trace"])
        # Per-image hashes stay in the per-run durable receipt. Duplicating tens
        # of thousands of them in collection.json makes every progress update
        # grow with every image of every prior episode.
        receipt_path = Path(attempt["run_dir"]) / "storage_receipt.json"
        attempt["storage"] = {
            "receipt_path": str(receipt_path), "receipt_sha256": digest(receipt_path),
            **{key: receipt[key] for key in (
                "schema", "kind", "trace_policy", "archive", "cleanup_status",
                "images_original_bytes", "images_removed", "traces",
                "preserved_pending_images_bytes", "preserved_pending_image_count") if key in receipt},
        }
        attempt["storage_complete"] = True
        attempt.pop("storage_error", None)
        _save(root, state)
        if not attempt.get("accepted") and receipt.get("preserved_pending_images_bytes", 0):
            raise RuntimeError(
                "This failed/interrupted attempt still contains PNG evidence. It was preserved; "
                "inspect the failed run before continuing")
    except Exception as error:
        attempt["storage_error"] = f"{type(error).__name__}: {error}"
        attempt["storage_complete"] = False
        _save(root, state)
        raise RuntimeError(
            f"Attempt {attempt['index']} storage did not finish; no further attempt was started. "
            "Keep this collection and retry with --resume. " + str(error)) from error
    detail = attempt.get("failure_diagnostics", {})
    outcome = ("accepted and archived" if attempt.get("accepted") else
               f"failed at {detail.get('stage') or 'unknown stage'}: {detail.get('reason')}; trace retention complete")
    print(f"attempt {attempt['index']:06d}: {outcome}; "
          f"successes {state['summary']['successes']}/{plan['successes']}", file=sys.stderr, flush=True)


def collect_compact(plan, output, *, execute, launcher, dataset_python, resume=False,
                    retry_after_failure_limit=False):
    from .stationary_collection import COLLECTION_SCHEMA, attempt_command, config_digest
    options = storage_options(**plan["storage_options"])
    if options["storage_mode"] != "compact":
        raise ValueError("Compact collector requires compact storage options")
    if retry_after_failure_limit and not resume:
        raise ValueError("Retry after the failure limit requires --resume")
    root = Path(output).resolve()
    if not execute:
        return {**_summary({"plan": plan, "attempts": []}), "dry_run": True,
                "storage_options": options, "capture_options": plan.get("capture_options", {}),
                "command": attempt_command(plan, root / "runs/attempt_000000", plan["seed"])}
    if dataset_python is None or not Path(dataset_python).is_file():
        raise ValueError("Compact collection requires an existing --dataset-python")
    if not resume and root.exists():
        raise FileExistsError("Collection output exists; use --resume for this compact collection")
    if resume and not root.is_dir():
        raise FileNotFoundError("Cannot resume a missing compact collection")
    if not resume:
        root.mkdir(parents=True)
    with _lock(root):
        _check_sources(plan)
        if resume:
            state = read_json(root / "collection.json")
            _check_state(root, state, plan)
        else:
            state = {"schema_version": COLLECTION_SCHEMA, "plan": plan,
                     "plan_sha256": config_digest(plan), "attempts": [], "complete": False}
        if retry_after_failure_limit:
            if not _failure_limit_reached(state):
                raise ValueError("No consecutive-failure limit is pending; omit --retry-after-failure-limit")
            state.setdefault("failure_retry_events", []).append({
                "attempt_count": len(state["attempts"]),
                "previous_consecutive_failures": _consecutive_failures(state),
                "requested_at_utc": datetime.now(timezone.utc).isoformat()})
            state.pop("stop_reason", None)
        elif _failure_limit_reached(state):
            state.update(complete=True, stop_reason="consecutive_failure_limit")
            _save(root, state)
            print("Consecutive-failure limit reached; inspect the failures before using "
                  "--resume --retry-after-failure-limit.", file=sys.stderr, flush=True)
            return state
        state.update(complete=False)
        state.pop("storage_error", None)
        _save(root, state)
        # Check archived hashes and finish interrupted cleanup before new capture.
        for attempt in state["attempts"]:
            _finish_attempt(root, state, attempt, dataset_python)
        while (len(state["attempts"]) < plan["max_attempts"]
               and not state["summary"]["quota_complete"] and not (root / "STOP").exists()
               and not _failure_limit_reached(state)):
            _check_sources(plan)
            free_gib = shutil.disk_usage(root).free / 1024**3
            if free_gib < options["min_free_gib"]:
                state["storage_error"] = (
                    f"{free_gib:.1f}GiB free; need {options['min_free_gib']:.1f}GiB before next attempt")
                _save(root, state)
                raise RuntimeError(state["storage_error"] + "; free space and use --resume")
            index = len(state["attempts"])
            run = root / "runs" / f"attempt_{index:06d}"
            attempt = {"index": index, "run_dir": str(run), "seed": plan["seed"] + index,
                       "accepted": False, "status": "running"}
            attempt["command"] = attempt_command(plan, run, attempt["seed"])
            state["attempts"].append(attempt)
            _save(root, state)

            def started(pid):
                attempt["pid"] = pid
                _save(root, state)

            try:
                outcome = launcher(attempt["command"], log_path=root / "logs" / f"attempt_{index:06d}.log",
                                   stop_path=root / "STOP", run_dir=run,
                                   max_wall_seconds=plan["max_wall_seconds"], on_start=started)
                attempt.update(outcome)
                attempt["status"] = "native_finished"
                _save(root, state)  # Preserve exit outcome even if validation is interrupted.
            except (OSError, ValueError) as error:
                # Popen may have succeeded before on_start/progress persistence
                # failed. Keep "running" so resume checks the PID; never compress
                # a trace that an orphaned native child could still be writing.
                attempt["launcher_error"] = f"{type(error).__name__}: {error}"
                _save(root, state)
                raise RuntimeError(
                    "Native launcher failed; no files were cleaned. Confirm that the recorded "
                    "child has exited before resuming: " + str(error)) from error
            _finish_attempt(root, state, attempt, dataset_python)
        if _failure_limit_reached(state):
            state["stop_reason"] = "consecutive_failure_limit"
            print(f"Stopped after {_consecutive_failures(state)} consecutive failures; "
                  "inspect failure_diagnostics before explicitly resuming.", file=sys.stderr, flush=True)
        state.update(complete=True, operator_stopped=(root / "STOP").exists())
        _save(root, state)
        return state


def export_compact_collection(collection_root, output, dataset_python, *, repo_id=None, allow_partial=False):
    """Merge verified per-episode official shards, without reopening deleted PNGs."""
    root, output = Path(collection_root).resolve(), Path(output).resolve()
    with _lock(root):
        state = read_json(root / "collection.json")
        _check_state(root, state, state["plan"])
        if not allow_partial and (state.get("complete") is not True or not _summary(state)["quota_complete"]
                or not all(a.get("storage_complete") for a in state["attempts"])):
            raise ValueError("Compact collection quota/storage is incomplete; resume collection first")
        for attempt in state["attempts"]:
            _ensure_not_running(attempt)
        accepted = [a for a in state["attempts"] if a.get("accepted") is True]
        if not accepted or any(a.get("storage_complete") is not True for a in accepted):
            raise ValueError("Export requires at least one admitted success and all accepted archives complete")
        from .compact_storage import verify_archived_success
        shards = []
        for attempt in accepted:
            shards.append(str(verify_archived_success(root, attempt, state["plan"])))
        identity = {"collection_sha256": digest(root / "collection.json"),
                    "output": str(output), "shards": shards}
        if allow_partial:
            identity["allow_partial"] = True
        receipt_path = root / "exports" / (digest(root / "collection.json")[:16] + ".json")
        if output.exists():
            from .compact_storage import verify_archive
            verify_archive(output)
            verified = output / "meta/nepes_storage_verification.json"
            if read_json(output / "meta/nepes_stationary_collection.json") != state:
                raise FileExistsError(f"Existing dataset belongs to a different collection: {output}")
            if receipt_path.is_file():
                receipt = read_json(receipt_path)
                if (receipt.get("identity") != identity
                        or digest(verified) != receipt["verification_sha256"]):
                    raise ValueError("Existing export verification receipt changed")
            else:
                # A crash may happen after the dataset's atomic publication but
                # before the collection-side receipt. Recover only an exact match.
                _write(receipt_path, {"identity": identity, "verification_sha256": digest(verified),
                                     "schema": "igris_c_nepes_compact_export_v1"})
            return output
        output.parent.mkdir(parents=True, exist_ok=True)
        needed = sum(p.stat().st_size for shard in shards for p in Path(shard).rglob("*") if p.is_file())
        if shutil.disk_usage(output.parent).free < needed + 1024**3:
            raise RuntimeError("Insufficient free space to merge compact shards; keep the shards and retry")
        shard_list = root / "compact_shards.json"
        _write(shard_list, shards)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
        command = [str(dataset_python), "-m", "robros_motion_planning.compact_dataset", "merge",
                   "--shards-json", str(shard_list), "--output-root", str(output),
                   "--repo-id", repo_id or "local/igris-c-stationary-" + state["plan"]["task"]["id"].replace("_", "-"),
                   "--collection-manifest", str(root / "collection.json")]
        try:
            subprocess.run(command, env=env, check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Compact dataset merge failed ({error.returncode}); shards are preserved") from error
        verification = output / "meta/nepes_storage_verification.json"
        _write(receipt_path, {"identity": identity, "verification_sha256": digest(verification),
                             "schema": "igris_c_nepes_compact_export_v1"})
        return output
