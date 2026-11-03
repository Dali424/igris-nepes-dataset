"""Verified, per-attempt retention for newly collected stationary recordings.

PNG removal is permitted only after an official single-episode archive and its
durable verification receipt exist. This module never scans old collections.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile


SCHEMA = "nepes_attempt_storage_v1"
VERIFICATION_SCHEMA = "nepes_verified_lerobot_storage_v1"
TRACE_POLICIES = frozenset(("gzip", "keep", "discard"))
GZIP_COMPRESSION_LEVEL = 1
_VERIFICATION_CHECKS = (
    "all_mp4_fully_decoded", "parquet_metadata_consistent",
    "camera_schema_consistent", "source_binding_consistent",
)


def _digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _canonical_digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class _FreshAdmission:
    """In-process binding minted only after the collector validates this attempt.

    This is not persisted or accepted from CLI/JSON. Resumed accepted attempts
    cannot reconstruct freshness merely from their stored ``accepted`` flag.
    """
    plan_sha256: str
    evidence_sha256: str


def _fresh_admission_token(plan, evidence):
    return _FreshAdmission(_canonical_digest(plan), _canonical_digest(evidence))


def _safe_path(root, path, *, missing=False, directory=False):
    """Reject lexical escapes and symlinks, including symlinked ancestors."""
    path = Path(path)
    if ".." in path.parts:
        raise ValueError("storage_path_contains_parent_reference")
    path = Path(os.path.abspath(path))
    root = Path(os.path.abspath(root))
    if not path.is_relative_to(root):
        raise ValueError("storage_path_escapes_collection")
    for parent in reversed((path, *path.parents)):
        try:
            mode = parent.lstat().st_mode
        except FileNotFoundError:
            if missing:
                continue
            raise
        if stat.S_ISLNK(mode):
            raise ValueError(f"storage_symlink_forbidden:{parent}")
    if not path.exists():
        if missing:
            return path
        raise FileNotFoundError(path)
    if directory and not path.is_dir():
        raise ValueError(f"storage_directory_required:{path}")
    if not directory and not path.is_file():
        raise ValueError(f"storage_regular_file_required:{path}")
    return path


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_receipt(path, value):
    _safe_path(path.parent, path, missing=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".storage-receipt-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _inventory(root, directory):
    _safe_path(root, directory, directory=True)
    files = []
    for folder, subdirs, names in os.walk(directory, followlinks=False):
        folder = Path(folder)
        for name in subdirs:
            _safe_path(root, folder / name, directory=True)
        for name in names:
            path = _safe_path(root, folder / name)
            files.append({"path": str(path.relative_to(root)),
                          "sha256": _digest(path), "size_bytes": path.stat().st_size})
    return sorted(files, key=lambda item: item["path"])


def _check_file(root, item, *, missing=False):
    path = _safe_path(root, root / item["path"], missing=missing)
    if not path.exists():
        return False
    if (path.stat().st_size != item["size_bytes"] or _digest(path) != item["sha256"]):
        raise ValueError(f"storage_file_changed:{item['path']}")
    return True


def _context(collection_root, attempt, plan=None, *, accepted):
    root = _safe_path(Path(collection_root), Path(collection_root), directory=True)
    index = attempt.get("index")
    if type(index) is not int or index < 0:
        raise ValueError("storage_attempt_index_invalid")
    run = _safe_path(root, root / "runs" / f"attempt_{index:06d}", missing=not accepted, directory=True)
    if Path(attempt.get("run_dir", "")) != run:
        raise ValueError("storage_attempt_run_directory_mismatch")
    if attempt.get("accepted") is not accepted or attempt.get("status") != "complete":
        raise ValueError("storage_attempt_not_completed_with_expected_admission")
    state = _json(_safe_path(root, root / "collection.json"))
    matches = [item for item in state.get("attempts", []) if item.get("index") == index]
    if len(matches) != 1:
        raise ValueError("storage_attempt_not_owned_by_collection")
    saved = matches[0]
    for key in ("index", "run_dir", "accepted", "status", "evidence"):
        if saved.get(key) != attempt.get(key):
            raise ValueError(f"storage_attempt_not_durably_admitted:{key}")
    if plan is not None and state.get("plan") != plan:
        raise ValueError("storage_plan_mismatch")
    if state.get("plan", {}).get("storage_options", {}).get("storage_mode") != "compact":
        raise ValueError("storage_compact_mode_not_explicitly_requested")
    if not run.exists():
        outcome_keys = ("returncode", "timed_out", "forced_termination")
        code = attempt.get("returncode")
        failed_process = (type(code) is int and code != 0
                          or attempt.get("timed_out") is True
                          or attempt.get("forced_termination") is True)
        if (accepted or not failed_process
                or any(saved.get(key) != attempt.get(key) for key in outcome_keys)):
            raise ValueError("storage_missing_run_has_no_durable_process_failure")
        # Only after the process failed and its outcome was durably saved.
        # Creating this before launching Isaac would violate fresh-output rules.
        run.mkdir(parents=True)
        _fsync_directory(run.parent)
    return root, run


@contextmanager
def _attempt_lock(run):
    path = _safe_path(run, run / ".storage.lock", missing=True)
    with path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def _check_policy(policy):
    if policy not in TRACE_POLICIES:
        raise ValueError("trace_policy_must_be_gzip_keep_or_discard")


def verify_archive(path):
    """Verify a published archive's complete file hashes and decode receipt.

    This checks the immutable results of the exporter's full decode/Parquet
    validation; it does not decode the same MP4s again on each collection resume.
    """
    shard = _safe_path(Path(path), Path(path), directory=True)
    receipt_path = _safe_path(shard, shard / "meta/nepes_storage_verification.json")
    receipt = _json(receipt_path)
    if (receipt.get("schema") != VERIFICATION_SCHEMA or receipt.get("verified") is not True
            or receipt.get("kind") not in ("single_episode_shard", "merged_dataset")
            or type(receipt.get("total_episodes")) is not int or receipt["total_episodes"] < 1
            or type(receipt.get("total_frames")) is not int or receipt["total_frames"] < 1
            or receipt.get("fps") != 30
            or any(receipt.get("checks", {}).get(key) is not True for key in _VERIFICATION_CHECKS)):
        raise ValueError("storage_archive_verification_failed")
    files = receipt.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("storage_archive_file_manifest_missing")
    paths = [item.get("path") for item in files]
    if len(set(paths)) != len(paths) or "meta/nepes_storage_sources.json" not in paths:
        raise ValueError("storage_archive_file_manifest_invalid")
    actual = _inventory(shard, shard)
    actual = [item for item in actual if item["path"] != "meta/nepes_storage_verification.json"]
    if sorted(files, key=lambda item: item["path"]) != actual:
        raise ValueError("storage_archive_files_changed")
    sources = _json(shard / "meta/nepes_storage_sources.json")
    bindings = receipt.get("source_episodes")
    if (sources.get("schema") != "nepes_compact_dataset_sources_v1"
            or sources.get("kind") != receipt["kind"]
            or sources.get("source_episodes") != bindings
            or not isinstance(bindings, list) or len(bindings) != receipt["total_episodes"]
            or [item.get("episode_index") for item in bindings] != list(range(receipt["total_episodes"]))
            or sum(item.get("frame_count", 0) for item in bindings) != receipt["total_frames"]):
        raise ValueError("storage_archive_source_manifest_changed")
    return receipt


def _verify_archive(shard, evidence):
    receipt = verify_archive(shard)
    receipt_path = shard / "meta/nepes_storage_verification.json"
    if (receipt["kind"] != "single_episode_shard" or receipt["total_episodes"] != 1
            or receipt["total_frames"] != evidence["frame_count"]):
        raise ValueError("storage_archive_single_episode_required")
    sources = receipt.get("source_episodes")
    if not isinstance(sources, list) or len(sources) != 1:
        raise ValueError("storage_archive_source_count_mismatch")
    source = sources[0]
    expected = {
        "episode_index": 0, "source_episode_id": evidence["episode_id"],
        "staging_episode_directory": evidence["episode_dir"],
        "episode_json_sha256": evidence["artifact_sha256"]["metadata"],
        "frames_sha256": evidence["frames_sha256"], "frame_count": evidence["frame_count"],
    }
    if any(source.get(key) != value for key, value in expected.items()):
        raise ValueError("storage_archive_source_binding_mismatch")
    files = receipt["files"]
    return {"path": str(shard), "verification_receipt_sha256": _digest(receipt_path),
            "total_frames": receipt["total_frames"], "file_count": len(files),
            "size_bytes": sum(item["size_bytes"] for item in files)}


def _export_archive(staging, shard, dataset_python, log_path):
    environment = dict(os.environ)
    package_root = str(Path(__file__).resolve().parent.parent)
    environment["PYTHONPATH"] = package_root + os.pathsep + environment.get("PYTHONPATH", "")
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = "1"
    command = [str(dataset_python), "-m", "robros_motion_planning.compact_dataset", "export",
               "--staging-root", str(staging), "--output-root", str(shard),
               "--repo-id", "local/nepes-compact-episode"]
    with log_path.open("ab") as log:
        outcome = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                 env=environment, check=False)
    if outcome.returncode:
        raise RuntimeError(f"storage_export_failed:{outcome.returncode}:{log_path}")


def _trace_item(run, path):
    path = _safe_path(run, path)
    return {"path": str(path.relative_to(run)), "sha256": _digest(path),
            "size_bytes": path.stat().st_size, "state": "pending"}


def _verify_gzip(path, item):
    result = hashlib.sha256()
    size = 0
    with gzip.open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            result.update(block)
    if size != item["size_bytes"] or result.hexdigest() != item["sha256"]:
        raise ValueError("storage_gzip_does_not_match_original_trace")


def _prepare_gzip(run, item):
    """Publish a checked gzip while retaining its source until cleanup commits."""
    path = _safe_path(run, run / item["path"], missing=True)
    compressed = _safe_path(run, path.with_suffix(path.suffix + ".gz"), missing=True)
    if compressed.exists():
        if item.get("compressed_sha256") is not None and _digest(compressed) != item["compressed_sha256"]:
            raise ValueError("storage_compressed_trace_changed")
        _verify_gzip(compressed, item)
    else:
        _check_file(run, item)
        descriptor, temporary = tempfile.mkstemp(prefix=".trace-gzip-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as target:
                with gzip.GzipFile(fileobj=target, mode="wb", compresslevel=GZIP_COMPRESSION_LEVEL, mtime=0) as zipped:
                    with path.open("rb") as source:
                        shutil.copyfileobj(source, zipped, 1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            _verify_gzip(Path(temporary), item)
            os.replace(temporary, compressed)
            _fsync_directory(path.parent)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return compressed


def _compact_trace(run, item, policy):
    path = _safe_path(run, run / item["path"], missing=True)
    if policy == "keep":
        _check_file(run, item)
        item["state"] = "kept"
        return
    if policy == "gzip":
        compressed = _prepare_gzip(run, item)
        if path.exists():
            _check_file(run, item)
            path.unlink()
            _fsync_directory(path.parent)
        item.update(state="gzipped", compressed_path=str(compressed.relative_to(run)),
                    compressed_sha256=_digest(compressed), compressed_size_bytes=compressed.stat().st_size,
                    compression_level=GZIP_COMPRESSION_LEVEL)
    else:
        if path.exists():
            _check_file(run, item)
            path.unlink()
            _fsync_directory(path.parent)
        item["state"] = "discarded"


def _remove_images(run, image_root, inventory):
    if not image_root.exists():
        _safe_path(run, image_root, missing=True, directory=True)
        return
    actual = _inventory(run, image_root)
    expected = {item["path"]: item for item in inventory}
    for item in actual:
        if expected.get(item["path"]) != item:
            raise ValueError("storage_images_changed_before_removal")
    for item in actual:
        path = _safe_path(run, run / item["path"])
        path.unlink()
    for folder, _, _ in os.walk(image_root, topdown=False, followlinks=False):
        _safe_path(run, Path(folder), directory=True).rmdir()
    _fsync_directory(image_root.parent)


def _retained_sources(run, evidence):
    episode = Path(evidence["episode_dir"])
    runtime = run / evidence["episode_id"]
    sources = {
        "result": runtime / "result.json", "initialization": runtime / "initialization.json",
        "metadata": episode / "episode.json", "robot_model_manifest": Path(evidence["robot_model_manifest"]),
        "run": run / "run.json",
    }
    result = []
    for key, path in sources.items():
        path = _safe_path(run, path)
        item = {"path": str(path.relative_to(run)), "sha256": evidence["artifact_sha256"][key],
                "size_bytes": path.stat().st_size}
        _check_file(run, item)
        result.append(item)
    path = _safe_path(run, episode / "frames.jsonl")
    result.append({"path": str(path.relative_to(run)), "sha256": evidence["frames_sha256"],
                   "size_bytes": path.stat().st_size})
    _check_file(run, result[-1])
    path = _safe_path(run, run / "config.json")
    result.append({"path": str(path.relative_to(run)), "sha256": _digest(path),
                   "size_bytes": path.stat().st_size})
    return result


def _check_images_manifest(run, episode, images, evidence):
    """Reproduce admission's ordered image fingerprint from the saved hashes."""
    expected = {item["path"]: item for item in images}
    if len(expected) != len(images):
        raise ValueError("storage_images_manifest_duplicate")
    cameras = _json(episode / "episode.json")["schema"]["camera_sizes"]
    digest = hashlib.sha256()
    used = set()
    for index in range(evidence["frame_count"]):
        for camera in cameras:
            relative = f"images/{camera}/{index:06d}.png"
            key = str((episode / relative).relative_to(run))
            item = expected.get(key)
            if item is None:
                raise ValueError("storage_images_manifest_missing")
            digest.update((relative + ":" + item["sha256"] + "\n").encode())
            used.add(key)
    if used != set(expected) or digest.hexdigest() != evidence["images_sha256"]:
        raise ValueError("storage_images_manifest_admission_mismatch")


def _check_saved_receipt(receipt, root, run, attempt, policy, kind):
    if (receipt.get("schema") != SCHEMA or receipt.get("kind") != kind
            or receipt.get("collection_root") != str(root) or receipt.get("run_dir") != str(run)
            or receipt.get("attempt_index") != attempt["index"] or receipt.get("trace_policy") != policy
            or receipt.get("admitted_evidence") != attempt.get("evidence")):
        raise ValueError("storage_receipt_identity_or_policy_mismatch")
    for item in receipt.get("retained_sources", []):
        _check_file(run, item)


def verify_archived_success(collection_root, attempt, plan):
    """Read-only check of a completed admitted archive; never retries cleanup."""
    root, run = _context(collection_root, attempt, plan, accepted=True)
    if attempt.get("storage_complete") is not True:
        raise ValueError("storage_success_archive_not_complete")
    receipt_path = _safe_path(run, run / "storage_receipt.json")
    saved = attempt.get("storage", {})
    if (saved.get("receipt_path") != str(receipt_path)
            or saved.get("receipt_sha256") != _digest(receipt_path)):
        raise ValueError("storage_ledger_receipt_changed")
    receipt = _json(receipt_path)
    policy = plan["storage_options"]["physics_trace"]
    _check_saved_receipt(receipt, root, run, attempt, policy, "accepted_episode")
    evidence = attempt.get("evidence", {})
    if (receipt.get("cleanup_status") != "complete" or receipt.get("images_removed") is not True
            or receipt.get("plan_sha256") != _canonical_digest(plan)
            or receipt.get("retained_sources") != _retained_sources(run, evidence)):
        raise ValueError("storage_success_receipt_incomplete_or_changed")
    shard = _safe_path(root, root / "archives" / f"attempt_{attempt['index']:06d}", directory=True)
    if _verify_archive(shard, evidence) != receipt.get("archive"):
        raise ValueError("storage_archive_changed_since_cleanup_started")
    episode = Path(evidence["episode_dir"])
    _check_images_manifest(run, episode, receipt["images"], evidence)
    traces = receipt.get("traces", [])
    if (len(traces) != 1 or traces[0].get("path") != f"{evidence['episode_id']}/trace.jsonl"
            or traces[0].get("sha256") != evidence["artifact_sha256"]["trace"]):
        raise ValueError("storage_trace_manifest_mismatch")
    item = traces[0]
    if policy == "gzip":
        if item.get("state") != "gzipped":
            raise ValueError("storage_trace_compression_incomplete")
        _check_file(run, {"path": item["compressed_path"], "sha256": item["compressed_sha256"],
                          "size_bytes": item["compressed_size_bytes"]})
    elif policy == "keep":
        _check_file(run, item)
    elif item.get("state") != "discarded":
        raise ValueError("storage_trace_retention_incomplete")
    return shard


def archive_success(collection_root, attempt, plan, dataset_python, trace_policy="gzip", *,
                    _fresh_admission=None):
    """Archive an admitted attempt, then remove only its owned episode images.

    ``collection.json`` must already contain the completed, accepted attempt and
    its ``validate_episode`` evidence. Re-entry after cleanup uses this durable
    admission, source hashes and verified archive instead of missing raw PNGs.
    The private in-process freshness token skips only a repeated physical/frame
    parse. Every source/image hash and post-export/deletion check still runs.
    """
    if _fresh_admission is not None and type(_fresh_admission) is not _FreshAdmission:
        raise ValueError("storage_fresh_admission_token_invalid")
    _check_policy(trace_policy)
    root, run = _context(collection_root, attempt, plan, accepted=True)
    evidence = attempt.get("evidence")
    if not isinstance(evidence, dict) or evidence.get("run_dir") != str(run):
        raise ValueError("storage_admitted_evidence_missing")
    name = evidence.get("episode_id", "")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("storage_episode_id_invalid")
    episode = run / "staging/episodes" / name
    if evidence.get("episode_dir") != str(episode) or evidence.get("config_sha256") != plan["config_sha256"]:
        raise ValueError("storage_episode_identity_mismatch")
    _safe_path(run, episode, directory=True)
    archive_root = _safe_path(root, root / "archives", missing=True, directory=True)
    archive_root.mkdir(exist_ok=True)
    shard = _safe_path(root, archive_root / f"attempt_{attempt['index']:06d}", missing=True, directory=True)
    receipt_path = _safe_path(run, run / "storage_receipt.json", missing=True)
    with _attempt_lock(run):
        if receipt_path.exists():
            receipt = _json(receipt_path)
            _check_saved_receipt(receipt, root, run, attempt, trace_policy, "accepted_episode")
            if receipt.get("plan_sha256") != _canonical_digest(plan):
                raise ValueError("storage_receipt_plan_changed")
            if receipt.get("retained_sources") != _retained_sources(run, evidence):
                raise ValueError("storage_retained_source_manifest_changed")
            if _verify_archive(shard, evidence) != receipt["archive"]:
                raise ValueError("storage_archive_changed_since_cleanup_started")
        else:
            from .stationary_collection import load_stationary_config, validate_episode
            if _fresh_admission is None:
                if validate_episode(run, plan["config_sha256"]) != evidence:
                    raise ValueError("storage_admitted_evidence_changed")
                admission_method = "full_episode_revalidated"
            else:
                if _fresh_admission != _fresh_admission_token(plan, evidence):
                    raise ValueError("storage_fresh_admission_binding_changed")
                # The caller checked runtime identity before admission. Recheck
                # now in case source/config files changed while it was parsed.
                from .compact_collection import _check_sources
                _check_sources(plan)
                load_stationary_config(_safe_path(run, run / "config.json"),
                                       expected_sha256=plan["config_sha256"])
                admission_method = "fresh_durable_admission_reused"
            retained = _retained_sources(run, evidence)
            images = _inventory(run, episode / "images")
            _check_images_manifest(run, episode, images, evidence)
            trace = _trace_item(run, run / name / "trace.jsonl")
            if trace["sha256"] != evidence["artifact_sha256"]["trace"]:
                raise ValueError("storage_trace_changed_since_admission")
            if not shard.exists():
                _export_archive(run / "staging", shard, dataset_python,
                                _safe_path(run, run / "storage_export.log", missing=True))
            archive = _verify_archive(shard, evidence)
            for item in retained + images + [trace]:
                _check_file(run, item)
            receipt = {
                "schema": SCHEMA, "kind": "accepted_episode", "collection_root": str(root),
                "run_dir": str(run), "attempt_index": attempt["index"], "trace_policy": trace_policy,
                "compression_level": GZIP_COMPRESSION_LEVEL if trace_policy == "gzip" else None,
                "plan_sha256": _canonical_digest(plan), "admitted_evidence": deepcopy(evidence),
                "admission_validation": admission_method,
                "archive": archive, "retained_sources": retained, "images": images,
                "images_original_bytes": sum(item["size_bytes"] for item in images),
                "traces": [trace], "cleanup_status": "verified_pending_cleanup",
            }
            # This durable write is the deletion commit point, including hashes
            # of every original image and trace, not just their resulting media.
            _write_receipt(receipt_path, receipt)
        _check_images_manifest(run, episode, receipt["images"], evidence)
        traces = receipt.get("traces")
        if (not isinstance(traces, list) or len(traces) != 1
                or traces[0].get("path") != f"{name}/trace.jsonl"
                or traces[0].get("sha256") != evidence["artifact_sha256"]["trace"]):
            raise ValueError("storage_trace_manifest_mismatch")
        # All archive and gzip verification happens before the first PNG unlink.
        # If compression fails, the raw images and trace are still available.
        for item in traces:
            if trace_policy == "gzip":
                _prepare_gzip(run, item)
            elif trace_policy == "keep":
                _check_file(run, item)
        _remove_images(run, episode / "images", receipt["images"])
        receipt["images_removed"] = True
        receipt["cleanup_status"] = "images_removed_trace_pending"
        _write_receipt(receipt_path, receipt)
        for item in receipt["traces"]:
            _compact_trace(run, item, trace_policy)
        receipt["cleanup_status"] = "complete"
        _write_receipt(receipt_path, receipt)
        return receipt


def compact_failed_attempt(collection_root, attempt, trace_policy="gzip"):
    """Retain failed diagnostics; optionally compress/discard raw traces only.

    Pending/failed PNGs are reported and preserved, never silently deleted.
    """
    _check_policy(trace_policy)
    root, run = _context(collection_root, attempt, accepted=False)
    receipt_path = _safe_path(run, run / "storage_receipt.json", missing=True)
    with _attempt_lock(run):
        if receipt_path.exists():
            receipt = _json(receipt_path)
            _check_saved_receipt(receipt, root, run, attempt, trace_policy, "failed_attempt")
        else:
            traces = []
            for child in sorted(run.iterdir()):
                if child.is_symlink():
                    raise ValueError("storage_symlink_forbidden")
                if child.is_dir() and (child / "trace.jsonl").exists():
                    traces.append(_trace_item(run, child / "trace.jsonl"))
            pending = []
            staging = run / "staging"
            if staging.exists():
                pending = [item for item in _inventory(run, staging) if item["path"].lower().endswith(".png")]
            receipt = {
                "schema": SCHEMA, "kind": "failed_attempt", "collection_root": str(root),
                "run_dir": str(run), "attempt_index": attempt["index"], "trace_policy": trace_policy,
                "compression_level": GZIP_COMPRESSION_LEVEL if trace_policy == "gzip" else None,
                "admitted_evidence": deepcopy(attempt.get("evidence")),
                "failure_reason": attempt.get("failure_reason"), "traces": traces,
                "preserved_pending_images_bytes": sum(item["size_bytes"] for item in pending),
                "preserved_pending_image_count": len(pending),
                "cleanup_status": "trace_pending",
            }
            _write_receipt(receipt_path, receipt)
        # An operator may have moved failed PNG diagnostics to separate storage.
        # Keep their original footprint while reporting the current footprint to
        # the collector, so a reviewed failure can resume without editing proof.
        receipt.setdefault("initial_pending_images_bytes", receipt["preserved_pending_images_bytes"])
        receipt.setdefault("initial_pending_image_count", receipt["preserved_pending_image_count"])
        staging = run / "staging"
        pending = []
        if staging.exists():
            pending = [item for item in _inventory(run, staging) if item["path"].lower().endswith(".png")]
        receipt["preserved_pending_images_bytes"] = sum(item["size_bytes"] for item in pending)
        receipt["preserved_pending_image_count"] = len(pending)
        for item in receipt["traces"]:
            relative = Path(item["path"])
            if len(relative.parts) != 2 or relative.name != "trace.jsonl" or relative.parts[0] in (".", ".."):
                raise ValueError("storage_failed_trace_path_invalid")
            _compact_trace(run, item, trace_policy)
        receipt["cleanup_status"] = "complete"
        _write_receipt(receipt_path, receipt)
        return receipt
