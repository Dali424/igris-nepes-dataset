"""Verified, atomic per-episode LeRobot export and lossless shard aggregation.

Run in the isolated dataset environment. This module never removes source PNGs
or physics evidence. A retention caller may do so only after checking the
published receipt and its source/file hashes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any, Sequence
from uuid import uuid4

import numpy as np

from .dataset import (LEROBOT_VERSION, STAGING_SCHEMA, _ordered_bounded_map, export_lerobot_v3,
                      lerobot_features, validate_camera_sizes)

RECEIPT_PATH = Path("meta/nepes_storage_verification.json")
SOURCES_PATH = Path("meta/nepes_storage_sources.json")
RECEIPT_SCHEMA = "nepes_verified_lerobot_storage_v1"
SOURCES_SCHEMA = "nepes_compact_dataset_sources_v1"
_RESERVED = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _fsync_directory(path: Path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_durable(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _files(root: Path) -> list[Path]:
    result = []
    for path in sorted(root.rglob("*")):
        _require(not path.is_symlink(), f"Dataset artifacts cannot be symlinks: {path}")
        if path.is_file() and path.relative_to(root) != RECEIPT_PATH:
            result.append(path)
    return result


def _inventory(root: Path) -> list[dict]:
    return [{"path": str(path.relative_to(root)), "sha256": _sha(path), "size_bytes": path.stat().st_size}
            for path in _files(root)]


def _relative(root: Path, name: str) -> Path:
    path = root / name
    _require(not Path(name).is_absolute() and path.resolve().is_relative_to(root),
             f"Dataset reference escapes its root: {name}")
    _require(path.is_file() and not path.is_symlink(), f"Missing dataset reference: {name}")
    return path


def _official_version():
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION
    _require(version("lerobot") == LEROBOT_VERSION and CODEBASE_VERSION == "v3.0",
             f"Requires official lerobot=={LEROBOT_VERSION}, dataset v3.0")


def _video_decode(path: Path, *, fps: int, width: int, height: int, count: int) -> dict:
    import av
    decoded = 0
    pts_first = pts_last = None
    max_error = 0.
    with av.open(str(path)) as container:
        _require(len(container.streams) == len(container.streams.video) == 1,
                 f"Expected one video-only stream: {path}")
        stream = container.streams.video[0]
        _require(stream.average_rate is not None and float(stream.average_rate) == fps,
                 f"Video fps mismatch: {path}")
        _require((stream.width, stream.height) == (width, height), f"Video resolution mismatch: {path}")
        _require(not stream.frames or stream.frames == count, f"Video container frame count mismatch: {path}")
        for frame in container.decode(stream):
            _require(frame.pts is not None and frame.time_base is not None and not frame.is_corrupt,
                     f"Missing timestamp or corrupt decoded video frame: {path}")
            _require((frame.width, frame.height) == (width, height), f"Decoded resolution changed: {path}")
            pts = float(frame.pts * frame.time_base)
            error = abs(pts - decoded / fps)
            _require(error <= max(1e-6, float(frame.time_base) / 2),
                     f"Non-contiguous video PTS at frame {decoded}: {path}")
            _require(pts_last is None or pts > pts_last, f"Repeated/out-of-order video PTS: {path}")
            pts_first = pts if pts_first is None else pts_first
            pts_last = pts
            max_error = max(max_error, error)
            decoded += 1
        _require(decoded == count and decoded > 0, f"Full MP4 decode frame count mismatch: {path}")
        duration = float(stream.duration * stream.time_base) if stream.duration is not None else None
        if duration is not None:
            _require(abs(duration - count / fps) <= max(1e-5, float(stream.time_base)),
                     f"Video duration mismatch: {path}")
        return {"decoded_frames": decoded, "fps": fps, "resolution": [width, height],
                "pts_first_s": pts_first, "pts_last_s": pts_last, "max_pts_error_s": max_error,
                "duration_s": duration, "codec": stream.codec_context.name, "full_decode": True}


def _decode_video_job(job):
    path, name, key, width, height, count = job
    return {"path": name, "camera_key": key,
            **_video_decode(path, fps=30, width=width, height=height, count=count)}


def verify_dataset(path: str | Path, expected_episodes: int | None = None, *, video_workers: int = 2) -> dict:
    """Read and fully verify a compact dataset; return a receipt, without edits."""
    if type(video_workers) is not int or video_workers < 0:
        raise ValueError("video_workers must be a nonnegative integer")
    started = perf_counter()
    import pyarrow.parquet as pq
    _official_version()
    root = Path(path).resolve(strict=True)
    info = _read(root / "meta/info.json")
    provenance = _read(root / "meta/nepes_collection.json")
    sources = _read(root / SOURCES_PATH)
    _require(sources.get("schema") == SOURCES_SCHEMA, "Missing compact dataset source bindings")
    capture = provenance["capture_schema"]
    cameras = validate_camera_sizes(capture["camera_sizes"])
    _require(capture.get("schema_version") == STAGING_SCHEMA, "Unsupported capture schema")
    _require(info.get("codebase_version") == "v3.0" and info.get("fps") == capture.get("fps") == 30,
             "Dataset version/fps mismatch")
    _require(info.get("robot_type") == "igris_c_hugwbc", "Unexpected robot schema")
    episode_count, total = info["total_episodes"], info["total_frames"]
    _require(type(episode_count) is int and episode_count > 0 and type(total) is int and total > 0,
             "Dataset episode/frame counts must be positive")
    if expected_episodes is not None:
        _require(type(expected_episodes) is int and expected_episodes > 0 and episode_count == expected_episodes,
                 "Unexpected number of episodes")
    _require(info["splits"].get("train") == f"0:{episode_count}", "Dataset train split mismatch")
    features = info["features"]
    expected_features = lerobot_features(capture)
    video_keys = {f"observation.images.{name}" for name in cameras}
    _require({key for key, value in features.items() if value["dtype"] == "video"} == video_keys,
             "Camera feature set mismatch")
    _require(set(features) == set(expected_features) | _RESERVED, "Dataset feature schema mismatch")
    statistics = _read(root / "meta/stats.json")
    _require(set(statistics) == set(features), "Missing official feature statistics")
    for key, expected in expected_features.items():
        actual = features[key]
        _require(actual["dtype"] == expected["dtype"] and list(actual["shape"]) == list(expected["shape"])
                 and actual.get("names") == expected.get("names"), f"Feature schema mismatch: {key}")
    for key in _RESERVED:
        _require(features[key]["dtype"] == ("float32" if key == "timestamp" else "int64")
                 and features[key]["shape"] == [1], f"Reserved feature schema mismatch: {key}")

    episode_paths = sorted((root / "meta/episodes").rglob("*.parquet"))
    _require(bool(episode_paths), "Missing official episode metadata")
    episodes = [row for file in episode_paths for row in pq.read_table(file).to_pylist()]
    _require([row["episode_index"] for row in episodes] == list(range(episode_count)),
             "Episode metadata indices are incomplete or duplicated")
    extensions = provenance["episodes"]
    bindings = sources["source_episodes"]
    _require([p["episode_index"] for p in extensions] == list(range(episode_count))
             and [p["episode_index"] for p in bindings] == list(range(episode_count)),
             "NEPES source episode indices are inconsistent")
    identities = [(p["staging_episode_directory"], p["episode_json_sha256"]) for p in bindings]
    _require(len(set(identities)) == episode_count, "The same source episode was included more than once")
    offset = 0
    references = defaultdict(list)
    data_references = {}
    for index, (episode, extension, binding) in enumerate(zip(episodes, extensions, bindings)):
        length = episode["length"]
        _require(type(length) is int and length > 0 and episode["dataset_from_index"] == offset
                 and episode["dataset_to_index"] == offset + length, "Episode data offsets mismatch")
        _require(extension["success"] is True and extension["source_episode_id"] == binding["source_episode_id"]
                 and extension["frames_sha256"] == binding["frames_sha256"]
                 and binding["frame_count"] == length, "Source episode binding mismatch")
        for key in ("frames_sha256", "episode_json_sha256"):
            _require(isinstance(binding[key], str) and len(binding[key]) == 64
                     and all(c in "0123456789abcdef" for c in binding[key]), f"Invalid source hash: {key}")
        data_name = info["data_path"].format(chunk_index=episode["data/chunk_index"],
                                             file_index=episode["data/file_index"])
        data_references[index] = _relative(root, data_name)
        for key in video_keys:
            prefix = f"videos/{key}/"
            video_name = info["video_path"].format(
                video_key=key, chunk_index=episode[prefix+"chunk_index"], file_index=episode[prefix+"file_index"])
            _relative(root, video_name)
            start, end = episode[prefix+"from_timestamp"], episode[prefix+"to_timestamp"]
            _require(math.isfinite(start) and math.isfinite(end) and start >= 0
                     and abs((end-start)*30-length) < 1e-5
                     and abs(start*30-round(start*30)) < 1e-5, "Episode video offsets mismatch")
            references[video_name].append((start, end, length, key))
        offset += length
    _require(offset == total, "Episode lengths do not sum to dataset frame count")
    actual_videos = {str(p.relative_to(root)) for p in (root / "videos").rglob("*.mp4")}
    _require(actual_videos == set(references), "Missing or unreferenced MP4 streams")
    video_jobs = []
    for name, segments in sorted(references.items()):
        segments.sort()
        cursor = 0.
        for start, end, _, _ in segments:
            _require(abs(start-cursor) < 1e-6, "Video episode intervals overlap or leave a gap")
            cursor = end
        key = segments[0][3]
        _require(all(segment[3] == key for segment in segments), "Different cameras share one video reference")
        width, height = cameras[key.removeprefix("observation.images.")]
        video_jobs.append((root / name, name, key, width, height, sum(segment[2] for segment in segments)))
    video_started = perf_counter()
    with closing(_ordered_bounded_map(_decode_video_job, video_jobs,
                                     workers=video_workers, max_pending=max(1, video_workers))) as decoded:
        videos = list(decoded)
    video_finished = perf_counter()

    tasks = pq.read_table(root / "meta/tasks.parquet").to_pandas()
    _require(sorted(tasks["task_index"].tolist()) == list(range(info["total_tasks"])), "Task indices mismatch")
    task_names = {int(row["task_index"]): str(name) for name, row in tasks.iterrows()}
    starts = np.array([row["dataset_from_index"] for row in episodes], dtype=np.int64)
    ends = np.array([row["dataset_to_index"] for row in episodes], dtype=np.int64)
    data_paths = sorted((root / "data").rglob("*.parquet"))
    _require(set(data_paths) == set(data_references.values()), "Data Parquet files and episode references differ")
    expected_columns = set(features) - video_keys
    payload_digests = [dict() for _ in episodes]
    global_index = 0
    for data_path in data_paths:
        parquet = pq.ParquetFile(data_path)
        _require(set(parquet.schema_arrow.names) == expected_columns, "Parquet column set mismatch")
        for batch in parquet.iter_batches(batch_size=4096):
            values = batch.to_pydict()
            n = batch.num_rows
            global_indices = np.arange(global_index, global_index+n)
            ep_indices = np.searchsorted(ends, global_indices, side="right")
            _require(np.all(ep_indices < episode_count), "Parquet contains extra frames")
            local_indices = global_indices-starts[ep_indices]
            for key, expected in (("index", global_indices), ("episode_index", ep_indices), ("frame_index", local_indices)):
                _require(np.array_equal(np.asarray(values[key]), expected), f"Parquet {key} is non-contiguous")
            nominal = local_indices/30.
            tolerance = np.maximum(1e-6, np.abs(nominal)*np.finfo(np.float32).eps)
            _require(np.all(np.abs(np.asarray(values["timestamp"])-nominal) <= tolerance), "Parquet timestamps mismatch")
            arrays = {}
            for key in sorted(expected_columns):
                dtype = np.dtype(features[key]["dtype"])
                arr = np.asarray(values[key])
                width = math.prod(features[key]["shape"])
                if width == 1 and arr.shape == (n,):
                    arr = arr[:, None]
                _require(arr.shape == (n, width) and np.issubdtype(arr.dtype, np.number)
                         and np.isfinite(arr).all(), f"Invalid numeric Parquet values: {key}")
                if np.issubdtype(dtype, np.integer):
                    _require(np.array_equal(arr, arr.astype(dtype)), f"Non-integral values in {key}")
                arrays[key] = np.ascontiguousarray(arr, dtype=dtype)
                if key not in ("index", "episode_index", "task_index"):
                    for ep_index in np.unique(ep_indices):
                        digest = payload_digests[ep_index].setdefault(key, sha256())
                        digest.update(arr[ep_indices == ep_index].astype(dtype).tobytes())
            _require(np.all(arrays["success"] == 1), "A failed frame entered the success dataset")
            _require(np.array_equal(arrays["action"],
                                   np.concatenate([arrays["action.policy"], arrays["action.upper_target"]], axis=1)),
                     "Combined action differs from recorded policy/upper targets")
            for key in video_keys:
                name = key.removeprefix("observation.images.")
                _require(np.all(abs(arrays[f"observation.render_time.{name}"]-arrays["observation.sim_time"]) <= 1e-5),
                         f"Camera physics/render time mismatch: {name}")
            for ep_index in np.unique(ep_indices):
                _require(data_references[ep_index] == data_path, "Episode metadata references the wrong data file")
                mask = ep_indices == ep_index
                _require(np.all(arrays["task_id"][mask] == extensions[ep_index]["task_id"]), "Task ID provenance mismatch")
                for task in np.unique(arrays["task_index"][mask]):
                    _require(int(task) in task_names and task_names[int(task)] in episodes[ep_index]["tasks"],
                             "Parquet task reference mismatch")
            global_index += n
    _require(global_index == total, "Parquet row count mismatch")
    payload_hashes = {str(i): sha256(json.dumps(
        {key: digest.hexdigest() for key, digest in sorted(digests.items())}, sort_keys=True).encode()).hexdigest()
        for i, digests in enumerate(payload_digests)}

    inventory_started = perf_counter()
    files = _inventory(root)
    if (root / RECEIPT_PATH).exists():
        previous = _read(root / RECEIPT_PATH)
        _require(previous.get("schema") == RECEIPT_SCHEMA and previous.get("verified") is True
                 and previous.get("files") == files and previous.get("source_episodes") == bindings,
                 "Published storage receipt no longer matches dataset files/source bindings")
    return {
        "schema": RECEIPT_SCHEMA, "verified": True, "kind": sources["kind"],
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_version": f"lerobot=={LEROBOT_VERSION}", "repo_id": sources["repo_id"],
        "total_episodes": episode_count, "total_frames": total, "fps": 30, "camera_keys": sorted(video_keys),
        "checks": {"all_mp4_fully_decoded": True, "parquet_metadata_consistent": True,
                   "camera_schema_consistent": True, "source_binding_consistent": True},
        "source_episodes": bindings, "episode_payload_sha256": payload_hashes,
        "videos": videos, "files": files, "total_bytes": sum(item["size_bytes"] for item in files),
        "verification_timing_seconds": {
            "schema_and_references": video_started - started,
            "full_video_decode": video_finished - video_started,
            "parquet_and_payload": inventory_started - video_finished,
            "file_hashes_and_existing_receipt": perf_counter() - inventory_started,
            "total": perf_counter() - started,
        },
        "video_verification_workers": video_workers,
        "limitations": ["Encoding is the official lossy H.264 export; PNG pixel identity is not asserted.",
                        "Storage integrity does not independently establish physical task success."],
    }


def _destination(output_root: str | Path) -> tuple[Path, Path]:
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"Destination already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output, output.parent / f".{output.name}.compact-{uuid4().hex}"


def _publish(temporary: Path, output: Path, receipt: dict):
    # A receipt becomes visible only after every underlying file is durable.
    for path in _files(temporary):
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    for path in sorted((p for p in temporary.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        _fsync_directory(path)
    _write_durable(temporary / RECEIPT_PATH, receipt)
    _fsync_directory(temporary)
    if output.exists():
        raise FileExistsError(f"Destination appeared during export: {output}")
    os.rename(temporary, output)
    _fsync_directory(output.parent)


def export_verified_episode(staging_root: str | Path, output_root: str | Path, repo_id: str | None = None) -> dict:
    """Export exactly one committed success and atomically publish its receipt."""
    source = Path(staging_root).resolve(strict=True)
    episodes = sorted((source / "episodes").glob("*/episode.json"))
    _require(len(episodes) == 1, "Verified shard export requires exactly one committed episode")
    episode_path = episodes[0]
    episode = _read(episode_path)
    frames_path = episode_path.parent / "frames.jsonl"
    binding = {"episode_index": 0, "source_episode_id": episode["episode_id"],
               "staging_episode_directory": str(episode_path.parent),
               "episode_json_sha256": _sha(episode_path), "frames_sha256": _sha(frames_path),
               "frame_count": episode["frame_count"]}
    _require(binding["frames_sha256"] == episode["frames_sha256"], "Source frames changed after commit")
    output, temporary = _destination(output_root)
    _require(not output.is_relative_to(source) and not source.is_relative_to(output), "Export destination overlaps staging")
    chosen_repo = repo_id or episode["schema"]["repo_id"]
    try:
        writer_started = perf_counter()
        export_lerobot_v3(source, temporary, repo_id=chosen_repo)
        writer_finished = perf_counter()
        _require(_sha(episode_path) == binding["episode_json_sha256"] and _sha(frames_path) == binding["frames_sha256"],
                 "Source episode changed while exporting")
        _write_durable(temporary / SOURCES_PATH,
                       {"schema": SOURCES_SCHEMA, "kind": "single_episode_shard", "repo_id": chosen_repo,
                        "source_episodes": [binding]})
        receipt = verify_dataset(temporary, expected_episodes=1)
        receipt["export_timing_seconds"] = {
            "official_writer": writer_finished - writer_started,
            "source_binding_and_verification": perf_counter() - writer_finished,
            "before_durable_publication": perf_counter() - writer_started,
        }
        _publish(temporary, output, receipt)
        return receipt
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def merge_verified_shards(shards: Sequence[str | Path], output_root: str | Path,
                          repo_id: str | None = None, collection_manifest: str | Path | None = None) -> dict:
    """Aggregate already encoded one-episode shards via the official API."""
    from lerobot.datasets.aggregate import aggregate_datasets
    _official_version()
    roots = [Path(path).resolve(strict=True) for path in shards]
    _require(roots and len(set(roots)) == len(roots), "Provide distinct verified shard paths")
    receipts = []
    for root in roots:
        _require((root / RECEIPT_PATH).is_file(), f"Shard has no published verification receipt: {root}")
        receipt = verify_dataset(root, expected_episodes=1)
        _require(receipt["kind"] == "single_episode_shard", "Only one-episode verified shards can be merged")
        receipts.append(receipt)
    extensions = [_read(root / "meta/nepes_collection.json") for root in roots]
    capture = extensions[0]["capture_schema"]
    _require(all(extension["capture_schema"] == capture for extension in extensions), "Incompatible shard capture schemas")
    output, temporary = _destination(output_root)
    _require(not any(output.is_relative_to(root) or root.is_relative_to(output) for root in roots),
             "Merge destination overlaps a source shard")
    chosen_repo = repo_id or receipts[0]["repo_id"]
    try:
        # Every MP4 exceeds one byte. This official size-rotation branch copies
        # each source file verbatim instead of concatenating/re-encoding videos.
        aggregate_datasets(repo_ids=[receipt["repo_id"] for receipt in receipts],
                           aggr_repo_id=chosen_repo, roots=roots, aggr_root=temporary,
                           video_files_size_in_mb=1e-6)
        merged_episodes, bindings = [], []
        for index, (extension, receipt) in enumerate(zip(extensions, receipts)):
            merged_episodes.append({**deepcopy(extension["episodes"][0]), "episode_index": index})
            bindings.append({**deepcopy(receipt["source_episodes"][0]), "episode_index": index})
        _write_durable(temporary / "meta/nepes_collection.json", {
            **deepcopy(extensions[0]), "episodes": merged_episodes,
            "official_aggregation": f"lerobot=={LEROBOT_VERSION}:aggregate_datasets"})
        source_metadata = {"schema": SOURCES_SCHEMA, "kind": "merged_dataset", "repo_id": chosen_repo,
                           "source_episodes": bindings, "source_shards": [
                               {"path": str(root), "receipt_sha256": _sha(root / RECEIPT_PATH)}
                               for root in roots]}
        if collection_manifest is not None:
            manifest_path = Path(collection_manifest).resolve(strict=True)
            manifest = _read(manifest_path)
            _write_durable(temporary / "meta/nepes_stationary_collection.json", manifest)
            source_metadata["collection_manifest"] = {"path": str(manifest_path), "sha256": _sha(manifest_path)}
        _write_durable(temporary / SOURCES_PATH, source_metadata)
        receipt = verify_dataset(temporary, expected_episodes=len(roots))
        for index, source_receipt in enumerate(receipts):
            _require(receipt["episode_payload_sha256"][str(index)] == source_receipt["episode_payload_sha256"]["0"],
                     "Official aggregation changed a recorded state/action/camera value")
        def video_hashes(items):
            hashes = {item["path"]: item["sha256"] for item in items["files"]}
            return Counter((video["camera_key"], hashes[video["path"]]) for video in items["videos"])
        expected_videos = Counter()
        for source_receipt in receipts:
            expected_videos.update(video_hashes(source_receipt))
        _require(video_hashes(receipt) == expected_videos, "Aggregation changed encoded MP4 bytes")
        receipt["checks"].update(video_bytes_preserved_without_reencoding=True, episode_numeric_payloads_preserved=True)
        _publish(temporary, output, receipt)
        return receipt
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--staging-root", required=True)
    export.add_argument("--output-root", required=True)
    export.add_argument("--repo-id")
    merge = commands.add_parser("merge")
    merge.add_argument("--shards-json", type=Path, required=True, help="JSON list of verified shard directories")
    merge.add_argument("--output-root", required=True)
    merge.add_argument("--repo-id")
    merge.add_argument("--collection-manifest")
    verify = commands.add_parser("verify")
    verify.add_argument("--dataset", required=True)
    verify.add_argument("--expected-episodes", type=int)
    args = parser.parse_args(argv)
    if args.command == "export":
        receipt = export_verified_episode(args.staging_root, args.output_root, args.repo_id)
    elif args.command == "merge":
        shards = _read(args.shards_json)
        _require(isinstance(shards, list) and all(isinstance(path, str) for path in shards), "shards-json must be a path list")
        receipt = merge_verified_shards(shards, args.output_root, args.repo_id, args.collection_manifest)
    else:
        receipt = verify_dataset(args.dataset, args.expected_episodes)
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
