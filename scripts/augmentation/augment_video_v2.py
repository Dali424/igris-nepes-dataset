#!/usr/bin/env python3
"""Offline, single-episode LeRobot v3 photometric pilot (no physics or rendering).

Requires the collection's dataset environment: lerobot==0.4.4, av, numpy,
pyarrow. The source must be an immutable, verified one-episode archive.
Only --execute writes a NEW output. Original verification receipts are never
copied or issued for the derivative; see meta/video_augmentation_audit.json.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys
import time
from uuid import uuid4

FLAG = "provenance.is_augmented"
RECEIPT = "meta/nepes_storage_verification.json"
AUDIT = "meta/video_augmentation_audit.json"
RECIPE = {
    "name": "mild_darker_v1", "exposure_gain_linear_rgb": 0.94,
    "contrast_linear_rgb": 0.98, "contrast_pivot_linear_rgb": 0.18,
    "color_space": "decoded RGB8 interpreted as sRGB; linear-light LUT",
    "temporal_scope": "constant over the whole episode",
    "camera_scope": "identical LUT for every camera",
    "geometric_changes": False, "framewise_noise_or_jitter": False,
    "encoding": {"codec": "libx264", "pixel_format": "yuv420p", "crf": 18,
                 "preset": "veryfast", "threads": 1},
}
PRESETS = {"mild_darker_v1": RECIPE}
for _name, _gain in (("visible_darker_v2", 0.5), ("visible_brighter_v2", 2.0)):
    PRESETS[_name] = {
        "name": _name,
        "tone_curve": "sigmoid(contrast * (logit(x) + ln(brightness_odds_gain)))",
        "brightness_odds_gain": _gain, "logit_contrast": 1.15,
        "parameter_semantics": "sRGB tone-curve odds multiplier and logit slope; not physical exposure or EV",
        "color_space": "decoded RGB8 interpreted as sRGB; same monotone channel LUT",
        "endpoint_policy": "x=0 maps to 0; x=1 maps to 1 exactly",
        "temporal_scope": "constant over the whole episode",
        "camera_scope": "identical LUT for every camera",
        "geometric_changes": False, "framewise_noise_or_jitter": False,
        "encoding": deepcopy(RECIPE["encoding"]),
    }


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def inventory(root, exclude=()):
    result = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"Symlinks are unsupported: {path}")
        if path.is_file() and path.relative_to(root).as_posix() not in exclude:
            result.append({"path": path.relative_to(root).as_posix(),
                           "size_bytes": path.stat().st_size, "sha256": sha(path)})
    return result


def relative_file(root, name):
    path = root / name
    require(not Path(name).is_absolute() and path.resolve().is_relative_to(root)
            and path.is_file() and not path.is_symlink(), f"Invalid source reference: {name}")
    return path


def layout(root):
    import pyarrow.parquet as pq
    info = read_json(root / "meta/info.json")
    require(info["codebase_version"] == "v3.0" and info["total_episodes"] == 1
            and info["fps"] == 30 and info["total_frames"] > 1,
            "Pilot accepts one complete, 30 FPS LeRobot v3 episode only")
    require(FLAG in info["features"] and info["features"][FLAG]["dtype"] == "int64"
            and info["features"][FLAG]["shape"] == [1], "Required augmentation flag feature is missing or incompatible")
    ep_paths = list((root / "meta/episodes").rglob("*.parquet"))
    require(len(ep_paths) == 1, "Expected one episode metadata file")
    table = pq.read_table(ep_paths[0])
    require(table.num_rows == 1, "Expected exactly one episode metadata row")
    episode = table.to_pylist()[0]
    length = info["total_frames"]
    require(episode["episode_index"] == 0 and episode["length"] == length
            and episode["dataset_from_index"] == 0 and episode["dataset_to_index"] == length,
            "Inconsistent episode offsets")
    data_name = info["data_path"].format(chunk_index=episode["data/chunk_index"],
                                         file_index=episode["data/file_index"])
    relative_file(root, data_name)
    require(FLAG in pq.read_schema(root / data_name).names, "Required augmentation flag column is missing")
    require({p.relative_to(root).as_posix() for p in (root / "data").rglob("*.parquet")}
            == {data_name}, "Unreferenced data files")
    cameras = {}
    for key, feature in info["features"].items():
        if feature["dtype"] != "video":
            continue
        prefix = f"videos/{key}/"
        name = info["video_path"].format(video_key=key, chunk_index=episode[prefix + "chunk_index"],
                                          file_index=episode[prefix + "file_index"])
        relative_file(root, name)
        require(episode[prefix + "from_timestamp"] == 0
                and abs(episode[prefix + "to_timestamp"] - length / 30) < 1e-6,
                "Pilot does not support partial/shared video files")
        require(len(feature["shape"]) == 3 and feature["shape"][2] == 3,
                "Only RGB video features are supported")
        cameras[key] = name
    require(len(cameras) == 4 and len(set(cameras.values())) == 4, "Expected four separate RGB cameras")
    require({p.relative_to(root).as_posix() for p in (root / "videos").rglob("*.mp4")}
            == set(cameras.values()), "Unreferenced videos")
    return info, episode, data_name, ep_paths[0].relative_to(root).as_posix(), cameras


def make_plan(source, seed, preset="mild_darker_v1"):
    require(version("lerobot") == "0.4.4", "Use the pinned lerobot==0.4.4 dataset environment")
    receipt = read_json(source / RECEIPT)
    require(receipt.get("schema") == "nepes_verified_lerobot_storage_v1"
            and receipt.get("verified") is True and receipt.get("kind") == "single_episode_shard"
            and receipt.get("total_episodes") == 1, "A verified original single-episode archive is required")
    actual_inventory = inventory(source, exclude=(RECEIPT,))
    require(actual_inventory == receipt["files"], "Source archive no longer matches its published receipt")
    info, episode, _, _, cameras = layout(source)
    extension = read_json(source / "meta/nepes_collection.json")
    require(len(extension["episodes"]) == 1, "Ambiguous source provenance")
    parent_episode = extension["episodes"][0]
    prov = parent_episode["provenance"]
    require(prov.get("kind") == "original" and parent_episode.get("success") is True,
            "Pilot requires an original successful episode, not an augmentation chain")
    pair = prov["single_arm_layout"]
    family_basis = {"config_sha256": prov["config_sha256"], "scene_sha256": pair["scene_sha256"],
                    "manipulation_arm": pair["manipulation_arm"],
                    "authored_source_slot_number": pair["source_slot_number"],
                    "target_slot_number": pair["target_slot_number"],
                    "source_object_prim_path": pair["source_object_prim_path"]}
    family = "motion_" + hashlib.sha256(canonical(family_basis)).hexdigest()
    parent_digest = hashlib.sha256(canonical({"files": actual_inventory, "receipt": sha(source / RECEIPT)})).hexdigest()
    require(preset in PRESETS, f"Unknown preset: {preset}")
    recipe = deepcopy(PRESETS[preset])
    recipe["seed"] = seed
    recipe["seed_usage"] = "identity/reproducibility only; fixed preset, no random perturbations"
    variant = hashlib.sha256(canonical({"parent": parent_digest, "recipe": recipe})).hexdigest()
    return {"schema": "nepes_offline_video_augmentation_v1", "kind": "video_augmentation",
            "augmentation_id": "videoaug_" + variant, "derived_episode_id": "videoaug_" + variant,
            "parent_id": "archive_" + parent_digest, "parent_archive": str(source),
            "parent_receipt_sha256": sha(source / RECEIPT), "parent_files": actual_inventory,
            "parent_episode_provenance": parent_episode,
            "source_episode_id_is_not_globally_unique": True,
            "motion_family": family, "split_group": family, "motion_family_basis": family_basis,
            "split_policy": "Keep all repetitions of this motion/configuration and their derivatives in one split; no train/test independence is claimed. info.splits is only the LeRobot storage partition.",
            "recipe": recipe, "total_frames": info["total_frames"], "fps": 30,
            "camera_keys": sorted(cameras), "tasks": episode["tasks"],
            "independent_physics_execution": False, "physical_success_revalidated": False,
            "success_label_semantics": "Inherited parent trajectory outcome; no additional successful physical rollout",
            "not_scene_rerendering": True, "not_initial_placement_augmentation": True,
            "tool_sha256": sha(Path(__file__)), "lerobot_version": version("lerobot"),
            "pyav_version": version("av"), "dry_run": True,
            "dry_run_scope": "Source receipt inventory and metadata checked; full video decode occurs only on execute"}


def make_lut(recipe):
    import numpy as np
    values = np.arange(256, dtype=np.float64) / 255
    if recipe["name"] in ("visible_darker_v2", "visible_brighter_v2"):
        # Endpoints are assigned explicitly; logit is evaluated only inside (0, 1).
        output = np.empty(256, dtype=np.float64)
        output[0], output[-1] = 0.0, 1.0
        x = values[1:-1]
        logits = recipe["logit_contrast"] * (
            np.log(x) - np.log1p(-x) + np.log(recipe["brightness_odds_gain"]))
        output[1:-1] = 1.0 / (1.0 + np.exp(-logits))
        return np.rint(output * 255).astype(np.uint8)
    linear = np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)
    pivot = recipe["contrast_pivot_linear_rgb"]
    linear = np.clip((linear * recipe["exposure_gain_linear_rgb"] - pivot)
                     * recipe["contrast_linear_rgb"] + pivot, 0, 1)
    encoded = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)
    return np.rint(encoded * 255).astype(np.uint8)


def video_header(container, width, height, count):
    require(len(container.streams) == len(container.streams.video) == 1, "Expected one video-only stream")
    stream = container.streams.video[0]
    stream.codec_context.thread_count = 1
    require((stream.width, stream.height) == (width, height) and stream.average_rate == 30,
            "Video dimensions/rate do not match the dataset")
    require(stream.frames in (0, count) and stream.time_base is not None, "Invalid video count/time base")
    return stream


def transform_video(source, target, shape, count, recipe):
    import av
    import numpy as np
    height, width, _ = shape
    lut = make_lut(recipe)
    target.parent.mkdir(parents=True, exist_ok=True)
    timing = hashlib.sha256()
    stride = max(width, height) // 150 if max(width, height) >= 300 else 1
    source_hist = np.zeros((3, 256), dtype=np.int64)
    with av.open(str(source)) as original:
        incoming = video_header(original, width, height, count)
        time_base = incoming.time_base
        with av.open(str(target), "w", options={"movflags": "+faststart"}) as output:
            outgoing = output.add_stream("libx264", rate=30)
            outgoing.width, outgoing.height, outgoing.pix_fmt = width, height, "yuv420p"
            outgoing.time_base = outgoing.codec_context.time_base = time_base
            outgoing.codec_context.thread_count = 1
            outgoing.options = {"crf": "18", "preset": "veryfast", "threads": "1"}
            decoded = 0
            for frame in original.decode(incoming):
                require(not frame.is_corrupt and frame.pts is not None and frame.time_base == time_base
                        and frame.pts * time_base == Fraction(decoded, 30), "Non-contiguous/corrupt source video")
                timing.update(f"{frame.pts}/{time_base}\n".encode())
                source_rgb = frame.to_ndarray(format="rgb24")
                sampled = source_rgb[::stride, ::stride]
                for channel in range(3):
                    source_hist[channel] += np.bincount(sampled[:, :, channel].ravel(), minlength=256)
                rgb = lut[source_rgb]
                transformed = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                transformed.pts, transformed.time_base = frame.pts, time_base
                for packet in outgoing.encode(transformed):
                    output.mux(packet)
                decoded += 1
            require(decoded == count, "Source full-decode frame count mismatch")
            for packet in outgoing.encode():
                output.mux(packet)
    # Reopen the ENCODED result: stats describe what a model actually decodes.
    hist = np.zeros((3, 256), dtype=np.int64)
    output_timing = hashlib.sha256()
    with av.open(str(target)) as output:
        stream = video_header(output, width, height, count)
        require(stream.time_base == time_base, "Encoder changed video time base")
        require(stream.duration is not None and stream.duration * time_base == Fraction(count, 30),
                "Encoded video duration changed")
        decoded = 0
        stride = max(width, height) // 150 if max(width, height) >= 300 else 1
        for frame in output.decode(stream):
            require(not frame.is_corrupt and frame.pts is not None and frame.time_base == time_base
                    and frame.pts * time_base == Fraction(decoded, 30), "Encoded frame timing changed")
            output_timing.update(f"{frame.pts}/{time_base}\n".encode())
            rgb = frame.to_ndarray(format="rgb24")[::stride, ::stride]
            for channel in range(3):
                hist[channel] += np.bincount(rgb[:, :, channel].ravel(), minlength=256)
            decoded += 1
        require(decoded == count and output_timing.digest() == timing.digest(), "Video timing/count changed")
    stats = histogram_stats(hist)
    source_stats = histogram_stats(source_hist)
    luma_weights = np.asarray([0.2126, 0.7152, 0.0722])
    source_luma = float(luma_weights @ np.asarray(source_stats["mean"]).reshape(3))
    output_luma = float(luma_weights @ np.asarray(stats["mean"]).reshape(3))
    return stats, {"decoded_frames": count, "time_base": str(time_base), "pts_sha256": timing.hexdigest(),
                   "source_sha256": sha(source), "output_sha256": sha(target),
                   "fps": 30, "resolution": [width, height], "full_source_and_output_decode": True,
                   "statistics": "Every decoded output frame, RGB spatial stride matching LeRobot auto_downsample_height_width; exact uint8 histogram, float64 moments, inverse-CDF quantiles",
                   "statistics_spatial_stride": stride,
                   "output_histogram_rgb": hist.tolist(), "source_histogram_rgb": source_hist.tolist(),
                   "source_mean_luma": source_luma, "output_mean_luma": output_luma,
                   "mean_luma_delta": output_luma - source_luma}


def histogram_stats(hist):
    import numpy as np
    totals = hist.sum(axis=1)
    require(np.all(totals == totals[0]) and totals[0] > 0, "Invalid RGB histogram")
    values = np.arange(256, dtype=np.float64) / 255
    mean = (hist * values).sum(axis=1) / totals
    std = np.sqrt(np.maximum(0, (hist * values**2).sum(axis=1) / totals - mean**2))
    result = {"min": np.array([values[np.flatnonzero(row)[0]] for row in hist]),
              "max": np.array([values[np.flatnonzero(row)[-1]] for row in hist]), "mean": mean, "std": std}
    for q in (1, 10, 50, 90, 99):
        result[f"q{q:02d}"] = np.array([values[np.searchsorted(np.cumsum(row), total * q / 100)]
                                         for row, total in zip(hist, totals)])
    result = {key: value.reshape(3, 1, 1).tolist() for key, value in result.items()}
    result["count"] = [int(totals[0])]
    return result


def column_digest(table, name, feature):
    import numpy as np
    values = np.ascontiguousarray(np.asarray(table[name].to_pylist(), dtype=feature["dtype"]))
    return hashlib.sha256(values.tobytes()).hexdigest()


def write_derivative(source, temporary, plan):
    """Internal transformation; caller must validate the parent's receipt first."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.video_utils import get_video_info
    info, _, data_name, ep_name, cameras = layout(source)
    stats = read_json(source / "meta/stats.json")
    updates = {}
    video_audit = {}
    for key, relative in sorted(cameras.items()):
        print(f"Encoding and validating {key}", file=sys.stderr, flush=True)
        updates[key], video_audit[key] = transform_video(
            source / relative, temporary / relative, info["features"][key]["shape"],
            info["total_frames"], plan["recipe"])
        info["features"][key]["info"] = get_video_info(temporary / relative)
    original = pq.read_table(source / data_name)
    require(original.num_rows == info["total_frames"], "Parquet frame count mismatch")
    table = original
    if FLAG in table.column_names:
        require(info["features"][FLAG]["dtype"] == "int64", "Unsupported augmentation flag type")
        old_values = table[FLAG].to_pylist()
        require(all(value in (0, [0]) for value in old_values), "Source augmentation flags are not all zero")
        new_values = [1 if value == 0 else [1] for value in old_values]
        field = table.schema.field(FLAG)
        table = table.set_column(table.schema.get_field_index(FLAG), field, pa.array(new_values, type=field.type))
        updates[FLAG] = {key: [0.0 if key == "std" else info["total_frames"] if key == "count" else 1.0]
                         for key in stats[FLAG]}
    (temporary / data_name).parent.mkdir(parents=True, exist_ok=True)
    if FLAG in table.column_names:
        pq.write_table(table, temporary / data_name, compression="zstd")
    else:
        shutil.copyfile(source / data_name, temporary / data_name)
    reread = pq.read_table(temporary / data_name)
    require(original.schema == reread.schema and original.num_rows == reread.num_rows, "Parquet schema changed")
    preserved = [name for name in original.column_names if name != FLAG]
    require(original.select(preserved).equals(reread.select(preserved)), "Unmodified Arrow columns changed")
    numeric_digests = {}
    for key in original.column_names:
        if key == FLAG:
            require(all(value in (1, [1]) for value in reread[key].to_pylist()), "Augmentation flag was not set")
        else:
            digest = column_digest(original, key, info["features"][key])
            require(digest == column_digest(reread, key, info["features"][key]), f"Numeric values changed: {key}")
            numeric_digests[key] = digest
    episode_table = pq.read_table(source / ep_name)
    for key, values in updates.items():
        stats[key] = values
        for stat, value in values.items():
            column = f"stats/{key}/{stat}"
            require(column in episode_table.column_names, f"Missing episode statistic: {column}")
            field = episode_table.schema.field(column)
            episode_table = episode_table.set_column(episode_table.schema.get_field_index(column), field,
                                                     pa.array([value], type=field.type))
    (temporary / ep_name).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(episode_table, temporary / ep_name, compression="zstd")
    shutil.copyfile(source / "meta/tasks.parquet", temporary / "meta/tasks.parquet")
    write_json(temporary / "meta/info.json", info)
    write_json(temporary / "meta/stats.json", stats)
    # Deliberately NOT copied: nepes_collection, storage_sources, storage_verification.
    provenance = deepcopy(plan)
    provenance.pop("dry_run_scope", None)
    provenance["dry_run"] = False
    provenance["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["original_verification_receipt_applies_to_parent_only"] = True
    provenance["augmentation_flag_written"] = FLAG in original.column_names
    write_json(temporary / "meta/video_augmentation.json", provenance)
    metadata = LeRobotDatasetMetadata(repo_id="local/" + plan["augmentation_id"], root=temporary)
    require(metadata.total_frames == info["total_frames"] and metadata.total_episodes == 1,
            "Official LeRobot metadata reload failed")
    for key, values in updates.items():
        require(metadata.stats[key]["mean"].tolist() == values["mean"], f"Official stats reload mismatch: {key}")
    return {"schema": "nepes_offline_video_augmentation_audit_v1", "kind": "video_augmentation",
            "augmentation_id": plan["augmentation_id"], "parent_id": plan["parent_id"],
            "audit_scope": "Offline media/data integrity only; not a compact-storage receipt or physical success verification",
            "independent_physics_execution": False, "physical_success_revalidated": False,
            "videos": video_audit, "unchanged_numeric_column_sha256": numeric_digests,
            "modified_numeric_columns": [FLAG] if FLAG in original.column_names else [],
            "action_state_timing_bit_exact": True, "official_lerobot_metadata_reloaded": True,
            "tasks_parquet_byte_identical": sha(source / "meta/tasks.parquet") == sha(temporary / "meta/tasks.parquet"),
            "source_receipts_not_copied": not (temporary / RECEIPT).exists(),
            "files": inventory(temporary), "created_at_utc": datetime.now(timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="Verified original one-episode archive")
    parser.add_argument("--output", type=Path, required=True, help="New output dataset; must not exist")
    parser.add_argument("--seed", type=int, default=20261028, help="Recorded identity seed; preset stays identical for LEFT/RIGHT")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="mild_darker_v1",
                        help="Fixed episode-wide color preset; visible v2 presets use an endpoint-preserving sRGB odds tone curve")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Validate source inventory and print recipe; write nothing")
    mode.add_argument("--execute", action="store_true", help="CPU-only serial encode/full verification and atomic publication")
    args = parser.parse_args()
    source, output = args.source.resolve(strict=True), args.output.resolve()
    require(not output.exists() and not output.is_relative_to(source) and not source.is_relative_to(output),
            "Output must be a new directory outside the source archive")
    require(args.seed >= 0, "Seed must be nonnegative")
    plan = make_plan(source, args.seed, args.preset)
    if args.dry_run:
        brief = {k: v for k, v in plan.items() if k not in ("parent_files", "parent_episode_provenance")}
        print(json.dumps(brief, indent=2, ensure_ascii=False))
        return
    os.nice(10)
    started = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.building-{uuid4().hex}")
    temporary.mkdir()
    try:
        audit = write_derivative(source, temporary, plan)
        require(inventory(source, exclude=(RECEIPT,)) == plan["parent_files"]
                and sha(source / RECEIPT) == plan["parent_receipt_sha256"], "Source archive changed during execution")
        audit["source_unchanged_after_execution"] = True
        audit["wall_seconds"] = time.monotonic() - started
        write_json(temporary / AUDIT, audit)
        # Publish only after all media, numeric columns, statistics and source hashes pass.
        for path in temporary.rglob("*"):
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        require(not output.exists(), "Output appeared during execution")
        temporary.rename(output)
        print(json.dumps({"output": str(output), "augmentation_id": plan["augmentation_id"],
                          "frames": plan["total_frames"], "wall_seconds": audit["wall_seconds"],
                          "audit": str(output / AUDIT)}, indent=2))
    except BaseException:
        shutil.rmtree(temporary)
        raise


if __name__ == "__main__":
    main()
