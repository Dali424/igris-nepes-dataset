"""Success-gated episodic staging and official LeRobot v3 export.

The Isaac process writes lossless RGB PNGs and low-dimensional JSONL incrementally.
These are explicitly an internal staging format, never a fabricated LeRobot tree.
An isolated export environment calls the official LeRobotDataset v0.4.4 API to
produce v3 Parquet, MP4, metadata, statistics and episode offsets.

The MP4 timeline is 30 fps. Actual physics/render timestamps are separate columns:
at nominal 250 Hz, tick scheduling has less than 4 ms skew. The simulator's
small cumulative float32 clock drift is retained in the actual timestamps.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import shutil
from time import monotonic
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from .constants import COMMAND_JOINT_NAMES

STAGING_SCHEMA = "igris_c_nepes_episode_staging_v1"
LEROBOT_VERSION = "0.4.4"
DEFAULT_CAMERA_SIZES = {"head": (1280, 720), "hand": (640, 480), "third_person": (1280, 720)}
FOUR_CAMERA_SIZES = {"head": (1280, 720), "wrist_left": (640, 480),
                     "wrist_right": (640, 480), "third_person": (1280, 720)}


def validate_camera_sizes(camera_sizes: Mapping[str, Sequence[int]], *,
                          require_standard_sizes: bool = False) -> dict[str, tuple[int, int]]:
    """Accept complete legacy or both-wrist layouts without mixing their keys.

    Small even resolutions remain available for synthetic writer/export tests.
    Collection admission additionally requires the agreed recording resolutions.
    The input order is retained so existing staging image fingerprints stay stable.
    """
    if not isinstance(camera_sizes, Mapping):
        raise ValueError("Camera sizes must be a mapping")
    layout = next((sizes for sizes in (DEFAULT_CAMERA_SIZES, FOUR_CAMERA_SIZES)
                   if set(camera_sizes) == set(sizes)), None)
    if layout is None:
        raise ValueError("Require exactly head, hand, third_person or "
                         "head, wrist_left, wrist_right, third_person cameras")
    normalized = {}
    for name, size in camera_sizes.items():
        if not isinstance(size, (list, tuple)) or len(size) != 2 or any(
            type(value) is not int or value <= 0 or value % 2 for value in size
        ):
            raise ValueError("Camera sizes must be positive even (width, height) integers")
        normalized[name] = tuple(size)
        if require_standard_sizes and normalized[name] != layout[name]:
            raise ValueError(f"Camera {name} requires standard resolution {layout[name]}")
    return normalized


def validate_declared_camera_layout(camera_sizes: Mapping[str, Sequence[int]], camera_layout: str) -> None:
    """Check an explicit run declaration; callers skip this for historical runs without one."""
    layouts = {"head_hand": DEFAULT_CAMERA_SIZES, "head_both_wrists": FOUR_CAMERA_SIZES}
    if not isinstance(camera_layout, str) or camera_layout not in layouts:
        raise ValueError("camera_layout must be head_hand or head_both_wrists")
    if set(camera_sizes) != set(layouts[camera_layout]):
        raise ValueError("Recorded cameras do not match the declared camera_layout")


def validate_physics_clock(*, sample_time: float, first_time: float, elapsed_ticks: int,
                           physics_hz: int, previous_time: float | None = None,
                           interval_ticks: int | None = None) -> None:
    """Allow step-size representation drift, but reject clock jumps or lost ticks.

    Isaac 5.1 advances its clock by float32(1 / physics_hz) per callback.
    Check each observed interval tightly as well as the whole episode; a large
    accumulated roundoff budget must never hide a one-tick discontinuity.
    """
    elapsed = elapsed_ticks / physics_hz
    budget = max(1e-8, abs(elapsed) * np.finfo(np.float32).eps)
    if not math.isfinite(sample_time) or abs(sample_time - first_time - elapsed) > budget:
        raise ValueError("Capture timing mismatch: physics clock drift exceeds step-size roundoff")
    if previous_time is not None:
        if interval_ticks is None or interval_ticks <= 0:
            raise ValueError("Capture timing mismatch: physics ticks must advance")
        interval = interval_ticks / physics_hz
        if abs(sample_time - previous_time - interval) > max(1e-8, interval * np.finfo(np.float32).eps):
            raise ValueError("Capture timing mismatch: physics clock interval does not match ticks")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=_jsonable) + "\n")
    temporary.replace(path)


def _names(names: Sequence[str], count: int, label: str) -> tuple[str, ...]:
    result = tuple(names)
    if not result or len(result) != count or len(set(result)) != count or any(not isinstance(n, str) or not n for n in result):
        raise ValueError(f"{label} requires {count} unique nonempty names")
    return result


def _vector(values: Any, names: Sequence[str], label: str) -> np.ndarray:
    if isinstance(values, Mapping):
        if set(values) != set(names):
            raise ValueError(f"{label} named values must exactly match the declared schema")
        values = [values[name] for name in names]
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (len(names),) or not np.isfinite(vector).all():
        raise ValueError(f"{label} requires {len(names)} finite values")
    return vector


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("episode_id must be a plain identifier without path separators")
    return value


class DatasetWriter:
    """Stream one active episode; only finish_episode(success=True) commits it.

    Action schema is the actual 13-D policy output plus the named
    upper-body targets for the selected robot model. The policy output must
    be supplied by the caller; it is never reconstructed from joint positions.
    """
    def __init__(self, root: str | Path, repo_id: str, joint_names: Sequence[str],
                 policy_action_names: Sequence[str], *, fps: int = 30, physics_hz: int = 250,
                 control_hz: int = 50, camera_sizes: Mapping[str, Sequence[int]] | None = None,
                 keep_failed_frames: bool = False,
                 upper_target_names: Sequence[str] = COMMAND_JOINT_NAMES,
                 image_workers: int = 0, max_pending_frames: int = 8):
        if fps != 30:
            raise ValueError("This collection contract requires 30 fps")
        if (not isinstance(control_hz, int) or control_hz <= 0 or not isinstance(physics_hz, int)
                or physics_hz < fps or physics_hz % control_hz):
            raise ValueError("physics_hz must support 30 fps and integer control decimation")
        if not repo_id or not isinstance(repo_id, str):
            raise ValueError("repo_id is required")
        if type(image_workers) is not int or image_workers < 0:
            raise ValueError("image_workers must be a nonnegative integer")
        if type(max_pending_frames) is not int or max_pending_frames < 1:
            raise ValueError("max_pending_frames must be a positive integer")
        self.root = Path(root).resolve()
        self.repo_id, self.fps, self.physics_hz, self.control_hz = repo_id, fps, physics_hz, control_hz
        self.joint_names = _names(joint_names, len(joint_names), "Joint state")
        self.upper_target_names = _names(upper_target_names, len(upper_target_names), "Upper targets")
        self.policy_action_names = _names(policy_action_names, 13, "Policy action")
        self.camera_sizes = validate_camera_sizes(DEFAULT_CAMERA_SIZES if camera_sizes is None else camera_sizes)
        self.keep_failed_frames = keep_failed_frames
        for directory in (".staging", "episodes", "failures"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        self._active: dict[str, Any] | None = None
        self._stream = None
        self.image_workers, self.max_pending_frames = image_workers, max_pending_frames
        self._image_writer = None
        self._last_image_writer_metrics = None

    @property
    def image_writer_metrics(self) -> dict | None:
        """Current or most recently joined asynchronous writer measurements."""
        if self._image_writer is not None:
            return self._image_writer.metrics
        return None if self._last_image_writer_metrics is None else dict(self._last_image_writer_metrics)

    @property
    def schema(self) -> dict[str, Any]:
        return {"schema_version": STAGING_SCHEMA, "repo_id": self.repo_id, "fps": self.fps,
                "physics_hz": self.physics_hz, "control_hz": self.control_hz,
                "joint_names": list(self.joint_names), "policy_action_names": list(self.policy_action_names),
                "upper_target_names": list(self.upper_target_names),
                "camera_sizes": {key: list(value) for key, value in self.camera_sizes.items()},
                "timestamp_contract": "nominal video k/30; actual sim/render times retained; ceil physics tick",
                "upper_target_semantics": f"{len(self.upper_target_names)} requested targets; actual {len(self.joint_names)} positions/velocities recorded separately"}

    def begin_episode(self, episode_id: str, task_id: int, task: str, provenance: Mapping[str, Any]) -> None:
        if self._active is not None:
            raise RuntimeError("Finish or abort the active episode first")
        episode_id = _identifier(episode_id)
        if not isinstance(task_id, int) or task_id not in range(1, 6) or not isinstance(task, str) or not task.strip():
            raise ValueError("Require task_id 1..5 and a nonempty task description")
        provenance = dict(provenance)
        if provenance.get("kind") not in ("original", "augmentation"):
            raise ValueError("provenance.kind must be original or augmentation")
        if provenance["kind"] == "augmentation" and (
            not provenance.get("parent_episode_id") or not provenance.get("augmentations")
        ):
            raise ValueError("Augmentation requires parent_episode_id and explicit augmentations")
        # Validate serializability before creating a partially initialized episode.
        provenance = json.loads(json.dumps(provenance, allow_nan=False, default=_jsonable))
        directory = self.root / ".staging" / episode_id
        if directory.exists() or (self.root / "episodes" / episode_id).exists() or (self.root / "failures" / f"{episode_id}.json").exists():
            raise FileExistsError(f"Episode identifier already exists: {episode_id}")
        directory.mkdir()
        for name in self.camera_sizes:
            (directory / "images" / name).mkdir(parents=True)
        self._active = {"episode_id": episode_id, "task_id": task_id, "task": task,
                        "provenance": provenance, "schema": self.schema, "frame_count": 0,
                        "status": "recording", "success": False, "phases": {}}
        _write_json(directory / "episode.json", self._active)
        self._stream = (directory / "frames.jsonl").open("w", encoding="utf-8")
        if self.image_workers:
            from .async_image_writer import BoundedImageWriter
            self._image_writer = BoundedImageWriter(self.image_workers, self.max_pending_frames)
            self._last_image_writer_metrics = None

    def add_frame(self, *, sim_time_s: float, joint_positions: Any, joint_velocities: Any,
                  policy_action: Any, upper_targets: Any, images: Mapping[str, np.ndarray],
                  phase_id: int, phase_name: str, physics_step: int, command_time_s: float,
                  camera_metadata: Mapping[str, Mapping[str, Any]],
                  annotations: Mapping[str, Any] | None = None) -> None:
        """Append one frame; optional ``annotations`` add new JSON fields to the row (e.g. holding_arm)."""
        if self._active is None:
            raise RuntimeError("Call begin_episode first")
        extra = {}
        if annotations is not None:
            if (not isinstance(annotations, Mapping) or not annotations
                    or any(not isinstance(key, str) or not key for key in annotations)):
                raise ValueError("Frame annotations must be a nonempty mapping of named fields")
            extra = json.loads(json.dumps(dict(annotations), allow_nan=False, default=_jsonable))
        if not np.isfinite([sim_time_s, command_time_s]).all():
            raise ValueError("Capture and command timestamps must be finite")
        if not isinstance(physics_step, int) or physics_step < 0:
            raise ValueError("physics_step must be a nonnegative integer")
        if not isinstance(phase_id, int) or phase_id < 0 or not isinstance(phase_name, str) or not phase_name:
            raise ValueError("Require a nonnegative phase_id and its name")
        count = self._active["frame_count"]
        if count == 0:
            first_step, first_time = physics_step, float(sim_time_s)
        else:
            first_step, first_time = self._active["first_physics_step"], self._active["first_sim_time_s"]
        expected_step = first_step + (count * self.physics_hz + self.fps - 1) // self.fps
        if physics_step != expected_step:
            raise ValueError(f"Capture timing mismatch: expected physics tick {expected_step}; got {physics_step}")
        validate_physics_clock(sample_time=float(sim_time_s), first_time=first_time,
            elapsed_ticks=physics_step - first_step, physics_hz=self.physics_hz,
            previous_time=self._active.get("last_sim_time_s"),
            interval_ticks=(physics_step - self._active["last_physics_step"] if count else None))
        if command_time_s > sim_time_s + 1e-5 or sim_time_s - command_time_s > 1 / self.control_hz + 1e-5:
            raise ValueError("Policy command timestamp must identify the current 50 Hz held command")
        state = _vector(joint_positions, self.joint_names, "joint_positions")
        velocity = _vector(joint_velocities, self.joint_names, "joint_velocities")
        action = _vector(policy_action, self.policy_action_names, "policy_action")
        upper = _vector(upper_targets, self.upper_target_names, "upper_targets")
        if set(images) != set(self.camera_sizes) or set(camera_metadata) != set(self.camera_sizes):
            raise ValueError("Every frame requires exactly the declared RGB cameras and their render metadata")
        camera_info = {}
        for name, (width, height) in self.camera_sizes.items():
            rgb = np.asarray(images[name])
            if rgb.dtype != np.uint8 or rgb.shape != (height, width, 3):
                raise ValueError(f"{name} requires uint8 RGB {(height, width, 3)}, got {rgb.shape}/{rgb.dtype}")
            info = dict(camera_metadata[name])
            render_time = float(info["render_time_s"])
            pose = np.asarray(info["pose_w_xyz_wxyz"], dtype=float)
            intrinsics = np.asarray(info["intrinsics"], dtype=float)
            if not math.isfinite(render_time) or abs(render_time - sim_time_s) > 1e-5:
                raise ValueError(f"Camera {name} is not synchronized to this physics frame")
            if pose.shape != (7,) or intrinsics.shape != (3, 3) or not np.isfinite(pose).all() or not np.isfinite(intrinsics).all():
                raise ValueError(f"Camera {name} requires finite pose and intrinsics")
            camera_info[name] = info
        previous_phase = self._active["phases"].get(str(phase_id))
        if previous_phase is not None and previous_phase != phase_name:
            raise ValueError("A phase_id cannot change its meaning within an episode")
        directory = self.root / ".staging" / self._active["episode_id"]
        paths, destinations = {}, {}
        for name in self.camera_sizes:
            relative = Path("images") / name / f"{count:06d}.png"
            paths[name] = str(relative)
            destinations[directory / relative] = images[name]
        row = {"frame_index": count, "timestamp": count / self.fps, "sim_time_s": float(sim_time_s),
               "physics_step": physics_step, "command_time_s": float(command_time_s),
               "capture_skew_s": float(sim_time_s - first_time - count / self.fps),
               "joint_positions": state.tolist(), "joint_velocities": velocity.tolist(),
               "policy_action": action.tolist(), "upper_targets": upper.tolist(),
               "phase_id": phase_id, "phase_name": phase_name, "images": paths, "cameras": camera_info}
        if set(extra) & set(row):
            raise ValueError("Frame annotations cannot replace recorded frame fields")
        row.update(extra)
        serialized = json.dumps(row, ensure_ascii=False, allow_nan=False, default=_jsonable) + "\n"
        # Validate the complete metadata before scheduling any side effects.
        # A rejected frame can then be retried without concurrent writes to
        # the same image paths.
        if self._image_writer is None:
            from PIL import Image
            for path, rgb in destinations.items():
                Image.fromarray(np.ascontiguousarray(rgb)).save(path, compress_level=1)
        else:
            self._image_writer.submit(destinations)
        self._stream.write(serialized)
        self._stream.flush()
        self._active.update(frame_count=count + 1, first_physics_step=first_step, first_sim_time_s=first_time,
                            last_sim_time_s=float(sim_time_s), last_physics_step=physics_step)
        if count == 0:
            self._active["camera_configuration"] = {
                name: {"prim_path": info.get("prim_path"), "camera_axes": info.get("camera_axes"),
                       "attachment_prim_path": info.get("attachment_prim_path"),
                       "pose_source": info.get("pose_source"),
                       "clipping_range_m": info.get("clipping_range_m"),
                       **{key: info[key] for key in (
                           "lens_distortion_model", "opencv_distortion_coefficients",
                           "intrinsics_source", "calibration_source") if key in info},
                       "resolution": list(self.camera_sizes[name])} for name, info in camera_info.items()
            }
        self._active["phases"][str(phase_id)] = phase_name

    def finish_episode(self, success: bool, metrics: Mapping[str, Any], reason: str = "") -> Path:
        if self._active is None:
            raise RuntimeError("No active episode")
        if type(success) is not bool:
            raise ValueError("success must be an explicit boolean outcome from task validation")
        if success and self._active["frame_count"] == 0:
            raise ValueError("An empty episode cannot be successful")
        report = {**self._active, "success": success, "status": "success" if success else "failure",
                  "metrics": dict(metrics), "reason": str(reason)}
        json.dumps(report, allow_nan=False, default=_jsonable)
        directory = self.root / ".staging" / report["episode_id"]
        image_error = None
        if self._image_writer is not None:
            try:
                self._image_writer.close()
            except Exception as exc:
                image_error = exc
                success = False
                prefix = report["reason"] + "; " if report["reason"] else ""
                report.update(success=False, status="failure",
                              reason=f"{prefix}image_writer_failed: {exc}")
            finally:
                self._last_image_writer_metrics = self._image_writer.metrics
                self._image_writer = None
            report["image_writing"] = dict(self._last_image_writer_metrics)
        self._stream.close()
        self._stream = None
        report["frames_sha256"] = sha256((directory / "frames.jsonl").read_bytes()).hexdigest()
        if success:
            _write_json(directory / "episode.json", report)
            result = self.root / "episodes" / report["episode_id"]
            directory.replace(result)
        else:
            result = self.root / "failures" / f"{report['episode_id']}.json"
            _write_json(result, report)
            if self.keep_failed_frames:
                _write_json(directory / "episode.json", report)
                failed = self.root / "failed_frames"
                failed.mkdir(exist_ok=True)
                directory.replace(failed / report["episode_id"])
            else:
                shutil.rmtree(directory)
        self._active = None
        if image_error is not None:
            raise image_error
        return result

    def finish(self, success: bool, metrics: Mapping[str, Any], reason: str = "") -> Path:
        return self.finish_episode(success, metrics, reason)

    def abort(self, reason: str, metrics: Mapping[str, Any] | None = None) -> Path | None:
        if self._active is None:
            return None
        return self.finish_episode(False, {} if metrics is None else metrics, reason)

    def finalize(self) -> None:
        if self._active is not None:
            self.abort("Writer finalized before an explicit successful task outcome")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self._active is not None:
            self.abort(str(exc) if exc is not None else "Episode context exited without finish")


def lerobot_features(schema: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    cameras = validate_camera_sizes(schema["camera_sizes"])
    def vector(dtype, names):
        return {"dtype": dtype, "shape": (len(names),), "names": list(names)}
    features = {
        "observation.state": vector("float32", schema["joint_names"]),
        "observation.joint_velocity": vector("float32", schema["joint_names"]),
        "action": vector("float32", [f"policy/{n}" for n in schema["policy_action_names"]] +
                         [f"upper_target/{n}" for n in schema["upper_target_names"]]),
        "action.policy": vector("float32", schema["policy_action_names"]),
        "action.upper_target": vector("float32", schema["upper_target_names"]),
        "observation.sim_time": vector("float64", ["seconds"]),
        "observation.command_time": vector("float64", ["seconds"]),
        "observation.capture_skew": vector("float64", ["seconds"]),
        "observation.physics_step": vector("int64", ["tick"]),
        "task_id": vector("int64", ["nepes_task_id"]),
        "phase_id": vector("int64", ["phase_id"]),
        "success": vector("int64", ["validated_episode_success"]),
        "episode_scope_id": vector("int64", ["0_legacy_task_1_grasp_carry_2_stationary_task"]),
        "full_task_completed": vector("int64", ["completed_declared_task_definition"]),
        "provenance.is_augmented": vector("int64", ["is_augmented"]),
    }
    for name, (width, height) in cameras.items():
        features[f"observation.images.{name}"] = {"dtype": "video", "shape": (height, width, 3),
                                                   "names": ["height", "width", "channels"]}
        features[f"observation.camera_pose.{name}"] = vector("float64", ["x", "y", "z", "qw", "qx", "qy", "qz"])
        features[f"observation.camera_intrinsics.{name}"] = vector("float64", [f"K{i}{j}" for i in range(3) for j in range(3)])
        features[f"observation.render_time.{name}"] = vector("float64", ["seconds"])
    return features


def _export_frame(row: dict, report: dict, directory: Path) -> dict[str, Any]:
    from PIL import Image
    cameras = validate_camera_sizes(report["schema"]["camera_sizes"])
    if set(row["images"]) != set(cameras) or set(row["cameras"]) != set(cameras):
        raise ValueError("Every exported frame must contain exactly the declared images and camera measurements")
    action = np.asarray(row["policy_action"], dtype=np.float32)
    upper = np.asarray(row["upper_targets"], dtype=np.float32)
    frame = {
        # The official API generates its reserved timestamp as frame_index/fps.
        # Actual simulation/render timestamps remain explicit custom features.
        "task": report["task"],
        "observation.state": np.asarray(row["joint_positions"], dtype=np.float32),
        "observation.joint_velocity": np.asarray(row["joint_velocities"], dtype=np.float32),
        "action": np.concatenate([action, upper]), "action.policy": action, "action.upper_target": upper,
        "observation.sim_time": np.array([row["sim_time_s"]], dtype=np.float64),
        "observation.command_time": np.array([row["command_time_s"]], dtype=np.float64),
        "observation.capture_skew": np.array([row["capture_skew_s"]], dtype=np.float64),
        "observation.physics_step": np.array([row["physics_step"]], dtype=np.int64),
        "task_id": np.array([report["task_id"]], dtype=np.int64),
        "phase_id": np.array([row["phase_id"]], dtype=np.int64),
        "success": np.array([1], dtype=np.int64),
        "episode_scope_id": np.array([{"task": 0, "grasp_carry": 1, "stationary_task": 2}[
            report["provenance"].get("episode_scope", "task")]], dtype=np.int64),
        "full_task_completed": np.array([int(report.get("metrics", {}).get(
            "full_task_completed", report["provenance"].get("episode_scope", "task") == "task"))], dtype=np.int64),
        "provenance.is_augmented": np.array([int(report["provenance"]["kind"] == "augmentation")], dtype=np.int64),
    }
    for name, size in cameras.items():
        relative = row["images"][name]
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Staged image path escapes its episode")
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != size:
                raise ValueError(f"Staged camera {name} has incorrect image mode or resolution")
            frame[f"observation.images.{name}"] = np.array(image, dtype=np.uint8)
        camera = row["cameras"][name]
        frame[f"observation.camera_pose.{name}"] = np.asarray(camera["pose_w_xyz_wxyz"], dtype=np.float64)
        frame[f"observation.camera_intrinsics.{name}"] = np.asarray(camera["intrinsics"], dtype=np.float64).reshape(9)
        frame[f"observation.render_time.{name}"] = np.array([camera["render_time_s"]], dtype=np.float64)
    return frame


def _ordered_bounded_map(function, items, *, workers: int, max_pending: int):
    """Yield results in source order with bounded work and joined failure cleanup.

    Unlike Executor.map on Python 3.11, this never submits the entire episode.
    A consumer must close this generator when it stops before exhaustion.
    """
    if type(workers) is not int or workers < 0 or type(max_pending) is not int or max_pending < 1:
        raise ValueError("workers must be a nonnegative integer and max_pending a positive integer")
    if workers == 0:
        yield from map(function, items)
        return
    source = iter(items)
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="nepes-export")
    pending = deque()
    try:
        for _ in range(max_pending):
            try:
                item = next(source)
            except StopIteration:
                break
            pending.append(executor.submit(function, item))
        while pending:
            yield pending.popleft().result()
            try:
                item = next(source)
            except StopIteration:
                continue
            pending.append(executor.submit(function, item))
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def _export_rows(stream):
    for index, line in enumerate(stream):
        row = json.loads(line)
        if row["frame_index"] != index or abs(row["timestamp"] - index / 30) > 1e-10:
            raise ValueError("Non-contiguous staged frame sequence")
        yield row


def _wait_for_streaming_capacity(encoder, *, timeout_s: float = 120.0):
    """Backpressure the sole producer before the pinned writer can drop RGB.

    LeRobot 0.4.4's feed_frame drops after a 0.1s queue timeout. Its queues have
    one producer here and consumers only remove items, so an available slot
    remains available until this caller supplies the next frame or finish
    sentinel. Wait without increasing queue sizes; dead or stalled encoder
    threads are errors, never skipped RGB.
    """
    if encoder is None or not encoder._episode_active:
        return
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Encoder capacity timeout must be finite and positive")
    deadline = monotonic() + timeout_s
    for key, queue in encoder._frame_queues.items():
        with queue.not_full:
            while queue.maxsize > 0 and queue._qsize() >= queue.maxsize:
                if not encoder._threads[key].is_alive():
                    raise RuntimeError(f"Video encoder stopped while waiting for capacity: {key}")
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Video encoder capacity wait exceeded {timeout_s:g}s: {key}")
                queue.not_full.wait(timeout=min(0.1, remaining))


def export_lerobot_v3(staging_root: str | Path, output_root: str | Path, *, repo_id: str | None = None,
                      decode_workers: int = 4, prefetch_frames: int = 8) -> Path:
    """Export all committed successes using official APIs; publish locally atomically.

    No Hub upload is performed. Failed/uncommitted episodes are never read.
    Run this function in scripts/planning/install_dataset.sh's isolated venv.
    PNG decoding is prefetched in bounded frame bundles; only the caller thread
    invokes the official writer. Use decode_workers=0 for the serial reference.
    """
    if (type(decode_workers) is not int or decode_workers < 0
            or type(prefetch_frames) is not int or prefetch_frames < 1):
        raise ValueError("decode_workers must be nonnegative and prefetch_frames positive integers")
    from importlib.metadata import version
    try:
        from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    except ImportError as error:
        raise RuntimeError("Install the isolated dataset environment with scripts/planning/install_dataset.sh") from error
    if version("lerobot") != LEROBOT_VERSION or CODEBASE_VERSION != "v3.0":
        raise RuntimeError(f"Exporter requires official lerobot=={LEROBOT_VERSION} with CODEBASE_VERSION v3.0")
    source, output = Path(staging_root).resolve(), Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"Export destination already exists: {output}")
    episodes = sorted((source / "episodes").glob("*/episode.json"))
    if not episodes:
        raise ValueError("No successfully committed episodes to export")
    reports = [json.loads(path.read_text()) for path in episodes]
    schema = reports[0]["schema"]
    cameras = validate_camera_sizes(schema["camera_sizes"])
    for path, report in zip(episodes, reports):
        if report.get("success") is not True or report.get("status") != "success":
            raise ValueError(f"Refusing a non-success episode in committed collection: {path}")
        if report["schema"] != schema or schema.get("schema_version") != STAGING_SCHEMA:
            raise ValueError("Episodes have incompatible staging schemas")
        if report["frame_count"] <= 0:
            raise ValueError("Cannot export empty episodes")
        actual_hash = sha256((path.parent / "frames.jsonl").read_bytes()).hexdigest()
        if actual_hash != report["frames_sha256"]:
            raise ValueError(f"Staged frame records changed after commit: {path.parent}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.export-{uuid4().hex}"
    dataset = None
    try:
        dataset = LeRobotDataset.create(repo_id=repo_id or schema["repo_id"], root=temporary,
                                        robot_type="igris_c_hugwbc", fps=30,
                                        features=lerobot_features(schema), use_videos=True,
                                        vcodec="h264", video_backend="pyav", streaming_encoding=True,
                                        encoder_threads=2, metadata_buffer_size=1)
        provenance = []
        for episode_index, (path, report) in enumerate(zip(episodes, reports)):
            seen = 0
            with (path.parent / "frames.jsonl").open(encoding="utf-8") as stream:
                with closing(_ordered_bounded_map(
                        lambda row: _export_frame(row, report, path.parent), _export_rows(stream),
                        workers=decode_workers, max_pending=prefetch_frames)) as frames:
                    for frame in frames:
                        _wait_for_streaming_capacity(dataset._streaming_encoder)
                        dataset.add_frame(frame)
                        if any(dataset._streaming_encoder._dropped_frames.values()):
                            raise RuntimeError("Official streaming encoder dropped a frame; refusing export")
                        seen += 1
            if seen != report["frame_count"]:
                raise ValueError("Staged frame count does not match the committed report")
            # The pinned SDK's finish_episode inserts None without a timeout.
            # Reserve one slot per queue before entering it: after the final
            # RGB there are no other producers that could fill those slots.
            _wait_for_streaming_capacity(dataset._streaming_encoder)
            dataset.save_episode(parallel_encoding=False)
            provenance.append({"episode_index": episode_index, "source_episode_id": report["episode_id"],
                               "task_id": report["task_id"], "phases": report["phases"],
                               "success": True, "metrics": report["metrics"],
                               "camera_configuration": report.get("camera_configuration", {}),
                               "provenance": report["provenance"], "frames_sha256": report["frames_sha256"]})
        dataset.finalize()
        dataset = None
        # Extension metadata: official LeRobot files above are exclusively API-produced.
        _write_json(temporary / "meta" / "nepes_collection.json",
                    {"extension_schema": "igris_c_nepes_provenance_v1", "capture_schema": schema,
                     "official_writer": f"lerobot=={LEROBOT_VERSION}", "episodes": provenance})
        info = json.loads((temporary / "meta" / "info.json").read_text())
        if info["codebase_version"] != "v3.0" or info["total_episodes"] != len(reports):
            raise RuntimeError("Official export metadata failed final verification")
        video_keys = {f"observation.images.{name}" for name in cameras}
        actual_video_keys = {key for key, feature in info["features"].items() if feature["dtype"] == "video"}
        if (actual_video_keys != video_keys or not list((temporary / "data").rglob("*.parquet"))
                or any(not list((temporary / "videos" / key).rglob("*.mp4")) for key in video_keys)):
            raise RuntimeError("Official export did not produce Parquet and all camera MP4 streams")
        temporary.replace(output)
    except Exception:
        if dataset is not None:
            try:
                dataset.finalize()
            except Exception:
                pass
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Export successful NEPES staging episodes through official LeRobot v3 APIs")
    parser.add_argument("--staging-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-id")
    arguments = parser.parse_args()
    print(export_lerobot_v3(arguments.staging_root, arguments.output_root, repo_id=arguments.repo_id))


if __name__ == "__main__":
    main()
