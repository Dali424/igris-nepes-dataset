"""Validate a successful joint plan and export the existing 50 Hz NPZ format.

This module intentionally depends only on NumPy and the standard library. It
does not plan motion or assert collision clearance: that is the planner's job.
All untouched command joints are held at the supplied initial snapshot.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .constants import (
    COMMAND_FPS,
    COMMAND_JOINT_NAMES,
    HUGWBC_COMMAND_SCHEMA,
    PLANNABLE_JOINT_NAMES,
    TELEOPERATION_JOINT_NAMES,
    TELEOPERATION_RANGES,
    WRIST_HAND_JOINT_NAMES,
)


@dataclass(frozen=True)
class ExportResult:
    npz_path: Path
    manifest_path: Path
    frame_count: int
    duration_s: float
    source_duration_s: float


def bounded_quintic_joint_path(start, goal, velocity_limits, acceleration_limits,
                               jerk_limits, *, sample_dt=.02, validation_dt=.004):
    """Time a straight joint-space segment without claiming collision safety.

    The quintic progress law has zero endpoint velocity/acceleration. Analytic
    maxima bound the entire reference curve. Validation positions additionally
    sample the *piecewise linear exported commands* at the requested physics
    interval; the caller must check every such pose against self/world geometry.
    """
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    if start.ndim != 1 or not len(start) or goal.shape != start.shape or not np.isfinite([start, goal]).all():
        raise ValueError("Require equal finite nonempty joint vectors")
    limits = [np.asarray(value, dtype=float) for value in (velocity_limits, acceleration_limits, jerk_limits)]
    if any(value.shape != start.shape or not np.isfinite(value).all() or np.any(value <= 0) for value in limits):
        raise ValueError("Require positive finite velocity, acceleration and jerk limits per joint")
    sample_dt = _positive_finite(sample_dt, "sample_dt")
    validation_dt = _positive_finite(validation_dt, "validation_dt")
    ratio = sample_dt / validation_dt
    if validation_dt > sample_dt or abs(ratio - round(ratio)) > 1e-9:
        raise ValueError("Validation interval must divide output interval")
    delta = goal - start
    distance = np.abs(delta)
    maximum_second_derivative = 10 * math.sqrt(3) / 3
    duration = max(sample_dt, float(np.max(1.875 * distance / limits[0])),
                   float(np.max(np.sqrt(maximum_second_derivative * distance / limits[1]))),
                   float(np.max(np.cbrt(60 * distance / limits[2]))))
    steps = int(math.ceil(duration / sample_dt - 1e-12))
    duration = steps * sample_dt
    time = np.arange(steps + 1) * sample_dt
    u = time / duration
    progress = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
    positions = start + progress[:, None] * delta
    positions[0], positions[-1] = start, goal
    validation_time = np.arange(steps * round(ratio) + 1) * validation_dt
    validation_positions = np.column_stack([np.interp(validation_time, time, positions[:, i])
                                            for i in range(len(start))])
    maxima = {"velocity_rad_s": 1.875 * distance / duration,
              "acceleration_rad_s2": maximum_second_derivative * distance / duration ** 2,
              "jerk_rad_s3": 60 * distance / duration ** 3}
    return {"positions": positions, "validation_positions": validation_positions,
            "sample_dt": sample_dt, "validation_dt": validation_dt,
            "metadata": {"method": "straight_joint_segment_quintic_time_law",
                "duration_s": duration, "analytic_reference_maxima": {k: v.tolist() for k, v in maxima.items()},
                "velocity_limits_rad_s": limits[0].tolist(),
                "acceleration_limits_rad_s2": limits[1].tolist(), "jerk_limits_rad_s3": limits[2].tolist(),
                "validation_sampling": "piecewise_linear_between_exported_command_positions",
                "continuous_collision_proof": False, "collision_checked": False,
                "trajectory_optimized": False}}


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _positive_finite(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be positive and finite.")
    return value


def _check_positions(
    positions: np.ndarray,
    names: Sequence[str],
    limits: Mapping[str, tuple[float, float]],
) -> None:
    for column, name in enumerate(names):
        if name not in limits:
            raise ValueError(f"Missing position limits for {name!r}.")
        bounds = np.asarray(limits[name], dtype=np.float64)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
            raise ValueError(f"Invalid position limits for {name!r}.")
        if np.any(positions[:, column] < bounds[0]) or np.any(positions[:, column] > bounds[1]):
            raise ValueError(f"Position limit exceeded for {name!r}.")


def _check_rates(
    positions: np.ndarray,
    names: Sequence[str],
    dt: float,
    limits: Mapping[str, float],
) -> None:
    rates = np.abs(np.diff(positions, axis=0)) / dt
    for column, name in enumerate(names):
        if name not in limits:
            raise ValueError(f"Missing velocity limit for {name!r}.")
        limit = _positive_finite(limits[name], f"Velocity limit for {name!r}")
        if np.any(rates[:, column] > limit + 1e-9):
            raise ValueError(f"Velocity limit exceeded for {name!r}.")


def _forward_velocity(positions: np.ndarray) -> np.ndarray:
    """Derivative of the exported linear segments; final command holds still."""
    velocity = np.zeros_like(positions)
    velocity[:-1] = np.diff(positions, axis=0) * COMMAND_FPS
    return velocity


def export_arm_npz(output_path, *, joint_names, positions, sample_dt,
                   initial_positions, position_limits, velocity_limits,
                   planner_success, snapshot_hash, planner_info=None,
                   start_tolerance=1e-4):
    """Export a checked 50 Hz single-arm path without a legacy hand schema.

    The live adapter holds the other joints using its actual robot model. This
    artifact cannot be loaded by the legacy 51-joint HugWBC NPZ replay reader.
    """
    if not isinstance(planner_success, (bool, np.bool_)) or not planner_success:
        raise ValueError("Cannot export a failed or unverified plan")
    names = tuple(joint_names)
    from .robot_model import arm_joint_names
    if len(names) != 7 or not any(set(names) == set(arm_joint_names(a)) for a in ("left", "right")):
        raise ValueError("Require exactly one complete named seven-joint arm")
    if not math.isfinite(sample_dt) or abs(sample_dt - 1 / COMMAND_FPS) > 1e-8:
        raise ValueError("Arm path must already be sampled at 50 Hz")
    if not math.isfinite(start_tolerance) or start_tolerance < 0:
        raise ValueError("Invalid start tolerance")
    source = np.asarray(positions, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 7 or len(source) < 1 or not np.isfinite(source).all():
        raise ValueError("Require finite nonempty frames by seven positions")
    initial = {name: float(value) for name, value in initial_positions.items()}
    if not all(isinstance(n, str) and n and math.isfinite(v) for n, v in initial.items()):
        raise ValueError("Initial state must contain finite named positions")
    if not set(names) <= initial.keys():
        raise ValueError("Initial state lacks planned joints")
    if np.any(np.abs(source[0] - [initial[n] for n in names]) > start_tolerance):
        raise ValueError("Plan start does not match initial snapshot")
    if (not isinstance(snapshot_hash, str) or len(snapshot_hash) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in snapshot_hash)):
        raise ValueError("snapshot_hash must be a SHA-256 digest")
    _check_positions(source, names, position_limits)
    _check_rates(source, names, sample_dt, velocity_limits)
    output_path = Path(output_path)
    manifest_path = output_path.with_suffix(".manifest.json")
    if output_path.suffix.lower() != ".npz":
        raise ValueError("Output must be .npz")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite an existing path or manifest")
    duration = (len(source) - 1) / COMMAND_FPS
    manifest = {"schema": "igris_c_named_arm_path_v1", "fps": COMMAND_FPS,
                "joint_names": list(names), "frame_count": len(source), "duration_s": duration,
                "snapshot_sha256": snapshot_hash, "initial_joint_positions": initial,
                "initial_state_sha256": hashlib.sha256(_json_bytes(initial)).hexdigest(),
                "planner": dict(planner_info or {}), "physics_execution_validated": False}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staged, created = [], []
    try:
        with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".npz", delete=False) as stream:
            staged.append(Path(stream.name))
            np.savez_compressed(stream, schema=np.asarray(manifest["schema"]),
                                joint_names=np.asarray(names), joint_pos=source,
                                joint_vel=_forward_velocity(source),
                                timestamp=np.arange(len(source)) / COMMAND_FPS,
                                fps=np.asarray(COMMAND_FPS),
                                provenance_json=np.asarray(_json_bytes(manifest).decode()))
        manifest["npz_sha256"] = hashlib.sha256(staged[0].read_bytes()).hexdigest()
        with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".json", delete=False) as stream:
            staged.append(Path(stream.name))
            stream.write(json.dumps(manifest, indent=2, allow_nan=False).encode() + b"\n")
        for temporary, destination in zip(staged, (output_path, manifest_path)):
            os.link(temporary, destination)
            created.append(destination)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in staged:
            path.unlink(missing_ok=True)
    return ExportResult(output_path, manifest_path, len(source), duration, duration)


def export_hugwbc_npz(
    output_path: str | Path,
    *,
    joint_names: Sequence[str],
    positions: np.ndarray,
    sample_dt: float,
    initial_positions: Mapping[str, float],
    position_limits: Mapping[str, tuple[float, float]],
    velocity_limits: Mapping[str, float],
    planner_success: bool,
    snapshot_hash: str,
    planner_info: Mapping[str, Any] | None = None,
    start_tolerance: float = 1e-4,
) -> ExportResult:
    """Write a successful stationary-base plan and a provenance JSON manifest.

    ``sample_dt`` is the planner's actual interval in seconds. Piecewise linear
    interpolation samples that timeline at 50 Hz; the final source waypoint is
    included on the next 20 ms tick when the duration is not an exact multiple.
    There is no global time scaling and no substitution of the planner's dt.
    Velocities describe the exported intervals, with zero at the final hold.
    A one-waypoint successful plan is a valid, zero-duration hold.

    Position/velocity bounds are required for every planned joint. Position
    bounds supplied for held command joints are checked too. Upper-body values
    must additionally fit the existing replay reader's limits. Acceleration,
    jerk, dynamics, and collisions are not revalidated here.

    ``snapshot_hash`` must be the SHA-256 hex digest of the source snapshot;
    its producer supplies it. A second digest covers initial joint values.
    Existing outputs are never intentionally overwritten. Validation occurs
    before either output file is created.
    """
    if not isinstance(planner_success, (bool, np.bool_)) or not planner_success:
        raise ValueError("Cannot export a failed or unverified plan.")
    names = tuple(joint_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("joint_names must contain non-empty joint names.")
    if len(set(names)) != len(names):
        raise ValueError("joint_names contains duplicate names.")
    unsupported = sorted(set(names) - set(PLANNABLE_JOINT_NAMES))
    if unsupported:
        raise ValueError(f"Unsupported planned joints: {unsupported}.")
    sample_dt = _positive_finite(sample_dt, "sample_dt")
    start_tolerance = float(start_tolerance)
    if not math.isfinite(start_tolerance) or start_tolerance < 0:
        raise ValueError("start_tolerance must be non-negative and finite.")
    source = np.asarray(positions, dtype=np.float64)
    if source.ndim != 2 or source.shape[0] < 1 or source.shape[1] != len(names):
        raise ValueError("positions must have non-empty shape (frames, len(joint_names)).")
    if not np.isfinite(source).all():
        raise ValueError("positions contains NaN or Inf.")
    missing = sorted(set(COMMAND_JOINT_NAMES) - initial_positions.keys())
    if missing:
        raise ValueError(f"Initial snapshot is missing command joints: {missing}.")
    initial = {name: float(value) for name, value in initial_positions.items()}
    if any(not isinstance(name, str) or not name for name in initial):
        raise ValueError("Initial snapshot joint names must be non-empty strings.")
    if not all(math.isfinite(value) for value in initial.values()):
        raise ValueError("Initial snapshot contains NaN or Inf.")
    for column, name in enumerate(names):
        if abs(source[0, column] - initial[name]) > start_tolerance:
            raise ValueError(f"Plan start does not match initial snapshot for {name!r}.")
    if (
        not isinstance(snapshot_hash, str)
        or len(snapshot_hash) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in snapshot_hash)
    ):
        raise ValueError("snapshot_hash must be a SHA-256 hex digest.")
    _check_positions(source, names, position_limits)
    _check_rates(source, names, sample_dt, velocity_limits)

    source_duration = (source.shape[0] - 1) * sample_dt
    if not math.isfinite(source_duration):
        raise ValueError("Source duration is not finite.")
    # Ceil with a small floating-point allowance avoids an extra tick for e.g.
    # 0.14 / 0.02. The final value is copied explicitly to preserve the endpoint.
    tick_count = source_duration * COMMAND_FPS
    if not math.isfinite(tick_count):
        raise ValueError("Source duration cannot be represented on the export timeline.")
    minimum_frames = 2 if source_duration > 0 else 1
    frame_count = max(minimum_frames, int(math.ceil(tick_count - 1e-12)) + 1)
    timestamps = np.arange(frame_count, dtype=np.float64) / COMMAND_FPS
    source_times = np.arange(source.shape[0], dtype=np.float64) * sample_dt
    sampled = np.column_stack(
        [np.interp(timestamps, source_times, source[:, column]) for column in range(len(names))]
    )
    sampled[-1] = source[-1]
    _check_positions(sampled, names, position_limits)
    _check_rates(sampled, names, 1 / COMMAND_FPS, velocity_limits)
    command = np.tile([initial[name] for name in COMMAND_JOINT_NAMES], (frame_count, 1))
    canonical_columns = {name: column for column, name in enumerate(COMMAND_JOINT_NAMES)}
    for column, name in enumerate(names):
        command[:, canonical_columns[name]] = sampled[:, column]
    upper_count = len(TELEOPERATION_JOINT_NAMES)
    teleop = command[:, :upper_count]
    wrist_hand = command[:, upper_count:]
    # The reader stores its bounds as float32. Respect exactly those bounds,
    # including snapshot values lying at a float32 joint-limit endpoint.
    canonical_limits = dict(zip(TELEOPERATION_JOINT_NAMES, np.asarray(TELEOPERATION_RANGES, dtype=np.float32)))
    _check_positions(teleop, TELEOPERATION_JOINT_NAMES, canonical_limits)
    held_with_limits = [name for name in COMMAND_JOINT_NAMES if name not in names and name in position_limits]
    if held_with_limits:
        _check_positions(
            command[:, [canonical_columns[name] for name in held_with_limits]],
            held_with_limits,
            position_limits,
        )

    duration = float(timestamps[-1])
    manifest = {
        "manifest_version": 1,
        "schema_version": HUGWBC_COMMAND_SCHEMA,
        "fps": COMMAND_FPS,
        "sample_dt_s": 1 / COMMAND_FPS,
        "frame_count": frame_count,
        "duration_s": duration,
        "source_sample_dt_s": sample_dt,
        "source_frame_count": int(source.shape[0]),
        "source_duration_s": source_duration,
        "planned_joint_names": list(names),
        "held_joint_names": [name for name in COMMAND_JOINT_NAMES if name not in names],
        "base_velocity": [0.0, 0.0, 0.0],
        "resampling": "piecewise_linear_on_source_timeline_with_final_ceiling_tick",
        "velocity_convention": "forward_segment_derivative_and_zero_final_hold",
        "snapshot_sha256": snapshot_hash.lower(),
        "initial_positions_sha256": hashlib.sha256(_json_bytes(initial)).hexdigest(),
        "planner_success": True,
        "planner": dict(planner_info or {}),
        "validation": {
            "position_limits": {name: list(map(float, position_limits[name])) for name in names},
            "velocity_limits": {name: float(velocity_limits[name]) for name in names},
            "start_tolerance_rad": start_tolerance,
        },
    }
    manifest_bytes = _json_bytes(manifest)  # Reject non-JSON/non-finite provenance before writing.
    payload = {
        "schema_version": np.asarray(HUGWBC_COMMAND_SCHEMA),
        "fps": np.asarray(COMMAND_FPS, dtype=np.float64),
        "base_velocity": np.zeros((frame_count, 3), dtype=np.float64),
        "teleop_joint_names": np.asarray(TELEOPERATION_JOINT_NAMES),
        "teleop_joint_pos": teleop,
        "teleop_joint_vel": _forward_velocity(teleop),
        "wrist_hand_joint_names": np.asarray(WRIST_HAND_JOINT_NAMES),
        "wrist_hand_joint_pos": wrist_hand,
        "wrist_hand_joint_vel": _forward_velocity(wrist_hand),
        "timestamp": timestamps,
        "planned_joint_names": np.asarray(names),
        "source_sample_dt": np.asarray(sample_dt, dtype=np.float64),
        "provenance_json": np.asarray(manifest_bytes.decode("utf-8")),
    }
    output_path = Path(output_path).expanduser()
    if output_path.suffix.lower() != ".npz":
        raise ValueError("output_path must have an .npz suffix.")
    manifest_path = output_path.with_suffix(".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite an existing NPZ or manifest.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    created: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".npz", delete=False) as stream:
            staged.append(Path(stream.name))
            np.savez_compressed(stream, **payload)
        manifest["npz_sha256"] = hashlib.sha256(staged[0].read_bytes()).hexdigest()
        with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".json", delete=False) as stream:
            staged.append(Path(stream.name))
            stream.write(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")
        # A hard link fails atomically if another process created the output;
        # unlike replace(), it cannot silently overwrite an existing artifact.
        for temporary, destination in zip(staged, (output_path, manifest_path)):
            os.link(temporary, destination)
            created.append(destination)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in staged:
            path.unlink(missing_ok=True)
    return ExportResult(output_path, manifest_path, frame_count, duration, source_duration)
