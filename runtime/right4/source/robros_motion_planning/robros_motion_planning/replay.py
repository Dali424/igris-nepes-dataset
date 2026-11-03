"""Initial-state checks for snapshot-specific cuRobo replay commands.

Call after reset and before issuing the first command, while USD object
transforms are synchronized with physics. This is a one-time compatibility
check, not runtime tracking, fall detection, or collision-geometry validation.
It does not monitor objects, base drift, or tracking error during execution and
does not establish dynamic stability. The caller must stop on a reset/fall.

Legacy command files without provenance, and non-cuRobo provenance, return None.
NumPy is the only nonstandard import until object poses require USD lazily.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .constants import HUGWBC_COMMAND_SCHEMA
from .snapshot import load_snapshot, matrix_to_pose, pose_to_matrix, snapshot_digest


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _parse_json(text: str, label: str) -> dict:
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_keys)
    except (ValueError, TypeError) as error:
        raise ValueError(f"Invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _numpy(value: Any, label: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"Current {label} contains nonfinite values")
    return array


def _check_pose(current: Any, expected: Any, label: str) -> None:
    current_matrix = pose_to_matrix(current)
    expected_matrix = pose_to_matrix(expected)
    distance = float(np.linalg.norm(current_matrix[:3, 3] - expected_matrix[:3, 3]))
    rotation = current_matrix[:3, :3].T @ expected_matrix[:3, :3]
    angle = float(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
    if distance > 0.002 or angle > 0.01:
        raise ValueError(
            f"Planned replay {label} differs from snapshot: {distance:.6f} m, "
            f"{angle:.6f} rad (limits 0.002 m and 0.01 rad)"
        )


def _validate_robot(robot: Any, snapshot: dict) -> None:
    expected = snapshot["joint_positions"]
    names = tuple(robot.joint_names)
    if len(names) != len(set(names)):
        raise ValueError("Current robot has duplicate joint names")
    if set(names) != set(expected):
        raise ValueError(
            "Current robot joint names differ from the complete snapshot: "
            f"missing={sorted(set(expected) - set(names))}, "
            f"extra={sorted(set(names) - set(expected))}"
        )
    positions = _numpy(robot.data.joint_pos, "joint positions")
    if positions.shape != (1, len(names)):
        raise ValueError("Planned replay requires exactly one robot environment and all joint positions")
    errors = np.abs(positions[0] - np.asarray([expected[name] for name in names]))
    if np.any(errors > 1e-4):
        index = int(np.argmax(errors))
        raise ValueError(
            f"Planned replay joint {names[index]} differs from snapshot by "
            f"{errors[index]:.6f} rad (limit 0.0001 rad)"
        )
    bodies = tuple(robot.body_names)
    if len(bodies) != len(set(bodies)) or "base_link" not in bodies:
        raise ValueError("Current robot must expose an unambiguous base_link body")
    body_positions = _numpy(robot.data.body_pos_w, "body positions")
    body_quaternions = _numpy(robot.data.body_quat_w, "body quaternions")
    if body_positions.shape != (1, len(bodies), 3) or body_quaternions.shape != (1, len(bodies), 4):
        raise ValueError("Current robot body pose arrays have invalid shapes")
    index = bodies.index("base_link")
    pose = np.r_[body_positions[0, index], body_quaternions[0, index]]
    _check_pose(pose, snapshot["planning_base_pose_w"], "base_link pose")


def _validate_objects(stage: Any, snapshot: dict) -> None:
    objects = snapshot.get("objects", {})
    if not isinstance(objects, dict):
        raise ValueError("Snapshot objects must be keyed by full USD prim paths")
    if not objects:
        return
    from pxr import Usd, UsdGeom

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for path, record in objects.items():
        if (
            not isinstance(path, str) or not path.startswith("/")
            or not isinstance(record, dict) or record.get("prim_path", path) != path
        ):
            raise ValueError("Snapshot object keys must match their full USD prim paths")
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsActive():
            raise ValueError(f"Planned replay object is absent or inactive: {path}")
        # USD uses row-vector matrices; snapshot helpers use column vectors.
        transform = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float).T
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(f"Current object transform is invalid: {path}")
        # Capture stores a rigid pose beside the authored object scale. Apply
        # the same polar decomposition, so offsets/pose checks stay in meters.
        left, _, right = np.linalg.svd(transform[:3, :3])
        rigid = transform.copy()
        rigid[:3, :3] = left @ np.diag([1.0, 1.0, np.linalg.det(left @ right)]) @ right
        _check_pose(matrix_to_pose(rigid), record["pose_w"], f"object {path}")


def validate_planned_replay(
    motion_path: str | Path,
    robot: Any,
    stage: Any,
    *,
    scene_usd: str | Path,
) -> dict | None:
    """Validate cuRobo provenance and reset state; return its planner metadata.

    The snapshot's complete joint set (51 joints for the current IGRIS-C model)
    must match by name within 1e-4 rad. Base and object poses must match within
    2 mm and 0.01 rad. Only the named base_link body defines the planning frame;
    the articulation root need not coincide with it. Relative snapshot paths in
    provenance resolve beside the NPZ. A cuRobo artifact requires its original
    manifest, snapshot, and hashed meshes. Scene path identity is checked after
    absolute resolution, and the source USD file hash is checked when provided.

    Successful validation checks the initial state only. Scene geometry edits
    in memory and changes during replay require a new capture and plan.
    """
    motion_path = Path(motion_path).expanduser().resolve()
    with np.load(motion_path, allow_pickle=False) as data:
        if "provenance_json" not in data.files:
            return None
        raw = np.asarray(data["provenance_json"])
        if raw.size != 1 or raw.dtype.kind not in ("U", "S"):
            raise ValueError("NPZ provenance_json must be one JSON string")
        text = raw.reshape(-1)[0]
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        embedded = _parse_json(str(text), "embedded provenance")
    planner = embedded.get("planner", {})
    if not isinstance(planner, dict):
        raise ValueError("Embedded planner metadata must be a JSON object")
    if planner.get("backend") != "curobo":
        return None
    manifest_path = motion_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Planned replay requires its manifest: {manifest_path}")
    manifest = _parse_json(manifest_path.read_text(encoding="utf-8"), "replay manifest")
    if manifest.get("npz_sha256") != _file_digest(motion_path):
        raise ValueError("Planned replay NPZ SHA256 mismatch")
    without_file_hash = {key: value for key, value in manifest.items() if key != "npz_sha256"}
    if _canonical_json(without_file_hash) != _canonical_json(embedded):
        raise ValueError("Embedded provenance and replay manifest differ")
    if embedded.get("schema_version") != HUGWBC_COMMAND_SCHEMA or embedded.get("planner_success") is not True:
        raise ValueError("Planned replay provenance does not describe a successful command plan")
    snapshot_reference = planner.get("snapshot_path")
    if not isinstance(snapshot_reference, str) or not snapshot_reference:
        raise ValueError("Planned replay provenance is missing its source snapshot_path")
    snapshot_path = Path(snapshot_reference).expanduser()
    if not snapshot_path.is_absolute():
        snapshot_path = motion_path.parent / snapshot_path
    snapshot = load_snapshot(snapshot_path)
    if embedded.get("snapshot_sha256") != snapshot_digest(snapshot):
        raise ValueError("Planned replay source snapshot SHA256 differs from provenance")
    initial = {name: float(value) for name, value in snapshot["joint_positions"].items()}
    initial_digest = hashlib.sha256(_canonical_json(initial).encode("utf-8")).hexdigest()
    if embedded.get("initial_positions_sha256") != initial_digest:
        raise ValueError("Planned replay initial joint digest differs from source snapshot")
    source_scene = snapshot.get("scene_usd")
    if not isinstance(source_scene, str) or not source_scene:
        raise ValueError("Planned replay source snapshot is missing scene_usd")
    source_scene_path = Path(source_scene).expanduser().resolve()
    if scene_usd is None:
        raise ValueError("Planned replay requires --scene_usd matching its planning snapshot")
    current_scene_path = Path(scene_usd).expanduser().resolve()
    if current_scene_path != source_scene_path:
        raise ValueError("Current scene_usd path differs from the planning snapshot")
    if not current_scene_path.is_file():
        raise FileNotFoundError(current_scene_path)
    if "scene_usd_sha256" in snapshot and snapshot["scene_usd_sha256"] != _file_digest(current_scene_path):
        raise ValueError("Current scene USD SHA256 differs from the planning snapshot")
    _validate_robot(robot, snapshot)
    _validate_objects(stage, snapshot)
    return dict(planner)
