"""Independent NEPES dataset goals and their configuration identities.

Two numbered catalogs exist. Catalog v2 (the default) is the user's corrected
five tasks, numbered in the historical ``nepes_tasks.TASK_IDS`` order; its task 1
picks the CRS up from an upright start and loads it into the JIG. Catalog v1
keeps the earlier component goals (task 1 lift-and-hold, task 2 held-start JIG
loading) so their profiles and admitted recordings stay reproducible. A spec's
schema_version names its catalog; a number is never resolved across catalogs.
``configured`` is not evidence of physical success.

This module contains no simulator imports and never starts collection.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .nepes_tasks import UPRIGHT_PICKUP_SEQUENCE, load_task_config
from .snapshot import pose_to_matrix


STATIONARY_SCHEMA = "igris_c_nepes_stationary_task_v1"
STATIONARY_SCHEMA_V2 = "igris_c_nepes_stationary_task_v2"
STATIONARY_SCOPE = "stationary_task"
DEFAULT_CATALOG = "v2"
# DEFAULT_CONFIG stays the catalog-v1 task-1 profile for legacy callers.
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config/nepes_stationary_crs_lift.json"
DEFAULT_CONFIG_V2 = DEFAULT_CONFIG.with_name("nepes_upright_crs_pickup_load_jig.json")
_DEFAULT_CONFIGS = MappingProxyType({
    1: DEFAULT_CONFIG,
    2: DEFAULT_CONFIG.with_name("nepes_stationary_crs_load_jig.json"),
})
_DEFAULT_CONFIGS_V2 = MappingProxyType({1: DEFAULT_CONFIG_V2})

# Catalog v1: earlier component goals; kept unchanged for reproducibility.
TASK_CATALOG = tuple(MappingProxyType(task) for task in (
    {"number": 1, "id": "crs_pick_lift", "legacy_task_id": "crs_load_jig",
     "title": "CRS 파지 후 들어올리기", "goal": "lift_hold", "configured": True},
    {"number": 2, "id": "crs_load_jig", "legacy_task_id": "crs_load_jig",
     "title": "CRS JIG 적재", "goal": "load_jig", "configured": True},
    {"number": 3, "id": "open_cleaner_cover", "legacy_task_id": "open_cleaner_cover",
     "title": "세척기 덮개 개방", "goal": "lift_off", "configured": False},
    {"number": 4, "id": "jig_extract_cleaner", "legacy_task_id": "jig_extract_cleaner",
     "title": "JIG 파지 후 인출", "goal": "extract_jig", "configured": False},
    {"number": 5, "id": "crs_final_place", "legacy_task_id": "crs_final_place",
     "title": "CRS 파지 최종 배치", "goal": "final_place", "configured": False},
))

# Catalog v2: the user's corrected five tasks in legacy TASK_IDS order.
TASK_CATALOG_V2 = tuple(MappingProxyType(task) for task in (
    {"number": 1, "id": "crs_pickup_load_jig", "legacy_task_id": "crs_load_jig",
     "title": "CRS 픽업 & JIG 적재", "goal": "pickup_load_jig", "configured": True},
    {"number": 2, "id": "open_cleaner_cover", "legacy_task_id": "open_cleaner_cover",
     "title": "세척기 도어 개방", "goal": "open_door", "configured": False},
    {"number": 3, "id": "jig_insert_cleaner", "legacy_task_id": "jig_insert_cleaner",
     "title": "JIG 파지 & 세척기 투입", "goal": "insert_jig", "configured": False},
    {"number": 4, "id": "jig_extract_cleaner", "legacy_task_id": "jig_extract_cleaner",
     "title": "세척 완료후 JIG 인출", "goal": "extract_jig", "configured": False},
    {"number": 5, "id": "crs_final_place", "legacy_task_id": "crs_final_place",
     "title": "CRS 최종 배치", "goal": "final_place", "configured": False},
))
CATALOGS = MappingProxyType({"v1": TASK_CATALOG, "v2": TASK_CATALOG_V2})
_CATALOG_SCHEMAS = MappingProxyType({"v1": STATIONARY_SCHEMA, "v2": STATIONARY_SCHEMA_V2})
_CATALOG_CONFIGS = MappingProxyType({"v1": _DEFAULT_CONFIGS, "v2": _DEFAULT_CONFIGS_V2})

_SPEC_KEYS = frozenset((
    "schema_version", "number", "id", "legacy_task_id", "goal", "hold_dwell_s",
    "minimum_lift_m", "require_hugwbc", "independent_scene",
    "max_object_linear_m_s", "max_object_angular_rad_s", "max_base_linear_m_s",
    "max_base_yaw_rad_s",
))
_MAX_SPEEDS = {
    "max_object_linear_m_s": .02, "max_object_angular_rad_s": .12,
    "max_base_linear_m_s": .03, "max_base_yaw_rad_s": .12,
}
_IDENTITY_KEYS = frozenset(("schema_version", "number", "id", "legacy_task_id", "goal",
                            "require_hugwbc", "independent_scene"))
_SPEC_KEYS_V2 = _IDENTITY_KEYS | frozenset((
    "preparation_dwell_s", "settle_dwell_s", "max_object_linear_m_s", "max_object_angular_rad_s",
    "max_base_linear_m_s", "max_base_yaw_rad_s", "slot_position_tolerance_m",
    "slot_normal_position_tolerance_m", "slot_rotation_tolerance_rad", "maximum_jig_position_drift_m",
    "maximum_jig_rotation_drift_rad", "release_separation_m", "minimum_retreat_m", "minimum_lift_m",
    "upright_joint_tolerance_rad", "maximum_pelvis_tilt_rad",
))
# The source cart POM slots hold the ring until a 0.1675 m vertical lift; no
# horizontal carry may begin below it.
_MINIMUM_UPRIGHT_LIFT_M = .17
_MAX_UPRIGHT_JOINT_TOLERANCE_RAD = .2
_MAX_PELVIS_TILT_RAD = .2
# A v2 attempt's runner wall cap exceeds the profile's own task timeout, so the
# machine's task_timeout (never the runner cap) ends an over-long attempt.
_TASK_WALL_MARGIN_S = 10.


def task_catalog(catalog: str = DEFAULT_CATALOG) -> tuple[Mapping[str, Any], ...]:
    """Return one numbered catalog; v1 and v2 numbers are never interchangeable."""
    if not isinstance(catalog, str) or catalog not in CATALOGS:
        raise ValueError("Unknown stationary catalog; use v1 or v2")
    return CATALOGS[catalog]


def stationary_spec_catalog(spec: Mapping[str, Any]) -> str:
    """Name the catalog selected by a stationary spec's schema_version."""
    schema = spec.get("schema_version") if isinstance(spec, Mapping) else None
    for name, expected in _CATALOG_SCHEMAS.items():
        if schema == expected:
            return name
    raise ValueError(f"Expected stationary schema_version {STATIONARY_SCHEMA} or {STATIONARY_SCHEMA_V2}")


def resolve_stationary_task(value: int | str, catalog: str = DEFAULT_CATALOG) -> Mapping[str, Any]:
    """Resolve a number or external dataset ID within one catalog, without legacy aliases."""
    entries = task_catalog(catalog)
    if isinstance(value, bool):
        raise ValueError("stationary task must be a number 1..5 or a dataset task ID")
    if isinstance(value, str) and value in ("1", "2", "3", "4", "5"):
        value = int(value)
    for entry in entries:
        if ((type(value) is int and value == entry["number"])
                or (isinstance(value, str) and value == entry["id"])):
            return entry
    raise ValueError("Unknown stationary task; use numbers 1..5 or a catalog dataset task ID")


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def default_stationary_config(task: int | str = 1, catalog: str = DEFAULT_CATALOG) -> Path:
    """Choose a profile by catalog task identity, never by the legacy task index."""
    entry = resolve_stationary_task(task, catalog)
    configs = _CATALOG_CONFIGS[catalog]
    if entry["number"] not in configs:
        raise ValueError(f"Stationary task {entry['number']} ({entry['id']}) is not configured yet")
    return configs[entry["number"]]


def validate_recording_boundary(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the opt-in prepared-scene recording boundary; absence keeps old runs valid."""
    from .recording_boundary import validate_recording_boundary as validate_boundary
    if settings is None:
        raise ValueError("recording_boundary must be an explicit prepared-scene configuration")
    if isinstance(settings, Mapping) and settings.get("kind") == "after_measured_grasp_hold":
        if set(settings) != {"kind", "readiness_dwell_s"}:
            raise ValueError("Held recording boundary requires exactly kind and readiness_dwell_s")
        dwell = _number(settings["readiness_dwell_s"], "recording readiness_dwell_s")
        if dwell < .5:
            raise ValueError("Held recording readiness requires at least 0.5 s")
        return {"kind": settings["kind"], "readiness_dwell_s": dwell}
    return validate_boundary(settings)


def validate_stationary_spec(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a configured independent goal; return an isolated copy of its spec.

    Legacy task configuration validation remains the responsibility of
    ``load_task_config`` / ``load_stationary_config``. A changed dataset ID cannot
    silently turn a lift-only episode into a successful JIG-placement episode.
    Tightening hold thresholds is permitted; relaxing them is rejected. The
    COM speed cap is 0.02 m/s; the machine additionally retains a 0.015 m/s
    measured-pose speed cap and a 5 mm / 0.05 rad hold-entry displacement band.
    """
    spec = config.get("stationary_task")
    boundary = (validate_recording_boundary(config["recording_boundary"])
                if "recording_boundary" in config else None)
    if not isinstance(spec, Mapping):
        raise ValueError("stationary_task must contain exactly the stationary schema fields")
    if spec.get("schema_version") == STATIONARY_SCHEMA_V2:
        return _validate_upright_pickup_spec(config, spec, boundary)
    if spec.get("schema_version") != STATIONARY_SCHEMA:
        raise ValueError(f"Expected stationary schema_version {STATIONARY_SCHEMA} or {STATIONARY_SCHEMA_V2}")
    if type(spec.get("number")) is not int:
        raise ValueError("stationary_task.number must be an integer")
    entry = resolve_stationary_task(spec["number"], "v1")
    if any(spec.get(key) != entry[key] for key in ("id", "legacy_task_id", "goal")):
        raise ValueError("stationary task number, ID, legacy ID and goal must match the catalog")
    if not entry["configured"]:
        raise ValueError(f"Stationary task {entry['number']} ({entry['id']}) is not configured yet")
    if entry["goal"] == "lift_hold" and set(spec) != _SPEC_KEYS:
        raise ValueError("stationary_task must contain exactly the stationary schema fields")
    if spec.get("require_hugwbc") is not True or spec.get("independent_scene") is not True:
        raise ValueError("Stationary collection requires live HugWBC and an independent scene")
    if config.get("attachment_mode") != "none":
        raise ValueError("Stationary collection requires physical grasps without attachments")
    if config.get("allowed_collection_scopes") != [STATIONARY_SCOPE]:
        raise ValueError("Stationary configurations must be restricted to stationary_task scope")
    if config.get("allowed_task_ids") != [entry["legacy_task_id"]]:
        raise ValueError("Stationary configurations must restrict the legacy task ID")
    if "third_person_orientation_wxyz" in config:
        from .camera_capture import default_camera_specs, with_third_person_orientation
        with_third_person_orientation(default_camera_specs(), config["third_person_orientation_wxyz"])
    task = config.get("tasks", {}).get(entry["legacy_task_id"])
    if not isinstance(task, Mapping):
        raise ValueError("Stationary configuration is missing its legacy source task")
    if task.get("object") != "crs" or task.get("mechanism") == "revolute_hinge":
        raise ValueError("Independent CRS goals require a free CRS object")
    if "stage_sequence" in task:
        raise ValueError("Catalog v1 stationary goals cannot use the upright pickup stage sequence")
    if entry["goal"] == "load_jig":
        from .stationary_place import validate_place_settings
        from .stationary_initialization import validate_prepared_grasp_initialization
        result = validate_place_settings(spec, task)
        if set(spec) != set(result):
            raise ValueError("stationary_task must contain exactly the stationary placement schema fields")
        if validate_prepared_grasp_initialization(config) is None:
            raise ValueError("Stationary placement requires an explicit prepared held-object initialization")
        thresholds = config.get("thresholds", {})
        if not 0. < _number(thresholds.get("capture_position_m"), "capture_position_m") <= .025:
            raise ValueError("Stationary placement cannot relax the original 0.025 m capture bound")
        if not 0. < _number(thresholds.get("capture_rotation_rad"), "capture_rotation_rad") <= .2:
            raise ValueError("Stationary placement cannot relax the original 0.2 rad capture bound")
        if task.get("target", {}).get("frame") != "jig" or task.get("placement_support", {}).get("object") != "jig":
            raise ValueError("Stationary placement requires a measured JIG-relative target and JIG support")
        if boundary is None or boundary["kind"] != "after_measured_grasp_hold":
            raise ValueError("Stationary placement requires an explicit measured grasp-hold recording boundary")
        if boundary["readiness_dwell_s"] < result["preparation_dwell_s"]:
            raise ValueError("Held recording readiness cannot shorten measured grasp preparation")
        return deepcopy(dict(result))
    if "prepared_grasp_initialization" in config:
        raise ValueError("Stationary lift cannot start from a prepared held-object snapshot")
    if boundary is not None and boundary["kind"] != "after_measured_pregrasp":
        raise ValueError("Stationary lift requires the measured pregrasp recording boundary")
    minimum = _number(spec["minimum_lift_m"], "minimum_lift_m")
    if minimum < max(.08, _number(task.get("min_lift_m"), "task.min_lift_m")):
        raise ValueError("minimum_lift_m cannot relax the original lift threshold or 0.08 m")
    offset = task.get("lift_offset_w", ())
    if not isinstance(offset, (list, tuple)) or len(offset) != 3:
        raise ValueError("CRS lift goal requires a three-dimensional lift offset")
    if minimum > _number(offset[2], "lift_offset_w.z"):
        raise ValueError("minimum_lift_m exceeds the configured lift goal")
    if _number(spec["hold_dwell_s"], "hold_dwell_s") < 1.:
        raise ValueError("Stationary lift requires at least 1.0 s of continuous measured hold")
    for key, maximum in _MAX_SPEEDS.items():
        if not 0. < _number(spec[key], key) <= maximum:
            raise ValueError(f"{key} must be positive and no greater than {maximum}")
    thresholds = config.get("thresholds", {})
    if not 0. < _number(thresholds.get("capture_position_m"), "capture_position_m") <= .025:
        raise ValueError("Stationary lift cannot relax the original 0.025 m capture bound")
    if not 0. < _number(thresholds.get("capture_rotation_rad"), "capture_rotation_rad") <= .2:
        raise ValueError("Stationary lift cannot relax the original 0.2 rad capture bound")
    return deepcopy(dict(spec))


def _validate_upright_pickup_spec(config: Mapping[str, Any], spec: Mapping[str, Any],
                                  boundary: Mapping[str, Any] | None) -> dict[str, Any]:
    """Catalog-v2 task 1: recorded pickup from an upright start, then released JIG loading.

    Placement acceptance reuses the validated task-2 bounds and may only tighten
    them. The pickup adds a lift that clears the cart POM slots and upright
    readiness tolerances; a held or pre-posed initialization is never accepted.
    """
    from .stationary_place import PLACEMENT_BOUNDS
    if set(spec) != _SPEC_KEYS_V2:
        raise ValueError("stationary_task must contain exactly the stationary schema fields")
    if type(spec.get("number")) is not int:
        raise ValueError("stationary_task.number must be an integer")
    entry = resolve_stationary_task(spec["number"], "v2")
    if any(spec.get(key) != entry[key] for key in ("id", "legacy_task_id", "goal")):
        raise ValueError("stationary task number, ID, legacy ID and goal must match the catalog")
    if not entry["configured"]:
        raise ValueError(f"Stationary task {entry['number']} ({entry['id']}) is not configured yet")
    if spec.get("require_hugwbc") is not True or spec.get("independent_scene") is not True:
        raise ValueError("Stationary collection requires live HugWBC and an independent scene")
    if config.get("attachment_mode") != "none":
        raise ValueError("Stationary collection requires physical grasps without attachments")
    if config.get("allowed_collection_scopes") != [STATIONARY_SCOPE]:
        raise ValueError("Stationary configurations must be restricted to stationary_task scope")
    if config.get("allowed_task_ids") != [entry["legacy_task_id"]]:
        raise ValueError("Stationary configurations must restrict the legacy task ID")
    if "third_person_orientation_wxyz" in config:
        from .camera_capture import default_camera_specs, with_third_person_orientation
        with_third_person_orientation(default_camera_specs(), config["third_person_orientation_wxyz"])
    task = config.get("tasks", {}).get(entry["legacy_task_id"])
    if not isinstance(task, Mapping):
        raise ValueError("Stationary configuration is missing its legacy source task")
    if task.get("object") != "crs" or task.get("mechanism") == "revolute_hinge":
        raise ValueError("Independent CRS goals require a free CRS object")
    result = deepcopy(dict(spec))
    for key in sorted(_SPEC_KEYS_V2 - _IDENTITY_KEYS):
        value = _number(spec[key], f"stationary_task.{key}")
        if value <= 0.:
            raise ValueError(f"stationary_task.{key} must be positive and finite")
        bound = PLACEMENT_BOUNDS.get(key)
        if bound is not None and (key.startswith(("max_", "maximum_", "slot_")) and value > bound
                                  or key.startswith("minimum_") and value < bound):
            raise ValueError(f"stationary_task.{key} cannot relax the validated placement acceptance bound")
        result[key] = value
    if result["preparation_dwell_s"] < .5 or result["settle_dwell_s"] < 1.:
        raise ValueError("Upright pickup/load requires 0.5 s measured preparation and 1 s released settling")
    if not .005 <= result["release_separation_m"] <= .03:
        raise ValueError("Release separation must be a bounded 5-30 mm movement")
    if result["upright_joint_tolerance_rad"] > _MAX_UPRIGHT_JOINT_TOLERANCE_RAD:
        raise ValueError("upright_joint_tolerance_rad must be at most 0.2 rad")
    if result["maximum_pelvis_tilt_rad"] > _MAX_PELVIS_TILT_RAD:
        raise ValueError("maximum_pelvis_tilt_rad must be at most 0.2 rad")
    minimum = result["minimum_lift_m"]
    if minimum < max(_MINIMUM_UPRIGHT_LIFT_M, _number(task.get("min_lift_m"), "task.min_lift_m")):
        raise ValueError("minimum_lift_m must clear the cart POM slots (0.17 m) and the original lift threshold")
    offset = task.get("lift_offset_w", ())
    if not isinstance(offset, (list, tuple)) or len(offset) != 3:
        raise ValueError("CRS lift goal requires a three-dimensional lift offset")
    if minimum > _number(offset[2], "lift_offset_w.z"):
        raise ValueError("minimum_lift_m exceeds the configured lift goal")
    retreat = task.get("retreat_offset_w", ())
    if (not isinstance(retreat, (list, tuple)) or len(retreat) != 3
            or math.sqrt(sum(_number(value, "retreat_offset_w") ** 2 for value in retreat)) < result["minimum_retreat_m"]):
        raise ValueError("Retreat target must clear the released CRS by the required distance")
    if task.get("target", {}).get("frame") != "jig" or task.get("placement_support", {}).get("object") != "jig":
        raise ValueError("Upright pickup/load requires the named JIG for both target and measured support")
    if task.get("requires_completed") or task.get("requires_events"):
        raise ValueError("Independent pickup/load cannot require preceding task execution")
    if task.get("stage_sequence") != UPRIGHT_PICKUP_SEQUENCE or not isinstance(task.get("upright_start"), Mapping):
        raise ValueError("Upright pickup/load requires the upright pickup stage sequence and an explicit upright start")
    if "prepared_grasp_initialization" in config:
        raise ValueError("Upright pickup/load cannot start from a prepared held-object snapshot")
    thresholds = config.get("thresholds", {})
    if not 0. < _number(thresholds.get("capture_position_m"), "capture_position_m") <= .025:
        raise ValueError("Upright pickup/load cannot relax the original 0.025 m capture bound")
    if not 0. < _number(thresholds.get("capture_rotation_rad"), "capture_rotation_rad") <= .2:
        raise ValueError("Upright pickup/load cannot relax the original 0.2 rad capture bound")
    if boundary is None or boundary["kind"] != "after_measured_upright_stance":
        raise ValueError("Upright pickup/load requires an explicit measured upright-stance recording boundary")
    if boundary["readiness_dwell_s"] < result["preparation_dwell_s"]:
        raise ValueError("Upright recording readiness cannot shorten measured upright preparation")
    return result


def stationary_max_task_seconds(config: Mapping[str, Any]) -> float | None:
    """Per-attempt ``--max_task_seconds`` of a v2 profile: its timeout_s plus a margin.

    None for catalog-v1 goals, which keep the runner and collector caps their
    admitted recordings were made with.
    """
    spec = config.get("stationary_task")
    if not isinstance(spec, Mapping) or spec.get("goal") != "pickup_load_jig":
        return None
    return float(config["tasks"][spec["legacy_task_id"]]["timeout_s"]) + _TASK_WALL_MARGIN_S


def recorded_inter_station_transport(config: Mapping[str, Any]) -> bool:
    """Whether the recorded demonstration walks between two distinct declared stations.

    Only the v2 pickup/load records a walk: its side-step (navigate_target) from
    the source to the target station, with the grasped CRS. A v2 target station
    that repeats the source (planar position and yaw) declares no transport;
    admission then requires an exactly zero base command in every recorded stage.
    Catalog-v1 goals never record an inter-station walk.
    """
    spec = config.get("stationary_task")
    if not isinstance(spec, Mapping) or spec.get("goal") != "pickup_load_jig":
        return False
    stations = config["tasks"][spec["legacy_task_id"]]["base_stations"]
    source, target = (pose_to_matrix(stations[name]) for name in ("source", "target"))
    yaw = math.atan2(source[1, 0], source[0, 0]) - math.atan2(target[1, 0], target[0, 0])
    return bool(math.hypot(*(source[:2, 3] - target[:2, 3])) > 1e-6
                or abs(math.atan2(math.sin(yaw), math.cos(yaw))) > 1e-6)


def config_digest(config: Mapping[str, Any]) -> str:
    """Canonical full-configuration hash, including the independent goal contract."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_stationary_config(path: str | Path = DEFAULT_CONFIG, *,
                           expected_sha256: str | None = None) -> dict[str, Any]:
    """Load a legacy-validated stationary profile and optionally pin its identity.

    ``expected_sha256`` refers to the normalized full configuration returned by
    this function, not raw file bytes. Formatting changes do not change this
    identity; physics, controller, goal and metadata changes do.
    """
    config = load_task_config(path)
    validate_stationary_spec(config)
    if expected_sha256 is not None:
        if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
                or any(char not in "0123456789abcdef" for char in expected_sha256)):
            raise ValueError("expected_sha256 must be a lowercase SHA-256 hex digest")
        if config_digest(config) != expected_sha256:
            raise ValueError("Stationary configuration hash mismatch")
    return config
