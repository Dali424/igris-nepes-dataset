"""Prepare independent NEPES episode fixtures without simulating prior tasks.

This pure function only returns poses. The runtime may apply them at an episode
boundary, before recording, then must measure the settled scene again. Applying
an earlier task's placement target is initialization, never a demonstration or
evidence that the earlier manipulation or washing process succeeded.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any
import math

import numpy as np

from .nepes_tasks import TASK_IDS, validate_task_config
from .snapshot import matrix_to_pose, pose_to_matrix
from .hinged_door import measure_hinge_state


def prepare_independent_task(
    config: Mapping[str, Any], task_id: str,
    baseline_object_poses: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an isolated task's object poses, source base pose and provenance.

    Input and output object mappings use full prim paths and xyz+wxyz poses.
    Include measured static fixture poses in the baseline: authored fixture
    positions are deliberately not substituted for missing measurements. Other
    scene objects (including the unselected CRS rings) are preserved.

    ``satisfied_prerequisites`` names fixture conditions supplied by this setup;
    it is not a list of successful task results. Only task 4 initializes the
    ``wash_complete`` supervisory event, explicitly as a pre-washed start state.
    No attachments, contacts, stability or collision-free motion are inferred.
    """
    document = validate_task_config(config)
    if task_id not in TASK_IDS:
        raise ValueError(f"Unknown task {task_id!r}")
    if not isinstance(baseline_object_poses, Mapping):
        raise ValueError("baseline_object_poses must map full prim paths to poses")
    objects = document["objects"]
    if len(set(objects.values())) != len(objects):
        raise ValueError("Independent initialization needs distinct object prim paths")
    previous = {
        TASK_IDS[0]: [], TASK_IDS[1]: [],
        TASK_IDS[2]: [TASK_IDS[0], TASK_IDS[1]],
        TASK_IDS[3]: [TASK_IDS[0], TASK_IDS[1], TASK_IDS[2]],
        TASK_IDS[4]: [TASK_IDS[0], TASK_IDS[3]],
    }[task_id]
    required = set()
    for required_id in [*previous, task_id]:
        specification = document["tasks"][required_id]
        required.add(objects[specification["object"]])
        if specification["target"]["frame"] != "world":
            required.add(objects[specification["target"]["frame"]])
        required.update(specification.get("required_fixture_paths", []))
        for relation in specification.get("required_object_relations", []):
            required.update((objects[relation["reference"]], objects[relation["object"]]))
    if TASK_IDS[1] in previous or document["tasks"][task_id].get("requires_door_open"):
        required.update((objects["cleaner"], objects["cover"]))
    missing = required - set(baseline_object_poses)
    if missing:
        raise ValueError("Missing measured object/fixture poses: " + ", ".join(sorted(missing)))
    poses = {}
    for path, pose in baseline_object_poses.items():
        if not isinstance(path, str) or not path.startswith("/") or path == "/":
            raise ValueError("baseline_object_poses keys must be full prim paths")
        try:
            poses[path] = pose_to_matrix(pose)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid baseline pose for {path}: {exc}") from exc

    # Independent tasks need only their physical inputs, not every earlier
    # operation in the production sequence. Opening a lid needs no loaded CRS.
    loaded = False
    changed_paths = set()
    pose_overrides = []
    # Compose the required earlier target states, without generating trajectories.
    # A loaded CRS moves rigidly with the JIG during initialization only. Physics
    # subsequently determines whether this relative placement remains supported.
    for previous_id in previous:
        earlier = document["tasks"][previous_id]
        path = objects[earlier["object"]]
        target = earlier["target"]
        initial_slot = document.get("jig_slots", {}).get("initial_pose_jig")
        if previous_id == "crs_load_jig" and initial_slot is not None:
            target_w = poses[objects["jig"]] @ pose_to_matrix(initial_slot)
            pose_overrides.append({"object": "crs", "reference": "jig", "pose": list(initial_slot),
                                   "source": "jig_slots.initial_pose_jig",
                                   "purpose": "episode_boundary_settling_clearance_only"})
        else:
            target_w = pose_to_matrix(target["pose"])
        if not (previous_id == "crs_load_jig" and initial_slot is not None) and target["frame"] != "world":
            target_w = poses[objects[target["frame"]]] @ target_w
        if earlier["object"] == "jig" and loaded:
            poses[objects["crs"]] = target_w @ np.linalg.inv(poses[path]) @ poses[objects["crs"]]
            changed_paths.add(objects["crs"])
        poses[path] = target_w
        changed_paths.add(path)
        if previous_id == "crs_load_jig":
            loaded = True

    task = document["tasks"][task_id]
    checks = []
    for relation in task.get("required_object_relations", []):
        actual = np.linalg.inv(poses[objects[relation["reference"]]]) @ poses[objects[relation["object"]]]
        desired = pose_to_matrix(relation["pose"])
        position_error = float(np.linalg.norm(actual[:3, 3] - desired[:3, 3]))
        relative = actual[:3, :3].T @ desired[:3, :3]
        rotation_error = float(math.acos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
        if (position_error > relation["position_tolerance_m"] or
                rotation_error > relation["rotation_tolerance_rad"]):
            raise ValueError(
                f"Initialized {task_id} violates required object relation "
                f"{relation['object']}:{relation['reference']}")
        checks.append({"kind": "required_object_relation", "object": relation["object"],
                       "reference": relation["reference"], "position_error_m": position_error,
                       "rotation_error_rad": rotation_error})

    if "open_cleaner_cover" in previous or task.get("requires_door_open"):
        if document["tasks"]["open_cleaner_cover"].get("mechanism") == "revolute_hinge":
            state = measure_hinge_state(document, matrix_to_pose(poses[objects["cover"]]))
            if not state["open"]:
                raise ValueError(f"Initialized {task_id} hinge does not reach its opening angle")
            checks.append({"kind": "hinged_door_open", **state})
        else:
            relative = np.linalg.inv(poses[objects["cleaner"]]) @ poses[objects["cover"]]
            clearance = document["door_clearance"]
            extent = np.abs(relative[:3, :3]) @ np.asarray(clearance["cover_half_extents_m"])
            required = np.asarray(clearance["opening_half_extents_m"]) + extent[:2] + clearance["margin_m"]
            clearance_xy = np.abs(relative[:2, 3]) - required
            if not np.any(clearance_xy > 0):
                raise ValueError(f"Initialized {task_id} cover parking does not clear the cleaner opening")
            checks.append({"kind": "cover_clears_opening", "axis_clearance_m": clearance_xy.tolist()})

    # An initializer must not silently synthesize arbitrary external events.
    events = ["wash_complete"] if task_id == "jig_extract_cleaner" else []
    unsupported_events = set(task.get("requires_events", [])) - set(events)
    if unsupported_events:
        raise ValueError("Independent initialization cannot supply external events: " +
                         ", ".join(sorted(unsupported_events)))
    source_pose = task.get("base_stations", {}).get("source")
    return {
        "schema_version": "igris_c_nepes_initialization_v1",
        "task_id": task_id,
        "object_poses_w": {
            path: matrix_to_pose(pose) if path in changed_paths
            else np.asarray(baseline_object_poses[path], dtype=float).tolist()
            for path, pose in poses.items()
        },
        "robot_source_base_pose_w": deepcopy(task.get("initial_robot_pose_w", source_pose)),
        "robot_manipulation_station_pose_w": deepcopy(source_pose),
        "satisfied_prerequisites": list(task.get("requires_completed", [])),
        "events": events,
        "initialization_only": True,
        "generated_demonstration": False,
        "physical_success_verified": False,
        "applied_target_tasks": previous,
        "initialization_pose_overrides": pose_overrides,
        "geometry_checks": checks,
        "prerequisite_source": "episode_boundary_fixture_initialization",
        "event_source": "prewashed_initial_state" if events else None,
        "requires_measured_settling_validation": True,
        "calibration_status": document["calibration_status"],
    }
