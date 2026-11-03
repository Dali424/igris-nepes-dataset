#!/usr/bin/env python3
"""Materialize a robot-only XY start perturbation without changing the baseline.

This changes requested reset/configuration values; it does not execute physics,
replan posture guides, or prove grasp success. Cartesian object/EE targets remain
fixed in their original frames. Runtime upper-body IK uses the measured base.
The seed here samples an initial pose; pass the same seed to the native runner.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil

DEFAULT_BUNDLE = Path(__file__).resolve().parents[2]
PROFILES = {2: "left2", 4: "right4"}
TASK_PREFIX = ("tasks", "crs_load_jig")
POSE_PATHS = (
    TASK_PREFIX + ("upright_start", "robot_root_pose_w"),
    TASK_PREFIX + ("base_stations", "source"),
    TASK_PREFIX + ("base_stations", "target"),
    TASK_PREFIX + ("navigation_control", "no_walk", "planned_pelvis_xy_yaw"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def get(document, path):
    value = document
    for key in path:
        value = value[key]
    return value


def changed_paths(a, b, prefix=""):
    """JSON pointers to changed scalar leaves, including unexpected structure."""
    if type(a) is not type(b):
        return [prefix]
    if isinstance(a, dict):
        return ([prefix + "/" + key for key in sorted(a.keys() ^ b.keys())]
                + [path for key in a.keys() & b.keys()
                   for path in changed_paths(a[key], b[key], prefix + "/" + key)])
    if isinstance(a, list):
        if len(a) != len(b):
            return [prefix]
        return [path for index, (left, right) in enumerate(zip(a, b))
                for path in changed_paths(left, right, prefix + "/" + str(index))]
    return [] if a == b else [prefix]


def sample_offset(pair, seed, xy_range_m):
    """Stable SHA256-to-53-bit uniforms, independent of Python RNG versions."""
    require(pair in PROFILES and type(seed) is int and seed >= 0, "Use pair 2/4 and a nonnegative integer seed")
    require(math.isfinite(xy_range_m) and xy_range_m >= 0, "XY range must be finite and nonnegative")
    uniforms = []
    for axis in ("x", "y"):
        digest = hashlib.sha256(f"nepes_robot_start_xy_v1|pair={pair}|seed={seed}|axis={axis}".encode()).digest()
        uniforms.append((int.from_bytes(digest[:8], "big") >> 11) / (1 << 53))
    return [(2 * value - 1) * xy_range_m for value in uniforms], uniforms


XY_RANGES = {2: .003, 4: .005}
LEFT_SYMMETRY = {"axis_object": [0, 0, 1], "pivot_radius_m": .127}


def effective_xy_range(pair, seed, requested_range_m=None):
    require(pair in PROFILES and type(seed) is int and 0 <= seed <= 2_147_483_647,
            "Use pair 2/4 and an integer seed in [0, 2147483647]")
    expected = XY_RANGES[pair]
    require(requested_range_m is None or requested_range_m == expected,
            f"{PROFILES[pair]} uses a fixed +/-{expected} m XY start range")
    return expected


def revision_plan():
    return {"schema": "nepes_portable_profile_policy_v1", "all_seeds_use_same_profile": True,
            "left2": {"recipe_id": "left_alignment_symmetry_v1", "xy_half_range_m": .003,
                      "approach_completion_position_m": .0008, "object_symmetry": deepcopy(LEFT_SYMMETRY)},
            "right4": {"recipe_id": "right_authored_v1", "xy_half_range_m": .005},
            "sampling": "SHA256 pair/seed/axis uniform square; independent seed ranges across PCs"}


def configure_profile(original, pair):
    config = deepcopy(original)
    if pair == 2:
        task = config["tasks"]["crs_load_jig"]
        require(task["guarded_approach"]["completion_position_m"] in (.003, .0008),
                "Unexpected source LEFT alignment; review the release configuration")
        require("object_symmetry" not in task or task["object_symmetry"] == LEFT_SYMMETRY,
                "Unexpected source LEFT symmetry")
        task["guarded_approach"]["completion_position_m"] = .0008
        task["object_symmetry"] = deepcopy(LEFT_SYMMETRY)
    return config


def lineage(sample):
    require(sample.get("schema") == "nepes_seeded_robot_initial_xy_v3", "Unknown sample schema")
    profile = sample["profile"]
    recipe = revision_plan()[profile]
    return {"sample_schema": sample["schema"], "profile": profile, "recipe_id": recipe["recipe_id"],
            "recipe_sha256": sha_bytes(json_bytes(recipe)), "effective_xy_half_range_m": sample["xy_half_range_m"],
            "source_config_sha256": sample["source_config_sha256"],
            "effective_config_file_sha256": sample["config_sha256"], "sampler_sha256": sample["sampler_sha256"]}


def build_config(bundle, pair, seed, xy_range_m=None, offset_xy=None):
    bundle = Path(bundle).resolve(strict=True)
    xy_range_m = effective_xy_range(pair, seed, xy_range_m)
    sampled_offset, uniforms = sample_offset(pair, seed, xy_range_m)
    offset = sampled_offset if offset_xy is None else list(offset_xy)
    require(len(offset) == 2 and all(isinstance(value, (float, int)) and math.isfinite(value)
                                    and abs(value) <= xy_range_m for value in offset),
            "Offset needs two finite world-frame meters, each within --xy-range-m")
    offset = [float(value) for value in offset]
    source = bundle / "configs" / f"{PROFILES[pair]}.json"
    original_bytes = source.read_bytes()
    original = json.loads(original_bytes)
    require(original.get("units") == "meters" and original.get("quaternion_order") == "wxyz",
            "Unsupported baseline units/quaternion convention")
    require(original["user_slot_pair"]["number"] == pair, "Baseline slot-pair identity mismatch")
    task = original["tasks"]["crs_load_jig"]
    require(task["arm"] == ("left" if pair == 2 else "right"), "Baseline arm mismatch")
    require("initial_robot_pose_w" not in task and "prepared_grasp" not in task,
            "Unexpected alternative robot initialization override; review reset ownership first")
    require(set(task["base_stations"]) == {"source", "target"}, "Unexpected base station schema")
    require(task["base_stations"]["source"][:2] == task["base_stations"]["target"][:2],
            "This sampler requires the single stationary stance")
    scene = (bundle / original["single_arm_layout"]["scene_usd"]).resolve(strict=True)
    require(scene.is_relative_to(bundle), "Scene must stay inside the selected bundle")
    scene_digest = sha_bytes(scene.read_bytes())
    require(scene_digest == original["single_arm_layout"]["scene_sha256"], "Scene checksum mismatch")
    profiled = configure_profile(original, pair)
    config = deepcopy(profiled)
    nominal, effective = {}, {}
    for path in POSE_PATHS:
        old = get(original, path)
        expected_length = 3 if path[-1] == "planned_pelvis_xy_yaw" else 7
        require(isinstance(old, list) and len(old) == expected_length
                and all(isinstance(value, (float, int)) and math.isfinite(value) for value in old),
                f"Invalid robot pose: {'/'.join(path)}")
        updated = get(config, path)
        updated[0], updated[1] = old[0] + offset[0], old[1] + offset[1]
        nominal["/" + "/".join(path)] = deepcopy(old)
        effective["/" + "/".join(path)] = deepcopy(updated)
    config["single_arm_layout"]["scene_usd"] = str(scene)
    allowed = {"/single_arm_layout/scene_usd"} | {
        "/" + "/".join(path) + "/" + str(axis) for path in POSE_PATHS for axis in (0, 1)}
    changed = changed_paths(profiled, config)
    require(set(changed) <= allowed, f"Unexpected configuration changes: {set(changed) - allowed}")
    config_bytes = json_bytes(config)
    manifest = {
        "schema": "nepes_seeded_robot_initial_xy_v3", "kind": "robot_initial_xy_variation",
        "profile_policy": revision_plan()[PROFILES[pair]],
        "profile_policy_scope": "Applied for every seed, independently of first-seed or attempt index",
        "profile_recipe_changes_from_source": sorted(changed_paths(original, profiled)),
        "pair": pair, "profile": PROFILES[pair], "seed": seed,
        "sampling_mode": "seeded_uniform_square" if offset_xy is None else "directed_offset_override",
        "coordinate_frame": "world", "units": "meters", "xy_half_range_m": xy_range_m,
        "sampling_algorithm": "SHA256 domain-separated pair/seed/axis; first 53 bits divided by 2^53; offset=(2*u-1)*range",
        "seed_uniform_xy": uniforms, "seed_sampled_offset_xy_m": sampled_offset,
        "applied_offset_xy_m": offset,
        "directed_offset_override": offset_xy is not None,
        "nominal_robot_pose_fields": nominal, "effective_requested_robot_pose_fields": effective,
        "nominal_reset_pose_w": nominal["/" + "/".join(POSE_PATHS[0])],
        "effective_requested_reset_pose_w": effective["/" + "/".join(POSE_PATHS[0])],
        "z_and_all_orientations_preserved": True,
        "world_object_and_end_effector_targets_preserved": True,
        "gripper_controller_and_posture_guides_preserved": True,
        "posture_guides_replanned": False,
        "runtime_ik_contract": "Upper-body servo uses measured pelvis/base and unchanged Cartesian targets; posture guides remain baseline regularization",
        "planned_pelvis_note": "Translated baseline settled-pelvis reference, not a measurement or prediction verified at the new start",
        "physical_execution_performed": False, "physical_success_verified": False,
        "must_validate": "Measure actual reset/settled root and new joint/action trajectory; validate contacts, grasp, placement and success before accepting data",
        "native_runner_seed_note": "Run the native collector with this same seed; this file only materializes initial geometry",
        "source_config": str(source), "source_config_sha256": sha_bytes(original_bytes),
        "config_sha256": sha_bytes(config_bytes), "scene_usd": str(scene), "scene_sha256": scene_digest,
        "scene_hash_scope": "overlay file only, not transitive dependencies",
        "changed_json_pointers": sorted(changed), "allowed_changed_json_pointers": sorted(allowed),
        "sampler_sha256": sha_bytes(Path(__file__).read_bytes()),
    }
    # Catch accidental source writes during materialization (all operations above are read-only).
    require(source.read_bytes() == original_bytes and sha_bytes(scene.read_bytes()) == scene_digest,
            "Source config/scene changed while materializing")
    return config, manifest


def materialize(output_dir, config, manifest):
    """Exclusive new directory; never overwrite or reuse an earlier sample."""
    output_dir = Path(output_dir).absolute()
    require(not output_dir.exists(), f"Output directory already exists: {output_dir}")
    source = Path(manifest["source_config"]).resolve()
    bundle = source.parent.parent
    require(not output_dir.resolve().is_relative_to(bundle) and not bundle.is_relative_to(output_dir.resolve()),
            "Output cannot overlap the immutable bundle")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()  # Exclusive creation also protects against concurrent writers.
    try:
        for name, value in (("config.json", config), ("sample.json", manifest)):
            with (output_dir / name).open("xb") as stream:
                stream.write(json_bytes(value))
    except BaseException:
        shutil.rmtree(output_dir)
        raise
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", type=int, choices=tuple(PROFILES), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--xy-range-m", type=float, default=None,
                        help="Optional assertion of the fixed profile range: LEFT .003 / RIGHT .005 m")
    parser.add_argument("--offset-xy", type=float, nargs=2, metavar=("DX", "DY"),
                        help="Directed pilot offset instead of seed sampling; must lie within the specified bounds")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE,
                        help="Read-only baseline release directory")
    args = parser.parse_args()
    config, manifest = build_config(args.bundle, args.pair, args.seed, args.xy_range_m, args.offset_xy)
    output = materialize(args.output_dir, config, manifest)
    print(json.dumps({"config": str(output / "config.json"), "sample": str(output / "sample.json"),
                      "pair": args.pair, "seed": args.seed, "offset_xy_m": manifest["applied_offset_xy_m"],
                      "config_sha256": manifest["config_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
