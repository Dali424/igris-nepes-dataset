"""runtime_v26 C: opt-in place/release keys of the upright pickup/load task (validation only).

Every key is opt-in; without it validation, commands, evidence and contracts are those of runtime_v25.
The keys change command timing and the jaw command only; no acceptance pose, JIG drift gate, capture
reference, success gate or physics parameter changes.

* placement_support.latch_dwell_s in [0.02, 0.3] s: PlacementHold latches the TCP only after valid support has
  been confirmed continuously for this long (placement_hold.py); requires the placement hold
  (placement_alignment.support_hold_dropout_grace_s). The pending phase is bounded: 1 s after the first loaded
  sample, a raw JIG load of 1.2 m g or a 1 mm JIG drift fall back to the runtime_v25 first-sample latch.
* placement_support.filter_s in (0, 0.3] s: the place-stage support decision uses the causal filtered JIG normal
  force (placement_support_filter.py); the raw jig_support_confirmed gate stays in the trace.
* placement_alignment.allow_press_in: true lets command_height_offset_m reach -0.0005 m (placement_alignment.py).
* object_pose_control_overrides.place.{fast_linear_speed, slow_zone_height_m}: the two-speed place descent
  (crs_manipulation.CrsPoseController two_speed_descent).
* retreat_pose_control_overrides.release_separating: release 'separating' subphase pose-control settings.
* place_pinch_reduction {start_height_m, delta_rad}: the experimental pre-seat reduced pinch
  (supported_grip_relief.py); requires supported_grip_relief and zero target_position_jitter_m.
* supported_grip_relief.{jump_start_rad, relief_profile, activation_minimum_load_fraction}
  (supported_grip_relief.py).
"""
from collections.abc import Mapping
import math
from numbers import Real

import numpy as np

PLACEMENT_SUPPORT_REQUIRED = frozenset(("object", "normal_w", "minimum_load_fraction", "maximum_linear_m_s",
                                        "maximum_angular_rad_s", "dwell_s"))
PLACEMENT_SUPPORT_OPTIONAL = frozenset(("latch_dwell_s", "filter_s"))
TWO_SPEED_DESCENT_KEYS = ("fast_linear_speed", "slow_zone_height_m")
FAST_LINEAR_SPEED_MAXIMUM_M_S = .02
# runtime_v26 review fix: the slow zone must cover run_116's 3.9 mm chamfer contact plus a servo-lag margin.
SLOW_ZONE_HEIGHT_BOUNDS_M = (.005, .02)
RELEASE_SEPARATING = "release_separating"
RETREAT_OVERRIDE_STAGES = frozenset(("retreat", RELEASE_SEPARATING))
# Explicit bounds of the empty-hand separating reference (the retreat override itself keeps TcpPoseController's
# positivity checks only, as in runtime_v25).
RELEASE_SEPARATING_BOUNDS = {"linear_speed": .05, "linear_acceleration": .2, "position_lead": .01}
RELEASE_SEPARATING_KEYS = frozenset((*RELEASE_SEPARATING_BOUNDS, "reference_profile"))


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)


def validate_placement_support_options(support, *, upright_pickup, placement_alignment):
    """The optional placement_support keys; the required six keep their runtime_v25 checks in nepes_tasks."""
    declared = set(support) & PLACEMENT_SUPPORT_OPTIONAL
    if not declared:
        return
    if not upright_pickup:
        raise ValueError("placement_support." + "/".join(sorted(declared))
                         + " requires the independent upright pickup/load task1")
    if "latch_dwell_s" in support:
        from .placement_hold import validate_latch_dwell_s
        validate_latch_dwell_s(support["latch_dwell_s"])
        if not isinstance(placement_alignment, Mapping) or "support_hold_dropout_grace_s" not in placement_alignment:
            raise ValueError("placement_support.latch_dwell_s requires the placement hold "
                             "(placement_alignment.support_hold_dropout_grace_s)")
    if "filter_s" in support:
        from .placement_support_filter import validate_filter_s
        validate_filter_s(support["filter_s"])


def split_two_speed_descent(item):
    """(place override without the two-speed keys, the declared two-speed keys)."""
    return ({key: value for key, value in item.items() if key not in TWO_SPEED_DESCENT_KEYS},
            {key: item[key] for key in TWO_SPEED_DESCENT_KEYS if key in item})


def validate_two_speed_descent(declared, effective, task, task_id):
    """object_pose_control_overrides.place two-speed keys against the effective place pose-control settings."""
    if not declared:
        return
    name = f"{task_id}.object_pose_control_overrides.place"
    if set(declared) != set(TWO_SPEED_DESCENT_KEYS):
        raise ValueError(f"{name} two-speed descent requires both fast_linear_speed and slow_zone_height_m")
    fast, zone = declared["fast_linear_speed"], declared["slow_zone_height_m"]
    slow = effective.get("linear_speed", .03)
    if not _finite(fast) or not 0. < fast <= FAST_LINEAR_SPEED_MAXIMUM_M_S or not _finite(slow) or fast <= slow:
        raise ValueError(f"{name}.fast_linear_speed must be finite, above the place linear_speed and at most 0.02 m/s")
    low, high = SLOW_ZONE_HEIGHT_BOUNDS_M
    if not _finite(zone) or not low <= zone <= high:
        raise ValueError(f"{name}.slow_zone_height_m must be finite in [0.005, 0.02] m")
    if effective.get("reference_profile", "quintic") != "online":
        raise ValueError(f"{name} two-speed descent requires the online reference profile")
    if "place_aligned_descent" not in task:
        raise ValueError(f"{name} two-speed descent requires place_aligned_descent (a vertical nominal descent)")
    offset = task.get("placement_alignment", {}).get("command_height_offset_m", 0.)
    if _finite(offset) and zone <= offset:
        raise ValueError(f"{name}.slow_zone_height_m must exceed placement_alignment.command_height_offset_m")


def validate_release_separating(value, task, task_id):
    """retreat_pose_control_overrides.release_separating: bounded empty-hand separation settings."""
    name = f"{task_id}.retreat_pose_control_overrides.release_separating"
    if not isinstance(value, Mapping) or not value or set(value) - RELEASE_SEPARATING_KEYS:
        unknown = sorted(map(str, set(value) - RELEASE_SEPARATING_KEYS)) if isinstance(value, Mapping) else []
        raise ValueError(f"{name} must be a nonempty mapping of " + ", ".join(sorted(RELEASE_SEPARATING_KEYS))
                         + (" (unknown: " + ", ".join(unknown) + ")" if unknown else ""))
    for key, bound in RELEASE_SEPARATING_BOUNDS.items():
        if key in value and (not _finite(value[key]) or not 0. < value[key] <= bound):
            raise ValueError(f"{name}.{key} must be finite in (0, {bound}]")
    if "reference_profile" in value and value["reference_profile"] not in ("quintic", "online"):
        raise ValueError(f"{name}.reference_profile must be quintic or online")
    if "release_clearance_offset_w" not in task:
        raise ValueError(f"{name} requires release_clearance_offset_w (separation starts only after the measured "
                         "contact-free clearing phase)")


def validate_place_release_options(task, task_id):
    """Cross-key checks of the upright pickup/load task, after every individual key was validated."""
    relief = task.get("supported_grip_relief")
    support = task.get("placement_support", {})
    if relief is not None and "activation_minimum_load_fraction" in relief:
        from .supported_grip_relief import validate_supported_grip_relief
        validate_supported_grip_relief(relief, minimum_load_fraction=support.get("minimum_load_fraction"))
    if "place_pinch_reduction" in task:
        from .supported_grip_relief import validate_place_pinch_reduction
        if relief is None:
            raise ValueError(f"{task_id}.place_pinch_reduction requires supported_grip_relief (the relief takes over "
                             "the reduced pinch and the admission replays both)")
        task["place_pinch_reduction"] = validate_place_pinch_reduction(task["place_pinch_reduction"], relief)
        if np.any(np.asarray(task.get("target_position_jitter_m", [0., 0., 0.]), float) != 0.):
            # The collection admission replays the reduction height against the configured (unjittered) slot.
            raise ValueError(f"{task_id}.place_pinch_reduction requires zero target_position_jitter_m (the admission "
                             "replays the reduction against the configured slot pose)")
