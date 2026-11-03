"""Explicit DelayedPD command-delay experiments and read-only reset evidence.

The omitted profile preserves the asset's random delay range. Fixed profiles
preserve the authored buffer allocation and change live DelayedPD min/max just
before each normal env.reset(). The SDK selects the delay and resets buffers;
this module never writes buffers. Ticks are physics steps, not policy steps.
These are simulation conditions, not hardware calibration.
"""
from collections.abc import Mapping
from copy import deepcopy
import math

import numpy as np


SCHEMA = "igris_c_actuator_delay_v2"
_BUFFERS = {"position": "positions_delay_buffer", "velocity": "velocities_delay_buffer",
            "effort": "efforts_delay_buffer"}


def _ticks(value, name):
    if type(value) is not int or not 0 <= value <= 2:
        raise ValueError(f"{name} must be an integer physics tick count in [0, 2]")
    return value


def validate_actuator_delay_profile(profile):
    """Validate an explicitly present JSON profile without importing Isaac."""
    if not isinstance(profile, Mapping) or profile.get("mode") != "fixed":
        raise ValueError("actuator_delay_profile needs mode='fixed'")
    keys = set(profile) - {"startup_group_physics_ticks"}
    if keys not in ({"mode", "physics_ticks"}, {"mode", "group_physics_ticks"}):
        raise ValueError("actuator_delay_profile requires mode and either physics_ticks or "
                         "group_physics_ticks (mutually exclusive), optional startup_group_physics_ticks; "
                         "unknown keys are not allowed")
    result = {"mode": "fixed"}
    if "physics_ticks" in profile:
        result["physics_ticks"] = _ticks(profile["physics_ticks"], "physics_ticks")
    for key in ("group_physics_ticks", "startup_group_physics_ticks"):
        if key not in profile:
            continue
        groups = profile[key]
        if (not isinstance(groups, Mapping) or not groups
                or any(not isinstance(name, str) or not name.strip() for name in groups)):
            raise ValueError(f"actuator_delay_profile.{key} needs named actuator groups")
        result[key] = {name: _ticks(value, f"{key}.{name}") for name, value in groups.items()}
    return result


def _delayed_pd_type():
    from isaaclab.actuators import DelayedPDActuator
    return DelayedPDActuator


def configure_actuator_delay_profile(env_cfg, profile=None):
    """Validate the profile before gym.make without changing buffer allocation.

    All authored min/max bounds remain intact during construction. None denotes an
    omitted field; callers must reject explicitly configured null beforehand.
    """
    profile = None if profile is None else validate_actuator_delay_profile(profile)
    configs = getattr(getattr(getattr(env_cfg, "scene", None), "robot", None), "actuators", None)
    if not isinstance(configs, Mapping):
        if profile is not None:
            raise ValueError("Fixed actuator_delay_profile requires robot actuator configuration")
        configs = {}
    delayed_type = _delayed_pd_type() if configs else None
    selected = {name: cfg for name, cfg in configs.items()
                if isinstance(getattr(cfg, "class_type", None), type)
                and issubclass(cfg.class_type, delayed_type)}
    if profile is not None and not selected:
        raise ValueError("Fixed actuator_delay_profile found no DelayedPD actuator groups")
    if profile is not None:
        for key in ("group_physics_ticks", "startup_group_physics_ticks"):
            if key in profile and set(profile[key]) != set(selected):
                raise ValueError(f"actuator_delay_profile.{key} must exactly cover DelayedPD groups: "
                                 + ", ".join(sorted(selected)))
    groups = {}
    for name, cfg in selected.items():
        old_min = _ticks(cfg.min_delay, f"{name}.min_delay")
        old_max = _ticks(cfg.max_delay, f"{name}.max_delay")
        if old_min > old_max:
            raise ValueError(f"{name}: min_delay exceeds max_delay")
        requested = (None if profile is None else profile["physics_ticks"] if "physics_ticks" in profile
                     else profile["group_physics_ticks"][name])
        startup = (None if profile is None else profile.get("startup_group_physics_ticks", {}).get(name, requested))
        if requested is not None and max(requested, startup) > old_max:
            raise ValueError(f"{name}: requested delay exceeds authored buffer allocation")
        groups[name] = {"class_type": f"{cfg.class_type.__module__}.{cfg.class_type.__qualname__}",
                        "authored_min_delay_physics_ticks": old_min,
                        "authored_max_delay_physics_ticks": old_max,
                        "allocation_min_delay_physics_ticks": old_min,
                        "allocation_max_delay_physics_ticks": old_max,
                        "buffer_history_length_physics_ticks": old_max,
                        "startup_fixed_physics_ticks": startup,
                        "episode_fixed_physics_ticks": requested,
                        "joint_names_expr": list(cfg.joint_names_expr)}
    return {"schema": SCHEMA, "requested_profile": deepcopy(profile),
            "selection": "authored_delay_range" if profile is None else "explicit_fixed_simulation_delay",
            "groups": groups, "non_delayed_groups_unchanged": sorted(set(configs) - set(selected)),
            "configuration_patched_before_environment_creation": False,
            "buffer_allocation": "preserve_authored_max_delay",
            "reset_policy": "authored_delay_range" if profile is None else "fixed_bounds_before_each_env_reset",
            "live_buffers_written": False, "hardware_calibrated": False,
            "material_gains_limits_and_success_gates_changed": False}


def _expected_bounds(configuration, boundary):
    fixed = configuration["requested_profile"] is not None
    key = "startup_fixed_physics_ticks" if boundary == "startup_env_reset" else "episode_fixed_physics_ticks"
    return {name: ((group[key], group[key]) if fixed else
                   (group["authored_min_delay_physics_ticks"], group["authored_max_delay_physics_ticks"]))
            for name, group in configuration["groups"].items()}


def prepare_actuator_delay_reset(robot, configuration, *, boundary):
    """Select bounds immediately before SDK env.reset(), without writing buffers.

    Check every group and allocation before changing any live cfg. The SDK's
    ordinary reset still performs random sampling and buffer initialization.
    With min==max the sampled value is fixed; startup may have a separate map.
    """
    result = {"boundary": str(boundary), "applied": False, "bounds_physics_ticks": {},
              "live_buffers_written": False, "reset_invoked": False}
    if configuration["requested_profile"] is None:
        return result
    actuators = getattr(robot, "actuators", None)
    if not isinstance(actuators, Mapping):
        raise RuntimeError("Fixed actuator_delay_profile requires runtime robot.actuators before reset")
    delayed_type = _delayed_pd_type()
    selected = {name: actuator for name, actuator in actuators.items() if isinstance(actuator, delayed_type)}
    expected = _expected_bounds(configuration, boundary)
    if set(selected) != set(expected):
        raise RuntimeError("actuator_delay_profile reset groups differ from configured DelayedPD groups")
    writes = []
    for name, actuator in selected.items():
        capacity = configuration["groups"][name]["buffer_history_length_physics_ticks"]
        for attribute in _BUFFERS.values():
            buffer = getattr(actuator, attribute, None)
            if buffer is None or getattr(buffer, "history_length", None) != capacity:
                raise RuntimeError(f"{name}: authored delay buffer allocation was not preserved")
        cfg = actuator.cfg
        old = (_ticks(cfg.min_delay, f"{name}.min_delay"), _ticks(cfg.max_delay, f"{name}.max_delay"))
        if old[0] > old[1]:
            raise RuntimeError(f"{name}: invalid pre-reset delay bounds")
        writes.append((cfg, old, expected[name]))
    try:
        for cfg, old, bounds in writes:
            cfg.min_delay, cfg.max_delay = bounds
    except Exception:
        for cfg, old, bounds in writes:
            cfg.min_delay, cfg.max_delay = old
        raise
    result.update(applied=True, bounds_physics_ticks={name: list(bounds) for name, bounds in expected.items()})
    return result


def _lag_values(buffer, num_envs):
    value = buffer.time_lags
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.shape != (num_envs,) or value.dtype.kind not in "iu":
        raise ValueError("time_lags must contain one integer per environment")
    return value.astype(np.int64).tolist()


def read_actuator_delay_state(robot, configuration, *, physics_dt_s, boundary, num_envs=1):
    """Read all three SDK buffers after reset without mutating simulation state.

    Return errors as evidence so the runtime can durably report a failed fixed
    profile before require_actuator_delay_match raises. An absent profile never
    adds a new failure gate to existing runs.
    """
    dt = float(physics_dt_s)
    if not math.isfinite(dt) or dt <= 0 or type(num_envs) is not int or num_envs < 1:
        raise ValueError("Actuator delay readback needs positive physics_dt_s and num_envs")
    report = {"schema": SCHEMA, "boundary": str(boundary), "physics_dt_s": dt,
              "physics_tick_ms": dt * 1000., "num_envs": num_envs,
              "source": "isaaclab_DelayedPDActuator_three_DelayBuffer_time_lags_after_env_reset",
              "requested_profile": deepcopy(configuration["requested_profile"]),
              "groups": {}, "errors": [], "available": False,
              "fixed_profile_verified": False, "live_buffers_written": False,
              "hardware_calibrated": False}
    actuators = getattr(robot, "actuators", None)
    expected = configuration["groups"]
    expected_bounds = _expected_bounds(configuration, boundary)
    if not isinstance(actuators, Mapping):
        report["errors"].append("robot.actuators is unavailable")
        return report
    delayed_type = _delayed_pd_type() if actuators else None
    selected = {name: actuator for name, actuator in actuators.items()
                if isinstance(actuator, delayed_type)}
    if set(selected) != set(expected):
        report["errors"].append("Runtime DelayedPD groups differ from configured groups")
    if not selected:
        report["errors"].append("No runtime DelayedPD actuator groups available")
    for name, actuator in selected.items():
        item = {"available": False, "errors": []}
        report["groups"][name] = item
        try:
            cfg = actuator.cfg
            low, high = _ticks(cfg.min_delay, f"{name}.min_delay"), _ticks(cfg.max_delay, f"{name}.max_delay")
            if low > high:
                raise ValueError("min_delay exceeds max_delay")
            names = list(actuator.joint_names)
            if not names or any(not isinstance(joint, str) or not joint for joint in names):
                raise ValueError("Actual actuator joint_names are unavailable")
            item.update(joint_names=names, min_delay_physics_ticks=low, max_delay_physics_ticks=high,
                        delay_physics_ticks_by_buffer={}, delay_ms_by_buffer={}, buffer_history_length_by_buffer={})
            for kind, attribute in _BUFFERS.items():
                buffer = getattr(actuator, attribute)
                lags = _lag_values(buffer, num_envs)
                item["delay_physics_ticks_by_buffer"][kind] = lags
                item["delay_ms_by_buffer"][kind] = [lag * dt * 1000. for lag in lags]
                item["buffer_history_length_by_buffer"][kind] = _ticks(buffer.history_length, f"{name}.{kind}.history_length")
                if (name in expected and buffer.history_length != expected[name]["buffer_history_length_physics_ticks"]):
                    item["errors"].append(f"{kind} buffer allocation differs from authored max_delay")
                if any(lag < low or lag > high for lag in lags):
                    item["errors"].append(f"{kind} time_lags lie outside configured min/max")
            values = item["delay_physics_ticks_by_buffer"]
            if not values["position"] == values["velocity"] == values["effort"]:
                item["errors"].append("Position, velocity and effort delay buffers disagree")
            expected_group = expected.get(name)
            item["expected_reset_bounds_physics_ticks"] = list(expected_bounds[name]) if name in expected_bounds else None
            if expected_group is None or (low, high) != expected_bounds[name]:
                item["errors"].append("Runtime min/max differ from this reset boundary's configured profile")
            if configuration["requested_profile"] is not None and expected_group is not None:
                target = expected_bounds[name][0]
                if any(lag != target for lags in values.values() for lag in lags):
                    item["errors"].append("Actual delay buffers differ from explicitly requested fixed ticks")
            item["available"] = not item["errors"]
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            item["errors"].append(f"{type(error).__name__}: {error}")
        report["errors"].extend(f"{name}: {error}" for error in item["errors"])
    report["available"] = not report["errors"]
    report["fixed_profile_verified"] = configuration["requested_profile"] is not None and report["available"]
    return report


def require_actuator_delay_match(readback):
    """An explicitly fixed profile must be observed in all live delay buffers."""
    if readback["requested_profile"] is not None and not readback["fixed_profile_verified"]:
        raise RuntimeError("actuator_delay_profile readback mismatch: " + "; ".join(readback["errors"]))
