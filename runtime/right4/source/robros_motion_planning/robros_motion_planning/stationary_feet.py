"""Measured stationarity of supported feet, independent of pelvis rotation.

This is a sampled application contract, not a balance or friction certificate.
Native link poses transform declared sole material points; filtered normal
forces and native contact identities refer to the last physics step observed
at each control tick. No simulation state or control command is modified here.
"""
from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Real

import numpy as np

from .snapshot import pose_to_matrix


SCHEMA_VERSION = "igris_c_stationary_feet_hold_v1"
_SIDES = ("left", "right")


def _number(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(value) or (positive and value <= 0)):
        raise ValueError("invalid_" + name)
    return float(value)


def _vector(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("invalid_" + name)
    return np.asarray([_number(x, name) for x in value])


def _path(value):
    return isinstance(value, str) and value.startswith("/") and not value.endswith("/")


def validate_stationary_feet_settings(settings):
    keys = {"schema_version", "maximum_sole_xy_displacement_m", "maximum_sole_z_displacement_m",
            "minimum_ground_normal_force_n", "ground_collider_paths", "feet"}
    if not isinstance(settings, Mapping) or set(settings) != keys or settings["schema_version"] != SCHEMA_VERSION:
        raise ValueError("invalid_stationary_feet_hold_schema")
    result = deepcopy(dict(settings))
    for name in keys - {"schema_version", "ground_collider_paths", "feet"}:
        result[name] = _number(settings[name], name, positive=True)
    ground = settings["ground_collider_paths"]
    if (not isinstance(ground, list) or not ground or not all(map(_path, ground))
            or len(set(ground)) != len(ground)):
        raise ValueError("invalid_stationary_feet_ground_paths")
    feet = settings["feet"]
    if not isinstance(feet, Mapping) or set(feet) != set(_SIDES):
        raise ValueError("invalid_stationary_feet_geometry")
    for side in _SIDES:
        foot = feet[side]
        if (not isinstance(foot, Mapping) or set(foot) != {"body_name", "body_path", "sole_points_link_m"}
                or not isinstance(foot["body_name"], str) or not foot["body_name"]
                or not _path(foot["body_path"]) or foot["body_path"].rsplit("/", 1)[-1] != foot["body_name"]):
            raise ValueError("invalid_stationary_foot_identity")
        points = foot["sole_points_link_m"]
        if not isinstance(points, list) or len(points) < 3:
            raise ValueError("invalid_stationary_sole_points")
        points = np.asarray([_vector(point, "sole_point") for point in points])
        if np.linalg.matrix_rank(points - points[0]) < 2:
            raise ValueError("stationary_sole_points_must_span_a_surface")
        result["feet"][side]["sole_points_link_m"] = points.tolist()
    if feet["left"]["body_path"] == feet["right"]["body_path"]:
        raise ValueError("stationary_feet_must_be_distinct")
    return result


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError("invalid_" + name)
    return value


class StationaryFeetHold:
    """Anchor once at the upright recording boundary and inspect fresh samples."""

    def __init__(self, settings):
        self.settings = validate_stationary_feet_settings(settings)
        self._anchor = self._anchor_time = self._previous = self._last_input = self._last_result = None

    def _measure(self, raw):
        now = _number(raw.get("time_s"), "time")
        if now < 0:
            raise ValueError("invalid_time")
        poses, contacts = raw.get("foot_observation"), raw.get("foot_contact_observation")
        if (not isinstance(poses, Mapping) or poses.get("schema") != "igris_c_native_foot_observation_v1"
                or poses.get("available") is not True or poses.get("frame") != "world"
                or poses.get("pose_reference") != "body_link_origin" or poses.get("quaternion_order") != "wxyz"):
            raise ValueError("native_foot_pose_unavailable")
        units = poses.get("units")
        if not isinstance(units, Mapping) or units.get("position") != "m":
            raise ValueError("native_foot_pose_units_not_meters")
        if abs(_number(poses.get("measurement_time_s"), "foot_pose_time") - now) > 1e-8:
            raise ValueError("stale_foot_pose")
        if (not isinstance(contacts, Mapping) or contacts.get("schema") != "igris_c_foot_contact_observation_v1"
                or contacts.get("available") is not True or contacts.get("force_frame") != "world"
                or contacts.get("force_units") != "N"
                or contacts.get("force_kind") != "normal_only_excludes_tangential_friction"
                or contacts.get("sampling") != "last_completed_physics_step_not_control_interval_average"):
            raise ValueError("native_ground_force_unavailable")
        if abs(_number(contacts.get("trace_time_s"), "contact_trace_time") - now) > 1e-8:
            raise ValueError("stale_ground_force")
        native_time = _number(contacts.get("measurement_time_s"), "native_contact_time")
        dt = _number(contacts.get("physics_dt_s"), "physics_dt", positive=True)
        grounds, measured, sequences, relative_times = self.settings["ground_collider_paths"], {}, [], []
        for side in _SIDES:
            expected = self.settings["feet"][side]
            pose = poses["feet"][side]
            contact = contacts["feet"][side]
            if (pose.get("available") is not True or pose.get("body_name") != expected["body_name"]
                    or contact.get("available") is not True or contact.get("body_path") != expected["body_path"]):
                raise ValueError(side + "_foot_identity_or_availability")
            native_pose = pose.get("pose_w")
            if not isinstance(native_pose, (list, tuple)) or len(native_pose) != 7:
                raise ValueError(side + "_invalid_native_pose")
            native_pose = [_number(value, "native_pose") for value in native_pose]
            # Same explicit unit tolerance as the native pose producer. Small
            # floating-point norm error is normalized only for geometry math.
            if abs(np.linalg.norm(native_pose[3:]) - 1.) > 1e-3:
                raise ValueError(side + "_native_quaternion_not_unit")
            transform = pose_to_matrix(native_pose)
            points = np.asarray(expected["sole_points_link_m"]) @ transform[:3, :3].T + transform[:3, 3]
            pairs = contact.get("ground_pairs")
            if not isinstance(pairs, list) or len(pairs) != len(grounds):
                raise ValueError(side + "_ground_filters_missing")
            tensors = {}
            for pair in pairs:
                ground = pair.get("ground_collider_filter_path")
                if (pair.get("available") is not True or pair.get("sensor_path") != expected["body_path"]
                        or ground not in grounds or ground in tensors):
                    raise ValueError(side + "_ground_filter_identity")
                tensors[ground] = _vector(pair.get("normal_force_w_n"), "ground_normal_force")
            total = sum(tensors.values(), np.zeros(3))
            if not np.allclose(total, _vector(contact.get("summed_ground_normal_force_w_n"), "ground_force_sum"),
                               rtol=1e-7, atol=1e-5):
                raise ValueError(side + "_ground_force_sum_mismatch")
            report = contact.get("native_contact_report")
            if not isinstance(report, Mapping) or report.get("available") is not True or report.get("truncated") is not False:
                raise ValueError(side + "_native_contact_report_unavailable")
            sequence = _integer(report.get("physics_step_sequence"), "contact_step_sequence")
            if report.get("report_step_sequence") != sequence:
                raise ValueError(side + "_stale_native_contact_report")
            sequences.append(sequence)
            relative_times.append(_number(report.get("physics_time_since_subscription_s"), "contact_relative_time"))
            native_pairs = report.get("pairs")
            if not isinstance(native_pairs, list) or not native_pairs:
                raise ValueError(side + "_ground_contact_missing")
            impulses = {ground: np.zeros(3) for ground in grounds}
            point_count = 0
            for pair in native_pairs:
                # Native11 qualified this exact dynamic-foot actor0 ordering.
                # An unqualified actor1 ordering fails closed, rather than guessing a sign.
                ground = pair.get("collider1")
                if (pair.get("actor0") != expected["body_path"] or ground not in grounds
                        or pair.get("actor1") != ground
                        or not isinstance(pair.get("collider0"), str)
                        or not pair["collider0"].startswith(expected["body_path"] + "/")):
                    raise ValueError(side + "_unqualified_native_contact_identity")
                native_points = pair.get("points")
                if not isinstance(native_points, list) or not native_points or pair.get("reported_point_count") != len(native_points):
                    raise ValueError(side + "_native_contact_points_missing")
                for point in native_points:
                    _vector(point.get("position_w_m"), "native_contact_position")
                    normal = _vector(point.get("normal_w_native"), "native_contact_normal")
                    if abs(np.linalg.norm(normal) - 1.) > 1e-3:
                        raise ValueError(side + "_native_contact_normal_not_unit")
                    impulses[ground] += _vector(point.get("impulse_w_native_n_s"), "native_contact_impulse") / dt
                    point_count += 1
            maximum_residual = 0.
            for ground in grounds:
                residual = float(np.linalg.norm(impulses[ground] - tensors[ground]))
                maximum_residual = max(maximum_residual, residual)
                if residual > 1e-3 + 1e-6 * float(np.linalg.norm(tensors[ground])):
                    raise ValueError(side + "_native_impulse_tensor_mismatch")
            measured[side] = {"pose_w": list(map(float, pose["pose_w"])), "points_w": points,
                              "normal_force_w_n": total.tolist(), "contact_point_count": point_count,
                              "maximum_impulse_tensor_residual_n": maximum_residual}
        if len(set(sequences)) != 1 or max(relative_times) - min(relative_times) > 1e-8:
            raise ValueError("feet_contact_samples_unsynchronized")
        if _integer(contacts.get("physics_step_sequence"), "contact_step_sequence") != sequences[0]:
            raise ValueError("contact_summary_step_mismatch")
        clock = {"time_s": now, "native_time_s": native_time, "sequence": sequences[0],
                 "relative_time_s": relative_times[0], "physics_dt_s": dt}
        if self._previous is not None:
            previous = self._previous
            if (now <= previous["time_s"] or native_time <= previous["native_time_s"]
                    or sequences[0] <= previous["sequence"] or dt != previous["physics_dt_s"]):
                raise ValueError("stale_or_reordered_foot_sample")
            expected_delta = (sequences[0] - previous["sequence"]) * dt
            if (abs(now - previous["time_s"] - expected_delta) > 1e-5
                    or abs(native_time - previous["native_time_s"] - expected_delta) > 1e-5
                    or abs(relative_times[0] - previous["relative_time_s"] - expected_delta) > 1e-5):
                raise ValueError("native_contact_clock_discontinuity")
        return measured, clock

    def check(self, raw, *, anchor=False):
        if not isinstance(raw, Mapping):
            raw = {}
        subset = deepcopy({key: raw.get(key) for key in ("time_s", "foot_observation", "foot_contact_observation")})
        if not anchor and self._last_input is not None and subset == self._last_input:
            return deepcopy(self._last_result)
        result = {"schema_version": SCHEMA_VERSION, "valid": False, "reason": None,
                  "measurement_time_s": raw.get("time_s"), "anchor_time_s": self._anchor_time,
                  "sampling": "control_samples_of_last_completed_physics_step",
                  "continuous_support_verified": False, "balance_verified": False,
                  "total_contact_wrench_verified": False, "zero_base_command_required": True,
                  "pelvis_fixed_claimed": False, "feet": {}, "violations": [],
                  "limits": {key: self.settings[key] for key in ("maximum_sole_xy_displacement_m",
                      "maximum_sole_z_displacement_m", "minimum_ground_normal_force_n")}}
        try:
            if anchor and self._anchor is not None:
                raise ValueError("stationary_feet_reanchoring_forbidden")
            measured, clock = self._measure(raw)
            if self._anchor is None:
                if not anchor:
                    raise ValueError("stationary_feet_anchor_missing")
                self._anchor, self._anchor_time = deepcopy(measured), clock["time_s"]
            result["anchor_time_s"] = self._anchor_time
            result["native_contact_clock"] = clock
            for side in _SIDES:
                current, initial = measured[side], self._anchor[side]
                delta = current["points_w"] - initial["points_w"]
                xy, z = float(np.linalg.norm(delta[:, :2], axis=1).max()), float(abs(delta[:, 2]).max())
                fz = current["normal_force_w_n"][2]
                gates = {"sole_xy_stationary": xy <= self.settings["maximum_sole_xy_displacement_m"],
                         "sole_z_stationary": z <= self.settings["maximum_sole_z_displacement_m"],
                         "ground_normal_force": fz >= self.settings["minimum_ground_normal_force_n"]}
                result["feet"][side] = {"body_path": self.settings["feet"][side]["body_path"],
                    "anchor_pose_w": initial["pose_w"], "measured_pose_w": current["pose_w"],
                    "maximum_sole_xy_displacement_m": xy, "maximum_sole_z_displacement_m": z,
                    "normal_force_w_n": current["normal_force_w_n"], "native_contact_point_count": current["contact_point_count"],
                    "maximum_impulse_tensor_residual_n": current["maximum_impulse_tensor_residual_n"], "gates": gates}
                result["violations"].extend(side + ":" + name for name, passed in gates.items() if not passed)
            self._previous = clock
            result["valid"] = not result["violations"]
            result["reason"] = result["violations"][0] if result["violations"] else None
        except (ValueError, TypeError, KeyError, IndexError, OverflowError, AttributeError) as exc:
            result["reason"] = str(exc)
            result["violations"] = [str(exc)]
        self._last_input, self._last_result = subset, deepcopy(result)
        return result
