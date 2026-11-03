"""Read-only native foot-link diagnostics, without FK or contact inference.

Inputs are one environment's native articulation arrays. Positions and velocities
must both refer to the link origin: IsaacLab's ``body_lin_vel_w`` is a COM
velocity alias, so callers must use ``body_link_lin_vel_w`` instead.
"""
from collections.abc import Sequence

import numpy as np


FOOT_BODY_NAMES = {"left": "Link_Ankle_Roll_Left", "right": "Link_Ankle_Roll_Right"}
_QUATERNION_NORM_TOLERANCE = 1e-3


def native_foot_observation(body_names, body_pos_w, body_quat_w,
                            body_link_lin_vel_w, body_link_ang_vel_w):
    """Copy measured foot poses and link-origin velocities into JSON data.

    Array shapes must be (number of bodies, 3/4/3/3), without an environment
    axis. Quaternion order is wxyz. Missing or invalid data make the affected
    diagnostic unavailable; this helper never raises a task fault, changes
    state, renormalizes measurements, or estimates contact/support.
    """
    report = {
        "schema": "igris_c_native_foot_observation_v1",
        "source": "native_articulation_body_link_state",
        "available": False,
        "frame": "world",
        "pose_reference": "body_link_origin",
        "linear_velocity_reference": "body_link_origin",
        "quaternion_order": "wxyz",
        "units": {"position": "m", "linear_velocity": "m/s", "angular_velocity": "rad/s"},
        "quaternion_norm_tolerance": _QUATERNION_NORM_TOLERANCE,
        "contact_observed": False,
        "support_verified": False,
        "feet": {side: {"body_name": name, "available": False}
                 for side, name in FOOT_BODY_NAMES.items()},
    }

    def unavailable(reason):
        for foot in report["feet"].values():
            foot["reason"] = reason
        return report

    if (not isinstance(body_names, Sequence) or isinstance(body_names, (str, bytes))
            or not all(isinstance(name, str) for name in body_names)):
        return unavailable("invalid_body_names")
    names = list(body_names)
    arrays = {}
    for name, values, width in (
        ("position", body_pos_w, 3), ("quaternion", body_quat_w, 4),
        ("linear_velocity", body_link_lin_vel_w, 3),
        ("angular_velocity", body_link_ang_vel_w, 3),
    ):
        try:
            array = np.asarray(values)
            if array.dtype.kind not in "iuf" or array.shape != (len(names), width):
                return unavailable("invalid_" + name + "_array")
            arrays[name] = array.astype(float, copy=False)
        except (TypeError, ValueError, OverflowError):
            return unavailable("invalid_" + name + "_array")

    for foot in report["feet"].values():
        count = names.count(foot["body_name"])
        if count != 1:
            foot["reason"] = "missing_body_name" if count == 0 else "ambiguous_body_name"
            continue
        index = names.index(foot["body_name"])
        foot["body_index"] = index
        rows = {key: array[index] for key, array in arrays.items()}
        invalid = next((key for key, row in rows.items() if not np.all(np.isfinite(row))), None)
        if invalid:
            foot["reason"] = "nonfinite_" + invalid
            continue
        norm = float(np.linalg.norm(rows["quaternion"]))
        if abs(norm - 1.) > _QUATERNION_NORM_TOLERANCE:
            foot["reason"] = "quaternion_not_unit"
            continue
        foot.update(
            available=True,
            pose_w=rows["position"].tolist() + rows["quaternion"].tolist(),
            linear_velocity_w_m_s=rows["linear_velocity"].tolist(),
            angular_velocity_w_rad_s=rows["angular_velocity"].tolist(),
        )
    report["available"] = all(foot["available"] for foot in report["feet"].values())
    return report
