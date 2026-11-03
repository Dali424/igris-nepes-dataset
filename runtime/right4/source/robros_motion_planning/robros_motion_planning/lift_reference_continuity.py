"""Eligibility checks for preserving an online reference across straight lift segments.

All acquisition/phase transforms and settings must match. Rejected boundaries
explicitly use the existing new-controller behavior; this is never a recapture.
"""
from copy import deepcopy

import numpy as np


def segment_context(stage, waypoint_index, arm, object_path, grasp, control_grasp, settings):
    return {"stage": stage, "waypoint_index": waypoint_index, "arm": arm,
            "object_path": object_path, "grasp": deepcopy(grasp),
            "control_grasp": deepcopy(control_grasp), "settings": deepcopy(settings)}


def preserve_lift_reference(controller, previous, current, goal_object_pose):
    def rejected(reason):
        return {"accepted": False, "reason": reason}

    if controller is None or previous is None:
        return rejected("initial_segment")
    if previous["stage"] != "lift" or current["stage"] != "lift":
        return rejected("stage_boundary")
    if current["waypoint_index"] != previous["waypoint_index"] + 1:
        return rejected("nonadjacent_waypoint")
    for name in ("arm", "object_path", "settings"):
        if previous[name] != current[name]:
            return rejected(name + "_changed")
    for name in ("grasp", "control_grasp"):
        if not np.allclose(previous[name], current[name], rtol=0., atol=1e-12):
            return rejected(name + "_changed")
    return controller.retarget_collinear_goal(goal_object_pose)
