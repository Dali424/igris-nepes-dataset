"""Measured fall guards for the torso-rooted, articulated-waist robot.

Torso pitch includes the commanded waist bend. Pelvis orientation is the
lower-body orientation relevant to tipping; neither pose proves balance.
"""
import math
import numpy as np
from .snapshot import pose_to_matrix


def measured_stability(torso_pose_w, pelvis_pose_w):
    torso, pelvis = pose_to_matrix(torso_pose_w), pose_to_matrix(pelvis_pose_w)
    torso_tilt = math.acos(float(np.clip(torso[2, 2], -1., 1.)))
    pelvis_tilt = math.acos(float(np.clip(pelvis[2, 2], -1., 1.)))
    faults = []
    if torso[2, 3] < .72:
        faults.append("low_base_height")
    if pelvis_tilt > .45:
        faults.append("unstable_pelvis_orientation")
    return {"torso_height_m": float(torso[2, 3]), "torso_tilt_rad": torso_tilt,
            "pelvis_height_m": float(pelvis[2, 3]), "pelvis_tilt_rad": pelvis_tilt,
            "minimum_torso_height_m": .72, "maximum_pelvis_tilt_rad": .45,
            "orientation_body": "Link_Waist_Pitch", "faults": faults,
            "balance_verified": False}
