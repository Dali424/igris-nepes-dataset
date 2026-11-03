"""Isaac-free copy of the existing HugWBC command-file contract.

Keep these names and upper-body ranges aligned with
``robros_lab/tasks/hug_wbc/controllers/npz_command_source.py``.  Importing that
package here would also import the simulator, so contract parity is tested.
"""

HUGWBC_COMMAND_SCHEMA = "igris_c_hugwbc_command_v1"
COMMAND_FPS = 50
TELEOPERATION_JOINT_NAMES = (
    "Joint_Waist_Yaw",
    "Joint_Waist_Roll",
    "Joint_Waist_Pitch",
    "Joint_Shoulder_Pitch_Left",
    "Joint_Shoulder_Roll_Left",
    "Joint_Shoulder_Yaw_Left",
    "Joint_Elbow_Pitch_Left",
    "Joint_Shoulder_Pitch_Right",
    "Joint_Shoulder_Roll_Right",
    "Joint_Shoulder_Yaw_Right",
    "Joint_Elbow_Pitch_Right",
)
TELEOPERATION_RANGES = (
    (-1.5, 1.5),
    (-0.2, 0.2),
    (-1.0, 0.28),
    (-3.0, 1.0),
    (-0.15, 3.0),
    (-1.5, 1.5),
    (-2.1, 0.0),
    (-3.0, 1.0),
    (-3.0, 0.15),
    (-1.5, 1.5),
    (-2.1, 0.0),
)
WRIST_HAND_JOINT_NAMES = (
    "Joint_Wrist_Yaw_Left",
    "Joint_Wrist_Roll_Left",
    "Joint_Wrist_Pitch_Left",
    "Joint_Wrist_Yaw_Right",
    "Joint_Wrist_Roll_Right",
    "Joint_Wrist_Pitch_Right",
    "l_0_joint_thumb_proximal",
    "l_1_joint_thumb_middle",
    "l_3_joint_index_middle",
    "l_5_joint_middle_middle",
    "l_7_joint_ring_middle",
    "l_9_joint_little_middle",
    "r_0_joint_thumb_proximal",
    "r_1_joint_thumb_middle",
    "r_3_joint_index_middle",
    "r_5_joint_middle_middle",
    "r_7_joint_ring_middle",
    "r_9_joint_little_middle",
)
PLANNABLE_JOINT_NAMES = (
    TELEOPERATION_JOINT_NAMES[0],
    *TELEOPERATION_JOINT_NAMES[3:],
    *WRIST_HAND_JOINT_NAMES[:6],
)
COMMAND_JOINT_NAMES = TELEOPERATION_JOINT_NAMES + WRIST_HAND_JOINT_NAMES
