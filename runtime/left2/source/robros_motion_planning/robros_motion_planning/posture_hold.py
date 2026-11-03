"""Optional immutable 19-joint command hold; never a robot-state write."""
from collections.abc import Mapping
from copy import deepcopy
import math
import xml.etree.ElementTree as ET

HOLD_MODE = "hold_last_applied"
HOLD_STAGES = ("carry_posture", "place_posture", "rest_posture")
HOLD_PATH_KEYS = frozenset(("reference_mode", "joint_names", "arrival_tolerance_rad",
    "settle_velocity_rad_s", "settle_dwell_s", "max_tracking_error_rad"))
HOLD_SOURCE = "observation.commanded_upper_joint_positions_before_stage_command"


def is_applied_hold(path):
    return isinstance(path, Mapping) and path.get("reference_mode") == HOLD_MODE


def applied_hold_targets(values, names):
    if not isinstance(values, Mapping) or any(n not in values or isinstance(values[n], bool)
            or not isinstance(values[n], (int, float)) or not math.isfinite(values[n]) for n in names):
        raise ValueError("hold_last_applied needs finite previous applied targets for every owned joint")
    return {n: float(values[n]) for n in names}


def hold_anchor(values, names, time_s):
    if isinstance(time_s, bool) or not isinstance(time_s, (int, float)) or not math.isfinite(time_s):
        raise ValueError("hold_last_applied needs finite entry observation time")
    return {"reference_mode": HOLD_MODE, "source": HOLD_SOURCE, "entry_observation_time_s": time_s,
            "joint_positions": applied_hold_targets(values, names)}


class AppliedTargetHoldReference:
    """Zero new motion. The existing FSM's fresh measured dwell determines arrival."""
    def __init__(self, start):
        self._target = deepcopy(start)
        self.duration = self.elapsed = 0.
        self.complete = True
        self.segment_index = 0

    @property
    def final(self):
        return deepcopy(self._target)

    @property
    def start(self):
        return deepcopy(self._target)

    def update(self, dt):
        if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("hold_last_applied dt must be positive and finite")
        return self.final

def canonical_upper_command_limits():
    """Read canonical 19-joint bounds from the authored latest URDF and shared limits.

    Jaw variants preserve these joints. No USD import or derived asset write is
    needed. Admission never trusts command limits supplied by a trace record.
    """
    from .constants import TELEOPERATION_JOINT_NAMES, TELEOPERATION_RANGES
    from .latest_robot import COMMAND_JOINT_NAMES, GRIPPER_JOINT_NAMES, default_model_dir, _aliases
    source = default_model_dir() / "urdf/igris_c_v2_torso_1dof.urdf"
    aliases, _ = _aliases(source)
    wanted = set(COMMAND_JOINT_NAMES) - set(GRIPPER_JOINT_NAMES)
    shared = dict(zip(TELEOPERATION_JOINT_NAMES, TELEOPERATION_RANGES))
    limits = {}
    for joint in ET.parse(source).getroot().findall("joint"):
        name = aliases[joint.get("name")]
        if name not in wanted:
            continue
        item = joint.find("limit")
        lo, hi = float(item.get("lower")), float(item.get("upper"))
        if name in shared:
            lo, hi = max(lo, shared[name][0]), min(hi, shared[name][1])
        limits[name] = (lo, hi)
    if set(limits) != wanted or any(not all(map(math.isfinite, pair)) or pair[0] >= pair[1]
                                    for pair in limits.values()):
        raise ValueError("Canonical upper command limits unavailable")
    return limits
