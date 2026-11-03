"""Opt-in world target motion for the existing bounded upper-body solver.

This derivative uses the object controller's commanded TCP reference, including
its bounded object-feedback correction. It does not differentiate the FSM's
endpoint or a moving robot frame. Pelvis compensation is a separate term.
"""
from collections.abc import Mapping

from .carry_feedforward import RelativePoseFeedforward


class UpperBodyTargetMotionFeedforward:
    """Differentiate a continuous world TCP reference, restarting at boundaries.

    A held final TCP reference produces zero twist. A nominal-path pause may
    still allow bounded object-feedback correction; that final target motion
    remains included. Missing references and placement support takeover disable
    the term and clear its history. Each stage/waypoint/arm segment starts at zero.
    This helper supplies a solver input only; joint limits and physics gates
    remain the caller's responsibility.
    """

    def __init__(self, stages):
        self.stages = frozenset(stages)
        self._derivative = RelativePoseFeedforward()
        self.reset()

    def reset(self):
        self._segment = None
        self._derivative.reset()
        self.last_result = None

    def update(self, stage_id, command, arm, target_pose_w, dt):
        reference = command.get("object_pose_reference")
        if (stage_id not in self.stages or target_pose_w is None
                or not isinstance(reference, Mapping)
                or reference.get("active", True) is False
                or reference.get("tcp_target_pose_w") is None
                or command.get("placement_support_active", False)
                or command.get("placement_support_hold_active", False)):
            self.reset()
            return None
        segment = (stage_id, command.get("object_waypoint_index", 0), arm)
        continuity = command.get("object_reference_continuity")
        if isinstance(continuity, Mapping) and continuity.get("run_id") is not None:
            # Opt-in blended pass-through (runtime_v26): one continuous reference spans its waypoints, so the
            # derivative history restarts with each new controller run, not with each waypoint.
            segment = (stage_id, ("blended_reference_run", continuity["run_id"]), arm)
        history_reset = segment != self._segment
        completed = command.get("completed_transfer_reference_handoff")
        if (history_reset and isinstance(completed, Mapping)
                and completed.get("controller_state_preserved") is True
                and completed.get("decision", {}).get("accepted") is True
                and tuple(completed.get("current_segment", ())) == (stage_id, command.get("object_waypoint_index", 0))
                and self._segment == (stage_id, command.get("object_waypoint_index", 0) - 1, arm)):
            # Same nominal endpoint and both correction-filter states were retained.
            # This is a reference derivative only; measured-arrival gates are unchanged.
            history_reset = False
        if history_reset:
            self._derivative.reset()
        # An identity base keeps both translational and angular components in
        # WORLD coordinates and includes no measured pelvis motion.
        twist = self._derivative.update([0., 0., 0., 1., 0., 0., 0.], target_pose_w, dt)
        self._segment = segment
        self.last_result = {
            "coordinate_frame": "world", "stage_id": stage_id,
            "object_waypoint_index": command.get("object_waypoint_index", 0), "arm": arm, "dt_s": float(dt),
            "history_reset": history_reset, "twist_world": twist.tolist(),
            "source": "consecutive_object_pose_tcp_references_in_world",
            "pelvis_motion_included": False, "pelvis_pose_written": False,
        }
        if segment[1] != command.get("object_waypoint_index", 0):
            self.last_result["blended_reference_run_id"] = segment[1][1]
        return twist
