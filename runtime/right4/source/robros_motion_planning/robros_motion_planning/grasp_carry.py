"""A separately labelled physical grasp/transport primitive, not a full task.

Uses all parent contact, payload, observation and fault checks. No object state
is written. Completion requires measured object transport followed by a quiet
hold with the gripper still closed; insertion/placement is not claimed.
"""
from copy import deepcopy
import math
import numpy as np

from .nepes_tasks import NepesTaskMachine, MANIPULATION_STAGES
from .snapshot import matrix_to_pose, pose_to_matrix


class GraspCarryMachine(NepesTaskMachine):
    def __init__(self, config, task_id, *, offset_w=(.25, 0.), seed=0,
                 completed_tasks=(), min_transport_m=.15, hold_seconds=.6):
        if task_id == "open_cleaner_cover":
            raise ValueError("Hinged doors are not free transport payloads")
        offset = np.asarray(offset_w, dtype=float)
        if offset.shape != (2,) or not np.isfinite(offset).all():
            raise ValueError("carry offset requires two finite world XY values")
        if not math.isfinite(min_transport_m) or min_transport_m <= 0:
            raise ValueError("minimum transport must be positive")
        if np.linalg.norm(offset) < min_transport_m + .08:
            raise ValueError("carry offset needs minimum transport plus arrival clearance")
        if not math.isfinite(hold_seconds) or hold_seconds < .6:
            raise ValueError("carry hold requires at least .6 seconds")
        cfg = deepcopy(config)
        station = cfg["tasks"][task_id]["base_stations"]["source"]
        if station is None:
            raise ValueError("carry primitive needs a measured base station")
        cfg["tasks"][task_id]["base_stations"]["target"] = list(station)
        super().__init__(cfg, task_id, seed=seed, completed_tasks=completed_tasks)
        self.stages = MANIPULATION_STAGES[:7]
        for relation in self.task.get("required_object_relations", []):
            if relation.get("check_until_stage", "settle") not in self.stages:
                relation["check_until_stage"] = "navigate_target"
        self.offset_w, self.min_transport_m = offset, min_transport_m
        self.hold_seconds = hold_seconds
        self.min_lift_m = max(.18, self.task["min_lift_m"])
        if self.task["lift_offset_w"][2] < self.min_lift_m:
            raise ValueError("configured lift is too short for the carry primitive")
        self._carry_start = self._hold_tcp = None
        self._last_base = None
        self._base_speed = self._base_yaw_speed = None
        self.transport_m = self.lift_m = 0.

    def graph(self):
        result = super().graph()
        result.update(episode_scope="grasp_carry", full_task_completed=False,
                      carry_offset_w=self.offset_w.tolist(), minimum_transport_m=self.min_transport_m,
                      minimum_lift_m=self.min_lift_m, hold_seconds=self.hold_seconds)
        result["stages"][-1]["success_condition"] = (
            "measured payload translation and lift, retained pinch and payload relation, "
            "followed by continuous low-speed object/base hold; no insertion or release")
        return result

    def step(self, observation):
        if self.status == "running" and "base_pose_w" in observation:
            base = pose_to_matrix(observation["base_pose_w"])
            now = float(observation["time_s"])
            if self._last_base is not None and now > self._last_base[0]:
                previous, dt = self._last_base[1], now - self._last_base[0]
                self._base_speed = float(np.linalg.norm(base[:3, 3] - previous[:3, 3]) / dt)
                yaw = math.atan2(base[1, 0], base[0, 0]) - math.atan2(previous[1, 0], previous[0, 0])
                self._base_yaw_speed = abs(math.atan2(math.sin(yaw), math.cos(yaw))) / dt
            self._last_base = (now, base)
        return super().step(observation)

    def _enter(self, index, observed):
        super()._enter(index, observed)
        if self.stage_id == "navigate_target":
            self._carry_start = observed["objects"][self.object_path].copy()
            base = observed["base"].copy()
            base[:2, 3] += self.offset_w
            self.task["base_stations"]["target"] = matrix_to_pose(base)

    def _base_near(self, observed, station):
        if station != "target":
            return super()._base_near(observed, station)
        obj = observed["objects"][self.object_path]
        self.transport_m = float(np.dot(obj[:2, 3] - self._carry_start[:2, 3],
                                       self.offset_w / np.linalg.norm(self.offset_w)))
        self.lift_m = float(obj[2, 3] - self._source_object[2, 3])
        geometry_ok = self.transport_m >= self.min_transport_m and self.lift_m >= self.min_lift_m
        if (self._hold_tcp is None and geometry_ok and self._grasp_valid(observed)
                and super()._base_near(observed, station)):
            self._hold_tcp = observed["tcp"].copy()
        return bool(self._hold_tcp is not None and geometry_ok and self._grasp_valid(observed)
                    and self._linear_speed is not None and self._linear_speed <= .015
                    and self._angular_speed is not None and self._angular_speed <= .12
                    and self._base_speed is not None and self._base_speed <= .03
                    and self._base_yaw_speed is not None and self._base_yaw_speed <= .12)

    def _dwell(self, condition, now, duration):
        return super()._dwell(condition, now,
                              self.hold_seconds if self.stage_id == "navigate_target" else duration)

    def _command(self):
        result = super()._command()
        carry_goal = None
        if self._carry_start is not None:
            carry_goal = self._carry_start.copy()
            carry_goal[:2, 3] += self.offset_w
            carry_goal = matrix_to_pose(carry_goal)
        result.update(episode_scope="grasp_carry", full_task_completed=False,
                      target_object_pose_w=carry_goal,
                      measured_carry_projection_m=self.transport_m,
                      measured_lift_m=self.lift_m,
                      primitive_phase="carry_hold" if self._hold_tcp is not None else self.stage_id)
        if self._hold_tcp is not None and self.status == "running":
            result.update(motion_mode="contact_servo", base_goal_pose_w=None,
                          tcp_target_pose_w=matrix_to_pose(self._hold_tcp), hand_command="close")
        return result
