"""Paths and loaders shared by the Pink IK suite.

This directory runs only with the isolated IK Python (requirements/ik_pink.txt):
    "$NEPES_IK_PYTHON" -B -m unittest discover -s tests/ik_pink -t tests/ik_pink
tests/test_ik_pink_bridge.py starts it from the documented package test command.
It has no __init__.py, so the dataset-environment discovery never imports Pinocchio.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

TESTS = Path(__file__).resolve().parent
RELEASE = TESTS.parents[1]
IK_DIR = RELEASE / "ik" / "pink"
FIXTURES = TESTS / "fixtures"
URDF = FIXTURES / "igris_c_260910_canonical.urdf"
PROFILES = {"left2": "left", "right4": "right"}
BACKENDS = ("baseline", "pinocchio", "pink")


def runtime_root(profile):
    return RELEASE / "runtime" / profile / "source" / "robros_motion_planning"


def native_module(profile):
    return runtime_root(profile) / "robros_motion_planning" / "upper_body_control.py"


# The worker serves one profile per process. In this suite both profiles share the
# left2 package for the dependencies that test_servo_contract proves identical.
for _path in (IK_DIR, runtime_root("left2")):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def snapshots():
    return read_json(FIXTURES / "measured_snapshots_v1.json")["samples"]


def provenance():
    return read_json(FIXTURES / "provenance.json")


_model = None


def model():
    global _model
    if _model is None:
        from robros_motion_planning.robot_model import load_robot_model
        _model = load_robot_model(URDF)
    return _model


def make(profile, backend, **kwargs):
    from servo_factory import make_servo
    return make_servo(model(), backend, native_module=native_module(profile),
                      runtime_root=runtime_root(profile), **kwargs)


def configure_stage(servo, sample):
    servo.stage_id = sample["command_limits_stage_id"]
    servo.stage_command_limits_rad = sample["stage_command_limits_rad"]
    return servo


def solve(servo, sample):
    return servo.solve_step(sample["joint_positions"], sample["pelvis_pose_w"], sample["target_pose_w"],
                            sample["arm"], sample["dt_s"], sample["previous_targets"], **sample["solve_kwargs"])
