#!/usr/bin/env python3
"""Stand-in for one Isaac process (no Kit): imports a runtime and uses UpperBodyServo.

Used by test_ik_pink_bridge.py through subprocesses so that every "Isaac
process" owns exactly one runtime profile and one Pink worker, as in collection.
Prints one JSON document. Modes: select, fixtures, hold, exit-normal,
exit-exception, crash-worker, missing-python, report-normal, report-kill.
The report-* modes write run.json through task_runtime's own helpers
(_with_ik_backend at the episode end, then _finalize_task_run).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import sys

RELEASE = Path(__file__).resolve().parents[1]
FIXTURES = RELEASE / "tests" / "ik_pink" / "fixtures"


def emit(document):
    print(json.dumps(document, sort_keys=True), flush=True)


def main(profile, mode):
    sys.path.insert(0, str(RELEASE / "runtime" / profile / "source" / "robros_motion_planning"))
    import robros_motion_planning.upper_body_control as control
    from robros_motion_planning import upper_body_ik_backend as backend
    from robros_motion_planning.robot_model import load_robot_model
    if mode == "select":
        result = {"class": control.UpperBodyServo.__name__,
                  "is_native": control.UpperBodyServo is control.NativeUpperBodyServo,
                  "report": backend.backend_report(), "worker_started": backend._client is not None,
                  "pin_imported": "pinocchio" in sys.modules or "pink" in sys.modules}
        try:
            control.UpperBodyServo(load_robot_model(FIXTURES / "igris_c_260910_canonical.urdf"),
                                   waist_command_limits={"Joint_Waist_Yaw": [-1.2, 1.2], "Joint_Waist_Roll": [-.15, .15],
                                                         "Joint_Waist_Pitch": [-.4, .15]})
            result["construct"] = "ok"
        except backend.IKBackendError as exc:
            result["construct"] = "IKBackendError: " + str(exc)
        result["report_after_construct"] = backend.backend_report()
        backend.shutdown_worker()
        return emit(result)
    model = load_robot_model(FIXTURES / "igris_c_260910_canonical.urdf")
    samples = [s for s in json.loads((FIXTURES / "measured_snapshots_v1.json").read_text())["samples"]
               if s["profile"] == profile]
    if mode == "missing-python":
        backend.DEFAULT_PYTHON = Path("/nonexistent/nepes/.venv/ik/bin/python")
        try:
            control.UpperBodyServo(model, **samples[0]["constructor_kwargs"])
            return emit({"error": None})
        except backend.IKBackendError as exc:
            return emit({"error": str(exc), "report": backend.backend_report()})
    if mode in ("report-normal", "report-kill"):
        import tempfile
        from robros_motion_planning import task_runtime
        config = json.loads((RELEASE / "configs" / (profile + ".json")).read_text())["tasks"]["crs_load_jig"]
        report = {"schema": "nepes_task_run_v1", "episodes": []}
        try:
            control.build_upper_body_servos(model, config["upper_body_ik"], config["arm"])
        finally:
            report["upper_body_ik_backend"] = backend.backend_report()  # as task_runtime.py does
        provenance = dict(report)  # begin_recording(): provenance = {**report, ...}
        worker = backend.shared_worker()
        sample = samples[0]
        servo = control.UpperBodyServo(model, **sample["constructor_kwargs"])
        servo.stage_id = sample["command_limits_stage_id"]
        servo.stage_command_limits_rad = sample["stage_command_limits_rad"]
        if mode == "report-kill":
            os.kill(worker.process.pid, signal.SIGKILL)
        reason = ""
        try:
            servo.solve_step(sample["joint_positions"], sample["pelvis_pose_w"], sample["target_pose_w"],
                             sample["arm"], sample["dt_s"], sample["previous_targets"], **sample["solve_kwargs"])
        except backend.IKBackendError as exc:
            reason = "IKBackendError: " + str(exc)
        report["episodes"].append({"success": not reason, "reason": reason})
        with tempfile.TemporaryDirectory(prefix="nepes-ik-report-") as directory:
            path = Path(directory) / "run.json"
            task_runtime._json(path, task_runtime._with_ik_backend(report))  # end-of-episode write
            episode_end = json.loads(path.read_text())["upper_body_ik_backend"]
            errors = task_runtime._finalize_task_run(report=report, report_path=path)
            final = json.loads(path.read_text())["upper_body_ik_backend"]
        return emit({"reason": reason, "provenance": provenance["upper_body_ik_backend"], "episode_end": episode_end,
                     "final": final, "cleanup_errors": errors, "worker_pid": worker.process.pid})
    if mode == "fixtures":
        worst, markers = 0., set()
        for sample in samples:
            servo = control.UpperBodyServo(model, **sample["constructor_kwargs"])
            servo.stage_id = sample["command_limits_stage_id"]
            servo.stage_command_limits_rad = sample["stage_command_limits_rad"]
            command, diagnostics = servo.solve_step(
                sample["joint_positions"], sample["pelvis_pose_w"], sample["target_pose_w"], sample["arm"],
                sample["dt_s"], sample["previous_targets"], **sample["solve_kwargs"])
            markers.add(diagnostics["ik_backend"]["backend"])
            worst = max(worst, max(abs(command[name] - sample["recorded_output"][name]) for name in command))
        config = json.loads((RELEASE / "configs" / (profile + ".json")).read_text())["tasks"]["crs_load_jig"]
        base, stages = control.build_upper_body_servos(model, config["upper_body_ik"], config["arm"])
        worker = backend.shared_worker()
        worker_ppid = int(Path(f"/proc/{worker.process.pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
        report = backend.backend_report()
        backend.shutdown_worker()
        return emit({"profile": profile, "samples": len(samples), "max_command_difference_rad": worst,
                     "markers": sorted(markers), "owner_pid": os.getpid(), "worker_pid": worker.process.pid,
                     "worker_ppid": worker_ppid, "worker_returncode": worker.returncode,
                     "stage_classes": sorted({type(s).__name__ for s in (base, *stages.values())}),
                     "report": report, "finite": math.isfinite(worst),
                     "pin_imported": "pinocchio" in sys.modules or "pink" in sys.modules,
                     "kit_imported": any(name.startswith(("isaacsim", "omni.kit", "carb")) for name in sys.modules)})
    servo = control.UpperBodyServo(model, **samples[0]["constructor_kwargs"])
    worker = backend.shared_worker()
    if mode == "crash-worker":
        os.kill(worker.process.pid, signal.SIGKILL)
        sample = samples[0]
        result = {"worker_pid": worker.process.pid}
        try:
            servo.solve_step(sample["joint_positions"], sample["pelvis_pose_w"], sample["target_pose_w"],
                             sample["arm"], sample["dt_s"], sample["previous_targets"], **sample["solve_kwargs"])
            result["solve"] = "returned"  # Would be a silent fallback: the test fails on this.
        except backend.IKBackendError as exc:
            result["solve"] = "IKBackendError: " + str(exc)
        try:
            control.UpperBodyServo(model, **sample["constructor_kwargs"])
            result["new_servo"] = "constructed"
        except backend.IKBackendError as exc:
            result["new_servo"] = "IKBackendError: " + str(exc)
        result["report"] = backend.backend_report()
        return emit(result)
    emit({"owner_pid": os.getpid(), "worker_pid": worker.process.pid})
    if mode == "hold":
        sys.stdin.read()  # Until the test closes stdin or signals this process.
    elif mode == "exit-exception":
        raise RuntimeError("simulated attempt failure after the worker started")
    elif mode != "exit-normal":
        raise SystemExit("unknown mode " + mode)
    return None


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: ik_pink_owner.py PROFILE MODE")
    main(sys.argv[1], sys.argv[2])
