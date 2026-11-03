#!/usr/bin/env python3
"""Pink IK worker: private JSON-lines protocol on stdin/stdout.

Started only by runtime/<profile>/.../upper_body_ik_backend.py with the isolated
IK Python (requirements/ik_pink.txt): one persistent worker per Isaac process,
no ports, temporary names or shared files. It serves one runtime profile and
executes that runtime's unchanged native solver source with Pinocchio
kinematics and pink.solve_ik/quadprog (servo_factory.py).

Failure policy: any error is answered once and the worker exits with code 2.
The caller then fails the attempt; nothing ever falls back to another solver.
EOF on stdin (the owner closed it or exited) ends the worker with code 0, and
a watchdog exits if the owner disappears. SIGINT is ignored so that only the
owner, EOF or SIGTERM end the worker.

``--check`` (used by setup.sh and check_release.py) verifies this interpreter
against requirements/ik_pink.txt and solves a small Pink QP; it prints JSON.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
RELEASE_ROOT = HERE.parents[1]
REQUIREMENTS = RELEASE_ROOT / "requirements" / "ik_pink.txt"
SOURCES = ("worker.py", "servo_factory.py", "pin_backend.py", "pink_qp.py")
PROTOCOL = 1
MUTABLE_FIELDS = (
    "stage_id", "stage_command_limits_rad", "waist_speed_scope", "limits",
    "waist_command_limits", "arm_command_limits_rad", "max_joint_speed",
    "max_tracking_error", "max_waist_speed", "arm_command_margin_rad",
    "posture_saturation_aware", "waist_speed_profile", "cartesian_gain_s_inv",
)
_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)$")
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))  # -I omits the script directory


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def reject_constant(value):
    raise ValueError("Nonfinite JSON number: " + value)


def read_pins(path=REQUIREMENTS):
    pins = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        match = _PIN.match(text)
        if match is None:
            raise ValueError(f"{path}:{number}: only exact name==version pins are allowed")
        pins[match.group(1)] = match.group(2)
    if not pins:
        raise ValueError(f"{path}: no pinned packages")
    return pins


def environment_report(names):
    """Interpreter identity, installed versions of ``names`` and worker source hashes."""
    from importlib.metadata import PackageNotFoundError, version
    import numpy  # noqa: F401  Import the numerical stack before reporting it.
    import pinocchio  # noqa: F401
    import pink  # noqa: F401
    import quadprog  # noqa: F401
    packages = {}
    for name in names:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"protocol": PROTOCOL, "pid": os.getpid(), "python": sys.executable,
            "python_version": platform.python_version(), "implementation": platform.python_implementation(),
            "prefix": sys.prefix, "base_prefix": sys.base_prefix, "isolated_mode": bool(sys.flags.isolated),
            "packages": packages,
            "sources_sha256": {"ik/pink/" + name: sha256_file(HERE / name) for name in SOURCES}}


def smoke_test():
    """Tiny bounded least squares through pink.solve_ik/quadprog with a known optimum."""
    import numpy as np
    from pink_qp import PinkBoundedLeastSquares
    solver = PinkBoundedLeastSquares()
    solver.dt = .02
    a = np.vstack([np.eye(3), .04 * np.eye(3)])
    b = np.r_[.5, -2., .1, 0., 0., 0.]
    out = solver(a, b, bounds=(-np.ones(3), np.ones(3)), method="bvls", tol=1e-10, max_iter=100)
    expected = np.clip(b[:3] / (1 + .04 ** 2), -1., 1.)
    error = float(np.max(np.abs(out.x - expected)))
    if not error < 1e-9:
        raise RuntimeError(f"Pink QP smoke test error {error}")
    return {"pink_qp_max_error": error}


def check(requirements=REQUIREMENTS):
    """CLI self-check: exact pins, imports, worker sources and a Pink QP smoke test."""
    result = {"schema": "nepes_pink_ik_environment_check_v1", "ok": False, "requirements": str(requirements)}
    try:
        pins = read_pins(requirements)
        report = environment_report(sorted(pins))
        mismatches = {name: {"expected": wanted, "installed": report["packages"].get(name)}
                      for name, wanted in pins.items() if report["packages"].get(name) != wanted}
        result.update(report, expected=pins, mismatches=mismatches)
        if sys.version_info[:2] != (3, 11):
            result["error"] = "Python 3.11 is required"
        elif mismatches:
            result["error"] = "Installed packages differ from requirements/ik_pink.txt"
        else:
            result.update(smoke_test(), ok=True)
    except Exception as exc:  # noqa: BLE001  A self-check reports every failure.
        result["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


def watch_owner(owner):
    while True:
        time.sleep(1.)
        if os.getppid() != owner:
            os._exit(0)  # The owning Isaac process is gone; never linger.


class Session:
    """Remote servos of one runtime profile."""

    def __init__(self):
        self.servos, self.runtime_root = {}, None

    def create(self, request):
        from servo_factory import make_servo, prepare_imports
        for path, expected in request["expected_sha256"].items():
            if sha256_file(path) != expected:
                raise ValueError("Source changed since the client hashed it: " + path)
        root = prepare_imports(request["runtime_root"])
        if self.runtime_root is None:
            self.runtime_root = root
        elif root != self.runtime_root:
            raise ValueError("One worker serves exactly one runtime profile")
        from robros_motion_planning.robot_model import load_robot_model
        model = load_robot_model(request["urdf"])
        servo = make_servo(model, request["backend"], native_module=request["native_module"],
                           runtime_root=root, **request["kwargs"])
        identifier = len(self.servos) + 1
        initial = {name: deepcopy(getattr(servo, name)) for name in MUTABLE_FIELDS if hasattr(servo, name)}
        self.servos[identifier] = (servo, initial)
        return {"servo_id": identifier, "urdf_sha256": sha256_file(request["urdf"]),
                "limits": {name: [float(low), float(high)] for name, (low, high) in servo.limits.items()},
                **servo.ik_backend_identity}

    def call(self, request):
        servo, initial = self.servos[request["servo_id"]]
        fields = request["attributes"]
        if set(fields) - set(MUTABLE_FIELDS):
            raise ValueError("Unsupported mutable servo field")
        for name in MUTABLE_FIELDS:
            if name in fields:
                setattr(servo, name, fields[name])
            elif name in initial:
                setattr(servo, name, deepcopy(initial[name]))
            elif hasattr(servo, name):
                delattr(servo, name)
        if request["method"] not in ("solve_step", "pose_and_jacobian"):
            raise ValueError("Unsupported remote method")
        return getattr(servo, request["method"])(*request["args"], **request["kwargs"])


def serve():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Only the private descriptor carries the protocol. Library output, including
    # native writes to file descriptor 1, goes to the inherited stderr log.
    protocol = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    threading.Thread(target=watch_owner, args=(os.getppid(),), name="owner-watchdog", daemon=True).start()
    session = Session()

    def send(response):
        try:
            protocol.write(json.dumps(response, default=encode, allow_nan=False, separators=(",", ":")) + "\n")
            protocol.flush()
        except BrokenPipeError:
            os._exit(0)  # The owner is gone.

    for line in sys.stdin:
        request = None
        try:
            request = json.loads(line, parse_constant=reject_constant)
            if (not isinstance(request, dict) or request.get("protocol") != PROTOCOL
                    or type(request.get("id")) is not int):
                raise ValueError("Invalid IK protocol or request ID")
            operation = request.get("operation")
            if operation == "shutdown":
                send({"id": request["id"], "ok": True, "result": None})
                return 0
            if operation == "hello":
                result = environment_report(request["packages"])
            elif operation == "create":
                result = session.create(request)
            elif operation == "call":
                result = session.call(request)
            else:
                raise ValueError("Unsupported IK operation")
            send({"id": request["id"], "ok": True, "result": result})
        except Exception as exc:  # noqa: BLE001  Report, then stop: never continue after a failure.
            identifier = request.get("id") if isinstance(request, dict) else None
            send({"id": identifier, "ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}})
            return 2
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--check"]:
        raise SystemExit(check(Path(sys.argv[2]) if len(sys.argv) > 2 else REQUIREMENTS))
    if len(sys.argv) != 1:
        raise SystemExit("usage: worker.py [--check [requirements.txt]]")
    raise SystemExit(serve())
