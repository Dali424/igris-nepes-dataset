"""Upper-body IK backend selection for the Pink release; standard library only.

The hook at the end of upper_body_control.py replaces ``UpperBodyServo`` with
the class returned by :func:`select_upper_body_servo`:

* ``pink`` (package default): construction, command limits and all other
  attributes stay native in this process; ``pose_and_jacobian`` and
  ``solve_step`` run the *same* native solver source in an isolated Python 3.11
  (``NEPES_IK_PYTHON``, requirements/ik_pink.txt) with Pinocchio kinematics and
  pink.solve_ik/quadprog (ik/pink/). One persistent worker per Isaac process
  speaks JSON lines over stdin/stdout: no ports, temporary names or shared
  files, so concurrent Isaac processes cannot interfere with each other.
* ``native``: the unchanged in-process NumPy/SciPy solver, for debugging.

``NEPES_IK_BACKEND`` selects explicitly; unset or empty means ``pink``. Any Pink
failure (missing interpreter, version or source mismatch, timeout, crash,
remote error, invalid result) raises :class:`IKBackendError`, closes the
worker and fails the attempt; there is never a fallback to another solver and
a failed worker is never restarted. Importing this module never starts a
worker and never imports Pinocchio or Pink into the Isaac process.
"""
from __future__ import annotations

import atexit
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import subprocess
import threading
import time

SCHEMA = "nepes_upper_body_ik_backend_v1"
BACKENDS = ("pink", "native")
DEFAULT_BACKEND = "pink"
BACKEND_ENV = "NEPES_IK_BACKEND"
PYTHON_ENV = "NEPES_IK_PYTHON"
HOOK_MARKER = "# --- NEPES upper-body IK backend (Pink release) ---"
HOOK_SEPARATOR = "\n\n" + HOOK_MARKER + "\n"
PROTOCOL = 1
STARTUP_TIMEOUT_S = 120.
REQUEST_TIMEOUT_S = 30.
MAX_MESSAGE_BYTES = 4 * 1024 * 1024
MUTABLE_FIELDS = (
    "stage_id", "stage_command_limits_rad", "waist_speed_scope", "limits",
    "waist_command_limits", "arm_command_limits_rad", "max_joint_speed",
    "max_tracking_error", "max_waist_speed", "arm_command_margin_rad",
    "posture_saturation_aware", "waist_speed_profile", "cartesian_gain_s_inv",
)
PACKAGE_DIR = Path(__file__).resolve().parent
# <release>/runtime/<profile>/source/robros_motion_planning/robros_motion_planning/
RELEASE_ROOT = PACKAGE_DIR.parents[4]
WORKER_DIR = RELEASE_ROOT / "ik" / "pink"
WORKER_SCRIPT = WORKER_DIR / "worker.py"
WORKER_SOURCES = ("worker.py", "servo_factory.py", "pin_backend.py", "pink_qp.py")
REQUIREMENTS = RELEASE_ROOT / "requirements" / "ik_pink.txt"
DEFAULT_PYTHON = RELEASE_ROOT / ".venv" / "ik" / "bin" / "python"
_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)$")


class IKBackendError(RuntimeError):
    """The selected upper-body IK backend failed; the attempt fails without fallback."""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def release_relative(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(RELEASE_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_backend(environ=None):
    """Return ``(backend, selected_by)``; unset or empty selects the package default."""
    raw = (os.environ if environ is None else environ).get(BACKEND_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_BACKEND, "package_default"
    if raw.strip() not in BACKENDS:
        raise IKBackendError(f"{BACKEND_ENV}={raw!r} is not supported; use one of {', '.join(BACKENDS)} "
                             f"(unset means {DEFAULT_BACKEND})")
    return raw.strip(), "environment:" + BACKEND_ENV


def resolve_worker_python(environ=None):
    """Return ``(python, selected_by)`` for the isolated IK interpreter, or raise clearly."""
    raw = (os.environ if environ is None else environ).get(PYTHON_ENV, "").strip()
    if raw:
        path, source = Path(raw).expanduser(), "environment:" + PYTHON_ENV
    elif DEFAULT_PYTHON.exists():
        path, source = DEFAULT_PYTHON, "release_default:.venv/ik/bin/python"
    else:
        raise IKBackendError(
            f"The Pink IK backend needs its isolated worker Python, but {PYTHON_ENV} is not set and "
            f"{DEFAULT_PYTHON} does not exist. Run `bash scripts/setup.sh ...` once and `source "
            f"<env-root>/env.sh` (it exports {PYTHON_ENV}=<env-root>/ik/bin/python). For single-run debugging "
            f"only, {BACKEND_ENV}=native selects the in-process solver (batches refuse it).")
    path = Path(os.path.abspath(path))  # Keep venv symlinks: resolving them would bypass the venv.
    if not path.is_file() or not os.access(path, os.X_OK):
        raise IKBackendError(f"{PYTHON_ENV} must name the executable IK environment Python (setup.sh "
                             f"creates <env-root>/ik/bin/python); not usable: {path}")
    return str(path), source


def read_pins(path=REQUIREMENTS):
    pins = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        match = _PIN.match(text)
        if match is None:
            raise IKBackendError(f"{path}:{number}: only exact name==version pins are allowed")
        pins[match.group(1)] = match.group(2)
    if not pins:
        raise IKBackendError(f"{path}: no pinned packages")
    return pins


def native_module_identity(module_file):
    """Hashes of upper_body_control.py and of its native solver part before the hook."""
    path = Path(module_file).resolve()
    data = path.read_bytes()
    text = data.decode("utf-8")
    if text.count(HOOK_MARKER) != 1 or HOOK_SEPARATOR not in text:
        raise IKBackendError(f"{path} must end with exactly one NEPES IK backend hook")
    prefix = text[:text.index(HOOK_SEPARATOR)].encode("utf-8")
    return {"module": release_relative(path), "file_sha256": sha256_bytes(data),
            "native_prefix_sha256": sha256_bytes(prefix)}


def _encode(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError("Unsupported IK request value: " + type(value).__name__)


def _reject_constant(value):
    raise ValueError("Nonfinite IK worker response: " + value)


class WorkerClient:
    """One persistent worker process. Requests are serialized; any failure closes it."""

    def __init__(self, python, script=WORKER_SCRIPT, *, pins, expected_sources=None, python_selected_by="explicit",
                 startup_timeout=STARTUP_TIMEOUT_S, request_timeout=REQUEST_TIMEOUT_S, on_failure=None):
        for value in (startup_timeout, request_timeout):
            if not 0 < value <= 600:
                raise ValueError("IK worker timeouts must be in (0, 600] seconds")
        self.python, self.script = str(python), str(Path(script).resolve())
        self.python_selected_by, self.request_timeout = python_selected_by, float(request_timeout)
        self.owner_pid, self.sequence, self.buffer = os.getpid(), 0, b""
        self.lock, self.closed, self.failure, self.returncode = threading.RLock(), False, None, None
        self.reap_error = None
        self.on_failure = on_failure
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", OPENBLAS_NUM_THREADS="1",
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        # Kit's Python and native library paths must never leak into the worker ABI.
        for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONEXECUTABLE", "LD_LIBRARY_PATH", "LD_PRELOAD"):
            environment.pop(name, None)
        try:
            # Same process group as the owner: a collector's group signal also reaches the worker.
            self.process = subprocess.Popen([self.python, "-I", "-B", "-u", self.script], stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=None, bufsize=0, env=environment,
                                            cwd=str(Path(self.script).parent), close_fds=True)
        except OSError as exc:
            raise IKBackendError(f"Cannot start the Pink IK worker with {self.python}: {exc}") from exc
        os.set_blocking(self.process.stdin.fileno(), False)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        hello = self.request("hello", timeout=startup_timeout, packages=sorted(pins))
        problems = []
        if hello.get("protocol") != PROTOCOL or hello.get("pid") != self.process.pid:
            problems.append("protocol/process identity")
        installed = hello.get("packages", {})
        problems += [f"{name}=={wanted} (installed {installed.get(name)})"
                     for name, wanted in sorted(pins.items()) if installed.get(name) != wanted]
        if expected_sources is not None and hello.get("sources_sha256") != expected_sources:
            problems.append("worker source hashes")
        if problems:
            self.fail("Pink IK worker environment differs from requirements/ik_pink.txt or the release: "
                      + "; ".join(problems) + f" [{self.python}]")
            raise IKBackendError(self.failure)
        self.hello = hello

    def request(self, operation, *, timeout=None, **payload):
        timeout = self.request_timeout if timeout is None else float(timeout)
        with self.lock:
            if self.closed:
                raise IKBackendError("The Pink IK worker is closed" + (f" after a failure ({self.failure})"
                                     if self.failure else "") + "; it is never restarted and there is no fallback")
            if self.owner_pid != os.getpid():
                raise IKBackendError("The Pink IK worker belongs to another process; there is no fallback")
            self.sequence += 1
            message = {"protocol": PROTOCOL, "id": self.sequence, "operation": operation, **payload}
            started = time.monotonic()
            try:
                raw = (json.dumps(message, default=_encode, allow_nan=False, separators=(",", ":")) + "\n").encode()
                if len(raw) > MAX_MESSAGE_BYTES:
                    raise IKBackendError("IK request exceeds the bounded protocol size")
                self._write(raw, started, timeout, operation)
                response = json.loads(self._read_line(started, timeout, operation), parse_constant=_reject_constant)
                if not isinstance(response, dict) or response.get("id") != self.sequence:
                    raise IKBackendError("Pink IK worker response ID mismatch")
                if response.get("ok") is not True:
                    error = response.get("error") or {}
                    raise IKBackendError(f"Pink IK worker {operation} failed: "
                                         f"{error.get('type', 'error')}: {error.get('message', '')}")
                return response["result"]
            except Exception as exc:
                failure = exc if isinstance(exc, IKBackendError) else IKBackendError(
                    f"Pink IK worker communication failed during {operation}: {type(exc).__name__}: {exc}")
                self.fail(str(failure))
                if failure is exc:
                    raise
                raise failure from exc

    def _write(self, raw, started, timeout, operation):
        with selectors.DefaultSelector() as writable:
            writable.register(self.process.stdin, selectors.EVENT_WRITE)
            offset = 0
            while offset < len(raw):
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0 or not writable.select(remaining):
                    raise IKBackendError(f"Pink IK worker timed out ({timeout:g} s) receiving {operation}")
                try:
                    offset += os.write(self.process.stdin.fileno(), raw[offset:])
                except BlockingIOError:
                    continue
                except BrokenPipeError as exc:
                    raise IKBackendError(f"Pink IK worker exited before {operation} "
                                         f"(returncode {self.process.poll()})") from exc

    def _read_line(self, started, timeout, operation):
        while b"\n" not in self.buffer:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0 or not self.selector.select(remaining):
                raise IKBackendError(f"Pink IK worker timed out ({timeout:g} s) during {operation}")
            block = os.read(self.process.stdout.fileno(), 65536)
            if not block:
                try:
                    code = self.process.wait(timeout=1.)
                except subprocess.TimeoutExpired:
                    code = None
                raise IKBackendError(f"Pink IK worker exited unexpectedly during {operation} (returncode {code})")
            self.buffer += block
            if len(self.buffer) > MAX_MESSAGE_BYTES:
                raise IKBackendError("IK response exceeds the bounded protocol size")
        line, self.buffer = self.buffer.split(b"\n", 1)
        return line

    def fail(self, reason):
        with self.lock:
            if self.failure is None:
                self.failure = reason
            self.close()
        if self.on_failure is not None:
            self.on_failure(self.failure)

    def close(self):
        """EOF asks the worker to exit; escalate to SIGTERM/SIGKILL only if it lingers."""
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if self.owner_pid != os.getpid():
                return
            try:
                try:
                    self.process.stdin.close()
                except OSError:
                    pass
                try:
                    self.returncode = self.process.wait(timeout=2.)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:
                        self.returncode = self.process.wait(timeout=2.)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        try:
                            self.returncode = self.process.wait(timeout=5.)
                        except subprocess.TimeoutExpired:
                            # Recorded, never raised: closing must not mask the failure being reported.
                            self.reap_error = f"Pink IK worker pid {self.process.pid} did not exit after SIGKILL"
            finally:
                self.selector.close()
                self.process.stdout.close()


_client = None
_client_lock = threading.RLock()
_STATE = {"schema": SCHEMA, "backend": None, "status": "not_selected"}


def _note_exit(client):
    """Worker exit code (and an unreapable worker) in the report; call with _client_lock held."""
    if client is not None and isinstance(_STATE.get("worker"), dict):
        _STATE["worker"]["returncode"] = client.returncode
        if client.reap_error:
            _STATE["worker"]["reap_error"] = client.reap_error


def _record_failure(reason):
    with _client_lock:
        _STATE.update(status="failed", error=reason)
        _note_exit(_client)


def shared_worker():
    """The single worker of this process, started on first use; never restarted after a failure."""
    global _client
    with _client_lock:
        if _client is not None:
            if _client.closed or _client.owner_pid != os.getpid():
                raise IKBackendError("The Pink IK worker of this process is closed"
                                     + (f" after a failure ({_client.failure})" if _client.failure else "")
                                     + "; it is never restarted and there is no fallback")
            return _client
        try:
            python, selected_by = resolve_worker_python()
            if not WORKER_SCRIPT.is_file():
                raise IKBackendError(f"Pink IK worker is missing from the release: {WORKER_SCRIPT}")
            pins = read_pins()
            expected = {"ik/pink/" + name: sha256_file(WORKER_DIR / name) for name in WORKER_SOURCES}
            client = WorkerClient(python, WORKER_SCRIPT, pins=pins, expected_sources=expected,
                                  python_selected_by=selected_by, on_failure=_record_failure)
        except IKBackendError as exc:
            _record_failure(str(exc))
            raise
        except (OSError, ValueError) as exc:
            _record_failure(f"{type(exc).__name__}: {exc}")
            raise IKBackendError(f"Cannot start the Pink IK worker: {type(exc).__name__}: {exc}") from exc
        _client = client
        hello = client.hello
        _STATE.update(status="ready", error=None, worker={
            "python": client.python, "python_selected_by": selected_by, "pid": client.process.pid,
            "transport": "persistent_stdio_jsonl_one_worker_per_process", "protocol": PROTOCOL,
            "python_version": hello["python_version"], "implementation": hello["implementation"],
            "prefix": hello["prefix"], "isolated_mode": hello["isolated_mode"], "packages": hello["packages"],
            "sources_sha256": hello["sources_sha256"]},
            requirements={"path": release_relative(REQUIREMENTS), "sha256": sha256_file(REQUIREMENTS)},
            remote_servos=0)
        return client


def shutdown_worker():
    """Close this process's worker (EOF; idempotent). Registered with atexit."""
    global _client
    with _client_lock:
        client, _client = _client, None
    if client is not None:
        client.close()
        with _client_lock:
            if _STATE.get("status") == "ready":
                _STATE["status"] = "closed"
            _note_exit(client)


atexit.register(shutdown_worker)


def backend_report():
    """JSON-ready snapshot: backend, selection, worker interpreter, versions and source hashes.

    task_runtime.py refreshes it before every run.json write and once more after
    shutdown_worker(), so run.json ends with ``closed`` and the worker exit code,
    or ``failed`` with the error that ended the attempt.
    """
    with _client_lock:
        return deepcopy(_STATE)


def _unavailable_class(native_class, message):
    class UnavailableUpperBodyServo(native_class):
        def __init__(self, *args, **kwargs):
            raise IKBackendError(message)

    UnavailableUpperBodyServo.__name__ = UnavailableUpperBodyServo.__qualname__ = "UnavailableUpperBodyServo"
    return UnavailableUpperBodyServo


def make_pink_servo_class(native_class, module_file, identity):
    module_path = Path(module_file).resolve()
    runtime_root = module_path.parents[1]

    class PinkUpperBodyServo(native_class):
        """Native API and limits; pose_and_jacobian/solve_step run in the Pink worker."""

        def __init__(self, model, **kwargs):
            super().__init__(model, **kwargs)
            worker = shared_worker()
            urdf = Path(model.urdf_path).resolve()
            urdf_sha256 = sha256_file(urdf)
            if native_module_identity(module_path) != identity:
                worker.fail("upper_body_control.py changed after it was imported")
                raise IKBackendError(worker.failure)
            response = worker.request(
                "create", backend="pink", urdf=str(urdf), runtime_root=str(runtime_root),
                native_module=str(module_path), kwargs=kwargs,
                expected_sha256={str(module_path): identity["file_sha256"], str(urdf): urdf_sha256})
            local = {name: [float(low), float(high)] for name, (low, high) in self.limits.items()}
            if (response.get("urdf_sha256") != urdf_sha256 or response.get("backend") != "pink"
                    or response.get("optimization") != "pink.solve_ik/quadprog"
                    or response.get("native_prefix_sha256") != identity["native_prefix_sha256"]
                    or response.get("limits") != local):
                worker.fail("Pink IK worker servo differs from the native model, solver source or command limits")
                raise IKBackendError(worker.failure)
            self._ik_worker, self._ik_remote_id = worker, response["servo_id"]
            with _client_lock:
                # The solver actually in use (one runtime profile per worker), not merely the last import.
                _STATE.update(native_solver=dict(identity), urdf_sha256=urdf_sha256,
                              remote_servos=_STATE.get("remote_servos", 0) + 1)

        def _ik_call(self, method, args, kwargs):
            attributes = {name: getattr(self, name) for name in MUTABLE_FIELDS if hasattr(self, name)}
            return self._ik_worker.request("call", servo_id=self._ik_remote_id, method=method, args=args,
                                           kwargs=kwargs, attributes=attributes)

        def pose_and_jacobian(self, joint_positions, pelvis_pose_w, arm):
            import numpy as np
            pose, jacobian = self._ik_call("pose_and_jacobian", [joint_positions, pelvis_pose_w, arm], {})
            pose, jacobian = np.asarray(pose, dtype=float), np.asarray(jacobian, dtype=float)
            if (pose.shape != (4, 4) or jacobian.shape != (6, len(self.names(arm)))
                    or not np.isfinite(pose).all() or not np.isfinite(jacobian).all()):
                self._ik_worker.fail("Pink IK worker returned invalid kinematic arrays")
                raise IKBackendError(self._ik_worker.failure)
            return pose, jacobian

        def solve_step(self, joint_positions, pelvis_pose_w, target_pose_w, arm, dt,
                       previous_targets=None, **options):
            result, diagnostics = self._ik_call(
                "solve_step", [joint_positions, pelvis_pose_w, target_pose_w, arm, dt, previous_targets], options)
            if (not isinstance(result, dict) or not isinstance(diagnostics, dict)
                    or set(result) != set(self.names(arm))
                    or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                           for value in result.values())):
                self._ik_worker.fail("Pink IK worker returned commands outside the owned joints or nonfinite values")
                raise IKBackendError(self._ik_worker.failure)
            # Per-solve evidence in the trace; run.json holds the full backend_report().
            diagnostics["ik_backend"] = {"backend": "pink", "worker_pid": self._ik_worker.process.pid,
                                         "request_id": self._ik_worker.sequence}
            return result, diagnostics

    PinkUpperBodyServo.__name__ = PinkUpperBodyServo.__qualname__ = "PinkUpperBodyServo"
    return PinkUpperBodyServo


def select_upper_body_servo(native_class, module_file):
    """Called once by the hook in upper_body_control.py at import; never starts a worker.

    An invalid selection or a malformed hook yields a class whose constructor
    raises, so tools that only import the module (verification, replay) work.
    """
    try:
        backend, selected_by = resolve_backend()
        identity = native_module_identity(module_file)
    except (IKBackendError, OSError, UnicodeDecodeError) as exc:
        message = str(exc)
        with _client_lock:
            _STATE.update(backend=None, status="invalid_configuration", error=message)
        return _unavailable_class(native_class, message)
    common = {"backend": backend, "selected_by": selected_by, "error": None,
              "fallback": "none: an IK backend failure fails the attempt",
              "native_solver": identity,
              "client": {"module": release_relative(__file__), "sha256": sha256_file(__file__)}}
    with _client_lock:
        if backend == "native":
            _STATE.update(common, status="in_process", worker=None, kinematics="native_numpy",
                          optimization="scipy.bvls")
            return native_class
        _STATE.update(common, status="not_started", worker=None, kinematics="pinocchio",
                      optimization="pink.solve_ik/quadprog", command_integration="clipped_previous_target")
    return make_pink_servo_class(native_class, module_file, identity)
