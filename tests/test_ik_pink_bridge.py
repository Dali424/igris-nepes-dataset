"""Pink IK backend wiring, checked from the dataset Python (CPU only; never starts Isaac).

* Hook/client integrity, backend and interpreter selection, release tooling,
  the batch preflight (before any output or Isaac start) and the worker
  protocol (with stand-in workers) run on every machine.
* With the IK environment installed (NEPES_IK_PYTHON from env.sh, or
  <release>/.venv/ik) the real worker is also driven from this interpreter
  (NumPy 1.26, like Isaac) through tests/ik_pink_owner.py, one profile per
  process, and the isolated suite tests/ik_pink runs in the IK Python.
  Without it those tests are skipped with the reason shown.
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

RELEASE = Path(__file__).resolve().parents[1]
OWNER = RELEASE / "tests" / "ik_pink_owner.py"
PROFILES = ("left2", "right4")
# Native solver of the validated production release (before the hook marker).
NATIVE_SHA256 = {"left2": "84bba28c5a58d54e9f7adcc60b589d9fbb2a49149cc6a969f5d152a14e7afeef",
                 "right4": "e4a4357e1f5a202e9b48a2bc413f5564b3d4a91ad415c27c1efbcdac20ad4b15"}
HOOK = '''

# --- NEPES upper-body IK backend (Pink release) ---
# Everything above this marker is the unchanged native solver. The Pink worker
# executes exactly that part in its isolated Python; see upper_body_ik_backend.py.
# NEPES_IK_BACKEND selects pink (package default) or native for this process.
from .upper_body_ik_backend import select_upper_body_servo as _select_upper_body_servo  # noqa: E402

NativeUpperBodyServo = UpperBodyServo
UpperBodyServo = _select_upper_body_servo(NativeUpperBodyServo, __file__)
'''
_names = itertools.count()


def package(profile):
    return RELEASE / "runtime" / profile / "source" / "robros_motion_planning" / "robros_motion_planning"


def load_script(path, label):
    spec = importlib.util.spec_from_file_location(f"_nepes_ik_test_{label}_{next(_names)}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_client(profile="left2"):
    """The stdlib-only client, loaded privately: no runtime package enters this test process."""
    return load_script(package(profile) / "upper_body_ik_backend.py", "client_" + profile)


def ik_python():
    value = os.environ.get("NEPES_IK_PYTHON", "").strip()
    path = Path(value).expanduser() if value else RELEASE / ".venv" / "ik" / "bin" / "python"
    return str(Path(os.path.abspath(path))) if path.is_file() and os.access(path, os.X_OK) else None


def child_environment(**changes):
    environment = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME")}
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for key, value in changes.items():
        if value is None:
            environment.pop(key, None)
        else:
            environment[key] = value
    return environment


def owner(profile, mode, *, timeout=300, **environment):
    result = subprocess.run([sys.executable, "-B", str(OWNER), profile, mode], capture_output=True, text=True,
                            timeout=timeout, env=child_environment(**environment), cwd=str(RELEASE))
    return result


def last_json(text):
    return json.loads(text.strip().splitlines()[-1])


def alive(pid):
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError):
        return False
    return state != "Z"


def wait_gone(pid, timeout=15.):
    deadline = time.monotonic() + timeout
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(.05)
    return not alive(pid)


class HookAndSelectionTests(unittest.TestCase):
    def test_runtime_hook_keeps_the_validated_native_solver(self):
        for profile in PROFILES:
            text = (package(profile) / "upper_body_control.py").read_text()
            marker = "# --- NEPES upper-body IK backend (Pink release) ---"
            self.assertEqual(text.count(marker), 1)
            prefix, suffix = text[:text.index("\n\n" + marker + "\n")], text[text.index("\n\n" + marker + "\n"):]
            self.assertEqual(hashlib.sha256(prefix.encode()).hexdigest(), NATIVE_SHA256[profile], profile)
            self.assertEqual(suffix, HOOK, profile)

    def test_client_is_identical_in_both_runtimes_and_stdlib_only(self):
        left, right = ((package(p) / "upper_body_ik_backend.py").read_bytes() for p in PROFILES)
        self.assertEqual(left, right)
        imports = set(re.findall(r"^(?:from|import) ([A-Za-z_.]+)", left.decode(), flags=re.M))
        self.assertLessEqual(imports, {"__future__", "atexit", "copy", "hashlib", "json", "math", "os", "pathlib",
                                       "re", "selectors", "subprocess", "threading", "time"})

    def test_backend_selection_is_explicit(self):
        client = load_client()
        self.assertEqual(client.resolve_backend({}), ("pink", "package_default"))
        self.assertEqual(client.resolve_backend({"NEPES_IK_BACKEND": " "}), ("pink", "package_default"))
        self.assertEqual(client.resolve_backend({"NEPES_IK_BACKEND": "native"}), ("native", "environment:NEPES_IK_BACKEND"))
        self.assertEqual(client.resolve_backend({"NEPES_IK_BACKEND": "pink"}), ("pink", "environment:NEPES_IK_BACKEND"))
        for value in ("Pink", "scipy", "pinocchio", "auto"):
            with self.subTest(value=value), self.assertRaisesRegex(client.IKBackendError, "pink, native"):
                client.resolve_backend({"NEPES_IK_BACKEND": value})

    def test_worker_python_resolution_and_clear_errors(self):
        client = load_client()
        client.DEFAULT_PYTHON = Path("/nonexistent/release/.venv/ik/bin/python")
        with self.assertRaises(client.IKBackendError) as raised:
            client.resolve_worker_python({})
        self.assertIn("NEPES_IK_PYTHON is not set", str(raised.exception))
        self.assertIn("setup.sh", str(raised.exception))
        self.assertIn("NEPES_IK_BACKEND=native", str(raised.exception))
        with self.assertRaisesRegex(client.IKBackendError, "not usable"):
            client.resolve_worker_python({"NEPES_IK_PYTHON": "/nonexistent/ik/bin/python"})
        self.assertEqual(client.resolve_worker_python({"NEPES_IK_PYTHON": sys.executable}),
                         (os.path.abspath(sys.executable), "environment:NEPES_IK_PYTHON"))
        with tempfile.TemporaryDirectory(prefix="nepes-ik-default-") as directory:
            default = Path(directory) / "python"
            default.symlink_to(sys.executable)
            client.DEFAULT_PYTHON = default
            # The venv symlink is kept, never resolved to the base interpreter.
            self.assertEqual(client.resolve_worker_python({}), (str(default), "release_default:.venv/ik/bin/python"))

    def test_requirements_pin_the_validated_closure_exactly(self):
        pins = load_client().read_pins()
        self.assertEqual({name: pins[name] for name in ("pin", "pin-pink", "quadprog", "qpsolvers", "numpy", "scipy",
                                                        "cmeel-urdfdom", "cmeel-tinyxml2")},
                         {"pin": "3.8.0", "pin-pink": "4.4.0", "quadprog": "0.1.14", "qpsolvers": "4.13.0",
                          "numpy": "2.3.5", "scipy": "1.17.1", "cmeel-urdfdom": "4.0.1", "cmeel-tinyxml2": "10.0.0"})
        self.assertNotIn("pytest", pins)

    def test_import_selects_backend_without_starting_a_worker(self):
        for profile in PROFILES:
            result = owner(profile, "select", NEPES_IK_BACKEND=None, NEPES_IK_PYTHON="/nonexistent/ik/bin/python")
            self.assertEqual(result.returncode, 0, result.stderr)
            document = last_json(result.stdout)
            self.assertEqual((document["class"], document["is_native"]), ("PinkUpperBodyServo", False))
            self.assertEqual(document["report"]["status"], "not_started")
            self.assertEqual(document["report"]["native_solver"]["native_prefix_sha256"], NATIVE_SHA256[profile])
            self.assertFalse(document["worker_started"] or document["pin_imported"])
            # No fallback: a missing interpreter fails construction with the reason recorded.
            self.assertIn("not usable", document["construct"])
            self.assertEqual(document["report_after_construct"]["status"], "failed")
        native = last_json(owner("left2", "select", NEPES_IK_BACKEND="native").stdout)
        self.assertTrue(native["is_native"])
        self.assertEqual((native["report"]["backend"], native["report"]["status"]), ("native", "in_process"))
        self.assertEqual(native["construct"], "ok")
        invalid = owner("right4", "select", NEPES_IK_BACKEND="scipy")
        self.assertEqual(invalid.returncode, 0, invalid.stderr)  # Import-only tools keep working.
        invalid = last_json(invalid.stdout)
        self.assertEqual(invalid["class"], "UnavailableUpperBodyServo")
        self.assertIn("is not supported", invalid["construct"])

    def test_missing_worker_python_fails_construction_with_clear_error(self):
        result = owner("left2", "missing-python", NEPES_IK_BACKEND=None, NEPES_IK_PYTHON=None)
        self.assertEqual(result.returncode, 0, result.stderr)
        document = last_json(result.stdout)
        self.assertIn("NEPES_IK_PYTHON is not set", document["error"])
        self.assertEqual(document["report"]["status"], "failed")


DUMMY = r'''
import json, os, sys, time
MODE, PINS = %r, %r
for line in sys.stdin:
    request = json.loads(line)
    if MODE == "sleep":
        time.sleep(60)
    if request["operation"] == "hello":
        packages = dict(PINS)
        if MODE == "badversion":
            packages["pin"] = "9.9.9"
        result = {"protocol": 1, "pid": os.getpid(), "packages": packages, "sources_sha256": {},
                  "python_version": "3.11", "implementation": "CPython", "prefix": sys.prefix, "isolated_mode": True}
        print(json.dumps({"id": request["id"], "ok": True, "result": result}), flush=True)
        continue
    if MODE == "error":
        print(json.dumps({"id": request["id"], "ok": False, "error": {"type": "ValueError", "message": "remote boom"}}), flush=True)
        sys.exit(2)
    if MODE == "badid":
        print(json.dumps({"id": request["id"] + 7, "ok": True, "result": 1}), flush=True)
    elif MODE == "nan":
        sys.stdout.write('{"id": %%d, "ok": true, "result": NaN}\n' %% request["id"]); sys.stdout.flush()
    elif MODE == "big":
        sys.stdout.write("x" * (5 * 1024 * 1024)); sys.stdout.flush()
    elif MODE == "exit":
        sys.exit(3)
    else:
        print(json.dumps({"id": request["id"], "ok": True, "result": {"echo": request["operation"]}}), flush=True)
'''
PINS = {"pin": "3.8.0", "pin-pink": "4.4.0"}


class DummyWorkerProtocolTests(unittest.TestCase):
    def setUp(self):
        self.client = load_client()
        self.directory = tempfile.TemporaryDirectory(prefix="nepes-ik-dummy-")
        self.workers = []

    def tearDown(self):
        for worker in self.workers:
            if getattr(worker, "process", None) is not None:
                worker.close()
        self.directory.cleanup()

    def start(self, mode, **options):
        script = Path(self.directory.name) / (mode + ".py")
        script.write_text(DUMMY % (mode, PINS))
        worker = self.client.WorkerClient.__new__(self.client.WorkerClient)
        self.workers.append(worker)
        worker.__init__(sys.executable, script, pins=PINS, **options)
        return worker

    def test_good_worker_round_trip_and_clean_eof_shutdown(self):
        worker = self.start("good")
        self.assertEqual(worker.request("call", servo_id=1), {"echo": "call"})
        worker.close()
        self.assertEqual(worker.returncode, 0)
        with self.assertRaisesRegex(self.client.IKBackendError, "never restarted"):
            worker.request("call")

    def test_startup_timeout_kills_and_reaps_the_worker(self):
        started = time.monotonic()
        with self.assertRaisesRegex(self.client.IKBackendError, "timed out"):
            self.start("sleep", startup_timeout=.3)
        self.assertLess(time.monotonic() - started, 8.)
        self.assertIsNotNone(self.workers[-1].process.poll())

    def test_version_mismatch_is_rejected(self):
        with self.assertRaisesRegex(self.client.IKBackendError, r"pin==3\.8\.0 \(installed 9\.9\.9\)"):
            self.start("badversion")
        self.assertIsNotNone(self.workers[-1].process.poll())

    def test_remote_error_closes_and_is_never_retried(self):
        failures = []
        worker = self.start("error", on_failure=failures.append)
        with self.assertRaisesRegex(self.client.IKBackendError, "ValueError: remote boom"):
            worker.request("call")
        self.assertEqual(worker.returncode, 2)
        self.assertEqual(len(failures), 1)
        with self.assertRaisesRegex(self.client.IKBackendError, "after a failure"):
            worker.request("call")

    def test_unreapable_worker_is_recorded_without_masking_the_failure(self):
        failures = []
        worker = self.client.WorkerClient.__new__(self.client.WorkerClient)
        process = mock.Mock(pid=4242)
        process.wait.side_effect = subprocess.TimeoutExpired("worker.py", 5)  # e.g. uninterruptible sleep
        worker.__dict__.update(lock=threading.RLock(), closed=False, owner_pid=os.getpid(), failure=None,
                               returncode=None, reap_error=None, on_failure=failures.append,
                               process=process, selector=mock.Mock())
        worker.fail("Pink IK worker timed out (30 s) during call")  # must not raise TimeoutExpired
        self.assertEqual(failures, ["Pink IK worker timed out (30 s) during call"])
        self.assertEqual((worker.closed, worker.returncode), (True, None))
        self.assertIn("pid 4242 did not exit after SIGKILL", worker.reap_error)
        process.terminate.assert_called_once()
        process.kill.assert_called_once()

    def test_invalid_responses_fail_closed(self):
        for mode, pattern in (("badid", "ID mismatch"), ("nan", "Nonfinite"), ("big", "bounded protocol size"),
                              ("exit", "exited unexpectedly")):
            with self.subTest(mode=mode):
                worker = self.start(mode, request_timeout=20)
                with self.assertRaisesRegex(self.client.IKBackendError, pattern):
                    worker.request("call")
                self.assertTrue(worker.closed)
                self.assertIsNotNone(worker.process.poll())

    def test_requests_are_bounded_and_finite(self):
        worker = self.start("good")
        with self.assertRaisesRegex(self.client.IKBackendError, "Out of range float|not JSON compliant"):
            worker.request("call", value=float("nan"))
        worker = self.start("good")
        with self.assertRaisesRegex(self.client.IKBackendError, "bounded protocol size"):
            worker.request("call", blob="x" * (5 * 1024 * 1024))


@unittest.skipUnless(ik_python(), "Pink IK environment not installed: run scripts/setup.sh and source env.sh")
class RealWorkerTests(unittest.TestCase):
    def test_isolated_ik_suite_passes_in_the_ik_python(self):
        result = subprocess.run([ik_python(), "-B", "-m", "unittest", "discover", "-s", "tests/ik_pink",
                                 "-t", "tests/ik_pink"], capture_output=True, text=True, timeout=1800,
                                cwd=str(RELEASE), env=child_environment(NEPES_IK_BACKEND=None))
        self.assertEqual(result.returncode, 0, result.stdout[-4000:] + result.stderr[-8000:])
        ran = re.search(r"Ran (\d+) tests? in", result.stderr)
        self.assertIsNotNone(ran)
        print(json.dumps({"isolated_ik_suite": {"python": ik_python(), "tests": int(ran.group(1)),
                                                "result": result.stderr.strip().splitlines()[-1]}}), flush=True)

    def test_isaac_like_processes_reproduce_recorded_commands_concurrently(self):
        """Three owner processes at once: each owns one worker; results equal the recorded native commands."""
        environment = child_environment(NEPES_IK_PYTHON=ik_python(), NEPES_IK_BACKEND=None)
        processes = [subprocess.Popen([sys.executable, "-B", str(OWNER), profile, "fixtures"], stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True, env=environment, cwd=str(RELEASE))
                     for profile in ("left2", "right4", "left2")]
        documents = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=600)
            self.assertEqual(process.returncode, 0, stderr[-4000:])
            documents.append(last_json(stdout))
        self.assertEqual(len({d["worker_pid"] for d in documents}), 3)
        for document in documents:
            self.assertEqual(document["worker_ppid"], document["owner_pid"])
            self.assertEqual(document["samples"], 25)
            self.assertLessEqual(document["max_command_difference_rad"], 1e-9)
            self.assertEqual(document["markers"], ["pink"])
            self.assertEqual(document["stage_classes"], ["PinkUpperBodyServo"])
            self.assertEqual(document["worker_returncode"], 0)
            self.assertFalse(document["pin_imported"] or document["kit_imported"])
            report = document["report"]
            self.assertEqual((report["backend"], report["status"], report["selected_by"]),
                             ("pink", "ready", "package_default"))
            self.assertEqual(report["native_solver"]["native_prefix_sha256"], NATIVE_SHA256[document["profile"]])
            self.assertEqual(report["worker"]["packages"]["pin-pink"], "4.4.0")
            self.assertEqual(set(report["worker"]["sources_sha256"]),
                             {"ik/pink/" + name for name in ("worker.py", "servo_factory.py", "pin_backend.py", "pink_qp.py")})
            self.assertFalse(alive(document["worker_pid"]))
        print(json.dumps({"isaac_like_owners": [{k: d[k] for k in ("profile", "max_command_difference_rad", "worker_pid")}
                                                for d in documents]}), flush=True)

    def test_worker_ends_with_its_owner_on_exit_exception_sigterm_and_sigkill(self):
        environment = child_environment(NEPES_IK_PYTHON=ik_python(), NEPES_IK_BACKEND=None)
        for mode in ("exit-normal", "exit-exception"):
            with self.subTest(mode=mode):
                result = subprocess.run([sys.executable, "-B", str(OWNER), "left2", mode], capture_output=True,
                                        text=True, timeout=300, env=environment, cwd=str(RELEASE))
                self.assertEqual(result.returncode, 0 if mode == "exit-normal" else 1, result.stderr[-2000:])
                self.assertTrue(wait_gone(last_json(result.stdout)["worker_pid"]), mode)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            with self.subTest(signal=sig.name):
                process = subprocess.Popen([sys.executable, "-B", str(OWNER), "right4", "hold"], stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment,
                                           cwd=str(RELEASE))
                try:
                    worker_pid = json.loads(process.stdout.readline())["worker_pid"]
                    self.assertTrue(alive(worker_pid))
                    process.send_signal(sig)
                    process.wait(timeout=30)
                    self.assertTrue(wait_gone(worker_pid), sig.name)
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.communicate(timeout=30)

    def test_batch_preflight_freezes_the_resolved_ik_environment(self):
        check = load_script(RELEASE / "scripts/check_release.py", "check_release")
        environment = child_environment(NEPES_IK_PYTHON=ik_python(), NEPES_IK_BACKEND=None)
        frozen = check.batch_preflight(environ=environment)
        self.assertEqual(frozen, check.batch_preflight(environ=environment))  # resume compares it exactly
        self.assertEqual((frozen["backend"], frozen["python"]), ("pink", ik_python()))
        self.assertEqual(frozen["packages"], load_client().read_pins())
        self.assertEqual(set(frozen["worker_sources_sha256"]),
                         {"ik/pink/" + name for name in ("worker.py", "servo_factory.py", "pin_backend.py", "pink_qp.py")})
        with tempfile.TemporaryDirectory(prefix="nepes-ik-dry-") as directory:
            sim = Path(directory) / "sim-python"  # Stand-in Isaac Python: the EULA probe finds no package.
            sim.write_text("#!/bin/sh\necho '{\"kit_app\": null}'\n")
            sim.chmod(0o755)
            output = Path(directory) / "batch"
            result = subprocess.run([sys.executable, "-B", str(RELEASE / "scripts/collect.py"), "--per-pair", "1",
                                     "--first-seed", "100000", "--output", str(output), "--isaac-python", str(sim),
                                     "--dataset-python", sys.executable, "--dry-run"], capture_output=True, text=True,
                                    timeout=600, env=environment, cwd=str(RELEASE))
            self.assertEqual(result.returncode, 0, result.stderr[-4000:])
            self.assertEqual(json.loads(result.stdout)["plan"]["upper_body_ik"], frozen)
            self.assertFalse(output.exists())

    def test_run_report_ends_with_the_final_ik_state(self):
        """run.json is refreshed at the episode end and after shutdown (never a stale ready)."""
        merger = load_script(RELEASE / "scripts/seeded/merge_seeded.py", "merge_seeded")
        environment = {"NEPES_IK_PYTHON": ik_python(), "NEPES_IK_BACKEND": None}
        result = owner("left2", "report-normal", **environment)
        self.assertEqual(result.returncode, 0, result.stderr[-4000:])
        normal = last_json(result.stdout)
        self.assertEqual([normal[key]["status"] for key in ("provenance", "episode_end", "final")],
                         ["ready", "ready", "closed"])
        self.assertEqual((normal["final"]["error"], normal["final"]["worker"]["returncode"], normal["cleanup_errors"]),
                         (None, 0, []))
        evidence = merger.upper_body_ik_evidence(RELEASE, {"upper_body_ik_backend": normal["final"]},
                                                 {"upper_body_ik_backend": normal["provenance"]})
        self.assertEqual((evidence["backend"], evidence["status"], evidence["worker_returncode"]), ("pink", "closed", 0))
        result = owner("right4", "report-kill", **environment)
        self.assertEqual(result.returncode, 0, result.stderr[-4000:])
        killed = last_json(result.stdout)
        self.assertRegex(killed["reason"], r"^IKBackendError: Pink IK worker (exited unexpectedly|exited before)")
        self.assertEqual(killed["episode_end"]["status"], "failed")
        self.assertEqual((killed["final"]["status"], killed["final"]["worker"]["returncode"]), ("failed", -signal.SIGKILL))
        self.assertIn("Pink IK worker exited", killed["final"]["error"])
        with self.assertRaisesRegex(ValueError, "did not close cleanly"):
            merger.upper_body_ik_evidence(RELEASE, {"upper_body_ik_backend": killed["final"]})

    def test_worker_crash_fails_the_attempt_without_fallback_or_restart(self):
        result = owner("left2", "crash-worker", NEPES_IK_PYTHON=ik_python(), NEPES_IK_BACKEND=None)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        document = last_json(result.stdout)
        self.assertRegex(document["solve"], r"^IKBackendError: Pink IK worker (exited unexpectedly|exited before)")
        self.assertRegex(document["new_servo"], r"^IKBackendError: .*never restarted")
        self.assertEqual(document["report"]["status"], "failed")


class BatchPreflightTests(unittest.TestCase):
    """Batch tools stop before any output or Isaac start unless the IK backend and Isaac are ready."""

    def setUp(self):
        self.check = load_script(RELEASE / "scripts/check_release.py", "check_release")
        self.directory = tempfile.TemporaryDirectory(prefix="nepes-ik-batch-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def script(self, name, body):
        path = self.root / name
        path.write_text("#!/bin/sh\n" + body)
        path.chmod(0o755)
        return path

    def test_native_invalid_or_unusable_ik_never_reaches_a_batch(self):
        for environ, pattern in (({"NEPES_IK_BACKEND": "native"}, "single-run debugging"),
                                 ({"NEPES_IK_BACKEND": "scipy"}, "is not supported"),
                                 ({"NEPES_IK_PYTHON": "/nonexistent/ik/bin/python"}, "not usable")):
            with self.subTest(environ=environ), self.assertRaisesRegex(ValueError, pattern):
                self.check.batch_preflight(environ=environ)
        stale = self.script("python", "echo '{\"ok\": false, \"error\": \"Installed packages differ\", "
                                      "\"mismatches\": {\"pin\": {\"expected\": \"3.8.0\", \"installed\": \"2.7.0\"}}}'\nexit 1\n")
        with self.assertRaisesRegex(ValueError, "before the batch.*2\\.7\\.0"):
            self.check.batch_preflight(environ={"NEPES_IK_PYTHON": str(stale)})

    def test_isaac_eula_prompt_is_detected_before_any_attempt(self):
        kit = self.root / "site" / "isaacsim" / "kit"
        kit.mkdir(parents=True)
        (kit / "kit_app.py").write_text("")
        sim = self.script("sim-python", "echo '" + json.dumps({"kit_app": str(kit / "kit_app.py")}) + "'\n")
        self.assertIn("import isaacsim", self.check.isaac_eula_problem(sim, environ={}))
        self.assertIsNone(self.check.isaac_eula_problem(sim, environ={"OMNI_KIT_ACCEPT_EULA": "YES"}))
        (kit / "EULA_ACCEPTED").write_text("yes")
        self.assertIsNone(self.check.isaac_eula_problem(sim, environ={}))
        unknown = self.script("other-python", "echo '{\"kit_app\": null}'\n")
        self.assertIsNone(self.check.isaac_eula_problem(unknown, environ={}))  # left to Kit's own check

    def test_collect_dry_run_stops_before_any_output_without_a_usable_ik_python(self):
        output = self.root / "batch"
        result = subprocess.run([sys.executable, "-B", str(RELEASE / "scripts/collect.py"), "--per-pair", "1",
                                 "--first-seed", "100000", "--output", str(output), "--isaac-python", sys.executable,
                                 "--dataset-python", sys.executable, "--dry-run"], capture_output=True, text=True,
                                timeout=300, cwd=str(RELEASE),
                                env=child_environment(NEPES_IK_BACKEND=None, NEPES_IK_PYTHON="/nonexistent/ik/bin/python"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not usable", result.stderr)
        self.assertFalse(output.exists())

    def test_resume_with_another_ik_backend_or_environment_is_refused(self):
        collector = load_script(RELEASE / "scripts/seeded/collect_seeded.py", "collect_seeded")
        output = self.root / "batch"
        output.mkdir()
        frozen = {"schema": "nepes_batch_upper_body_ik_v1", "backend": "pink", "python": "/old/ik/bin/python"}
        (output / "batch.json").write_text(json.dumps({"plan": {"upper_body_ik": frozen}, "attempts": [],
                                                       "status": "stopped"}))
        argv = ["--bundle", str(RELEASE), "--output", str(output), "--isaac-python", sys.executable,
                "--dataset-python", sys.executable, "--max-successes-per-pair", "1", "--resume"]
        with mock.patch.object(collector, "verify_bundle", return_value="fixture_manifest"), \
                mock.patch.object(collector, "upper_body_ik_plan", return_value=dict(frozen, python="/new/ik/bin/python")), \
                mock.patch.object(tempfile, "tempdir", str(self.root)), \
                self.assertRaisesRegex(ValueError, "IK backend/worker differs from the frozen batch plan"):
            collector.main(argv)

    def test_render_collection_uses_the_same_preflight(self):
        sys.path.insert(0, str(RELEASE / "scripts" / "augmentation"))
        try:
            render = load_script(RELEASE / "scripts/augmentation/render_collect.py", "render_collect")
        finally:
            sys.path.remove(str(RELEASE / "scripts" / "augmentation"))
        self.assertIsNone(render.upper_body_ik_plan(self.root, sys.executable))  # no manifest: no declared backend
        with mock.patch.dict(os.environ, {"NEPES_IK_BACKEND": "native"}), \
                self.assertRaisesRegex(ValueError, "single-run debugging"):
            render.upper_body_ik_plan(RELEASE, sys.executable)


class ReleaseToolingTests(unittest.TestCase):
    def test_setup_dry_run_builds_an_isolated_ik_env_and_never_touches_the_sim_python(self):
        # Spaces in the paths: setup.sh prints shell-quoted (printf %q) commands, compared as tokens.
        with tempfile.TemporaryDirectory(prefix="nepes ik setup ") as directory:
            env_root = Path(directory) / "env root"
            sim = Path(directory) / "isaac env" / "bin" / "python"
            for arguments in (["--sim-python", str(sim)], ["--python", "python3.11"], ["--ik-only", "--sim-python", str(sim)]):
                with self.subTest(arguments=arguments):
                    result = subprocess.run(["bash", str(RELEASE / "scripts/setup.sh"), "--dry-run", "--env-root",
                                             str(env_root), *arguments], capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    commands = [shlex.split(line[2:]) for line in result.stdout.splitlines() if line.startswith("+ ")]
                    base = str(sim) if "--sim-python" in arguments else "python3.11"
                    ik = f"{env_root}/ik/bin/python"
                    self.assertIn([base, "-m", "venv", f"{env_root}/ik"], commands)
                    self.assertIn([ik, "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--no-deps",
                                   "--only-binary=:all:", "-r", f"{RELEASE}/requirements/ik_pink.txt"], commands)
                    self.assertIn([ik, "-m", "pip", "check"], commands)
                    # Same isolation as the runtime client: -I, and no inherited native library paths.
                    self.assertIn(["env", "-u", "LD_LIBRARY_PATH", "-u", "LD_PRELOAD", ik, "-I", "-B",
                                   f"{RELEASE}/ik/pink/worker.py", "--check"], commands)
                    self.assertFalse([c for c in commands if c[:4] == [str(sim), "-m", "pip", "install"]])
                    if "--ik-only" in arguments:
                        self.assertFalse([token for command in commands for token in command
                                          if f"{env_root}/dataset" in token or f"{env_root}/sim" in token
                                          or "lerobot" in token or "isaacsim[" in token])
                    self.assertFalse(env_root.exists())

    def test_build_release_ships_ik_sources_and_manifest_declares_backend(self):
        build = load_script(RELEASE / "scripts/build_release.py", "build_release")
        names = [name for name, _ in build.release_files()]
        for required in ("ik/pink/worker.py", "ik/pink/servo_factory.py", "ik/pink/pin_backend.py", "ik/pink/pink_qp.py",
                         "ik/pink/provenance.json", "requirements/ik_pink.txt", "tests/ik_pink/_support.py",
                         "tests/ik_pink/fixtures/measured_snapshots_v1.json", "tests/ik_pink/fixtures/provenance.json",
                         "tests/ik_pink/fixtures/igris_c_260910_canonical.urdf",
                         *(f"runtime/{p}/source/robros_motion_planning/robros_motion_planning/upper_body_ik_backend.py"
                           for p in PROFILES)):
            self.assertIn(required, names)
        self.assertFalse([n for n in names if ".venv" in n.split("/") or "__pycache__" in n or n.endswith(".pyc")])
        manifest = json.loads((RELEASE / "manifest.json").read_text())
        self.assertEqual(manifest["schema"], "igris_nepes_portable_release_v1")
        self.assertEqual(manifest["ik_backend"], build.IK_BACKEND)
        self.assertEqual(manifest["ik_backend"]["default"], "pink")
        self.assertIn("pink_ik_backend", manifest["capabilities"])
        self.assertEqual(set(manifest["files"]), set(names), "Run python3 scripts/build_release.py after editing")

    def test_check_release_verifies_the_ik_environment(self):
        check = load_script(RELEASE / "scripts/check_release.py", "check_release")
        with tempfile.TemporaryDirectory(prefix="nepes-ik-check-") as directory:
            fake = Path(directory) / "python"
            fake.write_text("#!/bin/sh\necho '{\"ok\": false, \"error\": \"Installed packages differ\", "
                            "\"mismatches\": {\"pin\": {\"expected\": \"3.8.0\", \"installed\": \"3.9.0\"}}}'\nexit 1\n")
            fake.chmod(0o755)
            report, errors = check.verify_ik(fake)
            self.assertTrue(errors and "3.9.0" in errors[0])
            report, errors = check.verify_ik(Path(directory) / "missing")
            self.assertTrue(errors and "not executable" in errors[0])
        if ik_python():
            report, errors = check.verify_ik(ik_python())
            self.assertEqual(errors, [])
            self.assertTrue(report["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
