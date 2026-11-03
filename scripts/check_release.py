#!/usr/bin/env python3
"""Verify the shipped files without importing simulator dependencies.

When the Pink IK environment exists (--ik-python, NEPES_IK_PYTHON from env.sh,
or <release>/.venv/ik), also verify its exact pins, imports, worker source
hashes and a small Pink QP solve. Before setup.sh this check is skipped unless
--require-ik is given. With NEPES_SIM_PYTHON (or --sim-python) it also reports
whether that Isaac Sim installation still waits for its one-time EULA prompt.

batch_preflight() is the same check for batch tools (collect_seeded.py,
render_collect.py): it runs before any output or Isaac process exists, also in
--dry-run, and returns the IK backend/worker identity frozen in the batch plan.
"""
import argparse
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
WORKER = root / "ik" / "pink" / "worker.py"
DEFAULT_IK_PYTHON = root / ".venv" / "ik" / "bin" / "python"
REQUIREMENTS = root / "requirements" / "ik_pink.txt"
CLIENT = root / "runtime" / "left2" / "source" / "robros_motion_planning" / "robros_motion_planning" / \
    "upper_body_ik_backend.py"
# Same interpreter isolation as the runtime client (upper_body_ik_backend.WorkerClient).
STRIPPED_ENVIRONMENT = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONEXECUTABLE", "LD_LIBRARY_PATH", "LD_PRELOAD")
EULA_PROBE = (
    "import importlib.util, json, os\n"
    "spec = importlib.util.find_spec('isaacsim')\n"  # Locates the package without importing (no prompt).
    "places = list(spec.submodule_search_locations or []) if spec else []\n"
    "kit_app = os.path.join(places[0], 'kit', 'kit_app.py') if places else ''\n"
    "print(json.dumps({'kit_app': kit_app if os.path.isfile(kit_app) else None}))\n")


def verify_files():
    manifest = json.loads((root / "manifest.json").read_text())
    errors = []
    for relative, expected in manifest["files"].items():
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            errors.append(f"Missing/external file: {relative}")
            continue
        if path.stat().st_size < 1024 and path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1"):
            errors.append(f"Git LFS pointer instead of asset: {relative}; run git lfs install && git lfs pull")
            continue
        digest = sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            errors.append(f"Modified file: {relative}")
    return manifest, errors


def find_ik_python(explicit=None):
    if explicit:
        return Path(explicit).expanduser(), "--ik-python"
    if os.environ.get("NEPES_IK_PYTHON", "").strip():
        return Path(os.environ["NEPES_IK_PYTHON"].strip()).expanduser(), "NEPES_IK_PYTHON"
    if DEFAULT_IK_PYTHON.exists():
        return DEFAULT_IK_PYTHON, "release default .venv/ik"
    return None, None


def clean_environment(environ=None):
    environment = {key: value for key, value in (os.environ if environ is None else environ).items()
                   if key not in STRIPPED_ENVIRONMENT}
    environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    return environment


def verify_ik(python, worker=WORKER, timeout=300, environ=None):
    """Run ``worker.py --check`` in the IK Python; return (report or None, errors)."""
    python = Path(os.path.abspath(python))  # Keep venv symlinks.
    if not python.is_file() or not os.access(python, os.X_OK):
        return None, [f"Pink IK Python is not executable: {python}"]
    environment = clean_environment(environ)
    try:
        result = subprocess.run([str(python), "-I", "-B", str(worker), "--check"], capture_output=True,
                                text=True, timeout=timeout, env=environment)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, [f"Pink IK environment check could not run: {exc}"]
    try:
        report = json.loads(result.stdout)
    except ValueError:
        return None, [f"Pink IK environment check failed (exit {result.returncode}): {result.stderr.strip()[-2000:]}"]
    if result.returncode != 0 or report.get("ok") is not True:
        return report, [f"Pink IK environment check failed: {report.get('error')}; "
                        f"mismatches: {json.dumps(report.get('mismatches', {}), sort_keys=True)}"]
    return report, []


def load_client():
    """The runtime's stdlib-only IK client: the same backend and interpreter resolution as Isaac."""
    spec = importlib.util.spec_from_file_location("nepes_release_ik_client", CLIENT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def isaac_eula_problem(sim_python, environ=None):
    """Message if this Isaac Sim still asks for its EULA on first import, else None.

    Mirrors isaacsim/kit/kit_app.py check_eula(). That prompt would block an
    unattended attempt until its wall limit, so batches stop before starting one.
    An installation this probe cannot inspect is left to Kit's own check.
    """
    environ = os.environ if environ is None else environ
    if environ.get("OMNI_KIT_ACCEPT_EULA", "N").lower() in ("y", "yes", "1"):
        return None
    try:
        result = subprocess.run([str(sim_python), "-I", "-c", EULA_PROBE], capture_output=True, text=True,
                                timeout=120, env=clean_environment(environ))
        kit_app = json.loads(result.stdout).get("kit_app") if result.returncode == 0 else None
    except (OSError, ValueError, AttributeError, subprocess.TimeoutExpired):
        return None
    if not kit_app:
        return None
    try:
        with (Path(os.path.realpath(kit_app)).parent / "EULA_ACCEPTED").open() as stream:
            if stream.readline().strip().lower() in ("y", "yes", "1"):
                return None
    except OSError:
        pass
    return (f"Isaac Sim has not recorded EULA acceptance for {sim_python}. Run it once in a terminal, read the "
            f"notice and answer the prompt: \"{sim_python}\" -c 'import isaacsim'. Otherwise the first attempt "
            "waits for that prompt (its output goes to the attempt log) until the attempt wall limit.")


def batch_preflight(sim_python=None, environ=None):
    """Checks before any batch output or Isaac process; returns the IK block frozen in the batch plan.

    The backend and worker interpreter are resolved exactly as the Isaac runtime
    will resolve them (NEPES_IK_BACKEND, NEPES_IK_PYTHON, <release>/.venv/ik).
    Batches always use the release's declared backend: the native solver is a
    single-run debugging option and never enters a batch. Raises ValueError.
    """
    environ = os.environ if environ is None else environ
    declared = json.loads((root / "manifest.json").read_text()).get("ik_backend")
    if declared is None:
        return None
    client = load_client()
    try:
        backend, _ = client.resolve_backend(environ)
        if backend != declared["default"]:
            raise ValueError(
                f"{client.BACKEND_ENV}={backend} is a single-run debugging option; batches of this release always "
                f"use the {declared['default']} backend, so one batch can never mix solvers. Unset "
                f"{client.BACKEND_ENV} (or use the previous native-solver release for a native dataset).")
        python, _ = client.resolve_worker_python(environ)
    except client.IKBackendError as exc:
        raise ValueError(str(exc)) from exc
    report, errors = verify_ik(python, environ=environ)
    if errors:
        raise ValueError("Pink IK environment check failed before the batch (no Isaac started, nothing written): "
                         + "; ".join(errors) + ". Fix it with bash scripts/setup.sh --ik-only and source env.sh.")
    if sim_python is not None and (problem := isaac_eula_problem(sim_python, environ)):
        raise ValueError(problem)
    with REQUIREMENTS.open("rb") as stream:
        requirements = sha256(stream.read()).hexdigest()
    return {"schema": "nepes_batch_upper_body_ik_v1", "backend": backend, "python": python,
            "python_version": report["python_version"], "packages": report["packages"],
            "requirements_sha256": requirements, "worker_sources_sha256": report["sources_sha256"],
            "preflight": "ik/pink/worker.py --check passed"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ik-python", help="IK environment Python (default: NEPES_IK_PYTHON, then .venv/ik)")
    parser.add_argument("--require-ik", action="store_true", help="Fail if no installed IK environment is found")
    parser.add_argument("--sim-python", default=os.environ.get("NEPES_SIM_PYTHON"),
                        help="Isaac Python whose one-time EULA status is reported (default: NEPES_SIM_PYTHON)")
    args = parser.parse_args(argv)
    manifest, errors = verify_files()
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Verified {len(manifest['files'])} release files; existing runs and installed SDKs are excluded.")
    backend = manifest.get("ik_backend", {})
    print(f"Upper-body IK backend default: {backend.get('default', 'native')} "
          f"(override for debugging: {backend.get('selection_environment', 'NEPES_IK_BACKEND')}).")
    if args.sim_python and (problem := isaac_eula_problem(args.sim_python)):
        print("WARNING: " + problem, file=sys.stderr)
    python, source = find_ik_python(args.ik_python)
    if python is None:
        message = ("Pink IK environment not installed yet (bash scripts/setup.sh creates <env-root>/ik and "
                   "exports NEPES_IK_PYTHON in env.sh); IK check skipped.")
        print(message, file=sys.stderr if args.require_ik else sys.stdout)
        return 1 if args.require_ik else 0
    report, ik_errors = verify_ik(python)
    if ik_errors:
        print("\n".join(ik_errors), file=sys.stderr)
        return 1
    packages = report["packages"]
    print(f"Verified Pink IK environment ({source}): {python} - Python {report['python_version']}, "
          f"pin {packages['pin']}, pin-pink {packages['pin-pink']}, quadprog {packages['quadprog']}, "
          f"numpy {packages['numpy']}; Pink QP smoke error {report['pink_qp_max_error']:.1e}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
