"""Worker-side servo factory: the unchanged native solver with Pinocchio/Pink numerics.

Runs only in the isolated IK environment. Every servo executes its own copy of
the native solver source, i.e. the bytes of the runtime's upper_body_control.py
that precede the backend hook (see upper_body_ik_backend.py), so a QP adapter is
bound per instance and no module is ever patched in place. Backends:

* ``pink``: Pinocchio FK/Jacobian and pink.solve_ik/quadprog (production).
* ``pinocchio``: Pinocchio FK/Jacobian and the native SciPy BVLS (diagnostics).
* ``baseline``: the native NumPy/SciPy solver unchanged (tests, diagnostics).

All three keep the native objective, bounds, posture/feedforward terms and the
command integration at the clipped previous command (never the measured q).
Adapted from the validated 2026-10-30 experiment (experimental_servo.py); the
only change is that the native source is read from the shipped runtime instead
of a separate reference copy. See provenance.json.
"""
from __future__ import annotations

import hashlib
import itertools
from pathlib import Path
import sys
import types

HOOK_MARKER = "# --- NEPES upper-body IK backend (Pink release) ---"
HOOK_SEPARATOR = "\n\n" + HOOK_MARKER + "\n"
BACKENDS = ("baseline", "pinocchio", "pink")
_ids = itertools.count()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def prepare_imports(runtime_root):
    """Make the runtime's robros_motion_planning package importable; return its root."""
    path = Path(runtime_root).resolve(strict=True)
    if not (path / "robros_motion_planning" / "robot_model.py").is_file():
        raise FileNotFoundError(f"Not a robros_motion_planning source root: {path}")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def split_native_source(text):
    """Return the native solver source: everything before the single backend hook."""
    if text.count(HOOK_MARKER) != 1 or HOOK_SEPARATOR not in text:
        raise ValueError("upper_body_control.py must end with exactly one NEPES IK backend hook")
    return text[:text.index(HOOK_SEPARATOR)]


def load_native_module(native_module, runtime_root):
    """Execute the native solver source as an independent module copy.

    Returns ``(module, identity)``. The copy's relative imports resolve to the
    same runtime's package, and the module object is removed from sys.modules
    afterwards so that it can never shadow or alter the importable runtime.
    """
    root = prepare_imports(runtime_root)
    path = Path(native_module).resolve(strict=True)
    if path != root / "robros_motion_planning" / "upper_body_control.py":
        raise ValueError("The native solver must be this runtime's robros_motion_planning/upper_body_control.py")
    data = path.read_bytes()
    prefix = split_native_source(data.decode("utf-8"))
    name = f"robros_motion_planning._nepes_ik_native_{next(_ids)}"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = "robros_motion_planning"
    sys.modules[name] = module
    try:
        exec(compile(prefix, str(path), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    identity = {"file_sha256": sha256_bytes(data), "native_prefix_sha256": sha256_bytes(prefix.encode("utf-8"))}
    return module, identity


def make_servo(model, backend="pink", *, native_module, runtime_root, **kwargs):
    """Same constructor and solve_step inputs as the native UpperBodyServo."""
    if backend not in BACKENDS:
        raise ValueError("backend must be one of " + ", ".join(BACKENDS))
    module, identity = load_native_module(native_module, runtime_root)
    base = module.UpperBodyServo
    if backend == "baseline":
        servo = base(model, **kwargs)
        servo.ik_backend_identity = {"backend": backend, "kinematics": "native_numpy",
                                     "optimization": "scipy.bvls", **identity}
        return servo
    from pin_backend import PinocchioUpperBodyKinematics
    from pink_qp import PinkBoundedLeastSquares
    kinematics = PinocchioUpperBodyKinematics(model.urdf_path)
    qp = PinkBoundedLeastSquares() if backend == "pink" else None
    if qp is not None:
        # Bound to this private module copy only; scipy and the runtime stay untouched.
        module.lsq_linear = qp

    class WorkerUpperBodyServo(base):
        def pose_and_jacobian(self, joint_positions, pelvis_pose_w, arm):
            return kinematics.pose_and_jacobian(joint_positions, pelvis_pose_w, arm)

        def solve_step(self, joint_positions, pelvis_pose_w, target_pose_w, arm, dt,
                       previous_targets=None, **options):
            if qp is not None:
                qp.dt = dt
            return super().solve_step(joint_positions, pelvis_pose_w, target_pose_w, arm, dt,
                                      previous_targets, **options)

    servo = WorkerUpperBodyServo(model, **kwargs)
    servo.ik_backend_identity = {
        "backend": backend, "kinematics": "pinocchio",
        "optimization": "pink.solve_ik/quadprog" if qp is not None else "scipy.bvls",
        "command_integration": "clipped_previous_target", **identity}
    servo.ik_kinematics, servo.ik_qp = kinematics, qp
    return servo
