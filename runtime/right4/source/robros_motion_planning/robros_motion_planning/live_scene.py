"""Live NEPES object state, episode-boundary reset, and grasp evidence.

Object actor poses and COM velocities come from PhysX tensor views, never stale
USD transforms. This module does not attach objects to hands or move them during
an episode. Contact reports and stable relative lift are separate evidence: the
latter alone is explicitly not proof of hand-object contact.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import copy
import math
import weakref

import numpy as np

from .snapshot import capture_snapshot, matrix_to_pose, pose_to_matrix


NEPES_DYNAMIC_NAMES = tuple(f"Ring_{index:02d}" for index in range(1, 6)) + ("CleanerCover", "asset_jig_rev01")


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value).copy()


class _SupportContactDiagnostics:
    """Read native actor/shape identities after each physics step, without inference.

    Tensor filters can resolve a USD mesh path while the native static actor has
    a different identity. These bounded reports diagnose that case; they never
    replace the filtered force used by the task's support/success conditions.
    Native impulse signs remain in native actor order until independently
    validated. A new (possibly empty) report replaces the previous step's data.
    """

    def __init__(self, object_paths, *, point_limit=128, pair_limit=64):
        self.object_paths = tuple(object_paths)
        self.point_limit, self.pair_limit = point_limit, pair_limit
        self.sequence = 0
        self.elapsed = 0.
        self._reports = {}
        self._subscription = None
        self._read_contacts = None
        self._decode_path = None
        self._error = "Native diagnostic subscription has not started"

    def start(self):
        # Official installed ContactReportImmediateAPIDemo uses this immediate
        # contact-buffer API. An explicit post-step subscription avoids treating
        # an old callback as current when an object has no contacts this step.
        from omni.physx import get_physx_interface, get_physx_simulation_interface
        from pxr import PhysicsSchemaTools

        self._read_contacts = get_physx_simulation_interface().get_contact_report
        self._decode_path = lambda value: str(PhysicsSchemaTools.intToSdfPath(value))
        reference = weakref.ref(self)

        def after_step(dt):
            instance = reference()
            if instance is not None:
                instance._after_step(dt)

        self._subscription = get_physx_interface().subscribe_physics_on_step_events(
            after_step, False, 1000)
        self._error = "Awaiting first post-physics contact report"

    @staticmethod
    def _vector(value):
        vector = np.asarray(list(value), dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError("Invalid native contact vector")
        return vector.tolist()

    def _after_step(self, dt):
        self.sequence += 1
        self._reports = {}
        try:
            dt = float(dt)
            if not math.isfinite(dt) or dt <= 0:
                raise ValueError("Invalid native diagnostic physics dt")
            self.elapsed += dt
            headers, data = self._read_contacts()
            reports = {path: {"pairs": [], "points": 0, "truncated": False,
                              "matched_pair_count": 0} for path in self.object_paths}
            for header in headers:
                paths = {name: self._decode_path(getattr(header, name))
                         for name in ("actor0", "actor1", "collider0", "collider1")}
                matched = [path for path in self.object_paths
                           if any(value == path or value.startswith(path + "/")
                                  for value in paths.values())]
                if not matched:
                    continue
                offset, count = int(header.contact_data_offset), int(header.num_contact_data)
                if offset < 0 or count < 0 or offset + count > len(data):
                    raise ValueError("Native contact report has invalid point offsets")
                for path in matched:
                    report = reports[path]
                    report["matched_pair_count"] += 1
                    if len(report["pairs"]) >= self.pair_limit:
                        report["truncated"] = True
                        continue
                    pair = {**paths, "native_event_type": str(header.type),
                            "reported_point_count": count, "points": []}
                    take = min(count, self.point_limit - report["points"])
                    report["truncated"] |= take < count
                    for index in range(offset, offset + take):
                        contact = data[index]
                        separation = float(contact.separation)
                        if not math.isfinite(separation):
                            raise ValueError("Invalid native contact separation")
                        point = {"position_w_m": self._vector(contact.position),
                                 "normal_w_native": self._vector(contact.normal),
                                 "impulse_w_native_n_s": self._vector(contact.impulse),
                                 "separation_m": separation,
                                 "face_index0": int(contact.face_index0),
                                 "face_index1": int(contact.face_index1)}
                        for side in (0, 1):
                            name = f"material{side}"
                            # Invalid native face indices have no material
                            # lookup; accessing the binding emits a PhysX
                            # warning for every contact. Geometry/impulse
                            # diagnostics remain available without material.
                            point[name] = (None if point[f"face_index{side}"] in (-1, 0xFFFFFFFF)
                                           else self._decode_path(getattr(contact, name)))
                        pair["points"].append(point)
                    report["points"] += take
                    report["pairs"].append(pair)
            self._reports = reports
            self._error = None
        except Exception as error:
            # Diagnostic failure must not change or fabricate task evidence.
            self._error = f"{type(error).__name__}: {error}"

    def read(self, object_path):
        report = self._reports.get(object_path)
        return {"available": self._error is None and report is not None,
                "source": "physx_post_step_native_contact_report",
                "physics_step_sequence": self.sequence,
                "physics_time_since_subscription_s": self.elapsed,
                "report_step_sequence": self.sequence if report is not None else None,
                "used_for_support_or_success": False,
                "impulse_sign_convention": "native actor order; not converted to object force",
                "reason": self._error,
                **({} if report is None else copy.deepcopy(report))}

    def close(self):
        subscription, self._subscription = self._subscription, None
        try:
            if subscription is not None and hasattr(subscription, "unsubscribe"):
                subscription.unsubscribe()
        finally:
            self._read_contacts = self._decode_path = None
            self._reports = {}
            self._error = "Native contact diagnostic closed"


def discover_nepes_objects(stage: Any, scene_root: str = "/World/Scene") -> dict[str, str]:
    """Resolve the seven dynamic compound bodies required by the five tasks."""
    from pxr import UsdPhysics

    result = {}
    for name in NEPES_DYNAMIC_NAMES:
        path = scene_root.rstrip("/") + "/" + name
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsActive() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"NEPES dynamic rigid body is missing: {path}")
        api = UsdPhysics.RigidBodyAPI(prim)
        if api.GetRigidBodyEnabledAttr().Get() is False or api.GetKinematicEnabledAttr().Get() is True:
            raise ValueError(f"Task object must be a dynamic enabled rigid body: {path}")
        result[name] = path
    return result


def enable_hand_contact_reports(stage: Any, robot_root: str = "/World/envs/env_0/Robot") -> dict[str, list[str]]:
    """Enable reports on physical palm/finger bodies; leave joint/pose state alone.

    Prefer calling before the first physics step. The existing IGRIS asset also
    enables contact sensors at spawn; applying an already-present API is safe.
    """
    from pxr import PhysxSchema, Usd, UsdPhysics
    import carb.settings

    # IsaacLab disables CPU contact processing by default. Its ContactSensor
    # enables this setting in its constructor; direct tensor contact views must
    # do the same or a valid-looking view can report only zero forces.
    carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)

    root = stage.GetPrimAtPath(robot_root)
    if not root.IsValid():
        raise ValueError(f"Robot prim not found: {robot_root}")
    result: dict[str, list[str]] = {"left": [], "right": []}
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        name = prim.GetName()
        for arm, prefix in (("left", "l_"), ("right", "r_")):
            is_finger = name.startswith(prefix) and any(part in name for part in ("thumb", "index", "middle", "ring", "little"))
            is_gripper = name in (prefix + "1_dof_gripper_fixed", prefix + "1_dof_gripper_moving")
            if name == arm.title() + "_Hand" or is_finger or is_gripper:
                report = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                report.CreateThresholdAttr(0.0)
                result[arm].append(str(prim.GetPath()))
    return result


def _support_pair_paths(support_pairs: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(support_pairs, Mapping):
        raise ValueError("Support pairs must map exact object paths to exact static collider paths")
    result = dict(support_pairs)
    for object_path, support_path in result.items():
        for path in (object_path, support_path):
            if not isinstance(path, str) or not path.startswith("/") or any(c in path for c in "*?[]()|"):
                raise ValueError("Support pairs require exact absolute prim paths")
        if object_path == support_path:
            raise ValueError("An object cannot support itself")
    return result


def _support_contact_prims(stage: Any, object_path: str, support_path: str, *,
                           allow_dynamic_support: bool = False) -> tuple[Any, Any]:
    """Validate one dynamic sensor and one exact static CollisionAPI prim."""
    from pxr import Usd, UsdPhysics

    obj, support = (stage.GetPrimAtPath(path) for path in (object_path, support_path))
    if not obj.IsValid() or not obj.IsActive() or not obj.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError(f"Support sensor is not an active dynamic rigid body: {object_path}")
    body = UsdPhysics.RigidBodyAPI(obj)
    if body.GetRigidBodyEnabledAttr().Get() is False or body.GetKinematicEnabledAttr().Get() is True:
        raise ValueError(f"Support sensor must be dynamic and enabled: {object_path}")
    if allow_dynamic_support:
        if not support.IsValid() or not support.IsActive() or not support.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"Dynamic support must name one exact rigid actor: {support_path}")
        support_body = UsdPhysics.RigidBodyAPI(support)
        if support_body.GetRigidBodyEnabledAttr().Get() is False or support_body.GetKinematicEnabledAttr().Get() is True:
            raise ValueError(f"Dynamic support must remain enabled and nonkinematic: {support_path}")
        colliders = [prim for prim in Usd.PrimRange(support, Usd.TraverseInstanceProxies())
            if prim.IsActive() and prim.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False]
        if not colliders:
            raise ValueError(f"Dynamic support has no active collision geometry: {support_path}")
        return obj, support
    if not support.IsValid() or not support.IsActive() or not support.HasAPI(UsdPhysics.CollisionAPI):
        raise ValueError(f"Support must be one active static collider: {support_path}")
    if UsdPhysics.CollisionAPI(support).GetCollisionEnabledAttr().Get() is False:
        raise ValueError(f"Support collision is disabled: {support_path}")
    ancestor = support
    while ancestor.IsValid() and not ancestor.IsPseudoRoot():
        if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"Support collider cannot have a rigid-body ancestor: {support_path}")
        ancestor = ancestor.GetParent()
    return obj, support


def enable_support_contact_reports(stage: Any, support_pairs: Mapping[str, str], *,
                                   dynamic_support_paths: Sequence[str] = ()) -> dict[str, str]:
    """Validate authored supports and idempotently register object reports.

    The runtime first calls ``prepare_support_contact_reporting`` before native
    actor creation. This later preflight runs after static fixture authoring;
    adding the API here alone may not activate an already parsed native actor.
    No collider geometry, body state, mass, friction, or drives are changed.
    All pairs are validated before any API is applied.
    """
    from pxr import PhysxSchema
    import carb.settings

    pairs = _support_pair_paths(support_pairs)
    dynamic = set(dynamic_support_paths)
    if not dynamic <= set(pairs.values()):
        raise ValueError("Every explicit dynamic support must be a configured exact support path")
    objects = [_support_contact_prims(stage, obj, support, allow_dynamic_support=support in dynamic)[0]
               for obj, support in pairs.items()]
    if pairs:
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
    for prim in objects:
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
    return pairs


def prepare_support_contact_reporting(env, env_ids, *, object_paths, keep_awake_object_paths=()):
    """Enable configured support measurements before PhysX parses bodies.

    A support sensor must not rely on a report API authored after native actor
    creation being hot-reloaded. Static task fixtures may not exist yet, so this
    hook only validates and enables the requested dynamic reporters. The normal
    support-pair preflight still runs later. An explicit reporter-only opt-in
    sets sleepThreshold=0 before actor creation so resting support impulses
    remain observable. Its original value and the physics parameter change are
    reported. No pose, geometry, mass, collision offset or force is changed.
    """
    import omni.usd
    import carb.settings
    from pxr import UsdPhysics, PhysxSchema

    stage = omni.usd.get_context().get_stage()
    if isinstance(keep_awake_object_paths, (str, bytes)):
        raise ValueError("Keep-awake object paths must be an explicit sequence of reporter paths")
    awake_paths = set(keep_awake_object_paths)
    if not awake_paths <= set(object_paths):
        raise ValueError("Keep-awake objects must be selected exact dynamic contact reporters")
    prims = []
    for path in object_paths:
        if not isinstance(path, str) or not path.startswith("/") or any(c in path for c in "*?[]()|"):
            raise ValueError("Support contact reporters require exact absolute object paths")
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsActive() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"Support contact reporter is not a rigid body: {path}")
        body = UsdPhysics.RigidBodyAPI(prim)
        if body.GetRigidBodyEnabledAttr().Get() is False or body.GetKinematicEnabledAttr().Get() is True:
            raise ValueError(f"Support contact reporter must be dynamic and enabled: {path}")
        prims.append(prim)
    if prims:
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
    awake_overrides = []
    for prim in prims:
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
        if str(prim.GetPath()) in awake_paths:
            physics = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            original = physics.GetSleepThresholdAttr().Get()
            physics.CreateSleepThresholdAttr(0.0)
            actual = physics.GetSleepThresholdAttr().Get()
            if actual != 0.0:
                raise RuntimeError("Prestartup support reporter sleep threshold did not apply")
            awake_overrides.append({"object_path": str(prim.GetPath()),
                "original_sleep_threshold": original, "applied_sleep_threshold": actual,
                "verified_usd_attribute": "physxRigidBody:sleepThreshold",
                "phase": "prestartup_before_native_actor_creation"})
    env.nepes_support_contact_reporting_provenance = {
        "schema": "nepes_support_contact_reporting_v1",
        "mode": "prestartup", "dynamic_reporters": list(object_paths),
        "report_threshold_n": 0., "simulation_state_or_physics_parameters_changed": bool(awake_overrides),
        "simulation_state_written": False, "asset_files_modified": False,
        "keep_awake_overrides": awake_overrides,
        "force_values_synthesized": False,
        "purpose": "Activate configured filtered support-force measurements before native actor creation",
        "diagnostic_callback_required": False,
        "support_filter_paths_and_success_predicates_changed": False}


def prepare_support_contact_diagnostics(env, env_ids, *, object_paths):
    """Compatibility entry point for the original opt-in diagnostic event."""
    prepare_support_contact_reporting(env, env_ids, object_paths=object_paths)
    env.nepes_support_contact_diagnostic_provenance = env.nepes_support_contact_reporting_provenance


def read_usd_object_poses(stage: Any, paths: Sequence[str]) -> dict[str, list[float]]:
    """Read static/support poses after sim.render with Fabric disabled.

    Use LiveScene.read_states for dynamic objects. This helper intentionally does
    not claim that a USD transform is current physics state.
    """
    from pxr import UsdGeom

    cache = UsdGeom.XformCache()
    result = {}
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsActive():
            raise ValueError(f"Active USD object not found: {path}")
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float).T
        left, _, right = np.linalg.svd(matrix[:3, :3])
        matrix[:3, :3] = left @ np.diag([1, 1, np.linalg.det(left @ right)]) @ right
        result[path] = matrix_to_pose(matrix)
    return result


class RenderPoseValidationError(RuntimeError):
    """USD geometry cannot safely be recorded as the measured physics state."""

    def __init__(self, errors: Mapping[str, Mapping[str, Any]]):
        self.errors = {path: dict(details) for path, details in errors.items()}
        super().__init__(f"Rendered USD poses differ from measured physics state: {self.errors}")


def validate_render_poses(
    stage: Any,
    measured_poses_w: Mapping[str, Sequence[float]],
    position_tolerance: float = 0.002,
    rotation_tolerance: float = 0.01,
) -> dict[str, dict[str, float]]:
    """Fail closed when USD render geometry is stale relative to PhysX actors.

    Call after ``sim.render()`` with Fabric disabled and physics-to-USD writeback
    enabled. Keys are exact body prim paths; poses are actor xyz+wxyz, not COM
    poses. A fresh USD XformCache is used on every call, so a previously matching
    frame cannot hide a later writeback failure. Returned errors use metres and
    radians. Quaternion sign does not affect the rotation comparison.
    """
    tolerances = np.asarray([position_tolerance, rotation_tolerance], dtype=float)
    if not np.isfinite(tolerances).all() or np.any(tolerances <= 0):
        raise ValueError("Render pose tolerances must be positive and finite")
    if not isinstance(measured_poses_w, Mapping) or not measured_poses_w:
        raise ValueError("Render validation requires a nonempty mapping of exact body paths to poses")
    measured = {}
    for path, pose in measured_poses_w.items():
        if not isinstance(path, str) or not path.startswith("/") or any(c in path for c in "*?[]()|"):
            raise ValueError(f"Render validation requires an exact absolute prim path: {path}")
        measured[path] = pose_to_matrix(pose)

    errors = {}
    valid_paths = []
    for path in measured:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsActive():
            errors[path] = {"reason": "missing_or_inactive_usd_prim"}
        else:
            valid_paths.append(path)
    try:
        rendered_poses = read_usd_object_poses(stage, valid_paths)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        errors["usd_transform_read"] = {"reason": f"{type(error).__name__}: {error}"}
        raise RenderPoseValidationError(errors) from error

    differences = {}
    for path, pose in rendered_poses.items():
        rendered = pose_to_matrix(pose)
        reference = measured[path]
        position_error = float(np.linalg.norm(rendered[:3, 3] - reference[:3, 3]))
        rotation_delta = reference[:3, :3].T @ rendered[:3, :3]
        rotation_error = float(np.arccos(np.clip((np.trace(rotation_delta) - 1) / 2, -1, 1)))
        differences[path] = {"position_error_m": position_error, "rotation_error_rad": rotation_error}
        if position_error > position_tolerance or rotation_error > rotation_tolerance:
            errors[path] = {"reason": "stale_usd_transform", **differences[path],
                            "position_tolerance_m": float(position_tolerance),
                            "rotation_tolerance_rad": float(rotation_tolerance)}
    if errors:
        raise RenderPoseValidationError(errors)
    return differences


class LiveScene:
    """PhysX-backed views for independent workcell rigid bodies and hand contacts."""

    def __init__(
        self,
        simulation_view: Any,
        object_paths: Mapping[str, str] | Sequence[str],
        *,
        physics_dt: float,
        stage: Any = None,
        robot_root: str = "/World/envs/env_0/Robot",
        hand_paths: Mapping[str, Sequence[str]] | None = None,
        enable_contacts: bool = True,
        contact_force_threshold_n: float = 0.05,
        support_pairs: Mapping[str, str] | None = None,
        support_contact_diagnostics: bool = False,
        dynamic_support_paths: Sequence[str] = (),
    ):
        if type(support_contact_diagnostics) is not bool:
            raise ValueError("support_contact_diagnostics must be an explicit boolean")
        self.simulation_view = simulation_view
        self.physics_dt = float(physics_dt)
        self.contact_force_threshold_n = float(contact_force_threshold_n)
        if not math.isfinite(self.physics_dt) or self.physics_dt <= 0:
            raise ValueError("physics_dt must be positive and finite")
        if not math.isfinite(self.contact_force_threshold_n) or self.contact_force_threshold_n < 0:
            raise ValueError("Contact force threshold must be finite and nonnegative")
        self.object_paths = dict(object_paths) if isinstance(object_paths, Mapping) else {path.rsplit("/", 1)[-1]: path for path in object_paths}
        if not self.object_paths or len(set(self.object_paths.values())) != len(self.object_paths):
            raise ValueError("Object paths must be nonempty and unique")
        self._views = {}
        for path in self.object_paths.values():
            if not isinstance(path, str) or not path.startswith("/") or any(c in path for c in "*?[]()|"):
                raise ValueError(f"Object view requires one exact absolute prim path: {path}")
            view = simulation_view.create_rigid_body_view(path)
            if view.count != 1 or list(view.prim_paths) != [path]:
                raise ValueError(f"Expected exactly one rigid actor at {path}, found {list(view.prim_paths)}")
            self._views[path] = view
        self._active_episode = False
        self._closed = False
        self._initial_states: dict[str, dict[str, Any]] | None = None
        self._contact_views = {}
        self._contact_errors = {}
        self.support_pairs = _support_pair_paths({} if support_pairs is None else support_pairs)
        if not set(self.support_pairs) <= set(self._views):
            raise ValueError("Every support sensor must be an exact tracked dynamic object path")
        dynamic_support_paths = set(dynamic_support_paths)
        if not dynamic_support_paths <= set(self.support_pairs.values()):
            raise ValueError("Explicit dynamic support is absent from configured support pairs")
        self._support_views = {}
        self._support_errors = {}
        self._support_diagnostics = None
        self._support_diagnostics_error = None
        if enable_contacts and stage is not None:
            discovered = enable_hand_contact_reports(stage, robot_root)
            if hand_paths is None:
                hand_paths = discovered
        self.hand_paths = {arm: list(paths) for arm, paths in (hand_paths or {}).items()}
        for arm in ("left", "right"):
            sensors = self.hand_paths.get(arm, [])
            for object_path in self.object_paths.values():
                key = (arm, object_path)
                if not enable_contacts or not sensors:
                    self._contact_errors[key] = "Hand contact sensors disabled or no physical palm/finger bodies found"
                    continue
                try:
                    # One filtered view per object avoids assuming that filter
                    # indices correspond one-to-one to compound collision shapes.
                    view = simulation_view.create_rigid_contact_view(
                        sensors, filter_patterns=[[object_path] for _ in sensors],
                        max_contact_data_count=max(256, 64 * len(sensors)),
                    )
                    if view.sensor_count < 1 or view.filter_count < 1:
                        raise ValueError("PhysX contact view resolved no sensors or object filters")
                    resolved = list(view.sensor_paths)
                    if set(resolved) != set(sensors):
                        raise ValueError("PhysX contact sensor paths differ from requested hand bodies")
                    self._contact_views[key] = view
                except Exception as error:
                    self._contact_errors[key] = f"{type(error).__name__}: {error}"
        for object_path, support_path in self.support_pairs.items():
            if not enable_contacts:
                self._support_errors[object_path] = "Contact sensors disabled"
                continue
            try:
                if stage is not None:
                    from pxr import PhysxSchema
                    obj, _ = _support_contact_prims(stage, object_path, support_path,
                        allow_dynamic_support=support_path in dynamic_support_paths)
                    if not obj.HasAPI(PhysxSchema.PhysxContactReportAPI):
                        raise ValueError("Call enable_support_contact_reports before physics reset")
                view = simulation_view.create_rigid_contact_view(
                    [object_path], filter_patterns=[[support_path]], max_contact_data_count=256)
                if (view.sensor_count != 1 or view.filter_count != 1
                        or list(view.sensor_paths) != [object_path]
                        or np.asarray(view.filter_paths, dtype=object).reshape(-1).tolist() != [support_path]):
                    raise ValueError("Support contact view did not resolve the exact object/static collider pair")
                self._support_views[object_path] = view
            except Exception as error:
                self._support_errors[object_path] = f"{type(error).__name__}: {error}"
        if support_contact_diagnostics and enable_contacts and self.support_pairs:
            self._support_diagnostics = _SupportContactDiagnostics(self.support_pairs)
            try:
                self._support_diagnostics.start()
            except Exception as error:
                self._support_diagnostics_error = f"{type(error).__name__}: {error}"

    def _support_contact_diagnostic_evidence(self, path):
        report = self._support_diagnostics.read(path)
        if self._support_diagnostics_error is not None:
            report.update(available=False, reason=self._support_diagnostics_error)
        view = self._support_views.get(path)
        if view is not None:
            report["tensor_sensor_paths"] = list(view.sensor_paths)
            report["tensor_filter_paths"] = np.asarray(view.filter_paths, dtype=object).tolist()
            try:
                force = _numpy(view.get_net_contact_forces(dt=self.physics_dt))
                if force.shape != (1, 3) or not np.isfinite(force).all():
                    raise ValueError("Invalid unfiltered net contact force")
                report["unfiltered_net_normal_force_w_n"] = force[0].tolist()
                report["unfiltered_net_normal_force_available"] = True
            except Exception as error:
                report["unfiltered_net_normal_force_available"] = False
                report["unfiltered_net_normal_force_error"] = f"{type(error).__name__}: {error}"
        return report

    def resolve_path(self, object_name_or_path: str) -> str:
        self._ensure_open()
        path = self.object_paths.get(object_name_or_path, object_name_or_path)
        if path not in self._views:
            raise KeyError(f"Object is not tracked: {object_name_or_path}")
        return path

    def read_masses(self) -> dict[str, float]:
        """Read cooked actor masses once for probe provenance, without writes."""
        self._ensure_open()
        result = {}
        for path, view in self._views.items():
            mass = _numpy(view.get_masses()).reshape(-1)
            if mass.shape != (1,) or not np.isfinite(mass).all() or mass[0] <= 0:
                raise ValueError(f"Invalid live rigid-body mass at {path}")
            result[path] = float(mass[0])
        return result

    def read_states(self) -> dict[str, dict[str, Any]]:
        """Return copies: actor poses are xyz+wxyz, velocities are world COM twist."""
        self._ensure_open()
        states = {}
        for path, view in self._views.items():
            transforms = _numpy(view.get_transforms())
            velocities = _numpy(view.get_velocities())
            if transforms.shape != (1, 7) or velocities.shape != (1, 6):
                raise ValueError(f"Unexpected PhysX state shape for {path}")
            if not np.isfinite(transforms).all() or not np.isfinite(velocities).all():
                raise ValueError(f"Nonfinite PhysX object state: {path}")
            pose = transforms[0, [0, 1, 2, 6, 3, 4, 5]].astype(float).tolist()
            pose_to_matrix(pose)
            states[path] = {
                "pose_w": pose, "velocity_w": velocities[0].astype(float).tolist(),
                "pose_source": "physx_rigid_actor_tensor", "velocity_frame": "world_com_linear_angular",
            }
        return states

    def capture_initial_state(self) -> dict[str, dict[str, Any]]:
        if self._active_episode:
            raise RuntimeError("Cannot replace reset state during an active episode")
        self._initial_states = self.read_states()
        return copy.deepcopy(self._initial_states)

    def begin_episode(self) -> None:
        if self._active_episode:
            raise RuntimeError("Episode is already active")
        if self._initial_states is None:
            self.capture_initial_state()
        self._active_episode = True

    def end_episode(self) -> None:
        self._active_episode = False

    def reset_objects(self, *, episode_boundary: bool) -> None:
        """Restore captured actor poses AND COM velocities only between episodes."""
        self._ensure_open()
        if episode_boundary is not True or self._active_episode:
            raise RuntimeError("Object state writes are permitted only at episode boundaries")
        if self._initial_states is None:
            raise RuntimeError("Capture initial object states before resetting")
        for path, state in self._initial_states.items():
            self._write_state(path, state["pose_w"], state["velocity_w"])

    def set_initialization_poses(self, poses_w: Mapping[str, Sequence[float]], *, episode_boundary: bool = True) -> None:
        """Initialize independent-task fixtures between episodes, with zero twist.

        Keys must be exact tracked dynamic-body prim paths. All input poses are
        validated before any actor is changed. Unspecified actors and the
        captured original reset state remain unchanged; this is initialization,
        never an in-episode grasp, object attachment, or transport mechanism.
        """
        self._ensure_open()
        if episode_boundary is not True or self._active_episode:
            raise RuntimeError("Object state writes are permitted only at episode boundaries")
        if self._initial_states is None:
            raise RuntimeError("Capture initial object states before task initialization")
        if not isinstance(poses_w, Mapping) or not poses_w:
            raise ValueError("Initialization requires a nonempty mapping of exact dynamic-body paths to poses")
        prepared = {}
        for path, pose in poses_w.items():
            if path not in self._views:
                raise ValueError(f"Initialization path is not an exact tracked dynamic-body prim: {path}")
            pose_to_matrix(pose)
            value = np.asarray(pose, dtype=float).copy()
            value[3:] /= np.linalg.norm(value[3:])
            prepared[path] = value
        for path, pose in prepared.items():
            self._write_state(path, pose, np.zeros(6))

    def _write_state(self, path: str, pose_w: Sequence[float], velocity_w: Sequence[float]) -> None:
        view = self._views[path]
        template = view.get_transforms()
        pose = np.asarray(pose_w)[[0, 1, 2, 4, 5, 6, 3]][None]
        velocity = np.asarray(velocity_w)[None]
        if hasattr(template, "device") and hasattr(template, "detach"):
            import torch
            pose_data = torch.as_tensor(pose, device=template.device, dtype=template.dtype)
            velocity_data = torch.as_tensor(velocity, device=template.device, dtype=template.dtype)
            indices = torch.zeros(1, device=template.device, dtype=torch.int32)
        else:
            pose_data = pose.astype(_numpy(template).dtype)
            velocity_data = velocity.astype(_numpy(view.get_velocities()).dtype)
            indices = np.zeros(1, dtype=np.uint32)
        view.set_transforms(pose_data, indices=indices)
        view.set_velocities(velocity_data, indices=indices)

    def contact_evidence(self, arm: str, object_name_or_path: str, *, detail_limit: int = 0) -> dict[str, Any]:
        """Read hand-object filtered forces; unavailable sensors never confirm contact."""
        if arm not in ("left", "right"):
            raise ValueError("arm must be left or right")
        if type(detail_limit) is not int or not 0 <= detail_limit <= 256:
            raise ValueError("Contact detail limit must be an integer in [0, 256]")
        path = self.resolve_path(object_name_or_path)
        key = (arm, path)
        evidence = {"arm": arm, "object_path": path, "contact_confirmed": False,
                    "available": False, "source": "physx_filtered_hand_object_contacts"}
        if key not in self._contact_views:
            evidence["reason"] = self._contact_errors.get(key, "Contact view not initialized")
            return evidence
        try:
            view = self._contact_views[key]
            matrix = _numpy(view.get_contact_force_matrix(dt=self.physics_dt))
            if matrix.shape != (view.sensor_count, view.filter_count, 3) or not np.isfinite(matrix).all():
                raise ValueError("Invalid PhysX contact-force matrix")
            # Sum magnitudes per hand body/filter, rather than cancelling
            # opposing finger forces into a misleading zero net hand force.
            magnitude = float(np.linalg.norm(matrix, axis=-1).sum())
            evidence.update({"available": True, "force_magnitude_sum_n": magnitude,
                             "contact_confirmed": magnitude > self.contact_force_threshold_n,
                             "force_threshold_n": self.contact_force_threshold_n,
                             "sensor_paths": list(view.sensor_paths),
                             "forces_w": matrix.tolist()})
            try:
                forces, points, normals, separations, counts, starts = view.get_contact_data(dt=self.physics_dt)
                counts = _numpy(counts)
                evidence["contact_point_count"] = int(counts.sum())
                if detail_limit:
                    forces, points, normals, separations, starts = map(
                        _numpy, (forces, points, normals, separations, starts))
                    forces, separations = forces.reshape(-1), separations.reshape(-1)
                    shape = (view.sensor_count, view.filter_count)
                    if (counts.shape != shape or starts.shape != shape
                            or points.shape != (len(forces), 3) or normals.shape != points.shape
                            or len(separations) != len(forces)):
                        raise ValueError("Malformed detailed contact buffers")
                    details = []
                    for sensor in range(view.sensor_count):
                        for filter_index in range(view.filter_count):
                            count, start = int(counts[sensor, filter_index]), int(starts[sensor, filter_index])
                            if count < 0 or (count and (start < 0 or start + count > len(forces))):
                                raise ValueError("Detailed contact range lies outside the PhysX buffer")
                            for index in range(start, start + min(count, detail_limit - len(details))):
                                values = np.r_[forces[index], points[index], normals[index], separations[index]]
                                if not np.isfinite(values).all():
                                    raise ValueError("Nonfinite detailed contact sample")
                                details.append({"sensor_path": view.sensor_paths[sensor],
                                    "object_path": path, "filter_index": filter_index,
                                    "point_w_m": points[index].tolist(), "normal_w": normals[index].tolist(),
                                    "normal_force_n": float(forces[index]),
                                    "separation_m": float(separations[index])})
                    evidence.update(contact_details=details, contact_details_available=True,
                                    contact_details_truncated=len(details) < evidence["contact_point_count"],
                                    contact_details_source="physx_contact_patch_buffer; object collider identity unavailable")
            except Exception as error:
                # Filtered forces remain direct contact evidence even if the
                # optional detailed point buffer is unsupported/unavailable.
                evidence.setdefault("contact_point_count", None)
                if detail_limit:
                    evidence.update(contact_details_available=False,
                                    contact_details_error=f"{type(error).__name__}: {error}")
        except Exception as error:
            evidence["reason"] = f"{type(error).__name__}: {error}"
        return evidence

    def support_evidence(self, object_name_or_path: str) -> dict[str, Any]:
        """Read signed force *on the object* from its exact static support.

        The dynamic body is the sensor; the static collider is the filter. Do
        not negate the returned world vector or use its magnitude as an upward
        load. Velocities and mass are direct PhysX COM/actor measurements, not
        finite differences of rendered or actor poses. Missing views or invalid
        values return unavailable evidence and never a fabricated zero force.
        """
        path = self.resolve_path(object_name_or_path)
        evidence = {"available": False, "object_path": path,
                    "support_path": self.support_pairs.get(path),
                    "source": "physx_filtered_object_support_contacts",
                    "force_frame": "world", "force_is_on": "dynamic_object",
                    "velocity_frame": "world_com_linear_angular"}
        if self._support_diagnostics is not None:
            evidence["contact_diagnostics"] = self._support_contact_diagnostic_evidence(path)
        if path not in self._support_views:
            evidence["reason"] = self._support_errors.get(path, "No support pair configured")
            return evidence
        try:
            view = self._support_views[path]
            force = _numpy(view.get_contact_force_matrix(dt=self.physics_dt))
            velocity = _numpy(self._views[path].get_velocities())
            mass = _numpy(self._views[path].get_masses()).reshape(-1)
            if (force.shape != (1, 1, 3) or velocity.shape != (1, 6) or mass.shape != (1,)
                    or not np.isfinite(force).all() or not np.isfinite(velocity).all()
                    or not np.isfinite(mass).all() or mass[0] <= 0):
                raise ValueError("Invalid measured support force, object velocity or mass")
            evidence.update(available=True, force_w=force[0, 0].astype(float).tolist(),
                            linear_velocity_w=velocity[0, :3].astype(float).tolist(),
                            angular_velocity_w=velocity[0, 3:].astype(float).tolist(),
                            object_mass_kg=float(mass[0]))
        except Exception as error:
            evidence["reason"] = f"{type(error).__name__}: {error}"
        return evidence

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("LiveScene views have been closed")

    def close(self) -> None:
        """Release native view references before env.close()/SimulationApp.close().

        The shared simulation view must not be invalidated here: the robot and
        environment may still own it. Only this helper's references are dropped.
        """
        self._active_episode = False
        if self._support_diagnostics is not None:
            try:
                self._support_diagnostics.close()
            except Exception as error:
                self._support_diagnostics_error = f"Diagnostic cleanup: {type(error).__name__}: {error}"
            self._support_diagnostics = None
        self._contact_views.clear()
        self._support_views.clear()
        self._views.clear()
        self.simulation_view = None
        self._closed = True


class GraspEvidenceTracker:
    """Measure stable hand-relative object lift without inventing attachment/contact."""

    def __init__(self, *, window_s=0.25, minimum_lift_m=0.03,
                 relative_position_tolerance_m=0.015, relative_rotation_tolerance_rad=0.15,
                 maximum_hand_object_distance_m=0.60, minimum_contact_fraction=0.6):
        values = [window_s, minimum_lift_m, relative_position_tolerance_m,
                  relative_rotation_tolerance_rad, maximum_hand_object_distance_m]
        if not np.isfinite(values).all() or any(value <= 0 for value in values):
            raise ValueError("Grasp evidence tolerances must be positive and finite")
        self.window_s, self.minimum_lift_m = float(window_s), float(minimum_lift_m)
        self.position_tolerance = float(relative_position_tolerance_m)
        self.rotation_tolerance = float(relative_rotation_tolerance_rad)
        self.maximum_distance = float(maximum_hand_object_distance_m)
        self.minimum_contact_fraction = float(minimum_contact_fraction)
        if not math.isfinite(self.minimum_contact_fraction) or not 0 < self.minimum_contact_fraction <= 1:
            raise ValueError("minimum_contact_fraction must be in (0, 1]")
        self.reset()

    def reset(self) -> None:
        self._history = deque()
        self._baseline_z = None

    def update(self, timestamp_s: float, hand_pose_w: Sequence[float], object_pose_w: Sequence[float],
               *, contact_evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp) or self._history and timestamp <= self._history[-1][0]:
            raise ValueError("Evidence timestamps must be finite and strictly increasing")
        hand = pose_to_matrix(hand_pose_w)
        obj = pose_to_matrix(object_pose_w)
        if self._baseline_z is None:
            self._baseline_z = float(obj[2, 3])
        relative = np.linalg.inv(hand) @ obj
        contact = bool(contact_evidence and contact_evidence.get("available") and contact_evidence.get("contact_confirmed"))
        self._history.append((timestamp, relative, contact))
        while len(self._history) > 2 and self._history[1][0] <= timestamp - self.window_s:
            self._history.popleft()
        reference = self._history[0][1]
        position_error = max(float(np.linalg.norm(item[1][:3, 3] - reference[:3, 3])) for item in self._history)
        rotation_error = max(float(np.arccos(np.clip((np.trace(reference[:3, :3].T @ item[1][:3, :3]) - 1) / 2, -1, 1))) for item in self._history)
        duration = timestamp - self._history[0][0]
        lift = float(obj[2, 3] - self._baseline_z)
        near = float(np.linalg.norm(relative[:3, 3])) <= self.maximum_distance
        stable = duration >= self.window_s - 1e-9 and position_error <= self.position_tolerance and rotation_error <= self.rotation_tolerance
        lifted = lift >= self.minimum_lift_m
        stable_lift = bool(near and stable and lifted)
        contact_fraction = sum(item[2] for item in self._history) / len(self._history)
        sustained_contact = contact and contact_fraction >= self.minimum_contact_fraction
        confirmed = bool(stable_lift and sustained_contact)
        return {
            "source": "measured_relative_pose_and_object_lift",
            "stable_lift_observed": stable_lift,
            "contact_confirmed": contact,
            "grasp_confirmed": confirmed,
            "contact_sample_fraction": contact_fraction,
            "sustained_contact": sustained_contact,
            "evidence_label": "contact_and_stable_lift" if confirmed else ("contact_observed_grasp_unconfirmed" if contact else "kinematic_evidence_only_contact_unconfirmed"),
            "lift_m": lift, "window_duration_s": duration,
            "relative_position_error_m": position_error, "relative_rotation_error_rad": rotation_error,
            "hand_object_distance_m": float(np.linalg.norm(relative[:3, 3])),
            "proximity_within_limit": near,
        }


def two_finger_pinch_evidence(contact: Mapping[str, Any], arm: str, joint_position_rad: float,
                              *, closed_position_rad: float = 0.0,
                              maximum_closed_angle_rad: float = 0.15) -> dict[str, Any]:
    """Qualify a thin-part pinch using both actual 260910 jaw bodies.

    At the configured 125 mm pad depth, the source CAD has about 11.3 mm
    aperture at 0.08 rad and 16.7 mm at 0.15 rad. The latter is a conservative
    angular guard for the CRS/8 mm diagonal JIG bar, not proof of a grasp.
    Each jaw must independently contact this same object. The task machine
    additionally checks contact dwell, object lift and relative-pose retention.
    Original any-hand contact evidence remains unchanged for release checks.
    """
    if arm not in ("left", "right"):
        raise ValueError("arm must be left or right")
    angle = float(joint_position_rad)
    if not np.isfinite([angle, closed_position_rad, maximum_closed_angle_rad]).all() or maximum_closed_angle_rad <= 0:
        raise ValueError("Gripper angles must be finite and the closed-angle bound positive")
    result = {"pinch_confirmed": False, "bilateral_contact_confirmed": False,
              "jaw_force_magnitudes_n": {}, "joint_position_rad": angle,
              "maximum_closed_angle_rad": maximum_closed_angle_rad,
              "angle_within_closed_bound": abs(angle - closed_position_rad) <= maximum_closed_angle_rad,
              "pinch_source": "same_object_two_jaw_filtered_forces_and_angle"}
    if not contact.get("available", False):
        return result
    paths = contact.get("sensor_paths", [])
    matrix = np.asarray(contact.get("forces_w", []), dtype=float)
    threshold = float(contact.get("force_threshold_n", 0.05))
    if (matrix.ndim != 3 or matrix.shape[0] != len(paths) or matrix.shape[2] != 3
            or matrix.shape[1] == 0 or not np.isfinite(matrix).all()
            or not math.isfinite(threshold) or threshold <= 0):
        return result
    body_names = {"fixed": f"{arm.title()}_Hand", "moving": f"{arm[0]}_1_dof_gripper_moving"}
    for jaw, name in body_names.items():
        matches = [i for i, path in enumerate(paths) if str(path).rsplit("/", 1)[-1] == name]
        if len(matches) != 1:
            return result
        result["jaw_force_magnitudes_n"][jaw] = float(np.linalg.norm(matrix[matches[0]], axis=-1).sum())
    result["bilateral_contact_confirmed"] = all(v > threshold for v in result["jaw_force_magnitudes_n"].values())
    result["pinch_confirmed"] = bool(result["bilateral_contact_confirmed"] and result["angle_within_closed_bound"])
    return result


def capture_live_snapshot(stage, robot, live_scene: LiveScene, output_path: str | Path, *,
                          scene_usd: str, scene_root="/World/Scene",
                          robot_root="/World/envs/env_0/Robot") -> dict[str, Any]:
    """Snapshot current paused physics without writing object poses to USD or PhysX."""
    states = live_scene.read_states()
    return capture_snapshot(
        stage, robot, output_path, scene_usd=scene_usd, scene_root=scene_root,
        robot_root=robot_root,
        rigid_body_poses_w={path: state["pose_w"] for path, state in states.items()},
    )
