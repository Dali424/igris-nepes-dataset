"""Read-only foot/ground normal forces and native contact identities.

These are diagnostics, never balance, support, friction-wrench or task gates.
Tensor normal forces exclude tangential friction. Native impulses retain their
original actor order; no sign conversion or COP inference is performed here.
"""
from collections.abc import Mapping
import math

import numpy as np

from .foot_observation import FOOT_BODY_NAMES
from .live_scene import _SupportContactDiagnostics, _numpy


GROUND_COLLIDER_PATHS = (
    "/World/Scene/WorkcellFloorCollider",
    "/World/Scene/Colliders/FloorOutsideWorkcellSupport",
)


def enable_foot_contact_reports(stage, *, robot_root="/World/envs/env_0/Robot"):
    """Enable only two reporters in the session layer before native parsing.

    Validate both feet before authoring anything. Geometry, sleep thresholds,
    material, mass, drives and body/joint state are untouched. Failure is an
    unavailable diagnostic, rather than a new task/control failure.
    """
    report = {"schema": "igris_c_foot_contact_reporting_v1", "available": False,
              "mode": "prestartup", "foot_paths": {},
              "expected_ground_collider_paths": list(GROUND_COLLIDER_PATHS),
              "report_threshold_n": 0., "simulation_state_written": False,
              "physics_parameters_changed": False, "asset_files_modified": False,
              "used_for_support_or_success": False}
    try:
        from pxr import PhysxSchema, Usd, UsdPhysics
        root = stage.GetPrimAtPath(robot_root)
        if not root.IsValid() or not root.IsActive():
            raise ValueError("Robot root is missing or inactive")
        feet = {}
        for side, name in FOOT_BODY_NAMES.items():
            matches = [prim for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                       if prim.IsActive() and prim.GetName() == name
                       and prim.HasAPI(UsdPhysics.RigidBodyAPI)]
            if len(matches) != 1:
                raise ValueError(f"Expected one exact rigid foot named {name}")
            prim = matches[0]
            body = UsdPhysics.RigidBodyAPI(prim)
            if body.GetRigidBodyEnabledAttr().Get() is False or body.GetKinematicEnabledAttr().Get() is True:
                raise ValueError(f"Foot is not enabled and dynamic: {prim.GetPath()}")
            feet[side] = prim
        session = stage.GetSessionLayer()
        report["previous_report_thresholds_n"] = {
            side: PhysxSchema.PhysxContactReportAPI(prim).GetThresholdAttr().Get()
            if prim.HasAPI(PhysxSchema.PhysxContactReportAPI) else None
            for side, prim in feet.items()}
        before = session.ExportToString()
        try:
            with Usd.EditContext(stage, session):
                for prim in feet.values():
                    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        except Exception:
            session.ImportFromString(before)
            raise
        report.update(available=True, edit_layer="session",
                      foot_paths={side: str(prim.GetPath()) for side, prim in feet.items()})
    except Exception as error:
        report["reason"] = f"{type(error).__name__}: {error}"
    return report


def prepare_foot_contact_reporting(env, env_ids, *, robot_root="/World/envs/env_0/Robot"):
    """IsaacLab prestartup event; reporting switch only, no state writes."""
    report = {"schema": "igris_c_foot_contact_reporting_v1", "available": False,
              "foot_paths": {}, "used_for_support_or_success": False}
    try:
        import carb.settings
        import omni.usd
        report = enable_foot_contact_reports(omni.usd.get_context().get_stage(), robot_root=robot_root)
        if report["available"]:
            settings = carb.settings.get_settings()
            report["disable_contact_processing_before"] = settings.get("/physics/disableContactProcessing")
            settings.set_bool("/physics/disableContactProcessing", False)
            report["disable_contact_processing_after"] = settings.get("/physics/disableContactProcessing")
    except Exception as error:
        report.update(available=False, reason=f"{type(error).__name__}: {error}")
    env.nepes_foot_contact_reporting_provenance = report


def _validate_pair_prims(stage, foot_paths, ground_paths):
    from pxr import PhysxSchema, UsdPhysics

    if not isinstance(foot_paths, Mapping) or set(foot_paths) != set(FOOT_BODY_NAMES):
        raise ValueError("Both exact foot actor paths are required")
    if tuple(ground_paths) != GROUND_COLLIDER_PATHS:
        raise ValueError("Expected the two exact active NEPES floor collider paths")
    for side, path in foot_paths.items():
        if not isinstance(path, str) or not path.startswith("/") or path.rsplit("/", 1)[-1] != FOOT_BODY_NAMES[side]:
            raise ValueError("Foot actor name/path mismatch")
        prim = stage.GetPrimAtPath(path)
        if (not prim.IsValid() or not prim.IsActive()
                or not prim.HasAPI(UsdPhysics.RigidBodyAPI)
                or not prim.HasAPI(PhysxSchema.PhysxContactReportAPI)):
            raise ValueError("Foot reporter missing; enable it before native actor creation")
        body = UsdPhysics.RigidBodyAPI(prim)
        if body.GetRigidBodyEnabledAttr().Get() is False or body.GetKinematicEnabledAttr().Get() is True:
            raise ValueError("Foot reporter is not dynamic and enabled")
    for path in ground_paths:
        prim = stage.GetPrimAtPath(path)
        if (not prim.IsValid() or not prim.IsActive() or not prim.HasAPI(UsdPhysics.CollisionAPI)
                or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False):
            raise ValueError(f"Ground collider missing or disabled: {path}")
        ancestor = prim
        while ancestor and not ancestor.IsPseudoRoot():
            if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError("Ground collider must be static")
            ancestor = ancestor.GetParent()


class FootContactObservation:
    """Four exact 1:1 filtered views plus current-step native foot reports."""

    def __init__(self, simulation_view, *, physics_dt, stage, foot_paths,
                 ground_paths=GROUND_COLLIDER_PATHS):
        self._views, self._errors = {}, {}
        self._diagnostics = None
        self._initialization_error = None
        self._last_measurement_time = None
        self._last_native_sequence = None
        self._closed = False
        self.foot_paths = dict(foot_paths) if isinstance(foot_paths, Mapping) else {}
        self.ground_paths = ()
        self.physics_dt = None
        try:
            self.ground_paths = tuple(ground_paths)
            value = float(physics_dt)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("physics_dt must be finite and positive")
            self.physics_dt = value
            _validate_pair_prims(stage, self.foot_paths, self.ground_paths)
            for side, foot_path in self.foot_paths.items():
                for ground_path in self.ground_paths:
                    key = (side, ground_path)
                    try:
                        view = simulation_view.create_rigid_contact_view(
                            [foot_path], filter_patterns=[[ground_path]], max_contact_data_count=256)
                        if (view.sensor_count != 1 or view.filter_count != 1
                                or list(view.sensor_paths) != [foot_path]
                                or np.asarray(view.filter_paths, dtype=object).reshape(-1).tolist() != [ground_path]):
                            raise ValueError("Foot/ground view resolved a different actor/filter")
                        self._views[key] = view
                    except Exception as error:
                        self._errors[key] = f"{type(error).__name__}: {error}"
            self._diagnostics = _SupportContactDiagnostics(self.foot_paths.values(), point_limit=256, pair_limit=64)
            self._diagnostics.start()
        except Exception as error:
            self._initialization_error = f"{type(error).__name__}: {error}"

    def read(self, *, measurement_time_s):
        """Sample without mutation; unavailable readings never become zero forces."""
        report = {"schema": "igris_c_foot_contact_observation_v1", "available": False,
                  "source": "physx_filtered_foot_ground_normal_forces_and_native_reports",
                  "measurement_time_s": None, "physics_dt_s": self.physics_dt,
                  "sampling": "last_completed_physics_step_not_control_interval_average",
                  "force_frame": "world", "force_units": "N",
                  "force_kind": "normal_only_excludes_tangential_friction",
                  "tensor_documented_force_is_on": "foot_sensor",
                  "native_impulse_sign_validated": False,
                  "total_contact_force_available": False, "cop_available": False,
                  "cop_unavailable_reason": "Contact impulse sign and pressure reconstruction are not native-qualified",
                  "support_verified": False, "balance_verified": False,
                  "used_for_support_or_success": False, "feet": {}}
        try:
            measurement_time_s = float(measurement_time_s)
            if not math.isfinite(measurement_time_s):
                raise ValueError("Nonfinite measurement time")
            report["measurement_time_s"] = measurement_time_s
            if self._closed:
                raise ValueError("Foot diagnostic is closed")
            if self._initialization_error:
                raise ValueError(self._initialization_error)
            sequences = []
            for side, foot_path in self.foot_paths.items():
                native = self._diagnostics.read(foot_path)
                for pair in native.get("pairs", []):
                    # Native static actor names can differ from USD collider
                    # paths. Attribute ground identity by exact collider only.
                    pair["expected_ground_colliders"] = [ground for ground in self.ground_paths
                        if ground in (pair["collider0"], pair["collider1"])]
                pairs = []
                for ground_path in self.ground_paths:
                    evidence = {"available": False, "sensor_path": foot_path,
                                "ground_collider_filter_path": ground_path}
                    view = self._views.get((side, ground_path))
                    try:
                        if view is None:
                            raise ValueError(self._errors.get((side, ground_path), "View unavailable"))
                        values = _numpy(view.get_contact_force_matrix(dt=self.physics_dt))
                        if values.shape != (1, 1, 3) or not np.isfinite(values).all():
                            raise ValueError("Invalid native filtered normal-force matrix")
                        evidence.update(available=True, normal_force_w_n=values[0, 0].astype(float).tolist())
                    except Exception as error:
                        evidence["reason"] = f"{type(error).__name__}: {error}"
                    pairs.append(evidence)
                foot = {"body_path": foot_path, "ground_pairs": pairs, "native_contact_report": native,
                        "available": bool(native.get("available") and all(pair["available"] for pair in pairs))}
                if all(pair["available"] for pair in pairs):
                    foot["summed_ground_normal_force_w_n"] = np.sum(
                        [pair["normal_force_w_n"] for pair in pairs], axis=0).tolist()
                report["feet"][side] = foot
                sequences.append(native.get("physics_step_sequence"))
            if len(set(sequences)) != 1 or not isinstance(sequences[0], int) or sequences[0] <= 0:
                raise ValueError("Native feet reports do not share a completed physics step")
            sequence = sequences[0]
            report["physics_step_sequence"] = sequence
            if self._last_measurement_time is not None:
                if measurement_time_s < self._last_measurement_time:
                    raise ValueError("Measurement clock moved backwards")
                if measurement_time_s > self._last_measurement_time and sequence <= self._last_native_sequence:
                    raise ValueError("Native foot contact report is stale")
            self._last_measurement_time, self._last_native_sequence = measurement_time_s, sequence
            report["available"] = all(foot["available"] for foot in report["feet"].values())
            if not report["available"]:
                report["reason"] = "One or more filtered views or native identity reports are unavailable"
        except Exception as error:
            report["reason"] = f"{type(error).__name__}: {error}"
        return report

    def close(self):
        """Unsubscribe only this diagnostic; never invalidate shared views."""
        try:
            if self._diagnostics is not None:
                self._diagnostics.close()
        finally:
            self._diagnostics = None
            self._views.clear()
            self._closed = True
