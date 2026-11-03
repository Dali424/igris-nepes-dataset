"""Physics-clock capture scheduling and lazy Isaac Sim RGB cameras.

250 Hz physics and 30 fps are incommensurate. The default scheduler captures at
ceil(k * 250 / 30) physics ticks, retains the actual timestamps, and bounds skew
below 4 ms. It never manufactures intermediate renders from 50 Hz observations.
For exact intervals, explicitly use a physics rate divisible by both 50 and 30
(for example 750 Hz); changing physics is a separate simulation configuration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from collections.abc import Mapping
import json
import importlib.util
import math
from numbers import Real
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Sequence

import numpy as np

from .snapshot import matrix_to_pose, pose_to_matrix


class StaleCameraFrameError(RuntimeError):
    """The renderer has not produced the requested physics-time frame yet."""


def render_current_frame(camera, sim_time_s, render, current_time, *, max_attempts=3):
    """Drain a bounded render delay without advancing physics or accepting stale RGB."""
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    before = float(current_time())
    if abs(before - sim_time_s) > 1e-5:
        raise RuntimeError("Capture target differs from current physics time")
    for attempt in range(max_attempts):
        render()
        if abs(float(current_time()) - before) > 1e-8:
            raise RuntimeError("Physics advanced during synchronous camera rendering")
        try:
            bundle = camera.read(sim_time_s)
        except StaleCameraFrameError:
            if attempt + 1 == max_attempts:
                raise
            continue
        for metadata in bundle.metadata.values():
            metadata["render_attempts_at_same_physics_time"] = attempt + 1
        return bundle

# UsdGeom.Camera optical lengths are in tenths of one stage unit. Isaac Sim
# 5.1 Camera getters/setters use full stage units and perform the factor of 10
# internally. These are not millimetres unless the USD stage unit is 1 cm.
_USD_OPTICAL_TENTHS_PER_STAGE_UNIT = 10.0


def _set_camera_optics(camera: Any, spec: "CameraSpec") -> None:
    camera.set_focal_length(spec.focal_length_usd_tenths / _USD_OPTICAL_TENTHS_PER_STAGE_UNIT)
    if spec.calibrated_intrinsics is None:
        camera.set_horizontal_aperture(spec.horizontal_aperture_usd_tenths / _USD_OPTICAL_TENTHS_PER_STAGE_UNIT)
        return
    K = np.asarray(spec.calibrated_intrinsics)
    camera.set_horizontal_aperture(
        spec.horizontal_aperture_usd_tenths / _USD_OPTICAL_TENTHS_PER_STAGE_UNIT,
        maintain_square_pixels=False)
    camera.set_vertical_aperture(
        spec.height * spec.focal_length_usd_tenths / K[1, 1] / _USD_OPTICAL_TENTHS_PER_STAGE_UNIT,
        maintain_square_pixels=False)
    # Isaac Sim 5.1's native OpenCV schema preserves unequal fx/fy, principal
    # point and distortion. Its regular get_intrinsics_matrix ignores those.
    camera.set_opencv_pinhole_properties(
        cx=float(K[0, 2]), cy=float(K[1, 2]), fx=float(K[0, 0]), fy=float(K[1, 1]),
        pinhole=[*spec.distortion_coefficients, *([0.] * 7)])


def _camera_optics_metadata(camera: Any) -> dict[str, Any]:
    """Read actual SDK values, including its square-pixel aperture adjustment."""
    return {
        "unit": "usd_tenths_of_stage_unit",
        "focal_length": float(camera.get_focal_length()) * _USD_OPTICAL_TENTHS_PER_STAGE_UNIT,
        "horizontal_aperture": float(camera.get_horizontal_aperture()) * _USD_OPTICAL_TENTHS_PER_STAGE_UNIT,
        "vertical_aperture": float(camera.get_vertical_aperture()) * _USD_OPTICAL_TENTHS_PER_STAGE_UNIT,
        "source": "isaac_camera_sdk_readback",
    }


def _camera_intrinsics_metadata(camera: Any, spec: "CameraSpec") -> tuple[np.ndarray, dict[str, Any]]:
    if spec.calibrated_intrinsics is None:
        return _cpu_array(camera.get_intrinsics_matrix(device="cpu")), {}
    model = camera.get_lens_distortion_model()
    if model != "opencvPinhole":
        raise RuntimeError(f"Camera {spec.name} lost its calibrated OpenCV pinhole model: {model}")
    if tuple(camera.get_resolution()) != (spec.width, spec.height):
        raise RuntimeError(f"Camera {spec.name} calibrated image resolution changed")
    cx, cy, fx, fy, coefficients = camera.get_opencv_pinhole_properties()
    K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]], dtype=float)
    distortion = np.asarray(coefficients, dtype=float)
    expected_distortion = np.r_[spec.distortion_coefficients, np.zeros(7)]
    if (not np.isfinite(K).all() or not np.allclose(K, spec.calibrated_intrinsics, atol=2e-4, rtol=0)
            or distortion.shape != (12,) or not np.isfinite(distortion).all()
            or not np.allclose(distortion, expected_distortion, atol=1e-7, rtol=0)):
        raise RuntimeError(f"Camera {spec.name} OpenCV calibration readback differs from its configured K/distortion")
    return K, {
        "lens_distortion_model": model,
        "opencv_distortion_coefficients": distortion.tolist(),
        "intrinsics_source": "isaac_opencv_pinhole_sdk_readback",
        "calibration_source": json.loads(json.dumps(spec.calibration_source, allow_nan=False)),
    }


def prepare_calibrated_camera_plugins(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Register native lens schemas before SimulationApp creates a USD stage.

    USD caches schema definitions on its first registry use. Registering this
    plugin only when cameras are initialized is too late after an environment
    has loaded a stage. Discovery here imports neither Isaac nor the app.
    """
    if config.get("head_camera_calibration") is None:
        return ()
    _validated_head_calibration(config["head_camera_calibration"])
    installation = importlib.util.find_spec("isaacsim")
    roots = () if installation is None else installation.submodule_search_locations or ()
    paths = []
    for root in roots:
        for folder in ("extscache", "exts"):
            for info in sorted((Path(root) / folder).glob("omni.usd.schema.omni_lens_distortion*/usd_plugins/plugInfo.json")):
                # USD plugInfo permits whole-line '#' comments before JSON.
                document = json.loads("\n".join(line for line in info.read_text().splitlines()
                                                if not line.lstrip().startswith("#")))
                types = [value for plugin in document.get("Plugins", [])
                         for value in plugin.get("Info", {}).get("Types", {}).values()]
                if any(value.get("alias", {}).get("UsdSchemaBase") == "OmniLensDistortionOpenCvPinholeAPI"
                       for value in types):
                    path = str(info.parent.resolve())
                    if path not in paths:
                        paths.append(path)
    if not paths:
        raise RuntimeError("Cannot locate the installed Isaac Sim OpenCV pinhole USD plugin before app startup")
    existing = [p for p in os.environ.get("PXR_PLUGINPATH_NAME", "").split(os.pathsep) if p]
    os.environ["PXR_PLUGINPATH_NAME"] = os.pathsep.join(paths + [p for p in existing if p not in paths])
    # PXR may already be importable in the launcher's Python environment. Its
    # plugin registry is safe to populate now; do not touch Usd.SchemaRegistry.
    try:
        from pxr import Plug
    except ImportError:
        pass  # The environment path is read when SimulationApp loads USD.
    else:
        for path in paths:
            Plug.Registry().RegisterPlugins(path)
    return tuple(paths)


def _ensure_opencv_pinhole_schema() -> None:
    """Require the native schema preloaded by prepare_calibrated_camera_plugins."""
    import omni.usd.schema.omni_lens_distortion as lens_distortion_schema
    from pxr import Plug, Usd
    # Keep this fallback for applications that have imported USD without ever
    # constructing its schema registry; normal collection preloads at startup.
    plugin_dirs = [parent / "usd_plugins" for parent in Path(lens_distortion_schema.__file__).resolve().parents
                   if (parent / "usd_plugins/plugInfo.json").is_file()]
    for path in plugin_dirs:
        Plug.Registry().RegisterPlugins(str(path))
    if Usd.SchemaRegistry().FindAppliedAPIPrimDefinition("OmniLensDistortionOpenCvPinholeAPI") is None:
        raise RuntimeError("Isaac Sim OpenCV pinhole lens-distortion schema is not registered; "
                           "prepare_calibrated_camera_plugins must run before SimulationApp or the first USD stage")


def _cpu_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


@dataclass(frozen=True)
class CaptureEvent:
    frame_index: int
    physics_step: int
    nominal_time_s: float
    actual_time_s: float
    skew_s: float


class CaptureScheduler:
    """Call poll after EVERY physics tick, including the chosen starting tick."""
    def __init__(self, physics_hz: int = 250, fps: int = 30, start_step: int = 0, *, exact: bool = False):
        if not isinstance(physics_hz, int) or not isinstance(fps, int) or not 0 < fps <= physics_hz:
            raise ValueError("Require integer 0 < fps <= physics_hz")
        if exact and physics_hz % fps:
            raise ValueError("Exact capture requires physics_hz divisible by fps; 250/30 is not exact")
        self.physics_hz, self.fps, self.exact = physics_hz, fps, exact
        self.reset(start_step)

    def reset(self, start_step: int = 0) -> None:
        if not isinstance(start_step, int) or start_step < 0:
            raise ValueError("start_step must be a nonnegative integer")
        self.start_step, self.frame_index, self._last_step = start_step, 0, start_step

    @property
    def next_step(self) -> int:
        return self.start_step + (self.frame_index * self.physics_hz + self.fps - 1) // self.fps

    def poll(self, physics_step: int) -> CaptureEvent | None:
        if not isinstance(physics_step, int) or physics_step < self._last_step:
            raise ValueError("physics_step must be a monotonically increasing integer")
        self._last_step = physics_step
        if physics_step < self.next_step:
            return None
        if physics_step > self.next_step:
            raise RuntimeError(f"Missed capture at physics tick {self.next_step}; got {physics_step}. "
                               "Invoke the scheduler at physics rate, not only at 50 Hz control updates.")
        nominal = self.frame_index / self.fps
        actual = (physics_step - self.start_step) / self.physics_hz
        event = CaptureEvent(self.frame_index, physics_step, nominal, actual, actual - nominal)
        self.frame_index += 1
        return event

    @property
    def metadata(self) -> dict[str, Any]:
        return {"physics_hz": self.physics_hz, "fps": self.fps,
                "mode": "exact_physics_tick" if self.physics_hz % self.fps == 0 else "ceiling_physics_tick",
                "max_skew_exclusive_s": 1 / self.physics_hz,
                "timestamp_contract": "video timestamp k/fps; actual physics and render timestamps retained"}


@dataclass(frozen=True)
class CameraSpec:
    name: str
    prim_path: str
    width: int
    height: int
    translation: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    world_pose: bool = False
    focal_length_usd_tenths: float = 18.0
    horizontal_aperture_usd_tenths: float = 36.0
    attachment_prim_path: str | None = None
    near_clip_m: float = 0.01
    far_clip_m: float = 100.0
    source_camera_prim_path: str | None = None
    source_aperture_usd_tenths: tuple[float, float] | None = None
    intrinsics_policy: str = "configured_square_pixels"
    pose_policy: str = "configured_default"
    # Original local mount pose after conversion to camera_axes='world'.
    # These values remain unchanged by configured head/wrist overrides.
    authored_mount_pose_wxyz: tuple[float, ...] | None = None
    authored_focal_length_usd_tenths: float | None = None
    configured_mount_override_fields: tuple[str, ...] = ()
    calibrated_intrinsics: tuple[tuple[float, float, float], ...] | None = None
    distortion_coefficients: tuple[float, ...] | None = None
    calibration_source: dict[str, Any] | None = None

    def __post_init__(self):
        if not self.name or "/" in self.name or not self.prim_path.startswith("/"):
            raise ValueError("Camera needs a name and absolute USD prim_path")
        if min(self.width, self.height) <= 0 or self.width % 2 or self.height % 2:
            raise ValueError("Camera dimensions must be positive even numbers for MP4 encoding")
        pose = np.asarray((*self.translation, *self.orientation_wxyz), dtype=float)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError("Camera pose must contain finite xyz and wxyz values")
        if abs(np.linalg.norm(self.orientation_wxyz) - 1) > 1e-5:
            raise ValueError("Camera orientation must be a unit wxyz quaternion")
        if (not np.isfinite([self.focal_length_usd_tenths, self.horizontal_aperture_usd_tenths]).all()
                or min(self.focal_length_usd_tenths, self.horizontal_aperture_usd_tenths) <= 0):
            raise ValueError("Camera intrinsics must be positive")
        if not (math.isfinite(self.near_clip_m) and math.isfinite(self.far_clip_m)
                and 0 < self.near_clip_m < self.far_clip_m):
            raise ValueError("Camera clipping must satisfy 0 < near < far in meters")
        if self.attachment_prim_path is not None:
            if not self.attachment_prim_path.startswith("/") or not self.world_pose:
                raise ValueError("Measured camera attachments need a full link path and world_pose=True")
            if self.prim_path.startswith(self.attachment_prim_path.rstrip("/") + "/"):
                raise ValueError("A measured camera must be outside its moving attachment hierarchy")
        if self.pose_policy not in ("configured_default", "authored_camera_mount", "configured_mount_override",
                                   "calibrated_camera_mount"):
            raise ValueError("Unsupported camera pose_policy")
        if self.authored_mount_pose_wxyz is not None:
            authored = np.asarray(self.authored_mount_pose_wxyz, dtype=float)
            if (authored.shape != (7,) or not np.isfinite(authored).all()
                    or abs(np.linalg.norm(authored[3:]) - 1) > 1e-5):
                raise ValueError("Authored camera mount pose must contain finite xyz and a unit wxyz quaternion")
        if self.authored_focal_length_usd_tenths is not None:
            focal = self.authored_focal_length_usd_tenths
            if isinstance(focal, (bool, np.bool_)) or not isinstance(focal, Real) or not math.isfinite(focal) or focal <= 0:
                raise ValueError("Authored camera focal length must be positive and finite")
        if (len(set(self.configured_mount_override_fields)) != len(self.configured_mount_override_fields)
                or any(field not in ("translation", "orientation_wxyz", "focal_length_usd_tenths")
                       for field in self.configured_mount_override_fields)):
            raise ValueError("Unsupported or repeated configured mount override fields")
        calibration_fields = (self.calibrated_intrinsics, self.distortion_coefficients, self.calibration_source)
        if any(field is not None for field in calibration_fields):
            if (any(field is None for field in calibration_fields) or self.name != "head"
                    or self.pose_policy != "calibrated_camera_mount"
                    or self.intrinsics_policy != "calibrated_opencv_pinhole"):
                raise ValueError("Calibrated head cameras require complete K/distortion/source and calibrated policies")
            _validated_head_calibration({
                "attachment_body": (self.attachment_prim_path or "").rsplit("/", 1)[-1],
                "translation": self.translation, "orientation_wxyz": self.orientation_wxyz,
                "intrinsics": self.calibrated_intrinsics, "distortion": self.distortion_coefficients,
                "clipping_range_m": [self.near_clip_m, self.far_clip_m], "source": self.calibration_source,
            }, width=self.width, height=self.height)
        elif self.pose_policy == "calibrated_camera_mount":
            raise ValueError("Calibrated camera pose policy requires calibration fields")


def default_camera_specs(robot_prim_path: str = "/World/envs/env_0/Robot", *,
                         hand_arm: str = "right") -> tuple[CameraSpec, ...]:
    """Starting placements; review the rendered workcell views before collection.

    Quaternions use Isaac's camera_axes='world': +X forward and +Z up.
    Attachment paths are validated during initialize. Cameras live outside the
    articulation hierarchy; update_attachment_poses explicitly follows measured
    physics link poses before each render, even when USD link poses are stale.
    """
    if hand_arm not in ("left", "right", "both"):
        raise ValueError("hand_arm must be left, right or both")
    robot = robot_prim_path.rstrip("/")
    head = CameraSpec("head", "/World/NEPES_Capture/head", 1280, 720,
                   (0.10, 0.0, 0.03), (math.cos(math.pi / 24), 0.0, math.sin(math.pi / 24), 0.0),
                   True, attachment_prim_path=f"{robot}/Link_Neck_Pitch")
    arms = ("left", "right") if hand_arm == "both" else (hand_arm,)
    hands = tuple(CameraSpec(
                   f"wrist_{arm}" if hand_arm == "both" else "hand",
                   f"/World/NEPES_Capture/wrist_{arm}" if hand_arm == "both" else "/World/NEPES_Capture/hand",
                   640, 480,
                   (0.035, 0.0, 0.025), (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0),
                   True, attachment_prim_path=f"{robot}/Link_Wrist_Pitch_{arm.title()}")
                  for arm in arms)
    third_person = CameraSpec("third_person", "/World/NEPES_Capture/third_person", 1280, 720,
                   # Scene.usda's authored OperatorView is inside the scanned
                   # room. Its USD camera rotation is converted to +X-forward
                   # camera_axes='world'; the former distant guess hit a wall.
                   (-0.509923645832592, -1.1953450548195501, 1.807851752208838),
                   (0.561458867506176, -0.34010740567447, 0.27127653774053384, 0.7039175610649109),
                   True, focal_length_usd_tenths=8.0, horizontal_aperture_usd_tenths=20.955)
    return (head, *hands, third_person)


def latest_camera_specs(camera_attachments: Mapping[str, Mapping[str, Any]],
                        robot_prim_path: str = "/World/envs/env_0/Robot", *,
                        hand_arm: str = "right", head_camera: str = "LeftCamera") -> tuple[CameraSpec, ...]:
    """Use the actual 260910 mounted cameras and retain the scene OperatorView.

    The authored cameras look along local -Z with +Y up. Isaac's public world
    convention looks along +X with +Z up. Their poses are explicitly converted,
    then followed using the measured camera-mount rigid actors. Source apertures
    are centre-cropped to the requested image aspect ratio with square pixels;
    the original apertures and source prim path remain in capture configuration.
    Authored clipping is retained: the head's opaque visual shell sits about
    19 mm ahead of its camera mounts, and its source 50 mm near plane clears it.
    """
    if hand_arm not in ("left", "right", "both"):
        raise ValueError("hand_arm must be left, right or both")
    if head_camera not in ("LeftCamera", "RightCamera"):
        raise ValueError("head_camera must be the authored LeftCamera or RightCamera")
    robot = robot_prim_path.rstrip("/")
    usd_from_world_axes = np.eye(4)
    # Same U_W_TRANSFORM used by Isaac Sim 5.1 Camera.get_world_pose('world').
    usd_from_world_axes[:3, :3] = [[0, -1, 0], [0, 0, 1], [-1, 0, 0]]

    def mounted(name, source_name, width, height):
        if source_name not in camera_attachments:
            raise ValueError(f"Authored camera attachment missing: {source_name}")
        source = camera_attachments[source_name]
        if source.get("pose_convention") != "usd_camera_minus_z_forward_plus_y_up":
            raise ValueError(f"Unsupported source camera convention: {source_name}")
        source_pose = pose_to_matrix(source["relative_pose_wxyz"])
        pose = matrix_to_pose(source_pose @ usd_from_world_axes)
        # Adapter <=v5's *_mm keys contain raw UsdGeom attribute numbers despite
        # their historical suffix. Read old artifacts without interpreting them
        # as millimetres; v6 metadata names the USD units explicitly.
        if source.get("optics_units", "usd_tenths_of_stage_unit") != "usd_tenths_of_stage_unit":
            raise ValueError(f"Unsupported authored camera optics units: {source_name}")
        def optical_value(name):
            value = source.get(name + "_usd_tenths", source.get(name + "_mm"))
            if value is None:
                raise ValueError(f"Authored camera optical value missing: {source_name}/{name}")
            return float(value)
        horizontal, vertical = optical_value("horizontal_aperture"), optical_value("vertical_aperture")
        focal = optical_value("focal_length")
        if not np.isfinite([horizontal, vertical, focal]).all() or min(horizontal, vertical, focal) <= 0:
            raise ValueError(f"Invalid authored camera intrinsics: {source_name}")
        cropped_horizontal = min(horizontal, vertical * width / height)
        clipping = np.asarray(source.get("clipping_range_m", []), dtype=float)
        if clipping.shape != (2,) or not np.isfinite(clipping).all() or not 0 < clipping[0] < clipping[1]:
            raise ValueError(f"Invalid or missing authored camera clipping range: {source_name}")
        return CameraSpec(name, f"/World/NEPES_Capture/{name}", width, height,
            tuple(pose[:3]), tuple(pose[3:]), True, focal_length_usd_tenths=focal,
            horizontal_aperture_usd_tenths=cropped_horizontal,
            near_clip_m=float(clipping[0]), far_clip_m=float(clipping[1]),
            attachment_prim_path=robot + "/" + source["body_name"],
            source_camera_prim_path=robot + "/" + source["relative_prim_path"],
            source_aperture_usd_tenths=(horizontal, vertical),
            intrinsics_policy="authored_focal_length_center_crop_square_pixels",
            pose_policy="authored_camera_mount",
            authored_mount_pose_wxyz=tuple(float(v) for v in pose),
            authored_focal_length_usd_tenths=focal)

    arms = ("left", "right") if hand_arm == "both" else (hand_arm,)
    hands = tuple(mounted(f"wrist_{arm}" if hand_arm == "both" else "hand",
                          arm.title() + "WristCamera", 640, 480) for arm in arms)
    return (mounted("head", head_camera, 1280, 720), *hands,
            default_camera_specs(robot, hand_arm=hand_arm)[-1])


def _validated_pose_lens_changes(values: Mapping[str, Any], label: str) -> dict[str, Any]:
    """Validate explicit pose/lens replacements without importing Isaac."""
    changes = {}
    for name, size in (("translation", 3), ("orientation_wxyz", 4)):
        if name not in values:
            continue
        value = values[name]
        try:
            shape = np.shape(value)
        except (TypeError, ValueError):
            shape = None
        if (not isinstance(value, (list, tuple, np.ndarray))
                or shape != (size,)
                or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
                       or not math.isfinite(v) for v in value)):
            raise ValueError(f"{label}{name} requires {size} finite real numbers")
        changes[name] = tuple(float(v) for v in value)
        if name == "orientation_wxyz" and abs(np.linalg.norm(changes[name]) - 1) > 1e-5:
            raise ValueError(f"{label}{name} must be a unit wxyz quaternion")
    if "focal_length_usd_tenths" in values:
        focal_length_usd_tenths = values["focal_length_usd_tenths"]
        if (isinstance(focal_length_usd_tenths, (bool, np.bool_))
                or not isinstance(focal_length_usd_tenths, Real)
                or not math.isfinite(focal_length_usd_tenths) or focal_length_usd_tenths <= 0):
            raise ValueError(f"{label}focal_length_usd_tenths must be a positive finite real number")
        changes["focal_length_usd_tenths"] = float(focal_length_usd_tenths)
    return changes


def with_third_person_pose(specs: Sequence[CameraSpec], *, translation=None,
                           orientation_wxyz=None, focal_length_usd_tenths=None) -> tuple[CameraSpec, ...]:
    """Override the external world pose/lens; retain mounted-camera calibration.

    Position uses stage units; quaternion uses Isaac's +X-forward/+Z-up world
    camera convention. Focal length uses USD tenths of one stage unit, matching
    CameraSpec and the existing SDK conversion, not an implicit millimetre unit.
    """
    changes = _validated_pose_lens_changes(
        {name: value for name, value in (
            ("translation", translation), ("orientation_wxyz", orientation_wxyz),
            ("focal_length_usd_tenths", focal_length_usd_tenths)) if value is not None},
        "third_person_")
    if not changes:
        return tuple(specs)
    if sum(spec.name == "third_person" for spec in specs) != 1:
        raise ValueError("Exactly one external third_person camera is required")
    updated = []
    for spec in specs:
        if spec.name == "third_person":
            if not spec.world_pose or spec.attachment_prim_path is not None:
                raise ValueError("Cannot re-aim a mounted camera as the external view")
            spec = replace(spec, **changes)
        updated.append(spec)
    return tuple(updated)


def with_third_person_orientation(specs: Sequence[CameraSpec], orientation_wxyz=None) -> tuple[CameraSpec, ...]:
    """Backward-compatible orientation-only external-camera override."""
    return with_third_person_pose(specs, orientation_wxyz=orientation_wxyz)


def _validated_wrist_overrides(overrides, *, legacy: bool = False) -> dict[str, dict[str, Any]]:
    if overrides is None:
        return {}
    if not isinstance(overrides, Mapping) or any(arm not in ("left", "right") for arm in overrides):
        raise ValueError("wrist_camera_overrides must be a mapping with only left/right keys")
    validated = {}
    allowed = {"translation", "orientation_wxyz", "focal_length_usd_tenths"}
    for arm, values in overrides.items():
        if not isinstance(values, Mapping) or not values or any(field not in allowed for field in values):
            raise ValueError(f"wrist_camera_overrides.{arm} requires a nonempty mapping of translation, orientation_wxyz or focal_length_usd_tenths")
        validated[arm] = _validated_pose_lens_changes(values, f"wrist_camera_overrides.{arm}.")
    focal_field = "focal_length_usd_tenths"
    if legacy and any(focal_field in values for values in validated.values()):
        if (any(focal_field not in validated.get(arm, {}) for arm in ("left", "right"))
                or validated["left"][focal_field] != validated["right"][focal_field]):
            raise ValueError("head_hand wrist focal overrides require identical explicit left and right focal_length_usd_tenths; legacy switching shares one lens")
    return validated


def _validated_head_override(override) -> dict[str, Any]:
    if override is None:
        return {}
    allowed = {"translation", "orientation_wxyz", "focal_length_usd_tenths"}
    if not isinstance(override, Mapping) or any(field not in allowed for field in override):
        raise ValueError("head_camera_override must be a mapping of translation, orientation_wxyz or focal_length_usd_tenths")
    return _validated_pose_lens_changes(override, "head_camera_override.")


def _validated_head_calibration(calibration, *, width: int = 1280, height: int = 720) -> dict[str, Any] | None:
    if calibration is None:
        return None
    label = "head_camera_calibration"
    fields = {"attachment_body", "translation", "orientation_wxyz", "intrinsics",
              "distortion", "clipping_range_m", "source"}
    if not isinstance(calibration, Mapping) or set(calibration) != fields:
        raise ValueError(f"{label} requires exactly {sorted(fields)}")
    if calibration["attachment_body"] != "Link_Neck_Pitch":
        raise ValueError(f"{label}.attachment_body must be Link_Neck_Pitch")
    pose = _validated_pose_lens_changes(calibration, label + ".")

    def real_array(value, shape, name):
        try:
            arr = np.asarray(value, dtype=object)
        except (TypeError, ValueError):
            arr = np.empty(0, dtype=object)
        if (not isinstance(value, (list, tuple, np.ndarray)) or arr.shape != shape
                or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
                       or not math.isfinite(v) for v in arr.flat)):
            raise ValueError(f"{label}.{name} requires finite real values with shape {shape}")
        return np.asarray(arr, dtype=float)

    K = real_array(calibration["intrinsics"], (3, 3), "intrinsics")
    if (not np.array_equal(K[2], [0, 0, 1]) or K[0, 1] != 0 or K[1, 0] != 0
            or K[0, 0] <= 0 or K[1, 1] <= 0
            or not (0 <= K[0, 2] < width and 0 <= K[1, 2] < height)):
        raise ValueError(f"{label}.intrinsics needs positive fx/fy, zero skew, [0,0,1] last row and an in-image principal point")
    distortion = real_array(calibration["distortion"], (5,), "distortion")
    clip = real_array(calibration["clipping_range_m"], (2,), "clipping_range_m")
    if not 0 < clip[0] < clip[1]:
        raise ValueError(f"{label}.clipping_range_m must satisfy 0 < near < far")
    source = calibration["source"]
    required_source = {"path", "sha256", "input_resolution", "resize_crop"}
    if not isinstance(source, Mapping) or not required_source <= set(source):
        raise ValueError(f"{label}.source requires path, sha256, input_resolution and resize_crop")
    if (not isinstance(source["path"], str) or not source["path"]
            or not isinstance(source["sha256"], str) or len(source["sha256"]) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in source["sha256"])):
        raise ValueError(f"{label}.source requires a nonempty source path and hexadecimal SHA256")
    resolution = source["input_resolution"]
    if (not isinstance(resolution, (list, tuple)) or len(resolution) != 2
            or any(type(v) is not int or v <= 0 for v in resolution)):
        raise ValueError(f"{label}.source.input_resolution requires positive integer width,height")
    resize = source["resize_crop"]
    if not isinstance(resize, Mapping) or set(resize) != {"scale", "crop_xywh"}:
        raise ValueError(f"{label}.source.resize_crop requires scale and crop_xywh")
    scale = real_array(resize["scale"], (2,), "source.resize_crop.scale")
    crop = resize["crop_xywh"]
    if (np.any(scale <= 0) or not isinstance(crop, (list, tuple)) or len(crop) != 4
            or any(type(v) is not int for v in crop) or min(crop[:2]) < 0
            or tuple(crop[2:]) != (width, height)
            or crop[0] + width > resolution[0] * scale[0] + 1e-8
            or crop[1] + height > resolution[1] * scale[1] + 1e-8):
        raise ValueError(f"{label}.source.resize_crop must fit its scaled input and match camera width,height")
    try:
        source_copy = json.loads(json.dumps(dict(source), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}.source must be finite JSON-serializable provenance") from exc
    return {
        **pose, "attachment_body": calibration["attachment_body"],
        "intrinsics": tuple(tuple(float(v) for v in row) for row in K),
        "distortion": tuple(float(v) for v in distortion),
        "clipping_range_m": tuple(float(v) for v in clip), "source": source_copy,
    }


def _with_mounted_camera_changes(spec: CameraSpec, changes: Mapping[str, Any]) -> CameraSpec:
    """Retain original calibration when replacing a mounted sensor's settings."""
    if spec.calibrated_intrinsics is not None:
        raise ValueError("A calibrated camera cannot also receive a pose/lens override")
    provenance = {
        "configured_mount_override_fields": tuple(sorted(set(spec.configured_mount_override_fields) | set(changes))),
    }
    if "translation" in changes or "orientation_wxyz" in changes:
        provenance["pose_policy"] = "configured_mount_override"
    if "focal_length_usd_tenths" in changes:
        provenance["intrinsics_policy"] = "configured_focal_length_retained_aperture"
    return replace(spec, **changes, **provenance)


def with_head_camera_override(specs: Sequence[CameraSpec], override) -> tuple[CameraSpec, ...]:
    """Replace the head camera's fixed local pose/lens on its measured mount.

    Pose values are absolute replacements relative to the existing attachment
    body, using camera +X forward/+Z up. They do not follow a task object or
    alter the articulation. Authored calibration, aperture and clipping remain
    available unchanged alongside the configured camera settings.
    """
    specs = tuple(specs)
    changes = _validated_head_override(override)
    if not changes:
        return specs
    matches = [i for i, spec in enumerate(specs) if spec.name == "head"]
    if len(matches) != 1:
        raise ValueError("Head override requires exactly one measured head camera")
    index = matches[0]
    spec = specs[index]
    mount = (spec.attachment_prim_path or "").rsplit("/", 1)[-1]
    source_name = {"l_camera_link": "LeftCamera", "r_camera_link": "RightCamera"}.get(mount)
    source_matches = (spec.source_camera_prim_path is None
                      or (source_name is not None and spec.source_camera_prim_path
                          == f"{spec.attachment_prim_path}/{source_name}"))
    if (not spec.world_pose or mount not in ("l_camera_link", "r_camera_link", "Link_Neck_Pitch")
            or not source_matches):
        raise ValueError("Head override requires its measured head attachment, not a different or unmeasured camera")
    updated = list(specs)
    updated[index] = _with_mounted_camera_changes(spec, changes)
    return tuple(updated)


def with_head_camera_calibration(specs: Sequence[CameraSpec], calibration) -> tuple[CameraSpec, ...]:
    """Replace the eye camera with an explicitly calibrated neck-mounted sensor.

    Source pose uses Isaac +X-forward/+Z-up relative to Link_Neck_Pitch. K is
    expressed in the output image's pixels and distortion uses OpenCV ordering
    k1,k2,p1,p2,k3. The original eye calibration is cleared because it describes
    a different optical sensor; calibration_source identifies the replacement.
    """
    specs = tuple(specs)
    if calibration is None:
        return specs
    matches = [i for i, spec in enumerate(specs) if spec.name == "head"]
    if len(matches) != 1:
        raise ValueError("Head calibration requires exactly one measured head camera")
    index = matches[0]
    spec = specs[index]
    parsed = _validated_head_calibration(calibration, width=spec.width, height=spec.height)
    if (not spec.world_pose or not spec.attachment_prim_path
            or spec.attachment_prim_path.rsplit("/", 1)[-1]
            not in ("l_camera_link", "r_camera_link", "Link_Neck_Pitch")
            or spec.configured_mount_override_fields):
        raise ValueError("Head calibration requires a measured head mount without a simultaneous head override")
    robot_path = spec.attachment_prim_path.rsplit("/", 1)[0]
    focal = 24.0  # Same gauge as the source calibrated_pinhole_camera_cfg.
    updated = list(specs)
    updated[index] = replace(
        spec, translation=parsed["translation"], orientation_wxyz=parsed["orientation_wxyz"],
        attachment_prim_path=f"{robot_path}/{parsed['attachment_body']}",
        focal_length_usd_tenths=focal,
        horizontal_aperture_usd_tenths=spec.width * focal / parsed["intrinsics"][0][0],
        near_clip_m=parsed["clipping_range_m"][0], far_clip_m=parsed["clipping_range_m"][1],
        pose_policy="calibrated_camera_mount", intrinsics_policy="calibrated_opencv_pinhole",
        source_camera_prim_path=None, source_aperture_usd_tenths=None,
        authored_mount_pose_wxyz=None, authored_focal_length_usd_tenths=None,
        configured_mount_override_fields=(), calibrated_intrinsics=parsed["intrinsics"],
        distortion_coefficients=parsed["distortion"], calibration_source=parsed["source"])
    return tuple(updated)


def with_wrist_camera_overrides(specs: Sequence[CameraSpec], overrides, *,
                                hand_arm: str | None = None) -> tuple[CameraSpec, ...]:
    """Replace each wrist's local pose/lens while retaining its measured mount.

    Translation and wxyz are absolute local replacements, not deltas: they are
    relative to the existing attachment body, with camera +X forward/+Z up.
    Four-view wrist_left/wrist_right specs select their own arm. A legacy hand
    spec requires explicit hand_arm and applies only that arm's override; both
    overrides are validated so later arm switching cannot hide invalid values.
    Legacy focal overrides must specify equal values on both arms, since its
    single render product keeps fixed intrinsics when switching attachments.
    """
    specs = tuple(specs)
    legacy = any(spec.name == "hand" for spec in specs)
    if legacy:
        if hand_arm not in ("left", "right"):
            raise ValueError("Legacy hand wrist overrides require explicit hand_arm left or right")
        if sum(spec.name == "hand" for spec in specs) != 1 or any(spec.name in ("wrist_left", "wrist_right") for spec in specs):
            raise ValueError("Wrist camera layout cannot mix or duplicate legacy and dual wrist cameras")
    elif hand_arm is not None:
        raise ValueError("hand_arm is only used for the legacy hand camera")
    validated = _validated_wrist_overrides(overrides, legacy=legacy)
    if not validated:
        return specs
    updated = list(specs)
    for arm, changes in validated.items():
        if legacy and arm != hand_arm:
            continue
        name = "hand" if legacy else f"wrist_{arm}"
        matches = [i for i, spec in enumerate(specs) if spec.name == name]
        if len(matches) != 1:
            raise ValueError(f"Wrist override requires exactly one measured {name} camera")
        index = matches[0]
        spec = specs[index]
        mount = (spec.attachment_prim_path or "").rsplit("/", 1)[-1]
        if (not spec.world_pose
                or mount not in (f"{arm[0]}_wrist_camera_link", f"Link_Wrist_Pitch_{arm.title()}")
                or (spec.source_camera_prim_path is not None
                    and not spec.source_camera_prim_path.endswith(f"/{arm.title()}WristCamera"))):
            raise ValueError(f"Wrist override for {arm} requires its measured wrist attachment, not a different or unmeasured camera")
        updated[index] = _with_mounted_camera_changes(spec, changes)
    return tuple(updated)


def validate_camera_config(config: Mapping[str, Any]) -> str:
    """Validate camera options before starting Isaac or creating output files."""
    if not isinstance(config, Mapping):
        raise ValueError("Camera configuration must be a mapping")
    layout = config.get("camera_layout", "head_both_wrists")
    if not isinstance(layout, str) or layout not in ("head_both_wrists", "head_hand"):
        raise ValueError("camera_layout must be head_both_wrists or head_hand")
    _validated_head_override(config.get("head_camera_override"))
    _validated_head_calibration(config.get("head_camera_calibration"))
    if config.get("head_camera_calibration") is not None and config.get("head_camera_override") is not None:
        raise ValueError("head_camera_override and head_camera_calibration cannot be used together")
    _validated_wrist_overrides(config.get("wrist_camera_overrides"), legacy=layout == "head_hand")
    with_third_person_pose(
        default_camera_specs(hand_arm="both" if layout == "head_both_wrists" else "right"),
        translation=config.get("third_person_translation"),
        orientation_wxyz=config.get("third_person_orientation_wxyz"),
        focal_length_usd_tenths=config.get("third_person_focal_length_usd_tenths"),
    )
    return layout


@dataclass(frozen=True)
class CameraFrameBundle:
    images: dict[str, np.ndarray]
    sim_time_s: float
    metadata: dict[str, dict[str, Any]]


class CameraCapture:
    """Create after SimulationApp and env reset; caller owns render/physics steps.

    initialize() creates RGB render products. Warm up rendering, then call read
    immediately after an explicitly requested render at a scheduler deadline.
    A stale or missing camera frame raises instead of producing synthetic pixels.
    """
    def __init__(self, specs: Sequence[CameraSpec] | None = None, *,
                 hand_specs: Mapping[str, CameraSpec] | None = None):
        self.specs = tuple(default_camera_specs() if specs is None else specs)
        if len({spec.name for spec in self.specs}) != len(self.specs):
            raise ValueError("Camera names must be unique")
        if len({spec.prim_path for spec in self.specs}) != len(self.specs):
            raise ValueError("Camera prim paths must be unique")
        self._hand_specs = dict(hand_specs or {})
        if self._hand_specs:
            if set(self._hand_specs) != {"left", "right"}:
                raise ValueError("hand_specs must provide left and right camera specifications")
            current = next((spec for spec in self.specs if spec.name == "hand"), None)
            for spec in self._hand_specs.values():
                if (current is None or spec.name != "hand" or not spec.attachment_prim_path
                        or (spec.prim_path, spec.width, spec.height, spec.focal_length_usd_tenths, spec.horizontal_aperture_usd_tenths,
                            spec.near_clip_m, spec.far_clip_m)
                        != (current.prim_path, current.width, current.height, current.focal_length_usd_tenths, current.horizontal_aperture_usd_tenths,
                            current.near_clip_m, current.far_clip_m)):
                    raise ValueError("Hand camera variants must share the render product and intrinsics")
        self._cameras: dict[str, Any] = {}
        self._last_render_times: dict[str, float] = {}
        self._attachment_poses_w: dict[str, np.ndarray] = {}
        self._attachments_updated = False
        self.cleanup_errors: list[str] = []

    def initialize(self) -> None:
        if self._cameras:
            raise RuntimeError("CameraCapture is already initialized")
        # These imports must happen only after the caller starts SimulationApp.
        import omni.usd
        # IsaacLab's --enable_cameras enables rendering but does not necessarily
        # load the separate high-level Isaac Sim camera Python extension.
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("isaacsim.sensors.camera")
        from isaacsim.sensors.camera import Camera
        from pxr import UsdGeom
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Start SimulationApp and load/reset the environment before cameras")
        if any(spec.calibrated_intrinsics is not None for spec in self.specs):
            _ensure_opencv_pinhole_schema()
        for spec in self.specs:
            if spec.attachment_prim_path and not stage.GetPrimAtPath(spec.attachment_prim_path).IsValid():
                raise ValueError(f"Camera attachment link does not exist: {spec.attachment_prim_path}")
            parent = spec.prim_path.rsplit("/", 1)[0]
            if not stage.GetPrimAtPath(parent).IsValid():
                if spec.world_pose:
                    UsdGeom.Xform.Define(stage, parent)
                else:
                    raise ValueError(f"Camera attachment link does not exist: {parent}")
            camera = Camera(prim_path=spec.prim_path, name=f"nepes_{spec.name}",
                            resolution=(spec.width, spec.height), frequency=-1, annotator_device="cpu")
            # Retain partially initialized cameras so caller cleanup can reach
            # their annotators/render products if a later SDK operation fails.
            self._cameras[spec.name] = camera
            if spec.attachment_prim_path:
                # Its physical world pose is supplied before the first render.
                pass
            elif spec.world_pose:
                camera.set_world_pose(position=np.array(spec.translation),
                                      orientation=np.array(spec.orientation_wxyz), camera_axes="world")
            else:
                camera.set_local_pose(translation=np.array(spec.translation),
                                      orientation=np.array(spec.orientation_wxyz), camera_axes="world")
            _set_camera_optics(camera, spec)
            # USD's default near plane is 1 stage unit, i.e. a full meter here.
            # It otherwise clips the robot torso and all close hand interactions.
            camera.set_clipping_range(near_distance=spec.near_clip_m, far_distance=spec.far_clip_m)
            camera.initialize()

    def set_hand_arm(self, arm: str) -> None:
        """Switch the measured wrist attachment at an episode boundary.

        Reuses the same RGB annotator and render product; rebuilding SDK camera
        graphs between tasks is unnecessary and can fail during RTX teardown.
        A fresh update_attachment_poses call is required before the next frame.
        """
        if arm not in ("left", "right"):
            raise ValueError("hand_arm must be left or right")
        matches = [spec for spec in self.specs if spec.name == "hand"]
        if len(matches) != 1 or not matches[0].attachment_prim_path:
            raise ValueError("set_hand_arm requires a measured hand camera attachment")
        hand = matches[0]
        if self._hand_specs:
            updated = self._hand_specs[arm]
        else:
            current_path = hand.attachment_prim_path
            if current_path.rsplit("/", 1)[-1] not in ("Link_Wrist_Pitch_Left", "Link_Wrist_Pitch_Right"):
                raise ValueError("Authored hand camera switching requires left/right hand_specs")
            robot_path = current_path.rsplit("/", 1)[0]
            updated = replace(hand, attachment_prim_path=f"{robot_path}/Link_Wrist_Pitch_{arm.title()}")
        self.specs = tuple(updated if spec.name == "hand" else spec for spec in self.specs)
        self._attachment_poses_w.clear()
        self.reset_timestamps()

    def update_attachment_poses(self, body_poses_w: Mapping[str, Any]) -> None:
        """Set camera world poses from the current measured articulation links.

        Call after the physics state update and immediately before rendering.
        Keys may be full attachment prim paths or robot body names. Values are
        xyz+wxyz; no USD parent transform is read as a substitute for physics.
        All requested inputs are validated before any camera pose is changed.
        """
        if not self._cameras:
            raise RuntimeError("Initialize cameras before updating measured attachments")
        pending = {}
        for spec in self.specs:
            if spec.attachment_prim_path is None:
                continue
            path = spec.attachment_prim_path
            key = path if path in body_poses_w else path.rsplit("/", 1)[-1]
            if key not in body_poses_w:
                raise ValueError(f"Missing measured camera attachment pose: {path}")
            body_w = pose_to_matrix(_cpu_array(body_poses_w[key]))
            local = pose_to_matrix([*spec.translation, *spec.orientation_wxyz])
            pending[spec.name] = np.asarray(matrix_to_pose(body_w @ local))
        for name, pose in pending.items():
            self._cameras[name].set_world_pose(position=pose[:3], orientation=pose[3:], camera_axes="world")
        self._attachment_poses_w = pending
        self._attachments_updated = True

    def read(self, sim_time_s: float, *, tolerance_s: float = 1e-5) -> CameraFrameBundle:
        if not math.isfinite(sim_time_s) or not self._cameras:
            raise RuntimeError("A finite simulation time and initialized cameras are required")
        if any(spec.attachment_prim_path for spec in self.specs) and not self._attachments_updated:
            raise RuntimeError("Update measured camera attachment poses before every captured render")
        images, metadata = {}, {}
        for spec in self.specs:
            camera = self._cameras[spec.name]
            frame = camera.get_current_frame()
            rendered_at = float(frame.get("rendering_time", float("nan")))
            if not math.isfinite(rendered_at) or abs(rendered_at - sim_time_s) > tolerance_s:
                raise StaleCameraFrameError(f"Camera {spec.name} is stale: rendered {rendered_at}, simulation {sim_time_s}")
            if rendered_at <= self._last_render_times.get(spec.name, -math.inf):
                raise RuntimeError(f"Camera {spec.name} repeated a render timestamp")
            rgb = _cpu_array(camera.get_rgb(device="cpu"))
            if rgb.shape != (spec.height, spec.width, 3) or rgb.dtype != np.uint8:
                raise RuntimeError(f"Camera {spec.name} has no valid uint8 RGB frame: {rgb.shape}/{rgb.dtype}")
            position, quaternion = camera.get_world_pose(camera_axes="world")
            pose_w = np.concatenate([_cpu_array(position), _cpu_array(quaternion)])
            if spec.attachment_prim_path and not np.allclose(
                    pose_to_matrix(pose_w), pose_to_matrix(self._attachment_poses_w[spec.name]), atol=2e-5):
                raise RuntimeError(f"Camera {spec.name} world pose differs from its measured attachment")
            intrinsics, calibration_metadata = _camera_intrinsics_metadata(camera, spec)
            images[spec.name] = np.ascontiguousarray(rgb).copy()
            metadata[spec.name] = {
                "render_time_s": rendered_at,
                "pose_w_xyz_wxyz": pose_w.tolist(),
                "intrinsics": intrinsics.tolist(),
                "optics": _camera_optics_metadata(camera),
                "prim_path": spec.prim_path,
                "camera_axes": "world: +X forward, +Z up",
                "attachment_prim_path": spec.attachment_prim_path,
                "pose_source": "measured_physics_link" if spec.attachment_prim_path else "usd_camera",
                "clipping_range_m": [spec.near_clip_m, spec.far_clip_m],
                **calibration_metadata,
            }
        self._last_render_times = {name: data["render_time_s"] for name, data in metadata.items()}
        self._attachments_updated = False
        return CameraFrameBundle(images, float(sim_time_s), metadata)

    def reset_timestamps(self) -> None:
        """Call after an environment reset before capturing a new episode."""
        self._last_render_times.clear()
        self._attachments_updated = False

    @property
    def configuration(self) -> list[dict[str, Any]]:
        return [asdict(spec) for spec in self.specs]

    def close(self) -> list[str]:
        """Best-effort, idempotent release; report SDK errors before app shutdown.

        Isaac Sim 5.1 Camera.destroy handles custom annotators (including RGB)
        and its render product, but omits its ReferenceTime annotator. Explicitly
        detach that annotator and callbacks as well. One SDK failure must not
        prevent other camera releases or the caller's environment cleanup.
        """
        cameras, self._cameras = self._cameras, {}

        def attempt(name, operation, callback):
            try:
                callback()
            except Exception as exc:
                message = f"Camera {name} cleanup {operation} failed: {type(exc).__name__}: {exc}"
                self.cleanup_errors.append(message)
                print(f"[NEPES camera] {message}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()

        # The render products share SyntheticData dependency nodes. Detach
        # all consumers before destroying any product, or later detaches can
        # traverse graph nodes invalidated by the first product's destruction.
        for name, camera in cameras.items():
            attempt(name, "pause", camera.pause)
            for attribute in ("_acquisition_callback", "_stage_open_callback", "_timer_reset_callback"):
                if hasattr(camera, attribute):
                    setattr(camera, attribute, None)
        for name, camera in cameras.items():
            for annotator_name in list(getattr(camera, "_custom_annotators", {})):
                attempt(name, f"detach_{annotator_name}",
                        lambda c=camera, a=annotator_name: c.detach_annotator(a))
        for name, camera in cameras.items():
            reference_time = getattr(camera, "_fabric_time_annotator", None)
            if reference_time is not None:
                attempt(name, "detach_reference_time", reference_time.detach)
        for name, camera in cameras.items():
            attempt(name, "destroy", camera.destroy)
        for name, camera in cameras.items():
            # Camera.destroy omits the inherited SingleXFormPrim's two aliases
            # to its XFormPrim. That view owns a USDRT stage even with Fabric
            # simulation disabled. A retained SDK exception traceback can keep
            # Camera alive until Python finalization, after USDRT unloads.
            # Release only these known owned views while plugins are loaded.
            views = {}
            for attribute in ("_xform_prim_view", "_prim_view"):
                view = getattr(camera, attribute, None)
                if view is not None:
                    views[id(view)] = view
            for view in views.values():
                if hasattr(view, "destroy"):
                    attempt(name, "destroy_xform_view", view.destroy)
            for attribute in ("_xform_prim_view", "_prim_view"):
                if hasattr(camera, attribute):
                    setattr(camera, attribute, None)
        self._attachment_poses_w.clear()
        self.reset_timestamps()
        return list(self.cleanup_errors)
