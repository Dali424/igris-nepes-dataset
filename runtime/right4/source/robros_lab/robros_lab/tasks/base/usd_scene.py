"""Run an IGRIS-C task inside an authored USD scene such as the NEPES workcell."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


SCENE_PRIM_PATH = "/World/Scene"
DEFAULT_SCENE_ROBOT_PRIM = "/World/igris_c"
_DEFAULT_CAMERA_FOCUS_DISTANCE = 3.0
# NuRec volumes are primary-ray only; reflections trace the matte proxy mesh instead.
_NUREC_PROXY_REFLECTION_SETTINGS = {
    "/rtx/matteObject/enabled": True,
    "/rtx/post/matteObject/enableShadowCatcher": False,
    "/rtx/matteObject/enableAmbientShadowCatcher": False,
    "/rtx/matteObject/visibility/secondaryRays": True,
}


def spawn_usd_scene(
    prim_path: str,
    cfg: UsdSceneCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Reference a scene USD and deactivate the prims that the task replaces.

    Authored physics scenes are deactivated as well: the task's simulation context
    owns the only PhysX scene, and a second one would split the simulation.
    """

    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    stage = prim.GetStage()
    for relative_path in cfg.inactive_prim_paths:
        replaced_prim = stage.GetPrimAtPath(f"{prim_path}/{relative_path}")
        if not replaced_prim.IsValid():
            raise ValueError(f"Scene prim '{relative_path}' does not exist in {cfg.usd_path}.")
        replaced_prim.SetActive(False)
    physics_scenes = [child for child in Usd.PrimRange(prim) if child.IsA(UsdPhysics.Scene)]
    for physics_scene in physics_scenes:
        physics_scene.SetActive(False)
    return prim


@configclass
class UsdSceneCfg(sim_utils.UsdFileCfg):
    """Scene USD referenced as the static environment around a task."""

    func: Callable = spawn_usd_scene

    inactive_prim_paths: tuple[str, ...] = ()
    """Prim paths relative to the scene's default prim that are deactivated after referencing."""


@dataclass(frozen=True)
class UsdScenePlacement:
    """Where :func:`configure_usd_scene` placed the robot and the viewer."""

    usd_path: Path
    robot_prim_path: str
    robot_pos: tuple[float, float, float]
    robot_yaw: float
    camera_prim_path: str | None
    """Stage path of the camera the scene was saved with, if the scene references it."""


def configure_usd_scene(
    env_cfg: ManagerBasedEnvCfg,
    usd_path: str | Path,
    robot_prim_path: str = DEFAULT_SCENE_ROBOT_PRIM,
) -> UsdScenePlacement:
    """Replace the task's ground and lights with an authored USD scene.

    The scene provides the ground, colliders, and lighting. The controlled robot keeps
    the task's spawn height and joint state but takes the planar position and heading
    of ``robot_prim_path``; that scene prim is deactivated so the task's articulation
    replaces it. The viewer starts at the camera the scene was saved with.
    """

    if env_cfg.scene.num_envs != 1:
        raise ValueError(f"A USD scene holds a single robot, got num_envs={env_cfg.scene.num_envs}.")
    usd_path = Path(usd_path).expanduser().resolve()
    layer = Sdf.Layer.FindOrOpen(str(usd_path))
    if layer is None:
        raise FileNotFoundError(f"Scene USD does not exist: {usd_path}")
    if not layer.defaultPrim:
        raise ValueError(f"Scene USD has no defaultPrim: {usd_path}")
    default_prim_path = Sdf.Path.absoluteRootPath.AppendChild(layer.defaultPrim)
    robot_path = Sdf.Path(robot_prim_path)
    if robot_path == default_prim_path or not robot_path.HasPrefix(default_prim_path):
        raise ValueError(f"The scene robot prim must be below {default_prim_path}, got {robot_prim_path}.")
    camera_path = layer.customLayerData.get("cameraSettings", {}).get("boundCamera")

    # Compose only the robot and camera branches; the scene's heavy assets are not needed here.
    mask_paths = [robot_path] + ([Sdf.Path(camera_path)] if camera_path else [])
    stage = Usd.Stage.OpenMasked(layer, Usd.StagePopulationMask(mask_paths), Usd.Stage.LoadNone)
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z or not UsdGeom.LinearUnitsAre(
        UsdGeom.GetStageMetersPerUnit(stage), UsdGeom.LinearUnits.meters
    ):
        raise ValueError(f"Scene USD must be Z-up and in meters like the task stage: {usd_path}")
    # The default prim is referenced at SCENE_PRIM_PATH with an identity transform, so scene
    # coordinates are task coordinates only if the default prim is not transformed itself.
    xform_cache = UsdGeom.XformCache()
    if not Gf.IsClose(xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(default_prim_path)), Gf.Matrix4d(1.0), 1e-9):
        raise ValueError(f"The scene's default prim {default_prim_path} must not be transformed: {usd_path}")
    robot_prim = stage.GetPrimAtPath(robot_path)
    if not robot_prim.IsValid():
        raise ValueError(f"Scene robot prim {robot_prim_path} does not exist in {usd_path}.")
    robot_to_scene = xform_cache.GetLocalToWorldTransform(robot_prim)
    robot_pos = robot_to_scene.ExtractTranslation()
    heading = robot_to_scene.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
    robot_yaw = math.atan2(heading[1], heading[0])

    scene_cfg = env_cfg.scene
    for name, entity in list(vars(scene_cfg).items()):
        if isinstance(entity, TerrainImporterCfg) or isinstance(
            getattr(entity, "spawn", None), (sim_utils.GroundPlaneCfg, sim_utils.LightCfg)
        ):
            setattr(scene_cfg, name, None)
    robot_cfg = scene_cfg.robot
    spawn_pos = (float(robot_pos[0]), float(robot_pos[1]), robot_cfg.init_state.pos[2])
    scene_cfg.robot = robot_cfg.replace(
        init_state=robot_cfg.init_state.replace(
            pos=spawn_pos,
            rot=(math.cos(0.5 * robot_yaw), 0.0, 0.0, math.sin(0.5 * robot_yaw)),
        )
    )
    scene_cfg.usd_scene = AssetBaseCfg(
        prim_path=SCENE_PRIM_PATH,
        spawn=UsdSceneCfg(
            usd_path=str(usd_path),
            inactive_prim_paths=(str(robot_path.MakeRelativePath(default_prim_path)),),
        ),
        collision_group=-1,
    )

    camera_prim_path = None
    # Kit saves "/OmniverseKit_Persp" when the stage was viewed through its own camera.
    camera_prim = stage.GetPrimAtPath(camera_path) if camera_path else None
    if camera_prim and camera_prim.IsA(UsdGeom.Camera) and camera_prim.GetPath().HasPrefix(default_prim_path):
        camera_to_scene = xform_cache.GetLocalToWorldTransform(camera_prim)
        focus = camera_prim.GetAttribute("omni:kit:centerOfInterest").Get()
        focus = Gf.Vec3d(focus) if focus is not None else Gf.Vec3d(0.0, 0.0, -_DEFAULT_CAMERA_FOCUS_DISTANCE)
        # The task's viewer camera, used for --video, starts at the same pose.
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.eye = tuple(float(value) for value in camera_to_scene.Transform(Gf.Vec3d(0.0, 0.0, 0.0)))
        env_cfg.viewer.lookat = tuple(float(value) for value in camera_to_scene.Transform(focus))
        relative_camera_path = camera_prim.GetPath().MakeRelativePath(default_prim_path)
        camera_prim_path = str(Sdf.Path(SCENE_PRIM_PATH).AppendPath(relative_camera_path))

    return UsdScenePlacement(
        usd_path=usd_path,
        robot_prim_path=robot_prim_path,
        robot_pos=spawn_pos,
        robot_yaw=robot_yaw,
        camera_prim_path=camera_prim_path,
    )


def view_usd_scene_camera(placement: UsdScenePlacement) -> str | None:
    """Show the camera the scene was saved with in the GUI viewport.

    Returns the camera prim path, or ``None`` when the scene has no saved camera or
    there is no viewport.
    """

    from omni.kit.viewport.utility import get_active_viewport

    viewport = get_active_viewport()
    if placement.camera_prim_path is None or viewport is None:
        return None
    viewport.set_active_camera(placement.camera_prim_path)
    return placement.camera_prim_path


def apply_usd_scene_render_settings(usd_path: str | Path, scene_prim_path: str = SCENE_PRIM_PATH) -> int:
    """Apply the RTX settings that opening the scene in the Isaac Sim GUI applies.

    Mirrors ``renderer_nurec``'s GUI integration: render settings authored in the
    layers the scene uses, with the scene's own layer taking precedence, and, for NuRec
    volumes with a proxy mesh, the matte-object settings for proxy reflections.
    Returns the number of authored render settings that were applied.
    """

    import carb

    stage = sim_utils.get_current_stage()
    scene_layer = Sdf.Layer.FindOrOpen(str(Path(usd_path).expanduser().resolve()))
    render_settings: dict[str, object] = {}
    for layer in stage.GetUsedLayers():
        if layer != scene_layer:
            render_settings.update(layer.customLayerData.get("renderSettings", {}))
    render_settings.update(scene_layer.customLayerData.get("renderSettings", {}))

    settings = carb.settings.get_settings()
    for name, value in render_settings.items():
        settings.set("/" + name.replace(":", "/"), value)

    proxy_targets = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(scene_prim_path)):
        marker = prim.GetAttribute("omni:nurec:isNuRecVolume")
        proxy = prim.GetRelationship("proxy")
        if prim.GetTypeName() == "Volume" and marker.IsValid() and marker.Get() and proxy.IsValid():
            proxy_targets += proxy.GetTargets()
    if proxy_targets:
        for name, value in _NUREC_PROXY_REFLECTION_SETTINGS.items():
            settings.set_bool(name, value)
    return len(render_settings)


__all__ = [
    "DEFAULT_SCENE_ROBOT_PRIM",
    "SCENE_PRIM_PATH",
    "UsdSceneCfg",
    "UsdScenePlacement",
    "apply_usd_scene_render_settings",
    "configure_usd_scene",
    "spawn_usd_scene",
    "view_usd_scene_camera",
]
