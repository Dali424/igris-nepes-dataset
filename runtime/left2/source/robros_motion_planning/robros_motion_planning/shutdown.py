"""Orderly standalone shutdown while PhysX tensor plugins are still loaded."""

from __future__ import annotations


def release_articulation_views(articulation) -> None:
    """Release a finished IsaacLab articulation's references before Kit unloads.

    Call only after its environment has closed; the articulation cannot be used
    again. IsaacLab's STOP callback drops ``_root_physx_view`` but leaves strong
    ``SimulationView`` references on both Articulation and ArticulationData.
    In Isaac Sim 5.1 these can outlive the PhysX plugin and crash at Python exit.
    Tensor views have no public ``close`` method: remove these known owner
    references while the plugin is loaded, without modifying native backends.
    """
    if articulation is None:
        return
    # Release specific views before the simulation view that owns their scene.
    if hasattr(articulation, "_root_physx_view"):
        articulation._root_physx_view = None
    data = getattr(articulation, "_data", None)
    if data is not None and hasattr(data, "_physics_sim_view"):
        data._physics_sim_view = None
    if hasattr(articulation, "_physics_sim_view"):
        articulation._physics_sim_view = None


def close_simulation_app(app) -> None:
    """Call after environment/camera/contact owners have been closed.

    Isaac Sim 5.1's tensor API exposes ``reset`` specifically to release the
    stage's underlying tensor buffers. Stopping and dispatching the timeline
    first lets SimulationManager invalidate its torch/warp simulation views.
    Articulation owners must first call ``release_articulation_views``; reset
    does not remove their Python references to native simulation views.
    A generation-2 collection before app.close also releases unreachable camera
    frames while USDRT is loaded. Native debugger evidence in Isaac Sim 5.1
    showed an OSError traceback cycle retaining Camera -> XFormPrim -> USDRT
    until Py_FinalizeEx, after Kit had already unloaded the owning plugins.
    This performs normal application shutdown and preserves the exit status.
    """
    import omni.physics.tensors
    import omni.timeline

    try:
        print("[NEPES shutdown] Stopping timeline", flush=True)
        omni.timeline.get_timeline_interface().stop()
        print("[NEPES shutdown] Dispatching stop callbacks", flush=True)
        app.update()
        print("[NEPES shutdown] Resetting physics tensor buffers", flush=True)
        omni.physics.tensors.reset()
    finally:
        import gc
        print("[NEPES shutdown] Collecting unreachable camera/frame cycles before plugin unload", flush=True)
        collected = gc.collect()
        print(f"[NEPES shutdown] Collected {collected} unreachable objects", flush=True)
        print("[NEPES shutdown] Closing SimulationApp", flush=True)
        app.close()
        print("[NEPES shutdown] SimulationApp closed", flush=True)
