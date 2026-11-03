"""Wall-clock accounting and optional capture-owned rendering.

The physics/control clocks and capture deadlines are never changed here.
IsaacLab's periodic render is redundant when the headless recorder already
renders synchronously at every camera deadline.
"""
from contextlib import contextmanager
from functools import wraps
import sys
import time


class ThroughputProfile:
    def __init__(self, enabled=False):
        self.enabled = bool(enabled)
        self.started = time.perf_counter()
        self.sections = {}

    @contextmanager
    def measure(self, name):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            item = self.sections.setdefault(name, {"calls": 0, "wall_seconds": 0.0})
            item["calls"] += 1
            item["wall_seconds"] += elapsed

    def track(self, name):
        def decorate(function):
            if not self.enabled:
                return function
            @wraps(function)
            def wrapped(*args, **kwargs):
                with self.measure(name):
                    return function(*args, **kwargs)
            return wrapped
        return decorate

    def summary(self):
        return {
            "enabled": self.enabled,
            "runtime_wall_seconds": time.perf_counter() - self.started,
            "sections": {name: dict(item) for name, item in self.sections.items()},
            "accounting": "Sections can nest; do not add inclusive durations. Runtime excludes SimulationApp startup.",
        }


class CaptureRenderControl:
    """Suppress only periodic rendering; explicit sim.render() still works.

    The large interval is an integer supported by IsaacLab's modulo scheduling.
    No sim.step, sensor data, timestamps, or graphics-quality settings are
    overridden. Restore both the config and method lookup during cleanup.
    """
    def __init__(self, raw, *, mode, headless, recording, profile):
        if mode not in ("legacy", "capture_only"):
            raise ValueError("Unknown capture render mode")
        if mode == "capture_only" and not (headless and recording):
            raise ValueError("capture_only requires headless RGB recording")
        self.raw, self.mode, self.profile = raw, mode, profile
        self.original_interval = raw.cfg.sim.render_interval
        self.active = False
        self.calls = 0
        self._had_render_override = "render" in vars(raw.sim)
        self._original_render = raw.sim.render
        self._wrapped = False
        if profile.enabled:
            raw.sim.render = self._render
            self._wrapped = True

    def _render(self, *args, **kwargs):
        self.calls += 1
        with self.profile.measure("all_render_calls"):
            return self._original_render(*args, **kwargs)

    def activate(self):
        # Activate after camera initialization. A fresh explicit render is still
        # required at each capture deadline, including frame zero.
        if self.mode == "capture_only":
            if getattr(self.raw, "_sim_step_counter", 0) >= sys.maxsize:
                raise RuntimeError("Physics counter exceeds capture-only render interval")
            self.raw.cfg.sim.render_interval = sys.maxsize
        self.active = True

    def summary(self):
        return {
            "mode": self.mode,
            "original_render_interval_physics_ticks": self.original_interval,
            "periodic_render_suppressed": self.active and self.mode == "capture_only",
            "render_calls_observed": self.calls if self.profile.enabled else None,
            "physics_dt_s": float(self.raw.physics_dt),
            "control_dt_s": float(self.raw.step_dt),
            "capture_fps": 30,
        }

    def restore(self):
        self.raw.cfg.sim.render_interval = self.original_interval
        if self._wrapped:
            if self._had_render_override:
                self.raw.sim.render = self._original_render
            else:
                del self.raw.sim.render
            self._wrapped = False
