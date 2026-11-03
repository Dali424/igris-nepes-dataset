"""A bounded command hold while a physically supported object settles.

This changes the hand reference only. It never declares support or success,
relaxes retention, or commands an object pose. The FSM keeps its independent
load, actual velocity, target-pose and dwell checks before opening the hand.
"""
import math
import numpy as np


def validate_load_control(settings):
    """Validate a small, optional hand admittance; this is not a support gate."""
    limits = {"target_load_fraction": (0.5, 1.), "admittance_m_s_N": (0., .002),
              "max_speed_m_s": (0., .005), "max_displacement_m": (0., .005)}
    optional = {"contact_load_fraction", "contact_dwell_s"}
    if (not isinstance(settings, dict) or not set(limits) <= set(settings)
            or set(settings) - set(limits) not in (set(), optional)):
        raise ValueError("Support load control needs load fraction, admittance, speed and displacement")
    if optional <= set(settings):
        limits.update(contact_load_fraction=(.05, .49), contact_dwell_s=(.1, 1.))
    for key, (lower, upper) in limits.items():
        value = settings[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not (lower <= value <= upper if key in optional else lower < value <= upper)):
            raise ValueError(f"Invalid support load control {key}")
    return dict(settings)


class SupportSettlingHold:
    def __init__(self, duration_s, load_control=None):
        if isinstance(duration_s, bool) or not math.isfinite(duration_s) or not 0 < duration_s <= 5:
            raise ValueError("Support hold duration must be in (0, 5] seconds")
        self.duration_s = float(duration_s)
        self._target = None
        self._until = None
        self._armed = True
        self._last_time = None
        self.load_control = None if load_control is None else validate_load_control(load_control)
        self._anchor = None
        self._normal = None
        self._offset = 0.
        self._contact_since = None
        self._touch_consumed = False
        self.activation = None

    def update(self, now, *, loaded, valid, measured_tcp, normal_w=None, force_w=None, mass_kg=None):
        if not math.isfinite(now) or (self._last_time is not None and now < self._last_time):
            raise ValueError("Support hold needs a finite monotonic measurement time")
        dt = 0. if self._last_time is None else now - self._last_time
        self._last_time = now
        normal = None
        if self.load_control is not None and valid:
            try:
                normal = np.asarray(normal_w, dtype=float)
                force = np.asarray(force_w, dtype=float)
                valid = bool(normal.shape == force.shape == (3,) and np.isfinite(normal).all()
                    and np.isfinite(force).all() and abs(np.linalg.norm(normal) - 1.) < 1e-6
                    and normal[2] > 0 and not isinstance(mass_kg, bool)
                    and math.isfinite(mass_kg) and mass_kg > 0)
                if self._normal is not None:
                    valid = valid and np.allclose(normal, self._normal, atol=1e-9, rtol=0.)
            except (TypeError, ValueError):
                valid = False
        if not valid:
            self._target = self._until = None
            self._anchor = self._normal = None
            self._offset = 0.
            self._armed = True
            self._contact_since = None
            return None
        touch_ready = False
        touch_mode = self.load_control is not None and "contact_load_fraction" in self.load_control
        if touch_mode:
            contact = float(force @ normal) >= self.load_control["contact_load_fraction"] * mass_kg * 9.81
            if contact:
                if self._contact_since is None:
                    self._contact_since = now
                touch_ready = now - self._contact_since + 1e-10 >= self.load_control["contact_dwell_s"]
            else:
                self._contact_since = None
        if not loaded:
            self._armed = True
        if self._target is not None:
            if now < self._until:
                if self.load_control is not None:
                    cfg = self.load_control
                    error = cfg["target_load_fraction"] * mass_kg * 9.81 - float(force @ normal)
                    speed = np.clip(cfg["admittance_m_s_N"] * error,
                                    -cfg["max_speed_m_s"], cfg["max_speed_m_s"])
                    self._offset = float(np.clip(self._offset + speed * dt, 0., cfg["max_displacement_m"]))
                    self._target[:3, 3] = self._anchor[:3, 3] - self._normal * self._offset
                return self._target.copy()
            self._target = self._until = None
            self._anchor = self._normal = None
            self._armed = False
            return None
        if (loaded or touch_ready) and self._armed and not (touch_mode and self._touch_consumed):
            target = np.asarray(measured_tcp, dtype=float)
            if target.shape != (4, 4) or not np.isfinite(target).all():
                raise ValueError("Support hold requires a finite measured TCP transform")
            self._target = target.copy()
            self._anchor = target.copy()
            self._normal = None if normal is None else normal.copy()
            self._offset = 0.
            self._until = now + self.duration_s
            self._armed = False
            self._touch_consumed = touch_mode
            self.activation = {"time_s": now, "kind": "independent_loaded_gate" if loaded else "measured_support_touch",
                               "load_fraction": None if normal is None else float(force @ normal) / (mass_kg * 9.81),
                               "support_success_criteria_unchanged": True}
            return self._target.copy()
        return None
