"""Opt-in numerical contact-distance calibration for the thin CRS/cart guide.

This changes when contacts are generated, not the resting separation, mesh,
material, mass, or task acceptance. It is an experimental simulation profile.
"""
from __future__ import annotations

import math


PROFILE = "crs_cart_contact_distance_0p1mm"
ROOTS = ("/World/Scene/Ring_01", "/World/Scene/Cart")
CONTACT_OFFSET_M = 0.0001
REFERENCE = "https://nvidia-omniverse.github.io/PhysX/physx/5.1.2/docs/AdvancedCollisionDetection.html"


def author_contact_distance(stage, profile: str) -> dict:
    """Validate every collider first, then author only contact offsets in session."""
    from pxr import Sdf, Usd, UsdPhysics

    if profile != PROFILE:
        raise ValueError("Unknown contact_distance_profile")
    changes = []
    for root_path in ROOTS:
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid() or not root.IsActive():
            raise ValueError(f"Missing contact-distance root: {root_path}")
        count = 0
        for prim in Usd.PrimRange(root):
            if (not prim.HasAPI(UsdPhysics.CollisionAPI)
                    or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False):
                continue
            if prim.IsInstanceProxy():
                raise ValueError(f"Cannot calibrate instance-proxy collider: {prim.GetPath()}")
            contact = prim.GetAttribute("physxCollision:contactOffset").Get()
            rest = prim.GetAttribute("physxCollision:restOffset").Get()
            # Explicit original settings are required: no inferred auto offsets.
            if (contact is None or not math.isfinite(float(contact))
                    or abs(float(contact) - 0.0005) > 1e-9 or rest != 0.0):
                raise ValueError(f"Expected original 0.5mm/0mm offsets: {prim.GetPath()}")
            changes.append((prim, {"path": str(prim.GetPath()),
                                   "original_contact_offset_m": float(contact),
                                   "rest_offset_m": float(rest)}))
            count += 1
        if not count:
            raise ValueError(f"No active contact-distance colliders: {root_path}")
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for prim, record in changes:
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(CONTACT_OFFSET_M)
            record["contact_offset_m"] = float(prim.GetAttribute("physxCollision:contactOffset").Get())
    return {"schema": "nepes_contact_distance_calibration_v1", "profile": PROFILE,
            "experimental": True, "physical_success_verified": False,
            "reference": REFERENCE, "session_layer_only": True,
            "unchanged": ["geometry", "rest_offset", "mass", "material", "effort", "success_criteria"],
            "changes": [record for _, record in changes]}
