"""CPU-only IGRIS-C kinematics and conservative cuRobo 0.7 robot configuration.

The planning base is the URDF ``base_link`` (the torso), not the world frame.
Only one seven-joint arm moves; every other movable joint is held at the supplied
snapshot. Collision geometry uses authored collision shapes, or visual geometry
when a link has no collision shape. Spheres conservatively cover occupied mesh
cells; snapshot-adaptive refinement removes proxy collisions without disabling
link pairs. Legacy cuRobo's strict sphere-count limit is checked before loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence
import copy
import math
import xml.etree.ElementTree as ET

import numpy as np


ARM_JOINT_PARTS = (
    "Shoulder_Pitch", "Shoulder_Roll", "Shoulder_Yaw", "Elbow_Pitch",
    "Wrist_Yaw", "Wrist_Roll", "Wrist_Pitch",
)


def arm_joint_names(arm: str) -> tuple[str, ...]:
    """Canonical proximal-to-distal seven-joint order."""
    if arm not in ("left", "right"):
        raise ValueError("arm must be 'left' or 'right'")
    return tuple(f"Joint_{part}_{arm.title()}" for part in ARM_JOINT_PARTS)


def default_ee_link(arm: str) -> str:
    arm_joint_names(arm)
    return f"{arm.title()}_Hand"


def default_urdf_path() -> Path:
    """Locate the authored URDF in a source checkout, without importing Isaac."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / (
            "source/robros_lab/robros_lab/assets/robots/robros/igris_c/"
            "from_urdf/urdf/igris_c_v2_wrist_hand_isaac.urdf"
        )
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("IGRIS-C URDF not found; pass urdf_path explicitly")


def _vector(value: str | None, default: tuple[float, float, float]) -> np.ndarray:
    result = np.array(default if value is None else [float(v) for v in value.split()])
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"Expected three finite values, got {value!r}")
    return result


def _origin(element: ET.Element | None) -> np.ndarray:
    transform = np.eye(4)
    if element is None:
        return transform
    roll, pitch, yaw = _vector(element.get("rpy"), (0.0, 0.0, 0.0))
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    transform[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    transform[:3, 3] = _vector(element.get("xyz"), (0.0, 0.0, 0.0))
    return transform


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float | None
    upper: float | None
    velocity: float | None


@dataclass(frozen=True)
class RobotModel:
    urdf_path: Path
    root_link: str
    link_names: tuple[str, ...]
    joints: dict[str, Joint]
    xml: ET.Element

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(name for name, joint in self.joints.items() if joint.kind != "fixed")

    @property
    def joint_limits(self) -> dict[str, tuple[float, float]]:
        return {
            name: (float(j.lower), float(j.upper))
            for name, j in self.joints.items() if j.kind != "fixed"
        }

    def arm_joint_names(self, arm: str) -> tuple[str, ...]:
        names = arm_joint_names(arm)
        if not set(names).issubset(self.joints):
            raise ValueError(f"URDF does not contain the complete {arm} IGRIS-C arm")
        return names

    def default_ee_link(self, arm: str) -> str:
        link = default_ee_link(arm)
        if link not in self.link_names:
            raise ValueError(f"URDF has no end-effector link {link}")
        return link

    def validate_snapshot(self, joint_positions: Mapping[str, float]) -> dict[str, float]:
        missing = set(self.joint_names) - set(joint_positions)
        if missing:
            raise ValueError(f"Snapshot is missing joints: {sorted(missing)}")
        unknown = set(joint_positions) - set(self.joints)
        if unknown:
            raise ValueError(f"Snapshot has unknown joints: {sorted(unknown)}")
        values = {name: float(joint_positions[name]) for name in self.joint_names}
        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"Snapshot joint {name} is not finite")
            low, high = self.joint_limits[name]
            if value < low - 1e-6 or value > high + 1e-6:
                raise ValueError(f"Snapshot joint {name}={value} exceeds URDF limits [{low}, {high}]")
        return values

    def forward_kinematics(self, joint_positions: Mapping[str, float]) -> dict[str, np.ndarray]:
        """Return link transforms relative to root_link, using named joint values."""
        values = self.validate_snapshot(joint_positions)
        transforms = {self.root_link: np.eye(4)}
        pending = list(self.joints.values())
        while pending:
            progress = False
            for joint in pending[:]:
                if joint.parent not in transforms:
                    continue
                motion = np.eye(4)
                if joint.kind == "prismatic":
                    motion[:3, 3] = joint.axis * values[joint.name]
                elif joint.kind != "fixed":
                    x, y, z = joint.axis
                    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                    angle = values[joint.name]
                    motion[:3, :3] += math.sin(angle) * skew + (1 - math.cos(angle)) * skew @ skew
                transforms[joint.child] = transforms[joint.parent] @ joint.origin @ motion
                pending.remove(joint)
                progress = True
            if not progress:
                raise ValueError("URDF has a cycle or disconnected joints")
        return transforms


@dataclass(frozen=True)
class RobotModelConfig:
    """``robot_cfg`` is the inner dictionary passed to cuRobo, without a wrapper."""
    robot_cfg: dict[str, Any]
    active_joint_names: tuple[str, ...]
    locked_joint_positions: dict[str, float]
    ee_link: str
    generated_urdf_path: Path
    collision_spheres: dict[str, list[dict[str, Any]]]
    joint_limits: dict[str, tuple[float, float]]
    geometry_sources: dict[str, str]
    collision_model_metadata: dict[str, Any]


def load_robot_model(urdf_path: str | Path | None = None) -> RobotModel:
    path = (default_urdf_path() if urdf_path is None else Path(urdf_path)).resolve()
    root = ET.parse(path).getroot()
    links = tuple(link.attrib["name"] for link in root.findall("link"))
    if len(set(links)) != len(links):
        raise ValueError("Duplicate URDF link names")
    joints: dict[str, Joint] = {}
    children: set[str] = set()
    for element in root.findall("joint"):
        name, kind = element.attrib["name"], element.attrib["type"]
        if name in joints or kind not in ("fixed", "revolute", "prismatic"):
            raise ValueError(f"Duplicate or unsupported joint: {name} ({kind})")
        parent, child = element.find("parent"), element.find("child")
        if parent is None or child is None:
            raise ValueError(f"Missing parent/child for joint {name}")
        parent_name, child_name = parent.attrib["link"], child.attrib["link"]
        if parent_name not in links or child_name not in links or child_name in children:
            raise ValueError(f"Invalid URDF tree at joint {name}")
        children.add(child_name)
        axis_element, limit = element.find("axis"), element.find("limit")
        axis = _vector(None if axis_element is None else axis_element.get("xyz"), (1.0, 0.0, 0.0))
        if kind != "fixed" and (limit is None or np.linalg.norm(axis) == 0):
            raise ValueError(f"Missing limits or zero axis for {name}")
        lower = upper = velocity = None
        if kind != "fixed":
            lower, upper = float(limit.attrib["lower"]), float(limit.attrib["upper"])
            velocity = float(limit.attrib["velocity"])
            if not np.isfinite([lower, upper, velocity]).all() or lower >= upper or velocity <= 0:
                raise ValueError(f"Invalid joint limits for {name}")
        joints[name] = Joint(name, kind, parent_name, child_name, _origin(element.find("origin")),
                             axis / np.linalg.norm(axis), lower, upper, velocity)
    roots = set(links) - children
    if len(roots) != 1:
        raise ValueError(f"Expected one root link, found {sorted(roots)}")
    return RobotModel(path, roots.pop(), links, joints, root)


def resolve_mesh_path(filename: str, urdf_path: str | Path) -> Path:
    """Resolve authored relative or absolute mesh paths; reject unresolved ROS URIs."""
    if filename.startswith("file://"):
        filename = filename[7:]
    if "://" in filename:
        raise ValueError(f"Mesh URI requires explicit resolution: {filename}")
    path = Path(filename)
    path = (Path(urdf_path).parent / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"URDF mesh not found: {path}")
    return path


def tiled_aabb_spheres(bounds: np.ndarray, cell_size: float = 0.04) -> list[dict[str, Any]]:
    """Cover every point in an AABB by enclosing each grid cell with a sphere.

    Cell side lengths are at most cell_size; each radius is half the actual cell
    diagonal. This covers interiors and faces, unlike sampling mesh vertices.
    """
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] < bounds[0]):
        raise ValueError("bounds must be finite ordered (2, 3) coordinates")
    if not math.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("cell_size must be positive and finite")
    extent = bounds[1] - bounds[0]
    counts = np.maximum(1, np.ceil(extent / cell_size).astype(int))
    if math.prod(int(n) for n in counts) > 100_000:
        raise ValueError("Collision geometry requires over 100,000 spheres; check mesh units")
    steps = extent / counts
    radius = max(float(np.linalg.norm(steps) / 2), 1e-6)
    return [
        {"center": (bounds[0] + (np.array(index) + 0.5) * steps).tolist(), "radius": radius}
        for index in product(*(range(int(n)) for n in counts))
    ]


def _shape_bounds(geometry: ET.Element, urdf_path: Path) -> np.ndarray:
    if len(geometry) != 1:
        raise ValueError("A URDF geometry must contain exactly one shape")
    shape = geometry[0]
    if shape.tag == "box":
        half = _vector(shape.get("size"), (0, 0, 0)) / 2
    elif shape.tag == "sphere":
        half = np.full(3, float(shape.attrib["radius"]))
    elif shape.tag == "cylinder":
        half = np.array([float(shape.attrib["radius"])] * 2 + [float(shape.attrib["length"]) / 2])
    elif shape.tag == "mesh":
        import trimesh  # Optional for primitive-only URDFs; no simulator dependency.
        path = resolve_mesh_path(shape.attrib["filename"], urdf_path)
        mesh = trimesh.load(str(path), force="mesh", process=False)
        vertices = np.asarray(mesh.vertices) * _vector(shape.get("scale"), (1, 1, 1))
        if vertices.size == 0 or not np.isfinite(vertices).all():
            raise ValueError(f"Invalid mesh vertices in {path}")
        return np.stack([vertices.min(axis=0), vertices.max(axis=0)])
    else:
        raise ValueError(f"Unsupported collision shape: {shape.tag}")
    if not np.isfinite(half).all() or np.any(half <= 0):
        raise ValueError(f"Invalid {shape.tag} geometry")
    return np.stack([-half, half])


@dataclass
class _CollisionShape:
    link: str
    transform: np.ndarray
    bounds: np.ndarray
    mesh: Any = None
    approximation: dict[str, Any] | None = None


@dataclass
class _CollisionCell:
    shape: _CollisionShape
    lower: np.ndarray
    upper: np.ndarray
    enclosing_radius: float | None = None

    @property
    def center(self) -> np.ndarray:
        return self.shape.transform[:3, :3] @ ((self.lower + self.upper) / 2) + self.shape.transform[:3, 3]

    @property
    def radius(self) -> float:
        value = (np.linalg.norm(self.upper - self.lower) / 2
                 if self.enclosing_radius is None else self.enclosing_radius)
        return max(float(value), 1e-6)


def _collision_shapes(model: RobotModel) -> tuple[list[_CollisionShape], dict[str, str]]:
    shapes, sources = [], {}
    for link in model.xml.findall("link"):
        name = link.attrib["name"]
        source = "collision" if link.findall("collision") else "visual"
        entries = link.findall(source)
        if not entries:
            continue
        for entry in entries:
            geometry = entry.find("geometry")
            if geometry is None:
                raise ValueError(f"Missing {source} geometry on {name}")
            mesh = None
            approximation = None
            if len(geometry) == 1 and geometry[0].tag == "mesh":
                import trimesh
                element = geometry[0]
                mesh = trimesh.load(str(resolve_mesh_path(element.attrib["filename"], model.urdf_path)),
                                    force="mesh", process=True)
                mesh.vertices *= _vector(element.get("scale"), (1, 1, 1))
                if not len(mesh.vertices) or not np.isfinite(mesh.vertices).all():
                    raise ValueError(f"Invalid mesh geometry on {name}")
                bounds = mesh.bounds.copy()
                # Open or inconsistently oriented source meshes cannot support
                # a trustworthy winding test. Their convex hull still encloses
                # every authored triangle (each triangle is a convex combination
                # of its vertices), without the empty corners of a full AABB.
                # Degenerate/invalid hulls retain the full AABB as a safe fallback.
                if not mesh.is_watertight or not mesh.is_winding_consistent:
                    approximation = {
                        "link": name, "source": source,
                        "mesh_path": str(resolve_mesh_path(element.attrib["filename"], model.urdf_path)),
                        "source_watertight": bool(mesh.is_watertight),
                        "source_winding_consistent": bool(mesh.is_winding_consistent),
                        "source_vertex_count": len(mesh.vertices),
                    }
                    try:
                        hull = mesh.convex_hull
                        if not (hull.is_watertight and hull.is_winding_consistent and hull.is_volume):
                            raise ValueError("Convex hull is not a closed oriented volume")
                        signed = np.einsum(
                            "vfi,fi->vf", mesh.vertices[:, None] - hull.triangles[:, 0], hull.face_normals,
                        )
                        maximum_outside = float(signed.max())
                        tolerance = max(1e-12, float(np.ptp(mesh.vertices, axis=0).max()) * 1e-10)
                        if maximum_outside > tolerance:
                            raise ValueError(f"Convex hull excludes source vertices by {maximum_outside} m")
                        approximation.update(
                            method="convex_hull", source_max_outside_m=maximum_outside,
                            verification_tolerance_m=tolerance,
                        )
                        mesh = hull
                    except (ValueError, RuntimeError) as exc:
                        approximation.update(method="full_aabb", reason=str(exc).splitlines()[0])
                        mesh = None
            else:
                bounds = _shape_bounds(geometry, model.urdf_path)
            shapes.append(_CollisionShape(name, _origin(entry.find("origin")), bounds, mesh, approximation))
        sources[name] = source
    return shapes, sources


def _solid_contains(mesh: Any, points: np.ndarray) -> np.ndarray:
    """Float64 winding test for an oriented closed mesh, without optional rays.

    Trimesh silently selects Embree when installed. Its float32 ray offsets can
    jump across the millimetre-scale shells in this robot and classify cavities
    as solid (or the reverse). The signed solid angle uses the original triangles
    directly and is independent of optional acceleration libraries or RNG state.
    Callers separately retain points within the surface-distance tolerance.
    """
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    inside = np.empty(len(points), dtype=bool)
    for start in range(0, len(points), 64):
        vectors = triangles[None] - points[start:start + 64, None, None]
        first, second, third = vectors[:, :, 0], vectors[:, :, 1], vectors[:, :, 2]
        lengths = np.linalg.norm(vectors, axis=-1)
        numerator = np.einsum("pfi,pfi->pf", first, np.cross(second, third))
        denominator = np.prod(lengths, axis=-1)
        denominator += np.einsum("pfi,pfi->pf", first, second) * lengths[:, :, 2]
        denominator += np.einsum("pfi,pfi->pf", second, third) * lengths[:, :, 0]
        denominator += np.einsum("pfi,pfi->pf", third, first) * lengths[:, :, 1]
        solid_angle = 2 * np.arctan2(numerator, denominator).sum(axis=1)
        inside[start:start + 64] = np.abs(solid_angle) > 2 * np.pi
    return inside


def _occupied_cell(shape: _CollisionShape, lower: np.ndarray, upper: np.ndarray) -> _CollisionCell | None:
    """Bound the entire closed-solid/cell intersection, including its interior.

    An occupied polyhedral cell has extrema at clipped triangle vertices or at
    cube corners lying inside the solid. Bounding both sets therefore covers the
    volume, unlike a surface-only fit. Open meshes and primitives retain AABBs.
    """
    enclosing_radius = None
    if shape.mesh is not None:
        import trimesh
        mesh = shape.mesh
        midpoint = (lower + upper) / 2
        radius = np.linalg.norm(upper - lower) / 2
        _, distance, _ = trimesh.proximity.closest_point(mesh, midpoint[None])
        if distance[0] > radius + 1e-8 and not _solid_contains(mesh, midpoint[None])[0]:
            return None
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        for axis in range(3):
            normal = np.eye(3)[axis]
            vertices, faces, _ = trimesh.intersections.slice_faces_plane(vertices, faces, normal, lower)
            vertices, faces, _ = trimesh.intersections.slice_faces_plane(vertices, faces, -normal, upper)
            if not len(faces):
                break
        corners = np.array(list(product(*zip(lower, upper))))
        _, corner_distance, _ = trimesh.proximity.closest_point(mesh, corners)
        interior = corners[_solid_contains(mesh, corners) | (corner_distance <= 1e-8)]
        surface = vertices[np.unique(faces)].reshape(-1, 3)
        points = np.vstack([surface, interior])
        if not len(points):
            return None
        # Outward tolerance covers arithmetic at slice planes and mesh surfaces.
        lower = np.maximum(lower, points.min(axis=0) - 1e-8)
        upper = np.minimum(upper, points.max(axis=0) + 1e-8)
        # The clipped triangle vertices and contained cell corners include the
        # vertices of every polyhedral intersection component. A convex sphere
        # containing all of them contains its complete solid volume, including
        # nonconvex meshes. Keep the cell bounds for later bisection, but omit
        # empty AABB corners from this radius. No authored geometry is shrunk.
        midpoint = (lower + upper) / 2
        enclosing_radius = float(np.linalg.norm(points - midpoint, axis=1).max()) + 2e-8
    return _CollisionCell(shape, lower, upper, enclosing_radius)


def _grid_cells(shapes: list[_CollisionShape], cell_size: float) -> list[_CollisionCell]:
    if not math.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("cell_size must be positive and finite")
    cells = []
    for shape in shapes:
        extent = np.ptp(shape.bounds, axis=0)
        counts = np.maximum(1, np.ceil(extent / cell_size).astype(int))
        if math.prod(int(n) for n in counts) > 100_000:
            raise ValueError("Collision geometry requires over 100,000 cells; check mesh units")
        steps = extent / counts
        for index in product(*(range(int(n)) for n in counts)):
            lower = shape.bounds[0] + np.array(index) * steps
            cell = _occupied_cell(shape, lower, lower + steps)
            if cell is not None:
                cells.append(cell)
    return cells


def _cells_to_spheres(cells: list[_CollisionCell]) -> dict[str, list[dict[str, Any]]]:
    spheres: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        spheres.setdefault(cell.shape.link, []).append({"center": cell.center.tolist(), "radius": cell.radius})
    return spheres


def build_collision_spheres(
    model: RobotModel, cell_size: float = 0.04,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Cover geometry without a snapshot; build_robot_config also enforces GPU limits."""
    shapes, sources = _collision_shapes(model)
    spheres = _cells_to_spheres(_grid_cells(shapes, cell_size))
    return spheres, sources


def _self_collision_ignore(model: RobotModel, active: tuple[str, ...], links: list[str]) -> dict[str, list[str]]:
    # Links connected only by locked/fixed joints cannot move relative to one
    # another. Ignore those pairs, plus actual immediate parent/child links.
    components = {name: name for name in model.link_names}
    def root(name: str) -> str:
        while components[name] != name:
            name = components[name]
        return name
    adjacent = set()
    for joint in model.joints.values():
        adjacent.add(frozenset((joint.parent, joint.child)))
        if joint.name not in active:
            components[root(joint.child)] = root(joint.parent)
    result = {name: [] for name in links}
    for i, first in enumerate(links):
        for second in links[i + 1:]:
            if root(first) == root(second) or frozenset((first, second)) in adjacent:
                result[first].append(second)
    return result


def _cell_conflicts(
    cells: list[_CollisionCell], transforms: dict[str, np.ndarray],
    ignore: dict[str, list[str]], buffer: float, clearance: float,
) -> tuple[dict[int, float], list[tuple[str, str, float]], float]:
    groups: dict[str, list[tuple[int, np.ndarray, float]]] = {}
    for index, cell in enumerate(cells):
        transform = transforms[cell.shape.link]
        center = transform[:3, :3] @ cell.center + transform[:3, 3]
        groups.setdefault(cell.shape.link, []).append((index, center, cell.radius + buffer))
    arrays = {name: (np.array([s[0] for s in group]), np.array([s[1] for s in group]),
                     np.array([s[2] for s in group])) for name, group in groups.items()}
    candidates, overlaps, minimum = {}, [], math.inf
    names = list(groups)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            if second in ignore.get(first, []) or first in ignore.get(second, []):
                continue
            first_indices, first_centers, first_radii = arrays[first]
            second_indices, second_centers, second_radii = arrays[second]
            distance = np.linalg.norm(first_centers[:, None] - second_centers[None], axis=-1)
            distance -= first_radii[:, None] + second_radii[None]
            gap = float(distance.min())
            minimum = min(minimum, gap)
            if gap >= clearance:
                continue
            overlaps.append((first, second, gap))
            for a, b in zip(*np.where(distance < clearance)):
                index = int(first_indices[a] if first_radii[a] >= second_radii[b] else second_indices[b])
                candidates[index] = min(candidates.get(index, math.inf), float(distance[a, b]))
    return candidates, sorted(overlaps, key=lambda pair: pair[2]), minimum


def _adaptive_collision_spheres(
    model: RobotModel, snapshot: dict[str, float], active: tuple[str, ...],
    cell_size: float, buffer: float, max_spheres: int,
    validation_joint_positions: Sequence[Mapping[str, float]] = (),
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str], dict[str, Any]]:
    if not isinstance(max_spheres, int) or not 1 <= max_spheres <= 1023:
        raise ValueError("cuRobo 0.7.x requires max_spheres between 1 and 1023")
    if not math.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("cell_size must be positive and finite")
    shapes, sources = _collision_shapes(model)
    if len(shapes) > max_spheres:
        raise ValueError("Sphere budget is smaller than the number of robot geometries")
    # Start with one conservative enclosing sphere for each authored geometry.
    # The old uniform initial grid spent budget on distant locked feet/legs.
    # Refine only pairs that need clearance at a supplied fitting pose; no
    # geometry or self-collision pair is removed to meet the native limit.
    cells = [_occupied_cell(shape, shape.bounds[0].copy(), shape.bounds[1].copy()) for shape in shapes]
    cells = [cell for cell in cells if cell is not None]
    initial_count = len(cells)
    fit_poses = [snapshot, *validation_joint_positions]
    all_transforms = [model.forward_kinematics(pose) for pose in fit_poses]
    ignore = _self_collision_ignore(model, active, list(sources))
    target_clearance = 0.0001  # Avoid a zero-gap start under GPU float32 arithmetic.
    iterations = 0
    while True:
        candidates, overlaps, minimum = {}, [], math.inf
        pose_clearances = []
        for transforms in all_transforms:
            local, collisions, value = _cell_conflicts(cells, transforms, ignore, buffer, target_clearance)
            for index, gap in local.items():
                candidates[index] = min(candidates.get(index, math.inf), gap)
            overlaps.extend(collisions)
            minimum = min(minimum, value)
            pose_clearances.append(value)
        overlaps.sort(key=lambda pair: pair[2])
        if not candidates:
            break
        if iterations >= 8 * max_spheres:
            raise ValueError(f"Collision refinement did not converge: {overlaps[:3]}")
        selected = sorted(candidates, key=candidates.get)[:5]
        replacements = []
        for index in selected:
            cell = cells[index]
            axis = int(np.argmax(cell.upper - cell.lower))
            if cell.upper[axis] - cell.lower[axis] < 0.00005:
                raise ValueError(f"Snapshot has unresolved geometric self-collision: {overlaps[:3]}")
            midpoint = (cell.lower[axis] + cell.upper[axis]) / 2
            for upper_half in (False, True):
                lower, upper = cell.lower.copy(), cell.upper.copy()
                if upper_half:
                    lower[axis] = midpoint
                else:
                    upper[axis] = midpoint
                child = _occupied_cell(cell.shape, lower, upper)
                if child is not None:
                    replacements.append(child)
        updated_count = len(cells) - len(selected) + len(replacements)
        if updated_count > max_spheres:
            raise ValueError(
                f"Conservative collision cover exceeds {max_spheres} spheres before all fit poses are clear; "
                f"remaining pairs: {overlaps[:3]}. No collision links were disabled."
            )
        selected_set = set(selected)
        cells = [cell for index, cell in enumerate(cells) if index not in selected_set] + replacements
        iterations += 1
    spheres = _cells_to_spheres(cells)
    if set(spheres) != set(sources):
        raise ValueError("Collision fitting unexpectedly removed an entire physical link")
    metadata = {
        "method": ("occupied_mesh_cell_cover_with_multi_pose_refinement" if len(fit_poses) > 1
                   else "occupied_mesh_cell_cover_with_snapshot_adaptive_refinement"),
        "volume_coverage": "conservative; closed oriented meshes clipped to cells; open/inconsistent meshes use a verified enclosing convex hull, or full AABB if degenerate",
        "mesh_approximations": [shape.approximation for shape in shapes if shape.approximation is not None],
        "mesh_inside_test": "float64_signed_solid_angle; independent of optional Embree ray backend",
        "sphere_bound": "AABB midpoint; maximum distance to all clipped vertices and contained cell corners plus 2e-8 m numerical padding",
        "self_sphere_buffer_m": buffer,
        "base_world_sphere_buffer_m": buffer,
        "initialization": "one_enclosing_sphere_per_authored_geometry_then_conflict_refinement",
        "initial_cell_size_m": None,
        "legacy_requested_cell_size_m": cell_size,
        "initial_sphere_count": initial_count,
        "sphere_count": len(cells), "max_spheres": max_spheres,
        "refinement_iterations": iterations,
        "snapshot_min_self_clearance_m": pose_clearances[0],
        "fit_min_self_clearance_m": minimum,
        "fit_pose_count": len(fit_poses),
        "fit_poses": [{"role": "measured_start" if index == 0 else "candidate_only",
                       "active_joint_positions": {name: pose[name] for name in active},
                       "minimum_self_clearance_m": pose_clearances[index]}
                      for index, pose in enumerate(fit_poses)],
        "fit_scope": "Geometry is conservatively covered at all poses; start/candidate clearance checks do not certify an interpolated path or physical execution.",
        "snapshot_target_self_clearance_m": target_clearance,
        "extra_physical_adjacency_exclusions": [],
        "visual_fallback_links": [name for name, source in sources.items() if source == "visual"],
    }
    return spheres, sources, metadata


def planning_joint_limits(model: RobotModel, arm: str, soft_limit_factor: float = 0.97) -> dict[str, tuple[float, float]]:
    """The same URDF/soft/controller intersection for CPU IK and cuRobo."""
    if not math.isfinite(soft_limit_factor) or not 0 < soft_limit_factor <= 1:
        raise ValueError("soft_limit_factor must be in (0, 1]")
    command_ranges = [(-3.0, 1.0), (-0.15, 3.0) if arm == "left" else (-3.0, 0.15),
                      (-1.5, 1.5), (-2.1, 0.0)]
    limits = {}
    for index, name in enumerate(model.arm_joint_names(arm)):
        lower, upper = model.joint_limits[name]
        midpoint, half_range = (lower + upper) / 2, (upper - lower) * soft_limit_factor / 2
        lower, upper = midpoint - half_range, midpoint + half_range
        if index < len(command_ranges):
            lower, upper = max(lower, command_ranges[index][0]), min(upper, command_ranges[index][1])
        if lower >= upper:
            raise ValueError(f"No usable planning range for {name}")
        limits[name] = (lower, upper)
    return limits


def build_robot_config(
    model: RobotModel,
    joint_positions: Mapping[str, float],
    arm: str,
    output_dir: str | Path,
    *,
    sphere_cell_size: float = 0.04,
    sphere_buffer: float = 0.002,
    soft_limit_factor: float = 0.97,
    max_joint_velocity: float = 1.0,
    max_spheres: int = 1023,
    validation_joint_positions: Sequence[Mapping[str, float]] | None = None,
) -> RobotModelConfig:
    """Build a snapshot-specific, single-arm cuRobo 0.7.x robot_cfg.

    A derived URDF rewrites mesh paths and narrows active joint limits to the
    intersection of URDF, soft limits and HugWBC command ranges. The source URDF
    is never changed. Caller must use the same frozen non-arm snapshot during
    planning and invalidate/replan if the torso or opposite hand moves.
    Optional full-joint validation snapshots may vary only the active arm. They
    guide geometry refinement, never add exclusions or certify a motion path.
    """
    if model.root_link != "base_link":
        raise ValueError("IGRIS planning requires root link 'base_link'")
    if not 0 < soft_limit_factor <= 1 or not math.isfinite(soft_limit_factor):
        raise ValueError("soft_limit_factor must be in (0, 1]")
    if not math.isfinite(sphere_buffer) or sphere_buffer < 0:
        raise ValueError("sphere_buffer must be finite and nonnegative")
    if not math.isfinite(max_joint_velocity) or max_joint_velocity <= 0:
        raise ValueError("max_joint_velocity must be positive and finite")
    active = model.arm_joint_names(arm)
    ee_link = model.default_ee_link(arm)
    snapshot = model.validate_snapshot(joint_positions)
    locked = {name: value for name, value in snapshot.items() if name not in active}
    limits = planning_joint_limits(model, arm, soft_limit_factor)
    for name, (lower, upper) in limits.items():
        if not lower <= snapshot[name] <= upper:
            raise ValueError(f"Snapshot joint {name}={snapshot[name]} exceeds planning limits [{lower}, {upper}]")
    validation = []
    for index, positions in enumerate(validation_joint_positions or ()):
        pose = model.validate_snapshot(positions)
        changed = [name for name, value in locked.items() if abs(pose[name] - value) > 1e-9]
        if changed:
            raise ValueError(f"Validation pose {index} changes locked joints: {changed}")
        for name, (lower, upper) in limits.items():
            if not lower <= pose[name] <= upper:
                raise ValueError(f"Validation pose {index} joint {name} exceeds planning limits")
        validation.append(pose)
    spheres, sources, collision_metadata = _adaptive_collision_spheres(
        model, snapshot, active, sphere_cell_size, sphere_buffer, max_spheres, validation,
    )
    xml = copy.deepcopy(model.xml)
    for mesh in xml.findall(".//mesh"):
        mesh.set("filename", str(resolve_mesh_path(mesh.attrib["filename"], model.urdf_path)))
    for element in xml.findall("joint"):
        name = element.attrib["name"]
        if name in active:
            limit = element.find("limit")
            limit.set("lower", str(limits[name][0]))
            limit.set("upper", str(limits[name][1]))
            limit.set("velocity", str(min(float(limit.attrib["velocity"]), max_joint_velocity)))
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = output_dir / f"igris_c_{arm}_planning.urdf"
    if generated == model.urdf_path:
        raise ValueError("Refusing to overwrite the authored URDF")
    ET.indent(xml)
    ET.ElementTree(xml).write(generated, encoding="utf-8", xml_declaration=True)
    collision_links = list(spheres)
    names = list(model.joint_names)
    config = {"kinematics": {
        "use_usd_kinematics": False,
        "urdf_path": str(generated), "asset_root_path": str(model.urdf_path.parent),
        "base_link": model.root_link, "ee_link": ee_link,
        "link_names": list(model.link_names),
        "collision_link_names": collision_links, "collision_spheres": spheres,
        "collision_sphere_buffer": sphere_buffer,
        "self_collision_ignore": _self_collision_ignore(model, active, collision_links),
        # cuRobo subtracts collision_sphere_buffer when constructing its self
        # offsets. Set this explicitly so GPU self margins match CPU fitting.
        "self_collision_buffer": {name: sphere_buffer for name in collision_links},
        "lock_joints": locked,
        "use_global_cumul": True,
        "cspace": {
            "joint_names": names, "retract_config": [snapshot[name] for name in names],
            "null_space_weight": [1.0] * len(names),
            "cspace_distance_weight": [1.0] * len(names),
            "max_acceleration": 3.0, "max_jerk": 30.0,
        },
    }}
    return RobotModelConfig(config, active, locked, ee_link, generated, spheres, limits, sources, collision_metadata)
