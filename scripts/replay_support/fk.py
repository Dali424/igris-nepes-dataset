"""fk: minimal URDF forward kinematics (numpy only) for the kinematic replay.

Link poses are returned in the URDF root frame (base_link). The trace's base_pose_w is the articulation root
(base_link == torso, fixed joint with identity origin), so  T_world_link = T(base_pose_w) @ fk[link].
Quaternions are (w, x, y, z) like the trace.
"""
import math
import xml.etree.ElementTree as ET

import numpy as np


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def axis_angle_R(axis, a):
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)


def quat_to_R(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def pose_to_T(p):
    T = np.eye(4)
    T[:3, :3] = quat_to_R(p[3:7])
    T[:3, 3] = p[:3]
    return T


def T_to_pose(T):
    return np.r_[T[:3, 3], R_to_quat(T[:3, :3])]


def T_err(Ta, Tb):
    dp = float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]))
    dR = Ta[:3, :3].T @ Tb[:3, :3]
    return dp, math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2))))


class URDF:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.joints = {}
        children = set()
        for j in root.findall('joint'):
            o = j.find('origin')
            xyz = [float(v) for v in (o.get('xyz', '0 0 0') if o is not None else '0 0 0').split()]
            rpy = [float(v) for v in (o.get('rpy', '0 0 0') if o is not None else '0 0 0').split()]
            ax = j.find('axis')
            axis = np.array([float(v) for v in ax.get('xyz').split()]) if ax is not None else np.array([1.0, 0, 0])
            T0 = np.eye(4)
            T0[:3, :3] = rpy_to_R(*rpy)
            T0[:3, 3] = xyz
            L = j.find('limit')
            self.joints[j.get('name')] = {'type': j.get('type'), 'parent': j.find('parent').get('link'),
                                          'child': j.find('child').get('link'), 'T0': T0, 'axis': axis,
                                          'limit': (float(L.get('lower')), float(L.get('upper')))
                                          if L is not None and L.get('lower') is not None else None}
            children.add(j.find('child').get('link'))
        self.links = [l.get('name') for l in root.findall('link')]
        roots = [l for l in self.links if l not in children]
        assert len(roots) == 1, roots
        self.root = roots[0]
        by_parent = {}
        for n, j in self.joints.items():
            by_parent.setdefault(j['parent'], []).append(n)
        # breadth-first joint order (parents before children)
        self.order = []
        frontier = [self.root]
        while frontier:
            nxt = []
            for link in frontier:
                for jn in by_parent.get(link, []):
                    self.order.append(jn)
                    nxt.append(self.joints[jn]['child'])
            frontier = nxt
        self.movable = [n for n in self.order if self.joints[n]['type'] in ('revolute', 'continuous', 'prismatic')]

    def fk(self, q):
        """q: mapping joint name -> position. Returns {link: 4x4 in root frame}."""
        T = {self.root: np.eye(4)}
        for jn in self.order:
            J = self.joints[jn]
            M = T[J['parent']] @ J['T0']
            if J['type'] in ('revolute', 'continuous'):
                R = np.eye(4)
                R[:3, :3] = axis_angle_R(J['axis'], float(q[jn]))
                M = M @ R
            elif J['type'] == 'prismatic':
                P = np.eye(4)
                P[:3, 3] = J['axis'] * float(q[jn])
                M = M @ P
            T[J['child']] = M
        return T
