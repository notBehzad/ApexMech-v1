"""Geometry and 3D vector utilities for Mech Royale."""
import math
from typing import List, Optional, Tuple

Vec2 = List[float]
Vec3 = List[float]


def v2_dist(a: Vec2, b: Vec2) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def v3_dist(a: Vec3, b: Vec3) -> float:
    return math.dist(a, b)


def v2_norm(v: Vec2) -> Vec2:
    mag = math.hypot(v[0], v[1])
    if mag < 1e-6:
        return [0.0, 0.0]
    return [v[0] / mag, v[1] / mag]


def angle_diff(target_yaw: float, current_yaw: float) -> float:
    """Returns normalized angle difference in [-pi, pi]."""
    diff = (target_yaw - current_yaw + math.pi) % (2.0 * math.pi) - math.pi
    return diff


def aim_to_vec3(yaw: float, pitch: float) -> Vec3:
    """Converts (yaw, pitch) in radians to a 3D unit direction vector."""
    cp = math.cos(pitch)
    return [cp * math.cos(yaw), cp * math.sin(yaw), math.sin(pitch)]


def vec3_to_aim(dx: float, dy: float, dz: float) -> Tuple[float, float]:
    """Converts 3D vector to (yaw, pitch) in radians."""
    d_xy = math.hypot(dx, dy)
    yaw = math.atan2(dy, dx)
    pitch = math.atan2(dz, max(1e-4, d_xy))
    return yaw, pitch


def seg_intersect_2d(
    ax: float, ay: float, bx: float, by: float,
    cx: float, cy: float, dx: float, dy: float
) -> Optional[Tuple[float, float]]:
    """Checks intersection between segment AB and segment CD.
    Returns (t, u) parameters in [0, 1] if intersecting, None otherwise."""
    denom = (dx - cx) * (by - ay) - (dy - cy) * (bx - ax)
    if abs(denom) < 1e-9:
        return None
    num_t = (cx - ax) * (dy - cy) - (cy - ay) * (dx - cx)
    num_u = (cx - ax) * (by - ay) - (cy - ay) * (bx - ax)
    t = num_t / -denom
    u = num_u / -denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return t, u
    return None
