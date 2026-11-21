"""Ballistic lead calculations, trajectory solving, and trigger discipline."""
import math
from typing import List, Optional, Tuple

from geometry import angle_diff, vec3_to_aim

Vec2 = List[float]
Vec3 = List[float]


def solve_gun_lead(
    muzzle_pos: Vec3,
    bullet_speed: float,
    target_pos: Vec3,
    target_vel: Vec3,
    max_range: float = 230.0
) -> Optional[Tuple[float, Vec3, float, float]]:
    """Solves exact quadratic ballistic intercept for moving target.
    Returns (t_intercept, intercept_point, desired_yaw, desired_pitch) or None."""
    # Target center of mass is pos.z + 4.5
    tx, ty, tz = target_pos[0], target_pos[1], target_pos[2] + 4.5
    vx, vy, vz = target_vel[0], target_vel[1], target_vel[2]
    mx, my, mz = muzzle_pos[0], muzzle_pos[1], muzzle_pos[2]

    dx = tx - mx
    dy = ty - my
    dz = tz - mz

    # Quadratic equation: |(P_t - P_m) + V_t * t|^2 = (speed * t)^2
    # a * t^2 + b * t + c = 0
    a = (vx * vx + vy * vy + vz * vz) - (bullet_speed * bullet_speed)
    b = 2.0 * (dx * vx + dy * vy + dz * vz)
    c = dx * dx + dy * dy + dz * dz

    dist_init = math.sqrt(c)
    if dist_init > max_range:
        return None

    disc = b * b - 4.0 * a * c
    if disc < 0:
        return None

    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    t = None
    if t1 > 0 and t2 > 0:
        t = min(t1, t2)
    elif t1 > 0:
        t = t1
    elif t2 > 0:
        t = t2
    else:
        return None

    # Max flight time for gun range
    if t * bullet_speed > max_range:
        return None

    ix = tx + vx * t
    iy = ty + vy * t
    iz = tz + vz * t
    intercept_pt = [ix, iy, iz]

    yaw, pitch = vec3_to_aim(ix - mx, iy - my, iz - mz)
    return t, intercept_pt, yaw, pitch


def solve_mortar_lead(
    tube_pos: Vec3,
    mortar_speed: float,
    gravity: float,
    target_pos: Vec3,
    target_vel: Vec3,
    high_arc: bool = False
) -> Optional[Tuple[float, float, float]]:
    """Iteratively solves mortar ballistics for moving target.
    Returns (yaw, pitch, flight_time) or None."""
    tx, ty, tz = target_pos[0], target_pos[1], target_pos[2] + 2.0
    vx, vy, vz = target_vel[0], target_vel[1], target_vel[2]
    mx, my, mz = tube_pos[0], tube_pos[1], tube_pos[2]

    # Iterative refinement over flight time t
    t = math.hypot(tx - mx, ty - my) / max(1.0, mortar_speed)
    v2 = mortar_speed * mortar_speed
    v4 = v2 * v2

    for _ in range(4):
        # Predict target position at time t
        cur_tx = tx + vx * t
        cur_ty = ty + vy * t
        cur_tz = tz + vz * t

        dx = cur_tx - mx
        dy = cur_ty - my
        dz = cur_tz - mz
        d_xy = math.hypot(dx, dy)
        if d_xy < 1e-3:
            return None

        under_root = v4 - gravity * (gravity * d_xy * d_xy + 2.0 * dz * v2)
        if under_root < 0:
            return None  # Out of range

        root = math.sqrt(under_root)
        if high_arc:
            tan_theta = (v2 + root) / (gravity * d_xy)
        else:
            tan_theta = (v2 - root) / (gravity * d_xy)

        pitch = math.atan(tan_theta)
        yaw = math.atan2(dy, dx)
        v_horiz = mortar_speed * math.cos(pitch)
        if v_horiz <= 1e-3:
            return None
        t = d_xy / v_horiz

    return yaw, pitch, t


def solve_laser_aim(muzzle_pos: Vec3, target_pos: Vec3) -> Tuple[float, float]:
    """Hitscan aim directly at target center of mass."""
    dx = target_pos[0] - muzzle_pos[0]
    dy = target_pos[1] - muzzle_pos[1]
    dz = (target_pos[2] + 4.5) - muzzle_pos[2]
    return vec3_to_aim(dx, dy, dz)


def is_in_lightning_cone(
    actual_aim: Vec2,
    my_pos: Vec3,
    target_pos: Vec3,
    cone_deg: float = 30.0,
    range_m: float = 45.0
) -> bool:
    """Checks if target is within lightning range and cone of actual aim."""
    dx = target_pos[0] - my_pos[0]
    dy = target_pos[1] - my_pos[1]
    dz = (target_pos[2] + 4.5) - (my_pos[2] + 7.0)
    dist = math.hypot(dx, dy, dz)
    if dist > range_m:
        return False
    target_yaw = math.atan2(dy, dx)
    y_diff = abs(angle_diff(target_yaw, actual_aim[0]))
    cone_rad = math.radians(cone_deg * 0.5)
    return y_diff <= cone_rad


def check_trigger_discipline(
    actual_aim: Vec2,
    desired_yaw: float,
    desired_pitch: float,
    target_dist: float,
    target_radius: float = 2.5,
    spread_deg: float = 0.6
) -> bool:
    """Trigger discipline gatekeeper: only fire if actual torso aim is within hitting tolerance."""
    yaw_err = abs(angle_diff(desired_yaw, actual_aim[0]))
    pitch_err = abs(desired_pitch - actual_aim[1])
    total_err = math.hypot(yaw_err, pitch_err)

    # Angular radius of target cylinder at distance
    target_angular_radius = math.atan2(target_radius, max(1.0, target_dist))
    tolerance = target_angular_radius + math.radians(spread_deg)
    return total_err <= tolerance
