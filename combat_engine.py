"""Ballistics, intercept prediction, weapon selection, and trigger rules."""
import math
from typing import Any, Dict, List, Optional, Tuple


def solve_gun_intercept(
    muzzle_pos: List[float],
    bullet_speed: float,
    target_pos: List[float],
    target_vel: List[float],
    max_range: float = 180.0,
) -> Optional[Tuple[float, List[float], float, float]]:
    """Quadratic intercept solver for moving target.
    Targets center of mass (target.z + 4.5)."""
    tx, ty, tz = target_pos[0], target_pos[1], target_pos[2] + 4.5
    vx, vy, vz = target_vel[0], target_vel[1], target_vel[2]
    mx, my, mz = muzzle_pos[0], muzzle_pos[1], muzzle_pos[2]

    dx = tx - mx
    dy = ty - my
    dz = tz - mz

    init_dist = math.hypot(dx, dy, dz)
    if init_dist > max_range:
        return None

    # a * t^2 + b * t + c = 0
    a = (vx * vx + vy * vy + vz * vz) - (bullet_speed * bullet_speed)
    b = 2.0 * (dx * vx + dy * vy + dz * vz)
    c = dx * dx + dy * dy + dz * dz

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

    if t * bullet_speed > max_range:
        return None

    ix = tx + vx * t
    iy = ty + vy * t
    iz = tz + vz * t

    d_xy = math.hypot(ix - mx, iy - my)
    yaw = math.atan2(iy - my, ix - mx)
    pitch = math.atan2(iz - mz, max(1e-4, d_xy))
    return t, [ix, iy, iz], yaw, pitch


def solve_mortar_trajectory(
    tube_pos: List[float],
    speed: float,
    gravity: float,
    target_pos: List[float],
    target_vel: List[float],
    high_arc: bool = False,
) -> Optional[Tuple[float, float, float]]:
    """Iterative mortar trajectory solver for moving target."""
    tx, ty, tz = target_pos[0], target_pos[1], target_pos[2] + 2.0
    vx, vy, vz = target_vel[0], target_vel[1], target_vel[2]
    mx, my, mz = tube_pos[0], tube_pos[1], tube_pos[2]

    t = math.hypot(tx - mx, ty - my) / max(1.0, speed)
    v2 = speed * speed
    v4 = v2 * v2

    for _ in range(4):
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
            return None

        root = math.sqrt(under_root)
        tan_theta = (v2 + root if high_arc else v2 - root) / (gravity * d_xy)
        pitch = math.atan(tan_theta)
        yaw = math.atan2(dy, dx)
        v_horiz = speed * math.cos(pitch)
        if v_horiz <= 1e-3:
            return None
        t = d_xy / v_horiz

    return yaw, pitch, t


def can_hit_target(
    actual_aim: List[float],
    desired_yaw: float,
    desired_pitch: float,
    target_dist: float,
    target_radius: float = 2.5,
    spread_deg: float = 0.6,
) -> bool:
    """Trigger discipline gate: only fire when actual torso aim is within angular radius."""
    y_diff = abs(((desired_yaw - actual_aim[0] + math.pi) % (2.0 * math.pi)) - math.pi)
    p_diff = abs(desired_pitch - actual_aim[1])
    tot_err = math.hypot(y_diff, p_diff)

    angular_radius = math.atan2(target_radius, max(1.0, target_dist))
    return tot_err <= (angular_radius + math.radians(spread_deg))


def evaluate_engagement_permission(
    my_hp: float,
    my_max_hp: float,
    target_enemy,
    dist_to_target: float,
    enemies_with_los: int,
    has_cover: bool,
    was_damaged_recently: bool,
) -> bool:
    """Strict survival engagement rules: default to holding fire."""
    # Never fight if >= 2 enemies have line of sight
    if enemies_with_los >= 2:
        return False

    # Never fight if critically low on health (must prioritize disengage & regen)
    if (my_hp / my_max_hp) < 0.35:
        return False

    # Condition (a): Under active fire and cannot immediately break LOS
    if was_damaged_recently and not has_cover:
        return True

    # Check rear shot opportunity (target facing away by > 100 deg)
    yaw_to_target = math.atan2(target_enemy.pos[1], target_enemy.pos[0])
    target_facing_diff = abs(((target_enemy.aim[0] - yaw_to_target + math.pi) % (2.0 * math.pi)) - math.pi)
    is_rear_shot = target_facing_diff < math.radians(80.0)
    if is_rear_shot and dist_to_target < 80.0:
        return True

    # Condition (b): High HP, we have cover, target is weak (<40%) or distracted
    target_hp_ratio = target_enemy.hp / target_enemy.max_hp
    is_distracted = target_facing_diff < math.radians(90.0)
    if (my_hp / my_max_hp) > 0.50 and has_cover and (target_hp_ratio < 0.40 or is_distracted):
        return dist_to_target < 90.0

    # Farming isolated NPC drones
    if target_enemy.kind == "npc" and dist_to_target < 35.0:
        return True

    return False
