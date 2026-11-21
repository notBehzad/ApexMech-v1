"""High-level tactics: state machine, movement decisions, upgrade prioritization, and weapon management."""
import math
import random
from typing import Dict, List, Optional, Tuple, Any

from geometry import angle_diff, v2_dist, v2_norm, v3_dist

Vec2 = List[float]
Vec3 = List[float]

# Upgrade priority weighting
UPGRADE_WEIGHTS = {
    "armor": 120,             # Reactive armor: -8% damage taken (3 stacks = -24% total damage reduction)
    "nanites": 110,           # Repair nanites: +60% out-of-combat regen, delay -1.5s (full heal in seconds)
    "plating": 100,           # Composite plating: +15% max HP
    "servos": 90,             # Leg servos: +8% walk & boost speed
    "laser_damage": 85,       # Focusing lens: +18% laser damage
    "laser_charge": 85,       # Fast capacitors: -25% laser charge time
    "lightning_efficiency": 80,# Superconductor: -35% energy, +10% damage
    "lightning_range": 80,    # Tesla coil: +20% lightning range and cone
    "gun_rate": 75,           # Rotary feed: +15% arm gun fire rate
    "gun_damage": 75,         # Tungsten rounds: +15% arm gun damage
    "gun_velocity": 70,       # Rail barrel: +25% gun speed, +15% range
    "gun_mag": 70,            # Drum magazine: +8 rounds, -20% reload
    "mortar_splash": 65,      # Cluster shells: +30% splash radius
    "mortar_reload": 65,      # Autoloader: -18% mortar cooldown
    "capacitor": 60,          # Capacitor bank: +30 max energy
    "reactor": 55,            # Reactor tune: +30% energy regen
    "jets": 50,               # Efficient jets: -25% boost/jet drain
    "sensors": 45,            # Sensor suite: proximity & vision range
    "gyros": 40,              # Torso gyros: +30% turn rate
}


def score_upgrade_choice(choice: Dict[str, Any], held_weapons: List[str]) -> float:
    uid = choice.get("id", "")
    weight = UPGRADE_WEIGHTS.get(uid, 30.0)

    # Check weapon requirements
    if uid.startswith("laser_") and "laser" not in held_weapons:
        weight *= 0.3
    elif uid.startswith("lightning_") and "lightning" not in held_weapons:
        weight *= 0.3
    elif uid.startswith("mortar_") and "mortar" not in held_weapons:
        weight *= 0.3
    elif uid.startswith("gun_") and "gun" not in held_weapons:
        weight *= 0.3

    return float(weight)


def select_best_upgrade(offer: Dict[str, Any], held_weapons: List[str]) -> int:
    """Returns the index (0, 1, or 2) of the best upgrade choice."""
    choices = offer.get("choices", [])
    if not choices:
        return 0

    best_idx = 0
    best_score = -1.0
    for i, c in enumerate(choices):
        sc = score_upgrade_choice(c, held_weapons)
        if sc > best_score:
            best_score = sc
            best_idx = i

    return best_idx


def find_weapon_take_action(
    pickups: List[Dict[str, Any]],
    my_pos: Vec3,
    weapons: List[Any],
    reach: float
) -> Optional[Dict[str, Any]]:
    """Evaluates nearby weapon pickups and returns take/drop action if beneficial."""
    near_pickups = [
        p for p in pickups
        if v2_dist(p["pos"][:2], my_pos[:2]) < reach and abs(p["pos"][2] - my_pos[2]) < 4.0
    ]
    if not near_pickups:
        return None

    slot0 = weapons[0]
    slot1 = weapons[1]

    # Preference ranking: laser > lightning > mortar > gun
    tier = {"laser": 4, "lightning": 3, "mortar": 2, "gun": 1}

    # If slot 1 is empty, take highest tier pickup
    if slot1 is None:
        best_p = max(near_pickups, key=lambda p: tier.get(p.get("kind", ""), 0))
        if best_p.get("kind") != "gun":
            return {"take": best_p["id"]}

    # If both slots full, check if we can upgrade mortar to laser/lightning
    cur_kinds = [w.get("kind") for w in weapons if w]
    for p in near_pickups:
        p_kind = p.get("kind")
        if p_kind not in cur_kinds:
            if p_kind == "laser" and "mortar" in cur_kinds:
                drop_slot = 1 if cur_kinds[1] == "mortar" else 0
                return {"take": p["id"], "drop": drop_slot}
            elif p_kind == "lightning" and "mortar" in cur_kinds:
                drop_slot = 1 if cur_kinds[1] == "mortar" else 0
                return {"take": p["id"], "drop": drop_slot}

    return None


class CombatTactics:
    def __init__(self):
        self.strafe_sign = 1.0
        self.strafe_timer = 0.0

    def get_combat_movement(
        self,
        me_pos: Vec3,
        me_hp: float,
        enemy_pos: Vec3,
        enemy_aim_yaw: float,
        dt: float,
        has_lightning: bool,
        has_cover: bool,
        cover_pos: Optional[Vec3]
    ) -> Tuple[Vec2, bool]:
        """Calculates tactical combat movement: strafing, kiting, seeking cover, or flanking enemy rear."""
        # Strafe oscillation
        self.strafe_timer += dt
        if self.strafe_timer > 1.2:
            self.strafe_sign *= -1.0
            self.strafe_timer = 0.0

        # If HP is low (< 550) and cover exists, seek cover immediately
        if me_hp < 550.0 and cover_pos is not None:
            c_dx = cover_pos[0] - me_pos[0]
            c_dy = cover_pos[1] - me_pos[1]
            dist_c = math.hypot(c_dx, c_dy)
            if dist_c > 1.0:
                return [c_dx / dist_c, c_dy / dist_c], True

        dx = enemy_pos[0] - me_pos[0]
        dy = enemy_pos[1] - me_pos[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-4:
            return [1.0, 0.0], False

        fwd_x = dx / dist
        fwd_y = dy / dist

        # Tactical Flanking: determine which strafe direction gets closer to enemy's rear (aim_yaw + pi)
        # Enemy rear unit vector:
        rear_x = -math.cos(enemy_aim_yaw)
        rear_y = -math.sin(enemy_aim_yaw)

        # Candidate strafe directions: left and right
        strafe_left = [-fwd_y, fwd_x]
        strafe_right = [fwd_y, -fwd_x]

        dot_left = strafe_left[0] * rear_x + strafe_left[1] * rear_y
        dot_right = strafe_right[0] * rear_x + strafe_right[1] * rear_y

        # Prefer flanking toward their blind spot, mixed with oscillation
        flank_dir = strafe_left if dot_left > dot_right else strafe_right
        strafe_x = flank_dir[0] * 0.7 + (-fwd_y * self.strafe_sign) * 0.3
        strafe_y = flank_dir[1] * 0.7 + (fwd_x * self.strafe_sign) * 0.3

        # Distance management
        ideal_dist = 26.0 if has_lightning else 38.0
        if dist < ideal_dist - 8.0:
            # Backpedal while strafing
            move_x = -0.7 * fwd_x + 0.7 * strafe_x
            move_y = -0.7 * fwd_y + 0.7 * strafe_y
            should_boost = me_hp < 800.0
        elif dist > ideal_dist + 10.0:
            # Close in while strafing
            move_x = 0.8 * fwd_x + 0.5 * strafe_x
            move_y = 0.8 * fwd_y + 0.5 * strafe_y
            should_boost = False
        else:
            # Pure circle-strafe
            move_x = strafe_x
            move_y = strafe_y
            should_boost = False

        return v2_norm([move_x, move_y]), should_boost
