"""ApexMech: World #1 Autonomous Combat Agent for Mech Royale (Ohio State Buildathon)."""
import math
import sys
from typing import Any, Dict, List, Optional, Tuple

from arena_map import ArenaMap
from ballistics import (
    check_trigger_discipline,
    is_in_lightning_cone,
    solve_gun_lead,
    solve_laser_aim,
    solve_mortar_lead,
)
from geometry import aim_to_vec3, angle_diff, v2_dist, v2_norm, v3_dist, vec3_to_aim
import mechfight as mf
from tactics import (
    CombatTactics,
    find_weapon_take_action,
    select_best_upgrade,
)
from tracker import Tracker


class ApexBot:
    name = "ApexMech-v1"

    def __init__(self, hello: mf.Hello):
        cfg = hello.get("config", {})
        self.mech_cfg = cfg.get("mech", {})
        self.loot_cfg = cfg.get("loot", {})
        self.weapons_cfg = cfg.get("weapons", {})
        self.pickup_radius = self.loot_cfg.get("pickup_radius", 4.0)

        # Initialize spatial systems
        map_dir = hello.get("map", {}).get("dir", "arena")
        self.map = ArenaMap(map_dir)
        self.tracker = Tracker()
        self.combat = CombatTactics()

        # Navigation state
        self.current_path: List[List[float]] = []
        self.last_path_t: float = -999.0
        self.stuck_ticks: int = 0
        self.prev_pos: Optional[List[float]] = None

    def act(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        my_pos = me["pos"]
        my_hp = me["hp"]
        my_aim = me["aim"]  # Actual torso aim [yaw, pitch]
        grounded = me.get("grounded", True)
        in_zone = me.get("in_zone", True)
        tick = obs["tick"]
        t = obs.get("t", 0.0)

        # Update perception
        self.tracker.update(obs)

        # Action template echoing observation tick
        act: mf.Action = {"tick": tick}

        # ---------------------------------------------------------
        # 1. UPGRADE RESOLUTION
        # ---------------------------------------------------------
        if obs.get("offer"):
            held_kinds = [w["kind"] for w in me["weapons"] if w]
            act["pick"] = select_best_upgrade(obs["offer"], held_kinds)

        # ---------------------------------------------------------
        # 2. WEAPON LOOTING
        # ---------------------------------------------------------
        take_act = find_weapon_take_action(
            obs.get("pickups", []), my_pos, me["weapons"], self.pickup_radius
        )
        if take_act:
            act.update(take_act)

        # ---------------------------------------------------------
        # 3. THREAT DETECTION & EMERGENCY DODGING
        # ---------------------------------------------------------
        dodge_res = self.tracker.get_dodge_vector(
            my_pos,
            self.mech_cfg.get("radius", 2.5),
            self.mech_cfg.get("height", 9.0),
            obs.get("projectiles", []),
        )
        laser_dodge = self.tracker.get_laser_threat_dodge(my_pos)

        emergency_move: Optional[List[float]] = None
        emergency_boost = False

        if dodge_res is not None:
            emergency_move, threat_kind = dodge_res
            emergency_boost = True
            mf.draw.label(f"DODGING {threat_kind.upper()}!")
        elif laser_dodge is not None:
            emergency_move = laser_dodge
            emergency_boost = True
            mf.draw.label("DODGING CHARGED LASER!")

        # ---------------------------------------------------------
        # 4. TARGET SELECTION & COMBAT
        # ---------------------------------------------------------
        enemy = self.tracker.get_highest_threat_enemy(my_pos, my_aim)

        # Zone navigation setup
        zone = obs.get("zone", {})
        next_center = zone.get("next_center", [0.0, 0.0])
        next_radius = zone.get("next_radius", 300.0)

        desired_aim = [my_aim[0], 0.0]
        fire_slots = [False, False]
        combat_move: Optional[List[float]] = None
        combat_boost = False

        # Find cover if needed
        cover_pt = None
        if enemy:
            cover_pt = self.map.find_cover(my_pos, enemy["pos"], next_center, next_radius)

        if enemy is not None:
            e_pos = enemy["pos"]
            e_vel = enemy.get("vel", [0.0, 0.0, 0.0])
            e_aim_yaw = enemy.get("aim", [0.0, 0.0])[0]
            dist_to_enemy = v3_dist(my_pos, e_pos)

            has_lightning = any(w and w.get("kind") == "lightning" for w in me["weapons"])
            combat_move, combat_boost = self.combat.get_combat_movement(
                my_pos, my_hp, e_pos, e_aim_yaw, 0.1, has_lightning, cover_pt is not None, cover_pt
            )

            # Solve weapons aiming
            for slot_idx, slot in enumerate(me["weapons"]):
                if slot is None:
                    continue
                w_kind = slot.get("kind")
                w_stats = slot.get("stats", {})
                cooldown = slot.get("cooldown", 0.0)

                if w_kind == "gun":
                    g_speed = w_stats.get("speed", 252.0)
                    lead = solve_gun_lead(
                        [my_pos[0], my_pos[1], my_pos[2] + 7.0],
                        g_speed,
                        e_pos,
                        e_vel,
                    )
                    if lead is not None:
                        t_lead, pt_lead, yaw_lead, pitch_lead = lead
                        desired_aim = [yaw_lead, pitch_lead]

                        # Check line of sight
                        if self.map.has_line_of_sight(
                            [my_pos[0], my_pos[1], my_pos[2] + 7.0], pt_lead
                        ):
                            # Trigger discipline
                            if check_trigger_discipline(
                                my_aim,
                                yaw_lead,
                                pitch_lead,
                                dist_to_enemy,
                                spread_deg=w_stats.get("spread_deg", 0.6),
                            ):
                                fire_slots[slot_idx] = True
                                mf.draw.line(
                                    [my_pos[0], my_pos[1], my_pos[2] + 7.0],
                                    pt_lead,
                                    "#ff3333",
                                )

                elif w_kind == "laser":
                    yaw_las, pitch_las = solve_laser_aim(
                        [my_pos[0], my_pos[1], my_pos[2] + 7.0], e_pos
                    )
                    desired_aim = [yaw_las, pitch_las]
                    charge = slot.get("charge", 0.0)
                    max_charge = w_stats.get("charge", 0.75)

                    # Charging cycle: hold trigger until max charge, then release to fire hitscan
                    if charge < max_charge - 0.05:
                        fire_slots[slot_idx] = True  # Hold to charge
                    else:
                        # Full charge reached! Release trigger to unleash 315+ damage hitscan beam
                        fire_slots[slot_idx] = False

                elif w_kind == "lightning":
                    l_cone = w_stats.get("cone_deg", 30.0)
                    l_range = w_stats.get("range", 45.0)
                    if is_in_lightning_cone(my_aim, my_pos, e_pos, l_cone, l_range):
                        if self.map.has_line_of_sight(
                            [my_pos[0], my_pos[1], my_pos[2] + 7.0],
                            [e_pos[0], e_pos[1], e_pos[2] + 4.5],
                        ):
                            fire_slots[slot_idx] = True

                elif w_kind == "mortar":
                    m_speed = w_stats.get("speed", 72.0)
                    m_grav = w_stats.get("gravity", 22.0)
                    m_lead = solve_mortar_lead(
                        [my_pos[0], my_pos[1], my_pos[2] + 9.5],
                        m_speed,
                        m_grav,
                        e_pos,
                        e_vel,
                        high_arc=dist_to_enemy < 60.0,
                    )
                    if m_lead is not None:
                        m_yaw, m_pitch, m_time = m_lead
                        desired_aim = [m_yaw, m_pitch]
                        if check_trigger_discipline(
                            my_aim, m_yaw, m_pitch, dist_to_enemy, target_radius=6.0
                        ):
                            fire_slots[slot_idx] = True

        else:
            # -----------------------------------------------------
            # NO ENEMY MECH: FARM CRATES / DRONES / SUPPLY DROPS
            # -----------------------------------------------------
            crates = obs.get("crates", [])
            cubes = obs.get("cubes", [])
            drops = obs.get("drops", [])

            # Gun reload optimization when safe
            gun_slot = next((s for s in me["weapons"] if s and s.get("kind") == "gun"), None)
            if gun_slot and gun_slot.get("ammo", 12) < 8 and gun_slot.get("reload", 0.0) == 0:
                act["reload"] = True

            # Check supply drops
            target_drop = None
            for d in drops:
                if d.get("lands_in", 999.0) == 0.0:
                    target_drop = d
                    break

            if target_drop and v2_dist(target_drop["pos"][:2], my_pos[:2]) < 60.0:
                d_pos = target_drop["pos"]
                combat_move = v2_norm([d_pos[0] - my_pos[0], d_pos[1] - my_pos[1]])
                desired_aim = [math.atan2(d_pos[1] - my_pos[1], d_pos[0] - my_pos[0]), 0.0]
            elif cubes and v2_dist(cubes[0]["pos"][:2], my_pos[:2]) < 30.0:
                c_pos = cubes[0]["pos"]
                combat_move = v2_norm([c_pos[0] - my_pos[0], c_pos[1] - my_pos[1]])
                desired_aim = [math.atan2(c_pos[1] - my_pos[1], c_pos[0] - my_pos[0]), 0.0]
            elif crates and v2_dist(crates[0]["pos"][:2], my_pos[:2]) < 50.0:
                crate = crates[0]
                cr_pos = crate["pos"]
                yaw_cr, pitch_cr = vec3_to_aim(
                    cr_pos[0] - my_pos[0],
                    cr_pos[1] - my_pos[1],
                    cr_pos[2] + 1.0 - (my_pos[2] + 7.0),
                )
                desired_aim = [yaw_cr, pitch_cr]
                if check_trigger_discipline(my_aim, yaw_cr, pitch_cr, v3_dist(my_pos, cr_pos)):
                    fire_slots[0] = True
            else:
                # 1. Protect rear armor if recently hit by unseen attacker
                damaged_ev = next(
                    (e for e in obs.get("events", []) if e.get("type") == "damaged" and e.get("bearing") is not None),
                    None
                )
                if damaged_ev:
                    desired_aim = [damaged_ev["bearing"], 0.0]
                else:
                    # 2. Audio Radar: scan toward closest heard sound
                    sounds = obs.get("sounds", [])
                    near_snd = next(
                        (s for s in sounds if s.get("dist") in ("near", "mid") and s.get("bearing") is not None),
                        None
                    )
                    if near_snd:
                        desired_aim = [near_snd["bearing"], 0.0]
                    else:
                        # 3. Look ahead toward safe zone center
                        desired_aim = [math.atan2(next_center[1] - my_pos[1], next_center[0] - my_pos[0]), 0.0]

        # ---------------------------------------------------------
        # 5. GENERAL MOVEMENT & PATHFINDING
        # ---------------------------------------------------------
        final_move = emergency_move if emergency_move is not None else combat_move
        final_boost = emergency_boost or combat_boost

        # If not dodging or fighting, navigate to safe zone center
        if final_move is None:
            dist_to_next = math.hypot(my_pos[0] - next_center[0], my_pos[1] - next_center[1])
            if dist_to_next > max(15.0, next_radius - 20.0) or not in_zone:
                # Recompute path periodically
                if t - self.last_path_t > 3.0 or not self.current_path:
                    self.current_path = self.map.find_path(my_pos[:2], next_center[:2])
                    self.last_path_t = t

                if self.current_path:
                    # Pop waypoints reached
                    while self.current_path and v2_dist(my_pos[:2], self.current_path[0]) < 4.0:
                        self.current_path.pop(0)

                    if self.current_path:
                        wp = self.current_path[0]
                        final_move = v2_norm([wp[0] - my_pos[0], wp[1] - my_pos[1]])
                    else:
                        final_move = v2_norm([next_center[0] - my_pos[0], next_center[1] - my_pos[1]])
                else:
                    final_move = v2_norm([next_center[0] - my_pos[0], next_center[1] - my_pos[1]])

                if not in_zone:
                    final_boost = True
            else:
                final_move = [0.0, 0.0]

        # Stuck detection & mantling (vault over ledges/obstacles)
        jump = False
        if self.prev_pos is not None:
            dist_moved = v2_dist(my_pos[:2], self.prev_pos[:2])
            if (final_move[0] != 0.0 or final_move[1] != 0.0) and dist_moved < 0.2:
                self.stuck_ticks += 1
                if self.stuck_ticks >= 3:
                    jump = grounded
                    final_boost = True
                    if self.stuck_ticks > 8:
                        self.stuck_ticks = 0
            else:
                self.stuck_ticks = 0
        self.prev_pos = my_pos[:2]

        # Energy conservation: reserve energy for dodging and close combat
        my_energy = me.get("energy", 100.0)
        if not emergency_boost and in_zone and my_energy < 25.0:
            final_boost = False

        act.update(
            move=final_move or [0.0, 0.0],
            aim=desired_aim,
            fire=fire_slots,
            boost=final_boost,
            jump=jump,
        )

        # Debug drawing
        mf.draw.label(f"HP {my_hp:.0f} · Cubes {me.get('cubes', 0)} · Threat: {enemy['id'] if enemy else 'None'}")
        if enemy:
            mf.draw.point(enemy["pos"], "#ff2020", 7)
        if cover_pt:
            mf.draw.point(cover_pt, "#20ff20", 5)

        return act


if __name__ == "__main__":
    mf.run(ApexBot)
