"""Perception tracking, threat assessment, and dodge calculations."""
import math
from typing import Dict, List, Optional, Tuple, Any

from geometry import angle_diff, v2_dist, v3_dist

Vec2 = List[float]
Vec3 = List[float]


class Tracker:
    def __init__(self):
        self.enemies: Dict[int, Dict[str, Any]] = {}
        self.last_tick: int = 0
        self.last_t: float = 0.0

    def update(self, obs: Dict[str, Any]):
        self.last_tick = obs.get("tick", 0)
        self.last_t = obs.get("t", 0.0)

        # Update visible mechs
        seen_ids = set()
        for m in obs.get("mechs", []):
            mid = m["id"]
            seen_ids.add(mid)
            m_pos = m["pos"]
            m_vel = m.get("vel", [0.0, 0.0, 0.0])
            m_aim = m.get("aim", [0.0, 0.0])
            m_hp = m.get("hp", 1400.0)
            m_max_hp = m.get("max_hp", 1400.0)
            m_weapons = m.get("weapons", ["gun"])
            m_charging = m.get("charging", False)
            m_kind = m.get("kind", "mech")

            self.enemies[mid] = {
                "id": mid,
                "kind": m_kind,
                "pos": m_pos,
                "vel": m_vel,
                "aim": m_aim,
                "hp": m_hp,
                "max_hp": m_max_hp,
                "weapons": m_weapons,
                "charging": m_charging,
                "last_seen_t": self.last_t,
                "last_seen_tick": self.last_tick,
            }

        # Purge memories older than 5.0 seconds
        stale_ids = [
            mid for mid, data in self.enemies.items()
            if self.last_t - data["last_seen_t"] > 5.0
        ]
        for mid in stale_ids:
            del self.enemies[mid]

    def get_dodge_vector(self, me_pos: Vec3, me_radius: float, me_height: float, projectiles: List[Dict[str, Any]]) -> Optional[Tuple[Vec2, str]]:
        """Calculates emergency dodge vector for incoming bullets or mortar shells.
        Returns (dodge_dir_2d, threat_type) or None."""
        mx, my, mz = me_pos[0], me_pos[1], me_pos[2]

        for p in projectiles:
            kind = p.get("kind", "gun")
            px, py, pz = p["pos"]
            vx, vy, vz = p["vel"]

            if kind in ("gun", "npc"):
                v_sq = vx * vx + vy * vy + vz * vz
                if v_sq < 100.0:
                    continue

                dx = mx - px
                dy = my - py
                dz = (mz + me_height * 0.5) - pz

                t_closest = (dx * vx + dy * vy + dz * vz) / v_sq
                if 0.0 < t_closest < 0.6:
                    cx = px + vx * t_closest
                    cy = py + vy * t_closest
                    cz = pz + vz * t_closest

                    d_xy = math.hypot(cx - mx, cy - my)
                    d_z = abs(cz - (mz + me_height * 0.5))

                    if d_xy < (me_radius + 0.8) and d_z < (me_height * 0.5 + 0.8):
                        # Bullet on collision course! Sidestep perpendicular to bullet velocity
                        norm_v = math.hypot(vx, vy)
                        if norm_v > 1e-3:
                            p1 = [-vy / norm_v, vx / norm_v]
                            p2 = [vy / norm_v, -vx / norm_v]
                            # Pick whichever moves us away from trajectory
                            rel_x = cx - mx
                            rel_y = cy - my
                            dot1 = p1[0] * rel_x + p1[1] * rel_y
                            dodge_dir = p2 if dot1 > 0 else p1
                            return dodge_dir, "bullet"

            elif kind == "mortar":
                # Solve quadratic for mortar landing time: pz + vz*t - 0.5*g*t^2 = mz
                g = 22.0
                disc = vz * vz - 4.0 * (-0.5 * g) * (pz - mz)
                if disc > 0:
                    t_impact = (-vz - math.sqrt(disc)) / (-g)
                    if 0.0 < t_impact < 4.0:
                        land_x = px + vx * t_impact
                        land_y = py + vy * t_impact
                        dist_to_blast = math.hypot(land_x - mx, land_y - my)
                        if dist_to_blast < 8.5:  # Inside 7m splash + radius
                            # Escape away from blast center!
                            diff_x = mx - land_x
                            diff_y = my - land_y
                            mag = math.hypot(diff_x, diff_y)
                            if mag < 1e-4:
                                escape_dir = [1.0, 0.0]
                            else:
                                escape_dir = [diff_x / mag, diff_y / mag]
                            return escape_dir, "mortar"

        return None

    def get_laser_threat_dodge(self, me_pos: Vec3) -> Optional[Vec2]:
        """Checks if an enemy is charging a laser and aiming towards us.
        Returns perpendicular dodge vector or None."""
        mx, my, mz = me_pos[0], me_pos[1], me_pos[2] + 4.5

        for mid, e in self.enemies.items():
            if not e.get("charging"):
                continue

            ex, ey, ez = e["pos"][0], e["pos"][1], e["pos"][2] + 7.0
            eyaw, epitch = e["aim"][0], e["aim"][1]

            # Vector to me
            dx = mx - ex
            dy = my - ey
            dz = mz - ez
            dist = math.hypot(dx, dy, dz)
            if dist > 280.0:
                continue

            # Laser beam direction
            cp = math.cos(epitch)
            lx = cp * math.cos(eyaw)
            ly = cp * math.sin(eyaw)
            lz = math.sin(epitch)

            # Distance from me to the laser beam ray
            t_proj = max(0.0, dx * lx + dy * ly + dz * lz)
            bx = ex + lx * t_proj
            by = ey + ly * t_proj
            bz = ez + lz * t_proj

            d_beam = math.hypot(bx - mx, by - my, bz - mz)
            if d_beam < 3.5:  # Laser is aimed directly at us!
                # Strafe perpendicular to laser horizontal direction
                norm_l = math.hypot(lx, ly)
                if norm_l > 1e-3:
                    p1 = [-ly / norm_l, lx / norm_l]
                    p2 = [ly / norm_l, -lx / norm_l]
                    rel_x = bx - mx
                    rel_y = by - my
                    dot1 = p1[0] * rel_x + p1[1] * rel_y
                    dodge = p2 if dot1 > 0 else p1
                    return dodge

        return None

    def get_highest_threat_enemy(self, me_pos: Vec3, me_aim: Vec2) -> Optional[Dict[str, Any]]:
        """Scores all visible/tracked enemies and returns the highest priority threat."""
        best_enemy = None
        best_score = -9999.0

        for mid, e in self.enemies.items():
            epos = e["pos"]
            dist = v3_dist(me_pos, epos)
            if dist > 200.0:
                continue

            # Scoring factors:
            # 1. Distance: closer = much higher threat
            dist_score = max(0.0, 200.0 - dist)

            # 2. Mech vs NPC: mechs are priority threats
            kind_score = 150.0 if e.get("kind") == "mech" else 10.0

            # 3. Low health: finish off weakened targets
            hp = e.get("hp", 1400.0)
            max_hp = e.get("max_hp", 1400.0)
            hp_score = max(0.0, (max_hp - hp) / max_hp * 80.0)

            # 4. Enemy aiming at us: active attacker
            yaw_to_me = math.atan2(me_pos[1] - epos[1], me_pos[0] - epos[0])
            enemy_facing_diff = abs(angle_diff(yaw_to_me, e["aim"][0]))
            aiming_at_us = enemy_facing_diff < math.radians(25.0)
            aim_score = 120.0 if aiming_at_us else 0.0

            # 5. Charging laser: immediate threat
            charge_score = 200.0 if e.get("charging") else 0.0

            # 6. Exposed rear: target is facing away from us (opportunity for 1.5x rear multiplier!)
            rear_exposed = enemy_facing_diff > math.radians(100.0)
            rear_score = 60.0 if rear_exposed else 0.0

            total_score = dist_score + kind_score + hp_score + aim_score + charge_score + rear_score

            if total_score > best_score:
                best_score = total_score
                best_enemy = e

        return best_enemy
