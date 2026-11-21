"""Perception tracking, threat assessment, projectile evasion, and target scoring."""
import math
from typing import Any, Dict, List, Optional, Tuple


class TrackedEnemy:
    def __init__(self, m: Dict[str, Any], t: float):
        self.id: int = int(m["id"])
        self.kind: str = m.get("kind", "mech")
        self.pos: List[float] = list(m["pos"])
        self.vel: List[float] = list(m.get("vel", [0.0, 0.0, 0.0]))
        self.aim: List[float] = list(m.get("aim", [0.0, 0.0]))
        self.hp: float = float(m.get("hp", 1400.0))
        self.max_hp: float = float(m.get("max_hp", 1400.0))
        self.weapons: List[str] = list(m.get("weapons", ["gun"]))
        self.charging: bool = bool(m.get("charging", False))
        self.last_seen_t: float = t

    def update(self, m: Dict[str, Any], t: float):
        self.pos = list(m["pos"])
        self.vel = list(m.get("vel", self.vel))
        self.aim = list(m.get("aim", self.aim))
        if "hp" in m:
            self.hp = float(m["hp"])
        self.weapons = list(m.get("weapons", self.weapons))
        self.charging = bool(m.get("charging", False))
        self.last_seen_t = t

    def get_extrapolated_pos(self, current_t: float) -> List[float]:
        """Extrapolate position using velocity up to memory_s."""
        dt = max(0.0, current_t - self.last_seen_t)
        return [
            self.pos[0] + self.vel[0] * dt,
            self.pos[1] + self.vel[1] * dt,
            self.pos[2] + self.vel[2] * dt,
        ]


class Tracker:
    def __init__(self, memory_s: float = 5.0):
        self.memory_s = memory_s
        self.enemies: Dict[int, TrackedEnemy] = {}
        self.last_t: float = 0.0

    def update(self, obs: Dict[str, Any]):
        self.last_t = obs.get("t", 0.0)

        for m in obs.get("mechs", []):
            mid = int(m["id"])
            if mid in self.enemies:
                self.enemies[mid].update(m, self.last_t)
            else:
                self.enemies[mid] = TrackedEnemy(m, self.last_t)

        # Purge stale actors older than memory_s
        stale = [
            mid for mid, e in self.enemies.items()
            if self.last_t - e.last_seen_t > self.memory_s
        ]
        for mid in stale:
            del self.enemies[mid]

    def get_projectile_threat(
        self,
        my_pos: List[float],
        my_radius: float,
        my_height: float,
        projectiles: List[Dict[str, Any]],
    ) -> Optional[Tuple[List[float], str]]:
        """Evaluates incoming bullets and mortar shells for immediate collision course.
        Returns (dodge_dir_2d, threat_kind) or None."""
        mx, my, mz = my_pos[0], my_pos[1], my_pos[2]

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
                dz = (mz + my_height * 0.5) - pz

                t_closest = (dx * vx + dy * vy + dz * vz) / v_sq
                if 0.0 < t_closest < 0.6:  # Arriving in <= 600ms
                    cx = px + vx * t_closest
                    cy = py + vy * t_closest
                    cz = pz + vz * t_closest

                    d_xy = math.hypot(cx - mx, cy - my)
                    d_z = abs(cz - (mz + my_height * 0.5))

                    if d_xy < (my_radius + 0.8) and d_z < (my_height * 0.5 + 0.8):
                        # Perpendicular dodge vector
                        norm_v = math.hypot(vx, vy)
                        if norm_v > 1e-3:
                            p1 = [-vy / norm_v, vx / norm_v]
                            p2 = [vy / norm_v, -vx / norm_v]
                            rel_x = cx - mx
                            rel_y = cy - my
                            dot1 = p1[0] * rel_x + p1[1] * rel_y
                            dodge = p2 if dot1 > 0 else p1
                            return dodge, "bullet"

            elif kind == "mortar":
                # Solve quadratic for mortar ground landing time under g=22
                g = 22.0
                disc = vz * vz - 4.0 * (-0.5 * g) * (pz - mz)
                if disc > 0:
                    t_impact = (-vz - math.sqrt(disc)) / (-g)
                    if 0.0 < t_impact < 4.0:
                        land_x = px + vx * t_impact
                        land_y = py + vy * t_impact
                        dist_blast = math.hypot(land_x - mx, land_y - my)
                        if dist_blast < 8.5:  # Inside 7m splash + radius
                            diff_x = mx - land_x
                            diff_y = my - land_y
                            mag = math.hypot(diff_x, diff_y)
                            if mag < 1e-4:
                                escape = [1.0, 0.0]
                            else:
                                escape = [diff_x / mag, diff_y / mag]
                            return escape, "mortar"

        return None

    def get_laser_threat_dodge(self, my_pos: List[float]) -> Optional[List[float]]:
        """Checks if an enemy is charging a laser aiming near our mech.
        Uses 2D horizontal range (280m) to avoid false elevation aborts."""
        mx, my, mz = my_pos[0], my_pos[1], my_pos[2] + 4.5

        for mid, e in self.enemies.items():
            if not e.charging:
                continue

            ex, ey, ez = e.pos[0], e.pos[1], e.pos[2] + 7.0
            eyaw, epitch = e.aim[0], e.aim[1]

            # 2D horizontal distance to enemy
            dist_2d = math.hypot(mx - ex, my - ey)
            if dist_2d > 280.0:
                continue

            dx = mx - ex
            dy = my - ey
            dz = mz - ez

            # Laser beam direction vector
            cp = math.cos(epitch)
            lx = cp * math.cos(eyaw)
            ly = cp * math.sin(eyaw)
            lz = math.sin(epitch)

            # Closest point along beam to my center
            t_proj = max(0.0, dx * lx + dy * ly + dz * lz)
            bx = ex + lx * t_proj
            by = ey + ly * t_proj
            bz = ez + lz * t_proj

            d_beam = math.hypot(bx - mx, by - my, bz - mz)
            if d_beam < 4.0:  # Beam passes within 4m of center
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
