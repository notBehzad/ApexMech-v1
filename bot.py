"""ApexMech-v2: World #1 Autonomous Survival and Tactical Combat Agent."""
import math
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from combat_engine import (
    can_hit_target,
    evaluate_engagement_permission,
    solve_gun_intercept,
    solve_mortar_trajectory,
)
from diagnostics import MatchDiagnostics
from path_planner import PathPlanner
from sound_heat import SoundHeatMap
from tracker import Tracker
from world_model import WorldModel
import mechfight as mf


class Bot:
    name = "ApexMech-v2"

    def __init__(self, hello: mf.Hello):
        t0 = time.perf_counter()
        self.hello = hello
        cfg = hello.get("config", {})
        self.mech_cfg = cfg.get("mech", {})
        self.weapons_cfg = cfg.get("weapons", {})
        self.loot_cfg = cfg.get("loot", {})
        self.zone_cfg = cfg.get("zone", {})
        self.pickup_radius = float(self.loot_cfg.get("pickup_radius", 4.0))

        # Seed reproducible PRNG
        self.seed = hello.get("seed", 42)
        random.seed(self.seed)

        # 1. World Model (rasterization, 2m cover, 4m nav)
        map_dir = hello.get("map", {}).get("dir", "arena")
        self.wm = WorldModel(map_dir)
        self.planner = PathPlanner(self.wm)

        # 2. Perception & Audio Tracking
        self.tracker = Tracker(memory_s=float(cfg.get("vision", {}).get("memory_s", 5.0)))
        self.sound_heat = SoundHeatMap(decay_s=10.0)

        # 3. Zone & Tactical States
        from zone_manager import ZoneManager
        self.zone_mgr = ZoneManager(self.zone_cfg)

        # 4. Movement & Dodge State Machine
        self.strafe_sign: float = 1.0 if random.random() < 0.5 else -1.0
        self.strafe_commit_timer: float = 0.0
        self.strafe_commit_duration: float = random.uniform(0.5, 0.9)
        self.scan_yaw: float = 0.0
        self.last_path_t: float = -999.0
        self.current_path: List[List[float]] = []

        # 5. Fallback & Telemetry
        self.last_threat_bearing: float = 0.0
        self.last_damage_t: float = -999.0
        self.diag = MatchDiagnostics(enabled=True)
        self.diag.on_hello(hello)

        t1 = time.perf_counter()
        print(f"[ApexMech-v2] Bot ready in {(t1 - t0) * 1000:.1f}ms", file=sys.stderr, flush=True)

    def act(self, obs: mf.Observation) -> mf.Action:
        tick_start = time.perf_counter()
        tick = obs["tick"]
        t = obs.get("t", 0.0)
        me = obs["self"]
        my_pos = me["pos"]
        my_hp = me["hp"]
        my_max_hp = me.get("max_hp", 1400.0)
        my_aim = me["aim"]
        my_energy = me.get("energy", 100.0)
        in_zone = me.get("in_zone", True)
        grounded = me.get("grounded", True)

        # Safe fallback action prepared in advance
        act: mf.Action = {
            "tick": tick,
            "move": [0.0, 0.0],
            "aim": [my_aim[0], 0.0],
            "fire": [False, False],
            "boost": False,
            "jump": False,
        }

        try:
            # -----------------------------------------------------
            # UPDATE SENSORS & STATE
            # -----------------------------------------------------
            self.tracker.update(obs)
            self.sound_heat.update(t)
            for s in obs.get("sounds", []):
                self.sound_heat.add_sound(t, s)

            # Check damage events
            for e in obs.get("events", []):
                if e.get("type") == "damaged":
                    self.last_damage_t = t
                    if e.get("bearing") is not None:
                        self.last_threat_bearing = float(e["bearing"])

            # -----------------------------------------------------
            # UPGRADES RESOLUTION
            # -----------------------------------------------------
            if obs.get("offer"):
                held = [w["kind"] for w in me["weapons"] if w]
                act["pick"] = self._choose_upgrade(obs["offer"], held)

            # -----------------------------------------------------
            # WEAPON & SUPPLY DROP PICKUPS
            # -----------------------------------------------------
            take_action = self._evaluate_pickups(obs.get("pickups", []), my_pos, me["weapons"])
            if take_action:
                act.update(take_action)

            # -----------------------------------------------------
            # PRIORITY ARBITER: SELECT ACTIVE STATE
            # 1. SURVIVE-ZONE
            # 2. EVADE (incoming fire)
            # 3. DISENGAGE (break LOS & regen)
            # 4. LOOT/UPGRADE
            # 5. AMBUSH/ENGAGE
            # 6. HIDE/SCAN
            # -----------------------------------------------------
            zone_obs = obs.get("zone", {})
            urgent_zone, dist_to_zone_rim = self.zone_mgr.is_evacuation_urgent(my_pos, zone_obs, t)

            # Threat analysis
            proj_dodge = self.tracker.get_projectile_threat(
                my_pos,
                self.mech_cfg.get("radius", 2.5),
                self.mech_cfg.get("height", 9.0),
                obs.get("projectiles", []),
            )
            laser_dodge = self.tracker.get_laser_threat_dodge(my_pos)

            # Enemy threat assessment
            target_enemy, enemies_with_los = self._find_target_and_los_count(my_pos)
            has_cover = self.wm.get_cover_score(my_pos[0], my_pos[1]) >= 8.0
            under_fire = (t - self.last_damage_t) < 3.0

            # State decision variables
            state_move: List[float] = [0.0, 0.0]
            state_boost: bool = False
            state_jump: bool = False
            state_aim: List[float] = [my_aim[0], 0.0]
            fire_slots: List[bool] = [False, False]

            # -----------------------------------------------------
            # 1. SURVIVE-ZONE (Top priority if collapsing or outside)
            # -----------------------------------------------------
            if urgent_zone or not in_zone:
                mf.draw.label("STATE: SURVIVE-ZONE")
                safe_pt = self.zone_mgr.get_safe_hold_point(my_pos, zone_obs, self.wm)
                state_move, state_boost = self._navigate_toward(my_pos, safe_pt, t, force_sprint=not in_zone)
                # If active incoming projectile or laser threat, blend evasion
                if proj_dodge is not None or laser_dodge is not None:
                    d_vec = proj_dodge[0] if proj_dodge else laser_dodge
                    state_move = [state_move[0] * 0.6 + d_vec[0] * 0.4, state_move[1] * 0.6 + d_vec[1] * 0.4]
                    state_boost = my_energy > 20.0
                # Face threat while retreating to prevent rear damage multiplier (1.5x)
                if target_enemy:
                    state_aim = [math.atan2(target_enemy.pos[1] - my_pos[1], target_enemy.pos[0] - my_pos[0]), 0.0]
                    dist_e = math.hypot(target_enemy.pos[0] - my_pos[0], target_enemy.pos[1] - my_pos[1])
                    if dist_e < 80.0 and under_fire:
                        fire_slots[0] = True
                elif under_fire:
                    state_aim = [self.last_threat_bearing, 0.0]
                else:
                    state_aim = [math.atan2(safe_pt[1] - my_pos[1], safe_pt[0] - my_pos[0]), 0.0]

            # -----------------------------------------------------
            # 2. EVADE (Incoming bullets, shells, charged lasers)
            # -----------------------------------------------------
            elif proj_dodge is not None or laser_dodge is not None:
                mf.draw.label("STATE: EVADE")
                dodge_vec = proj_dodge[0] if proj_dodge else laser_dodge
                state_move = dodge_vec
                state_boost = my_energy > 20.0
                state_jump = grounded and random.random() < 0.2
                if target_enemy:
                    state_aim = [math.atan2(target_enemy.pos[1] - my_pos[1], target_enemy.pos[0] - my_pos[0]), 0.0]

            # -----------------------------------------------------
            # 3. DISENGAGE (HP < 35% or 2+ enemies have LOS)
            # -----------------------------------------------------
            elif (my_hp / my_max_hp) < 0.35 or enemies_with_los >= 2:
                mf.draw.label("STATE: DISENGAGE")
                cover_pos = self._find_nearest_cover(my_pos, target_enemy.pos if target_enemy else None, zone_obs)
                if cover_pos:
                    state_move, state_boost = self._navigate_toward(my_pos, cover_pos, t, force_sprint=under_fire)
                # Face attacker while retreating to prevent rear damage (1.5x)
                if target_enemy:
                    state_aim = [math.atan2(target_enemy.pos[1] - my_pos[1], target_enemy.pos[0] - my_pos[0]), 0.0]
                elif under_fire:
                    state_aim = [self.last_threat_bearing, 0.0]

            # -----------------------------------------------------
            # 4. LOOT / UPGRADE (Supply drops or safe nearby pickups)
            # -----------------------------------------------------
            elif self._can_loot_supply_drop(obs.get("drops", []), my_pos, enemies_with_los):
                mf.draw.label("STATE: LOOT-DROP")
                drop_pos = obs["drops"][0]["pos"]
                state_move, state_boost = self._navigate_toward(my_pos, drop_pos[:2], t, force_sprint=False)
                state_aim = [math.atan2(drop_pos[1] - my_pos[1], drop_pos[0] - my_pos[0]), 0.0]

            # -----------------------------------------------------
            # 5. AMBUSH / ENGAGE (Only if strict conditions met)
            # -----------------------------------------------------
            elif target_enemy is not None and evaluate_engagement_permission(
                my_hp, my_max_hp, target_enemy, math.hypot(target_enemy.pos[0] - my_pos[0], target_enemy.pos[1] - my_pos[1]),
                enemies_with_los, has_cover, under_fire
            ):
                mf.draw.label("STATE: ENGAGE")
                state_move, state_boost, state_aim, fire_slots = self._handle_combat(
                    my_pos, my_hp, my_aim, me["weapons"], target_enemy, t
                )

            # -----------------------------------------------------
            # 6. HIDE / SCAN (Default survival: hold cover, sweep scan)
            # -----------------------------------------------------
            else:
                mf.draw.label("STATE: HIDE/SCAN")
                safe_hold = self.zone_mgr.get_safe_hold_point(my_pos, zone_obs, self.wm)
                dist_to_hold = math.hypot(my_pos[0] - safe_hold[0], my_pos[1] - safe_hold[1])

                if dist_to_hold > 8.0:
                    state_move, state_boost = self._navigate_toward(my_pos, safe_hold, t, force_sprint=False)
                else:
                    # Patrol silently below 5 m/s near cover
                    state_move = self._silent_cover_patrol(my_pos, t)
                    state_boost = False

                # Aim scan: sweep +/- 90 deg around threat axis or sound heat
                state_aim = self._compute_scanning_aim(my_pos, my_aim, t)

                # Gun reload optimization when concealed
                gun_slot = next((s for s in me["weapons"] if s and s.get("kind") == "gun"), None)
                if gun_slot and gun_slot.get("ammo", 12) <= 4 and gun_slot.get("reload", 0.0) == 0:
                    act["reload"] = True

            # -----------------------------------------------------
            # CONTINUOUS DODGE WHEN THREAT PLAUSIBLE
            # -----------------------------------------------------
            if (target_enemy or under_fire) and not (urgent_zone or not in_zone):
                state_move, state_boost = self._apply_continuous_dodge(my_pos, state_move, target_enemy, t)

            # Apply final action parameters
            act.update(
                move=self._clamp_norm(state_move),
                aim=state_aim,
                fire=fire_slots,
                boost=state_boost and (my_energy > 20.0 or urgent_zone or not in_zone),
                jump=state_jump,
            )

        except Exception as ex:
            # Time guard / safety catch: emit safe fallback
            print(f"[ApexMech ERROR at tick {tick}]: {ex}", file=sys.stderr, flush=True)
            act["move"] = [0.0, 0.0]
            act["aim"] = [self.last_threat_bearing, 0.0]

        # Ensure tick execution stays well inside 20ms target
        elapsed = (time.perf_counter() - tick_start) * 1000.0
        if elapsed > 20.0:
            print(f"[ApexMech WARN] Tick {tick} took {elapsed:.1f}ms", file=sys.stderr, flush=True)

        self.diag.update(obs, act)
        return act

    def on_end(self, end_msg: Dict[str, Any]):
        self.diag.on_end(end_msg)

    # -----------------------------------------------------------------
    # HELPER IMPLEMENTATIONS
    # -----------------------------------------------------------------
    def _find_target_and_los_count(self, my_pos: List[float]) -> Tuple[Optional[Any], int]:
        """Finds closest priority threat and counts enemies that have LOS to us."""
        los_count = 0
        best_target = None
        min_dist = 999.0

        for mid, e in self.tracker.enemies.items():
            dist = math.hypot(e.pos[0] - my_pos[0], e.pos[1] - my_pos[1])
            has_los = self.wm.has_line_of_sight([my_pos[0], my_pos[1], my_pos[2] + 8.0], [e.pos[0], e.pos[1], e.pos[2] + 4.5])
            if has_los:
                los_count += 1
            if dist < min_dist:
                min_dist = dist
                best_target = e

        return best_target, los_count

    def _navigate_toward(
        self, my_pos: List[float], goal_pos: List[float], current_t: float, force_sprint: bool = False
    ) -> Tuple[List[float], bool]:
        """Paths toward goal using A* with caching."""
        if current_t - self.last_path_t > 2.0 or not self.current_path:
            self.current_path = self.planner.find_path(my_pos[:2], goal_pos[:2])
            self.last_path_t = current_t

        while self.current_path and math.hypot(my_pos[0] - self.current_path[0][0], my_pos[1] - self.current_path[0][1]) < 4.0:
            self.current_path.pop(0)

        if self.current_path:
            wp = self.current_path[0]
            dx = wp[0] - my_pos[0]
            dy = wp[1] - my_pos[1]
            mag = math.hypot(dx, dy)
            if mag < 1e-4:
                return [0.0, 0.0], force_sprint
            return [dx / max(1e-4, mag), dy / max(1e-4, mag)], force_sprint
        return [0.0, 0.0], False

    def _apply_continuous_dodge(
        self, my_pos: List[float], base_move: List[float], target_enemy: Optional[Any], current_t: float
    ) -> Tuple[List[float], bool]:
        """Commit to perpendicular strafe for 0.4 - 1.0s, then flip."""
        dt = 0.1
        self.strafe_commit_timer += dt
        if self.strafe_commit_timer >= self.strafe_commit_duration:
            self.strafe_sign *= -1.0
            self.strafe_commit_timer = 0.0
            self.strafe_commit_duration = random.uniform(0.4, 0.9)

        threat_yaw = self.last_threat_bearing
        if target_enemy:
            threat_yaw = math.atan2(target_enemy.pos[1] - my_pos[1], target_enemy.pos[0] - my_pos[0])

        # Perpendicular strafe vector
        perp_yaw = threat_yaw + (math.pi * 0.5 * self.strafe_sign)
        strafe_x = math.cos(perp_yaw) * 0.75
        strafe_y = math.sin(perp_yaw) * 0.75

        combined_x = base_move[0] * 0.4 + strafe_x * 0.6
        combined_y = base_move[1] * 0.4 + strafe_y * 0.6
        return [combined_x, combined_y], False

    def _silent_cover_patrol(self, my_pos: List[float], current_t: float) -> List[float]:
        """Drifts below 5 m/s along cover to remain completely silent (no footsteps)."""
        patrol_angle = (current_t * 0.3) % (2.0 * math.pi)
        # Length 0.35 corresponds to ~4 m/s (below 5 m/s step_speed)
        return [math.cos(patrol_angle) * 0.35, math.sin(patrol_angle) * 0.35]

    def _compute_scanning_aim(self, my_pos: List[float], current_aim: List[float], current_t: float) -> List[float]:
        """Sweeps +/- 90 deg around threat axis or sound heat."""
        sound_bearing = self.sound_heat.get_most_recent_threat_bearing()
        axis = sound_bearing if sound_bearing is not None else self.last_threat_bearing
        # Periodic sweep
        sweep_offset = math.sin(current_t * 1.5) * math.radians(75.0)
        return [axis + sweep_offset, 0.0]

    def _handle_combat(
        self, my_pos: List[float], my_hp: float, my_aim: List[float], weapons: List[Any], enemy: Any, current_t: float
    ) -> Tuple[List[float], bool, List[float], List[bool]]:
        """Aims with intercept leading and fires active weapons."""
        dist = math.hypot(enemy.pos[0] - my_pos[0], enemy.pos[1] - my_pos[1])
        fire_slots = [False, False]
        aim_cmd = [math.atan2(enemy.pos[1] - my_pos[1], enemy.pos[0] - my_pos[0]), 0.0]

        # Slot 0 Gun lead
        g_lead = solve_gun_intercept([my_pos[0], my_pos[1], my_pos[2] + 7.0], 252.0, enemy.pos, enemy.vel, max_range=160.0)
        if g_lead is not None:
            _, _, gyaw, gpitch = g_lead
            aim_cmd = [gyaw, gpitch]
            if can_hit_target(my_aim, gyaw, gpitch, dist):
                fire_slots[0] = True

        # Slot 1 Laser / Lightning / Mortar
        slot1 = weapons[1] if len(weapons) > 1 else None
        if slot1 is not None:
            wkind = slot1.get("kind")
            if wkind == "laser":
                charge = slot1.get("charge", 0.0)
                max_charge = slot1.get("stats", {}).get("charge", 0.75)
                # Avoid charging if lightning holder nearby
                has_lightning_threat = "lightning" in enemy.weapons and dist < 50.0
                if not has_lightning_threat:
                    if charge < max_charge - 0.05:
                        fire_slots[1] = True
                    else:
                        fire_slots[1] = False
            elif wkind == "lightning" and dist < 45.0:
                fire_slots[1] = True

        # Flank movement
        move_dir = [-math.sin(aim_cmd[0]) * self.strafe_sign, math.cos(aim_cmd[0]) * self.strafe_sign]
        return move_dir, False, aim_cmd, fire_slots

    def _find_nearest_cover(
        self, my_pos: List[float], threat_pos: Optional[List[float]], zone_obs: Dict[str, Any]
    ) -> Optional[List[float]]:
        next_center = zone_obs.get("next_center", [0.0, 0.0])
        next_radius = float(zone_obs.get("next_radius", 380.0))

        # Check 2 concentric rings (8m, 16m), 6 angles each
        for r in (8.0, 16.0):
            for k in range(6):
                ang = k * (2.0 * math.pi / 6)
                cx = my_pos[0] + r * math.cos(ang)
                cy = my_pos[1] + r * math.sin(ang)

                if math.hypot(cx - next_center[0], cy - next_center[1]) > next_radius - 5.0:
                    continue
                if not self.wm.is_walkable(cx, cy):
                    continue
                if self.wm.get_wall_dist(cx, cy) > 8.0:
                    continue

                if threat_pos:
                    if not self.wm.has_line_of_sight([threat_pos[0], threat_pos[1], threat_pos[2] + 7.0], [cx, cy, my_pos[2] + 4.5]):
                        return [cx, cy]
                elif self.wm.get_cover_score(cx, cy) >= 7.0:
                    return [cx, cy]

        return None

    def _can_loot_supply_drop(self, drops: List[Dict[str, Any]], my_pos: List[float], enemies_with_los: int) -> bool:
        if enemies_with_los > 0 or not drops:
            return False
        d = drops[0]
        if d.get("lands_in", 999.0) == 0.0 and math.hypot(d["pos"][0] - my_pos[0], d["pos"][1] - my_pos[1]) < 35.0:
            return True
        return False

    def _choose_upgrade(self, offer: Dict[str, Any], held_weapons: List[str]) -> int:
        weights = {"armor": 120, "nanites": 110, "plating": 100, "sensors": 95, "servos": 85}
        best_i = 0
        best_w = -1.0
        for i, c in enumerate(offer.get("choices", [])):
            uid = c.get("id", "")
            w = weights.get(uid, 40.0)
            if any(uid.startswith(k) for k in held_weapons):
                w += 35.0
            if w > best_w:
                best_w = w
                best_i = i
        return best_i

    def _evaluate_pickups(self, pickups: List[Dict[str, Any]], my_pos: List[float], weapons: List[Any]) -> Optional[Dict[str, Any]]:
        near = [p for p in pickups if math.hypot(p["pos"][0] - my_pos[0], p["pos"][1] - my_pos[1]) < self.pickup_radius]
        if not near:
            return None
        tier = {"laser": 4, "lightning": 3, "mortar": 2, "gun": 1}
        slot1 = weapons[1] if len(weapons) > 1 else None
        if slot1 is None:
            best_p = max(near, key=lambda p: tier.get(p.get("kind"), 0))
            if best_p.get("kind") != "gun":
                return {"take": best_p["id"]}
        return None

    def _clamp_norm(self, v: List[float]) -> List[float]:
        mag = math.hypot(v[0], v[1])
        if mag > 1.0:
            return [v[0] / mag, v[1] / mag]
        return v


if __name__ == "__main__":
    mf.run(Bot)
