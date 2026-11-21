"""Diagnostic and performance telemetry system for Mech Royale (mechfight/1).
Collects observable metrics without altering gameplay or violating protocol timing.
"""
import json
import os
import sys
from typing import Any, Dict, List, Optional


class MatchDiagnostics:
    def __init__(self, enabled: bool = True, log_file: Optional[str] = None):
        # Can be toggled via parameter or environment variable
        env_setting = os.environ.get("APEX_DIAGNOSTICS")
        if env_setting is not None:
            self.enabled = env_setting.lower() in ("1", "true", "yes", "on")
        else:
            self.enabled = enabled

        self.log_file = log_file or os.environ.get("APEX_DIAGNOSTICS_FILE")
        self.reset()

    def reset(self):
        """Reset all match-specific counters for a new match."""
        self.match_started = False
        self.start_t: float = 0.0
        self.last_t: float = 0.0
        self.total_ticks: int = 0
        
        # Placement and survival
        self.final_place: Optional[int] = None
        self.final_kills: Optional[int] = None

        # Health and zone
        self.prev_hp: Optional[float] = None
        self.ticks_outside_zone: int = 0
        self.hp_recovered_regen: float = 0.0
        self.last_damage_taken_t: float = -999.0
        self.last_damage_dealt_t: float = -999.0

        # Damage counters
        self.total_damage_dealt: float = 0.0
        self.total_damage_received: float = 0.0
        self.damage_from_zone: float = 0.0
        self.damage_received_by_weapon: Dict[str, float] = {}
        self.damage_dealt_by_weapon: Dict[str, float] = {}

        # Economy & pickups
        self.cubes_collected: int = 0
        self.weapons_taken: List[str] = []
        self.supply_drops_captured: int = 0

        # Weapons, firing & downtime
        # Ticks where trigger was pulled: {slot: count}
        self.trigger_held_ticks: Dict[int, int] = {0: 0, 1: 0}
        # Ticks weapon was actively reloading or in cooldown
        self.weapon_reloading_ticks: Dict[int, int] = {0: 0, 1: 0}
        self.weapon_cooldown_ticks: Dict[int, int] = {0: 0, 1: 0}
        self.hits_by_weapon: Dict[str, int] = {}
        self.rear_hits_dealt: int = 0
        self.rear_hits_received: int = 0

        # Death context
        self.last_damage_event: Optional[Dict[str, Any]] = None

    def on_hello(self, hello: Dict[str, Any]):
        if not self.enabled:
            return
        self.reset()
        self.slot = hello.get("slot")

    def update(self, obs: Dict[str, Any], act: Dict[str, Any]):
        if not self.enabled:
            return

        t = obs.get("t", 0.0)
        if not self.match_started:
            self.start_t = t
            self.match_started = True

        self.last_t = t
        self.total_ticks += 1
        me = obs.get("self", {})
        cur_hp = me.get("hp", 0.0)

        # 1. Zone status
        if not me.get("in_zone", True):
            self.ticks_outside_zone += 1

        # 2. Process events
        events = obs.get("events", [])
        for e in events:
            etype = e.get("type")
            if etype == "damaged":
                amt = float(e.get("amount", 0.0))
                wpn = str(e.get("weapon", "unknown"))
                self.total_damage_received += amt
                self.damage_received_by_weapon[wpn] = self.damage_received_by_weapon.get(wpn, 0.0) + amt
                if wpn == "zone":
                    self.damage_from_zone += amt
                if e.get("rear"):
                    self.rear_hits_received += 1
                self.last_damage_taken_t = t
                self.last_damage_event = dict(e)

            elif etype == "hit":
                amt = float(e.get("amount", 0.0))
                wpn = str(e.get("weapon", "unknown"))
                self.total_damage_dealt += amt
                self.damage_dealt_by_weapon[wpn] = self.damage_dealt_by_weapon.get(wpn, 0.0) + amt
                self.hits_by_weapon[wpn] = self.hits_by_weapon.get(wpn, 0) + 1
                if e.get("rear"):
                    self.rear_hits_dealt += 1
                self.last_damage_dealt_t = t

            elif etype == "pickup":
                self.cubes_collected += 1

            elif etype == "took":
                self.weapons_taken.append(str(e.get("kind", "unknown")))

            elif etype == "drop":
                self.supply_drops_captured += 1

        # 3. Infer health recovered through passive regeneration
        # Protocol: regen occurs after 5s without taking/dealing damage while in_zone
        if self.prev_hp is not None:
            delta_hp = cur_hp - self.prev_hp
            # If HP went up without a cube pickup event on this tick
            had_pickup = any(e.get("type") == "pickup" for e in events)
            had_drop = any(e.get("type") == "drop" for e in events)
            if delta_hp > 0 and not had_pickup and not had_drop:
                time_since_combat = min(t - self.last_damage_taken_t, t - self.last_damage_dealt_t)
                if time_since_combat >= 4.9:  # 5s delay tolerance
                    self.hp_recovered_regen += delta_hp

        self.prev_hp = cur_hp

        # 4. Weapons usage and downtime
        fire = act.get("fire", [False, False])
        for slot_idx, slot in enumerate(me.get("weapons", [])):
            if slot is not None:
                if slot_idx < len(fire) and fire[slot_idx]:
                    self.trigger_held_ticks[slot_idx] = self.trigger_held_ticks.get(slot_idx, 0) + 1
                if slot.get("reload", 0.0) > 0.0:
                    self.weapon_reloading_ticks[slot_idx] = self.weapon_reloading_ticks.get(slot_idx, 0) + 1
                elif slot.get("cooldown", 0.0) > 0.0:
                    self.weapon_cooldown_ticks[slot_idx] = self.weapon_cooldown_ticks.get(slot_idx, 0) + 1

    def on_end(self, end_msg: Dict[str, Any]):
        if not self.enabled:
            return

        self.final_place = int(end_msg.get("place", 16))
        self.final_kills = int(end_msg.get("kills", 0))

        report = self.generate_report()
        self.emit(report)

    def generate_report(self) -> Dict[str, Any]:
        """Compile all collected metrics into a structured report dictionary."""
        survival_s = max(0.0, self.last_t - self.start_t)
        time_outside_zone_s = self.ticks_outside_zone * 0.1

        return {
            "summary": {
                "placement": self.final_place,
                "kills": self.final_kills,
                "survival_duration_s": round(survival_s, 2),
                "total_ticks": self.total_ticks,
                "score": 16 - (self.final_place or 16),
            },
            "damage": {
                "dealt_total": round(self.total_damage_dealt, 1),
                "received_total": round(self.total_damage_received, 1),
                "received_from_zone": round(self.damage_from_zone, 1),
                "dealt_by_weapon": {k: round(v, 1) for k, v in self.damage_dealt_by_weapon.items()},
                "received_by_weapon": {k: round(v, 1) for k, v in self.damage_received_by_weapon.items()},
                "rear_hits_received": self.rear_hits_received,
                "rear_hits_dealt": self.rear_hits_dealt,
            },
            "survival_and_health": {
                "time_outside_zone_s": round(time_outside_zone_s, 2),
                "health_recovered_regen": round(self.hp_recovered_regen, 1),
                "death_context": self._format_death_context(),
            },
            "economy_and_loot": {
                "cubes_collected": self.cubes_collected,
                "weapons_acquired": self.weapons_taken,
                "supply_drops_captured": self.supply_drops_captured,
            },
            "weapons_and_activity": {
                "trigger_held_seconds": {
                    f"slot_{k}": round(v * 0.1, 2) for k, v in self.trigger_held_ticks.items()
                },
                "reloading_seconds": {
                    f"slot_{k}": round(v * 0.1, 2) for k, v in self.weapon_reloading_ticks.items()
                },
                "cooldown_seconds": {
                    f"slot_{k}": round(v * 0.1, 2) for k, v in self.weapon_cooldown_ticks.items()
                },
                "hits_landed": self.hits_by_weapon,
            },
        }

    def _format_death_context(self) -> Optional[Dict[str, Any]]:
        if not self.last_damage_event:
            return None
        return {
            "last_attacker_id": self.last_damage_event.get("by"),
            "weapon": self.last_damage_event.get("weapon"),
            "amount": self.last_damage_event.get("amount"),
            "rear_attack": bool(self.last_damage_event.get("rear", False)),
            "bearing": self.last_damage_event.get("bearing"),
        }

    def format_text_summary(self, r: Dict[str, Any]) -> str:
        s = r["summary"]
        d = r["damage"]
        h = r["survival_and_health"]
        e = r["economy_and_loot"]
        w = r["weapons_and_activity"]
        dc = h["death_context"]

        lines = [
            "==================== APEX MATCH DIAGNOSTIC REPORT ====================",
            f"Result:       Place: {s['placement']}th | Kills: {s['kills']} | Match Score: {s['score']}/15",
            f"Survival:     Duration: {s['survival_duration_s']}s ({s['total_ticks']} decisions)",
            f"Health/Regen: Recovered via Regen: {h['health_recovered_regen']} HP | Time Outside Zone: {h['time_outside_zone_s']}s",
            f"Damage Dealt: Total: {d['dealt_total']} | Weapons: {d['dealt_by_weapon']} | Rear Hits: {d['rear_hits_dealt']}",
            f"Damage Taken: Total: {d['received_total']} (Zone: {d['received_from_zone']}) | By: {d['received_by_weapon']}",
            f"Economy:      Cubes: {e['cubes_collected']} | Pickups: {e['weapons_acquired']} | Supply Drops: {e['supply_drops_captured']}",
            f"Trigger Time: Slot 0: {w['trigger_held_seconds'].get('slot_0', 0)}s | Slot 1: {w['trigger_held_seconds'].get('slot_1', 0)}s",
            f"Reload Time:  Slot 0: {w['reloading_seconds'].get('slot_0', 0)}s | Cooldown Slot 0: {w['cooldown_seconds'].get('slot_0', 0)}s",
        ]
        if dc:
            rear_str = "REAR HIT" if dc["rear_attack"] else "Front/Side"
            lines.append(f"Death Cause:  Attacker #{dc['last_attacker_id']} with {dc['weapon']} ({dc['amount']} dmg, {rear_str})")
        lines.append("======================================================================")
        return "\n".join(lines)

    def emit(self, report: Dict[str, Any]):
        """Write diagnostic summary strictly to stderr and optional diagnostic log file."""
        text_summary = self.format_text_summary(report)
        print(text_summary, file=sys.stderr, flush=True)

        if self.log_file:
            try:
                # Ensure directory exists if path includes folder
                log_dir = os.path.dirname(self.log_file)
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(report) + "\n")
            except Exception as ex:
                print(f"[DIAGNOSTICS] Failed to write log file: {ex}", file=sys.stderr, flush=True)
