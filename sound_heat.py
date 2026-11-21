"""Sound heat map for tracking unseen combat activity, fights, and danger areas."""
import math
from typing import Any, Dict, List, Optional, Tuple


class SoundHeatMap:
    def __init__(self, decay_s: float = 10.0):
        self.decay_s = decay_s
        # List of recorded sound events: (t, kind, bearing, dist_min, dist_max, weight)
        self.events: List[Dict[str, float]] = []

    def add_sound(self, t: float, sound_obs: Dict[str, Any]):
        kind = sound_obs.get("kind", "gun")
        bearing = sound_obs.get("bearing", 0.0)
        dist_cls = sound_obs.get("dist", "mid")
        age = sound_obs.get("age", 0.0)

        # Distance estimation from protocol dist class
        if dist_cls == "near":
            dist_min, dist_max = 10.0, 60.0
        elif dist_cls == "mid":
            dist_min, dist_max = 60.0, 160.0
        else:
            dist_min, dist_max = 160.0, 400.0

        # Weapon weights (higher = deadlier fight)
        weights = {
            "laser": 2.5,
            "laser_charge": 2.0,
            "lightning": 2.0,
            "mortar": 2.0,
            "explosion": 1.8,
            "death": 2.2,
            "drop": 2.0,
            "gun": 1.0,
            "npc_gun": 0.5,
            "step": 0.8,
            "boost": 1.0,
            "reload": 0.6,
        }
        w = weights.get(kind, 1.0)

        self.events.append({
            "t": t - age,
            "kind": kind,
            "bearing": bearing,
            "dist_min": dist_min,
            "dist_max": dist_max,
            "weight": w,
        })

    def update(self, current_t: float):
        """Prune decayed sounds older than decay_s."""
        self.events = [e for e in self.events if current_t - e["t"] <= self.decay_s]

    def get_heat_at(self, my_pos: List[float], candidate_pos: List[float], current_t: float) -> float:
        """Computes fight danger heat for candidate position based on heard sounds."""
        if not self.events:
            return 0.0

        total_heat = 0.0
        dx = candidate_pos[0] - my_pos[0]
        dy = candidate_pos[1] - my_pos[1]
        cand_dist = math.hypot(dx, dy)
        cand_bearing = math.atan2(dy, dx)

        for e in self.events:
            dt = current_t - e["t"]
            if dt > self.decay_s or dt < 0:
                continue
            decay = 1.0 - (dt / self.decay_s)

            # Angle difference to bearing +/- 7 deg noise window
            ang_diff = abs(((cand_bearing - e["bearing"] + math.pi) % (2.0 * math.pi)) - math.pi)
            if ang_diff < math.radians(25.0):
                # Within hearing cone
                if e["dist_min"] <= cand_dist <= e["dist_max"]:
                    total_heat += e["weight"] * decay * 2.0
                elif cand_dist < e["dist_min"]:
                    total_heat += e["weight"] * decay * 0.8

        return total_heat

    def get_most_recent_threat_bearing(self) -> Optional[float]:
        """Returns bearing of the most recent weapon/combat sound."""
        if not self.events:
            return None
        # Sort by most recent
        recent = sorted(self.events, key=lambda x: x["t"], reverse=True)
        return recent[0]["bearing"]
