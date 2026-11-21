"""Zone timeline modeling and safe point computation."""
import math
from typing import Any, Dict, List, Optional, Tuple


class ZoneManager:
    def __init__(self, zone_cfg: Dict[str, Any]):
        self.cfg = zone_cfg
        self.phases = zone_cfg.get("phases", [])

    def get_safe_hold_point(
        self,
        my_pos: List[float],
        zone_obs: Dict[str, Any],
        world_model,
        time_to_arrival_s: float = 12.0,
    ) -> List[float]:
        """Calculates the optimal survival position inside the next circle.
        - Early phases (dps <= 35): sit 8-15m inside next rim, nearest to my current position.
        - Late phases (dps >= 60 / radius <= 140): move toward center.
        - Endgame (radius <= 25): claim best cover near final center.
        """
        next_center = zone_obs.get("next_center", [0.0, 0.0])
        next_radius = float(zone_obs.get("next_radius", 380.0))
        dps = float(zone_obs.get("dps", 20.0))

        # Vector from next center to me
        dx = my_pos[0] - next_center[0]
        dy = my_pos[1] - next_center[1]
        dist_to_center = math.hypot(dx, dy)

        if dist_to_center < 1e-4:
            angle_to_me = 0.0
        else:
            angle_to_me = math.atan2(dy, dx)

        # Determine target radial distance from next center
        if next_radius <= 30.0:
            # Endgame: hold within 5-10m of center with best cover
            target_r = min(8.0, next_radius * 0.4)
        elif dps >= 60.0 or next_radius <= 140.0:
            # Mid-late phases: sit well inside circle (40% radius)
            target_r = next_radius * 0.45
        else:
            # Early phases: sit 12-16m inside the rim on the side nearest to us
            target_r = max(10.0, next_radius - 15.0)

        # Candidate point along our approach axis
        base_x = next_center[0] + target_r * math.cos(angle_to_me)
        base_y = next_center[1] + target_r * math.sin(angle_to_me)

        # Search nearby walkable cells with high cover score
        best_pt = [base_x, base_y]
        best_score = -999.0

        for r_offset in (-8.0, 0.0, 8.0):
            for ang_offset in (-0.2, 0.0, 0.2):
                cand_r = max(5.0, min(next_radius - 5.0, target_r + r_offset))
                cand_ang = angle_to_me + ang_offset
                cx = next_center[0] + cand_r * math.cos(cand_ang)
                cy = next_center[1] + cand_r * math.sin(cand_ang)

                if not world_model.is_walkable(cx, cy):
                    continue

                cover = world_model.get_cover_score(cx, cy)
                wall_d = world_model.get_wall_dist(cx, cy)
                # Score: prefer high cover and wall proximity 3m-8m
                score = cover * 2.0 - abs(wall_d - 5.0) * 0.5
                if score > best_score:
                    best_score = score
                    best_pt = [cx, cy]

        return best_pt

    def is_evacuation_urgent(
        self,
        my_pos: List[float],
        zone_obs: Dict[str, Any],
        current_t: float,
        walk_speed: float = 11.7,
    ) -> Tuple[bool, float]:
        """Evaluates whether we must sprint immediately to avoid zone collapse."""
        next_center = zone_obs.get("next_center", [0.0, 0.0])
        next_radius = float(zone_obs.get("next_radius", 380.0))
        cur_center = zone_obs.get("center", [0.0, 0.0])
        cur_radius = float(zone_obs.get("radius", 560.0))
        shrink_start = float(zone_obs.get("shrink_start", 0.0))
        shrink_end = float(zone_obs.get("shrink_end", 999.0))

        dist_to_cur_center = math.hypot(my_pos[0] - cur_center[0], my_pos[1] - cur_center[1])
        outside_cur_zone = dist_to_cur_center >= (cur_radius - 8.0)

        dist_to_next_center = math.hypot(my_pos[0] - next_center[0], my_pos[1] - next_center[1])
        dist_to_next_rim = max(0.0, dist_to_next_center - (next_radius - 10.0))

        time_to_walk = dist_to_next_rim / max(1.0, walk_speed)
        time_left = max(0.0, shrink_end - current_t)

        is_shrinking_soon = (shrink_start - current_t) < 15.0
        time_critical = is_shrinking_soon and ((time_to_walk + 8.0) >= time_left)

        is_urgent = outside_cur_zone or (time_critical and dist_to_next_center > next_radius - 5.0)
        return is_urgent, dist_to_next_rim
