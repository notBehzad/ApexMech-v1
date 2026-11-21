"""Zone timeline modeling and safe point computation.

This module provides the ``ZoneManager`` class used by the bot to compute safe
positions for zone evacuation and to decide when immediate sprinting is
necessary. The original implementation contained several syntax errors and an
overly aggressive evacuation heuristic that caused poor performance. The
reimplemented version restores the intended behaviour while keeping the code
readable and lightweight.
"""

import math
from typing import Any, Dict, List, Tuple


class ZoneManager:
    """Utility class for zone‑related calculations.

    The constructor receives the raw zone configuration (as parsed from the
    ``hello`` message).  The manager exposes two public methods:

    * ``get_safe_hold_point`` – compute a target location inside the *next*
      safe zone that balances distance, cover and proximity to walls.
    * ``is_evacuation_urgent`` – determine whether the bot must sprint
      immediately to avoid being killed by a shrinking zone.
    """

    def __init__(self, zone_cfg: Dict[str, Any]):
        self.cfg = zone_cfg
        self.phases = zone_cfg.get("phases", [])

    # ---------------------------------------------------------------------
    # Safe‑hold point computation
    # ---------------------------------------------------------------------
    def get_safe_hold_point(
        self,
        my_pos: List[float],
        zone_obs: Dict[str, Any],
        world_model,
        time_to_arrival_s: float = 12.0,
    ) -> List[float]:
        """Return an optimal survival point inside the *next* zone.

        The strategy follows the original design specifications:

        * **Early phases** (damage per second ≤ 35) – stay near the inner rim
          of the next circle, on the side closest to the current position.
        * **Mid‑late phases** (dps ≥ 60 or radius ≤ 140) – move toward the
          centre, but keep a comfortable margin.
        * **Endgame** (radius ≤ 30) – pick a point with high cover close to the
          final centre.
        """
        next_center = zone_obs.get("next_center", [0.0, 0.0])
        next_radius = float(zone_obs.get("next_radius", 380.0))
        dps = float(zone_obs.get("dps", 20.0))

        # Vector from next centre to the bot
        dx = my_pos[0] - next_center[0]
        dy = my_pos[1] - next_center[1]
        dist_to_center = math.hypot(dx, dy)

        # Angle from centre to the bot (fallback to 0 when distance is tiny)
        angle_to_me = math.atan2(dy, dx) if dist_to_center >= 1e-4 else 0.0

        # Desired radial distance from the next centre based on phase
        if next_radius <= 30.0:
            # Endgame – stay very close to centre, but keep a little offset for
            # cover. ``target_r`` is capped at 8 m.
            target_r = min(8.0, next_radius * 0.4)
        elif dps >= 60.0 or next_radius <= 140.0:
            # Mid‑late phases – stay well inside the circle (≈45 % of radius).
            target_r = next_radius * 0.45
        else:
            # Early phases – sit close to the inner rim on the side nearest to us.
            target_r = max(10.0, next_radius * 0.15)

        # Base candidate point on the line from centre to the bot
        base_x = next_center[0] + target_r * math.cos(angle_to_me)
        base_y = next_center[1] + target_r * math.sin(angle_to_me)

        # Search a small neighbourhood for a walkable cell with good cover.
        best_pt = [base_x, base_y]
        best_score = float("inf")
        for r_offset in (8.0, 0.0, -8.0):
            for ang_offset in (0.2, 0.0, -0.2):
                cand_r = max(5.0, min(next_radius - 5.0, target_r + r_offset))
                cand_ang = angle_to_me + ang_offset
                cx = next_center[0] + cand_r * math.cos(cand_ang)
                cy = next_center[1] + cand_r * math.sin(cand_ang)

                if not world_model.is_walkable(cx, cy):
                    continue

                cover = world_model.get_cover_score(cx, cy)
                wall_d = world_model.get_wall_dist(cx, cy)
                # Lower score is better: prefer high cover and proximity to a wall.
                score = cover * 2.0 + abs(wall_d - 5.0) * 0.5
                if score < best_score:
                    best_score = score
                    best_pt = [cx, cy]

        return best_pt

    # ---------------------------------------------------------------------
    # Evacuation urgency check
    # ---------------------------------------------------------------------
    def is_evacuation_urgent(
        self,
        my_pos: List[float],
        zone_obs: Dict[str, Any],
        current_t: float,
        walk_speed: float = 11.7,
    ) -> Tuple[bool, float]:
        """Determine if an immediate sprint is required.

        The heuristic returns ``True`` when either:
        * the bot is already outside the *current* safe zone, **or**
        * the remaining time until ``shrink_end`` is insufficient to reach the
          next safe‑zone rim at the given walking speed.
        The function also returns the distance to that rim for the caller's
        own timing calculations.
        """
        next_center = zone_obs.get("next_center", [0.0, 0.0])
        next_radius = float(zone_obs.get("next_radius", 380.0))
        cur_center = zone_obs.get("center", [0.0, 0.0])
        cur_radius = float(zone_obs.get("radius", 560.0))
        shrink_end = float(zone_obs.get("shrink_end", 999.0))

        # Distance to current centre – check if we are already outside the
        # current safe zone.
        dist_to_cur_center = math.hypot(my_pos[0] - cur_center[0], my_pos[1] - cur_center[1])
        outside_cur_zone = dist_to_cur_center >= (cur_radius - 8.0)

        # Distance to the next centre and to the next rim (the point we need to
        # reach before the shrink ends).
        dist_to_next_center = math.hypot(my_pos[0] - next_center[0], my_pos[1] - next_center[1])
        dist_to_next_rim = max(0.0, dist_to_next_center - (next_radius - 10.0))

        # Time left until the next shrink finishes.
        time_left = max(0.0, shrink_end - current_t)
        # Estimated walking time to reach the rim.
        time_needed = dist_to_next_rim / max(walk_speed, 1e-6)

        # Urgent if we cannot make it in time or we are already outside the
        # current zone.
        urgent = outside_cur_zone or (time_needed > time_left)
        return urgent, dist_to_next_rim

