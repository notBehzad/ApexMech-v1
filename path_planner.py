"""A* Pathfinding over 4m cost grid with exposure, cover, and distance weighting."""
import heapq
import math
from typing import List, Optional, Tuple

import numpy as np


class PathPlanner:
    def __init__(self, world_model):
        self.wm = world_model
        self.nx = world_model.nx_4m
        self.ny = world_model.ny_4m
        self.step = world_model.res_4m
        self.min_x = world_model.min_x
        self.min_y = world_model.min_y
        self.solid = world_model.solid_4m
        self.wall_dist = world_model.wall_dist_4m
        self.cover_score = world_model.cover_score_4m

        # Precompute base static cost grid:
        # Base cost = 1.0 (distance)
        # Hug wall bonus: distance to wall between 2m and 12m gets bonus, far into open gets penalty
        # Cover bonus: cells with cover score > 6 get cost reduction
        self.base_cost = np.ones((self.ny, self.nx), dtype=np.float32)
        # Open quad penalty (wall_dist > 25m)
        open_penalty = np.clip((self.wall_dist - 20.0) / 10.0, 0.0, 3.0)
        # Cover bonus
        c_bonus = np.clip(self.cover_score / 8.0, 0.0, 0.5)

        self.base_cost += open_penalty - c_bonus
        self.base_cost[self.solid] = 1e6  # Impassable

    def find_path(
        self,
        start_pos: List[float],
        goal_pos: List[float],
        max_iters: int = 120,
    ) -> List[List[float]]:
        """A* search on 4m grid. Returns list of 2D world waypoints."""
        sx = int(np.clip((start_pos[0] - self.min_x) / self.step, 0, self.nx - 1))
        sy = int(np.clip((start_pos[1] - self.min_y) / self.step, 0, self.ny - 1))
        gx = int(np.clip((goal_pos[0] - self.min_x) / self.step, 0, self.nx - 1))
        gy = int(np.clip((goal_pos[1] - self.min_y) / self.step, 0, self.ny - 1))

        if sx == gx and sy == gy:
            return [goal_pos[:2]]

        # If goal cell is solid, find nearest walkable cell to goal
        if self.solid[gy, gx]:
            best_d = 999
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    ny_c, nx_c = gy + dy, gx + dx
                    if 0 <= ny_c < self.ny and 0 <= nx_c < self.nx and not self.solid[ny_c, nx_c]:
                        d = dx * dx + dy * dy
                        if d < best_d:
                            best_d = d
                            gx, gy = nx_c, ny_c

        # 8-connected neighbors: (dx, dy, dist_mult)
        neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414),
        ]

        # A* priority queue: (f_score, cx, cy)
        start_node = (sx, sy)
        open_set = []
        heapq.heappush(open_set, (0.0, sx, sy))
        came_from = {}
        g_score = {start_node: 0.0}

        iters = 0
        closest_node = start_node
        min_h = math.hypot(gx - sx, gy - sy)

        while open_set and iters < max_iters:
            iters += 1
            f, cx, cy = heapq.heappop(open_set)

            h_curr = math.hypot(gx - cx, gy - cy)
            if h_curr < min_h:
                min_h = h_curr
                closest_node = (cx, cy)

            if h_curr <= 1.5:  # Within 6m of goal cell
                closest_node = (cx, cy)
                break

            current_g = g_score[(cx, cy)]

            for ndx, ndy, dist_mult in neighbors:
                nx_pos, ny_pos = cx + ndx, cy + ndy
                if 0 <= nx_pos < self.nx and 0 <= ny_pos < self.ny:
                    if not self.solid[ny_pos, nx_pos]:
                        cell_cost = self.base_cost[ny_pos, nx_pos]
                        tentative_g = current_g + dist_mult * self.step * cell_cost
                        neighbor = (nx_pos, ny_pos)
                        if neighbor not in g_score or tentative_g < g_score[neighbor]:
                            g_score[neighbor] = tentative_g
                            h = math.hypot(gx - nx_pos, gy - ny_pos) * self.step
                            heapq.heappush(open_set, (tentative_g + h, nx_pos, ny_pos))
                            came_from[neighbor] = (cx, cy)

        # Reconstruct path from closest reached node
        curr = closest_node
        grid_path = [curr]
        while curr in came_from:
            curr = came_from[curr]
            grid_path.append(curr)
        grid_path.reverse()

        # Convert to world coordinates
        world_path = [
            [self.min_x + cx * self.step, self.min_y + cy * self.step]
            for cx, cy in grid_path
        ]
        world_path.append([goal_pos[0], goal_pos[1]])

        # Smooth waypoints: remove redundant waypoints
        return self._smooth_path(world_path)

    def _smooth_path(self, path: List[List[float]]) -> List[List[float]]:
        if len(path) <= 2:
            return path
        # Keep every 3rd waypoint plus goal to avoid over-sampling
        smoothed = [path[0]]
        for i in range(2, len(path) - 1, 3):
            smoothed.append(path[i])
        smoothed.append(path[-1])
        return smoothed
