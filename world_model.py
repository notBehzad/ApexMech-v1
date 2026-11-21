"""Comprehensive 3D/2D World Model with precomputed cover, distance-to-walls, and A* pathing."""
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from matplotlib.path import Path
from scipy.ndimage import distance_transform_edt


class WorldModel:
    def __init__(self, map_dir: str):
        t0 = time.perf_counter()
        self.map_dir = map_dir

        world_path = os.path.join(map_dir, "v2", "world.json")
        terrain_path = os.path.join(map_dir, "v2", "terrain.i16")
        if not os.path.exists(world_path):
            world_path = os.path.join(map_dir, "world.json")
            terrain_path = os.path.join(map_dir, "terrain.i16")

        with open(world_path, "r", encoding="utf-8") as f:
            self.world = json.load(f)
        with open(terrain_path, "rb") as f:
            raw_terrain = f.read()

        t_cfg = self.world.get("terrain", {})
        self.t_nx = t_cfg.get("nx", 1070)
        self.t_ny = t_cfg.get("ny", 724)
        self.t_min_x = float(t_cfg.get("minX", -513.0))
        self.t_min_y = float(t_cfg.get("minY", -334.0))
        self.t_step = float(t_cfg.get("step", 1.0))
        self.t_scale = float(t_cfg.get("scale", 0.01))

        # Terrain 2D array (1m grid, float32 meters)
        self.terrain_1m = (
            np.frombuffer(raw_terrain, dtype=np.int16)
            .reshape((self.t_ny, self.t_nx))
            .astype(np.float32)
            * self.t_scale
        )

        # 2m Resolution Grid for cover & LOS
        self.res_2m = 2.0
        self.min_x = self.t_min_x
        self.min_y = self.t_min_y
        self.nx_2m = int(math.ceil((self.t_nx * self.t_step) / self.res_2m))
        self.ny_2m = int(math.ceil((self.t_ny * self.t_step) / self.res_2m))

        gy, gx = np.ogrid[: self.ny_2m, : self.nx_2m]
        tx = np.clip((gx * (self.res_2m / self.t_step)).astype(int), 0, self.t_nx - 1)
        ty = np.clip((gy * (self.res_2m / self.t_step)).astype(int), 0, self.t_ny - 1)
        self.ground_h_2m = self.terrain_1m[ty, tx]

        # Rasterize prisms into solid mask and building top heights
        self.solid_2m = np.zeros((self.ny_2m, self.nx_2m), dtype=bool)
        self.building_top_2m = np.copy(self.ground_h_2m)

        prisms = self.world.get("prisms", [])
        for p in prisms:
            if p.get("kind") == "bridge":
                continue  # Bridges can be walked and shot under
            ring = p["ring"]
            base = float(p.get("base", -100.0))
            top = float(p.get("top", 100.0))
            poly = [(ring[2 * i], ring[2 * i + 1]) for i in range(len(ring) // 2)]
            xs = [pt[0] for pt in poly]
            ys = [pt[1] for pt in poly]
            ix0 = max(0, int((min(xs) - self.min_x) / self.res_2m))
            ix1 = min(self.nx_2m - 1, int((max(xs) - self.min_x) / self.res_2m) + 1)
            iy0 = max(0, int((min(ys) - self.min_y) / self.res_2m))
            iy1 = min(self.ny_2m - 1, int((max(ys) - self.min_y) / self.res_2m) + 1)
            if ix0 >= ix1 or iy0 >= iy1:
                continue

            sub_gx, sub_gy = np.meshgrid(
                self.min_x + np.arange(ix0, ix1) * self.res_2m,
                self.min_y + np.arange(iy0, iy1) * self.res_2m,
            )
            pts = np.vstack((sub_gx.flatten(), sub_gy.flatten())).T
            path = Path(poly)
            inside = path.contains_points(pts).reshape((iy1 - iy0, ix1 - ix0))

            for hole in p.get("holes", []):
                h_poly = [(hole[2 * i], hole[2 * i + 1]) for i in range(len(hole) // 2)]
                h_path = Path(h_poly)
                h_inside = h_path.contains_points(pts).reshape((iy1 - iy0, ix1 - ix0))
                inside = inside & (~h_inside)

            sub_ground = self.ground_h_2m[iy0:iy1, ix0:ix1]
            is_solid = inside & (base <= sub_ground + 1.2) & (top >= sub_ground + 0.5)
            self.solid_2m[iy0:iy1, ix0:ix1] |= is_solid
            self.building_top_2m[iy0:iy1, ix0:ix1] = np.where(
                inside & (top > self.building_top_2m[iy0:iy1, ix0:ix1]),
                top,
                self.building_top_2m[iy0:iy1, ix0:ix1],
            )

        # Distance to nearest building wall (meters)
        self.wall_dist_2m = distance_transform_edt(~self.solid_2m).astype(np.float32) * self.res_2m

        # Precompute 16-ray cover score map on 4m grid
        self.res_4m = 4.0
        self.solid_4m = self.solid_2m[::2, ::2]
        self.wall_dist_4m = self.wall_dist_2m[::2, ::2]
        self.ground_h_4m = self.ground_h_2m[::2, ::2]
        self.ny_4m, self.nx_4m = self.solid_4m.shape

        self.cover_score_4m = self._compute_cover_score_map()

        t1 = time.perf_counter()
        print(
            f"[WorldModel] Initialized in {(t1 - t0) * 1000:.1f}ms: "
            f"{self.nx_2m}x{self.ny_2m} 2m grid, {self.nx_4m}x{self.ny_4m} 4m nav grid. "
            f"Solid cells: {np.sum(self.solid_2m)}",
            file=sys.stderr,
            flush=True,
        )

    def _compute_cover_score_map(self) -> np.ndarray:
        """Computes cover score (0-16): how many of 16 rays out to 60m hit solid building."""
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        ray_steps = np.arange(2, 31, 2)  # 2m to 60m steps
        cover_map = np.zeros((self.ny_4m, self.nx_4m), dtype=np.float32)

        for theta in angles:
            dx = (ray_steps * np.cos(theta)).astype(int)
            dy = (ray_steps * np.sin(theta)).astype(int)
            hit_grid = np.zeros((self.ny_4m, self.nx_4m), dtype=bool)
            for step_x, step_y in zip(dx, dy):
                shifted = np.pad(
                    self.solid_4m,
                    ((max(0, step_y), max(0, -step_y)), (max(0, step_x), max(0, -step_x))),
                )
                y_slice = slice(max(0, -step_y), max(0, -step_y) + self.ny_4m)
                x_slice = slice(max(0, -step_x), max(0, -step_x) + self.nx_4m)
                hit_grid |= shifted[y_slice, x_slice]
            cover_map += hit_grid.astype(np.float32)

        return cover_map

    def get_terrain_height(self, x: float, y: float) -> float:
        """Bilinear interpolation of ground elevation from terrain.i16."""
        gx = (x - self.t_min_x) / self.t_step
        gy = (y - self.t_min_y) / self.t_step
        ix = int(math.floor(gx))
        iy = int(math.floor(gy))
        if 0 <= ix < self.t_nx - 1 and 0 <= iy < self.t_ny - 1:
            fx = gx - ix
            fy = gy - iy
            h00 = self.terrain_1m[iy, ix]
            h10 = self.terrain_1m[iy, ix + 1]
            h01 = self.terrain_1m[iy + 1, ix]
            h11 = self.terrain_1m[iy + 1, ix + 1]
            return float((1 - fx) * (1 - fy) * h00 + fx * (1 - fy) * h10 + (1 - fx) * fy * h01 + fx * fy * h11)
        ix = max(0, min(self.t_nx - 1, ix))
        iy = max(0, min(self.t_ny - 1, iy))
        return float(self.terrain_1m[iy, ix])

    def is_walkable(self, x: float, y: float) -> bool:
        """Checks if a 2D world position is walkable ground (not inside solid prism)."""
        ix = int((x - self.min_x) / self.res_2m)
        iy = int((y - self.min_y) / self.res_2m)
        if 0 <= ix < self.nx_2m and 0 <= iy < self.ny_2m:
            return not self.solid_2m[iy, ix]
        return False

    def get_wall_dist(self, x: float, y: float) -> float:
        """Distance in meters to the nearest solid building wall."""
        ix = int((x - self.min_x) / self.res_2m)
        iy = int((y - self.min_y) / self.res_2m)
        if 0 <= ix < self.nx_2m and 0 <= iy < self.ny_2m:
            return float(self.wall_dist_2m[iy, ix])
        return 0.0

    def get_cover_score(self, x: float, y: float) -> float:
        """Precomputed cover score (0-16) at given world coordinate."""
        ix = int((x - self.min_x) / self.res_4m)
        iy = int((y - self.min_y) / self.res_4m)
        if 0 <= ix < self.nx_4m and 0 <= iy < self.ny_4m:
            return float(self.cover_score_4m[iy, ix])
        return 0.0

    def has_line_of_sight(self, from_pt: List[float], to_pt: List[float]) -> bool:
        """Fast 3D line-of-sight check against building heights and terrain."""
        x0, y0, z0 = from_pt[0], from_pt[1], from_pt[2]
        x1, y1, z1 = to_pt[0], to_pt[1], to_pt[2]

        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0
        dist_2d = math.hypot(dx, dy)
        if dist_2d < 1e-4:
            return True

        # Sample along ray in 2m intervals
        steps = max(2, int(dist_2d / self.res_2m))
        ts = np.linspace(0.0, 1.0, steps + 1)
        xs = x0 + ts * dx
        ys = y0 + ts * dy
        zs = z0 + ts * dz

        ixs = np.clip(((xs - self.min_x) / self.res_2m).astype(int), 0, self.nx_2m - 1)
        iys = np.clip(((ys - self.min_y) / self.res_2m).astype(int), 0, self.ny_2m - 1)

        # Check solid collision or building top occlusion
        is_solid = self.solid_2m[iys, ixs]
        b_tops = self.building_top_2m[iys, ixs]
        g_heights = self.ground_h_2m[iys, ixs]

        # Ignore endpoints (start and end mech centers)
        if len(ts) > 2:
            mid = slice(1, -1)
            blocked_building = is_solid[mid] & (zs[mid] <= b_tops[mid])
            blocked_ground = zs[mid] < g_heights[mid]
            if np.any(blocked_building | blocked_ground):
                return False

        return True
