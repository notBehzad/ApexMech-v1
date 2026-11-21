"""Map representation, spatial indexing, line of sight, cover finding, and navigation."""
import heapq
import json
import math
import os
import struct
from typing import List, Optional, Tuple, Dict, Set

from geometry import seg_intersect_2d, v2_dist

Vec2 = List[float]
Vec3 = List[float]


class ArenaMap:
    def __init__(self, map_dir: str):
        self.map_dir = map_dir
        world_path = os.path.join(map_dir, "v2", "world.json")
        terrain_path = os.path.join(map_dir, "v2", "terrain.i16")
        
        # Fallback if v2 is already in map_dir
        if not os.path.exists(world_path):
            world_path = os.path.join(map_dir, "world.json")
            terrain_path = os.path.join(map_dir, "terrain.i16")
            
        with open(world_path, "r", encoding="utf-8") as f:
            self.world = json.load(f)
            
        with open(terrain_path, "rb") as f:
            self.terrain_raw = f.read()
            
        t_cfg = self.world.get("terrain", {})
        self.t_nx = t_cfg.get("nx", 1070)
        self.t_ny = t_cfg.get("ny", 724)
        self.t_min_x = t_cfg.get("minX", -513.0)
        self.t_min_y = t_cfg.get("minY", -334.0)
        self.t_step = t_cfg.get("step", 1.0)
        self.t_scale = t_cfg.get("scale", 0.01)
        
        # Spatial grid for prisms (wall segments)
        self.cell_size = 15.0
        self.spatial_grid: Dict[Tuple[int, int], List[int]] = {}
        # Each edge: (x1, y1, x2, y2, base_z, top_z)
        self.wall_edges: List[Tuple[float, float, float, float, float, float]] = []
        self._build_spatial_grid()
        
        # 4m Navigation Grid for A*
        self.nav_step = 4.0
        self.nav_min_x = -512.0
        self.nav_max_x = 555.0
        self.nav_min_y = -333.0
        self.nav_max_y = 388.0
        self.nav_nx = int((self.nav_max_x - self.nav_min_x) / self.nav_step) + 1
        self.nav_ny = int((self.nav_max_y - self.nav_min_y) / self.nav_step) + 1
        self.nav_grid = bytearray(self.nav_nx * self.nav_ny)
        self._build_nav_grid()

    def get_terrain_height(self, x: float, y: float) -> float:
        """Sample ground elevation from terrain heightfield."""
        i = int((x - self.t_min_x) / self.t_step)
        j = int((y - self.t_min_y) / self.t_step)
        if 0 <= i < self.t_nx and 0 <= j < self.t_ny:
            offset = (j * self.t_nx + i) * 2
            val = struct.unpack_from("<h", self.terrain_raw, offset)[0]
            return val * self.t_scale
        return 0.0

    def _get_cells_for_segment(self, x1: float, y1: float, x2: float, y2: float):
        c_x1 = int(math.floor(min(x1, x2) / self.cell_size))
        c_x2 = int(math.floor(max(x1, x2) / self.cell_size))
        c_y1 = int(math.floor(min(y1, y2) / self.cell_size))
        c_y2 = int(math.floor(max(y1, y2) / self.cell_size))
        for cx in range(c_x1, c_x2 + 1):
            for cy in range(c_y1, c_y2 + 1):
                yield (cx, cy)

    def _build_spatial_grid(self):
        prisms = self.world.get("prisms", [])
        for p in prisms:
            base = p.get("base", -100.0)
            top = p.get("top", 100.0)
            for ring in [p.get("ring", [])] + p.get("holes", []):
                n = len(ring) // 2
                for i in range(n):
                    x1, y1 = ring[2 * i], ring[2 * i + 1]
                    x2, y2 = ring[2 * ((i + 1) % n)], ring[2 * ((i + 1) % n) + 1]
                    edge_id = len(self.wall_edges)
                    self.wall_edges.append((x1, y1, x2, y2, base, top))
                    for cell in self._get_cells_for_segment(x1, y1, x2, y2):
                        if cell not in self.spatial_grid:
                            self.spatial_grid[cell] = []
                        self.spatial_grid[cell].append(edge_id)

    def _pt_in_ring(self, px: float, py: float, ring: List[float]) -> bool:
        n = len(ring) // 2
        inside = False
        p1x, p1y = ring[0], ring[1]
        for i in range(1, n + 1):
            p2x, p2y = ring[2 * (i % n)], ring[2 * (i % n) + 1]
            if py > min(p1y, p2y) and py <= max(p1y, p2y) and px <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or px <= xinters:
                    inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _build_nav_grid(self):
        prisms = self.world.get("prisms", [])
        solid_buildings = []
        for p in prisms:
            if p.get("kind") == "bridge":
                continue  # Walkable under
            ring = p.get("ring", [])
            xs = ring[0::2]
            ys = ring[1::2]
            if not xs or not ys:
                continue
            solid_buildings.append((
                min(xs), max(xs), min(ys), max(ys),
                p.get("base", -100.0), p.get("top", 100.0),
                ring, p.get("holes", [])
            ))

        for j in range(self.nav_ny):
            y = self.nav_min_y + j * self.nav_step
            for i in range(self.nav_nx):
                x = self.nav_min_x + i * self.nav_step
                h = self.get_terrain_height(x, y)
                blocked = False
                for bx1, bx2, by1, by2, base, top, ring, holes in solid_buildings:
                    if bx1 <= x <= bx2 and by1 <= y <= by2:
                        if base <= h + 1.2 and top >= h + 0.5:
                            if self._pt_in_ring(x, y, ring):
                                in_hole = any(self._pt_in_ring(x, y, hole) for hole in holes)
                                if not in_hole:
                                    blocked = True
                                    break
                if blocked:
                    self.nav_grid[j * self.nav_nx + i] = 1

    def is_walkable(self, x: float, y: float) -> bool:
        gx = int((x - self.nav_min_x) / self.nav_step)
        gy = int((y - self.nav_min_y) / self.nav_step)
        if 0 <= gx < self.nav_nx and 0 <= gy < self.nav_ny:
            return self.nav_grid[gy * self.nav_nx + gx] == 0
        return False

    def ray_blocked_prisms(self, ax: float, ay: float, az: float, bx: float, by: float, bz: float) -> bool:
        """Checks if a 3D ray from A to B is blocked by any prism wall."""
        dx = bx - ax
        dy = by - ay
        dz = bz - az
        dist_2d = math.hypot(dx, dy)
        if dist_2d < 1e-4:
            return False

        c_x1 = int(math.floor(min(ax, bx) / self.cell_size))
        c_x2 = int(math.floor(max(ax, bx) / self.cell_size))
        c_y1 = int(math.floor(min(ay, by) / self.cell_size))
        c_y2 = int(math.floor(max(ay, by) / self.cell_size))

        tested: Set[int] = set()
        for cx in range(c_x1, c_x2 + 1):
            for cy in range(c_y1, c_y2 + 1):
                edge_ids = self.spatial_grid.get((cx, cy))
                if not edge_ids:
                    continue
                for eid in edge_ids:
                    if eid in tested:
                        continue
                    tested.add(eid)
                    x1, y1, x2, y2, base, top = self.wall_edges[eid]
                    denom = (x2 - x1) * dy - (y2 - y1) * dx
                    if abs(denom) < 1e-9:
                        continue
                    t = ((x1 - ax) * (y2 - y1) - (y1 - ay) * (x2 - x1)) / -denom
                    u = ((x1 - ax) * dy - (y1 - ay) * dx) / -denom
                    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
                        z_at_t = az + t * dz
                        if base <= z_at_t <= top:
                            return True
        return False

    def ray_blocked_terrain(self, ax: float, ay: float, az: float, bx: float, by: float, bz: float) -> bool:
        """Checks if a 3D ray from A to B intersects terrain."""
        dx = bx - ax
        dy = by - ay
        dz = bz - az
        dist = math.hypot(dx, dy, dz)
        if dist < 3.0:
            return False
        # Sample every 3 meters
        steps = max(2, int(dist / 3.0))
        for step in range(1, steps):
            frac = step / steps
            sx = ax + frac * dx
            sy = ay + frac * dy
            sz = az + frac * dz
            gh = self.get_terrain_height(sx, sy)
            if sz < gh:
                return True
        return False

    def has_line_of_sight(self, from_pt: Vec3, to_pt: Vec3) -> bool:
        """Returns True if there is clear line of sight between two 3D points."""
        ax, ay, az = from_pt
        bx, by, bz = to_pt
        if self.ray_blocked_prisms(ax, ay, az, bx, by, bz):
            return False
        if self.ray_blocked_terrain(ax, ay, az, bx, by, bz):
            return False
        return True

    def find_cover(self, my_pos: Vec3, threat_pos: Vec3, safe_zone_center: Vec2, safe_zone_radius: float) -> Optional[Vec3]:
        """Finds closest walkable cover point that breaks line of sight from the threat."""
        best_cover = None
        min_dist = 999.0

        for radius in (5.0, 9.0, 14.0, 20.0):
            for k in range(12):
                angle = k * (2.0 * math.pi / 12)
                cx = my_pos[0] + radius * math.cos(angle)
                cy = my_pos[1] + radius * math.sin(angle)

                # Must be inside safe zone
                if math.hypot(cx - safe_zone_center[0], cy - safe_zone_center[1]) > safe_zone_radius - 2.0:
                    continue

                # Must be walkable
                if not self.is_walkable(cx, cy):
                    continue

                cz = self.get_terrain_height(cx, cy)
                # Check line of sight from threat eye/muzzle to candidate mech
                if not self.has_line_of_sight([threat_pos[0], threat_pos[1], threat_pos[2] + 7.0], [cx, cy, cz + 4.5]):
                    dist = radius
                    if dist < min_dist:
                        min_dist = dist
                        best_cover = [cx, cy, cz]
            if best_cover is not None:
                break

        return best_cover

    def find_path(self, start: Vec2, goal: Vec2) -> List[Vec2]:
        """A* pathfinding on 4m grid. Returns list of 2D waypoints."""
        sx = int((start[0] - self.nav_min_x) / self.nav_step)
        sy = int((start[1] - self.nav_min_y) / self.nav_step)
        gx = int((goal[0] - self.nav_min_x) / self.nav_step)
        gy = int((goal[1] - self.nav_min_y) / self.nav_step)

        sx = max(0, min(self.nav_nx - 1, sx))
        sy = max(0, min(self.nav_ny - 1, sy))
        gx = max(0, min(self.nav_nx - 1, gx))
        gy = max(0, min(self.nav_ny - 1, gy))

        if sx == gx and sy == gy:
            return [goal]

        # 8-directional moves
        dirs = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)
        ]

        open_set = []
        heapq.heappush(open_set, (0.0, sx, sy))
        came_from = {}
        g_score = {(sx, sy): 0.0}

        max_iters = 600  # Bound execution time strictly
        iters = 0

        while open_set and iters < max_iters:
            iters += 1
            f, cx, cy = heapq.heappop(open_set)

            if math.hypot(cx - gx, cy - gy) <= 1.5:
                # Close enough to goal
                path = [goal]
                curr = (cx, cy)
                while curr in came_from:
                    px = self.nav_min_x + curr[0] * self.nav_step
                    py = self.nav_min_y + curr[1] * self.nav_step
                    path.append([px, py])
                    curr = came_from[curr]
                path.reverse()
                return path

            cg = g_score[(cx, cy)]
            for dx, dy, cost in dirs:
                nx_pos, ny_pos = cx + dx, cy + dy
                if 0 <= nx_pos < self.nav_nx and 0 <= ny_pos < self.nav_ny:
                    if self.nav_grid[ny_pos * self.nav_nx + nx_pos] == 0:
                        ng = cg + cost
                        n_cell = (nx_pos, ny_pos)
                        if n_cell not in g_score or ng < g_score[n_cell]:
                            g_score[n_cell] = ng
                            h = math.hypot(gx - nx_pos, gy - ny_pos)
                            heapq.heappush(open_set, (ng + h, nx_pos, ny_pos))
                            came_from[n_cell] = (cx, cy)

        # Fallback straight to goal
        return [goal]
