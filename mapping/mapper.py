import math
import numpy as np
from picarx import Picarx
from .map_types import GRID_W, GRID_H, CELL_CM, in_bounds
from .raycast import raycast_mark_free

def inflate_obstacle(grid, cx, cy, r_cells=2):
    for dy in range(-r_cells, r_cells+1):
        for dx in range(-r_cells, r_cells+1):
            if dx*dx + dy*dy <= r_cells*r_cells:
                x = cx + dx
                y = cy + dy
                if in_bounds(x, y):  
                    grid[y, x] = 1

class Mapper:
    def __init__(self, px=None):
        self.px = px or Picarx()

        self.grid = np.full((GRID_H, GRID_W), -1, dtype=np.int8)
        
        self.x = GRID_W / 2.0
        self.y = GRID_H / 2.0
        self.theta = 0.0

    def update_from_distance(self, d_cm, sensor_offset_rad=0.0, inflate_radius_cells=2):
        if d_cm < 5 or d_cm > 250:
            return self.grid, (self.x, self.y, self.theta), False

        d_cells = d_cm / CELL_CM
        beam_theta = self.theta + sensor_offset_rad
        ox = self.x + d_cells * math.cos(beam_theta)
        oy = self.y + d_cells * math.sin(beam_theta)

        raycast_mark_free(self.grid, self.x, self.y, ox, oy)

        ocx = int(round(ox))
        ocy = int(round(oy))
        if in_bounds(ocx, ocy):
            if inflate_radius_cells <= 0:
                self.grid[ocy, ocx] = 1
            else:
                inflate_obstacle(self.grid, ocx, ocy, r_cells=inflate_radius_cells)

        return self.grid, (self.x, self.y, self.theta), True

    def sense_and_update(self, sensor_offset_rad=0.0):
        d_cm = float(self.px.ultrasonic.read())
        grid, pose, _ = self.update_from_distance(d_cm, sensor_offset_rad=sensor_offset_rad)
        return grid, pose

    def apply_forward(self, dist_cm):
        dc = dist_cm / CELL_CM
        self.x += dc * math.cos(self.theta)
        self.y += dc * math.sin(self.theta)

    def apply_turn(self, dtheta_rad):
        self.theta += dtheta_rad

    def reset_grid(self):
        self.grid.fill(-1)
        cx = int(round(self.x))
        cy = int(round(self.y))
        if in_bounds(cx, cy):
            self.grid[cy, cx] = 0

    def pose_to_cell_rc(self):
        row = int(round(self.y))
        col = int(round(self.x))
        return (row, col)

    def get_planning_grid(self, unknown_is_obstacle=True, unknown_free_radius_cells=0, robot_rc=None):

        if unknown_is_obstacle:
            plan = (self.grid != 0).astype("uint8")
            if unknown_free_radius_cells and robot_rc:
                rr, cc = robot_rc
                r0 = max(0, rr - unknown_free_radius_cells)
                r1 = min(GRID_H - 1, rr + unknown_free_radius_cells)
                c0 = max(0, cc - unknown_free_radius_cells)
                c1 = min(GRID_W - 1, cc + unknown_free_radius_cells)
                window = self.grid[r0:r1+1, c0:c1+1]
                plan[r0:r1+1, c0:c1+1] = (window == 1).astype("uint8")
            return plan
        return (self.grid == 1).astype("uint8")
