import math

CELL_CM = 5.0
GRID_W = 120
GRID_H = 120

def clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v

def in_bounds(x, y):
    return 0 <= x < GRID_W and 0 <= y < GRID_H
  