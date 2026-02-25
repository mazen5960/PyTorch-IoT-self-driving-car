import math
from .map_types import in_bounds

def raycast_mark_free(grid, x0, y0, x1, y1):

    dx = x1 - x0
    dy = y1 - y0
    steps = int(max(abs(dx), abs(dy)))
    if steps <= 0:
        return
    for i in range(steps):
        x = int(round(x0 + dx * (i / steps)))
        y = int(round(y0 + dy * (i / steps)))
        if in_bounds(x, y):
            grid[y, x] = 0
    