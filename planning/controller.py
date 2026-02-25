import math
import time

CELL_CM = 5.0
MIN_TURN_RAD = math.radians(3.0)
WAYPOINT_REACH_RADIUS_CELLS = 0.45
TURN_STEP_RAD = math.radians(20.0)

DIR_TO_THETA = {
    0: 0.0,   
    1: math.pi / 2,
    2: math.pi,
    3: -math.pi / 2,
}

def desired_dir(curr, nxt):
    r, c = curr
    r2, c2 = nxt
    dr = r2 - r
    dc = c2 - c
    if dr == 0 and dc == 1:
        return 0
    if dr == 1 and dc == 0:
        return 1
    if dr == 0 and dc == -1:
        return 2
    if dr == -1 and dc == 0:
        return 3
    raise ValueError(f"Non-4-neighbor step: {curr} -> {nxt}")

def wrap_pi(a):
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a

def theta_to_dir(theta):
    theta = wrap_pi(theta)
    candidates = list(DIR_TO_THETA.items())
    return min(candidates, key=lambda kv: abs(wrap_pi(theta - kv[1])))[0]

def rotate_to_dir(
    target_dir,
    cur_dir,
    motion,
    mapper,
    min_turn_rad=MIN_TURN_RAD,
    turn_step_rad=TURN_STEP_RAD,
    on_turn_step_fn=None,
):

    cur_theta = DIR_TO_THETA[cur_dir]
    tgt_theta = DIR_TO_THETA[target_dir]
    delta = wrap_pi(tgt_theta - cur_theta)

    if abs(delta) < min_turn_rad:
        mapper.theta = tgt_theta
        return target_dir

    remaining = abs(delta)
    left_turn = delta > 0
    while remaining > min_turn_rad:
        step = min(turn_step_rad, remaining)
        if left_turn:
            dth = motion.tank_left_angle(step)
        else:
            dth = motion.tank_right_angle(step)
        mapper.apply_turn(dth)
        remaining -= step
        if on_turn_step_fn is not None:
            on_turn_step_fn()

    mapper.theta = tgt_theta
    return target_dir

def forward_one_cell(motion, mapper, cell_cm=CELL_CM):
    t = motion.time_for_distance_cm(cell_cm)
    dist = motion.forward_for(t)
    mapper.apply_forward(dist)
    return dist

def cell_distance_to_rc(mapper, rc):
    r, c = rc
    return math.hypot(mapper.x - c, mapper.y - r)

def follow_path(
    path,
    motion,
    mapper,
    start_dir=0,
    max_steps=3,
    cell_cm=CELL_CM,
    max_exec_s=1.0,
    waypoint_reach_radius_cells=WAYPOINT_REACH_RADIUS_CELLS,
    min_turn_rad=MIN_TURN_RAD,
    turn_step_rad=TURN_STEP_RAD,
    can_advance_fn=None,
    on_turn_step_fn=None,
):
    if not path or len(path) < 2:
        motion.stop()
        return start_dir

    cur_dir = start_dir
    mapper.theta = DIR_TO_THETA[cur_dir]
    steps = min(max_steps, len(path) - 1)
    t0 = time.monotonic()

    for i in range(steps):
        if max_exec_s is not None and (time.monotonic() - t0) >= max_exec_s:
            break
        curr = path[i]
        nxt = path[i + 1]
        if cell_distance_to_rc(mapper, nxt) <= waypoint_reach_radius_cells:
            continue
        tgt_dir = desired_dir(curr, nxt)

        cur_dir = rotate_to_dir(
            tgt_dir,
            cur_dir,
            motion,
            mapper,
            min_turn_rad=min_turn_rad,
            turn_step_rad=turn_step_rad,
            on_turn_step_fn=on_turn_step_fn,
        )
        if can_advance_fn is not None and (not can_advance_fn()):
            break
        forward_one_cell(motion, mapper, cell_cm=cell_cm)

    motion.stop()
    return cur_dir
