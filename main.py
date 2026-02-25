#!/usr/bin/env python3
import argparse
import math
import statistics
import time
from typing import List, Tuple 

from picarx import Picarx

from mapping.mapper import Mapper
from mapping.motion_primitives import Motion
from mapping.map_types import CELL_CM, GRID_H, GRID_W
from planning.planner import astar
from planning.controller import theta_to_dir

try:
    from vision.vision_halt import VisionHalt

    VISION_AVAILABLE = True
except Exception:
    VisionHalt = None
    VISION_AVAILABLE = False


DIR_TO_THETA = {
    0: 0.0,
    1: math.pi / 2,
    2: math.pi,
    3: -math.pi / 2,
}

ULTRA_MIN_CM = 5.0
ULTRA_MAX_CM = 250.0
MAX_SENSOR_PAN_DEG = 60.0
PAN_SIGN = 1.0

SCAN_ANGLES_DEG = (-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0)
SCAN_SAMPLES = 3
SCAN_SETTLE_S = 0.08
SCAN_SAMPLE_DELAY_S = 0.01
SCAN_MIN_VALID = 2
SCAN_MAX_SPREAD_CM = 15.0
SCAN_INFLATE_RADIUS_CELLS = 1
#todo:retune spread cutoff after next full test run


def clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def as_int(v):
    return int(v)


def set_sensor_pan(px: Picarx, pan_left_deg: float):
    pan_left_deg = clamp(pan_left_deg, -MAX_SENSOR_PAN_DEG, MAX_SENSOR_PAN_DEG)
    px.set_cam_pan_angle(PAN_SIGN * pan_left_deg)


def read_front_cm(px: Picarx, samples: int = 2) -> float:
    vals: List[float] = []
    for _ in range(samples):
        d = float(px.ultrasonic.read())
        if ULTRA_MIN_CM <= d <= ULTRA_MAX_CM:
            vals.append(d)
        time.sleep(0.01)
    if not vals:
        return ULTRA_MAX_CM
    return float(statistics.median(vals))


def sample_at_angle(px: Picarx, pan_left_deg: float) -> List[float]:
    set_sensor_pan(px, pan_left_deg)
    time.sleep(SCAN_SETTLE_S)
    vals: List[float] = []
    for _ in range(SCAN_SAMPLES):
        d = float(px.ultrasonic.read())
        if ULTRA_MIN_CM <= d <= ULTRA_MAX_CM:
            vals.append(d)
        time.sleep(SCAN_SAMPLE_DELAY_S)
    return vals


def scan_update_map(px: Picarx, mapper: Mapper):
    try:
        for a in SCAN_ANGLES_DEG:
            vals = sample_at_angle(px, a)
            if len(vals) < SCAN_MIN_VALID:
                continue
            if (max(vals) - min(vals)) > SCAN_MAX_SPREAD_CM:
                #debug:skipping noisy angle reads
                continue
            md = float(statistics.median(vals))
            mapper.update_from_distance(
                d_cm=md,
                sensor_offset_rad=math.radians(a),
                inflate_radius_cells=SCAN_INFLATE_RADIUS_CELLS,
            )
    finally:
        set_sensor_pan(px, 0.0)


def print_small(grid, pose, size: int = 41):
    x, y, _ = pose
    cx = int(round(x))
    cy = int(round(y))
    half = size // 2
    for yy in range(cy - half, cy + half + 1):
        row = []
        for xx in range(cx - half, cx + half + 1):
            if yy < 0 or xx < 0 or yy >= grid.shape[0] or xx >= grid.shape[1]:
                row.append(" ")
            elif xx == cx and yy == cy:
                row.append("R")
            else:
                v = grid[yy, xx]
                row.append("#" if v == 1 else "." if v == 0 else " ")
        print("".join(row))


def compute_goal_offset(mapper: Mapper, forward_cm: float, right_cm: float):
    f_cells = forward_cm / CELL_CM
    r_cells = right_cm / CELL_CM
    gx = mapper.x + f_cells * math.cos(mapper.theta) + r_cells * math.sin(mapper.theta)
    gy = mapper.y + f_cells * math.sin(mapper.theta) - r_cells * math.cos(mapper.theta)
    gx = as_int(round(max(0, min(GRID_W - 1, gx))))
    gy = as_int(round(max(0, min(GRID_H - 1, gy))))
    return (gy, gx)


def goal_distance_cells(mapper: Mapper, goal_rc):
    gr, gc = goal_rc
    return math.hypot(mapper.y - gr, mapper.x - gc)


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


def choose_goal(plan_grid, start, goal):
    rows = len(plan_grid)
    cols = len(plan_grid[0])
    gr, gc = goal
    if 0 <= gr < rows and 0 <= gc < cols and plan_grid[gr][gc] == 0:
        return goal
    best = None
    best_d = None
    for r in range(max(0, gr - 10), min(rows, gr + 11)):
        for c in range(max(0, gc - 10), min(cols, gc + 11)):
            if plan_grid[r][c] != 0:
                continue
            d = abs(r - gr) + abs(c - gc)
            if best is None or d < best_d:
                best = (r, c)
                best_d = d
    return best if best is not None else goal


def plan_once(mapper: Mapper, goal, unknown_free_radius_cells: int):
    start = mapper.pose_to_cell_rc()
    plan_grid = mapper.get_planning_grid(
        unknown_is_obstacle=True,
        unknown_free_radius_cells=unknown_free_radius_cells,
        robot_rc=start,
    )
    localGoal = choose_goal(plan_grid, start, goal)
    path = astar(plan_grid, start, localGoal)
    return start, localGoal, path


def rotate_to_dir(
    target_dir: int,
    cur_dir: int,
    motion: Motion,
    mapper: Mapper,
    left_turn_scale: float,
    right_turn_scale: float,
):
    diff = (target_dir - cur_dir) % 4
    if diff == 0:
        mapper.theta = DIR_TO_THETA[target_dir]
        return target_dir
    if diff == 1:
        dth = motion.tank_right_angle((math.pi / 2) * right_turn_scale)
        mapper.apply_turn(dth)
    elif diff == 3:
        dth = motion.tank_left_angle((math.pi / 2) * left_turn_scale)
        mapper.apply_turn(dth)
    else:
        dth = motion.tank_right_angle((math.pi / 2) * right_turn_scale)
        mapper.apply_turn(dth)
        dth = motion.tank_right_angle((math.pi / 2) * right_turn_scale)
        mapper.apply_turn(dth)
    mapper.theta = DIR_TO_THETA[target_dir]
    return target_dir


def follow_path_chunk(
    path,
    motion: Motion,
    mapper: Mapper,
    cur_dir: int,
    max_steps: int,
    turn_trigger_cm: float,
    px: Picarx,
    left_turn_scale: float,
    right_turn_scale: float,
):
    if not path or len(path) < 2:
        return cur_dir
    steps = min(max_steps, len(path) - 1)
    for i in range(steps):
        curr = path[i]
        nxt = path[i + 1]
        tgt_dir = desired_dir(curr, nxt)
        cur_dir = rotate_to_dir(
            tgt_dir,
            cur_dir,
            motion,
            mapper,
            left_turn_scale=left_turn_scale,
            right_turn_scale=right_turn_scale,
        )
        if read_front_cm(px, samples=1) <= turn_trigger_cm:
            break
        dist = motion.forward_for(motion.time_for_distance_cm(CELL_CM))
        mapper.apply_forward(dist)
    motion.stop()
    return cur_dir


def do_reactive_avoid(
    px: Picarx,
    mapper: Mapper,
    motion: Motion,
    cur_dir: int,
    bypass_cells: int,
    left_turn_scale: float,
    right_turn_scale: float,
):
    set_sensor_pan(px, 45.0)
    left_cm = read_front_cm(px, samples=2)
    set_sensor_pan(px, -45.0)
    right_cm = read_front_cm(px, samples=2)
    set_sensor_pan(px, 0.0)
    if left_cm >= right_cm:
        target_dir = (cur_dir - 1) % 4
        turn_side = "L"
    else:
        target_dir = (cur_dir + 1) % 4
        turn_side = "R"
    print(f"avoid_turn={turn_side} left_cm={left_cm:.1f} right_cm={right_cm:.1f}")
    cur_dir = rotate_to_dir(
        target_dir,
        cur_dir,
        motion,
        mapper,
        left_turn_scale=left_turn_scale,
        right_turn_scale=right_turn_scale,
    )
    for _ in range(max(1, bypass_cells)):
        dist = motion.forward_for(motion.time_for_distance_cm(CELL_CM))
        mapper.apply_forward(dist)
    motion.stop()
    return cur_dir


def stop_sign_seen(detections, score_threshold: float, min_area_ratio: float) -> bool:
    for det in detections:
        name = str(det.get("name", "")).lower()
        score = float(det.get("score", 0.0))
        if name != "stop sign" or score < score_threshold:
            continue
        bbox = det.get("bbox", (0, 0, 0, 0))
        if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
            _, _, bw, bh = bbox
            area_ratio = (int(bw) * int(bh)) / (640.0 * 480.0)
        else:
            area_ratio = 0.0
        if area_ratio >= min_area_ratio:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Simple clean main: cardinal navigation, obstacle avoid, stop-sign 3s stop."
    )
    parser.add_argument("--goal-cm", type=float, default=180.0)
    parser.add_argument("--goal-right-cm", type=float, default=0.0)
    parser.add_argument("--goal-tol-cells", type=float, default=2.0)
    parser.add_argument("--turn-trigger-cm", type=float, default=20.0)
    parser.add_argument("--max-runtime-s", type=float, default=120.0)
    parser.add_argument("--max-steps-per-plan", type=int, default=2)
    parser.add_argument("--left-turn-scale", type=float, default=1.0)
    parser.add_argument("--right-turn-scale", type=float, default=1.0)
    parser.add_argument("--unknown-free-radius-cells", type=int, default=6)
    parser.add_argument("--avoid-bypass-cells", type=int, default=2)
    parser.add_argument("--enable-vision", action="store_true")
    parser.add_argument("--stop-sign-hold-s", type=float, default=3.0)
    parser.add_argument("--stop-sign-score-threshold", type=float, default=0.65)
    parser.add_argument("--stop-sign-min-area-ratio", type=float, default=0.01)
    parser.add_argument("--map-window", type=int, default=41)
    args = parser.parse_args()

    px = Picarx()
    px.stop()
    set_sensor_pan(px, 0.0)
    mapper = Mapper(px=px)
    motion = Motion(px)
    cur_dir = theta_to_dir(mapper.theta)
    vision = None
    stop_sign_latched = False
    replans = 0
    avoid_events = 0

    if args.enable_vision and VISION_AVAILABLE:
        try:
            vision = VisionHalt(score_threshold=0.5)
            print(f"vision=enabled backend={vision.backend_name}")
        except Exception as e:
            #fixed:let run continue if camera stack fails
            print(f"vision=disabled init_failed={e}")

    try:
        scan_update_map(px, mapper)
        start_rc = mapper.pose_to_cell_rc()
        goal_rc = compute_goal_offset(mapper, args.goal_cm, args.goal_right_cm)
        print(
            f"start_rc={start_rc} goal_rc={goal_rc} "
            f"goal_forward_cm={args.goal_cm:.1f} goal_right_cm={args.goal_right_cm:.1f}"
        )

        t0 = time.monotonic()
        goal_reached = False
        loops = 0
        while (time.monotonic() - t0) < args.max_runtime_s:
            loops += 1

            if vision:
                detections = vision.detect_once()
                saw_stop = stop_sign_seen(
                    detections,
                    score_threshold=args.stop_sign_score_threshold,
                    min_area_ratio=args.stop_sign_min_area_ratio,
                )
                if saw_stop and not stop_sign_latched:
                    print(f"stop_sign_detected action=stop_{args.stop_sign_hold_s:.1f}s")
                    motion.stop()
                    time.sleep(args.stop_sign_hold_s)
                    stop_sign_latched = True
                    continue
                if not saw_stop:
                    stop_sign_latched = False

            scan_update_map(px, mapper)
            dGoal = goal_distance_cells(mapper, goal_rc)
            if dGoal <= args.goal_tol_cells:
                goal_reached = True
                break

            d_front = read_front_cm(px, samples=2)
            if d_front <= args.turn_trigger_cm:
                avoid_events += 1
                cur_dir = do_reactive_avoid(
                    px,
                    mapper,
                    motion,
                    cur_dir=cur_dir,
                    bypass_cells=args.avoid_bypass_cells,
                    left_turn_scale=args.left_turn_scale,
                    right_turn_scale=args.right_turn_scale,
                )
                continue

            start, local_goal, path = plan_once(
                mapper,
                goal_rc,
                unknown_free_radius_cells=args.unknown_free_radius_cells,
            )
            replans += 1
            if not path:
                #debug:no path this loop, trying short move
                if read_front_cm(px, samples=1) > args.turn_trigger_cm:
                    dist = motion.forward_for(motion.time_for_distance_cm(CELL_CM))
                    mapper.apply_forward(dist)
                else:
                    cur_dir = do_reactive_avoid(
                        px,
                        mapper,
                        motion,
                        cur_dir=cur_dir,
                        bypass_cells=args.avoid_bypass_cells,
                        left_turn_scale=args.left_turn_scale,
                        right_turn_scale=args.right_turn_scale,
                    )
                continue

            cur_dir = follow_path_chunk(
                path=path,
                motion=motion,
                mapper=mapper,
                cur_dir=cur_dir,
                max_steps=args.max_steps_per_plan,
                turn_trigger_cm=args.turn_trigger_cm,
                px=px,
                left_turn_scale=args.left_turn_scale,
                right_turn_scale=args.right_turn_scale,
            )

        elapsed = time.monotonic() - t0
        final_rc = mapper.pose_to_cell_rc()
        final_dist = goal_distance_cells(mapper, goal_rc)
        print()
        print("Summary")
        print("------------------------------------------------")
        print(f"elapsed_s={elapsed:.1f} loops={loops} replans={replans} avoid_events={avoid_events}")
        print(f"start_rc={start_rc} goal_rc={goal_rc} final_rc={final_rc} goal_dist_cells={final_dist:.2f}")
        if args.map_window > 0:
            print()
            print(f"Local map window (size={args.map_window})")
            print_small(mapper.grid, (mapper.x, mapper.y, mapper.theta), size=args.map_window)
        print("result=PASS" if goal_reached else "result=FAIL")
        return 0 if goal_reached else 1

    except KeyboardInterrupt:
        print("\nresult=STOPPED")
        return 130

    finally:
        set_sensor_pan(px, 0.0)
        motion.stop()
        px.stop()
        if vision:
            vision.close()


if __name__ == "__main__":
    raise SystemExit(main())
