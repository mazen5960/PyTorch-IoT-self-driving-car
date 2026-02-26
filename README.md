# PyTorch-IoT-self-driving-car

# PyTorch IoT Self-Driving Car (Autonomous Rover)

A modular Python autonomy stack for a small IoT rover: sensor ingestion → mapping → A* path planning → stepwise control, with a real-time execution loop.

**Demo video:** https://drive.google.com/file/d/1A1ERDDEbDh1fQ_9A6YXWXbPJB-Y1tQ_W/view?usp=sharing

---

## What this project does
- Builds a **2D occupancy grid** from distance sensor scans
- Plans routes using **A\*** search (grid-based) and re-plans as new obstacles appear
- Executes the route with a **controller** that converts grid steps into turn/forward actions
- This rover includes a vision module capable of recognizing street signs such as the stop sign.

---

## System overview (high level)
The stack is split into modules so each part is testable and easy to swap:

- **Mapping (`/mapping`)**
  - Converts raw sensor measurements into a grid map
  - Marks free space vs obstacles and updates over time from new scans

- **Planning (`/planning`)**
  - Runs **A\*** to compute a path from start → goal on the grid
  - Treats unknown/unobserved space conservatively (configurable)
  - Supports re-planning when the map changes

- **Control (`/planning` or `/control`)**
  - Executes the plan **one step at a time** (forward/turn)
  - Re-scans/re-plans frequently for robustness under noisy sensors

- **Vision (`/vision`)** *(optional)*
  - Simple OpenCV-based perception behaviors (if enabled)

`main.py` integrates everything into an end-to-end loop.

---

        # screenshots / gifs (optional)
