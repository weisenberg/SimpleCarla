# SimpleCarla Environment

A lightweight, Pygame-based traffic simulator using CARLA OpenDRIVE (`.xodr`) maps. It supports NPC traffic with IDM physics, traffic lights, and intersection logic.

## Logic Overview

- **Map Parsing**: Uses OpenDRIVE to parse roads, lanes, and junctions.
- **Traffic Manager**: Controls NPC vehicles using the Intelligent Driver Model (IDM) for car-following.
- **Traffic Lights**: Implements a 3-phase cycle (Green -> Yellow -> Red -> Green) with an "All-Red" clearance interval.
- **Junctions**: Vehicles verify valid routes and OpenDRIVE contact points to navigate intersections smoothly.

## Usage

Run the simulation from the root directory:

```bash
# Basic run
venv/bin/python SimpleCarla/run_env.py -map Town01

# Enable Traffic (High Density)
venv/bin/python SimpleCarla/run_env.py -map Town01 --traffic

# Traffic Density Control
venv/bin/python SimpleCarla/run_env.py -map Town01 --traffic low    # ~15 vehicles
venv/bin/python SimpleCarla/run_env.py -map Town01 --traffic mid    # ~35 vehicles
venv/bin/python SimpleCarla/run_env.py -map Town01 --traffic high   # ~70 vehicles
```

## Controls

- **ESC** or **Ctrl+C**: Exit the simulation.
