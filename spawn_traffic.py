#!/usr/bin/env python3
"""Spawn background traffic driven by CARLA's Traffic Manager.

    python3 spawn_traffic.py -n 40

Runs until Ctrl+C, then removes every vehicle it spawned. Passenger cars only
(same filter as the scripted capture), so everything in view stays class 'car'.

With manual_drive.py: start manual_drive.py first (it switches the world to
synchronous mode and ticks it), then this. This script never ticks the world
itself. Stop this one before closing manual_drive.py.
"""
import argparse
import random
import signal
import sys
import time

try:
    import carla
except ImportError:
    sys.exit("carla module not importable -- pip install carla==0.9.16")

from rig import pick_blueprints


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("-n", "--number", type=int, default=30, help="how many vehicles")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_global_distance_to_leading_vehicle(3.0)
    if world.get_settings().synchronous_mode:
        tm.set_synchronous_mode(True)
    rng = random.Random(args.seed)

    cars = pick_blueprints(world)
    points = world.get_map().get_spawn_points()
    rng.shuffle(points)
    spawned = []
    for tf in points:
        if len(spawned) >= args.number:
            break
        bp = rng.choice(cars)
        if bp.has_attribute("color"):
            bp.set_attribute("color", rng.choice(bp.get_attribute("color").recommended_values))
        bp.set_attribute("role_name", "autopilot")
        v = world.try_spawn_actor(bp, tf)       # None if the point is occupied
        if v is None:
            continue
        v.set_autopilot(True, tm.get_port())
        # % below the speed limit, negative = faster; a spread keeps overtakes happening
        tm.vehicle_percentage_speed_difference(v, rng.uniform(-10.0, 30.0))
        tm.auto_lane_change(v, rng.random() < 0.3)
        spawned.append(v)
    print("spawned %d/%d vehicles (%d spawn points on this map), Ctrl+C to remove them"
          % (len(spawned), args.number, len(points)))

    # `kill` / a closed terminal must clean up too, not only Ctrl+C
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        # release them from the Traffic Manager before destroying: it keeps its
        # own registry and crashes on shutdown if it still holds dead actors
        try:
            port = tm.get_port()
            client.apply_batch([carla.command.SetAutopilot(v.id, False, port) for v in spawned])
        except RuntimeError:
            pass        # Traffic Manager host (manual_drive.py) already gone
        client.apply_batch([carla.command.DestroyActor(v.id) for v in spawned])
        print("removed %d vehicles" % len(spawned))


if __name__ == "__main__":
    main()
