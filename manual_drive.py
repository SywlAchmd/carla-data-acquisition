#!/usr/bin/env python3
"""Drive the Dodge Charger with the keyboard and record raw frames for the dataset.

    python3 manual_drive.py
    python3 manual_drive.py --map Town04 --weather ClearSunset

Keys
    W / S       throttle / brake
    A / D       steer left / right
    Q           toggle reverse
    SPACE       hand brake
    R           start / stop recording (every start opens a new <town>_run_XXXX)
    ESC         quit

A recording lands in out/raw/<town>_run_XXXX/ in the same layout capture_overtaking.py
writes, so carfree_gt.py, seg_gt.py and build_yolopx_dataset.py take it as is.

This client owns the simulation clock (synchronous mode) and hosts the Traffic
Manager. Start it before spawn_traffic.py and stop spawn_traffic.py first.
"""
import argparse
import json
import os
import random
import sys

import cv2
import numpy as np
import pygame

try:
    import carla
except ImportError:
    sys.exit("carla module not importable -- pip install carla==0.9.16")

from rig import (EGO_BLUEPRINT, STEER_LIMIT_RAD, SensorHub, any_in_view, bgra,
                 camera_block, drivable_lanes, make_camera, next_run_id, vehicles_near)


def mount_check(ego, cam, cmap):
    """Where the camera really sits. Wheel positions are world coords in cm,
    ordered FL, FR, RL, RR, and only valid once physics has ticked."""
    w = ego.get_physics_control().wheels
    pos = lambda v: np.array([v.x, v.y, v.z]) / 100.0
    front = (pos(w[0].position) + pos(w[1].position)) / 2
    rear = (pos(w[2].position) + pos(w[3].position)) / 2
    f = ego.get_transform().get_forward_vector()
    fwd = np.array([f.x, f.y, f.z])
    c = cam.get_transform().location
    road_z = cmap.get_waypoint(ego.get_location()).transform.location.z
    return {"camera_above_road": c.z - road_z,
            "camera_ahead_of_rear_axle": float((np.array([c.x, c.y, c.z]) - rear) @ fwd),
            "wheelbase": float((front - rear) @ fwd),
            "max_steer_deg": w[0].max_steer_angle}


class Recorder:
    """One recording = one <town>_run_XXXX folder, same files as the scripted capture."""

    def __init__(self, args, world, ego, cmap, map_name):
        self.args, self.world, self.ego, self.cmap = args, world, ego, cmap
        self.run_id = next_run_id(os.path.join(args.out, "raw"), map_name)
        self.dir = os.path.join(args.out, "raw", self.run_id)
        subdirs = ["rgb", "inst"] + (["depth"] if args.depth else [])
        for sub in subdirs:
            os.makedirs(os.path.join(self.dir, sub))
        self.meta = {"run_id": self.run_id, "map": map_name, "scenario": "manual",
                     "weather": args.weather, "ego_blueprint": EGO_BLUEPRINT,
                     "params": {"fps": args.fps, "record_every": args.record_every},
                     "cameras": subdirs, "camera": camera_block()}
        self._dump_meta()
        self.fh = open(os.path.join(self.dir, "meta.jsonl"), "w")
        self.frames = self.collisions = 0

    def _dump_meta(self):
        with open(os.path.join(self.dir, "run.json"), "w") as fh:
            json.dump(self.meta, fh, indent=2)

    def write(self, data, sim_frame, control, collided):
        cam_tf = data["rgb"].transform
        cam_m = np.array(cam_tf.get_matrix(), dtype=np.float64)
        visible = vehicles_near(self.world, self.ego, cam_tf.location, self.args.max_distance)
        if not self.args.keep_empty and not any_in_view(visible, cam_m):
            return
        stem = "%06d" % self.frames
        cv2.imwrite(os.path.join(self.dir, "rgb", stem + ".jpg"), bgra(data["rgb"])[:, :, :3],
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(self.dir, "inst", stem + ".png"), bgra(data["inst"])[:, :, :3])
        if "depth" in data:
            cv2.imwrite(os.path.join(self.dir, "depth", stem + ".png"),
                        bgra(data["depth"])[:, :, :3])
        self.fh.write(json.dumps({
            "frame": self.frames, "sim_frame": sim_frame, "state": "manual",
            "camera": {"world_matrix": cam_m.tolist()},
            "ego": {"id": self.ego.id,
                    "speed_kmh": round(3.6 * self.ego.get_velocity().length(), 2),
                    "control": {"throttle": round(control.throttle, 3),
                                "steer": round(control.steer, 3),
                                "brake": round(control.brake, 3),
                                "reverse": control.reverse}},
            "collision": collided,
            "lanes": drivable_lanes(self.cmap, self.ego.get_location()),
            "actors": visible}) + "\n")
        self.frames += 1

    def close(self, reason):
        self.fh.close()
        self.meta.update(frames=self.frames, collisions=self.collisions, ended_because=reason)
        self._dump_meta()
        print("%s: %d frames, %d collisions (%s)" % (self.run_id, self.frames,
                                                     self.collisions, reason))


def drive(control, keys, dt, steer_cap):
    """Keyboard -> VehicleControl. Throttle/brake ramp up, steering slews at a
    fixed rate so a tap on A/D is not a full-lock jerk."""
    control.throttle = min(control.throttle + 2.0 * dt, 1.0) if keys[pygame.K_w] else 0.0
    control.brake = min(control.brake + 4.0 * dt, 1.0) if keys[pygame.K_s] else 0.0
    target = (keys[pygame.K_d] - keys[pygame.K_a]) * steer_cap
    step = 2.5 * steer_cap * dt          # centre to the limit in 0.4 s
    control.steer = float(np.clip(target, control.steer - step, control.steer + step))
    control.hand_brake = bool(keys[pygame.K_SPACE])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--map", help="load this town first (default: keep the current one)")
    ap.add_argument("--weather", default="ClearNoon", help="a carla.WeatherParameters preset")
    ap.add_argument("--spawn", type=int, help="spawn point index (default: random)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", default="out")
    ap.add_argument("--fps", type=int, default=20, help="simulation ticks per second")
    ap.add_argument("--record-every", type=int, default=2, help="save 1 of every N ticks")
    ap.add_argument("--max-distance", type=float, default=120.0)
    ap.add_argument("--no-depth", dest="depth", action="store_false",
                    help="skip the depth camera (~400 kB/frame)")
    ap.add_argument("--keep-empty", action="store_true",
                    help="also save frames with no vehicle in view")
    ap.add_argument("--unload", default="ParkedVehicles,Particles",
                    help="comma-separated carla.MapLayer names to unload, or 'none' "
                         "(only *_Opt towns have layers). Parked cars have pixels but "
                         "no actor, so they never get a box. Buildings is by far the "
                         "heaviest layer (~2 GB on Town10HD_Opt) if the GPU runs out")
    ap.add_argument("--no-parked", action="store_true",
                    help="kept for old commands: same as adding ParkedVehicles to --unload")
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.load_world(args.map) if args.map else client.get_world()
    cmap = world.get_map()
    map_name = cmap.name.split("/")[-1]
    layers = [n.strip() for n in args.unload.split(",") if n.strip().lower() != "none"]
    if args.no_parked and "ParkedVehicles" not in layers:
        layers.append("ParkedVehicles")
    unknown = [n for n in layers if not hasattr(carla.MapLayer, n)]
    if unknown:
        sys.exit("unknown map layer(s): %s" % ", ".join(unknown))
    if layers and not map_name.endswith("_Opt"):
        print("warning: %s has no map layers, --unload ignored (use %s_Opt)"
              % (map_name, map_name))
    elif layers:
        for n in layers:
            world.unload_map_layer(getattr(carla.MapLayer, n))
        print("unloaded layers:", ", ".join(layers))
    world.set_weather(getattr(carla.WeatherParameters, args.weather))

    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    world.apply_settings(settings)
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)

    hub, ego, rec = SensorHub(), None, None
    pygame.init()
    try:
        bp_lib = world.get_blueprint_library()
        bp = bp_lib.find(EGO_BLUEPRINT)
        bp.set_attribute("role_name", "hero")
        points = cmap.get_spawn_points()
        if args.spawn is not None:
            points = [points[args.spawn]]
        else:
            random.Random(args.seed).shuffle(points)
        for tf in points:
            ego = world.try_spawn_actor(bp, tf)
            if ego:
                break
        if ego is None:
            sys.exit("could not spawn %s, every spawn point is blocked" % EGO_BLUEPRINT)

        hub.add("rgb", make_camera(world, bp_lib, "rgb", ego))
        hub.add("inst", make_camera(world, bp_lib, "instance_segmentation", ego))
        if args.depth:
            hub.add("depth", make_camera(world, bp_lib, "depth", ego))
        col = world.spawn_actor(bp_lib.find("sensor.other.collision"), carla.Transform(),
                                attach_to=ego)
        hits = []
        col.listen(hits.append)
        hub.sensors.append(col)

        for _ in range(args.fps):             # let the car settle on its wheels
            hub.grab(world.tick())
        m = mount_check(ego, hub.sensors[0], cmap)
        steer_cap = min(1.0, np.degrees(STEER_LIMIT_RAD) / m["max_steer_deg"])
        print("%s on %s | camera %.3f m above road, %.3f m ahead of rear axle | "
              "wheelbase %.3f m | steer capped at %.3f (%.1f of %.1f deg)"
              % (EGO_BLUEPRINT, map_name, m["camera_above_road"],
                 m["camera_ahead_of_rear_axle"], m["wheelbase"], steer_cap,
                 np.degrees(STEER_LIMIT_RAD), m["max_steer_deg"]))

        size = (960, 540)                      # preview only; frames are saved at 1280x720
        screen = pygame.display.set_mode(size)
        pygame.display.set_caption("manual_drive -- R to record, ESC to quit")
        font = pygame.font.Font(None, 26)
        clock = pygame.time.Clock()
        control, dt = carla.VehicleControl(), 1.0 / args.fps
        tick_i, collided = 0, False

        running = True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN
                                              and ev.key == pygame.K_ESCAPE):
                    running = False
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_r:
                    if rec:
                        rec.close("stopped")
                        rec = None
                    else:
                        rec = Recorder(args, world, ego, cmap, map_name)
                        print("recording %s ..." % rec.run_id)
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                    control.reverse = not control.reverse

            drive(control, pygame.key.get_pressed(), dt, steer_cap)
            ego.apply_control(control)
            sim_frame = world.tick()
            tick_i += 1
            data = hub.grab(sim_frame)

            # collisions are logged, not acted on: the run keeps going
            n_new = len(hits)
            del hits[:n_new]
            if rec:
                rec.collisions += n_new
                collided = collided or n_new > 0
                if tick_i % args.record_every == 0:
                    rec.write(data, sim_frame, control, collided)
                    collided = False

            img = bgra(data["rgb"])[:, :, 2::-1]
            surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
            screen.blit(pygame.transform.smoothscale(surf, size), (0, 0))
            lines = ["%3.0f km/h  steer %+.2f%s" % (3.6 * ego.get_velocity().length(),
                                                   control.steer,
                                                   "  REVERSE" if control.reverse else "")]
            if rec:
                lines.append("REC %s  %d frames" % (rec.run_id, rec.frames))
            for i, text in enumerate(lines):
                color = (255, 60, 60) if text.startswith("REC") else (255, 255, 255)
                screen.blit(font.render(text, True, color), (10, 10 + 24 * i))
            pygame.display.flip()
            clock.tick(args.fps)           # keep the sim close to real time
    finally:
        if rec:
            rec.close("quit")
        hub.destroy()
        if ego:
            ego.destroy()
        # the Traffic Manager lives in this process and segfaults the server if
        # it goes down still holding cars (spawn_traffic.py left running)
        port = tm.get_port()
        client.apply_batch([carla.command.SetAutopilot(v.id, False, port)
                            for v in world.get_actors().filter("vehicle.*")])
        world.tick()
        tm.set_synchronous_mode(False)
        world.apply_settings(original)
        pygame.quit()


if __name__ == "__main__":
    main()
