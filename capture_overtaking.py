#!/usr/bin/env python3
"""CARLA 0.9.16 raw capture: overtaking scenarios on Town04 for a YOLOPX detection set.

Mirrors CarFree's extract.py role -- it only gathers raw data (RGB, semantic /
instance segmentation, optional depth, 3D boxes, camera pose). The bounding
boxes themselves are produced offline by carfree_gt.py.

  python3 capture_overtaking.py --total-frames 4000 --out out

Ego is the Dodge Charger 2020 with the camera rig from rig.py: forward RGB,
centred, 1.65 m above the road, 1.68 m ahead of the rear axle, level, 90 deg
horizontal FOV, 1280x720 (vertical FOV falls out at ~58.7 deg from 16:9).
"""
import argparse
import json
import os
import queue
import random
import sys
import time

import numpy as np

try:
    import carla
except ImportError:
    sys.exit("carla module not importable -- pip install carla==0.9.16")

# agents.navigation.controller ships with CARLA; reuse its PID instead of writing one
_CARLA_ROOT = os.environ.get("CARLA_ROOT", os.path.expanduser("~/Sawal/CARLA_0.9.16"))
for _p in (os.path.join(_CARLA_ROOT, "PythonAPI", "carla"),
           os.path.join(os.path.dirname(carla.__file__), "..", "..", "PythonAPI", "carla")):
    if os.path.isdir(os.path.join(_p, "agents")):
        sys.path.append(_p)
        break
try:
    from agents.navigation.controller import VehiclePIDController
except ImportError:
    sys.exit("cannot import agents.navigation.controller -- set CARLA_ROOT to your "
             "CARLA install (needs PythonAPI/carla/agents)")

import cv2

from carfree_gt import run_number
from rig import (DRIVING, EGO_BLUEPRINT, SensorHub, any_in_view, bgra, camera_block,
                 drivable_lanes, make_camera, pick_blueprints,
                 same_direction_neighbours, vehicles_near)


# ---------------------------------------------------------------- geometry utils

def yaw_delta(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def adjacent_lanes(wp):
    out = {}
    for side, nb in (("left", wp.get_left_lane()), ("right", wp.get_right_lane())):
        if nb and nb.lane_type == DRIVING and (nb.lane_id * wp.lane_id) > 0:
            out[side] = nb.lane_id
    return out


def find_straight_segments(cmap, min_len=200.0, step=5.0, max_dev=2.0, spacing=25.0):
    """Straight, long, multi-lane stretches (Town04's highway ring qualifies).

    Heading is compared against the segment start; anything drifting more than
    max_dev degrees over min_len metres is rejected. A side ("left"/"right") is
    only kept if a same-direction lane exists on that side for the WHOLE length
    -- Town04 grows and drops ramp lanes, and a passing lane that ends mid-
    overtake strands the car and cuts the run short.

    Returns [(start_waypoint, {"left"|"right"}), ...].
    """
    segs = []
    for wp in cmap.generate_waypoints(spacing):
        if wp.lane_type != DRIVING:
            continue
        sides = set(adjacent_lanes(wp))
        if not sides:
            continue
        yaw0 = wp.transform.rotation.yaw
        cur, dist, ok = wp, 0.0, True
        while dist < min_len:
            nxt = cur.next(step)
            if not nxt:
                ok = False
                break
            cur = nxt[0]
            sides &= set(adjacent_lanes(cur))
            if yaw_delta(cur.transform.rotation.yaw, yaw0) > max_dev or not sides:
                ok = False
                break
            dist += step
        if ok:
            segs.append((wp, sides))
    return segs


def waypoint_in_lane(cmap, location, lane_id):
    """Nearest waypoint, forced into `lane_id` by hopping sideways."""
    wp = cmap.get_waypoint(location, project_to_road=True, lane_type=DRIVING)
    for _ in range(4):
        if wp.lane_id == lane_id:
            return wp
        cands = [c for c in (wp.get_left_lane(), wp.get_right_lane())
                 if c and c.lane_type == DRIVING and (c.lane_id * lane_id) > 0]
        if not cands:
            return wp
        best = min(cands, key=lambda c: abs(c.lane_id - lane_id))
        if abs(best.lane_id - lane_id) >= abs(wp.lane_id - lane_id):
            return wp
        wp = best
    return wp


def gap_along(a, b):
    """Signed longitudinal distance from vehicle a to vehicle b, a's heading."""
    fwd = a.get_transform().get_forward_vector()
    d = b.get_location() - a.get_location()
    return d.x * fwd.x + d.y * fwd.y + d.z * fwd.z


def kmh(v):
    return 3.6 * v.get_velocity().length()


# ---------------------------------------------------------------- control

class Driver:
    """Lane-follower: holds a target lane id and target speed; a lane change is
    just a change of target_lane_id -- the PID steers across."""

    def __init__(self, vehicle, cmap, dt):
        self.v = vehicle
        self.map = cmap
        self.pid = VehiclePIDController(
            vehicle,
            args_lateral={"K_P": 1.15, "K_I": 0.05, "K_D": 0.2, "dt": dt},
            args_longitudinal={"K_P": 1.0, "K_I": 0.05, "K_D": 0.0, "dt": dt},
            max_throttle=0.9, max_brake=0.7)
        self.target_lane_id = cmap.get_waypoint(vehicle.get_location()).lane_id
        self.target_kmh = 30.0
        self.speed_cap = None      # set per tick from the gap to the car in front

    def step(self):
        loc = self.v.get_location()
        wp = waypoint_in_lane(self.map, loc, self.target_lane_id)
        la = float(np.clip(3.0 + 0.55 * (kmh(self.v) / 3.6), 4.0, 14.0))
        # a lane can end a few metres ahead: shorten the lookahead, then fall
        # back to whatever lane the car is physically in, before giving up
        nxt = wp.next(la) or wp.next(3.0) or self.map.get_waypoint(loc).next(la)
        if not nxt:
            return False                        # road really ran out
        goal = self.target_kmh if self.speed_cap is None \
            else min(self.target_kmh, self.speed_cap)
        self.v.apply_control(self.pid.run_step(goal, nxt[0]))
        return True

    def lateral_error(self):
        wp = waypoint_in_lane(self.map, self.v.get_location(), self.target_lane_id)
        return wp.transform.location.distance(self.v.get_location())


class OvertakeFSM:
    """approach -> pull out -> alongside -> merge back -> done."""

    def __init__(self, driver, target, start_lane, pass_lane, trigger_gap,
                 merge_gap, cruise_kmh, pass_kmh):
        self.d, self.t = driver, target
        self.start_lane, self.pass_lane = start_lane, pass_lane
        self.trigger_gap, self.merge_gap = trigger_gap, merge_gap
        self.cruise_kmh, self.pass_kmh = cruise_kmh, pass_kmh
        self.state = "approach"
        self.d.target_lane_id = start_lane
        self.d.target_kmh = pass_kmh

    def step(self, pass_lane_clear=True):
        g = gap_along(self.d.v, self.t)          # >0 -> target is ahead of us
        if self.state == "approach":
            self.d.target_kmh = self.pass_kmh
            if g < self.trigger_gap and pass_lane_clear:
                self.d.target_lane_id = self.pass_lane
                self.state = "alongside"
        elif self.state == "alongside":
            if g < -self.merge_gap:
                self.d.target_lane_id = self.start_lane
                self.state = "merge"
        elif self.state == "merge":
            if self.d.lateral_error() < 0.6:
                self.d.target_kmh = self.cruise_kmh
                self.state = "done"
        return self.state


# ---------------------------------------------------------------- scenario run

def lane_gaps(cmap, vehicle, lane_id, others, max_d=60.0):
    """(nearest vehicle ahead, nearest behind) that sits in `lane_id`.

    Lane membership beats a lateral-offset test here: mid-lane-change the car is
    pointing across the lanes, so a heading-aligned corridor misses exactly the
    vehicle it is about to cut in front of.
    """
    tf = vehicle.get_transform()
    fwd = tf.get_forward_vector()
    ahead = behind = max_d
    for o in others:
        wp = cmap.get_waypoint(o.get_location(), project_to_road=True, lane_type=DRIVING)
        if wp.lane_id != lane_id:
            continue
        d = o.get_location() - tf.location
        lon = d.x * fwd.x + d.y * fwd.y
        if 0.0 < lon < ahead:
            ahead = lon
        elif -behind < lon <= 0.0:
            behind = -lon
    return ahead, behind


def clear_ahead(vehicle, others, max_d=30.0, half_width=1.9):
    """Longitudinal distance to the nearest vehicle in this one's path."""
    tf = vehicle.get_transform()
    fwd, right = tf.get_forward_vector(), tf.get_right_vector()
    best = max_d
    for o in others:
        d = o.get_location() - tf.location
        lon = d.x * fwd.x + d.y * fwd.y
        lat = d.x * right.x + d.y * right.y
        if 0.0 < lon < best and abs(lat) < half_width:
            best = lon
    return best


def spawn_traffic(world, cmap, seg, cars, rng, n, exclude=()):
    """Background traffic strung along the segment, across all same-direction
    lanes, some ahead of the scripted pair and some behind it."""
    spawned = []
    d_fwd, d_back = 35.0, 15.0
    for _ in range(n * 6):
        if len(spawned) >= n:
            break
        if rng.random() < 0.7:
            d_fwd += rng.uniform(10.0, 28.0)
            base = seg.next(d_fwd)
        else:
            d_back += rng.uniform(10.0, 25.0)
            base = seg.previous(d_back)
        if not base:
            continue
        opts = [base[0]] + same_direction_neighbours(base[0])
        v = spawn_at(world, rng.choice(cars), rng.choice(opts), rng)
        if v is not None:
            spawned.append(v)
    return spawned


def spawn_at(world, bp, wp, rng, z=0.4):
    if bp.has_attribute("color"):
        bp.set_attribute("color", rng.choice(bp.get_attribute("color").recommended_values))
    tf = carla.Transform(wp.transform.location + carla.Location(z=z), wp.transform.rotation)
    return world.try_spawn_actor(bp, tf)


def run_scenario(world, cmap, segs, cars, rng, args, run_id, kind, weather_name, tm):
    """One overtaking episode. Returns the number of frames written."""
    out_dir = os.path.join(args.out, "raw", run_id)
    pakai_sem = args.semantic_camera or args.no_instance
    subdirs = ["rgb"] + (["sem"] if pakai_sem else []) \
        + ([] if args.no_instance else ["inst"]) + (["depth"] if args.depth else [])
    for sub in subdirs:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    world.set_weather(getattr(carla.WeatherParameters, weather_name))

    lead_kmh = rng.uniform(30.0, 42.0)
    pass_kmh = lead_kmh + rng.uniform(16.0, 30.0)
    start_gap = rng.uniform(20.0, 40.0)
    trigger_gap = rng.uniform(12.0, 20.0)
    merge_gap = rng.uniform(14.0, 22.0)

    actors, hub = [], SensorHub()
    try:
        # --- pick a straight stretch with the lane we need -----------------
        want = "left" if kind in ("overtake_left", "being_overtaken") else "right"
        pool = [s for s in segs if want in s[1]] or segs
        seg, sides = rng.choice(pool)
        side = want if want in sides else sorted(sides)[0]
        pass_lane = adjacent_lanes(seg)[side]
        start_lane = seg.lane_id

        # --- place the vehicles -------------------------------------------
        bp_ego = world.get_blueprint_library().find(EGO_BLUEPRINT)
        bp_npc = rng.choice(cars)
        ahead = seg.next(start_gap)
        if not ahead:
            return 0, "no room ahead of the spawn point"
        if kind == "being_overtaken":
            ego_wp, npc_wp = ahead[0], seg
        else:
            ego_wp, npc_wp = seg, ahead[0]

        ego = spawn_at(world, bp_ego, ego_wp, rng)
        npc = spawn_at(world, bp_npc, npc_wp, rng)
        if ego is None or npc is None:
            for a in (ego, npc):
                if a:
                    a.destroy()
            return 0, "spawn blocked"
        actors += [ego, npc]

        traffic = spawn_traffic(world, cmap, seg, cars, rng, args.traffic,
                                exclude={ego.id, npc.id})
        actors += traffic

        # --- sensors -------------------------------------------------------
        bp_lib = world.get_blueprint_library()
        hub.add("rgb", make_camera(world, bp_lib, "rgb", ego))
        if not args.no_instance:
            hub.add("inst", make_camera(world, bp_lib, "instance_segmentation", ego))
        # kamera semantic hanya kalau diminta: kanal R kamera instance sudah
        # membawa tag yang sama persis, jadi satu render target bisa dihemat
        if pakai_sem:
            hub.add("sem", make_camera(world, bp_lib, "semantic_segmentation", ego))
        if args.depth:
            hub.add("depth", make_camera(world, bp_lib, "depth", ego))

        col_bp = bp_lib.find("sensor.other.collision")
        collisions = queue.Queue()
        col = world.spawn_actor(col_bp, carla.Transform(), attach_to=ego)
        col.listen(collisions.put)
        hub.sensors.append(col)

        # --- controllers ---------------------------------------------------
        dt = 1.0 / args.fps
        ego_d, npc_d = Driver(ego, cmap, dt), Driver(npc, cmap, dt)
        for v in traffic:
            v.set_autopilot(True, tm.get_port())
            # negative = faster than the speed limit; spread the pack out a bit
            tm.vehicle_percentage_speed_difference(v, rng.uniform(10.0, 45.0))
            tm.auto_lane_change(v, rng.random() < 0.3)
            tm.distance_to_leading_vehicle(v, rng.uniform(4.0, 10.0))

        if kind == "being_overtaken":
            ego_d.target_lane_id = start_lane
            ego_d.target_kmh = lead_kmh
            fsm = OvertakeFSM(npc_d, ego, start_lane, pass_lane, trigger_gap,
                              merge_gap, lead_kmh + 6.0, pass_kmh)
        else:
            npc_d.target_lane_id = start_lane
            npc_d.target_kmh = lead_kmh
            fsm = OvertakeFSM(ego_d, npc, start_lane, pass_lane, trigger_gap,
                              merge_gap, lead_kmh, pass_kmh)

        # let physics settle before recording
        for _ in range(int(args.fps * 1.0)):
            world.tick()

        run_meta = {
            "run_id": run_id, "map": "Town04", "scenario": kind, "weather": weather_name,
            "params": {"lead_kmh": round(lead_kmh, 2), "pass_kmh": round(pass_kmh, 2),
                       "start_gap_m": round(start_gap, 2), "trigger_gap_m": round(trigger_gap, 2),
                       "merge_gap_m": round(merge_gap, 2), "traffic": args.traffic,
                       "start_lane_id": start_lane, "pass_lane_id": pass_lane,
                       "fps": args.fps, "record_every": args.record_every},
            "cameras": subdirs,
            "ego_blueprint": EGO_BLUEPRINT,
            "camera": camera_block(),
        }
        json.dump(run_meta, open(os.path.join(out_dir, "run.json"), "w"), indent=2)

        # --- main loop ------------------------------------------------------
        meta_fh = open(os.path.join(out_dir, "meta.jsonl"), "w")
        written, tick_i, tail = 0, 0, 0
        reason = "frame budget"
        spectator = world.get_spectator()
        while written < args.max_frames_per_run and tick_i < args.max_frames_per_run * 8:
            snap_frame = world.tick()
            tick_i += 1

            if not collisions.empty():
                reason = "collision"
                break
            others = list(world.get_actors().filter("vehicle.*"))
            rest = lambda me: [o for o in others if o.id != me.id]
            p_ahead, p_behind = lane_gaps(cmap, fsm.d.v, pass_lane, rest(fsm.d.v))
            state = fsm.step(p_ahead > 32.0 and p_behind > 14.0)
            for drv in (ego_d, npc_d):
                peers = rest(drv.v)
                free = min(clear_ahead(drv.v, peers),
                           lane_gaps(cmap, drv.v, drv.target_lane_id, peers)[0])
                drv.speed_cap = None if free > 22.0 else max(5.0, 2.6 * free)
            if not ego_d.step():
                reason = "ego ran out of road"
                break
            if not npc_d.step():
                reason = "npc ran out of road"
                break
            if state == "done":
                tail += 1
                if tail > args.fps * args.tail_seconds:
                    reason = "overtake complete"
                    break

            data = hub.grab(snap_frame)
            if tick_i % args.record_every:
                continue

            cam_m = np.array(data["rgb"].transform.get_matrix(), dtype=np.float64)
            visible = vehicles_near(world, ego, data["rgb"].transform.location,
                                    args.max_distance)
            if args.skip_empty and not any_in_view(visible, cam_m):
                continue

            stem = "%06d" % written
            cv2.imwrite(os.path.join(out_dir, "rgb", stem + ".jpg"),
                        bgra(data["rgb"])[:, :, :3],
                        [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            if "sem" in data:
                cv2.imwrite(os.path.join(out_dir, "sem", stem + ".png"),
                            bgra(data["sem"])[:, :, 2])
            if "inst" in data:
                cv2.imwrite(os.path.join(out_dir, "inst", stem + ".png"),
                            bgra(data["inst"])[:, :, :3])
            if "depth" in data:
                cv2.imwrite(os.path.join(out_dir, "depth", stem + ".png"),
                            bgra(data["depth"])[:, :, :3])

            meta_fh.write(json.dumps({
                "frame": written, "sim_frame": snap_frame, "state": state,
                "camera": {"world_matrix": cam_m.tolist()},
                "ego": {"id": ego.id, "speed_kmh": round(kmh(ego), 2)},
                "lanes": drivable_lanes(cmap, ego.get_location(),
                                        args.da_ahead, args.da_behind, args.da_step),
                "actors": visible}) + "\n")
            written += 1

            if args.follow:
                tf = ego.get_transform()
                spectator.set_transform(carla.Transform(
                    tf.location + carla.Location(z=25) - 12 * tf.get_forward_vector(),
                    carla.Rotation(pitch=-30, yaw=tf.rotation.yaw)))

        meta_fh.close()
        run_meta["frames"] = written
        run_meta["ended_because"] = reason
        json.dump(run_meta, open(os.path.join(out_dir, "run.json"), "w"), indent=2)
        return written, reason
    finally:
        hub.destroy()
        # hand the traffic back before destroying it: the Traffic Manager keeps
        # its own registry and crashes on shutdown if it still holds dead actors
        for a in actors:
            try:
                a.set_autopilot(False, tm.get_port())
            except RuntimeError:
                pass
        for a in actors:
            try:
                a.destroy()
            except RuntimeError:
                pass
        try:
            world.tick()          # let the server process the destruction
        except RuntimeError:
            pass


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--out", default="out")
    ap.add_argument("--total-frames", type=int, default=4000)
    ap.add_argument("--max-frames-per-run", type=int, default=320)
    ap.add_argument("--fps", type=int, default=20, help="simulation ticks per second")
    ap.add_argument("--record-every", type=int, default=2, help="save 1 of every N ticks")
    ap.add_argument("--tail-seconds", type=float, default=3.0)
    ap.add_argument("--max-distance", type=float, default=120.0)
    ap.add_argument("--traffic", type=int, default=12,
                    help="background vehicles per run, driven by the Traffic Manager")
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--da-ahead", type=float, default=200.0,
                    help="how far ahead the drivable-area geometry reaches (m); it is "
                         "clipped to the visible road later, so overshooting is free")
    ap.add_argument("--da-behind", type=float, default=8.0)
    ap.add_argument("--da-step", type=float, default=6.0,
                    help="station spacing; on the straight segments we pick the chord "
                         "error of a 6 m step is far below one pixel")
    ap.add_argument("--weathers", default="ClearNoon",
                    help="comma list, e.g. ClearNoon,ClearSunset,CloudyNoon")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--depth", action="store_true", help="also record the depth camera")
    ap.add_argument("--no-instance", action="store_true",
                    help="rekam kamera semantic saja, tanpa instance (persis paper "
                         "CarFree; mask dua mobil yang bertumpuk akan menyatu)")
    ap.add_argument("--semantic-camera", action="store_true",
                    help="rekam juga kamera semantic terpisah. Tidak perlu di 0.9.16: "
                         "kanal R kamera instance identik dengan peta semantic, dan "
                         "load_tags() membacanya dari sana")
    ap.add_argument("--skip-empty", dest="skip_empty", action="store_true", default=True)
    ap.add_argument("--keep-empty", dest="skip_empty", action="store_false")
    ap.add_argument("--follow", action="store_true", help="move the spectator behind the ego")
    ap.add_argument("--min-straight", type=float, default=200.0)
    ap.add_argument("--resume", action="store_true",
                    help="keep the runs already in --out and number new ones after them")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    weathers = [w.strip() for w in args.weathers.split(",") if w.strip()]

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    if world.get_map().name.split("/")[-1] != "Town04":
        print("loading Town04 ...")
        world = client.load_world("Town04")
    cmap = world.get_map()

    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    world.apply_settings(settings)

    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(5.0)

    try:
        min_len = args.min_straight
        segs = []
        while min_len >= 80.0 and not segs:
            segs = find_straight_segments(cmap, min_len=min_len)
            if not segs:
                print("no straight segment >= %.0f m, relaxing" % min_len)
                min_len -= 50.0
        if not segs:
            sys.exit("Town04: no usable straight multi-lane segment found")
        n_left = sum("left" in s[1] for s in segs)
        n_right = sum("right" in s[1] for s in segs)
        print("found %d straight segments (>= %.0f m, heading dev < 2 deg; "
              "%d with a continuous left lane, %d with a right one)"
              % (len(segs), min_len, n_left, n_right))

        kinds = ["overtake_left", "overtake_left", "overtake_right", "being_overtaken"]
        run_i, done_before = 0, 0
        raw_root = os.path.join(args.out, "raw")
        if args.resume and os.path.isdir(raw_root):
            existing = sorted(d for d in os.listdir(raw_root) if run_number(d) is not None)
            run_i = max((run_number(d) for d in existing), default=-1) + 1
            done_before = sum(len(os.listdir(os.path.join(raw_root, d, "rgb")))
                              for d in existing if os.path.isdir(os.path.join(raw_root, d, "rgb")))
            rng = random.Random(args.seed + run_i)
            print("resuming at run %04d with %d frames already captured" % (run_i, done_before))
        total, t0 = done_before, time.time()
        while total < args.total_frames:
            kind = kinds[run_i % len(kinds)]
            weather = weathers[run_i % len(weathers)]
            run_id = "town04_run_%04d" % run_i
            try:
                n, reason = run_scenario(world, cmap, segs, pick_blueprints(world), rng,
                                         args, run_id, kind, weather, tm)
            except (RuntimeError, queue.Empty) as exc:
                print("simulator lost during %s: %s" % (run_id, exc))
                raise SystemExit(3)
            total += n
            run_i += 1
            print("[%3d] %-16s %-12s %4d frames  %-26s (total %5d/%d, %.0fs)"
                  % (run_i, kind, weather, n, "<" + reason + ">", total,
                     args.total_frames, time.time() - t0))
            if n == 0 and run_i > 8 and total == 0:
                sys.exit("every run produced 0 frames -- check the CARLA server")
        print("done: %d frames in %s/raw" % (total, args.out))
    finally:
        tm.set_synchronous_mode(False)
        world.apply_settings(original)


if __name__ == "__main__":
    main()
