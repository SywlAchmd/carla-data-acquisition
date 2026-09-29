#!/usr/bin/env python3
"""Live top-down map of the running CARLA world: roads, lane directions, vehicles.

    python3 minimap.py

Start it next to manual_drive.py or capture_overtaking.py. It only reads actor
transforms -- no camera, no spectator, no world settings -- so it costs the
server nothing and never touches the synchronous clock the other client owns.
A world reload (another --map) is picked up automatically.

Red = ego (role_name 'hero', or the ego blueprint), grey = everything else,
arrows = driving direction of each lane.

Keys
    F       follow the ego (zoomed) / whole map
    S       show / hide spawn point indices (for manual_drive.py --spawn)
    ESC, Q  quit
"""
import argparse
import sys
import time

import cv2
import numpy as np
import pygame

try:
    import carla
except ImportError:
    sys.exit("carla module not importable -- pip install carla==0.9.16")

from rig import DRIVING, EGO_BLUEPRINT

PX_PER_M = 4.0          # resolution of the pre-rendered road layer
MARGIN = 20.0           # m around the road network
ROAD = (70, 70, 70)
ARROW = (150, 150, 150)
BG = (40, 45, 40)
EGO = (40, 40, 230)
OTHER = (200, 200, 200)


class RoadLayer:
    """The static map, drawn once per world at PX_PER_M."""

    def __init__(self, cmap):
        self.name = cmap.name.split("/")[-1]
        segs = []
        for a, _ in cmap.get_topology():
            if a.lane_type != DRIVING:
                continue
            segs.append([a] + a.next_until_lane_end(2.0))
        pts = np.array([[wp.transform.location.x, wp.transform.location.y]
                        for s in segs for wp in s])
        self.origin = pts.min(axis=0) - MARGIN
        w, h = ((pts.max(axis=0) + MARGIN - self.origin) * PX_PER_M).astype(int)
        img = np.full((h, w, 3), BG, np.uint8)
        for s in segs:
            poly = np.array([self.px(wp.transform.location) for wp in s], np.int32)
            cv2.polylines(img, [poly], False, ROAD,
                          max(1, int(s[0].lane_width * PX_PER_M)), cv2.LINE_AA)
        for s in segs:
            for wp in s[2::8]:                         # one arrow every ~16 m
                f = wp.transform.get_forward_vector()
                p = np.array(self.px(wp.transform.location))
                d = np.array([f.x, f.y]) * 2.5 * PX_PER_M
                cv2.arrowedLine(img, tuple(map(int, p - d)), tuple(map(int, p + d)),
                                ARROW, 1, cv2.LINE_AA, tipLength=0.4)
        self.img = img
        self.cmap = cmap
        self.spawns = cmap.get_spawn_points()

    def px(self, loc):
        return ((loc.x - self.origin[0]) * PX_PER_M, (loc.y - self.origin[1]) * PX_PER_M)


def is_ego(actor):
    role = actor.attributes.get("role_name", "")
    return role == "hero" or (actor.type_id == EGO_BLUEPRINT and role != "autopilot")


def draw_vehicle(img, v, to_px, scale, color):
    tf, bb = v.get_transform(), v.bounding_box
    f, r = tf.get_forward_vector(), tf.get_right_vector()
    c = np.array(to_px(tf.location))
    fx, rx = np.array([f.x, f.y]), np.array([r.x, r.y])
    ex, ey = max(bb.extent.x, 1.5) * scale, max(bb.extent.y, 0.8) * scale
    box = [c + fx * ex + rx * ey, c + fx * ex - rx * ey,
           c - fx * ex - rx * ey, c - fx * ex + rx * ey]
    cv2.fillConvexPoly(img, np.array(box, np.int32), color, cv2.LINE_AA)
    cv2.line(img, tuple(map(int, c)), tuple(map(int, c + fx * ex * 1.6)), color, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--size", type=int, default=900, help="window size in px")
    ap.add_argument("--zoom-m", type=float, default=150.0,
                    help="metres across the window in follow mode")
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world, layer, world_id = None, None, None
    follow, show_spawns, trail = False, False, []
    pygame.init()
    screen = pygame.display.set_mode((args.size, args.size))
    pygame.display.set_caption("minimap -- F follow, S spawn points, ESC quit")

    while True:
        try:
            w = client.get_world()
            if w.id != world_id:                        # first run or map reloaded
                world, world_id, trail = w, w.id, []
                print("building road layer for", w.get_map().name)
                layer = RoadLayer(world.get_map())
            vehicles = world.get_actors().filter("vehicle.*")
            ego = next((v for v in vehicles if is_ego(v)), None)
        except RuntimeError as e:                      # server busy or reloading
            print("waiting for server:", e)
            pygame.event.pump()
            time.sleep(1.0)
            continue

        if ego is not None:
            p = layer.px(ego.get_location())
            if not trail or np.hypot(p[0] - trail[-1][0], p[1] - trail[-1][1]) > 4:
                trail = (trail + [p])[-600:]

        base = layer.img
        if follow and ego is not None:
            half = int(args.zoom_m * PX_PER_M / 2)
            cx, cy = map(int, layer.px(ego.get_location()))
            pad = cv2.copyMakeBorder(base, half, half, half, half,
                                     cv2.BORDER_CONSTANT, value=BG)
            view = pad[cy:cy + 2 * half, cx:cx + 2 * half]
            k, off = args.size / (2 * half), np.array([cx - half, cy - half])
        else:
            h, w_ = base.shape[:2]
            k, off = args.size / max(h, w_), np.zeros(2)
            view = base
        img = cv2.resize(view, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        to_px = lambda loc: (np.array(layer.px(loc)) - off) * k
        scale = PX_PER_M * k

        if show_spawns:
            for i, sp in enumerate(layer.spawns):
                q = tuple(map(int, to_px(sp.location)))
                cv2.circle(img, q, 3, (0, 200, 255), -1)
                cv2.putText(img, str(i), (q[0] + 4, q[1] - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, (0, 200, 255), 1, cv2.LINE_AA)
        if len(trail) > 1:
            pts = ((np.array(trail) - off) * k).astype(np.int32)
            cv2.polylines(img, [pts], False, (60, 60, 160), 2, cv2.LINE_AA)
        for v in vehicles:
            if ego is None or v.id != ego.id:
                draw_vehicle(img, v, to_px, scale, OTHER)
        if ego is not None:
            draw_vehicle(img, ego, to_px, scale, EGO)
            l = ego.get_location()
            wp = layer.cmap.get_waypoint(l)
            vel = ego.get_velocity()
            status = "ego x=%.0f y=%.0f  road %d lane %d%s  %.0f km/h" % (
                l.x, l.y, wp.road_id, wp.lane_id, "  JUNCTION" if wp.is_junction else "",
                3.6 * np.hypot(vel.x, vel.y))
        else:
            status = "no ego (role_name 'hero') in the world"
        for i, txt in enumerate((layer.name + "  %d vehicles" % len(vehicles), status,
                                 "F follow  S spawn points  ESC quit")):
            cv2.putText(img, txt, (10, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
        screen.fill(BG)
        screen.blit(pygame.surfarray.make_surface(img[:, :, ::-1].swapaxes(0, 1)), (0, 0))
        pygame.display.flip()

        quit_ = False
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN
                                          and ev.key in (pygame.K_ESCAPE, pygame.K_q)):
                quit_ = True
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_f:
                follow = not follow
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_s:
                show_spawns = not show_spawns
        if quit_:
            break
        time.sleep(0.1)                                 # ~10 Hz is plenty
    pygame.quit()


if __name__ == "__main__":
    main()
