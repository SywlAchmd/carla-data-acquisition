"""Ego vehicle, sensor rig and the capture helpers shared by
manual_drive.py and capture_overtaking.py.

Vehicle: Dodge Charger 2020 (5.008 x 1.882 m, 1920 kg in CARLA).
Camera:  RGB + depth + instance segmentation at the same transform,
         1280x720, 90 deg horizontal FOV, 1.65 m above the road (KITTI cam2
         height), 1.68 m ahead of the rear axle, on the centre line, level.
"""
import os
import queue

import numpy as np
import carla

from carfree_gt import build_K, project, run_number, world_to_cam

EGO_BLUEPRINT = "vehicle.dodge.charger_2020"

IMG_W, IMG_H, FOV = 1280, 720, 90.0

# CARLA puts the vehicle origin at the body centre, on the ground. The camera is
# specified from the rear axle, so shift it back into the origin's frame.
# manual_drive.py prints the measured values at start-up; tune these if they drift.
REAR_AXLE_TO_CENTRE = 1.433     # m
CAM_AHEAD_OF_REAR_AXLE = 1.68   # m
CAM_X = CAM_AHEAD_OF_REAR_AXLE - REAR_AXLE_TO_CENTRE    # 0.247 m ahead of the origin
CAM_Z = 1.65                    # m above the road

STEER_LIMIT_RAD = 0.5           # 28.6 deg, same limit the MPC uses (wheel max is 70 deg)

DRIVING = carla.LaneType.Driving


def next_run_id(raw_root, town):
    """'<town>_run_NNNN'. Numbers continue across towns, so each one is unique."""
    os.makedirs(raw_root, exist_ok=True)
    nums = [n for n in map(run_number, os.listdir(raw_root)) if n is not None]
    return "%s_run_%04d" % (town.lower(), max(nums, default=-1) + 1)


# ---------------------------------------------------------------- sensors

class SensorHub:
    """One queue per sensor; grab() returns the samples of one sim frame."""

    def __init__(self):
        self.queues = {}
        self.sensors = []

    def add(self, name, sensor):
        q = queue.Queue()
        sensor.listen(q.put)
        self.queues[name] = q
        self.sensors.append(sensor)

    def grab(self, frame, timeout=8.0):
        out = {}
        for name, q in self.queues.items():
            while True:
                data = q.get(timeout=timeout)
                if data.frame == frame:
                    out[name] = data
                    break
        return out

    def destroy(self):
        for s in self.sensors:
            s.stop()
            s.destroy()
        self.sensors, self.queues = [], {}


def make_camera(world, bp_lib, kind, ego, extra=None):
    bp = bp_lib.find("sensor.camera." + kind)
    bp.set_attribute("image_size_x", str(IMG_W))
    bp.set_attribute("image_size_y", str(IMG_H))
    bp.set_attribute("fov", str(FOV))
    for k, v in (extra or {}).items():
        bp.set_attribute(k, v)
    tf = carla.Transform(carla.Location(x=CAM_X, y=0.0, z=CAM_Z),
                         carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))
    return world.spawn_actor(bp, tf, attach_to=ego, attachment_type=carla.AttachmentType.Rigid)


def camera_block():
    """The 'camera' entry of run.json, read back by carfree_gt/seg_gt/build."""
    return {"width": IMG_W, "height": IMG_H, "fov": FOV,
            "mount_xyz": [CAM_X, 0.0, CAM_Z], "mount_rpy": [0.0, 0.0, 0.0]}


def bgra(image):
    return np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)


# ---------------------------------------------------------------- vehicles

def pick_blueprints(world):
    """Four-wheel passenger cars only -- no bikes, no trucks/vans (they carry a
    different semantic tag, which would break the single-class 'car' label)."""
    out = []
    for bp in world.get_blueprint_library().filter("vehicle.*"):
        if not bp.has_attribute("number_of_wheels"):
            continue
        if bp.get_attribute("number_of_wheels").as_int() != 4:
            continue
        if bp.has_attribute("base_type") and bp.get_attribute("base_type") != "car":
            continue
        out.append(bp)
    return out


def world_verts(actor):
    """8 corners of the actor's 3D box in world coordinates.

    Corner index = 4*i(x) + 2*i(y) + i(z) with i in {0:-extent, 1:+extent}, so
    corners differing in one bit share a cuboid edge (see EDGES in carfree_gt)."""
    bb = actor.bounding_box
    m = np.array(actor.get_transform().get_matrix(), dtype=np.float64)
    e, c = bb.extent, bb.location
    local = np.array([[c.x + sx * e.x, c.y + sy * e.y, c.z + sz * e.z, 1.0]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return (m @ local.T).T[:, :3].tolist()


def vehicles_near(world, ego, cam_location, max_distance):
    """Every other vehicle within max_distance of the camera, with its 3D box
    corners: the per-frame 'actors' list carfree_gt.py consumes."""
    out = []
    for a in world.get_actors().filter("vehicle.*"):
        if a.id == ego.id:
            continue
        d = a.get_location().distance(cam_location)
        if d > max_distance:
            continue
        out.append({"id": a.id, "type_id": a.type_id, "category": "car",
                    "distance": d, "verts_world": world_verts(a)})
    return out


def any_in_view(visible, cam_m):
    """Coarse pre-filter: does any actor's centre land inside the image?
    The real visibility decision is Algorithm 2, offline."""
    if not visible:
        return False
    K = build_K(IMG_W, IMG_H, FOV)
    for a in visible:
        c = np.mean(np.asarray(a["verts_world"]), axis=0, keepdims=True)
        cam = world_to_cam(c, cam_m)
        if cam[0, 2] < 0.5:
            continue
        uv = project(cam, K)[0]
        if -IMG_W * 0.15 < uv[0] < IMG_W * 1.15 and -IMG_H * 0.15 < uv[1] < IMG_H * 1.15:
            return True
    return False


# ---------------------------------------------------------------- road geometry

def same_direction_neighbours(wp, max_hops=3):
    """All same-direction Driving lanes beside `wp`, nearest first."""
    out = []
    for getter in ("get_left_lane", "get_right_lane"):
        cur = wp
        for _ in range(max_hops):
            nb = getattr(cur, getter)()
            if not (nb and nb.lane_type == DRIVING and nb.lane_id * wp.lane_id > 0):
                break
            out.append(nb)
            cur = nb
    return out


def drivable_lanes(cmap, location, ahead=200.0, behind=8.0, step=6.0):
    """Left/right edge points of the ego's own carriageway, in world coordinates.

    Only same-direction lanes -- the opposite carriageway is road surface but not
    drivable, which is the distinction BDD100K's da_seg makes and a plain
    semantic 'road' mask cannot. seg_gt.py turns consecutive stations into quads.
    At a junction only the first branch returned by next() is followed.
    """
    ego_wp = cmap.get_waypoint(location, project_to_road=True, lane_type=DRIVING)
    lanes = [(ego_wp, "ego")] + [(w, "adjacent") for w in same_direction_neighbours(ego_wp)]
    out = []
    for wp0, kind in lanes:
        back = wp0.previous(behind)
        cur = back[0] if back else wp0
        edges = []
        for _ in range(int((ahead + behind) / step) + 1):
            tf = cur.transform
            r, hw = tf.get_right_vector(), cur.lane_width * 0.5
            off = carla.Location(x=r.x * hw, y=r.y * hw, z=r.z * hw)
            le, ri = tf.location - off, tf.location + off
            edges.append([[round(le.x, 2), round(le.y, 2), round(le.z, 2)],
                          [round(ri.x, 2), round(ri.y, 2), round(ri.z, 2)]])
            nxt = cur.next(step)
            if not nxt:
                break
            cur = nxt[0]
        if len(edges) >= 2:
            out.append({"kind": kind, "edges": edges})
    return out
