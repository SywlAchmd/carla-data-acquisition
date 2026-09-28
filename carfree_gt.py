#!/usr/bin/env python3
"""CarFree ground-truth generation (Jang, Lee & Kim, Appl. Sci. 2022, 12, 281).

Offline post-processing: raw CARLA capture -> precise 2D bounding boxes.
Implements the paper's four steps verbatim:

  5.1 Coordinate transformation   world -> camera -> image
  5.2 Algorithm 1                 8 projected points -> initial 2D box
  5.3 Algorithm 2                 center-pixel visibility test on the seg mask
  5.4 Algorithm 3                 shrink each side until it touches the silhouette

Resolution-agnostic: everything works in pixel space, so 1280x720 is fine even
though the paper used 960x540.

Run standalone:  python3 carfree_gt.py --raw out/raw
Or import:       from carfree_gt import frame_labels
"""
import argparse
import collections
import json
import os
import sys

import cv2
import numpy as np

# CARLA semantic tags (0.9.14+ palette, same in 0.9.16)
TAG_ROAD = 1
TAG_ROADLINE = 24
TAG_PEDESTRIAN = 12
TAG_CAR = 14
TAG_TRUCK = 15
TAG_BUS = 16
VEHICLE_TAGS = (TAG_CAR, TAG_TRUCK, TAG_BUS)

CATEGORY_TO_TAGS = {"car": (TAG_CAR,), "truck": (TAG_TRUCK,), "bus": (TAG_BUS,),
                    "person": (TAG_PEDESTRIAN,)}

# the 12 edges of a cuboid: corner indices that differ in exactly one bit
EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]

NEAR_PLANE = 0.20  # m; corners closer than this are clipped, not projected


# ---------------------------------------------------------------- 5.1 transform

def build_K(width, height, fov_deg):
    """Pinhole intrinsics. CARLA sets the HORIZONTAL fov; fy == fx (square px),
    so the vertical fov follows from the aspect ratio (58.7 deg @ 16:9, 90 deg h)."""
    f = width / (2.0 * np.tan(np.radians(fov_deg) * 0.5))
    return np.array([[f, 0.0, width * 0.5],
                     [0.0, f, height * 0.5],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def world_to_cam(points_world, cam_world_matrix):
    """UE world coords -> OpenCV-style camera coords (x right, y down, z forward)."""
    w2c = np.linalg.inv(np.asarray(cam_world_matrix, dtype=np.float64))
    pts = np.asarray(points_world, dtype=np.float64)
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    ue = (w2c @ hom.T).T[:, :3]          # UE camera frame: x fwd, y right, z up
    return np.stack([ue[:, 1], -ue[:, 2], ue[:, 0]], axis=1)


def clip_near(cam_pts, near=NEAR_PLANE):
    """Clip the cuboid's 12 edges against the near plane.

    Without this, a vehicle straddling the camera plane (the 'alongside' phase of
    an overtake) projects to garbage instead of a truncated box.
    """
    keep = []
    for a, b in EDGES:
        pa, pb = cam_pts[a], cam_pts[b]
        za, zb = pa[2], pb[2]
        if za >= near and zb >= near:
            keep.append(pa)
            keep.append(pb)
        elif za >= near or zb >= near:
            t = (near - za) / (zb - za)
            keep.append(pa if za >= near else pb)
            keep.append(pa + t * (pb - pa))
    return np.asarray(keep) if keep else None


def clip_polygon_near(cam_pts, near=NEAR_PLANE):
    """Sutherland-Hodgman clip of a convex polygon against the near plane.
    Used for the drivable-area quads, which routinely straddle the camera."""
    out = []
    n = len(cam_pts)
    for i in range(n):
        a, b = cam_pts[i], cam_pts[(i + 1) % n]
        a_in, b_in = a[2] >= near, b[2] >= near
        if a_in:
            out.append(a)
        if a_in != b_in:
            t = (near - a[2]) / (b[2] - a[2])
            out.append(a + t * (b - a))
    return np.asarray(out) if len(out) >= 3 else None


def project(cam_pts, K):
    uv = (K @ cam_pts.T).T
    return uv[:, :2] / uv[:, 2:3]


# ---------------------------------------------------------------- 5.2 Algorithm 1

def initial_bbox(pts2d):
    """Algorithm 1: minimum-area box enclosing the 8 projected corner points."""
    return (float(pts2d[:, 0].min()), float(pts2d[:, 1].min()),
            float(pts2d[:, 0].max()), float(pts2d[:, 1].max()))


# ---------------------------------------------------------------- 5.3 Algorithm 2

def is_visible(box, mask, mode="center"):
    """Algorithm 2: the center pixel of the initial box must belong to the target
    class. `multi5` is the extension the paper suggests for partially occluded
    objects (center + the four corners, inset one pixel)."""
    h, w = mask.shape
    x1, y1, x2, y2 = box
    xc, yc = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    probes = [(xc, yc)]
    if mode == "multi5":
        ix1, iy1 = x1 + 1, y1 + 1
        ix2, iy2 = x2 - 1, y2 - 1
        probes += [(ix1, iy1), (ix2, iy1), (ix1, iy2), (ix2, iy2)]
    for px, py in probes:
        xi = int(np.clip(round(px), 0, w - 1))
        yi = int(np.clip(round(py), 0, h - 1))
        if mask[yi, xi]:
            return True
    return False


# ---------------------------------------------------------------- 5.4 Algorithm 3

def fit_to_boundary(box, mask):
    """Algorithm 3: move each side inward until it touches a target-class pixel.

    The paper's loop tests a whole column S[x_min] / row S[y_min]; we test only
    the part of that column inside the current box. Same result for an isolated
    object, and it stops a *second* car in the same column from halting the
    shrink early. The two-pass any()/flatnonzero below is the vectorised
    equivalent of the while-loop -- both stop at the first line containing a
    target pixel.
    """
    h, w = mask.shape
    x1 = int(np.clip(np.floor(box[0]), 0, w - 1))
    y1 = int(np.clip(np.floor(box[1]), 0, h - 1))
    x2 = int(np.clip(np.ceil(box[2]), 0, w - 1))
    y2 = int(np.clip(np.ceil(box[3]), 0, h - 1))
    if x2 < x1 or y2 < y1:
        return None
    sub = mask[y1:y2 + 1, x1:x2 + 1]
    if not sub.any():
        return None
    cols = np.flatnonzero(sub.any(axis=0))
    rows = np.flatnonzero(sub.any(axis=1))
    # x2/y2 are exclusive so that width == x2 - x1 (BDD100K convention)
    return (float(x1 + cols[0]), float(y1 + rows[0]),
            float(x1 + cols[-1] + 1), float(y1 + rows[-1] + 1))


# ---------------------------------------------------------------- masks

def load_semantic(path):
    """Raw tag image saved as single-channel PNG by the capture script."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(path)
    return img if img.ndim == 2 else img[:, :, 2]


def load_instance(path):
    """PNG written straight from CARLA's raw_data byte order: byte0 and byte1
    carry the actor id (byte0 = high), byte2 the semantic tag.

    Verified against a live capture -- `(byte0 << 8) | byte1` reproduces
    `actor.id & 0xFFFF` exactly; the opposite order yields garbage."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(path)
    tag = img[:, :, 2].astype(np.uint8)
    iid = (img[:, :, 0].astype(np.uint16) << 8) | img[:, :, 1].astype(np.uint16)
    return tag, iid


def load_tags(run_dir, stem):
    """Peta tag semantic untuk satu frame.

    Dari kamera semantic kalau memang direkam; kalau tidak, dari kanal R kamera
    instance. Di CARLA 0.9.16 keduanya identik (diuji 100% pada 25 frame), jadi
    kamera semantic terpisah tidak perlu di-spawn lagi.
    """
    p = os.path.join(run_dir, "sem", stem + ".png")
    if os.path.exists(p):
        return load_semantic(p)
    return load_instance(os.path.join(run_dir, "inst", stem + ".png"))[0]


def decode_depth(path):
    """CARLA depth PNG -> metres."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float64)
    norm = (img[:, :, 2] + img[:, :, 1] * 256.0 + img[:, :, 0] * 65536.0) / (256 ** 3 - 1)
    return 1000.0 * norm


# ---------------------------------------------------------------- attributes

def occlusion_attrs(fitted, actor_mask, other_vehicle_mask):
    """`occluded` heuristic: another vehicle's pixels sit inside this object's
    tight box, or the box is suspiciously empty. Cheap, and adequate for the
    sparse 1-2 vehicle traffic this dataset targets.
    ponytail: swap for a depth-buffer test if you ever need dense traffic."""
    x1, y1, x2, y2 = (int(v) for v in fitted)
    a = int(actor_mask[y1:y2, x1:x2].sum())
    box_area = max(1, (x2 - x1) * (y2 - y1))
    fill = a / box_area
    if other_vehicle_mask is None:
        return bool(fill < 0.45), float(fill), a
    o = int(other_vehicle_mask[y1:y2, x1:x2].sum())
    return bool(o > 0.05 * max(a, 1) or fill < 0.45), float(fill), a


# ---------------------------------------------------------------- per-frame driver

def frame_labels(meta, K, sem_tag, inst=None, mask_source="semantic",
                 visibility="center", min_box_px=8, max_distance=120.0, stats=None,
                 min_box_h=0.0):
    """Full CarFree pipeline for one frame. Returns a list of BDD-style labels."""
    h, w = sem_tag.shape
    inst_tag, inst_id = (None, None) if inst is None else inst
    veh_sem = np.isin(sem_tag, VEHICLE_TAGS)
    stats = collections.Counter() if stats is None else stats
    labels = []

    for actor in meta["actors"]:
        if actor["distance"] > max_distance:
            continue
        cat = actor["category"]
        tags = CATEGORY_TO_TAGS.get(cat, (TAG_CAR,))

        # 5.1 -------------------------------------------------------------
        cam_pts = world_to_cam(actor["verts_world"], meta["camera"]["world_matrix"])
        clipped = clip_near(cam_pts)
        if clipped is None:
            continue
        pts2d = project(clipped, K)

        # 5.2 -------------------------------------------------------------
        raw = initial_bbox(pts2d)
        if raw[2] < 0 or raw[0] > w or raw[3] < 0 or raw[1] > h:
            continue
        truncated = bool(raw[0] < -1 or raw[1] < -1 or raw[2] > w + 1 or raw[3] > h + 1
                         or len(clipped) < 24)

        # target mask: paper uses semantic; instance is the optional upgrade
        actor_mask = None
        if inst_id is not None:
            actor_mask = (inst_id == (actor["id"] & 0xFFFF)) & np.isin(inst_tag, tags)
            if not actor_mask.any():
                actor_mask = None          # no instance pixels: too far, or fully hidden
                stats["instance_miss"] += 1
        if mask_source == "instance":
            # zero instance pixels means the object is genuinely not visible.
            # Falling back to the semantic blob here would fit this box to
            # whichever car is occluding it -- exactly the failure the instance
            # mask exists to avoid.
            if actor_mask is None:
                continue
            target = actor_mask
        else:
            target = np.isin(sem_tag, tags)

        # 5.3 -------------------------------------------------------------
        if not is_visible(raw, target, visibility):
            continue

        # 5.4 -------------------------------------------------------------
        fitted = fit_to_boundary(raw, target)
        if fitted is None:
            continue
        if fitted[2] - fitted[0] < min_box_px or fitted[3] - fitted[1] < min_box_px:
            continue
        # KITTI ignores objects under a minimum box height; below roughly 16 px
        # here the car is also under 8 px once YOLOPX letterboxes 1280x720 to
        # 640, i.e. under one output stride, so it cannot be learnt anyway.
        if fitted[3] - fitted[1] < min_box_h:
            stats["too_small"] += 1
            continue

        stats["boxes"] += 1
        if actor_mask is not None:
            others = veh_sem & ~actor_mask
            occluded, fill, npx = occlusion_attrs(fitted, actor_mask, others)
        else:
            occluded, fill, npx = occlusion_attrs(fitted, target, None)

        labels.append({
            "id": actor["id"],
            "category": cat,
            "box2d": {"x1": round(fitted[0], 2), "y1": round(fitted[1], 2),
                      "x2": round(fitted[2], 2), "y2": round(fitted[3], 2)},
            "attributes": {"occluded": occluded, "truncated": truncated},
            "carla": {"distance": round(actor["distance"], 2),
                      "type_id": actor["type_id"],
                      "visible_ratio": round(fill, 3),
                      "mask_pixels": npx},
        })
    return labels


# ---------------------------------------------------------------- CLI

def process_run(run_dir, mask_source, visibility, min_box_px, max_distance,
                min_box_h=0.0):
    run = json.load(open(os.path.join(run_dir, "run.json")))
    cam = run["camera"]
    K = build_K(cam["width"], cam["height"], cam["fov"])
    out = os.path.join(run_dir, "det")
    os.makedirs(out, exist_ok=True)

    n_frames = n_boxes = 0
    stats = collections.Counter()
    with open(os.path.join(run_dir, "meta.jsonl")) as fh:
        for line in fh:
            meta = json.loads(line)
            stem = "%06d" % meta["frame"]
            sem = load_tags(run_dir, stem)
            inst_path = os.path.join(run_dir, "inst", stem + ".png")
            inst = load_instance(inst_path) if os.path.exists(inst_path) else None
            labels = frame_labels(meta, K, sem, inst, mask_source, visibility,
                                  min_box_px, max_distance, stats, min_box_h)
            json.dump({"name": stem + ".jpg", "labels": labels},
                      open(os.path.join(out, stem + ".json"), "w"))
            n_frames += 1
            n_boxes += len(labels)
    return n_frames, n_boxes, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="out/raw", help="directory holding run_* folders")
    ap.add_argument("--mask-source", choices=["semantic", "instance"], default="semantic",
                    help="semantic = the paper as published; instance = per-actor masks")
    ap.add_argument("--visibility", choices=["center", "multi5"], default="center",
                    help="Algorithm 2 probe: single center pixel or 5-point sampling")
    ap.add_argument("--min-box-px", type=int, default=8)
    ap.add_argument("--min-box-h", type=float, default=25.0,
                    help="drop boxes shorter than this (px). 25 is KITTI's minimum "
                         "for its Moderate/Hard levels and what yolopx_dataset_v2 "
                         "used; a car hits it at ~40 m. 16 (~60 m) is where it falls "
                         "below one output stride after YOLOPX letterboxes to 640")
    ap.add_argument("--max-distance", type=float, default=120.0)
    args = ap.parse_args()

    runs = sorted(d for d in os.listdir(args.raw)
                  if os.path.isdir(os.path.join(args.raw, d)))
    if not runs:
        sys.exit("no runs found in %s" % args.raw)
    tf = tb = total_miss = total_small = 0
    for r in runs:
        f, b, st = process_run(os.path.join(args.raw, r), args.mask_source,
                               args.visibility, args.min_box_px, args.max_distance,
                               args.min_box_h)
        miss = st["instance_miss"]
        note = "" if not miss else "  [%d/%d without instance pixels]" % (miss, st["boxes"] + miss)
        print("%-14s %5d frames  %5d boxes  (%.2f boxes/frame)%s"
              % (r, f, b, b / max(f, 1), note))
        tf += f
        tb += b
        total_miss += miss
        total_small += st["too_small"]
    print("-" * 52)
    print("TOTAL          %5d frames  %5d boxes  (%.2f boxes/frame)"
          % (tf, tb, tb / max(tf, 1)))
    if total_small:
        print("dropped %d boxes shorter than %.0f px" % (total_small, args.min_box_h))
    if total_miss:
        fate = ("were dropped as not visible" if args.mask_source == "instance"
                else "fell back to the semantic mask")
        print("note: %d objects had no instance-mask pixels and %s (expected for "
              "tiny, distant or fully hidden cars; a large share means the "
              "instance-id decoding is wrong)" % (total_miss, fate))


if __name__ == "__main__":
    main()
