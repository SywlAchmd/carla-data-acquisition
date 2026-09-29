#!/usr/bin/env python3
"""Drivable-area (da_seg) and lane-line (ll_seg) masks from the raw CARLA capture.

    python3 seg_gt.py --raw out/raw

Writes out/raw/<town>_run_XXXX/{da,ll,ll_cont}/NNNNNN.png -- 8-bit, 0 = background,
255 = foreground, which is what YOLOPX's loader expects (it thresholds at >1).

Drivable area (--da road, the default)
    Every road pixel of the semantic camera (Road + RoadLine): the ego lane, the
    oncoming lane, every branch of a junction. Vehicles, sidewalks and barriers
    have their own tags, so they are already cut out. Needs no lane geometry.

Drivable area (--da carriageway)
    The ego's own carriageway only: its lane plus every same-direction Driving
    lane beside it, taken from the OpenDRIVE geometry the capture stored per
    frame, projected into the image and filled. That geometric region is then
    intersected with the road pixels of the semantic camera, which clips it to
    the road actually rendered and punches out anything in the way -- vehicles,
    guard rails, barriers.

    A plain semantic 'road' mask cannot do this: on a divided highway it also
    returns the opposite carriageway, which is road surface but not drivable.

Lane lines
    CARLA renders lane markings as their own semantic class (tag 24), so the
    ground truth is already pixel-exact and just needs extracting. Every visible
    marking is kept, including the ones across a median -- BDD100K labels lane
    lines wherever they are visible too.

Continuous lane lines (ll_cont)
    The lane boundaries of the ego carriageway drawn as unbroken lines from the
    same OpenDRIVE geometry, so dashed markings join up and every lane shows as
    a closed strip. Clipped to the road pixels like the drivable area, so a car
    in front still hides the line behind it. Only the ego carriageway: lines
    across the median exist in `ll` but not here.
    build_yolopx_dataset.py --lanes picks which of the two goes into the dataset.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

from carfree_gt import (TAG_ROADLINE, TAG_ROAD, build_K, clip_polygon_near,
                        load_tags, project, world_to_cam)

FG = 255


def drivable_mask(meta, K, sem_tag):
    """Project the stored lane strips, fill them, keep only real road pixels."""
    h, w = sem_tag.shape
    geo = np.zeros((h, w), np.uint8)
    cam_m = meta["camera"]["world_matrix"]
    for lane in meta.get("lanes", []):
        edges = lane["edges"]
        for i in range(len(edges) - 1):
            quad = [edges[i][0], edges[i][1], edges[i + 1][1], edges[i + 1][0]]
            poly = clip_polygon_near(world_to_cam(quad, cam_m))
            if poly is None:
                continue
            uv = project(poly, K)
            if not np.isfinite(uv).all():
                continue
            uv = np.clip(uv, -1e4, 1e4).astype(np.int32)
            cv2.fillConvexPoly(geo, uv, FG)
    road = np.isin(sem_tag, (TAG_ROAD, TAG_ROADLINE))
    return np.where(road, geo, 0).astype(np.uint8)


def road_mask(sem_tag):
    return np.where(np.isin(sem_tag, (TAG_ROAD, TAG_ROADLINE)), FG, 0).astype(np.uint8)


def lane_mask(sem_tag):
    return np.where(sem_tag == TAG_ROADLINE, FG, 0).astype(np.uint8)


def lane_mask_continuous(meta, K, sem_tag, width=8, near=1.0):
    """Every lane edge as one unbroken line, kept only where road is visible.

    Clipped at 1 m instead of NEAR_PLANE: the road is only visible from ~3 m
    ahead anyway, and it keeps the projected end points in a sane pixel range.
    """
    h, w = sem_tag.shape
    geo = np.zeros((h, w), np.uint8)
    cam_m = meta["camera"]["world_matrix"]
    for lane in meta.get("lanes", []):
        for side in (0, 1):                      # left edge, right edge
            cam = world_to_cam([e[side] for e in lane["edges"]], cam_m)
            for a, b in zip(cam[:-1], cam[1:]):
                if a[2] < near and b[2] < near:
                    continue
                if a[2] < near:
                    a = a + (near - a[2]) / (b[2] - a[2]) * (b - a)
                elif b[2] < near:
                    b = b + (near - b[2]) / (a[2] - b[2]) * (a - b)
                uv = np.round(project(np.array([a, b]), K)).astype(np.int64)
                cv2.line(geo, tuple(map(int, uv[0])), tuple(map(int, uv[1])), FG, width)
    road = np.isin(sem_tag, (TAG_ROAD, TAG_ROADLINE))
    return np.where(road, geo, 0).astype(np.uint8)


def process_run(run_dir, ll_width=8, da_mode="road"):
    run = json.load(open(os.path.join(run_dir, "run.json")))
    cam = run["camera"]
    K = build_K(cam["width"], cam["height"], cam["fov"])
    for sub in ("da", "ll", "ll_cont"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)

    n, da_cov, ll_cov, no_geo = 0, 0.0, 0.0, 0
    gaps = []
    with open(os.path.join(run_dir, "meta.jsonl")) as fh:
        for line in fh:
            meta = json.loads(line)
            stem = "%06d" % meta["frame"]
            sem = load_tags(run_dir, stem)
            if not meta.get("lanes"):
                no_geo += 1
            da = road_mask(sem) if da_mode == "road" else drivable_mask(meta, K, sem)
            ll = lane_mask(sem)
            cv2.imwrite(os.path.join(run_dir, "da", stem + ".png"), da)
            cv2.imwrite(os.path.join(run_dir, "ll", stem + ".png"), ll)
            cv2.imwrite(os.path.join(run_dir, "ll_cont", stem + ".png"),
                        lane_mask_continuous(meta, K, sem, ll_width))
            n += 1
            da_cov += (da > 0).mean()
            ll_cov += (ll > 0).mean()
            # visible road sitting ABOVE the labelled region: if this grows, the
            # capture's --da-ahead is too short and the horizon goes unlabelled
            da_rows = np.flatnonzero((da > 0).any(axis=1))
            rd_rows = np.flatnonzero(np.isin(sem, (TAG_ROAD, TAG_ROADLINE)).any(axis=1))
            if len(da_rows) and len(rd_rows):
                gaps.append(int(da_rows[0] - rd_rows[0]))
    return (n, da_cov / max(n, 1), ll_cov / max(n, 1), no_geo,
            float(np.median(gaps)) if gaps else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="out/raw")
    ap.add_argument("--ll-width", type=int, default=8,
                    help="line thickness in px for the continuous lane lines (ll_cont)")
    ap.add_argument("--da", choices=("road", "carriageway"), default="road",
                    help="road = every road pixel, oncoming lanes and junctions "
                         "included; carriageway = ego lane + same-direction lanes only")
    args = ap.parse_args()

    runs = sorted(d for d in os.listdir(args.raw)
                  if os.path.isdir(os.path.join(args.raw, d)))
    if not runs:
        sys.exit("no runs found in %s" % args.raw)
    tn, tda, tll, tno, gaps = 0, 0.0, 0.0, 0, []
    for r in runs:
        n, da, ll, no_geo, gap = process_run(os.path.join(args.raw, r), args.ll_width,
                                          args.da)
        print("%-14s %5d frames   drivable %5.1f%%   lane lines %4.2f%%   "
              "horizon gap %3.0f px%s"
              % (r, n, 100 * da, 100 * ll, gap, "" if not no_geo else
                 "   [%d frames without lane geometry]" % no_geo))
        gaps.append(gap)
        tn += n
        tda += da * n
        tll += ll * n
        tno += no_geo
    print("-" * 72)
    med_gap = float(np.median(gaps)) if gaps else 0.0
    print("TOTAL          %5d frames   drivable %5.1f%%   lane lines %4.2f%%   "
          "horizon gap %3.0f px" % (tn, 100 * tda / max(tn, 1),
                                    100 * tll / max(tn, 1), med_gap))
    if med_gap > 25:
        print("warning: %.0f px of visible road sits above the drivable label -- "
              "raise --da-ahead on the capture side" % med_gap)
    if tno and args.da == "carriageway":
        print("warning: %d frames had no lane geometry in meta.jsonl -- they were "
              "captured before --da-* existed and their da masks are empty" % tno)


if __name__ == "__main__":
    main()
