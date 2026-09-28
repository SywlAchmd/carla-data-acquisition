#!/usr/bin/env python3
"""Overlay the generated boxes on the images -- the eyeball check for precision.

    python3 visualize_boxes.py --dataset dataset_root --n 40 --out check/
    python3 visualize_boxes.py --raw out/raw/run_0000 --n 40 --out check/

Green = clean, orange = truncated, red = occluded. --masks also tints the
drivable area and the lane lines so you can check all three heads at once.
Prints a box-size histogram so you can see whether the far/near phases are
actually represented.
"""
import argparse
import json
import os
import random

import cv2
import numpy as np


def load_labels(path):
    """Accepts both the schema YOLOPX reads (frames[0].objects) and the raw
    per-run files carfree_gt.py writes (labels)."""
    d = json.load(open(path))
    return d["frames"][0]["objects"] if "frames" in d else d["labels"]


def tint(img, mask_path, color, alpha=0.4):
    m = cv2.imread(mask_path, 0)
    if m is None:
        return img
    layer = np.zeros_like(img)
    layer[m > 0] = color
    return np.where(m[:, :, None] > 0, cv2.addWeighted(img, 1 - alpha, layer, alpha, 0), img)


def draw(img, labels):
    for l in labels:
        b = l["box2d"]
        p1 = (int(round(b["x1"])), int(round(b["y1"])))
        p2 = (int(round(b["x2"])), int(round(b["y2"])))
        a = l.get("attributes", {})
        color = (0, 0, 255) if a.get("occluded") else (0, 165, 255) if a.get("truncated") \
            else (0, 255, 0)
        cv2.rectangle(img, p1, p2, color, 2)
        tag = "%s %.0fm" % (l["category"], l.get("carla", {}).get("distance", 0))
        cv2.putText(img, tag, (p1[0], max(12, p1[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset")
    ap.add_argument("--raw", help="a single run_XXXX directory instead")
    ap.add_argument("--split", default="all")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--masks", action="store_true",
                    help="tint da_seg (blue) and ll_seg (magenta) over the image")
    args = ap.parse_args()

    if args.raw:
        img_dir, ann_dir, ext = (os.path.join(args.raw, "rgb"),
                                 os.path.join(args.raw, "det"), ".jpg")
        da_dir, ll_dir = os.path.join(args.raw, "da"), os.path.join(args.raw, "ll")
    elif args.dataset:
        img_dir = os.path.join(args.dataset, "images", args.split)
        ann_dir = os.path.join(args.dataset, "det_annotations", args.split)
        da_dir = os.path.join(args.dataset, "da_seg_annotations", args.split)
        ll_dir = os.path.join(args.dataset, "ll_seg_annotations", args.split)
        ext = ".jpg"
    else:
        raise SystemExit("pass --dataset or --raw")

    os.makedirs(args.out, exist_ok=True)
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(ann_dir) if f.endswith(".json"))
    random.Random(args.seed).shuffle(names)

    areas, n_box = [], 0
    for n in names:
        labels = load_labels(os.path.join(ann_dir, n + ".json"))
        n_box += len(labels)
        for l in labels:
            b = l["box2d"]
            areas.append(np.sqrt(max(1.0, (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))))

    for n in names[:args.n]:
        img = cv2.imread(os.path.join(img_dir, n + ext))
        if img is None:
            continue
        if args.masks:
            img = tint(img, os.path.join(da_dir, n + ".png"), (255, 120, 0))
            img = tint(img, os.path.join(ll_dir, n + ".png"), (255, 0, 255))
        cv2.imwrite(os.path.join(args.out, n + ".jpg"),
                    draw(img, load_labels(os.path.join(ann_dir, n + ".json"))))

    a = np.array(areas) if areas else np.zeros(1)
    print("frames %d, boxes %d (%.2f/frame)" % (len(names), n_box, n_box / max(len(names), 1)))
    print("box sqrt(area) px: min %.0f  p25 %.0f  median %.0f  p75 %.0f  max %.0f"
          % (a.min(), np.percentile(a, 25), np.median(a), np.percentile(a, 75), a.max()))
    print("small(<32px) %.1f%%  medium %.1f%%  large(>96px) %.1f%%"
          % (100 * (a < 32).mean(), 100 * ((a >= 32) & (a <= 96)).mean(), 100 * (a > 96).mean()))
    print("overlays -> %s" % os.path.abspath(args.out))


if __name__ == "__main__":
    main()
