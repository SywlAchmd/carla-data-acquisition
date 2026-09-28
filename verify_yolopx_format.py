#!/usr/bin/env python3
"""Check the dataset against what YOLOPX's loader actually does.

Replays lib/dataset/bdd.py::_get_db and the mask handling in
AutoDriveDataset.__getitem__ -- the same path derivation, the same JSON keys,
the same thresholds -- without importing them, so it runs even where
albumentations is not installed.

    python3 verify_yolopx_format.py --dataset dataset_root --splits train,val,test

Exits non-zero if anything YOLOPX needs is missing or malformed.
"""
import argparse
import json
import os
import sys

import cv2

# lib/dataset/convert.py, with single_cls = True in bdd.py
ID_DICT_SINGLE = {"car": 0, "bus": 1, "truck": 2, "train": 3}
ORG_IMG_SIZE = (720, 1280)      # cfg.DATASET.ORG_IMG_SIZE


def check_split(root, split):
    # bdd.py drives the whole dataset off the da_seg mask listing
    mask_root = os.path.join(root, "da_seg_annotations", split)
    if not os.path.isdir(mask_root):
        return ["%s: da_seg_annotations/%s missing" % (split, split)], 0, 0
    masks = sorted(f for f in os.listdir(mask_root) if f.endswith(".png"))
    problems, n_box = [], 0

    for fn in masks:
        mask_path = os.path.join(mask_root, fn)
        label_path = (mask_path.replace(os.path.join(root, "da_seg_annotations"),
                                        os.path.join(root, "det_annotations"))
                      .replace(".png", ".json"))
        image_path = (mask_path.replace(os.path.join(root, "da_seg_annotations"),
                                        os.path.join(root, "images"))
                      .replace(".png", ".jpg"))
        lane_path = mask_path.replace(os.path.join(root, "da_seg_annotations"),
                                      os.path.join(root, "ll_seg_annotations"))
        for p, what in ((label_path, "det json"), (image_path, "image"),
                        (lane_path, "ll mask")):
            if not os.path.exists(p):
                problems.append("%s: %s missing for %s" % (split, what, fn))
        if problems and len(problems) > 8:
            break
        if not all(os.path.exists(p) for p in (label_path, image_path, lane_path)):
            continue

        label = json.load(open(label_path))
        try:
            objects = label["frames"][0]["objects"]
        except (KeyError, IndexError, TypeError):
            problems.append("%s: %s has no frames[0].objects" % (split, fn))
            continue
        for obj in objects:
            if "box2d" not in obj:
                problems.append("%s: %s object without box2d" % (split, fn))
                continue
            if obj["category"] not in ID_DICT_SINGLE:
                problems.append("%s: %s unknown category %r"
                                % (split, fn, obj["category"]))
                continue
            b = obj["box2d"]
            h, w = ORG_IMG_SIZE
            cx = (b["x1"] + b["x2"]) / 2.0 / w
            cy = (b["y1"] + b["y2"]) / 2.0 / h
            bw = (b["x2"] - b["x1"]) / w
            bh = (b["y2"] - b["y1"]) / h
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                problems.append("%s: %s box out of range %s" % (split, fn, b))
            n_box += 1

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        seg = cv2.imread(mask_path, 0)
        lane = cv2.imread(lane_path, 0)
        if img is None or seg is None or lane is None:
            problems.append("%s: %s unreadable image/mask" % (split, fn))
            continue
        if img.shape[:2] != ORG_IMG_SIZE:
            problems.append("%s: %s is %s, cfg says %s"
                            % (split, fn, img.shape[:2], ORG_IMG_SIZE))
        if seg.shape != ORG_IMG_SIZE or lane.shape != ORG_IMG_SIZE:
            problems.append("%s: %s mask size mismatch" % (split, fn))
        # __getitem__ does cv2.threshold(x, 1, 255, BINARY): anything > 1 is
        # foreground, so a mask whose max is 1 or 0 silently trains on nothing
        if seg.max() <= 1:
            problems.append("%s: %s da mask has no foreground" % (split, fn))
        if lane.max() <= 1 and lane.max() != 0:
            problems.append("%s: %s ll mask max is 1, threshold would drop it"
                            % (split, fn))
    return problems, len(masks), n_box


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="dataset_root")
    ap.add_argument("--splits", default="train,val,test")
    args = ap.parse_args()

    all_problems, total = [], 0
    for split in args.splits.split(","):
        problems, n, n_box = check_split(args.dataset, split.strip())
        status = "OK" if not problems else "%d PROBLEMS" % len(problems)
        print("%-6s %5d frames %6d boxes   %s" % (split, n, n_box, status))
        all_problems += problems
        total += n
    print("-" * 52)
    if all_problems:
        for p in all_problems[:15]:
            print("  !", p)
        if len(all_problems) > 15:
            print("  ... and %d more" % (len(all_problems) - 15))
        sys.exit(1)
    print("%d frames pass every check YOLOPX's loader performs" % total)
    print("""
config to set in lib/config/default.py (or your yaml):
  _C.DATASET.DATAROOT  = '%s/images'
  _C.DATASET.LABELROOT = '%s/det_annotations'
  _C.DATASET.MASKROOT  = '%s/da_seg_annotations'
  _C.DATASET.LANEROOT  = '%s/ll_seg_annotations'
  _C.DATASET.TRAIN_SET = 'train'
  _C.DATASET.TEST_SET  = 'val'        # 'test' for the held-out split
  _C.DATASET.ORG_IMG_SIZE = [720, 1280]
  _C.num_seg_class = 2""" % ((os.path.abspath(args.dataset),) * 4))


if __name__ == "__main__":
    main()
