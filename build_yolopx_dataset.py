#!/usr/bin/env python3
"""Raw CARLA capture + CarFree labels -> YOLOPX (BDD100K-style) dataset tree.

    dataset_root/
      images/{all,train,val}/town04_seq0001_frame00012.jpg
      det_annotations/{all,train,val}/town04_seq0001_frame00012.json
      da_seg_annotations/{train,val}      (empty: detection first)
      ll_seg_annotations/{train,val}      (empty)
      calib/seq0001.txt                   (KITTI-style, for audit only)

Everything lands in `all/` -- run split_dataset.py afterwards.
Images are hardlinked by default (instant, no extra disk); --copy to duplicate.

    python3 build_yolopx_dataset.py --raw out/raw --dataset dataset_root
"""
import argparse
import json
import os
import shutil

from carfree_gt import build_K

SUBSETS = ("all", "train", "val")
LANE_DIRS = {"marking": "ll", "continuous": "ll_cont"}      # written by seg_gt.py


def link_or_copy(src, dst, copy):
    if os.path.exists(dst):
        os.remove(dst)
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_calib(path, cam, town, seq):
    K = build_K(cam["width"], cam["height"], cam["fov"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P2 = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]
    x, y, z = cam["mount_xyz"]
    with open(path, "w") as fh:
        fh.write("# %s %s -- forward RGB camera, KITTI cam2 style\n" % (town, seq))
        fh.write("image_size: %d %d\n" % (cam["width"], cam["height"]))
        fh.write("fov_h_deg: %.4f\n" % cam["fov"])
        fh.write("K: %s\n" % " ".join("%.6f" % v for v in K.reshape(-1)))
        fh.write("P2: %s\n" % " ".join("%.6f" % v for v in P2))
        # camera pose in the ego-vehicle frame (UE convention: x fwd, y right, z up)
        fh.write("Tr_cam_to_ego: %.6f %.6f %.6f  roll_pitch_yaw_deg: %.3f %.3f %.3f\n"
                 % (x, y, z, cam["mount_rpy"][0], cam["mount_rpy"][1], cam["mount_rpy"][2]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="out/raw")
    ap.add_argument("--dataset", default="dataset_root")
    ap.add_argument("--copy", action="store_true", help="copy images instead of hardlinking")
    ap.add_argument("--lanes", choices=sorted(LANE_DIRS), default="marking",
                    help="marking = painted lines as rendered (dashes stay dashed); "
                         "continuous = lane edges joined into unbroken lines")
    ap.add_argument("--keep-empty", action="store_true",
                    help="also emit frames whose label list is empty")
    args = ap.parse_args()

    for kind in ("images", "det_annotations", "da_seg_annotations", "ll_seg_annotations"):
        for sub in SUBSETS:
            os.makedirs(os.path.join(args.dataset, kind, sub), exist_ok=True)
        # wipe the pool: after re-running the GT with stricter settings, frames that
        # no longer qualify would otherwise stay here with their old labels
        pool = os.path.join(args.dataset, kind, "all")
        for stale in os.listdir(pool):
            os.remove(os.path.join(pool, stale))
    os.makedirs(os.path.join(args.dataset, "calib"), exist_ok=True)

    runs = sorted(d for d in os.listdir(args.raw) if os.path.isdir(os.path.join(args.raw, d)))
    n_img = n_box = n_empty = n_trunc = n_occ = n_noseg = 0
    for i, run in enumerate(runs):
        rd = os.path.join(args.raw, run)
        det_dir = os.path.join(rd, "det")
        if not os.path.isdir(det_dir):
            print("skip %s: no det/ (run carfree_gt.py first)" % run)
            continue
        meta = json.load(open(os.path.join(rd, "run.json")))
        town = meta.get("map", "Town04").lower()
        seq = "seq%04d" % i
        write_calib(os.path.join(args.dataset, "calib", seq + ".txt"), meta["camera"], town, seq)

        for fn in sorted(os.listdir(det_dir)):
            stem = os.path.splitext(fn)[0]
            ann = json.load(open(os.path.join(det_dir, fn)))
            if not ann["labels"] and not args.keep_empty:
                n_empty += 1
                continue
            name = "%s_%s_frame%05d" % (town, seq, int(stem))
            da_src = os.path.join(rd, "da", stem + ".png")
            ll_src = os.path.join(rd, LANE_DIRS[args.lanes], stem + ".png")
            if not (os.path.exists(da_src) and os.path.exists(ll_src)):
                n_noseg += 1
                continue
            link_or_copy(os.path.join(rd, "rgb", stem + ".jpg"),
                         os.path.join(args.dataset, "images", "all", name + ".jpg"),
                         args.copy)
            link_or_copy(da_src, os.path.join(args.dataset, "da_seg_annotations",
                                              "all", name + ".png"), args.copy)
            link_or_copy(ll_src, os.path.join(args.dataset, "ll_seg_annotations",
                                              "all", name + ".png"), args.copy)
            # schema that lib/dataset/bdd.py reads: frames[0].objects
            json.dump({"name": name + ".jpg",
                       "attributes": {"weather": meta.get("weather"), "scene": "highway",
                                      "timeofday": "daytime",
                                      "scenario": meta.get("scenario")},
                       "frames": [{"objects": ann["labels"]}]},
                      open(os.path.join(args.dataset, "det_annotations", "all",
                                        name + ".json"), "w"))
            n_img += 1
            n_box += len(ann["labels"])
            n_trunc += sum(l["attributes"]["truncated"] for l in ann["labels"])
            n_occ += sum(l["attributes"]["occluded"] for l in ann["labels"])

    print("images        : %d  (%d empty frames dropped)" % (n_img, n_empty))
    if n_noseg:
        print("skipped       : %d frames without da/ll masks -- run seg_gt.py first"
              % n_noseg)
    print("boxes         : %d  (%.2f per image)" % (n_box, n_box / max(n_img, 1)))
    print("truncated     : %d  (%.1f%%)" % (n_trunc, 100.0 * n_trunc / max(n_box, 1)))
    print("occluded      : %d  (%.1f%%)" % (n_occ, 100.0 * n_occ / max(n_box, 1)))
    print("dataset root  : %s" % os.path.abspath(args.dataset))


if __name__ == "__main__":
    main()
