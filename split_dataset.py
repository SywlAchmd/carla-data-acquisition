#!/usr/bin/env python3
"""Random per-frame train/val(/test) split of the `all/` folder.

Per the project brief this splits by FRAME, not by sequence -- neighbouring
frames of one run can land on both sides. That is deliberate for a quick
pipeline check; switch to --by-sequence before quoting any mAP number.

    python3 split_dataset.py --dataset dataset_root --val 0.2
    python3 split_dataset.py --dataset dataset_root --val-seqs 32 --test-seqs 33,34

--val-seqs / --test-seqs pick whole recordings by their run number (the NNNN
in <town>_run_NNNN); every other recording goes to train and --val/--test are
ignored.
"""
import argparse
import json
import os
import random
import shutil


def place(src, dst, mode):
    if os.path.exists(dst):
        os.remove(dst)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(src, dst)
    else:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="dataset_root")
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--test", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--mode", choices=["link", "copy", "move"], default="link")
    ap.add_argument("--by-sequence", action="store_true",
                    help="split whole sequences instead of individual frames")
    ap.add_argument("--val-seqs", default="",
                    help="comma-separated run numbers for val, e.g. 32,33")
    ap.add_argument("--test-seqs", default="",
                    help="comma-separated run numbers for test, e.g. 34")
    args = ap.parse_args()

    all_img = os.path.join(args.dataset, "images", "all")
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(all_img) if f.endswith(".jpg"))
    if not names:
        raise SystemExit("no images in %s" % all_img)

    rng = random.Random(args.seed)
    val_seqs = {int(x) for x in args.val_seqs.split(",") if x.strip()}
    test_seqs = {int(x) for x in args.test_seqs.split(",") if x.strip()}
    if val_seqs & test_seqs:
        raise SystemExit("run(s) %s are in both --val-seqs and --test-seqs"
                         % sorted(val_seqs & test_seqs))
    if val_seqs or test_seqs:
        seq_of = lambda n: int(n.split("_seq")[1].split("_frame")[0])
        found = {seq_of(n) for n in names}
        missing = (val_seqs | test_seqs) - found
        if missing:
            raise SystemExit("run(s) %s not in %s" % (sorted(missing), all_img))
        of = lambda n: ("val" if seq_of(n) in val_seqs
                        else "test" if seq_of(n) in test_seqs else "train")
        splits = {s: [n for n in names if of(n) == s] for s in ("train", "val", "test")}
    elif args.by_sequence:
        seqs = sorted({n.split("_frame")[0] for n in names})
        rng.shuffle(seqs)
        n_val = max(1, int(len(seqs) * args.val))
        n_test = int(len(seqs) * args.test)
        pick = {"val": set(seqs[:n_val]), "test": set(seqs[n_val:n_val + n_test])}
        of = lambda n: ("val" if n.split("_frame")[0] in pick["val"]
                        else "test" if n.split("_frame")[0] in pick["test"] else "train")
        splits = {s: [n for n in names if of(n) == s] for s in ("train", "val", "test")}
    else:
        rng.shuffle(names)
        n_val = int(len(names) * args.val)
        n_test = int(len(names) * args.test)
        splits = {"val": names[:n_val], "test": names[n_val:n_val + n_test],
                  "train": names[n_val + n_test:]}

    for split, items in splits.items():
        for kind, ext in (("images", ".jpg"), ("det_annotations", ".json"),
                          ("da_seg_annotations", ".png"), ("ll_seg_annotations", ".png")):
            d = os.path.join(args.dataset, kind, split)
            os.makedirs(d, exist_ok=True)
            # wipe first: without this, re-splitting with different ratios leaves
            # the previous run's files behind and the same frame ends up in two
            # splits at once -- silent train/val/test leakage
            for stale in os.listdir(d):
                os.remove(os.path.join(d, stale))
        if not items:
            continue
        for kind, ext in (("images", ".jpg"), ("det_annotations", ".json"),
                          ("da_seg_annotations", ".png"), ("ll_seg_annotations", ".png")):
            d = os.path.join(args.dataset, kind, split)
            for n in items:
                place(os.path.join(args.dataset, kind, "all", n + ext),
                      os.path.join(d, n + ext), args.mode)
        print("%-6s %5d frames" % (split, len(items)))

    json.dump({s: sorted(v) for s, v in splits.items() if v},
              open(os.path.join(args.dataset, "split.json"), "w"), indent=1)
    print("manifest -> %s" % os.path.join(args.dataset, "split.json"))


if __name__ == "__main__":
    main()
