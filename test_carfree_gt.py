#!/usr/bin/env python3
"""Self-check for the CarFree GT algorithms. No CARLA, no framework:

    python3 test_carfree_gt.py
"""
import numpy as np

from carfree_gt import (TAG_CAR, TAG_ROADLINE, build_K, clip_near, fit_to_boundary,
                        initial_bbox, is_visible, occlusion_attrs, project,
                        world_to_cam)

H, W = 720, 1280


def car_mask(x1, y1, x2, y2):
    m = np.zeros((H, W), bool)
    m[y1:y2, x1:x2] = True
    return m


def test_run_number():
    from carfree_gt import run_number
    assert run_number("town04_run_0003") == 3
    assert run_number("town10hd_opt_run_0120") == 120
    assert run_number("run_0007") == 7              # folders from before the town prefix
    assert run_number("check") is None and run_number("town04_run_x") is None


def test_projection():
    K = build_K(W, H, 90.0)
    assert abs(K[0, 0] - 640.0) < 1e-6, K            # f = w / 2tan(45)
    # identity camera pose: UE +x is straight ahead, +y is to the right
    uv = project(world_to_cam([[10.0, 0.0, 0.0]], np.eye(4)), K)[0]
    assert np.allclose(uv, [640.0, 360.0]), uv
    uv = project(world_to_cam([[10.0, 1.0, 0.0]], np.eye(4)), K)[0]
    assert np.allclose(uv, [704.0, 360.0]), uv       # 1 m right at 10 m -> +64 px
    uv = project(world_to_cam([[10.0, 0.0, 1.0]], np.eye(4)), K)[0]
    assert np.allclose(uv, [640.0, 296.0]), uv       # UE +z is up -> smaller v


def test_clip_near():
    # cuboid straddling the camera plane (the 'alongside' phase)
    cam = np.array([[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)
                    for sz in (-2.0, 2.0)])
    kept = clip_near(cam)
    assert kept is not None and (kept[:, 2] >= 0.2 - 1e-9).all()
    # entirely behind -> dropped
    behind = cam.copy()
    behind[:, 2] -= 10.0
    assert clip_near(behind) is None


def test_algorithm1():
    pts = np.array([[10.0, 20.0], [30.0, 5.0], [-4.0, 44.0]])
    assert initial_bbox(pts) == (-4.0, 5.0, 30.0, 44.0)


def test_algorithm2_visibility():
    m = car_mask(600, 300, 700, 400)
    assert is_visible((590.0, 290.0, 710.0, 410.0), m, "center")
    # centre lands on a hole (occluder) -> paper's single-probe test fails ...
    m2 = m.copy()
    m2[340:370, 630:670] = False
    assert not is_visible((590.0, 290.0, 710.0, 410.0), m2, "center")
    # ... and the 5-point extension recovers it
    assert is_visible((601.0, 301.0, 699.0, 399.0), m2, "multi5")


def test_algorithm3_fit():
    m = car_mask(600, 300, 700, 400)
    loose = (500.0, 200.0, 800.0, 500.0)             # deliberately far too big
    assert fit_to_boundary(loose, m) == (600.0, 300.0, 700.0, 400.0)
    # already tight -> unchanged
    assert fit_to_boundary((600.0, 300.0, 700.0, 400.0), m) == (600.0, 300.0, 700.0, 400.0)
    # empty region -> rejected
    assert fit_to_boundary((0.0, 0.0, 100.0, 100.0), m) is None


def test_truncation_clamped_to_frame():
    m = car_mask(0, 300, 90, 500)                    # car running off the left edge
    fitted = fit_to_boundary((-260.0, 250.0, 120.0, 560.0), m)
    assert fitted == (0.0, 300.0, 90.0, 500.0), fitted
    m = car_mask(1180, 300, W, 500)                  # ... and off the right edge
    fitted = fit_to_boundary((1100.0, 250.0, 1600.0, 560.0), m)
    assert fitted == (1180.0, 300.0, float(W), 500.0), fitted


def test_second_car_does_not_stop_the_shrink():
    """Why fit_to_boundary scans inside the box only: a car sharing a column with
    the target must not halt the shrink early."""
    m = car_mask(600, 300, 700, 400)
    m[600:650, 500:900] = True                       # another car lower in the frame
    assert fit_to_boundary((450.0, 250.0, 950.0, 450.0), m) == (600.0, 300.0, 700.0, 400.0)


def test_occlusion_attrs():
    actor = car_mask(600, 300, 700, 400)
    clean = occlusion_attrs((600, 300, 700, 400), actor, np.zeros((H, W), bool))
    assert clean[0] is False and clean[1] == 1.0
    other = car_mask(660, 300, 700, 400)
    assert occlusion_attrs((600, 300, 700, 400), actor, other)[0] is True


def test_instance_id_decoding(tmp="/tmp/_carfree_inst_test.png"):
    """Locks the byte order that a live CARLA 0.9.16 capture actually produces:
    byte0 = actor-id high byte, byte1 = low byte, byte2 = semantic tag. Getting
    this backwards silently empties every per-actor mask."""
    import cv2
    from carfree_gt import load_instance
    actor_id = 399
    img = np.zeros((4, 4, 3), np.uint8)
    img[1:3, 1:3] = [(actor_id >> 8) & 0xFF, actor_id & 0xFF, TAG_CAR]
    cv2.imwrite(tmp, img)
    tag, iid = load_instance(tmp)
    assert tag[2, 2] == TAG_CAR
    assert iid[2, 2] == actor_id, iid[2, 2]
    assert (iid == actor_id).sum() == 4


def test_drivable_mask_is_clipped_to_real_road_pixels():
    """The geometric lane strip must survive projection AND be cut down to the
    road pixels the semantic camera actually rendered."""
    from seg_gt import drivable_mask
    K = build_K(W, H, 90.0)
    # one 3.5 m lane running 2..40 m ahead; camera at the origin, road 1.65 m below
    edges = [[[float(d), -1.75, -1.65], [float(d), 1.75, -1.65]]
             for d in range(2, 42, 4)]
    meta = {"camera": {"world_matrix": np.eye(4).tolist()},
            "lanes": [{"kind": "ego", "edges": edges}]}
    sem = np.full((H, W), 11, np.uint8)          # sky
    sem[H // 2:, :] = 1                          # road across the bottom half
    m = drivable_mask(meta, K, sem)
    assert m.max() == 255
    assert (m > 0).sum() > 1000, (m > 0).sum()
    assert not (m[:H // 2] > 0).any()            # never outside the rendered road
    # a car parked on the road removes that patch from the drivable area
    occluded = sem.copy()
    occluded[H // 2:, 500:800] = 14
    assert (drivable_mask(meta, K, occluded) > 0).sum() < (m > 0).sum()
    # and with no road rendered at all, nothing survives
    assert (drivable_mask(meta, K, np.full((H, W), 11, np.uint8)) > 0).sum() == 0


def test_road_mask_keeps_every_road_pixel_but_not_cars():
    from seg_gt import road_mask
    sem = np.full((H, W), 11, np.uint8)
    sem[H // 2:, :] = 1
    sem[H // 2:, 600:610] = 24                   # a marking is road too
    sem[H // 2:, 800:900] = 14                   # a car is not
    m = road_mask(sem)
    assert (m[H // 2:, :800] == 255).all()
    assert not m[H // 2:, 800:900].any() and not m[:H // 2].any()


def test_continuous_lanes_join_dashes_and_hide_behind_cars():
    """ll_cont draws the edges unbroken even where no marking is painted, and
    drops them where something other than road covers the pixel."""
    from seg_gt import lane_mask_continuous
    K = build_K(W, H, 90.0)
    edges = [[[float(d), -1.75, -1.65], [float(d), 1.75, -1.65]] for d in range(0, 80, 6)]
    meta = {"camera": {"world_matrix": np.eye(4).tolist()},
            "lanes": [{"kind": "ego", "edges": edges}]}
    sem = np.full((H, W), 11, np.uint8)
    sem[H // 2:, :] = 1                          # road, no painted marking at all
    m = lane_mask_continuous(meta, K, sem)
    assert m.max() == 255
    # left edge at 1.75 m, 10 m ahead: u = 640 - 64*1.75, v = 360 + 64*1.65
    assert m[466, 528] == 255 and m[466, 752] == 255, "both edges drawn"
    assert m[466, 640] == 0, "nothing between the edges"
    assert not (m[:H // 2] > 0).any()
    car = sem.copy()
    car[440:500, 500:560] = 14                   # a car standing on the left line
    assert lane_mask_continuous(meta, K, car)[466, 528] == 0


def test_lane_mask_reads_the_roadline_class():
    from seg_gt import lane_mask
    sem = np.full((H, W), 1, np.uint8)
    sem[400:404, 100:1100] = 24
    m = lane_mask(sem)
    assert m.max() == 255 and (m > 0).sum() == 4 * 1000


def test_load_tags_falls_back_to_the_instance_camera(tmp="/tmp/_carfree_tags"):
    """Tanpa kamera semantic, peta tag harus terbaca dari kanal R kamera
    instance -- keduanya identik di CARLA 0.9.16."""
    import os
    import shutil
    import cv2
    from carfree_gt import load_tags
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "inst"))
    img = np.zeros((4, 4, 3), np.uint8)
    img[1:3, 1:3] = [1, 143, TAG_CAR]          # B, G, R
    cv2.imwrite(os.path.join(tmp, "inst", "000000.png"), img)
    assert load_tags(tmp, "000000")[2, 2] == TAG_CAR

    os.makedirs(os.path.join(tmp, "sem"))      # kalau ada, yang ini menang
    cv2.imwrite(os.path.join(tmp, "sem", "000000.png"),
                np.full((4, 4), TAG_ROADLINE, np.uint8))
    assert load_tags(tmp, "000000")[2, 2] == TAG_ROADLINE
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  %s" % name)
    print("all good")
