"""Tests for the real-image camera, its perception stack and the tracker."""

import math
import unittest

import numpy as np

from src.perception.onnx_detector import DEFAULT_ONNX, box_iou, letterbox, nms
from src.perception.tracking import Detection3D, GroundTracker
from src.simulation.photo_camera import PhotoCamera, available
from src.simulation.warehouse_env import (
    BOX, PALLET, PERSON, EnvConfig, Obstacle, WarehouseEnv,
)

HAVE_ASSETS = available()
HAVE_DETECTOR = HAVE_ASSETS and DEFAULT_ONNX.exists()


def empty_env(seed=0, **kw):
    cfg = EnvConfig(n_workers=(0, 0), n_forklifts=(0, 0), n_pallets=(0, 0),
                    n_boxes=(0, 0), shadow_prob=(0.0, 0.0), **kw)
    env = WarehouseEnv(cfg)
    env.reset(seed=seed)
    return env


def ahead(env, category, dist, **kw):
    x = env.x + dist * math.cos(env.theta)
    y = env.y + dist * math.sin(env.theta)
    if category == PERSON:
        o = Obstacle(PERSON, x, y, radius=0.28, uid=777)
    else:
        o = Obstacle(category, x, y, 0.3, 0.3, uid=777, **kw)
    env.statics.append(o)
    return o


class TestDetectorMaths(unittest.TestCase):

    def test_letterbox_keeps_aspect(self):
        img = np.zeros((100, 400, 3), np.uint8)
        out, r, left, top = letterbox(img, (384, 640))
        self.assertEqual(out.shape, (384, 640, 3))
        self.assertAlmostEqual(r, 1.6)
        self.assertEqual((left, top), (0, 112))

    def test_nms_drops_duplicates(self):
        b = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]],
                     np.float32)
        keep = nms(b, np.array([0.9, 0.8, 0.7]), 0.5)
        self.assertEqual(keep, [0, 2])
        self.assertAlmostEqual(float(box_iou(b[:1], b[:1])[0, 0]), 1.0,
                               places=5)


class TestTracker(unittest.TestCase):

    def det(self, x, y, conf=0.8, cls=0):
        return Detection3D(x, y, math.hypot(x, y), math.atan2(y, x), cls,
                           conf, (0, 0, 10, 20), 0.3, 1.7)

    def test_confirm_velocity_and_coasting(self):
        tr = GroundTracker(dt=0.2)
        for k in range(6):                      # walks 1 m/s along x
            tracks = tr.step([self.det(3.0 + 0.2 * k, 1.0, conf=0.5)], 0.0)
        self.assertEqual(len(tracks), 1)
        t = tracks[0]
        self.assertTrue(t.confirmed)
        self.assertAlmostEqual(t.state[2], 1.0, delta=0.35)
        tid = t.tid
        tracks = tr.step([], 0.0)               # one missed frame
        self.assertEqual(tracks[0].tid, tid)
        self.assertFalse(tracks[0].matched)
        for _ in range(6):                      # gone for good
            tracks = tr.step([], 0.0)
        self.assertEqual(tracks, [])

    def test_weak_detection_cannot_start_a_track(self):
        tr = GroundTracker(dt=0.2)
        for _ in range(4):
            tracks = tr.step([self.det(2.0, 0.0, conf=0.2)], 0.0)
        self.assertEqual(tracks, [])
        tr.step([self.det(2.0, 0.0, conf=0.7)], 0.0)
        tracks = tr.step([self.det(2.0, 0.0, conf=0.2)], 0.0)
        self.assertEqual(len(tracks), 1)        # ... but it can extend one
        self.assertTrue(tracks[0].matched)


@unittest.skipUnless(HAVE_ASSETS, "real-image asset bank missing")
class TestPhotoCamera(unittest.TestCase):

    def setUp(self):
        self.cam = PhotoCamera(effects=False)

    def test_frame_depth_and_ground_truth(self):
        env = empty_env(3)
        ahead(env, PERSON, 3.0)
        self.cam.reset(env, 3)
        fr = self.cam.render(env)
        self.assertEqual(fr.rgb.shape, (384, 640, 3))
        self.assertEqual(fr.rgb.dtype, np.uint8)
        self.assertEqual(fr.depth.shape, (384, 640))
        self.assertEqual(len(fr.gt), 1)
        g = fr.gt[0]
        self.assertEqual(g["class_name"], "person")
        self.assertAlmostEqual(g["range"], 3.0 - 0.28, delta=0.05)
        x0, y0, x1, y1 = g["bbox"]
        # in the middle of the image, feet on the floor line of that range
        self.assertLess(abs((x0 + x1) / 2 - self.cam.cx), 25)
        feet = self.cam.cy + self.cam.f * self.cam.cam_h / g["z"]
        self.assertAlmostEqual(y1, feet, delta=4)
        # the depth image shows the person nearer than the wall behind
        cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
        self.assertAlmostEqual(float(fr.depth_true[cy, cx]), g["z"],
                               delta=0.02)

    def test_static_depth_matches_render_without_objects(self):
        env = empty_env(4)
        self.cam.reset(env, 4)
        fr = self.cam.render(env)
        prior = self.cam.static_depth(env)
        self.assertLess(float(np.abs(prior - fr.depth_true).max()), 1e-3)

    def test_same_seed_same_frame(self):
        env = empty_env(5)
        ahead(env, PALLET, 2.5)
        frames = []
        for _ in range(2):
            self.cam.reset(env, 11)
            frames.append(self.cam.render(env, np.random.default_rng(1)).rgb)
        self.assertTrue(np.array_equal(frames[0], frames[1]))

    def test_held_out_sprites_only_by_default(self):
        self.assertEqual(set(self.cam.bank.splits), {"val", "test"})
        for lst in self.cam.bank.by_class.values():
            self.assertTrue(all(s.split in ("val", "test") for s in lst))


@unittest.skipUnless(HAVE_DETECTOR, "ONNX detector or assets missing")
class TestRealImagePerception(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.perception.onnx_detector import OnnxDetector
        cls.det = OnnxDetector(threads=1)

    def env(self, seed, **kw):
        from src.perception.camera_perception import attach_real_camera
        cfg = EnvConfig(n_workers=(0, 0), n_forklifts=(0, 0),
                        n_pallets=(0, 0), n_boxes=(0, 0),
                        shadow_prob=(0.0, 0.0), **kw)
        env = attach_real_camera(WarehouseEnv(cfg), detector=self.det)
        env.reset(seed=seed)
        return env

    def test_observation_shape_is_unchanged(self):
        env = self.env(1)
        obs = env._observation()
        self.assertEqual(obs.shape, (env.obs_dim,))
        self.assertTrue(np.all(np.abs(obs) <= 1.0))
        self.assertEqual(env.frame.rgb.shape, (384, 640, 3))

    def test_person_is_detected_and_ranged(self):
        env = self.env(3)
        ahead(env, PERSON, 3.0)
        for _ in range(3):                 # a few frames: track confirms
            env._perceive()
        objs = [o for o in env.vision_result.objects if o["name"] == "person"]
        self.assertEqual(len(objs), 1)
        self.assertAlmostEqual(objs[0]["range"], 2.72, delta=0.35)
        mid = env.cfg.n_rays // 2
        self.assertEqual(env.ray_labels[mid], PERSON)
        self.assertLess(env.fused_rays[mid], 3.1)

    def test_unknown_object_is_found_by_depth_only(self):
        env = self.env(3)
        ahead(env, BOX, 2.0)
        for _ in range(3):
            env._perceive()
        mid = env.cfg.n_rays // 2
        self.assertLess(env.fused_rays[mid], 2.0)
        self.assertEqual(env.ray_labels[mid], BOX)      # "unknown"
        self.assertGreaterEqual(len(env.vision_result.unknown), 1)

        blind = self.env(3, perception_mode="detector_only")
        ahead(blind, BOX, 2.0)
        for _ in range(3):
            blind._perceive()
        self.assertGreater(blind.fused_rays[mid], 2.0)  # invisible

    def test_empty_aisle_reads_like_the_map(self):
        env = self.env(6)
        for _ in range(3):
            env._perceive()
        m = env.true_rays < env.cfg.max_range - 0.3
        err = np.abs(env.fused_rays[m] - env.true_rays[m]) / env.true_rays[m]
        self.assertLess(float(np.median(err)), 0.08)
        self.assertEqual(env.vision_result.objects, [])


if __name__ == "__main__":
    unittest.main()
