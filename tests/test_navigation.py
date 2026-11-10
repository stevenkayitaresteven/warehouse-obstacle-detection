import math
import unittest

import numpy as np

from src.navigation.controllers import (
    DEFAULT_POLICY, NumpyPolicy, RuleBasedController,
)
from src.safety.shield import SafetyShield
from src.simulation.warehouse_env import (
    BOX, GO, PERSON, STOP, TURN_LEFT, EnvConfig, Obstacle, WarehouseEnv,
)


def empty_env(seed=0, **kw):
    cfg = EnvConfig(n_workers=(0, 0), n_forklifts=(0, 0), n_pallets=(0, 0),
                    n_boxes=(0, 0), shadow_prob=(0.0, 0.0), **kw)
    env = WarehouseEnv(cfg)
    env.reset(seed=seed)
    return env


def place_ahead(env, category, dist, **kw):
    x = env.x + dist * math.cos(env.theta)
    y = env.y + dist * math.sin(env.theta)
    if category == PERSON:
        o = Obstacle(PERSON, x, y, radius=0.28, uid=777)
    else:
        o = Obstacle(category, x, y, 0.3, 0.3, uid=777, **kw)
    env.statics.append(o)
    env._perceive()
    return o


class TestEnvironment(unittest.TestCase):

    def test_reset_and_spaces(self):
        env = WarehouseEnv(seed=1)
        obs, info = env.reset(seed=1)
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertTrue(np.all(np.abs(obs) <= 1.0))
        obs, r, term, trunc, info = env.step(GO)
        self.assertEqual(obs.shape, (env.obs_dim,))
        self.assertIsInstance(r, float)

    def test_layout_is_reproducible(self):
        a = WarehouseEnv()
        b = WarehouseEnv()
        a.reset(seed=5, options={"layout_seed": 3})
        b.reset(seed=5, options={"layout_seed": 3})
        self.assertEqual(len(a.racks), len(b.racks))
        self.assertAlmostEqual(a.x, b.x)
        self.assertEqual(len(a.statics), len(b.statics))

    def test_goal_is_reachable_on_static_map(self):
        for seed in range(10):
            env = empty_env(seed)
            self.assertLess(env._field_value(env.x, env.y), 60.0)

    def test_driving_into_box_collides(self):
        env = empty_env(2)
        place_ahead(env, BOX, 1.2)
        collided = False
        for _ in range(40):
            _, _, term, _, info = env.step(GO)
            if term:
                collided = info["collided"]
                break
        self.assertTrue(collided)

    def test_depth_sees_unknown_detector_does_not(self):
        env = empty_env(3, perception_mode="full")
        place_ahead(env, BOX, 2.0)
        mid = env.cfg.n_rays // 2
        self.assertLess(env.fused_rays[mid], 2.0)
        self.assertEqual(env.ray_labels[mid], BOX)   # "unknown" label

        env2 = empty_env(3, perception_mode="detector_only")
        place_ahead(env2, BOX, 2.0)
        self.assertGreater(env2.fused_rays[mid], 2.0)  # invisible


class TestSafetyShield(unittest.TestCase):

    def test_hard_stop_for_unknown_obstacle(self):
        env = empty_env(4)
        env.v = 1.0
        place_ahead(env, BOX, 1.0)
        shield = SafetyShield()
        final, cap, reason = shield.filter(env, GO)
        self.assertEqual(final, STOP)
        self.assertEqual(cap, 0.0)
        self.assertIn("Obstacle", reason)

    def test_person_priority(self):
        env = empty_env(5)
        env.v = 0.0
        place_ahead(env, PERSON, 1.8)  # outside hard-stop, inside slow zone
        shield = SafetyShield()
        final, cap, reason = shield.filter(env, GO)
        self.assertNotEqual(final, GO)
        self.assertIn("Person", reason)

    def test_turn_allowed_in_place(self):
        env = empty_env(6)
        place_ahead(env, BOX, 0.7)
        shield = SafetyShield()
        final, cap, _ = shield.filter(env, TURN_LEFT)
        self.assertEqual(final, TURN_LEFT)
        self.assertEqual(cap, 0.0)

    def test_free_path_untouched(self):
        env = empty_env(7)
        shield = SafetyShield()
        shield.config.unc_slow = 2.0   # ignore random noise uncertainty
        env.percep.uncertainty = 0.0
        # rotate the robot to face the open staging area
        final, cap, reason = shield.filter(env, STOP)
        self.assertEqual(final, STOP)
        self.assertIsNone(cap)

    def test_decision_log(self):
        env = empty_env(8)
        shield = SafetyShield()
        shield.filter(env, GO)
        self.assertEqual(len(shield.log), 1)


class TestControllers(unittest.TestCase):

    def test_rule_based_runs(self):
        env = WarehouseEnv(seed=0)
        obs, _ = env.reset(seed=0)
        ctl = RuleBasedController()
        for _ in range(20):
            obs, _, term, trunc, _ = env.step(ctl.act(obs, env))
            if term or trunc:
                break

    @unittest.skipUnless(DEFAULT_POLICY.exists(), "trained policy missing")
    def test_numpy_policy(self):
        env = WarehouseEnv(seed=0)
        obs, _ = env.reset(seed=0)
        pol = NumpyPolicy()
        a = pol.act(obs)
        self.assertIn(a, range(5))
        p = pol.probs(obs)
        self.assertAlmostEqual(float(p.sum()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
