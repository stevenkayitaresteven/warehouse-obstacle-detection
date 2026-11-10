import unittest

import numpy as np

from src.perception.obstacle_fusion import create_driving_corridor_mask
from src.perception.live_pipeline import unknown_obstacles


class TestUnknownObstacles(unittest.TestCase):

    def setUp(self):
        h, w = 120, 160
        # Relative inverse depth of a flat floor: closer (larger) at the
        # bottom of the image, farther toward the horizon.
        rows = np.linspace(0.1, 1.0, h)[:, None]
        self.floor = np.repeat(rows, w, axis=1).astype(np.float32)
        self.corridor = create_driving_corridor_mask((h, w))

    def test_flat_floor_has_no_unknown_obstacle(self):
        found, _ = unknown_obstacles(self.floor, self.corridor, [])
        self.assertEqual(found, [])

    def test_box_on_floor_is_found(self):
        depth = self.floor.copy()
        depth[70:95, 70:95] = 0.95          # upright object, close
        found, _ = unknown_obstacles(depth, self.corridor, [])
        self.assertEqual(len(found), 1)
        x1, y1, x2, y2 = found[0]["bbox"]
        self.assertTrue(65 <= x1 <= 75 and 65 <= y1 <= 75)

    def test_detected_box_is_not_unknown(self):
        depth = self.floor.copy()
        depth[70:95, 70:95] = 0.95
        dets = [{"class_name": "pallet", "confidence": 0.9,
                 "bbox": (68, 68, 97, 97)}]
        found, _ = unknown_obstacles(depth, self.corridor, dets)
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
