import unittest
import numpy as np

from src.perception.obstacle_fusion import (
    fuse_detections_with_depth,
    create_driving_corridor_mask,
    create_uncovered_candidate_mask,
)


class TestObstacleFusion(unittest.TestCase):

    def setUp(self):
        self.depth_map = np.full((100, 100), 3.0, dtype=np.float32)

        self.detections = [
            {
                "class_name": "pallet",
                "confidence": 0.90,
                "bbox": (20, 20, 50, 50),
            }
        ]

    def test_detection_depth_fusion(self):
        fused = fuse_detections_with_depth(
            self.detections,
            self.depth_map,
        )

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["class_name"], "pallet")
        self.assertAlmostEqual(fused[0]["relative_depth"], 3.0)
        self.assertGreater(fused[0]["valid_depth_pixels"], 0)

    def test_nan_depth_handling(self):
        depth = self.depth_map.copy()
        depth[25:30, 25:30] = np.nan

        fused = fuse_detections_with_depth(
            self.detections,
            depth,
        )

        self.assertEqual(len(fused), 1)
        self.assertTrue(np.isfinite(fused[0]["relative_depth"]))

    def test_corridor_mask_shape(self):
        mask = create_driving_corridor_mask(
            image_shape=self.depth_map.shape
        )

        self.assertEqual(mask.shape, self.depth_map.shape)
        self.assertGreater(np.count_nonzero(mask), 0)

    def test_candidate_mask_without_detections(self):
        corridor = create_driving_corridor_mask(
            image_shape=self.depth_map.shape
        )

        candidates = create_uncovered_candidate_mask(
            corridor_mask=corridor,
            detections=[],
        )

        np.testing.assert_array_equal(candidates, corridor)

    def test_detection_reduces_candidate_region(self):
        corridor = create_driving_corridor_mask(
            image_shape=self.depth_map.shape
        )

        candidates = create_uncovered_candidate_mask(
            corridor_mask=corridor,
            detections=self.detections,
        )

        self.assertEqual(candidates.shape, corridor.shape)
        self.assertLessEqual(
            np.count_nonzero(candidates),
            np.count_nonzero(corridor),
        )


if __name__ == "__main__":
    unittest.main()
