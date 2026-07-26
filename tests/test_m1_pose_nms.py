from __future__ import annotations

import unittest

import numpy as np

from panoptic_10v3.m1 import suppress_duplicate_poses
from panoptic_10v3.model import Detection


class PoseNmsTest(unittest.TestCase):
    def test_nested_boxes_with_same_pose_are_suppressed(self) -> None:
        base = np.column_stack(
            (
                np.linspace(100, 180, 17),
                np.linspace(80, 300, 17),
                np.full(17, 0.9),
            )
        )
        duplicate = base.copy()
        duplicate[:, :2] += 1.5
        other = base.copy()
        other[:, 0] += 300
        detections = [
            Detection("50_01", 0, base, np.array([80, 50, 210, 340]), instance_score=0.9),
            Detection(
                "50_01",
                1,
                duplicate,
                np.array([100, 70, 190, 250]),
                instance_score=0.5,
            ),
            Detection(
                "50_01",
                2,
                other,
                np.array([380, 50, 510, 340]),
                instance_score=0.8,
            ),
        ]
        kept = suppress_duplicate_poses(detections)
        self.assertEqual(len(kept), 2)
        self.assertEqual([item.instance_score for item in kept], [0.9, 0.8])


if __name__ == "__main__":
    unittest.main()
