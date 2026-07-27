from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from panoptic_10v3.m2 import detection_pair_cost
from panoptic_10v3.model import Detection


class M2FullPoseCostTest(unittest.TestCase):
    def test_pair_cost_uses_all_reliable_joints(self) -> None:
        keypoints = np.column_stack(
            (
                np.arange(17, dtype=float),
                np.zeros(17, dtype=float),
                np.ones(17, dtype=float),
            )
        )
        first = Detection("A", 0, keypoints, np.zeros(4, dtype=float))
        second = Detection("B", 0, keypoints, np.zeros(4, dtype=float))
        torso = {5, 6, 11, 12}

        def fake_epipolar_distance(_camera_a, points_a, _camera_b, _points_b):
            return np.asarray(
                [1.0 if int(point[0]) in torso else 40.0 for point in points_a],
                dtype=float,
            )

        with patch(
            "panoptic_10v3.m2.symmetric_epipolar_distance_px",
            side_effect=fake_epipolar_distance,
        ) as distance:
            cost = detection_pair_cost(first, second, {"A": object(), "B": object()})

        self.assertEqual(distance.call_args.args[1].shape, (17, 2))
        self.assertEqual(cost, 40.0)


if __name__ == "__main__":
    unittest.main()
