from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from panoptic_10v3.constants import ALL_CAMERAS
from panoptic_10v3.geometry import project_points, triangulate_robust
from panoptic_10v3.io import (
    build_frame_table,
    load_cameras,
    load_gt_coco17,
    read_jsonl,
)


SEQUENCE = Path("data/cmu_panoptic/160906_band1")


@unittest.skipUnless(SEQUENCE.exists(), "CMU Panoptic pilot data is not downloaded")
class RealPanopticTest(unittest.TestCase):
    def test_frame_table_uses_common_universal_time(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "frames.jsonl"
            summary = build_frame_table(
                SEQUENCE,
                path,
                stride=10,
                max_frames=3,
                require_all_cameras=True,
            )
            self.assertEqual(summary["selected_frames"], 3)
            for frame in read_jsonl(path):
                self.assertEqual(set(frame["available_cameras"]), set(ALL_CAMERAS))
                self.assertTrue(
                    all(abs(frame["cameras"][name]["delta_ms"]) <= 30 for name in ALL_CAMERAS)
                )

    def test_real_calibration_round_trip(self) -> None:
        cameras = load_cameras(SEQUENCE)
        gt_path = (
            SEQUENCE
            / "hdPose3d_stage1_coco19"
            / "body3DScene_00000220.json"
        )
        person = load_gt_coco17(gt_path)[0]
        joint_index = 5
        world = person["joints_cm"][joint_index]
        observations = []
        for name in ALL_CAMERAS:
            pixel, depth = project_points(cameras[name], world.reshape(1, 3))
            self.assertGreater(depth[0], 0)
            observations.append((cameras[name], pixel[0], 1.0))
        result = triangulate_robust(observations, reprojection_threshold_px=0.1)
        self.assertIsNotNone(result)
        self.assertLess(np.linalg.norm(result.point_cm - world), 1e-5)


if __name__ == "__main__":
    unittest.main()
