from __future__ import annotations

import unittest

import numpy as np

from panoptic_10v3.geometry import project_points, triangulate_robust
from panoptic_10v3.model import Camera


def look_at_camera(name: str, center: np.ndarray, target: np.ndarray) -> Camera:
    forward = target - center
    forward = forward / np.linalg.norm(forward)
    down_hint = np.array([0.0, 1.0, 0.0])
    right = np.cross(down_hint, forward)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack((right, down, forward))
    translation = -rotation @ center
    return Camera(
        name=name,
        node=int(name),
        width=1920,
        height=1080,
        K=np.array([[1100.0, 0.0, 960.0], [0.0, 1100.0, 540.0], [0.0, 0.0, 1.0]]),
        dist=np.array([0.04, -0.03, 0.001, -0.0005, 0.005]),
        R=rotation,
        t=translation,
    )


class GeometryTest(unittest.TestCase):
    def test_distortion_aware_robust_triangulation(self) -> None:
        target = np.array([0.0, 0.0, 0.0])
        cameras = [
            look_at_camera("1", np.array([250.0, -40.0, 0.0]), target),
            look_at_camera("2", np.array([-130.0, -30.0, 220.0]), target),
            look_at_camera("3", np.array([-150.0, -20.0, -210.0]), target),
            look_at_camera("4", np.array([20.0, -280.0, 30.0]), target),
        ]
        world = np.array([21.0, -65.0, 14.0])
        rng = np.random.default_rng(7)
        observations = []
        for camera in cameras:
            pixel, depth = project_points(camera, world.reshape(1, 3))
            self.assertGreater(depth[0], 0)
            observations.append((camera, pixel[0] + rng.normal(0, 0.4, 2), 0.95))
        observations[-1] = (
            observations[-1][0],
            observations[-1][1] + np.array([90.0, -70.0]),
            observations[-1][2],
        )
        result = triangulate_robust(observations, reprojection_threshold_px=5.0)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.inlier_indices), 3)
        self.assertLess(np.linalg.norm(result.point_cm - world), 0.5)


if __name__ == "__main__":
    unittest.main()
