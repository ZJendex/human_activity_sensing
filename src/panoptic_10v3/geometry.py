"""Projection, epipolar geometry, and robust multi-view triangulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

from .model import Camera


def project_points(camera: Camera, points_world_cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_world_cm, dtype=float).reshape(-1, 3)
    camera_xyz = (camera.R @ points.T + camera.t.reshape(3, 1)).T
    pixels, _ = cv2.projectPoints(
        points,
        cv2.Rodrigues(camera.R)[0],
        camera.t.reshape(3, 1),
        camera.K,
        camera.dist,
    )
    return pixels.reshape(-1, 2), camera_xyz[:, 2]


def undistort_normalized(camera: Camera, pixels: np.ndarray) -> np.ndarray:
    value = cv2.undistortPoints(
        np.asarray(pixels, dtype=float).reshape(-1, 1, 2),
        camera.K,
        camera.dist,
    )
    return value.reshape(-1, 2)


def skew(value: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(value, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def fundamental_matrix(camera_a: Camera, camera_b: Camera) -> np.ndarray:
    """Return F such that x_b.T @ F @ x_a = 0, for distorted-free pixels."""

    r_ba = camera_b.R @ camera_a.R.T
    t_ba = camera_b.t - r_ba @ camera_a.t
    essential = skew(t_ba) @ r_ba
    return np.linalg.inv(camera_b.K).T @ essential @ np.linalg.inv(camera_a.K)


def symmetric_epipolar_distance_px(
    camera_a: Camera,
    pixels_a: np.ndarray,
    camera_b: Camera,
    pixels_b: np.ndarray,
) -> np.ndarray:
    """Symmetric point-to-epipolar-line distance after undistortion."""

    a_norm = undistort_normalized(camera_a, pixels_a)
    b_norm = undistort_normalized(camera_b, pixels_b)
    a_und = np.column_stack(
        (
            a_norm[:, 0] * camera_a.K[0, 0] + camera_a.K[0, 2],
            a_norm[:, 1] * camera_a.K[1, 1] + camera_a.K[1, 2],
            np.ones(len(a_norm)),
        )
    )
    b_und = np.column_stack(
        (
            b_norm[:, 0] * camera_b.K[0, 0] + camera_b.K[0, 2],
            b_norm[:, 1] * camera_b.K[1, 1] + camera_b.K[1, 2],
            np.ones(len(b_norm)),
        )
    )
    f = fundamental_matrix(camera_a, camera_b)
    lines_b = (f @ a_und.T).T
    lines_a = (f.T @ b_und.T).T
    numer = np.abs(np.sum(b_und * lines_b, axis=1))
    dist_a = numer / np.maximum(np.linalg.norm(lines_a[:, :2], axis=1), 1e-9)
    dist_b = numer / np.maximum(np.linalg.norm(lines_b[:, :2], axis=1), 1e-9)
    return 0.5 * (dist_a + dist_b)


@dataclass
class TriangulationResult:
    point_cm: np.ndarray
    inlier_indices: np.ndarray
    reprojection_errors_px: np.ndarray
    rmse_px: float


def _weighted_dlt(
    observations: Sequence[Tuple[Camera, np.ndarray, float]],
    indices: Sequence[int],
) -> Optional[np.ndarray]:
    rows: List[np.ndarray] = []
    for index in indices:
        camera, pixel, score = observations[index]
        xy = undistort_normalized(camera, np.asarray(pixel).reshape(1, 2))[0]
        projection = np.column_stack((camera.R, camera.t.reshape(3)))
        weight = np.sqrt(max(float(score), 1e-4))
        rows.append(weight * (xy[0] * projection[2] - projection[0]))
        rows.append(weight * (xy[1] * projection[2] - projection[1]))
    if len(rows) < 4:
        return None
    _, _, vt = np.linalg.svd(np.asarray(rows), full_matrices=False)
    homogeneous = vt[-1]
    if abs(homogeneous[3]) < 1e-10:
        return None
    return homogeneous[:3] / homogeneous[3]


def triangulate_robust(
    observations: Sequence[Tuple[Camera, np.ndarray, float]],
    reprojection_threshold_px: float = 12.0,
    min_score: float = 0.05,
) -> Optional[TriangulationResult]:
    usable = [
        i
        for i, (_, pixel, score) in enumerate(observations)
        if float(score) >= min_score and np.all(np.isfinite(pixel))
    ]
    if len(usable) < 2:
        return None

    active = list(usable)
    point = _weighted_dlt(observations, active)
    if point is None:
        return None
    while len(active) > 2:
        errors = _reprojection_errors(point, observations, active)
        worst_at = int(np.argmax(errors))
        if float(errors[worst_at]) <= reprojection_threshold_px:
            break
        del active[worst_at]
        point = _weighted_dlt(observations, active)
        if point is None:
            return None

    def residual(world: np.ndarray) -> np.ndarray:
        values: List[float] = []
        for index in active:
            camera, pixel, score = observations[index]
            projected, depth = project_points(camera, world.reshape(1, 3))
            if depth[0] <= 0:
                values.extend((1000.0, 1000.0))
            else:
                weight = np.sqrt(max(float(score), 1e-4))
                values.extend(((projected[0] - pixel) * weight).tolist())
        return np.asarray(values)

    optimized = least_squares(
        residual,
        point,
        loss="huber",
        f_scale=max(1.0, reprojection_threshold_px / 2.0),
        max_nfev=60,
    ).x
    all_errors = _reprojection_errors(optimized, observations, usable)
    inlier_mask = all_errors <= reprojection_threshold_px
    inliers = np.asarray(usable, dtype=int)[inlier_mask]
    if inliers.size < 2:
        return None
    depths = [project_points(observations[i][0], optimized.reshape(1, 3))[1][0] for i in inliers]
    if any(depth <= 0 for depth in depths):
        return None
    inlier_errors = all_errors[inlier_mask]
    return TriangulationResult(
        point_cm=optimized,
        inlier_indices=inliers,
        reprojection_errors_px=all_errors,
        rmse_px=float(np.sqrt(np.mean(inlier_errors**2))),
    )


def _reprojection_errors(
    point_cm: np.ndarray,
    observations: Sequence[Tuple[Camera, np.ndarray, float]],
    indices: Sequence[int],
) -> np.ndarray:
    errors: List[float] = []
    for index in indices:
        camera, pixel, _ = observations[index]
        projected, depth = project_points(camera, np.asarray(point_cm).reshape(1, 3))
        error = np.linalg.norm(projected[0] - np.asarray(pixel))
        errors.append(float(error if depth[0] > 0 else 1e6))
    return np.asarray(errors, dtype=float)
