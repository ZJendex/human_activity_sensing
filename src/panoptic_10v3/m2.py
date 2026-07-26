"""M2: same-person association across calibrated RGB views."""

from __future__ import annotations

from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .constants import TORSO_JOINTS
from .geometry import symmetric_epipolar_distance_px
from .model import Camera, Detection


Cluster = List[Detection]


def detection_pair_cost(
    first: Detection,
    second: Detection,
    cameras: Mapping[str, Camera],
    score_threshold: float = 0.10,
) -> float:
    if first.camera == second.camera:
        return float("inf")
    score_a = first.keypoints[:, 2]
    score_b = second.keypoints[:, 2]
    valid = (score_a >= score_threshold) & (score_b >= score_threshold)
    torso_valid = valid[np.asarray(TORSO_JOINTS)]
    selected = np.flatnonzero(valid)
    if np.count_nonzero(torso_valid) >= 2:
        selected = np.asarray(TORSO_JOINTS, dtype=int)[torso_valid]
    if selected.size < 2:
        return float("inf")
    distances = symmetric_epipolar_distance_px(
        cameras[first.camera],
        first.keypoints[selected, :2],
        cameras[second.camera],
        second.keypoints[selected, :2],
    )
    weights = np.sqrt(score_a[selected] * score_b[selected])
    order = np.argsort(distances)
    distances = distances[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    median_index = int(np.searchsorted(cumulative, cumulative[-1] * 0.5))
    return float(distances[min(median_index, len(distances) - 1)])


def detection_cluster_cost(
    detection: Detection,
    cluster: Cluster,
    cameras: Mapping[str, Camera],
) -> float:
    costs = [
        detection_pair_cost(detection, member, cameras)
        for member in cluster
        if member.camera != detection.camera
    ]
    finite = [value for value in costs if np.isfinite(value)]
    if not finite:
        return float("inf")
    return float(np.median(finite))


def associate_across_views(
    detections_by_camera: Mapping[str, Sequence[Detection]],
    cameras: Mapping[str, Camera],
    epipolar_threshold_px: float = 25.0,
    min_views: int = 2,
) -> List[Cluster]:
    """Greedy camera-by-camera assignment with global Hungarian steps."""

    order = sorted(
        detections_by_camera,
        key=lambda name: (-len(detections_by_camera[name]), name),
    )
    clusters: List[Cluster] = []
    for camera_name in order:
        detections = list(detections_by_camera[camera_name])
        if not detections:
            continue
        if not clusters:
            clusters = [[detection] for detection in detections]
            continue
        cost = np.full((len(clusters), len(detections)), epipolar_threshold_px * 10.0)
        for cluster_index, cluster in enumerate(clusters):
            for detection_index, detection in enumerate(detections):
                value = detection_cluster_cost(detection, cluster, cameras)
                if np.isfinite(value):
                    cost[cluster_index, detection_index] = value
        rows, columns = linear_sum_assignment(cost)
        assigned = set()
        for row, column in zip(rows, columns):
            if cost[row, column] <= epipolar_threshold_px:
                clusters[row].append(detections[column])
                assigned.add(column)
        for detection_index, detection in enumerate(detections):
            if detection_index not in assigned:
                clusters.append([detection])
    clusters = [cluster for cluster in clusters if len({item.camera for item in cluster}) >= min_views]
    clusters.sort(key=lambda cluster: (-len(cluster), min(item.camera for item in cluster)))
    return clusters


def association_pair_counts(clusters: Sequence[Cluster]) -> Optional[Tuple[int, int, int]]:
    """Pairwise TP/FP/FN when control detections carry source_person_id."""

    all_detections = [item for cluster in clusters for item in cluster]
    if not any(item.source_person_id is not None for item in all_detections):
        return None
    predicted_pairs = set()
    for cluster in clusters:
        for i, first in enumerate(cluster):
            for second in cluster[i + 1 :]:
                key_a = (first.camera, first.detection_id)
                key_b = (second.camera, second.detection_id)
                predicted_pairs.add(tuple(sorted((key_a, key_b))))
    true_pairs = set()
    for i, first in enumerate(all_detections):
        if first.source_person_id is None:
            continue
        for second in all_detections[i + 1 :]:
            if (
                second.source_person_id == first.source_person_id
                and second.camera != first.camera
            ):
                key_a = (first.camera, first.detection_id)
                key_b = (second.camera, second.detection_id)
                true_pairs.add(tuple(sorted((key_a, key_b))))
    tp = len(predicted_pairs & true_pairs)
    fp = len(predicted_pairs - true_pairs)
    fn = len(true_pairs - predicted_pairs)
    return tp, fp, fn
