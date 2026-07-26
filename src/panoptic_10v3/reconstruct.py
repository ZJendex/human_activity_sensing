"""M3/M4: 3D reconstruction and lightweight persistent tracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import triangulate_robust
from .io import load_cameras, read_jsonl, write_jsonl
from .m1 import load_m1_by_frame
from .m2 import Cluster, associate_across_views, association_pair_counts
from .model import Camera, Reconstruction


def reconstruct_cluster(
    cluster: Cluster,
    cameras: Mapping[str, Camera],
    reprojection_threshold_px: float = 12.0,
    min_keypoint_score: float = 0.05,
    local_id: int = 0,
) -> Reconstruction:
    joints = np.full((17, 3), np.nan, dtype=float)
    valid = np.zeros(17, dtype=bool)
    support = np.zeros(17, dtype=int)
    rmse = np.full(17, np.nan, dtype=float)
    for joint_index in range(17):
        observations = [
            (
                cameras[detection.camera],
                detection.keypoints[joint_index, :2],
                float(detection.keypoints[joint_index, 2]),
            )
            for detection in cluster
        ]
        result = triangulate_robust(
            observations,
            reprojection_threshold_px=reprojection_threshold_px,
            min_score=min_keypoint_score,
        )
        if result is None:
            continue
        joints[joint_index] = result.point_cm
        valid[joint_index] = True
        support[joint_index] = len(result.inlier_indices)
        rmse[joint_index] = result.rmse_px
    return Reconstruction(
        local_id=local_id,
        joints_cm=joints,
        joint_valid=valid,
        joint_support=support,
        reprojection_rmse_px=rmse,
        members=[(item.camera, item.detection_id) for item in cluster],
        source_person_ids=[item.source_person_id for item in cluster],
    )


def pelvis(reconstruction: Reconstruction) -> Optional[np.ndarray]:
    hips = reconstruction.joints_cm[[11, 12]]
    valid = reconstruction.joint_valid[[11, 12]]
    if np.all(valid):
        return np.mean(hips, axis=0)
    chosen = hips[valid]
    return None if not len(chosen) else np.mean(chosen, axis=0)


def reconstruction_quality(reconstruction: Reconstruction) -> Tuple[int, int, int, float]:
    """Rank a reconstruction by independent multi-view evidence.

    ``joint_support`` counts the triangulation inlier views, so it is a more
    useful first signal than the number of valid joints alone.  The final term
    favors lower reprojection error when support is tied.
    """

    finite_rmse = reconstruction.reprojection_rmse_px[
        np.isfinite(reconstruction.reprojection_rmse_px)
    ]
    mean_rmse = float(np.mean(finite_rmse)) if len(finite_rmse) else float("inf")
    return (
        int(np.sum(reconstruction.joint_support)),
        int(np.count_nonzero(reconstruction.joint_valid)),
        len(reconstruction.members),
        -mean_rmse,
    )


def reconstruction_distance_cm(
    left: Reconstruction,
    right: Reconstruction,
) -> Optional[float]:
    """Return a torso-based distance suitable for duplicate suppression."""

    torso = np.asarray([5, 6, 11, 12], dtype=int)
    common = left.joint_valid[torso] & right.joint_valid[torso]
    if np.count_nonzero(common) >= 2:
        indices = torso[common]
        return float(
            np.mean(
                np.linalg.norm(
                    left.joints_cm[indices] - right.joints_cm[indices],
                    axis=1,
                )
            )
        )
    left_pelvis = pelvis(left)
    right_pelvis = pelvis(right)
    if left_pelvis is None or right_pelvis is None:
        return None
    return float(np.linalg.norm(left_pelvis - right_pelvis))


def suppress_duplicate_reconstructions(
    reconstructions: Sequence[Reconstruction],
    distance_threshold_cm: float = 30.0,
) -> List[Reconstruction]:
    """Spatial NMS for duplicate 3D people produced by split M2 clusters.

    Candidates are sorted by multi-view triangulation support. A lower-quality
    candidate is suppressed only when its reconstructed torso is within the
    configured world-space threshold of a retained candidate.
    """

    if distance_threshold_cm <= 0:
        return list(reconstructions)
    ordered = sorted(reconstructions, key=reconstruction_quality, reverse=True)
    kept: List[Reconstruction] = []
    for candidate in ordered:
        duplicate = False
        for accepted in kept:
            distance = reconstruction_distance_cm(candidate, accepted)
            if distance is not None and distance < distance_threshold_cm:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    for local_id, item in enumerate(kept):
        item.local_id = local_id
    return kept


@dataclass
class Track:
    track_id: int
    position_cm: np.ndarray
    velocity_cm_per_s: np.ndarray
    univ_time_ms: float
    missed: int = 0


class PelvisTracker:
    def __init__(self, gate_cm: float = 80.0, max_missed: int = 15):
        self.gate_cm = gate_cm
        self.max_missed = max_missed
        self.tracks: Dict[int, Track] = {}
        self.next_id = 0

    def update(
        self,
        reconstructions: Sequence[Reconstruction],
        univ_time_ms: float,
    ) -> None:
        observations = [(index, pelvis(item)) for index, item in enumerate(reconstructions)]
        observations = [(index, value) for index, value in observations if value is not None]
        track_ids = sorted(self.tracks)
        assigned_tracks = set()
        assigned_observations = set()
        if track_ids and observations:
            cost = np.zeros((len(track_ids), len(observations)), dtype=float)
            for row, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                dt = max(0.0, (univ_time_ms - track.univ_time_ms) / 1000.0)
                predicted = track.position_cm + track.velocity_cm_per_s * dt
                for column, (_, position) in enumerate(observations):
                    cost[row, column] = np.linalg.norm(predicted - position)
            rows, columns = linear_sum_assignment(cost)
            for row, column in zip(rows, columns):
                if cost[row, column] > self.gate_cm:
                    continue
                track_id = track_ids[row]
                reconstruction_index, position = observations[column]
                track = self.tracks[track_id]
                dt = max(1e-3, (univ_time_ms - track.univ_time_ms) / 1000.0)
                measured_velocity = (position - track.position_cm) / dt
                track.velocity_cm_per_s = 0.65 * track.velocity_cm_per_s + 0.35 * measured_velocity
                track.position_cm = position
                track.univ_time_ms = univ_time_ms
                track.missed = 0
                reconstructions[reconstruction_index].track_id = track_id
                assigned_tracks.add(track_id)
                assigned_observations.add(column)
        for track_id in list(self.tracks):
            if track_id not in assigned_tracks:
                self.tracks[track_id].missed += 1
                if self.tracks[track_id].missed > self.max_missed:
                    del self.tracks[track_id]
        for column, (reconstruction_index, position) in enumerate(observations):
            if column in assigned_observations:
                continue
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = Track(
                track_id=track_id,
                position_cm=position,
                velocity_cm_per_s=np.zeros(3, dtype=float),
                univ_time_ms=univ_time_ms,
            )
            reconstructions[reconstruction_index].track_id = track_id


def reconstruct_views(
    sequence_dir: Path,
    frame_table_path: Path,
    m1_path: Path,
    output_path: Path,
    camera_names: Sequence[str],
    label: str,
    association_threshold_px: float = 25.0,
    reprojection_threshold_px: float = 12.0,
    duplicate_distance_cm: float = 30.0,
) -> Dict[str, Any]:
    cameras = load_cameras(sequence_dir)
    m1 = load_m1_by_frame(m1_path)
    tracker = PelvisTracker()
    totals = np.zeros(3, dtype=int)
    labeled_frames = 0

    def rows() -> Iterator[Dict[str, Any]]:
        nonlocal labeled_frames
        for frame in read_jsonl(frame_table_path):
            hd_index = int(frame["hd_index"])
            by_camera = {
                name: m1.get(hd_index, {}).get(name, [])
                for name in camera_names
                if name in m1.get(hd_index, {})
            }
            clusters = associate_across_views(
                by_camera,
                cameras,
                epipolar_threshold_px=association_threshold_px,
            )
            reconstructions = [
                reconstruct_cluster(
                    cluster,
                    cameras,
                    reprojection_threshold_px=reprojection_threshold_px,
                    local_id=index,
                )
                for index, cluster in enumerate(clusters)
            ]
            reconstructions = [
                item for item in reconstructions if np.count_nonzero(item.joint_valid) >= 4
            ]
            reconstructions = suppress_duplicate_reconstructions(
                reconstructions,
                distance_threshold_cm=duplicate_distance_cm,
            )
            tracker.update(reconstructions, float(frame["univ_time_ms"]))
            pair_counts = association_pair_counts(clusters)
            if pair_counts is not None:
                totals[:] += np.asarray(pair_counts)
                labeled_frames += 1
            yield {
                "sequence": frame["sequence"],
                "condition": label,
                "cameras": list(camera_names),
                "hd_index": hd_index,
                "univ_time_ms": float(frame["univ_time_ms"]),
                "m2_pair_counts": (
                    {
                        "tp": pair_counts[0],
                        "fp": pair_counts[1],
                        "fn": pair_counts[2],
                    }
                    if pair_counts is not None
                    else None
                ),
                "people": [item.to_json() for item in reconstructions],
            }

    frame_count = write_jsonl(output_path, rows())
    tp, fp, fn = totals.tolist()
    precision = tp / (tp + fp) if labeled_frames and tp + fp else None
    recall = tp / (tp + fn) if labeled_frames and tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "condition": label,
        "cameras": list(camera_names),
        "output": str(output_path),
        "frames": frame_count,
        "duplicate_distance_cm": duplicate_distance_cm,
        "m2_pair_precision": precision,
        "m2_pair_recall": recall,
        "m2_pair_f1": f1,
    }
