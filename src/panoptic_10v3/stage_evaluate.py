"""Evaluation-only metrics for M1 detections and M2 associations.

This module projects official 3D joints only after the RGB predictions have
been frozen.  The resulting identity assignments are never consumed by M2 or
M3 inference.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .constants import ALL_CAMERAS, COCO17_NAMES
from .geometry import project_points
from .io import load_cameras, load_gt_coco17, read_jsonl, write_json, write_jsonl
from .m1 import load_m1_by_frame


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    return float(numerator / denominator) if denominator else None


def _empty_m1_accumulator() -> Dict[str, Any]:
    return {
        "gt_people": 0,
        "predicted_people": 0,
        "matched_people": 0,
        "eligible_gt_joints": 0,
        "localized_gt_joints": 0,
        "errors_px": [],
        "normalized_errors": [],
        "joint_errors_px": [[] for _ in range(17)],
    }


def _summarize_m1(value: Mapping[str, Any]) -> Dict[str, Any]:
    errors = np.asarray(value["errors_px"], dtype=float)
    normalized = np.asarray(value["normalized_errors"], dtype=float)
    return {
        "gt_people": int(value["gt_people"]),
        "predicted_people": int(value["predicted_people"]),
        "matched_people": int(value["matched_people"]),
        "eligible_gt_joints": int(value["eligible_gt_joints"]),
        "localized_gt_joints": int(value["localized_gt_joints"]),
        "person_precision": _safe_div(
            value["matched_people"], value["predicted_people"]
        ),
        "person_recall": _safe_div(value["matched_people"], value["gt_people"]),
        "joint_availability": _safe_div(
            value["localized_gt_joints"], value["eligible_gt_joints"]
        ),
        "mean_joint_error_px": float(np.mean(errors)) if errors.size else None,
        "median_joint_error_px": float(np.median(errors)) if errors.size else None,
        "pck_bbox_005": (
            float(np.mean(normalized <= 0.05)) if normalized.size else None
        ),
        "pck_bbox_010": (
            float(np.mean(normalized <= 0.10)) if normalized.size else None
        ),
        "pck_bbox_020": (
            float(np.mean(normalized <= 0.20)) if normalized.size else None
        ),
    }


def evaluate_m1(
    sequence_dir: Path,
    frame_table_path: Path,
    m1_path: Path,
    output_dir: Path,
    cameras: Sequence[str] = ALL_CAMERAS,
    gt_confidence_threshold: float = 0.1,
    keypoint_score_threshold: float = 0.05,
    person_gate_px: float = 250.0,
) -> Dict[str, Any]:
    """Evaluate frozen monocular detections against calibrated GT projections."""

    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = load_cameras(sequence_dir)
    m1 = load_m1_by_frame(m1_path)
    overall = _empty_m1_accumulator()
    per_camera = {camera: _empty_m1_accumulator() for camera in cameras}
    assignment_rows: List[Dict[str, Any]] = []

    for frame in read_jsonl(frame_table_path):
        hd_index = int(frame["hd_index"])
        gt_people = load_gt_coco17(Path(frame["gt_path"]))
        for camera_name in cameras:
            camera = calibration[camera_name]
            detections = m1.get(hd_index, {}).get(camera_name, [])
            projected_people: List[Dict[str, Any]] = []
            for person in gt_people:
                pixels, depths = project_points(camera, person["joints_cm"])
                confidence = np.asarray(person["confidence"], dtype=float)
                valid = (
                    (confidence > gt_confidence_threshold)
                    & (depths > 0)
                    & (pixels[:, 0] >= 0)
                    & (pixels[:, 0] < camera.width)
                    & (pixels[:, 1] >= 0)
                    & (pixels[:, 1] < camera.height)
                )
                if np.count_nonzero(valid) < 4:
                    continue
                extent = np.ptp(pixels[valid], axis=0)
                bbox_scale = max(float(np.max(extent)), 1.0)
                projected_people.append(
                    {
                        "id": int(person["id"]),
                        "pixels": pixels,
                        "valid": valid,
                        "bbox_scale": bbox_scale,
                    }
                )

            cost = np.full(
                (len(detections), len(projected_people)),
                person_gate_px * 10.0,
                dtype=float,
            )
            for detection_index, detection in enumerate(detections):
                predicted_valid = (
                    np.asarray(detection.keypoints[:, 2], dtype=float)
                    >= keypoint_score_threshold
                )
                for gt_index, person in enumerate(projected_people):
                    valid = predicted_valid & person["valid"]
                    if np.count_nonzero(valid) < 4:
                        continue
                    errors = np.linalg.norm(
                        detection.keypoints[valid, :2] - person["pixels"][valid],
                        axis=1,
                    )
                    cost[detection_index, gt_index] = float(np.mean(errors))

            matches: List[Tuple[int, int, float]] = []
            if detections and projected_people:
                rows, columns = linear_sum_assignment(cost)
                matches = [
                    (int(row), int(column), float(cost[row, column]))
                    for row, column in zip(rows, columns)
                    if cost[row, column] <= person_gate_px
                ]
            matched_by_detection = {
                detection_index: (gt_index, match_error)
                for detection_index, gt_index, match_error in matches
            }
            camera_assignments = []
            for detection_index, detection in enumerate(detections):
                match = matched_by_detection.get(detection_index)
                camera_assignments.append(
                    {
                        "detection_id": int(detection.detection_id),
                        "gt_id": (
                            None
                            if match is None
                            else int(projected_people[match[0]]["id"])
                        ),
                        "matching_mean_error_px": (
                            None if match is None else float(match[1])
                        ),
                    }
                )

            accumulators = (overall, per_camera[camera_name])
            for accumulator in accumulators:
                accumulator["gt_people"] += len(projected_people)
                accumulator["predicted_people"] += len(detections)
                accumulator["matched_people"] += len(matches)
                accumulator["eligible_gt_joints"] += sum(
                    int(np.count_nonzero(person["valid"]))
                    for person in projected_people
                )
            for detection_index, gt_index, _ in matches:
                detection = detections[detection_index]
                person = projected_people[gt_index]
                valid = (
                    (detection.keypoints[:, 2] >= keypoint_score_threshold)
                    & person["valid"]
                )
                selected = np.flatnonzero(valid)
                errors = np.linalg.norm(
                    detection.keypoints[selected, :2] - person["pixels"][selected],
                    axis=1,
                )
                normalized = errors / person["bbox_scale"]
                for accumulator in accumulators:
                    accumulator["localized_gt_joints"] += len(selected)
                    accumulator["errors_px"].extend(errors.tolist())
                    accumulator["normalized_errors"].extend(normalized.tolist())
                    for joint_index, error in zip(selected, errors):
                        accumulator["joint_errors_px"][int(joint_index)].append(
                            float(error)
                        )

            assignment_rows.append(
                {
                    "hd_index": hd_index,
                    "camera": camera_name,
                    "eligible_gt_ids": [
                        int(person["id"]) for person in projected_people
                    ],
                    "detections": camera_assignments,
                }
            )

    summary = {
        "depth_used": False,
        "gt_assignment_used_by_inference": False,
        "gt_confidence_threshold": gt_confidence_threshold,
        "keypoint_score_threshold": keypoint_score_threshold,
        "person_gate_px": person_gate_px,
        **_summarize_m1(overall),
        "joints": [
            {
                "joint": COCO17_NAMES[index],
                "count": len(values),
                "mean_error_px": (
                    float(np.mean(values)) if len(values) else None
                ),
                "median_error_px": (
                    float(np.median(values)) if len(values) else None
                ),
            }
            for index, values in enumerate(overall["joint_errors_px"])
        ],
        "per_camera": {
            camera: _summarize_m1(per_camera[camera]) for camera in cameras
        },
    }
    write_jsonl(output_dir / "assignments.jsonl", assignment_rows)
    write_json(output_dir / "summary.json", summary)
    return summary


def evaluate_m2_from_assignments(
    reconstruction_path: Path,
    assignments_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """Score predicted M2 clusters using evaluation-only M1-to-GT matches."""

    assignments: Dict[int, Dict[Tuple[str, int], Optional[int]]] = defaultdict(dict)
    for row in read_jsonl(assignments_path):
        hd_index = int(row["hd_index"])
        camera = str(row["camera"])
        for detection in row["detections"]:
            assignments[hd_index][
                (camera, int(detection["detection_id"]))
            ] = detection["gt_id"]

    totals = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "predicted_pairs": 0,
        "true_pairs": 0,
        "clusters": 0,
        "wrong_person_merge_clusters": 0,
        "gt_person_instances": 0,
        "split_person_instances": 0,
        "matched_detections": 0,
        "unclustered_matched_detections": 0,
    }
    purity_numerator = 0
    purity_denominator = 0
    completeness_values: List[float] = []
    frame_rows = []
    condition = None

    for row in read_jsonl(reconstruction_path):
        hd_index = int(row["hd_index"])
        condition = row.get("condition", condition)
        selected_cameras = set(row.get("cameras", []))
        labels = {
            key: value
            for key, value in assignments.get(hd_index, {}).items()
            if key[0] in selected_cameras
        }
        clusters = [
            [tuple(member) for member in person.get("members", [])]
            for person in row.get("people", [])
        ]
        predicted_pairs = {
            tuple(sorted(pair))
            for cluster in clusters
            for pair in itertools.combinations(cluster, 2)
        }
        by_gt: Dict[int, set] = defaultdict(set)
        for key, gt_id in labels.items():
            if gt_id is not None:
                by_gt[int(gt_id)].add(key)
        true_pairs = {
            tuple(sorted(pair))
            for keys in by_gt.values()
            for pair in itertools.combinations(sorted(keys), 2)
        }
        tp = len(predicted_pairs & true_pairs)
        fp = len(predicted_pairs - true_pairs)
        fn = len(true_pairs - predicted_pairs)

        clustered = {key for cluster in clusters for key in cluster}
        matched_keys = {key for keys in by_gt.values() for key in keys}
        wrong_merges = 0
        for cluster_index, cluster in enumerate(clusters):
            cluster_labels = [
                labels.get(key)
                for key in cluster
                if labels.get(key) is not None
            ]
            counts: Dict[int, int] = defaultdict(int)
            for gt_id in cluster_labels:
                counts[int(gt_id)] += 1
            if len(counts) > 1:
                wrong_merges += 1
            if cluster:
                purity_numerator += max(counts.values(), default=0)
                purity_denominator += len(cluster)

        split_people = 0
        for keys in by_gt.values():
            hit_counts = [len(set(cluster) & keys) for cluster in clusters]
            nonempty = sum(count > 0 for count in hit_counts)
            split_people += int(nonempty > 1)
            completeness_values.append(
                max(hit_counts, default=0) / len(keys) if keys else 0.0
            )

        frame_metrics = {
            "hd_index": hd_index,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "clusters": len(clusters),
            "wrong_person_merge_clusters": wrong_merges,
            "gt_person_instances": len(by_gt),
            "split_person_instances": split_people,
            "matched_detections": len(matched_keys),
            "unclustered_matched_detections": len(matched_keys - clustered),
        }
        frame_rows.append(frame_metrics)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["predicted_pairs"] += len(predicted_pairs)
        totals["true_pairs"] += len(true_pairs)
        for key in (
            "clusters",
            "wrong_person_merge_clusters",
            "gt_person_instances",
            "split_person_instances",
            "matched_detections",
            "unclustered_matched_detections",
        ):
            totals[key] += frame_metrics[key]

    precision = _safe_div(totals["tp"], totals["tp"] + totals["fp"])
    recall = _safe_div(totals["tp"], totals["tp"] + totals["fn"])
    summary = {
        "condition": condition,
        "gt_assignment_used_by_inference": False,
        "pairwise_precision": precision,
        "pairwise_recall": recall,
        "pairwise_f1": (
            2 * precision * recall / (precision + recall)
            if precision is not None
            and recall is not None
            and precision + recall
            else None
        ),
        "cluster_purity": _safe_div(purity_numerator, purity_denominator),
        "cluster_completeness": (
            float(np.mean(completeness_values)) if completeness_values else None
        ),
        "wrong_person_merge_rate": _safe_div(
            totals["wrong_person_merge_clusters"], totals["clusters"]
        ),
        "split_person_rate": _safe_div(
            totals["split_person_instances"], totals["gt_person_instances"]
        ),
        "unclustered_matched_detection_rate": _safe_div(
            totals["unclustered_matched_detections"], totals["matched_detections"]
        ),
        "counts": totals,
        "frames": frame_rows,
    }
    write_json(output_path, summary)
    return summary
