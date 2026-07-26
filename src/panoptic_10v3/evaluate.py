"""Evaluation for V10/V3 reconstruction and paired camera-count comparisons."""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .constants import ALL_CAMERAS, BALANCED_THREE, COCO17_NAMES
from .io import load_cameras, load_gt_coco17, read_jsonl, write_json, write_jsonl
from .m1 import load_m1_by_frame
from .m2 import associate_across_views
from .model import Reconstruction
from .reconstruct import reconstruct_cluster, suppress_duplicate_reconstructions


def _load_reconstructions(path: Path) -> Dict[int, Dict[str, Any]]:
    return {int(row["hd_index"]): row for row in read_jsonl(path)}


def match_people(
    predicted: Sequence[Reconstruction],
    gt_people: Sequence[Mapping[str, Any]],
    gt_confidence_threshold: float = 0.1,
    person_gate_mm: float = 1000.0,
) -> List[Tuple[int, int, float]]:
    if not predicted or not gt_people:
        return []
    cost = np.full((len(predicted), len(gt_people)), person_gate_mm * 10.0, dtype=float)
    for pred_index, pred in enumerate(predicted):
        for gt_index, gt in enumerate(gt_people):
            gt_conf = np.asarray(gt["confidence"], dtype=float)
            valid = pred.joint_valid & (gt_conf > gt_confidence_threshold)
            if np.count_nonzero(valid) < 4:
                continue
            errors_mm = (
                np.linalg.norm(
                    pred.joints_cm[valid] - np.asarray(gt["joints_cm"])[valid],
                    axis=1,
                )
                * 10.0
            )
            cost[pred_index, gt_index] = float(np.mean(errors_mm))
    rows, columns = linear_sum_assignment(cost)
    return [
        (int(row), int(column), float(cost[row, column]))
        for row, column in zip(rows, columns)
        if cost[row, column] <= person_gate_mm
    ]


def evaluate_frame(
    predicted: Sequence[Reconstruction],
    gt_people: Sequence[Mapping[str, Any]],
    gt_confidence_threshold: float = 0.1,
    person_gate_mm: float = 1000.0,
) -> Dict[str, Any]:
    matches = match_people(
        predicted,
        gt_people,
        gt_confidence_threshold=gt_confidence_threshold,
        person_gate_mm=person_gate_mm,
    )
    errors_by_joint: List[List[float]] = [[] for _ in range(17)]
    matched_details: List[Dict[str, Any]] = []
    eligible_joint_count = sum(
        int(np.count_nonzero(np.asarray(person["confidence"]) > gt_confidence_threshold))
        for person in gt_people
    )
    reconstructed_eligible = 0
    for pred_index, gt_index, match_cost in matches:
        pred = predicted[pred_index]
        gt = gt_people[gt_index]
        gt_conf = np.asarray(gt["confidence"], dtype=float)
        valid = pred.joint_valid & (gt_conf > gt_confidence_threshold)
        errors = np.linalg.norm(
            pred.joints_cm[valid] - np.asarray(gt["joints_cm"])[valid],
            axis=1,
        ) * 10.0
        reconstructed_eligible += int(np.count_nonzero(valid))
        selected_indices = np.flatnonzero(valid)
        for joint_index, error in zip(selected_indices, errors):
            errors_by_joint[int(joint_index)].append(float(error))
        hip_indices = np.asarray([11, 12], dtype=int)
        pelvis_valid = np.all(pred.joint_valid[hip_indices]) and np.all(
            gt_conf[hip_indices] > gt_confidence_threshold
        )
        pelvis_error_mm = None
        if pelvis_valid:
            pred_pelvis = np.mean(pred.joints_cm[hip_indices], axis=0)
            gt_pelvis = np.mean(np.asarray(gt["joints_cm"])[hip_indices], axis=0)
            pelvis_error_mm = float(np.linalg.norm(pred_pelvis - gt_pelvis) * 10.0)
        matched_details.append(
            {
                "pred_index": pred_index,
                "gt_index": gt_index,
                "gt_id": int(gt["id"]),
                "track_id": pred.track_id,
                "matching_mpjpe_mm": match_cost,
                "valid_joint_count": int(np.count_nonzero(valid)),
                "pelvis_error_mm": pelvis_error_mm,
            }
        )
    all_errors = np.asarray(
        [error for joint_errors in errors_by_joint for error in joint_errors],
        dtype=float,
    )
    return {
        "gt_people": len(gt_people),
        "predicted_people": len(predicted),
        "matched_people": len(matches),
        "eligible_gt_joints": eligible_joint_count,
        "reconstructed_gt_joints": reconstructed_eligible,
        "mpjpe_mm": float(np.mean(all_errors)) if all_errors.size else None,
        "pck_50": float(np.mean(all_errors <= 50.0)) if all_errors.size else None,
        "pck_100": float(np.mean(all_errors <= 100.0)) if all_errors.size else None,
        "joint_errors_mm": errors_by_joint,
        "matches": matched_details,
    }


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    return float(numerator / denominator) if denominator else None


def evaluate_reconstruction(
    frame_table_path: Path,
    reconstruction_path: Path,
    output_dir: Path,
    gt_confidence_threshold: float = 0.1,
    person_gate_mm: float = 1000.0,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    recon_by_frame = _load_reconstructions(reconstruction_path)
    joint_errors: List[List[float]] = [[] for _ in range(17)]
    totals = {
        "gt_people": 0,
        "predicted_people": 0,
        "matched_people": 0,
        "eligible_gt_joints": 0,
        "reconstructed_gt_joints": 0,
        "m2_tp": 0,
        "m2_fp": 0,
        "m2_fn": 0,
        "m2_labeled_frames": 0,
    }
    frame_rows: List[Dict[str, Any]] = []
    pelvis_errors: List[float] = []
    gt_track_history: Dict[int, List[Optional[int]]] = {}
    condition = None
    cameras: List[str] = []
    for frame in read_jsonl(frame_table_path):
        hd_index = int(frame["hd_index"])
        recon_row = recon_by_frame.get(hd_index, {})
        if condition is None and recon_row:
            condition = recon_row.get("condition")
            cameras = list(recon_row.get("cameras", []))
        predicted = [
            Reconstruction.from_json(item) for item in recon_row.get("people", [])
        ]
        gt_people = load_gt_coco17(Path(frame["gt_path"]))
        metrics = evaluate_frame(
            predicted,
            gt_people,
            gt_confidence_threshold=gt_confidence_threshold,
            person_gate_mm=person_gate_mm,
        )
        counts = recon_row.get("m2_pair_counts") or {}
        metrics.update(
            {
                "hd_index": hd_index,
                "univ_time_ms": float(frame["univ_time_ms"]),
                "m2_tp": int(counts.get("tp", 0)),
                "m2_fp": int(counts.get("fp", 0)),
                "m2_fn": int(counts.get("fn", 0)),
                "m2_labeled": bool(counts),
            }
        )
        for key in (
            "gt_people",
            "predicted_people",
            "matched_people",
            "eligible_gt_joints",
            "reconstructed_gt_joints",
            "m2_tp",
            "m2_fp",
            "m2_fn",
        ):
            totals[key] += int(metrics[key])
        totals["m2_labeled_frames"] += int(metrics["m2_labeled"])
        for joint_index, errors in enumerate(metrics.pop("joint_errors_mm")):
            joint_errors[joint_index].extend(errors)
        for match in metrics["matches"]:
            if match["pelvis_error_mm"] is not None:
                pelvis_errors.append(float(match["pelvis_error_mm"]))
            gt_track_history.setdefault(int(match["gt_id"]), []).append(match["track_id"])
        frame_rows.append(metrics)

    all_errors = np.asarray([x for values in joint_errors for x in values], dtype=float)
    joint_summary = []
    for index, values in enumerate(joint_errors):
        array = np.asarray(values, dtype=float)
        joint_summary.append(
            {
                "joint": COCO17_NAMES[index],
                "count": len(values),
                "mpjpe_mm": float(np.mean(array)) if array.size else None,
                "median_mm": float(np.median(array)) if array.size else None,
                "pck_50": float(np.mean(array <= 50.0)) if array.size else None,
                "pck_100": float(np.mean(array <= 100.0)) if array.size else None,
            }
        )
    person_precision = _safe_div(totals["matched_people"], totals["predicted_people"])
    person_recall = _safe_div(totals["matched_people"], totals["gt_people"])
    m2_precision = (
        _safe_div(totals["m2_tp"], totals["m2_tp"] + totals["m2_fp"])
        if totals["m2_labeled_frames"]
        else None
    )
    m2_recall = (
        _safe_div(totals["m2_tp"], totals["m2_tp"] + totals["m2_fn"])
        if totals["m2_labeled_frames"]
        else None
    )
    id_switches = 0
    fragmentation = 0
    for history in gt_track_history.values():
        filtered = [track_id for track_id in history if track_id is not None]
        id_switches += sum(first != second for first, second in zip(filtered, filtered[1:]))
        fragmentation += max(0, len(set(filtered)) - 1)
    pelvis_array = np.asarray(pelvis_errors, dtype=float)
    summary = {
        "condition": condition,
        "cameras": cameras,
        "gt_confidence_threshold": gt_confidence_threshold,
        "person_gate_mm": person_gate_mm,
        "frames": len(frame_rows),
        "mpjpe_mm": float(np.mean(all_errors)) if all_errors.size else None,
        "median_joint_error_mm": float(np.median(all_errors)) if all_errors.size else None,
        "pck_50": float(np.mean(all_errors <= 50.0)) if all_errors.size else None,
        "pck_100": float(np.mean(all_errors <= 100.0)) if all_errors.size else None,
        "person_precision": person_precision,
        "person_recall": person_recall,
        "pelvis_trajectory_rmse_mm": (
            float(np.sqrt(np.mean(pelvis_array**2))) if pelvis_array.size else None
        ),
        "track_id_switches": int(id_switches),
        "track_fragmentation": int(fragmentation),
        "joint_availability": _safe_div(
            totals["reconstructed_gt_joints"], totals["eligible_gt_joints"]
        ),
        "m2_pair_precision": m2_precision,
        "m2_pair_recall": m2_recall,
        "m2_pair_f1": (
            2 * m2_precision * m2_recall / (m2_precision + m2_recall)
            if m2_precision is not None
            and m2_recall is not None
            and m2_precision + m2_recall
            else None
        ),
        "counts": totals,
        "joints": joint_summary,
    }
    write_jsonl(output_dir / "per_frame.jsonl", frame_rows)
    write_json(output_dir / "summary.json", summary)
    return summary


def paired_block_bootstrap(
    frames_a: Sequence[Mapping[str, Any]],
    frames_b: Sequence[Mapping[str, Any]],
    block_duration_ms: float = 2000.0,
    iterations: int = 2000,
    seed: int = 20260724,
) -> Dict[str, Any]:
    by_a = {int(row["hd_index"]): row for row in frames_a}
    by_b = {int(row["hd_index"]): row for row in frames_b}
    paired = []
    for hd_index in sorted(set(by_a) & set(by_b)):
        a = by_a[hd_index].get("mpjpe_mm")
        b = by_b[hd_index].get("mpjpe_mm")
        if a is None or b is None:
            continue
        paired.append((float(by_a[hd_index]["univ_time_ms"]), float(a) - float(b)))
    if not paired:
        return {
            "paired_frames": 0,
            "mean_delta_mm": None,
            "ci95_mm": [None, None],
            "definition": "A - B",
        }
    start = paired[0][0]
    blocks: Dict[int, List[float]] = {}
    for timestamp, difference in paired:
        block = int((timestamp - start) // block_duration_ms)
        blocks.setdefault(block, []).append(difference)
    arrays = [np.asarray(values, dtype=float) for _, values in sorted(blocks.items())]
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        selected = rng.integers(0, len(arrays), size=len(arrays))
        sample = np.concatenate([arrays[index] for index in selected])
        estimates[iteration] = np.mean(sample)
    values = np.asarray([difference for _, difference in paired])
    return {
        "paired_frames": len(paired),
        "blocks": len(arrays),
        "block_duration_ms": block_duration_ms,
        "iterations": iterations,
        "mean_delta_mm": float(np.mean(values)),
        "ci95_mm": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "definition": "A - B",
    }


def compare_conditions(
    evaluation_a_dir: Path,
    evaluation_b_dir: Path,
    output_path: Path,
) -> Dict[str, Any]:
    summary_a = json.loads((evaluation_a_dir / "summary.json").read_text())
    summary_b = json.loads((evaluation_b_dir / "summary.json").read_text())
    frames_a = list(read_jsonl(evaluation_a_dir / "per_frame.jsonl"))
    frames_b = list(read_jsonl(evaluation_b_dir / "per_frame.jsonl"))
    paired = paired_block_bootstrap(frames_a, frames_b)
    comparison = {
        "a": summary_a["condition"],
        "b": summary_b["condition"],
        "definition": "delta = A - B; positive MPJPE delta means A is worse",
        "mpjpe_delta_mm": (
            summary_a["mpjpe_mm"] - summary_b["mpjpe_mm"]
            if summary_a["mpjpe_mm"] is not None and summary_b["mpjpe_mm"] is not None
            else None
        ),
        "pck_50_delta": (
            summary_a["pck_50"] - summary_b["pck_50"]
            if summary_a["pck_50"] is not None and summary_b["pck_50"] is not None
            else None
        ),
        "joint_availability_delta": (
            summary_a["joint_availability"] - summary_b["joint_availability"]
            if summary_a["joint_availability"] is not None
            and summary_b["joint_availability"] is not None
            else None
        ),
        "paired_block_bootstrap_mpjpe": paired,
        "summary_a": summary_a,
        "summary_b": summary_b,
    }
    write_json(output_path, comparison)
    return comparison


def run_all_triplets(
    sequence_dir: Path,
    frame_table_path: Path,
    m1_path: Path,
    output_csv: Path,
    max_frames: Optional[int] = None,
    gt_confidence_threshold: float = 0.1,
    association_threshold_px: float = 25.0,
    reprojection_threshold_px: float = 12.0,
    duplicate_distance_cm: float = 30.0,
) -> Dict[str, Any]:
    cameras = load_cameras(sequence_dir)
    m1 = load_m1_by_frame(m1_path)
    frames = list(read_jsonl(frame_table_path))
    if max_frames is not None:
        frames = frames[:max_frames]
    rows: List[Dict[str, Any]] = []
    for camera_triplet in itertools.combinations(ALL_CAMERAS, 3):
        all_errors: List[float] = []
        gt_joint_count = 0
        reconstructed_joint_count = 0
        matched_people = 0
        gt_people_count = 0
        predicted_people = 0
        for frame in frames:
            hd_index = int(frame["hd_index"])
            by_camera = {
                name: m1.get(hd_index, {}).get(name, []) for name in camera_triplet
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
            gt = load_gt_coco17(Path(frame["gt_path"]))
            metrics = evaluate_frame(
                reconstructions,
                gt,
                gt_confidence_threshold=gt_confidence_threshold,
            )
            gt_people_count += metrics["gt_people"]
            predicted_people += metrics["predicted_people"]
            matched_people += metrics["matched_people"]
            gt_joint_count += metrics["eligible_gt_joints"]
            reconstructed_joint_count += metrics["reconstructed_gt_joints"]
            all_errors.extend(
                error
                for values in metrics["joint_errors_mm"]
                for error in values
            )
        errors = np.asarray(all_errors, dtype=float)
        centers = np.asarray([cameras[name].center_world_cm for name in camera_triplet])
        baseline_cm = max(
            np.linalg.norm(centers[i] - centers[j])
            for i in range(3)
            for j in range(i + 1, 3)
        )
        rows.append(
            {
                "cameras": "+".join(camera_triplet),
                "contains_balanced_primary": camera_triplet == tuple(sorted(BALANCED_THREE)),
                "frames": len(frames),
                "max_baseline_cm": float(baseline_cm),
                "mpjpe_mm": float(np.mean(errors)) if errors.size else "",
                "pck_50": float(np.mean(errors <= 50.0)) if errors.size else "",
                "pck_100": float(np.mean(errors <= 100.0)) if errors.size else "",
                "joint_availability": _safe_div(reconstructed_joint_count, gt_joint_count),
                "person_precision": _safe_div(matched_people, predicted_people),
                "person_recall": _safe_div(matched_people, gt_people_count),
            }
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    numeric = [float(row["mpjpe_mm"]) for row in rows if row["mpjpe_mm"] != ""]
    summary = {
        "output": str(output_csv),
        "triplets": len(rows),
        "frames_per_triplet": len(frames),
        "mpjpe_mm_median": float(np.median(numeric)) if numeric else None,
        "mpjpe_mm_best": float(np.min(numeric)) if numeric else None,
        "mpjpe_mm_worst": float(np.max(numeric)) if numeric else None,
    }
    write_json(output_csv.with_suffix(".summary.json"), summary)
    return summary
