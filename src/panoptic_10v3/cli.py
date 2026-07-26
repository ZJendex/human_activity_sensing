"""Command-line entry point for the reproducible study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from .constants import ALL_CAMERAS, BALANCED_THREE
from .evaluate import (
    compare_conditions,
    evaluate_reconstruction,
    run_all_triplets,
)
from .io import (
    build_frame_table,
    file_sha256,
    load_gt_coco17,
    read_jsonl,
    write_json,
)
from .m1 import run_oracle_noise, run_transformers_vitpose, run_vitpose
from .reconstruct import reconstruct_views
from .stage_evaluate import evaluate_m1, evaluate_m2_from_assignments


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _print_result(result: Dict[str, Any]) -> None:
    print(json.dumps(result, indent=2, sort_keys=True))


def inspect_reused_m1(
    frame_table_path: Path,
    m1_path: Path,
    cameras: Sequence[str],
    requested_backend: str,
) -> Dict[str, Any]:
    """Verify a frozen M1 cache before reusing it in a paired comparison."""

    expected = {
        (
            int(frame["hd_index"]),
            camera,
            int(frame["cameras"][camera]["source_index"]),
        )
        for frame in read_jsonl(frame_table_path)
        for camera in cameras
        if frame["cameras"].get(camera, {}).get("valid")
    }
    actual = set()
    backends = set()
    detection_count = 0
    record_count = 0
    for row in read_jsonl(m1_path):
        record_count += 1
        actual.add(
            (
                int(row["hd_index"]),
                str(row["camera"]),
                int(row["source_index"]),
            )
        )
        backends.add(str(row["backend"]))
        detection_count += len(row["detections"])
    aliases = {
        "oracle-noise": {"oracle-noise"},
        "vitpose": {"mmpose-vitpose-b"},
        "hf-vitpose": {"transformers-vitpose-base-simple+rtdetr-r50"},
    }
    if backends != aliases[requested_backend]:
        raise ValueError(
            f"Frozen M1 backend mismatch: expected {aliases[requested_backend]}, "
            f"found {sorted(backends)}"
        )
    if record_count != len(actual):
        raise ValueError("Frozen M1 cache contains duplicate camera/frame records")
    if actual != expected:
        missing = sorted(expected - actual)[:5]
        extra = sorted(actual - expected)[:5]
        raise ValueError(
            f"Frozen M1 cache does not match the frame table; "
            f"missing examples={missing}, extra examples={extra}"
        )
    result = {
        "backend": next(iter(backends)),
        "output": str(m1_path),
        "reused": True,
        "camera_frame_records": record_count,
        "detections": detection_count,
        "sha256": file_sha256(m1_path),
    }
    if result["backend"] == "transformers-vitpose-base-simple+rtdetr-r50":
        result.update(
            {
                "pose_model": "usyd-community/vitpose-base-simple",
                "pose_revision": "a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45",
                "detector_model": "PekingU/rtdetr_r50vd_coco_o365",
                "detector_revision": "457857cec8ac28ddede40ecee9eed2beca321af8",
            }
        )
    return result


def command_prepare(args: argparse.Namespace) -> None:
    _print_result(
        build_frame_table(
            args.sequence_dir,
            args.output,
            stride=args.stride,
            max_frames=args.max_frames,
            require_all_cameras=not args.allow_missing_cameras,
        )
    )


def command_m1(args: argparse.Namespace) -> None:
    cameras = tuple(
        item.strip() for item in args.cameras.split(",") if item.strip()
    )
    if args.resume_m1 and args.backend != "hf-vitpose":
        raise ValueError("--resume-m1 is only supported by --backend hf-vitpose")
    if args.backend == "oracle-noise":
        result = run_oracle_noise(
            args.sequence_dir,
            args.frame_table,
            args.output,
            cameras=cameras,
            noise_px=args.noise_px,
            miss_probability=args.miss_probability,
            joint_dropout_probability=args.joint_dropout_probability,
            seed=args.seed,
        )
    elif args.backend == "vitpose":
        result = run_vitpose(
            args.sequence_dir,
            args.frame_table,
            args.output,
            cameras=cameras,
            device=args.device,
        )
    else:
        result = run_transformers_vitpose(
            args.sequence_dir,
            args.frame_table,
            args.output,
            cameras=cameras,
            device=args.device,
            detector_threshold=args.detector_threshold,
            resume=args.resume_m1,
        )
    _print_result(result)


def command_reconstruct(args: argparse.Namespace) -> None:
    camera_names = tuple(x.strip() for x in args.cameras.split(",") if x.strip())
    _print_result(
        reconstruct_views(
            args.sequence_dir,
            args.frame_table,
            args.m1,
            args.output,
            camera_names=camera_names,
            label=args.label,
            association_threshold_px=args.association_threshold_px,
            reprojection_threshold_px=args.reprojection_threshold_px,
            duplicate_distance_cm=args.duplicate_distance_cm,
        )
    )


def command_evaluate(args: argparse.Namespace) -> None:
    _print_result(
        evaluate_reconstruction(
            args.frame_table,
            args.reconstruction,
            args.output_dir,
            gt_confidence_threshold=args.gt_confidence_threshold,
        )
    )


def command_compare(args: argparse.Namespace) -> None:
    _print_result(compare_conditions(args.eval_a, args.eval_b, args.output))


def command_evaluate_m1(args: argparse.Namespace) -> None:
    cameras = tuple(item.strip() for item in args.cameras.split(",") if item.strip())
    _print_result(
        evaluate_m1(
            args.sequence_dir,
            args.frame_table,
            args.m1,
            args.output_dir,
            cameras=cameras,
            gt_confidence_threshold=args.gt_confidence_threshold,
            keypoint_score_threshold=args.keypoint_score_threshold,
            person_gate_px=args.person_gate_px,
        )
    )


def command_evaluate_m2(args: argparse.Namespace) -> None:
    _print_result(
        evaluate_m2_from_assignments(
            args.reconstruction,
            args.assignments,
            args.output,
        )
    )


def command_triplets(args: argparse.Namespace) -> None:
    _print_result(
        run_all_triplets(
            args.sequence_dir,
            args.frame_table,
            args.m1,
            args.output,
            max_frames=args.max_frames,
            association_threshold_px=args.association_threshold_px,
            reprojection_threshold_px=args.reprojection_threshold_px,
            duplicate_distance_cm=args.duplicate_distance_cm,
        )
    )


def command_visualize(args: argparse.Namespace) -> None:
    from .visualize import (
        build_viewer_data,
        render_calibration_audit,
        render_comparison_video,
        render_m1_m2_diagnostics,
        render_summary_figure,
        write_interactive_viewer,
    )

    data = build_viewer_data(
        args.sequence_dir,
        args.frame_table,
        args.v10,
        args.v3,
        max_frames=args.max_frames,
        colored_cloud_index_path=args.cloud_index,
        cloud_point_limit=args.cloud_point_limit,
        near_body_distance_cm=args.near_body_distance_cm,
    )
    write_interactive_viewer(data, args.output_dir / "interactive_comparison.html")
    render_summary_figure(
        args.eval_v10,
        args.eval_v3,
        args.output_dir / "summary.png",
    )
    render_calibration_audit(
        args.sequence_dir,
        args.frame_table,
        args.output_dir / "calibration_audit.png",
    )
    diagnostics = None
    if args.eval_m1 and args.eval_m2_v10 and args.eval_m2_v3:
        diagnostics = args.output_dir / "m1_m2_diagnostics.png"
        render_m1_m2_diagnostics(
            args.eval_m1,
            args.eval_m2_v10,
            args.eval_m2_v3,
            diagnostics,
        )
    if not args.skip_video:
        render_comparison_video(data, args.output_dir / "comparison.mp4", fps=args.video_fps)
    result = {
        "frames": len(data["frames"]),
        "colored_cloud_index": (
            None if args.cloud_index is None else str(args.cloud_index)
        ),
        "interactive_html": str(args.output_dir / "interactive_comparison.html"),
        "summary_figure": str(args.output_dir / "summary.png"),
        "calibration_audit": str(args.output_dir / "calibration_audit.png"),
        "m1_m2_diagnostics": None if diagnostics is None else str(diagnostics),
        "comparison_video": (
            None if args.skip_video else str(args.output_dir / "comparison.mp4")
        ),
    }
    manifest_path = args.output_dir.parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["final_visuals"] = result
        manifest.setdefault("sha256", {})
        manifest["sha256"]["interactive_comparison"] = file_sha256(
            args.output_dir / "interactive_comparison.html"
        )
        if not args.skip_video:
            manifest["sha256"]["comparison_video"] = file_sha256(
                args.output_dir / "comparison.mp4"
            )
        write_json(manifest_path, manifest)
    _print_result(result)


def command_depth_fuse(args: argparse.Namespace) -> None:
    from .depth_eval import (
        fuse_depth_at_hd_time,
        render_pointcloud_skeleton_overlay,
        write_ply,
    )

    points, metadata = fuse_depth_at_hd_time(
        args.sequence_dir,
        args.hd_index,
        sample_step=args.sample_step,
    )
    write_ply(args.output, points)
    write_json(args.output.with_suffix(".json"), metadata)
    gt_path = (
        args.sequence_dir
        / "hdPose3d_stage1_coco19"
        / f"body3DScene_{args.hd_index:08d}.json"
    )
    gt = load_gt_coco17(gt_path)
    overlay_path = args.output.with_suffix(".png")
    render_pointcloud_skeleton_overlay(
        overlay_path,
        points,
        [
            (
                int(person["id"]),
                person["joints_cm"],
                person["confidence"] > 0.1,
            )
            for person in gt
        ],
    )
    _print_result({"ply": str(args.output), "overlay": str(overlay_path), **metadata})


def command_depth_cache(args: argparse.Namespace) -> None:
    from .depth_eval import build_colored_cloud_cache

    result = build_colored_cloud_cache(
        args.sequence_dir,
        args.frame_table,
        args.output_dir,
        sample_step=args.sample_step,
        voxel_size_cm=args.voxel_size_cm,
        max_points=args.max_points,
        minimum_nodes=args.minimum_nodes,
        depth_filename=args.depth_filename,
    )
    manifest_path = args.output_dir.parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["evaluation_only_colored_cloud"] = result
        manifest.setdefault("sha256", {})
        manifest["sha256"]["colored_cloud_index"] = file_sha256(
            args.output_dir / "index.jsonl"
        )
        write_json(manifest_path, manifest)
    _print_result(result)


def command_run_study(args: argparse.Namespace) -> None:
    from .visualize import (
        build_viewer_data,
        render_calibration_audit,
        render_comparison_video,
        render_m1_m2_diagnostics,
        render_summary_figure,
        render_triplet_geometry_figure,
        write_interactive_viewer,
    )

    root = args.output_dir
    if args.resume_m1 and args.backend != "hf-vitpose":
        raise ValueError("--resume-m1 is only supported by --backend hf-vitpose")
    if args.resume_m1 and args.reuse_m1:
        raise ValueError("--resume-m1 and --reuse-m1 are mutually exclusive")
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "frames": root / "frame_table.jsonl",
        "m1": root / "m1_2d.jsonl",
        "v10": root / "m3_v10.jsonl",
        "v3": root / "m3_v3_balanced.jsonl",
        "eval10": root / "evaluation_v10",
        "eval3": root / "evaluation_v3",
        "eval_m1": root / "evaluation_m1",
        "eval_m2_v10": root / "evaluation_m2_v10.json",
        "eval_m2_v3": root / "evaluation_m2_v3.json",
        "comparison": root / "comparison.json",
        "triplets": root / "all_120_triplets.csv",
        "visuals": root / "visuals",
    }
    manifest: Dict[str, Any] = {
        "sequence": args.sequence_dir.name,
        "backend": args.backend,
        "depth_used_by_inference": False,
        "primary_v10": list(ALL_CAMERAS),
        "primary_v3": list(BALANCED_THREE),
        "configuration": {
            "association_threshold_px": args.association_threshold_px,
            "reprojection_threshold_px": args.reprojection_threshold_px,
            "duplicate_distance_cm": args.duplicate_distance_cm,
            "gt_confidence_threshold": args.gt_confidence_threshold,
            "gt_confidence_sensitivity": [
                float(value)
                for value in args.gt_confidence_sensitivity.split(",")
                if value.strip()
            ],
        },
        "artifacts": {key: str(value) for key, value in paths.items()},
    }
    manifest["prepare"] = build_frame_table(
        args.sequence_dir,
        paths["frames"],
        stride=args.stride,
        max_frames=args.max_frames,
        require_all_cameras=True,
    )
    if args.reuse_m1:
        if not paths["m1"].exists():
            raise FileNotFoundError(
                f"--reuse-m1 requested but cache does not exist: {paths['m1']}"
            )
        manifest["m1"] = inspect_reused_m1(
            paths["frames"],
            paths["m1"],
            ALL_CAMERAS,
            args.backend,
        )
    elif args.backend == "oracle-noise":
        manifest["m1"] = run_oracle_noise(
            args.sequence_dir,
            paths["frames"],
            paths["m1"],
            noise_px=args.noise_px,
            miss_probability=args.miss_probability,
            joint_dropout_probability=args.joint_dropout_probability,
            seed=args.seed,
        )
    elif args.backend == "vitpose":
        manifest["m1"] = run_vitpose(
            args.sequence_dir,
            paths["frames"],
            paths["m1"],
            device=args.device,
        )
    else:
        manifest["m1"] = run_transformers_vitpose(
            args.sequence_dir,
            paths["frames"],
            paths["m1"],
            device=args.device,
            detector_threshold=args.detector_threshold,
            resume=args.resume_m1,
        )
    manifest["v10"] = reconstruct_views(
        args.sequence_dir,
        paths["frames"],
        paths["m1"],
        paths["v10"],
        ALL_CAMERAS,
        "V10",
        association_threshold_px=args.association_threshold_px,
        reprojection_threshold_px=args.reprojection_threshold_px,
        duplicate_distance_cm=args.duplicate_distance_cm,
    )
    manifest["v3"] = reconstruct_views(
        args.sequence_dir,
        paths["frames"],
        paths["m1"],
        paths["v3"],
        BALANCED_THREE,
        "V3-balanced",
        association_threshold_px=args.association_threshold_px,
        reprojection_threshold_px=args.reprojection_threshold_px,
        duplicate_distance_cm=args.duplicate_distance_cm,
    )
    manifest["evaluation_m1"] = evaluate_m1(
        args.sequence_dir,
        paths["frames"],
        paths["m1"],
        paths["eval_m1"],
        cameras=ALL_CAMERAS,
        gt_confidence_threshold=args.gt_confidence_threshold,
    )
    manifest["evaluation_m2_v10"] = evaluate_m2_from_assignments(
        paths["v10"],
        paths["eval_m1"] / "assignments.jsonl",
        paths["eval_m2_v10"],
    )
    manifest["evaluation_m2_v3"] = evaluate_m2_from_assignments(
        paths["v3"],
        paths["eval_m1"] / "assignments.jsonl",
        paths["eval_m2_v3"],
    )
    manifest["evaluation_v10"] = evaluate_reconstruction(
        paths["frames"],
        paths["v10"],
        paths["eval10"],
        gt_confidence_threshold=args.gt_confidence_threshold,
    )
    manifest["evaluation_v3"] = evaluate_reconstruction(
        paths["frames"],
        paths["v3"],
        paths["eval3"],
        gt_confidence_threshold=args.gt_confidence_threshold,
    )
    manifest["gt_confidence_sensitivity"] = {}
    for threshold in manifest["configuration"]["gt_confidence_sensitivity"]:
        if abs(threshold - args.gt_confidence_threshold) < 1e-12:
            continue
        suffix = f"gt{int(round(threshold * 100)):02d}"
        eval10_sensitivity = root / f"evaluation_v10_{suffix}"
        eval3_sensitivity = root / f"evaluation_v3_{suffix}"
        manifest["gt_confidence_sensitivity"][str(threshold)] = {
            "v10": evaluate_reconstruction(
                paths["frames"],
                paths["v10"],
                eval10_sensitivity,
                gt_confidence_threshold=threshold,
            ),
            "v3": evaluate_reconstruction(
                paths["frames"],
                paths["v3"],
                eval3_sensitivity,
                gt_confidence_threshold=threshold,
            ),
        }
    manifest["comparison"] = compare_conditions(
        paths["eval3"], paths["eval10"], paths["comparison"]
    )
    if not args.skip_triplets:
        manifest["all_triplets"] = run_all_triplets(
            args.sequence_dir,
            paths["frames"],
            paths["m1"],
            paths["triplets"],
            max_frames=args.triplet_max_frames,
            association_threshold_px=args.association_threshold_px,
            reprojection_threshold_px=args.reprojection_threshold_px,
            duplicate_distance_cm=args.duplicate_distance_cm,
        )
        manifest["triplet_geometry_figure"] = render_triplet_geometry_figure(
            paths["triplets"],
            paths["visuals"] / "triplet_geometry.png",
        )
    viewer_data = build_viewer_data(
        args.sequence_dir,
        paths["frames"],
        paths["v10"],
        paths["v3"],
        max_frames=args.viewer_max_frames,
    )
    write_interactive_viewer(
        viewer_data, paths["visuals"] / "interactive_comparison.html"
    )
    render_summary_figure(
        paths["eval10"], paths["eval3"], paths["visuals"] / "summary.png"
    )
    render_m1_m2_diagnostics(
        paths["eval_m1"] / "summary.json",
        paths["eval_m2_v10"],
        paths["eval_m2_v3"],
        paths["visuals"] / "m1_m2_diagnostics.png",
    )
    manifest["calibration_audit"] = render_calibration_audit(
        args.sequence_dir,
        paths["frames"],
        paths["visuals"] / "calibration_audit.png",
    )
    if not args.skip_video:
        render_comparison_video(
            viewer_data,
            paths["visuals"] / "comparison.mp4",
            fps=args.video_fps,
        )
    checksum_paths = {
        "frame_table": paths["frames"],
        "m1_2d": paths["m1"],
        "m3_v10": paths["v10"],
        "m3_v3": paths["v3"],
        "comparison": paths["comparison"],
    }
    if paths["triplets"].exists():
        checksum_paths["all_120_triplets"] = paths["triplets"]
    manifest["sha256"] = {
        name: file_sha256(path) for name, path in checksum_paths.items()
    }
    write_json(root / "manifest.json", manifest)
    _print_result(manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="panoptic-10v3",
        description="RGB-only CMU Panoptic ten-view versus three-view study",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Build synchronized common-frame table")
    prepare.add_argument("--sequence-dir", type=_path, required=True)
    prepare.add_argument("--output", type=_path, required=True)
    prepare.add_argument("--stride", type=int, default=1)
    prepare.add_argument("--max-frames", type=int)
    prepare.add_argument("--allow-missing-cameras", action="store_true")
    prepare.set_defaults(func=command_prepare)

    m1 = subparsers.add_parser("m1", help="Cache per-camera COCO-17 detections")
    m1.add_argument("--sequence-dir", type=_path, required=True)
    m1.add_argument("--frame-table", type=_path, required=True)
    m1.add_argument("--output", type=_path, required=True)
    m1.add_argument("--cameras", default=",".join(ALL_CAMERAS))
    m1.add_argument(
        "--backend",
        choices=("oracle-noise", "vitpose", "hf-vitpose"),
        default="oracle-noise",
    )
    m1.add_argument("--noise-px", type=float, default=5.0)
    m1.add_argument("--miss-probability", type=float, default=0.02)
    m1.add_argument("--joint-dropout-probability", type=float, default=0.03)
    m1.add_argument("--seed", type=int, default=20260724)
    m1.add_argument(
        "--device",
        choices=("auto", "full-mps", "mps", "hybrid", "cpu", "cuda"),
        default="auto",
    )
    m1.add_argument("--detector-threshold", type=float, default=0.30)
    m1.add_argument(
        "--resume-m1",
        action="store_true",
        help="Resume a validated *.partial.jsonl Transformers inference cache",
    )
    m1.set_defaults(func=command_m1)

    reconstruction = subparsers.add_parser("reconstruct", help="Run M2 and M3")
    reconstruction.add_argument("--sequence-dir", type=_path, required=True)
    reconstruction.add_argument("--frame-table", type=_path, required=True)
    reconstruction.add_argument("--m1", type=_path, required=True)
    reconstruction.add_argument("--output", type=_path, required=True)
    reconstruction.add_argument("--cameras", required=True)
    reconstruction.add_argument("--label", required=True)
    reconstruction.add_argument("--association-threshold-px", type=float, default=25.0)
    reconstruction.add_argument("--reprojection-threshold-px", type=float, default=12.0)
    reconstruction.add_argument("--duplicate-distance-cm", type=float, default=30.0)
    reconstruction.set_defaults(func=command_reconstruct)

    evaluation = subparsers.add_parser("evaluate", help="Evaluate one reconstruction")
    evaluation.add_argument("--frame-table", type=_path, required=True)
    evaluation.add_argument("--reconstruction", type=_path, required=True)
    evaluation.add_argument("--output-dir", type=_path, required=True)
    evaluation.add_argument("--gt-confidence-threshold", type=float, default=0.1)
    evaluation.set_defaults(func=command_evaluate)

    comparison = subparsers.add_parser("compare", help="Paired block-bootstrap comparison")
    comparison.add_argument("--eval-a", type=_path, required=True)
    comparison.add_argument("--eval-b", type=_path, required=True)
    comparison.add_argument("--output", type=_path, required=True)
    comparison.set_defaults(func=command_compare)

    m1_evaluation = subparsers.add_parser(
        "evaluate-m1",
        help="Evaluate frozen 2D poses against calibrated GT projections",
    )
    m1_evaluation.add_argument("--sequence-dir", type=_path, required=True)
    m1_evaluation.add_argument("--frame-table", type=_path, required=True)
    m1_evaluation.add_argument("--m1", type=_path, required=True)
    m1_evaluation.add_argument("--output-dir", type=_path, required=True)
    m1_evaluation.add_argument("--cameras", default=",".join(ALL_CAMERAS))
    m1_evaluation.add_argument("--gt-confidence-threshold", type=float, default=0.1)
    m1_evaluation.add_argument("--keypoint-score-threshold", type=float, default=0.05)
    m1_evaluation.add_argument("--person-gate-px", type=float, default=250.0)
    m1_evaluation.set_defaults(func=command_evaluate_m1)

    m2_evaluation = subparsers.add_parser(
        "evaluate-m2",
        help="Evaluate frozen M2 clusters with evaluation-only GT assignments",
    )
    m2_evaluation.add_argument("--reconstruction", type=_path, required=True)
    m2_evaluation.add_argument("--assignments", type=_path, required=True)
    m2_evaluation.add_argument("--output", type=_path, required=True)
    m2_evaluation.set_defaults(func=command_evaluate_m2)

    triplets = subparsers.add_parser("all-triplets", help="Evaluate all 120 camera triplets")
    triplets.add_argument("--sequence-dir", type=_path, required=True)
    triplets.add_argument("--frame-table", type=_path, required=True)
    triplets.add_argument("--m1", type=_path, required=True)
    triplets.add_argument("--output", type=_path, required=True)
    triplets.add_argument("--max-frames", type=int)
    triplets.add_argument("--association-threshold-px", type=float, default=25.0)
    triplets.add_argument("--reprojection-threshold-px", type=float, default=12.0)
    triplets.add_argument("--duplicate-distance-cm", type=float, default=30.0)
    triplets.set_defaults(func=command_triplets)

    visual = subparsers.add_parser("visualize", help="Build HTML, PNG, and MP4 comparisons")
    visual.add_argument("--sequence-dir", type=_path, required=True)
    visual.add_argument("--frame-table", type=_path, required=True)
    visual.add_argument("--v10", type=_path, required=True)
    visual.add_argument("--v3", type=_path, required=True)
    visual.add_argument("--eval-v10", type=_path, required=True)
    visual.add_argument("--eval-v3", type=_path, required=True)
    visual.add_argument("--eval-m1", type=_path)
    visual.add_argument("--eval-m2-v10", type=_path)
    visual.add_argument("--eval-m2-v3", type=_path)
    visual.add_argument("--output-dir", type=_path, required=True)
    visual.add_argument("--max-frames", type=int, default=300)
    visual.add_argument(
        "--cloud-index",
        type=_path,
        help="Evaluation-only colored point-cloud index.jsonl",
    )
    visual.add_argument("--cloud-point-limit", type=int, default=5_000)
    visual.add_argument("--near-body-distance-cm", type=float, default=35.0)
    visual.add_argument("--skip-video", action="store_true")
    visual.add_argument("--video-fps", type=float, default=10.0)
    visual.set_defaults(func=command_visualize)

    depth = subparsers.add_parser(
        "depth-fuse", help="Evaluation-only ten-depth point-cloud fusion"
    )
    depth.add_argument("--sequence-dir", type=_path, required=True)
    depth.add_argument("--hd-index", type=int, required=True)
    depth.add_argument("--sample-step", type=int, default=6)
    depth.add_argument("--output", type=_path, required=True)
    depth.set_defaults(func=command_depth_fuse)

    depth_cache = subparsers.add_parser(
        "depth-cache",
        help="Build evaluation-only synchronized colored RGB-D point clouds",
    )
    depth_cache.add_argument("--sequence-dir", type=_path, required=True)
    depth_cache.add_argument("--frame-table", type=_path, required=True)
    depth_cache.add_argument("--output-dir", type=_path, required=True)
    depth_cache.add_argument("--sample-step", type=int, default=4)
    depth_cache.add_argument("--voxel-size-cm", type=float, default=2.0)
    depth_cache.add_argument("--max-points", type=int, default=20_000)
    depth_cache.add_argument("--minimum-nodes", type=int, default=6)
    depth_cache.add_argument(
        "--depth-filename",
        default="depthdata.dat",
        help=(
            "Depth stream basename inside each KINECTNODE directory; use "
            "depthdata.window.dat for a frame-table-bounded prefix"
        ),
    )
    depth_cache.set_defaults(func=command_depth_cache)

    study = subparsers.add_parser("run-study", help="Run the reproducible pilot end to end")
    study.add_argument("--sequence-dir", type=_path, required=True)
    study.add_argument("--output-dir", type=_path, required=True)
    study.add_argument(
        "--backend",
        choices=("oracle-noise", "vitpose", "hf-vitpose"),
        default="oracle-noise",
    )
    study.add_argument("--stride", type=int, default=5)
    study.add_argument("--max-frames", type=int, default=120)
    study.add_argument("--noise-px", type=float, default=5.0)
    study.add_argument("--miss-probability", type=float, default=0.02)
    study.add_argument("--joint-dropout-probability", type=float, default=0.03)
    study.add_argument("--seed", type=int, default=20260724)
    study.add_argument(
        "--device",
        choices=("auto", "full-mps", "mps", "hybrid", "cpu", "cuda"),
        default="auto",
    )
    study.add_argument("--detector-threshold", type=float, default=0.30)
    study.add_argument(
        "--resume-m1",
        action="store_true",
        help="Resume interrupted hf-vitpose inference from m1_2d.partial.jsonl",
    )
    study.add_argument(
        "--reuse-m1",
        action="store_true",
        help="Reuse and validate output-dir/m1_2d.jsonl instead of running inference",
    )
    study.add_argument("--association-threshold-px", type=float, default=25.0)
    study.add_argument("--reprojection-threshold-px", type=float, default=12.0)
    study.add_argument("--duplicate-distance-cm", type=float, default=30.0)
    study.add_argument("--gt-confidence-threshold", type=float, default=0.1)
    study.add_argument(
        "--gt-confidence-sensitivity",
        default="0.2,0.5",
        help="Comma-separated evaluation-only GT confidence thresholds",
    )
    study.add_argument("--skip-triplets", action="store_true")
    study.add_argument("--triplet-max-frames", type=int, default=60)
    study.add_argument("--viewer-max-frames", type=int, default=120)
    study.add_argument("--skip-video", action="store_true")
    study.add_argument("--video-fps", type=float, default=12.0)
    study.set_defaults(func=command_run_study)
    return parser


def main(argv: Sequence[str] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
