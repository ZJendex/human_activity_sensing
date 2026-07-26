"""Evaluation-only Kinect depth fusion.

This module is intentionally not imported by M1, M2, or reconstruction. Depth
may be used to inspect body-surface consistency after RGB-only inference; it is
not an inference input and is not a joint-center ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .constants import COCO17_EDGES
from .io import (
    VideoReaderPool,
    load_cameras,
    load_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


DEPTH_WIDTH = 512
DEPTH_HEIGHT = 424
DEPTH_PIXELS = DEPTH_WIDTH * DEPTH_HEIGHT


def read_depth_frame(path: Path, sync_position: int) -> np.ndarray:
    """Read one zero-based sync position, matching the official MATLAB reader."""

    with path.open("rb") as handle:
        handle.seek(2 * DEPTH_PIXELS * int(sync_position))
        raw = np.fromfile(handle, dtype="<u2", count=DEPTH_PIXELS)
    if raw.size != DEPTH_PIXELS:
        raise EOFError(f"Depth frame {sync_position} is incomplete in {path}")
    return raw.reshape(DEPTH_HEIGHT, DEPTH_WIDTH)[:, ::-1]


def unproject_depth_local_m(
    depth_mm: np.ndarray,
    sensor: Mapping[str, Any],
    sample_step: int = 4,
) -> np.ndarray:
    yy, xx = np.mgrid[0:DEPTH_HEIGHT:sample_step, 0:DEPTH_WIDTH:sample_step]
    depth = depth_mm[::sample_step, ::sample_step].reshape(-1).astype(float)
    pixels = np.column_stack((xx.reshape(-1), yy.reshape(-1))).astype(float)
    valid = depth > 0
    depth = depth[valid]
    pixels = pixels[valid]
    normalized = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2),
        np.asarray(sensor["K_depth"], dtype=float),
        np.asarray(sensor["distCoeffs_depth"], dtype=float).reshape(-1)[:5],
    ).reshape(-1, 2)
    points = np.column_stack(
        (normalized[:, 0] * depth * 0.001, normalized[:, 1] * depth * 0.001, depth * 0.001)
    )
    m_depth = np.asarray(sensor["M_depth"], dtype=float)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (np.linalg.inv(m_depth) @ homogeneous.T).T[:, :3]


def local_depth_to_world_cm(
    points_local_m: np.ndarray,
    sensor: Mapping[str, Any],
    camera_world: Any,
) -> np.ndarray:
    world_to_color = np.eye(4, dtype=float)
    world_to_color[:3, :3] = camera_world.R
    world_to_color[:3, 3] = camera_world.t
    scale = np.eye(4, dtype=float)
    scale[:3, :3] *= 100.0
    color_to_local = np.asarray(sensor["M_color"], dtype=float)
    transform = np.linalg.inv(world_to_color) @ scale @ np.linalg.inv(color_to_local)
    homogeneous = np.column_stack((points_local_m, np.ones(len(points_local_m))))
    return (transform @ homogeneous.T).T[:, :3]


def sample_local_points_from_color(
    points_local_m: np.ndarray,
    color_bgr: np.ndarray,
    sensor: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Project local Kinect points into the synchronized RGB image.

    This follows the CMU toolbox calibration chain: ``M_color`` maps the
    Kinect local coordinates into the color camera before lens distortion and
    bilinear RGB sampling.
    """

    if not len(points_local_m):
        return points_local_m, np.empty((0, 3), dtype=np.uint8)
    m_color = np.asarray(sensor["M_color"], dtype=float)
    rotation_vector, _ = cv2.Rodrigues(m_color[:3, :3])
    pixels, _ = cv2.projectPoints(
        np.asarray(points_local_m, dtype=float),
        rotation_vector,
        m_color[:3, 3],
        np.asarray(sensor["K_color"], dtype=float),
        np.asarray(sensor["distCoeffs_color"], dtype=float).reshape(-1),
    )
    pixels = pixels.reshape(-1, 2)
    homogeneous = np.column_stack(
        (points_local_m, np.ones(len(points_local_m), dtype=float))
    )
    color_camera = (m_color @ homogeneous.T).T
    height, width = color_bgr.shape[:2]
    valid = (
        (color_camera[:, 2] > 0)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] <= width - 1)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] <= height - 1)
    )
    if not np.any(valid):
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.uint8)
    map_x = pixels[valid, 0].astype(np.float32).reshape(-1, 1)
    map_y = pixels[valid, 1].astype(np.float32).reshape(-1, 1)
    sampled_bgr = cv2.remap(
        color_bgr,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).reshape(-1, 3)
    return points_local_m[valid], sampled_bgr[:, ::-1].copy()


def voxel_downsample_colored(
    points_cm: np.ndarray,
    colors_rgb: np.ndarray,
    voxel_size_cm: float = 2.0,
    max_points: int = 20_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministically average XYZ and RGB within metric voxels."""

    points = np.asarray(points_cm, dtype=float)
    colors = np.asarray(colors_rgb, dtype=np.uint8)
    if not len(points):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    if voxel_size_cm <= 0:
        raise ValueError("voxel_size_cm must be positive")
    keys = np.floor(points / voxel_size_cm).astype(np.int32)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    point_sums = np.zeros((len(counts), 3), dtype=np.float64)
    color_sums = np.zeros((len(counts), 3), dtype=np.float64)
    np.add.at(point_sums, inverse, points)
    np.add.at(color_sums, inverse, colors)
    averaged_points = point_sums / counts[:, None]
    averaged_colors = np.rint(color_sums / counts[:, None]).astype(np.uint8)
    if max_points > 0 and len(averaged_points) > max_points:
        chosen = np.linspace(
            0,
            len(averaged_points) - 1,
            num=max_points,
            dtype=int,
        )
        averaged_points = averaged_points[chosen]
        averaged_colors = averaged_colors[chosen]
    return averaged_points.astype(np.float32), averaged_colors


class ColoredDepthFusion:
    """Reusable synchronized RGB-D fusion context for a complete sequence."""

    def __init__(
        self,
        sequence_dir: Path,
        sample_step: int = 4,
        depth_tolerance_ms: float = 17.0,
        color_depth_tolerance_ms: float = 6.5,
        remove_dome: bool = True,
        depth_filename: str = "depthdata.dat",
    ):
        self.sequence_dir = sequence_dir
        self.sequence = sequence_dir.name
        self.sample_step = sample_step
        self.depth_tolerance_ms = depth_tolerance_ms
        self.color_depth_tolerance_ms = color_depth_tolerance_ms
        self.remove_dome = remove_dome
        self.depth_filename = depth_filename
        sync = load_json(sequence_dir / f"synctables_{self.sequence}.json")
        self.hd_times = np.asarray(sync["hd"]["univ_time"], dtype=float)
        self.ksync = load_json(
            sequence_dir / f"ksynctables_{self.sequence}.json"
        )["kinect"]
        self.sensors = load_json(
            sequence_dir / f"kcalibration_{self.sequence}.json"
        )["sensors"]
        self.cameras = load_cameras(sequence_dir)
        self.videos = VideoReaderPool(sequence_dir)

    def close(self) -> None:
        self.videos.close()

    def __enter__(self) -> "ColoredDepthFusion":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def fuse(
        self,
        hd_index: int,
        voxel_size_cm: float = 2.0,
        max_points: int = 20_000,
        minimum_nodes: int = 6,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        target = float(self.hd_times[hd_index])
        clouds: List[np.ndarray] = []
        colors: List[np.ndarray] = []
        nodes: List[Dict[str, Any]] = []
        accepted_depth_times: List[float] = []
        for node in range(1, 11):
            camera_name = f"50_{node:02d}"
            key = f"KINECTNODE{node}"
            depth_sync = self.ksync["depth"][key]
            color_sync = self.ksync["color"][key]
            depth_times = np.asarray(depth_sync["univ_time"], dtype=float)
            color_times = np.asarray(color_sync["univ_time"], dtype=float)
            valid_depth = np.flatnonzero(depth_times > 0)
            valid_color = np.flatnonzero(color_times > 0)
            if valid_depth.size == 0 or valid_color.size == 0:
                continue
            depth_position = int(
                valid_depth[np.argmin(np.abs(depth_times[valid_depth] - target))]
            )
            color_position = int(
                valid_color[
                    np.argmin(np.abs((color_times[valid_color] - 6.25) - target))
                ]
            )
            depth_delta = float(depth_times[depth_position] - target)
            corrected_color_time = float(color_times[color_position] - 6.25)
            color_delta = corrected_color_time - target
            raw_color_depth_delta = float(
                depth_times[depth_position] - color_times[color_position]
            )
            accepted = (
                abs(depth_delta) <= self.depth_tolerance_ms
                and abs(color_delta) <= 30.0
                and abs(raw_color_depth_delta) <= self.color_depth_tolerance_ms
            )
            color_indices = np.asarray(color_sync["index"], dtype=int)
            color_source_index = int(color_indices[color_position])
            status = {
                "node": node,
                "camera": camera_name,
                "depth_sync_position": depth_position,
                "color_sync_position": color_position,
                "color_source_index": color_source_index,
                "depth_delta_ms": depth_delta,
                "color_delta_ms": color_delta,
                "raw_color_depth_delta_ms": raw_color_depth_delta,
                "accepted": accepted,
            }
            nodes.append(status)
            if not accepted:
                continue
            depth_path = (
                self.sequence_dir
                / "kinect_shared_depth"
                / key
                / self.depth_filename
            )
            depth = read_depth_frame(depth_path, depth_position)
            sensor = self.sensors[node - 1]
            local = unproject_depth_local_m(
                depth,
                sensor,
                sample_step=self.sample_step,
            )
            if self.remove_dome:
                center = np.asarray(sensor["domeCenter"], dtype=float).reshape(3)
                distance = np.linalg.norm(local - center, axis=1)
                local = local[(distance <= 2.5) & (local[:, 1] <= 2.3)]
            color_bgr = self.videos.read(camera_name, color_source_index)
            local, rgb = sample_local_points_from_color(local, color_bgr, sensor)
            world = local_depth_to_world_cm(
                local,
                sensor,
                self.cameras[camera_name],
            )
            clouds.append(world)
            colors.append(rgb)
            accepted_depth_times.append(float(depth_times[depth_position]))
        accepted_nodes = len(accepted_depth_times)
        warnings = [
            "Evaluation-only RGB-D surface reference; it is not consumed by "
            "M1/M2/triangulation and is not joint-center ground truth."
        ]
        temporal_span_ms = (
            float(max(accepted_depth_times) - min(accepted_depth_times))
            if accepted_depth_times
            else None
        )
        if accepted_nodes < minimum_nodes:
            warnings.append(
                f"Only {accepted_nodes} depth nodes passed synchronization; "
                f"minimum is {minimum_nodes}, so the cloud was suppressed."
            )
            fused = np.empty((0, 3), dtype=float)
            fused_rgb = np.empty((0, 3), dtype=np.uint8)
        else:
            fused = np.concatenate(clouds, axis=0)
            fused_rgb = np.concatenate(colors, axis=0)
        if temporal_span_ms is not None and temporal_span_ms > 30.0:
            warnings.append(
                f"Accepted depth-node temporal span is {temporal_span_ms:.2f} ms."
            )
        fused, fused_rgb = voxel_downsample_colored(
            fused,
            fused_rgb,
            voxel_size_cm=voxel_size_cm,
            max_points=max_points,
        )
        metadata = {
            "sequence": self.sequence,
            "hd_index": int(hd_index),
            "univ_time_ms": target,
            "sample_step": self.sample_step,
            "voxel_size_cm": voxel_size_cm,
            "maximum_points": max_points,
            "depth_filename": self.depth_filename,
            "accepted_nodes": accepted_nodes,
            "temporal_span_ms": temporal_span_ms,
            "point_count": int(len(fused)),
            "nodes": nodes,
            "warnings": warnings,
        }
        return fused, fused_rgb, metadata


def build_colored_cloud_cache(
    sequence_dir: Path,
    frame_table_path: Path,
    output_dir: Path,
    sample_step: int = 4,
    voxel_size_cm: float = 2.0,
    max_points: int = 20_000,
    minimum_nodes: int = 6,
    depth_filename: str = "depthdata.dat",
) -> Dict[str, Any]:
    """Build resumable per-frame colored point-cloud NPZ files and an index."""

    frames = list(read_jsonl(frame_table_path))
    cloud_dir = output_dir / "clouds"
    cloud_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    built, reused = 0, 0
    with ColoredDepthFusion(
        sequence_dir,
        sample_step=sample_step,
        depth_filename=depth_filename,
    ) as fusion:
        for index, frame in enumerate(frames):
            hd_index = int(frame["hd_index"])
            cloud_path = cloud_dir / f"hd_{hd_index:08d}.npz"
            metadata: Dict[str, Any]
            if cloud_path.exists():
                with np.load(cloud_path, allow_pickle=False) as cached:
                    metadata = json.loads(str(cached["metadata"].item()))
                compatible = (
                    metadata.get("sample_step") == sample_step
                    and metadata.get("voxel_size_cm") == voxel_size_cm
                    and metadata.get("maximum_points") == max_points
                    and metadata.get("depth_filename", "depthdata.dat")
                    == depth_filename
                )
            else:
                compatible = False
            if compatible:
                reused += 1
            else:
                points, rgb, metadata = fusion.fuse(
                    hd_index,
                    voxel_size_cm=voxel_size_cm,
                    max_points=max_points,
                    minimum_nodes=minimum_nodes,
                )
                np.savez_compressed(
                    cloud_path,
                    xyz_cm=points.astype(np.float32),
                    rgb=rgb.astype(np.uint8),
                    metadata=np.asarray(json.dumps(metadata, separators=(",", ":"))),
                )
                built += 1
            rows.append(
                {
                    "hd_index": hd_index,
                    "univ_time_ms": float(frame["univ_time_ms"]),
                    "cloud_path": str(cloud_path),
                    **metadata,
                }
            )
            if (index + 1) % 10 == 0:
                print(
                    f"RGB-D cache: {index + 1}/{len(frames)} frames "
                    f"(built={built}, reused={reused})",
                    flush=True,
                )
    index_path = output_dir / "index.jsonl"
    write_jsonl(index_path, rows)
    summary = {
        "sequence": sequence_dir.name,
        "frame_table": str(frame_table_path),
        "index": str(index_path),
        "frames": len(rows),
        "built": built,
        "reused": reused,
        "sample_step": sample_step,
        "voxel_size_cm": voxel_size_cm,
        "maximum_points_per_frame": max_points,
        "minimum_nodes": minimum_nodes,
        "depth_filename": depth_filename,
        "depth_used_by_inference": False,
        "surface_reference_only": True,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def fuse_depth_at_hd_time(
    sequence_dir: Path,
    hd_index: int,
    sample_step: int = 6,
    depth_tolerance_ms: float = 17.0,
    color_depth_tolerance_ms: float = 6.5,
    remove_dome: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    sequence = sequence_dir.name
    panoptic_sync = load_json(sequence_dir / f"synctables_{sequence}.json")
    ksync = load_json(sequence_dir / f"ksynctables_{sequence}.json")["kinect"]
    calibration = load_json(sequence_dir / f"kcalibration_{sequence}.json")
    cameras = load_cameras(sequence_dir)
    target = float(panoptic_sync["hd"]["univ_time"][hd_index])
    clouds: List[np.ndarray] = []
    nodes: List[Dict[str, Any]] = []
    for node in range(1, 11):
        camera_name = f"50_{node:02d}"
        key = f"KINECTNODE{node}"
        depth_times = np.asarray(ksync["depth"][key]["univ_time"], dtype=float)
        color_times = np.asarray(ksync["color"][key]["univ_time"], dtype=float)
        valid_depth = np.flatnonzero(depth_times > 0)
        valid_color = np.flatnonzero(color_times > 0)
        if valid_depth.size == 0 or valid_color.size == 0:
            continue
        depth_position = int(valid_depth[np.argmin(np.abs(depth_times[valid_depth] - target))])
        color_position = int(
            valid_color[np.argmin(np.abs((color_times[valid_color] - 6.25) - target))]
        )
        depth_delta = float(depth_times[depth_position] - target)
        color_delta = float((color_times[color_position] - 6.25) - target)
        raw_color_depth_delta = float(
            depth_times[depth_position] - color_times[color_position]
        )
        accepted = (
            abs(depth_delta) <= depth_tolerance_ms
            and abs(color_delta) <= 30.0
            and abs(raw_color_depth_delta) <= color_depth_tolerance_ms
        )
        node_status = {
            "node": node,
            "camera": camera_name,
            "depth_sync_position": depth_position,
            "color_sync_position": color_position,
            "depth_delta_ms": depth_delta,
            "color_delta_ms": color_delta,
            "raw_color_depth_delta_ms": raw_color_depth_delta,
            "accepted": accepted,
        }
        nodes.append(node_status)
        if not accepted:
            continue
        sensor = calibration["sensors"][node - 1]
        depth_path = (
            sequence_dir
            / "kinect_shared_depth"
            / key
            / "depthdata.dat"
        )
        depth = read_depth_frame(depth_path, depth_position)
        local = unproject_depth_local_m(depth, sensor, sample_step=sample_step)
        if remove_dome:
            center = np.asarray(sensor["domeCenter"], dtype=float).reshape(3)
            distance = np.linalg.norm(local - center, axis=1)
            local = local[(distance <= 2.5) & (local[:, 1] <= 2.3)]
        clouds.append(local_depth_to_world_cm(local, sensor, cameras[camera_name]))
    fused = np.concatenate(clouds, axis=0) if clouds else np.empty((0, 3), dtype=float)
    metadata = {
        "sequence": sequence,
        "hd_index": hd_index,
        "univ_time_ms": target,
        "sample_step": sample_step,
        "accepted_nodes": sum(int(item["accepted"]) for item in nodes),
        "point_count": int(len(fused)),
        "nodes": nodes,
        "warning": (
            "Evaluation-only surface reference. Do not use nearest surface-point "
            "distance as joint-center MPJPE."
        ),
    }
    return fused, metadata


def write_ply(
    path: Path,
    points_cm: np.ndarray,
    colors_rgb: Optional[np.ndarray] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points_cm, dtype=float)
    colors = None if colors_rgb is None else np.asarray(colors_rgb, dtype=np.uint8)
    if colors is not None and colors.shape != points.shape:
        raise ValueError("colors_rgb must have the same Nx3 shape as points_cm")
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            handle.write(
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            )
        handle.write("end_header\n")
        for index, point in enumerate(points):
            values = f"{point[0]:.5f} {point[1]:.5f} {point[2]:.5f}"
            if colors is not None:
                values += (
                    f" {int(colors[index, 0])} {int(colors[index, 1])}"
                    f" {int(colors[index, 2])}"
                )
            handle.write(values + "\n")


def render_pointcloud_skeleton_overlay(
    output_path: Path,
    points_cm: np.ndarray,
    skeletons: Sequence[Tuple[int, np.ndarray, np.ndarray]],
) -> None:
    """Create a static QA mosaic in three orthographic world projections."""

    width, height = 1800, 600
    image = np.full((height, width, 3), 248, dtype=np.uint8)
    projections = ((0, 2, "world X/Z - top"), (0, 1, "world X/Y"), (2, 1, "world Z/Y"))
    colors = ((235, 99, 37), (23, 139, 229), (45, 122, 101), (123, 68, 192))
    ranges = {
        0: (-300.0, 300.0),
        1: (-220.0, 80.0),
        2: (-300.0, 300.0),
    }
    panel_width = width // 3
    sampled = points_cm[:: max(1, len(points_cm) // 40000)]
    for panel, (axis_x, axis_y, label) in enumerate(projections):
        x0 = panel * panel_width
        if panel:
            cv2.line(image, (x0, 0), (x0, height), (220, 224, 230), 1)

        def project(values: np.ndarray) -> np.ndarray:
            xlo, xhi = ranges[axis_x]
            ylo, yhi = ranges[axis_y]
            x = x0 + 30 + (values[:, axis_x] - xlo) / (xhi - xlo) * (panel_width - 60)
            y = height - 30 - (values[:, axis_y] - ylo) / (yhi - ylo) * (height - 80)
            return np.column_stack((x, y)).astype(int)

        cloud_pixels = project(sampled)
        inside = (
            (cloud_pixels[:, 0] >= x0)
            & (cloud_pixels[:, 0] < x0 + panel_width)
            & (cloud_pixels[:, 1] >= 0)
            & (cloud_pixels[:, 1] < height)
        )
        for pixel in cloud_pixels[inside]:
            image[pixel[1], pixel[0]] = (172, 177, 186)
        for person_index, (person_id, joints, valid) in enumerate(skeletons):
            pixels = project(joints)
            color = colors[person_index % len(colors)]
            for first, second in COCO17_EDGES:
                if valid[first] and valid[second]:
                    cv2.line(
                        image,
                        tuple(pixels[first]),
                        tuple(pixels[second]),
                        color,
                        3,
                        cv2.LINE_AA,
                    )
            for joint_index in np.flatnonzero(valid):
                cv2.circle(image, tuple(pixels[joint_index]), 4, color, -1, cv2.LINE_AA)
            valid_pixels = pixels[valid]
            if len(valid_pixels):
                anchor = valid_pixels[0]
                cv2.putText(
                    image,
                    f"GT ID {person_id}",
                    (int(anchor[0] + 6), int(anchor[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        cv2.putText(
            image,
            label,
            (x0 + 18, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (24, 33, 47),
            2,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not write {output_path}")


def point_to_skeleton_segments_cm(
    points_cm: np.ndarray,
    joints_cm: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    distances = np.full(len(points_cm), np.inf, dtype=float)
    for first, second in COCO17_EDGES:
        if not valid[first] or not valid[second]:
            continue
        a, b = joints_cm[first], joints_cm[second]
        ab = b - a
        denominator = float(ab @ ab)
        if denominator < 1e-9:
            continue
        t = np.clip(((points_cm - a) @ ab) / denominator, 0.0, 1.0)
        closest = a + t[:, None] * ab
        distances = np.minimum(distances, np.linalg.norm(points_cm - closest, axis=1))
    return distances


def near_body_surface_mask(
    points_cm: np.ndarray,
    gt_people: Sequence[Mapping[str, Any]],
    maximum_distance_cm: float = 35.0,
    confidence_threshold: float = 0.1,
) -> np.ndarray:
    """Evaluation-only crop around any official GT skeleton limb segment."""

    points = np.asarray(points_cm, dtype=float)
    minimum = np.full(len(points), np.inf, dtype=float)
    for person in gt_people:
        joints = np.asarray(person["joints_cm"], dtype=float)
        valid = np.asarray(person["confidence"], dtype=float) > confidence_threshold
        minimum = np.minimum(
            minimum,
            point_to_skeleton_segments_cm(points, joints, valid),
        )
    return minimum <= maximum_distance_cm


def surface_consistency(
    points_cm: np.ndarray,
    skeletons: Sequence[Tuple[np.ndarray, np.ndarray]],
    crop_margin_cm: float = 35.0,
    coverage_radius_cm: float = 25.0,
) -> Dict[str, Any]:
    """Secondary capsule-style surface coverage, never a joint MPJPE."""

    results = []
    for index, (joints, valid) in enumerate(skeletons):
        valid_points = joints[valid]
        if len(valid_points) < 4:
            continue
        lower = np.min(valid_points, axis=0) - crop_margin_cm
        upper = np.max(valid_points, axis=0) + crop_margin_cm
        crop = points_cm[np.all((points_cm >= lower) & (points_cm <= upper), axis=1)]
        distances = point_to_skeleton_segments_cm(crop, joints, valid)
        finite = distances[np.isfinite(distances)]
        results.append(
            {
                "skeleton_index": index,
                "cropped_surface_points": int(len(crop)),
                "surface_coverage_within_25cm": (
                    float(np.mean(finite <= coverage_radius_cm)) if finite.size else None
                ),
                "median_surface_to_bone_axis_cm": (
                    float(np.median(finite)) if finite.size else None
                ),
            }
        )
    return {
        "definition": (
            "Body-surface coverage around skeleton bone axes; descriptive secondary "
            "check only, not joint-center ground truth."
        ),
        "people": results,
    }
