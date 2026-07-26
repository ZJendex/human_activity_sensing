import json

import numpy as np

from panoptic_10v3.cli import build_parser
from panoptic_10v3.depth_eval import voxel_downsample_colored
from panoptic_10v3.m1 import _load_partial_m1_records
from panoptic_10v3.visualize import (
    GROUND_Y_CM,
    _ground_grid_segments,
    _pack_cloud_points,
    _select_cloud_points,
    _unpack_cloud_points,
)


def test_partial_m1_reader_repairs_torn_final_line(tmp_path):
    path = tmp_path / "m1_2d.partial.jsonl"
    rows = [{"hd_index": 1}, {"hd_index": 2}]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n{\"hd_index\":",
        encoding="utf-8",
    )

    assert _load_partial_m1_records(path) == rows
    assert list(map(json.loads, path.read_text(encoding="utf-8").splitlines())) == rows


def test_cli_accepts_full_mps_and_resume():
    args = build_parser().parse_args(
        [
            "m1",
            "--sequence-dir",
            "sequence",
            "--frame-table",
            "frames.jsonl",
            "--output",
            "m1.jsonl",
            "--backend",
            "hf-vitpose",
            "--device",
            "full-mps",
            "--resume-m1",
        ]
    )

    assert args.device == "full-mps"
    assert args.resume_m1 is True


def test_depth_cache_accepts_frame_table_bounded_depth_filename():
    args = build_parser().parse_args(
        [
            "depth-cache",
            "--sequence-dir",
            "sequence",
            "--frame-table",
            "frames.jsonl",
            "--output-dir",
            "clouds",
            "--depth-filename",
            "depthdata.window.dat",
        ]
    )

    assert args.depth_filename == "depthdata.window.dat"


def test_ground_grid_is_metric_world_aligned_and_on_floor():
    segments = _ground_grid_segments([25.0, -80.0, -25.0])

    assert segments
    assert all(start[1] == GROUND_Y_CM for start, _, _, _ in segments)
    assert all(end[1] == GROUND_Y_CM for _, end, _, _ in segments)
    assert any(major and not axis for _, _, major, axis in segments)
    assert sum(axis for _, _, _, axis in segments) == 2


def test_colored_voxel_average_is_deterministic():
    points = np.asarray(
        [[0.1, 0.2, 0.3], [0.8, 0.4, 0.7], [2.2, 0.0, 0.0]],
        dtype=float,
    )
    colors = np.asarray([[10, 20, 30], [30, 40, 50], [90, 80, 70]], dtype=np.uint8)

    downsampled, rgb = voxel_downsample_colored(
        points,
        colors,
        voxel_size_cm=2.0,
        max_points=10,
    )

    np.testing.assert_allclose(downsampled[0], [0.45, 0.3, 0.5])
    np.testing.assert_array_equal(rgb[0], [20, 30, 40])
    np.testing.assert_allclose(downsampled[1], points[2])
    np.testing.assert_array_equal(rgb[1], colors[2])


def test_cloud_packing_preserves_geometry_color_and_near_flag():
    points = np.asarray([[1.24, -2.26, 3.05], [4.0, 5.0, 6.0]], dtype=float)
    colors = np.asarray([[1, 2, 3], [250, 251, 252]], dtype=np.uint8)
    near = np.asarray([True, False])
    cloud = {
        "packed": _pack_cloud_points(points, colors, near),
        "quantizationCm": 0.1,
    }

    decoded, decoded_colors, decoded_near = _unpack_cloud_points(cloud)

    np.testing.assert_allclose(decoded, np.round(points * 10) / 10, atol=1e-6)
    np.testing.assert_array_equal(decoded_colors, colors)
    np.testing.assert_array_equal(decoded_near, near)


def test_cloud_selection_prioritizes_near_body_points():
    near = np.zeros(100, dtype=bool)
    near[:60] = True

    chosen = _select_cloud_points(near, maximum_points=50)

    assert len(chosen) == 50
    assert np.count_nonzero(near[chosen]) == 40
