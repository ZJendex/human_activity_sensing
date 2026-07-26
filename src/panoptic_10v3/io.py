"""Dataset loading, synchronization, and compact JSONL artifact helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

import cv2
import numpy as np

from .constants import ALL_CAMERAS, COCO17_FROM_PANOPTIC19
from .model import Camera


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False))
            handle.write("\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cameras(sequence_dir: Path) -> Dict[str, Camera]:
    sequence = sequence_dir.name
    calibration = load_json(sequence_dir / f"calibration_{sequence}.json")
    cameras: Dict[str, Camera] = {}
    for raw in calibration["cameras"]:
        if raw.get("type") != "kinect-color":
            continue
        width, height = raw["resolution"]
        camera = Camera(
            name=str(raw["name"]),
            node=int(raw["node"]),
            width=int(width),
            height=int(height),
            K=np.asarray(raw["K"], dtype=float),
            dist=np.asarray(raw["distCoef"], dtype=float).reshape(-1),
            R=np.asarray(raw["R"], dtype=float),
            t=np.asarray(raw["t"], dtype=float).reshape(3),
        )
        cameras[camera.name] = camera
    missing = sorted(set(ALL_CAMERAS) - set(cameras))
    if missing:
        raise ValueError(f"Missing Kinect color cameras in calibration: {missing}")
    return cameras


def load_gt_coco17(path: Path) -> List[Dict[str, Any]]:
    frame = load_json(path)
    people: List[Dict[str, Any]] = []
    mapping = np.asarray(COCO17_FROM_PANOPTIC19, dtype=int)
    for body in frame.get("bodies", []):
        joints19 = np.asarray(body["joints19"], dtype=float).reshape(19, 4)
        people.append(
            {
                "id": int(body["id"]),
                "joints_cm": joints19[mapping, :3],
                "confidence": joints19[mapping, 3],
            }
        )
    return people


def build_frame_table(
    sequence_dir: Path,
    output_path: Path,
    stride: int = 1,
    max_frames: Optional[int] = None,
    require_all_cameras: bool = True,
    color_tolerance_ms: float = 30.0,
) -> Dict[str, Any]:
    """Map HD/skeleton time to each Kinect RGB source frame.

    The official Kinoptic convention compares the Panoptic HD universal time
    against ``kinect_color_univ_time - 6.25 ms``.
    """

    sequence = sequence_dir.name
    sync = load_json(sequence_dir / f"synctables_{sequence}.json")
    ksync = load_json(sequence_dir / f"ksynctables_{sequence}.json")["kinect"]["color"]
    hd_times = np.asarray(sync["hd"]["univ_time"], dtype=float)
    color_times: Dict[str, np.ndarray] = {}
    color_indices: Dict[str, np.ndarray] = {}
    for camera in ALL_CAMERAS:
        node = int(camera.split("_")[1])
        raw = ksync[f"KINECTNODE{node}"]
        color_times[camera] = np.asarray(raw["univ_time"], dtype=float) - 6.25
        color_indices[camera] = np.asarray(raw["index"], dtype=int)

    gt_dir = sequence_dir / "hdPose3d_stage1_coco19"
    if not any(gt_dir.glob("body3DScene_*.json")) and (gt_dir / "hd").is_dir():
        gt_dir = gt_dir / "hd"
    rows: List[Dict[str, Any]] = []
    candidates = 0
    for hd_index in range(0, len(hd_times), max(1, stride)):
        gt_path = gt_dir / f"body3DScene_{hd_index:08d}.json"
        if not gt_path.exists():
            continue
        gt = load_json(gt_path)
        target = float(hd_times[hd_index])
        if target <= 0 or float(gt.get("univTime", -1)) <= 0:
            continue
        if abs(float(gt["univTime"]) - target) > 1e-3:
            raise ValueError(f"GT/sync time mismatch at HD index {hd_index}")
        candidates += 1
        mapped: Dict[str, Any] = {}
        for camera in ALL_CAMERAS:
            times = color_times[camera]
            valid_positions = np.flatnonzero(times > 0)
            if valid_positions.size == 0:
                continue
            nearest_pos = int(valid_positions[np.argmin(np.abs(times[valid_positions] - target))])
            delta = float(times[nearest_pos] - target)
            mapped[camera] = {
                "source_index": int(color_indices[camera][nearest_pos]),
                "sync_position": nearest_pos,
                "delta_ms": delta,
                "valid": abs(delta) <= color_tolerance_ms,
            }
        available = [name for name, item in mapped.items() if item["valid"]]
        if require_all_cameras and len(available) != len(ALL_CAMERAS):
            continue
        rows.append(
            {
                "sequence": sequence,
                "hd_index": hd_index,
                "univ_time_ms": target,
                "gt_path": str(gt_path),
                "person_count": len(gt.get("bodies", [])),
                "cameras": mapped,
                "available_cameras": available,
            }
        )
        if max_frames is not None and len(rows) >= max_frames:
            break

    write_jsonl(output_path, rows)
    summary = {
        "sequence": sequence,
        "frame_table": str(output_path),
        "candidate_gt_frames": candidates,
        "selected_frames": len(rows),
        "stride": max(1, stride),
        "max_frames": max_frames,
        "require_all_cameras": require_all_cameras,
        "color_tolerance_ms": color_tolerance_ms,
    }
    write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def kinect_video_path(sequence_dir: Path, camera: str) -> Path:
    sequence = sequence_dir.name
    candidates = (
        sequence_dir / "kinectVideos" / f"{camera}.mp4",
        sequence_dir / "kinectVideos" / f"{camera}_{sequence}.mp4",
        sequence_dir / "kinectVideos" / f"kinect_{camera}.mp4",
        sequence_dir / "kinectVideos" / f"{camera}.avi",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted((sequence_dir / "kinectVideos").glob(f"*{camera}*"))
    if not matches:
        raise FileNotFoundError(f"No RGB video found for {camera}")
    return matches[0]


class VideoReaderPool:
    """Random-access OpenCV readers, one per Kinect video."""

    def __init__(self, sequence_dir: Path):
        self.sequence_dir = sequence_dir
        self._captures: Dict[str, cv2.VideoCapture] = {}

    def read(self, camera: str, source_index: int) -> np.ndarray:
        capture = self._captures.get(camera)
        if capture is None:
            path = kinect_video_path(self.sequence_dir, camera)
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise RuntimeError(f"Could not open {path}")
            self._captures[camera] = capture
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(source_index))
        ok, image = capture.read()
        if not ok or image is None:
            raise RuntimeError(f"Could not read {camera} source frame {source_index}")
        return image

    def close(self) -> None:
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()

    def __enter__(self) -> "VideoReaderPool":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
