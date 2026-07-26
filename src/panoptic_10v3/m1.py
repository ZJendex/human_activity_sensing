"""M1: per-camera 2D joint inference and deterministic study controls."""

from __future__ import annotations

import hashlib
import json
import os
import time
import types
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

import numpy as np

from .constants import ALL_CAMERAS
from .geometry import project_points
from .io import VideoReaderPool, load_cameras, load_gt_coco17, read_jsonl, write_jsonl
from .model import Camera, Detection


def _frame_seed(seed: int, hd_index: int, camera: str, person_id: int) -> int:
    payload = f"{seed}:{hd_index}:{camera}:{person_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def detections_from_gt_projection(
    gt_people: Sequence[Mapping[str, Any]],
    camera: Camera,
    hd_index: int,
    noise_px: float,
    miss_probability: float,
    joint_dropout_probability: float,
    seed: int,
) -> List[Detection]:
    """Synthetic M1 control: GT projection plus independent pixel noise.

    This backend is deliberately named ``oracle-noise`` in every output. It is
    for geometry/association validation and is never reported as ViTPose.
    """

    detections: List[Detection] = []
    for person in gt_people:
        person_id = int(person["id"])
        rng = np.random.default_rng(_frame_seed(seed, hd_index, camera.name, person_id))
        if rng.random() < miss_probability:
            continue
        joints = np.asarray(person["joints_cm"], dtype=float)
        gt_conf = np.asarray(person["confidence"], dtype=float)
        pixels, depth = project_points(camera, joints)
        pixels = pixels + rng.normal(0.0, noise_px, size=pixels.shape)
        inside = (
            (depth > 0)
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] < camera.width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < camera.height)
        )
        scores = np.clip(0.35 + 0.65 * gt_conf, 0.0, 1.0)
        scores[~inside] = 0.0
        scores[rng.random(len(scores)) < joint_dropout_probability] = 0.0
        valid = scores > 0.05
        if np.count_nonzero(valid) < 4:
            continue
        confidence_noise = np.exp(-np.linalg.norm(pixels - project_points(camera, joints)[0], axis=1) / 18.0)
        scores = np.clip(scores * confidence_noise, 0.0, 1.0)
        bbox = _bbox_from_keypoints(pixels, valid, camera.width, camera.height)
        keypoints = np.column_stack((pixels, scores))
        detections.append(
            Detection(
                camera=camera.name,
                detection_id=len(detections),
                keypoints=keypoints,
                bbox=bbox,
                source_person_id=person_id,
                instance_score=1.0,
            )
        )
    rng_order = np.random.default_rng(_frame_seed(seed, hd_index, camera.name, 999_999))
    rng_order.shuffle(detections)
    for detection_id, detection in enumerate(detections):
        detection.detection_id = detection_id
    return detections


def _bbox_from_keypoints(
    pixels: np.ndarray,
    valid: np.ndarray,
    width: int,
    height: int,
    padding: float = 24.0,
) -> np.ndarray:
    chosen = np.asarray(pixels)[np.asarray(valid, dtype=bool)]
    x1, y1 = np.min(chosen, axis=0) - padding
    x2, y2 = np.max(chosen, axis=0) + padding
    return np.array(
        [max(0.0, x1), max(0.0, y1), min(float(width - 1), x2), min(float(height - 1), y2)],
        dtype=float,
    )


def run_oracle_noise(
    sequence_dir: Path,
    frame_table_path: Path,
    output_path: Path,
    cameras: Sequence[str] = ALL_CAMERAS,
    noise_px: float = 5.0,
    miss_probability: float = 0.02,
    joint_dropout_probability: float = 0.03,
    seed: int = 20260724,
) -> Dict[str, Any]:
    calibration = load_cameras(sequence_dir)

    def rows() -> Iterator[Dict[str, Any]]:
        for frame in read_jsonl(frame_table_path):
            gt_people = load_gt_coco17(Path(frame["gt_path"]))
            for camera_name in cameras:
                if not frame["cameras"][camera_name]["valid"]:
                    continue
                detections = detections_from_gt_projection(
                    gt_people=gt_people,
                    camera=calibration[camera_name],
                    hd_index=int(frame["hd_index"]),
                    noise_px=noise_px,
                    miss_probability=miss_probability,
                    joint_dropout_probability=joint_dropout_probability,
                    seed=seed,
                )
                yield {
                    "backend": "oracle-noise",
                    "sequence": frame["sequence"],
                    "hd_index": int(frame["hd_index"]),
                    "univ_time_ms": float(frame["univ_time_ms"]),
                    "camera": camera_name,
                    "source_index": int(frame["cameras"][camera_name]["source_index"]),
                    "detections": [item.to_json() for item in detections],
                }

    record_count = write_jsonl(output_path, rows())
    return {
        "backend": "oracle-noise",
        "output": str(output_path),
        "camera_frame_records": record_count,
        "noise_px": noise_px,
        "miss_probability": miss_probability,
        "joint_dropout_probability": joint_dropout_probability,
        "seed": seed,
    }


class MMPoseViTPoseBackend:
    """Thin optional adapter around the official MMPose ViTPose-B alias."""

    def __init__(self, device: Optional[str] = None):
        try:
            from mmpose.apis import MMPoseInferencer
        except ImportError as error:
            raise RuntimeError(
                "ViTPose requires the optional MMPose environment. "
                "Run scripts/install_vitpose_env.sh, then retry from that environment."
            ) from error
        self.inferencer = MMPoseInferencer(pose2d="vitpose-b", device=device)

    def infer(self, image_bgr: np.ndarray, camera: str) -> List[Detection]:
        # MMPose accepts a numpy image. Its OpenCV loader convention is BGR.
        result = next(
            self.inferencer(
                image_bgr,
                return_vis=False,
                show=False,
                draw_bbox=False,
            )
        )
        predictions = result.get("predictions", [])
        if len(predictions) == 1 and isinstance(predictions[0], list):
            predictions = predictions[0]
        detections: List[Detection] = []
        for raw in predictions:
            keypoints = np.asarray(raw.get("keypoints", []), dtype=float)
            scores = np.asarray(raw.get("keypoint_scores", []), dtype=float).reshape(-1)
            if keypoints.shape != (17, 2) or scores.shape != (17,):
                continue
            raw_bbox = np.asarray(raw.get("bbox", []), dtype=float)
            if raw_bbox.ndim > 1:
                raw_bbox = raw_bbox[0]
            if raw_bbox.size < 4:
                valid = scores > 0.05
                if np.count_nonzero(valid) < 2:
                    continue
                raw_bbox = _bbox_from_keypoints(
                    keypoints,
                    valid,
                    image_bgr.shape[1],
                    image_bgr.shape[0],
                )
            detections.append(
                Detection(
                    camera=camera,
                    detection_id=len(detections),
                    keypoints=np.column_stack((keypoints, scores)),
                    bbox=raw_bbox[:4],
                    source_person_id=None,
                    instance_score=float(
                        np.asarray(
                            raw.get("bbox_scores", raw.get("bbox_score", [1.0]))
                        ).reshape(-1)[0]
                    ),
                )
            )
        return suppress_duplicate_poses(detections)


def suppress_duplicate_poses(
    detections: Sequence[Detection],
    normalized_keypoint_distance: float = 0.12,
    keypoint_score_threshold: float = 0.20,
) -> List[Detection]:
    """Pose-NMS for nested/partial duplicate detector boxes.

    RT-DETR can return several nested boxes for the same heavily occluded
    person. IoU NMS is insufficient because a partial box may have low IoU
    with the full box; comparing normalized pose coordinates is much more
    discriminative.
    """

    kept: List[Detection] = []
    for candidate in sorted(
        detections,
        key=lambda item: (-item.instance_score, item.detection_id),
    ):
        duplicate = False
        for existing in kept:
            valid = (
                (candidate.keypoints[:, 2] >= keypoint_score_threshold)
                & (existing.keypoints[:, 2] >= keypoint_score_threshold)
            )
            if np.count_nonzero(valid) < 4:
                continue
            candidate_diag = np.hypot(
                candidate.bbox[2] - candidate.bbox[0],
                candidate.bbox[3] - candidate.bbox[1],
            )
            existing_diag = np.hypot(
                existing.bbox[2] - existing.bbox[0],
                existing.bbox[3] - existing.bbox[1],
            )
            scale = max(1.0, 0.5 * (candidate_diag + existing_diag))
            distances = np.linalg.norm(
                candidate.keypoints[valid, :2] - existing.keypoints[valid, :2],
                axis=1,
            )
            if float(np.median(distances) / scale) < normalized_keypoint_distance:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    for index, detection in enumerate(kept):
        detection.detection_id = index
    return kept


class TransformersViTPoseBackend:
    """Cross-platform ViTPose-B + RT-DETR adapter with staged fallback.

    On Apple silicon, ``full-mps`` keeps both models on Metal. Transformers'
    RT-DETR sine-position helper explicitly constructs float64 tensors, which
    MPS does not support, so only that small deterministic helper runs on CPU.
    A failed full-MPS inference falls back to CPU detector + MPS pose
    (``hybrid``), followed by all-CPU as the last recovery tier.
    """

    detector_id = "PekingU/rtdetr_r50vd_coco_o365"
    detector_revision = "457857cec8ac28ddede40ecee9eed2beca321af8"
    pose_id = "usyd-community/vitpose-base-simple"
    pose_revision = "a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45"
    valid_devices = {"auto", "full-mps", "mps", "hybrid", "cpu", "cuda"}

    def __init__(self, device: Optional[str] = None, detector_threshold: float = 0.30):
        try:
            import torch
            from PIL import Image
            from transformers import (
                AutoImageProcessor,
                AutoProcessor,
                RTDetrForObjectDetection,
                VitPoseForPoseEstimation,
            )
        except ImportError as error:
            raise RuntimeError(
                "The Transformers ViTPose backend needs transformers>=4.48, "
                "torch, and Pillow. Install the isolated environment with "
                "scripts/install_hf_vitpose_env.sh."
            ) from error
        self.torch = torch
        self.Image = Image
        requested = "auto" if device is None else str(device).lower()
        if requested not in self.valid_devices:
            raise ValueError(
                f"Unsupported device {device!r}; choose from "
                f"{sorted(self.valid_devices)}"
            )
        if requested == "mps":
            requested = "full-mps"
        self.requested_device = requested
        self.detector_threshold = detector_threshold
        self.started_at = time.perf_counter()
        self.stats: Dict[str, Any] = {
            "images": 0,
            "detections": 0,
            "detector_seconds": 0.0,
            "pose_seconds": 0.0,
            "fallbacks": [],
        }
        self.detector_processor = AutoImageProcessor.from_pretrained(
            self.detector_id,
            revision=self.detector_revision,
        )
        self.detector = RTDetrForObjectDetection.from_pretrained(
            self.detector_id,
            revision=self.detector_revision,
        ).eval()
        self.pose_processor = AutoProcessor.from_pretrained(
            self.pose_id,
            revision=self.pose_revision,
        )
        self.pose = VitPoseForPoseEstimation.from_pretrained(
            self.pose_id,
            revision=self.pose_revision,
        ).eval()
        self._install_mps_safe_rtdetr_position_embedding()
        self._set_execution_mode(self._resolve_mode(requested))

    def _resolve_mode(self, requested: str) -> str:
        torch = self.torch
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if (
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ):
                return "full-mps"
            return "cpu"
        if requested in {"full-mps", "hybrid"}:
            available = (
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            )
            if not available:
                self.stats["fallbacks"].append(
                    {
                        "from": requested,
                        "to": "cpu",
                        "reason": "MPS is unavailable",
                    }
                )
                return "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            self.stats["fallbacks"].append(
                {
                    "from": "cuda",
                    "to": "cpu",
                    "reason": "CUDA is unavailable",
                }
            )
            return "cpu"
        return requested

    def _install_mps_safe_rtdetr_position_embedding(self) -> None:
        """Keep RT-DETR's float64 sinusoid construction off MPS."""

        try:
            from transformers.models.rt_detr.modeling_rt_detr import (
                RTDetrSinePositionEmbedding,
            )
        except ImportError:
            return

        for module in self.detector.modules():
            if not isinstance(module, RTDetrSinePositionEmbedding):
                continue
            if getattr(module, "_panoptic_mps_safe", False):
                continue
            original_forward = module.forward
            cache: Dict[Any, Any] = {}

            def safe_forward(
                module_self: Any,
                width: int,
                height: int,
                device: Any,
                dtype: Any,
                *,
                _original: Any = original_forward,
                _cache: Dict[Any, Any] = cache,
            ) -> Any:
                destination = self.torch.device(device)
                if destination.type != "mps":
                    return _original(width, height, device, dtype)
                key = (int(width), int(height), str(dtype))
                if key not in _cache:
                    _cache[key] = _original(width, height, "cpu", dtype)
                return _cache[key].to(destination)

            module.forward = types.MethodType(safe_forward, module)
            module._panoptic_mps_safe = True

    def _set_execution_mode(self, mode: str) -> None:
        torch = self.torch
        if mode == "full-mps":
            detector_device, pose_device = "mps", "mps"
        elif mode == "hybrid":
            detector_device, pose_device = "cpu", "mps"
        elif mode == "cuda":
            detector_device, pose_device = "cuda", "cuda"
        else:
            detector_device, pose_device = "cpu", "cpu"
            mode = "cpu"
        self.mode = mode
        self.device = mode
        self.detector_device = torch.device(detector_device)
        self.pose_device = torch.device(pose_device)
        self.detector.to(self.detector_device).eval()
        self.pose.to(self.pose_device).eval()

    def _synchronize(self, device: Any) -> None:
        if device.type == "mps":
            self.torch.mps.synchronize()
        elif device.type == "cuda":
            self.torch.cuda.synchronize(device)

    def _infer_once(
        self,
        image_bgr: np.ndarray,
        camera: str,
    ) -> tuple[List[Detection], float, float]:
        torch = self.torch
        image = self.Image.fromarray(image_bgr[:, :, ::-1])
        detector_inputs = self.detector_processor(images=image, return_tensors="pt")
        detector_inputs = {
            key: value.to(self.detector_device)
            for key, value in detector_inputs.items()
        }
        self._synchronize(self.detector_device)
        detector_start = time.perf_counter()
        with torch.inference_mode():
            detector_outputs = self.detector(**detector_inputs)
        self._synchronize(self.detector_device)
        detector_seconds = time.perf_counter() - detector_start
        detected = self.detector_processor.post_process_object_detection(
            detector_outputs,
            target_sizes=torch.tensor(
                [(image.height, image.width)],
                device=self.detector_device,
            ),
            threshold=self.detector_threshold,
        )[0]
        person_mask = detected["labels"] == 0
        boxes_xyxy = detected["boxes"][person_mask].detach().cpu().numpy()
        person_scores = detected["scores"][person_mask].detach().cpu().numpy()
        if not len(boxes_xyxy):
            return [], detector_seconds, 0.0
        boxes_xywh = boxes_xyxy.copy()
        boxes_xywh[:, 2] -= boxes_xywh[:, 0]
        boxes_xywh[:, 3] -= boxes_xywh[:, 1]
        pose_inputs = self.pose_processor(
            image,
            boxes=[boxes_xywh],
            return_tensors="pt",
        )
        pose_inputs = {
            key: value.to(self.pose_device) for key, value in pose_inputs.items()
        }
        self._synchronize(self.pose_device)
        pose_start = time.perf_counter()
        with torch.inference_mode():
            pose_outputs = self.pose(**pose_inputs)
        self._synchronize(self.pose_device)
        pose_seconds = time.perf_counter() - pose_start
        pose_results = self.pose_processor.post_process_pose_estimation(
            pose_outputs,
            boxes=[boxes_xywh],
        )[0]
        detections: List[Detection] = []
        for index, raw in enumerate(pose_results):
            keypoints = raw["keypoints"].detach().cpu().numpy()
            scores = raw["scores"].detach().cpu().numpy()
            if keypoints.shape != (17, 2) or scores.shape != (17,):
                continue
            detections.append(
                Detection(
                    camera=camera,
                    detection_id=len(detections),
                    keypoints=np.column_stack((keypoints, scores)),
                    bbox=boxes_xyxy[index],
                    source_person_id=None,
                    instance_score=float(person_scores[index]),
                )
            )
        return suppress_duplicate_poses(detections), detector_seconds, pose_seconds

    def infer(self, image_bgr: np.ndarray, camera: str) -> List[Detection]:
        attempted = set()
        while True:
            attempted.add(self.mode)
            try:
                detections, detector_seconds, pose_seconds = self._infer_once(
                    image_bgr,
                    camera,
                )
                self.stats["images"] += 1
                self.stats["detections"] += len(detections)
                self.stats["detector_seconds"] += detector_seconds
                self.stats["pose_seconds"] += pose_seconds
                return detections
            except (RuntimeError, TypeError, NotImplementedError) as error:
                fallback = {
                    "full-mps": "hybrid",
                    "hybrid": "cpu",
                }.get(self.mode)
                if fallback is None or fallback in attempted:
                    raise
                self.stats["fallbacks"].append(
                    {
                        "from": self.mode,
                        "to": fallback,
                        "camera": camera,
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
                self._set_execution_mode(fallback)

    def runtime_summary(self) -> Dict[str, Any]:
        elapsed = time.perf_counter() - self.started_at
        summary: Dict[str, Any] = {
            "requested_device": self.requested_device,
            "execution_mode": self.mode,
            "detector_device": str(self.detector_device),
            "pose_device": str(self.pose_device),
            "dtype": "float32",
            "images": int(self.stats["images"]),
            "detections": int(self.stats["detections"]),
            "detector_seconds": float(self.stats["detector_seconds"]),
            "pose_seconds": float(self.stats["pose_seconds"]),
            "elapsed_seconds_including_model_load": elapsed,
            "fallbacks": list(self.stats["fallbacks"]),
        }
        if self.detector_device.type == "mps" or self.pose_device.type == "mps":
            try:
                summary["mps_current_allocated_bytes"] = int(
                    self.torch.mps.current_allocated_memory()
                )
                summary["mps_driver_allocated_bytes"] = int(
                    self.torch.mps.driver_allocated_memory()
                )
            except (AttributeError, RuntimeError):
                pass
        return summary


def run_vitpose(
    sequence_dir: Path,
    frame_table_path: Path,
    output_path: Path,
    cameras: Sequence[str] = ALL_CAMERAS,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    backend = MMPoseViTPoseBackend(device=device)

    def rows() -> Iterator[Dict[str, Any]]:
        with VideoReaderPool(sequence_dir) as videos:
            for frame in read_jsonl(frame_table_path):
                for camera in cameras:
                    mapping = frame["cameras"][camera]
                    if not mapping["valid"]:
                        continue
                    image = videos.read(camera, int(mapping["source_index"]))
                    detections = backend.infer(image, camera)
                    yield {
                        "backend": "mmpose-vitpose-b",
                        "sequence": frame["sequence"],
                        "hd_index": int(frame["hd_index"]),
                        "univ_time_ms": float(frame["univ_time_ms"]),
                        "camera": camera,
                        "source_index": int(mapping["source_index"]),
                        "detections": [item.to_json() for item in detections],
                    }

    record_count = write_jsonl(output_path, rows())
    return {
        "backend": "mmpose-vitpose-b",
        "output": str(output_path),
        "camera_frame_records": record_count,
        "device": device,
    }


def run_transformers_vitpose(
    sequence_dir: Path,
    frame_table_path: Path,
    output_path: Path,
    cameras: Sequence[str] = ALL_CAMERAS,
    device: Optional[str] = "auto",
    detector_threshold: float = 0.30,
    resume: bool = False,
) -> Dict[str, Any]:
    started = time.perf_counter()
    frames = list(read_jsonl(frame_table_path))
    expected = [
        (
            frame,
            camera,
            frame["cameras"][camera],
        )
        for frame in frames
        for camera in cameras
        if frame["cameras"][camera]["valid"]
    ]
    expected_keys = {
        (
            int(frame["hd_index"]),
            camera,
            int(mapping["source_index"]),
        )
        for frame, camera, mapping in expected
    }
    partial_path = output_path.with_suffix(".partial.jsonl")
    completed_keys: set[tuple[int, str, int]] = set()
    resumed_records = 0
    if resume and partial_path.exists():
        partial_rows = _load_partial_m1_records(partial_path)
        for row in partial_rows:
            if row.get("backend") != (
                "transformers-vitpose-base-simple+rtdetr-r50"
            ):
                raise ValueError("Partial M1 cache has a different backend")
            if row.get("pose_revision") != TransformersViTPoseBackend.pose_revision:
                raise ValueError("Partial M1 cache has a different pose revision")
            if (
                row.get("detector_revision")
                != TransformersViTPoseBackend.detector_revision
            ):
                raise ValueError("Partial M1 cache has a different detector revision")
            if abs(
                float(row.get("detector_threshold", detector_threshold))
                - detector_threshold
            ) > 1e-12:
                raise ValueError("Partial M1 cache has a different detector threshold")
            key = (
                int(row["hd_index"]),
                str(row["camera"]),
                int(row["source_index"]),
            )
            if key in completed_keys:
                raise ValueError(f"Duplicate record in partial M1 cache: {key}")
            if key not in expected_keys:
                raise ValueError(f"Unexpected record in partial M1 cache: {key}")
            completed_keys.add(key)
        resumed_records = len(completed_keys)
    else:
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_text("", encoding="utf-8")

    backend = TransformersViTPoseBackend(
        device=device,
        detector_threshold=detector_threshold,
    )
    new_records = 0
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    with VideoReaderPool(sequence_dir) as videos, partial_path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        for frame, camera, mapping in expected:
            key = (
                int(frame["hd_index"]),
                camera,
                int(mapping["source_index"]),
            )
            if key in completed_keys:
                continue
            image = videos.read(camera, int(mapping["source_index"]))
            detections = backend.infer(image, camera)
            row = {
                "backend": "transformers-vitpose-base-simple+rtdetr-r50",
                "pose_model": backend.pose_id,
                "pose_revision": backend.pose_revision,
                "detector_model": backend.detector_id,
                "detector_revision": backend.detector_revision,
                "detector_threshold": detector_threshold,
                "sequence": frame["sequence"],
                "hd_index": int(frame["hd_index"]),
                "univ_time_ms": float(frame["univ_time_ms"]),
                "camera": camera,
                "source_index": int(mapping["source_index"]),
                "detections": [item.to_json() for item in detections],
            }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            handle.flush()
            new_records += 1
            completed_keys.add(key)
            if new_records % 10 == 0:
                os.fsync(handle.fileno())
            if len(completed_keys) % 50 == 0:
                print(
                    f"M1 progress: {len(completed_keys)}/{len(expected_keys)} "
                    f"camera frames ({backend.mode})",
                    flush=True,
                )
        os.fsync(handle.fileno())
    if completed_keys != expected_keys:
        missing = sorted(expected_keys - completed_keys)[:5]
        raise RuntimeError(f"M1 cache is incomplete; missing examples: {missing}")
    os.replace(partial_path, output_path)
    record_count = len(completed_keys)
    return {
        "backend": "transformers-vitpose-base-simple+rtdetr-r50",
        "pose_model": backend.pose_id,
        "pose_revision": backend.pose_revision,
        "detector_model": backend.detector_id,
        "detector_revision": backend.detector_revision,
        "detector_threshold": detector_threshold,
        "output": str(output_path),
        "camera_frame_records": record_count,
        "device": backend.mode,
        "resume_requested": resume,
        "resumed_camera_frame_records": resumed_records,
        "new_camera_frame_records": new_records,
        "wall_seconds": time.perf_counter() - started,
        "runtime": backend.runtime_summary(),
    }


def _load_partial_m1_records(path: Path) -> List[Dict[str, Any]]:
    """Read a partial JSONL cache and repair only a torn final record."""

    lines = path.read_text(encoding="utf-8").splitlines()
    rows: List[Dict[str, Any]] = []
    repaired = False
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError(
                    f"Malformed non-final line {index + 1} in partial M1 cache"
                )
            repaired = True
    if repaired:
        temporary = path.with_suffix(path.suffix + ".repair")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    return rows


def load_m1_by_frame(path: Path) -> Dict[int, Dict[str, List[Detection]]]:
    grouped: Dict[int, Dict[str, List[Detection]]] = {}
    for row in read_jsonl(path):
        hd_index = int(row["hd_index"])
        grouped.setdefault(hd_index, {})[str(row["camera"])] = [
            Detection.from_json(item) for item in row["detections"]
        ]
    return grouped
