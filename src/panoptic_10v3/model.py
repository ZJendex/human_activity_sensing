"""Small serializable data models used across stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Camera:
    name: str
    node: int
    width: int
    height: int
    K: np.ndarray
    dist: np.ndarray
    R: np.ndarray
    t: np.ndarray

    @property
    def center_world_cm(self) -> np.ndarray:
        return -(self.R.T @ self.t.reshape(3))

    @property
    def projection(self) -> np.ndarray:
        return self.K @ np.column_stack((self.R, self.t.reshape(3)))


@dataclass
class Detection:
    camera: str
    detection_id: int
    keypoints: np.ndarray  # [17, 3] = x, y, score
    bbox: np.ndarray  # [4] = x1, y1, x2, y2
    source_person_id: Optional[int] = None
    instance_score: float = 1.0

    def to_json(self) -> Dict[str, Any]:
        return {
            "camera": self.camera,
            "detection_id": int(self.detection_id),
            "keypoints": self.keypoints.tolist(),
            "bbox": self.bbox.tolist(),
            "source_person_id": self.source_person_id,
            "instance_score": float(self.instance_score),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "Detection":
        source = value.get("source_person_id")
        return cls(
            camera=str(value["camera"]),
            detection_id=int(value["detection_id"]),
            keypoints=np.asarray(value["keypoints"], dtype=float),
            bbox=np.asarray(value["bbox"], dtype=float),
            source_person_id=None if source is None else int(source),
            instance_score=float(value.get("instance_score", 1.0)),
        )


@dataclass
class Reconstruction:
    local_id: int
    joints_cm: np.ndarray  # [17, 3]
    joint_valid: np.ndarray  # [17]
    joint_support: np.ndarray  # [17]
    reprojection_rmse_px: np.ndarray  # [17]
    members: List[Tuple[str, int]]
    source_person_ids: List[Optional[int]]
    track_id: Optional[int] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "local_id": int(self.local_id),
            "track_id": self.track_id,
            "joints_cm": _nan_to_none(self.joints_cm),
            "joint_valid": self.joint_valid.astype(bool).tolist(),
            "joint_support": self.joint_support.astype(int).tolist(),
            "reprojection_rmse_px": _nan_to_none(self.reprojection_rmse_px),
            "members": [[camera, int(det_id)] for camera, det_id in self.members],
            "source_person_ids": self.source_person_ids,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "Reconstruction":
        return cls(
            local_id=int(value["local_id"]),
            track_id=value.get("track_id"),
            joints_cm=_none_to_nan(value["joints_cm"]),
            joint_valid=np.asarray(value["joint_valid"], dtype=bool),
            joint_support=np.asarray(value["joint_support"], dtype=int),
            reprojection_rmse_px=_none_to_nan(value["reprojection_rmse_px"]),
            members=[(str(x[0]), int(x[1])) for x in value["members"]],
            source_person_ids=list(value.get("source_person_ids", [])),
        )


def _nan_to_none(array: np.ndarray) -> Any:
    value = np.asarray(array, dtype=float).tolist()

    def clean(item: Any) -> Any:
        if isinstance(item, list):
            return [clean(x) for x in item]
        return None if not np.isfinite(item) else float(item)

    return clean(value)


def _none_to_nan(value: Any) -> np.ndarray:
    def clean(item: Any) -> Any:
        if isinstance(item, list):
            return [clean(x) for x in item]
        return np.nan if item is None else float(item)

    return np.asarray(clean(value), dtype=float)
