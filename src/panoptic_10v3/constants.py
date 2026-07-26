"""Joint definitions and fixed study camera sets."""

from __future__ import annotations

COCO17_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

PANOPTIC19_NAMES = (
    "neck",
    "nose",
    "body_center",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_eye",
    "left_ear",
    "right_eye",
    "right_ear",
)

# COCO-17 index -> Panoptic COCO19 index.
COCO17_FROM_PANOPTIC19 = (1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14)

COCO17_EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

TORSO_JOINTS = (5, 6, 11, 12)
ALL_CAMERAS = tuple(f"50_{node:02d}" for node in range(1, 11))
BALANCED_THREE = ("50_06", "50_04", "50_02")

PERSON_COLORS = (
    "#2563eb",
    "#e58b17",
    "#657a2d",
    "#c0447b",
    "#6941c6",
    "#0f766e",
)
