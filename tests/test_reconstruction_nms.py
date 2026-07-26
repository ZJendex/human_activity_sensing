from __future__ import annotations

import unittest

import numpy as np

from panoptic_10v3.model import Reconstruction
from panoptic_10v3.reconstruct import suppress_duplicate_reconstructions


def make_reconstruction(
    local_id: int,
    offset_cm: float,
    support: int,
) -> Reconstruction:
    joints = np.zeros((17, 3), dtype=float)
    joints[:, 0] = offset_cm
    joints[:, 1] = np.linspace(0.0, 160.0, 17)
    return Reconstruction(
        local_id=local_id,
        joints_cm=joints,
        joint_valid=np.ones(17, dtype=bool),
        joint_support=np.full(17, support, dtype=int),
        reprojection_rmse_px=np.full(17, 2.0, dtype=float),
        members=[("50_01", local_id)] * support,
        source_person_ids=[None] * support,
    )


class ReconstructionNmsTest(unittest.TestCase):
    def test_keeps_supported_duplicate_and_separate_person(self) -> None:
        strong = make_reconstruction(0, 0.0, 8)
        duplicate = make_reconstruction(1, 12.0, 3)
        separate = make_reconstruction(2, 100.0, 5)

        kept = suppress_duplicate_reconstructions(
            [duplicate, separate, strong],
            distance_threshold_cm=30.0,
        )

        self.assertEqual(len(kept), 2)
        self.assertIs(kept[0], strong)
        self.assertIs(kept[1], separate)
        self.assertEqual([item.local_id for item in kept], [0, 1])

    def test_zero_threshold_disables_suppression(self) -> None:
        one = make_reconstruction(0, 0.0, 8)
        two = make_reconstruction(1, 5.0, 3)
        self.assertEqual(
            len(suppress_duplicate_reconstructions([one, two], 0.0)),
            2,
        )


if __name__ == "__main__":
    unittest.main()
