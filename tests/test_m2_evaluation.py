from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from panoptic_10v3.stage_evaluate import evaluate_m2_from_assignments


class M2EvaluationTest(unittest.TestCase):
    def test_evaluation_only_assignments_score_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assignments = root / "assignments.jsonl"
            reconstruction = root / "m3.jsonl"
            output = root / "m2.json"
            assignment_rows = [
                {
                    "hd_index": 10,
                    "camera": camera,
                    "detections": [
                        {
                            "detection_id": 0,
                            "gt_id": gt_id,
                            "matching_mean_error_px": 2.0,
                        }
                    ],
                }
                for camera, gt_id in (
                    ("50_01", 7),
                    ("50_02", 7),
                    ("50_03", 9),
                )
            ]
            assignments.write_text(
                "".join(json.dumps(row) + "\n" for row in assignment_rows),
                encoding="utf-8",
            )
            reconstruction.write_text(
                json.dumps(
                    {
                        "hd_index": 10,
                        "condition": "V3",
                        "cameras": ["50_01", "50_02", "50_03"],
                        "people": [
                            {
                                "members": [["50_01", 0], ["50_02", 0]],
                            },
                            {
                                "members": [["50_03", 0]],
                            },
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = evaluate_m2_from_assignments(
                reconstruction,
                assignments,
                output,
            )

            self.assertEqual(summary["pairwise_f1"], 1.0)
            self.assertEqual(summary["cluster_purity"], 1.0)
            self.assertEqual(summary["cluster_completeness"], 1.0)
            self.assertFalse(summary["gt_assignment_used_by_inference"])
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
