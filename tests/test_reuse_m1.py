from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from panoptic_10v3.cli import build_parser, inspect_reused_m1


class ReuseM1Test(unittest.TestCase):
    def test_reuse_flag_belongs_only_to_run_study(self) -> None:
        parser = build_parser()
        m1 = parser.parse_args(
            [
                "m1",
                "--sequence-dir",
                "sequence",
                "--frame-table",
                "frames.jsonl",
                "--output",
                "m1.jsonl",
            ]
        )
        study = parser.parse_args(
            [
                "run-study",
                "--sequence-dir",
                "sequence",
                "--output-dir",
                "output",
                "--reuse-m1",
            ]
        )
        self.assertFalse(hasattr(m1, "reuse_m1"))
        self.assertTrue(study.reuse_m1)

    def test_reused_cache_must_match_frame_camera_and_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = root / "frames.jsonl"
            m1 = root / "m1.jsonl"
            frames.write_text(
                json.dumps(
                    {
                        "hd_index": 12,
                        "cameras": {
                            "50_01": {
                                "valid": True,
                                "source_index": 34,
                            }
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            row = {
                "backend": "oracle-noise",
                "hd_index": 12,
                "camera": "50_01",
                "source_index": 34,
                "detections": [],
            }
            m1.write_text(json.dumps(row) + "\n", encoding="utf-8")

            result = inspect_reused_m1(
                frames,
                m1,
                ("50_01",),
                "oracle-noise",
            )
            self.assertTrue(result["reused"])
            self.assertEqual(result["camera_frame_records"], 1)

            row["source_index"] = 35
            m1.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                inspect_reused_m1(
                    frames,
                    m1,
                    ("50_01",),
                    "oracle-noise",
                )


if __name__ == "__main__":
    unittest.main()
