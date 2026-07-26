from __future__ import annotations

import ast
import unittest
from pathlib import Path


class DepthIsolationTest(unittest.TestCase):
    def test_inference_modules_do_not_import_depth(self) -> None:
        root = Path("src/panoptic_10v3")
        inference_modules = ("m1.py", "m2.py", "geometry.py", "reconstruct.py")
        for name in inference_modules:
            tree = ast.parse((root / name).read_text(encoding="utf-8"))
            imported = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imported.append(node.module or "")
            self.assertFalse(
                any("depth_eval" in value for value in imported),
                f"{name} must not import depth_eval",
            )


if __name__ == "__main__":
    unittest.main()
