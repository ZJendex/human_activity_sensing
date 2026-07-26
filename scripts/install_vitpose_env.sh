#!/usr/bin/env bash
set -euo pipefail

echo "Create a Python 3.10 or 3.11 environment before running this script."
python - <<'PY'
import sys
if sys.version_info[:2] not in {(3, 10), (3, 11)}:
    raise SystemExit(f"MMPose study environment requires Python 3.10/3.11; got {sys.version.split()[0]}")
PY

python -m pip install --upgrade pip openmim
mim install "mmengine>=0.9,<1.0"
mim install "mmcv>=2.0.1,<2.2.0"
mim install "mmdet>=3.1.0,<3.4.0"
python -m pip install "mmpose==1.3.2" "mmpretrain>=1.0.0"

echo "ViTPose-B is selected by the official MMPose alias: vitpose-b"
