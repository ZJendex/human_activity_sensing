#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="${repo_root}/.venv-vitpose"

uv venv --python 3.11 "${env_dir}"
uv pip install --python "${env_dir}/bin/python" \
  "numpy<2" \
  "scipy>=1.10" \
  "opencv-python>=4.8" \
  "matplotlib>=3.7" \
  "PyYAML>=6" \
  "torch>=2.2" \
  "torchvision>=0.17" \
  "transformers>=4.48,<6" \
  "Pillow>=10"

echo "Environment ready: ${env_dir}"
echo "Model weights download on the first --backend hf-vitpose run."
