#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PANOPTIC_PYTHON:-/Users/zjendex/.conda/envs/GAN/bin/python}"
sequence_dir="${1:-${repo_root}/data/cmu_panoptic/160906_band1}"
output_dir="${2:-${repo_root}/artifacts/cmu_panoptic_10v3/oracle_noise_pilot}"

export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${python_bin}" -m panoptic_10v3.cli run-study \
  --sequence-dir "${sequence_dir}" \
  --output-dir "${output_dir}" \
  --backend oracle-noise \
  --stride 5 \
  --max-frames 120 \
  --triplet-max-frames 60
