#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PANOPTIC_PYTHON:-${repo_root}/.venv-vitpose/bin/python}"
sequence_dir="${1:-${repo_root}/data/cmu_panoptic/160906_band1}"
output_dir="${2:-${repo_root}/artifacts/cmu_panoptic_10v3/hf_vitpose_full59s_10hz}"
max_frames="${PANOPTIC_MAX_FRAMES:-591}"
depth_filename="${PANOPTIC_DEPTH_FILENAME:-depthdata.dat}"

mkdir -p "${output_dir}/.cache/matplotlib" "${output_dir}/.cache/xdg"

export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${repo_root}/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export MPLCONFIGDIR="${output_dir}/.cache/matplotlib"
export XDG_CACHE_HOME="${output_dir}/.cache/xdg"

"${python_bin}" -m panoptic_10v3.cli run-study \
  --sequence-dir "${sequence_dir}" \
  --output-dir "${output_dir}" \
  --backend hf-vitpose \
  --device full-mps \
  --resume-m1 \
  --stride 3 \
  --max-frames "${max_frames}" \
  --skip-triplets \
  --viewer-max-frames "${max_frames}" \
  --skip-video

"${python_bin}" -m panoptic_10v3.cli depth-cache \
  --sequence-dir "${sequence_dir}" \
  --frame-table "${output_dir}/frame_table.jsonl" \
  --output-dir "${output_dir}/depth_rgbd" \
  --sample-step 4 \
  --voxel-size-cm 2 \
  --max-points 20000 \
  --minimum-nodes 6 \
  --depth-filename "${depth_filename}"

"${python_bin}" -m panoptic_10v3.cli visualize \
  --sequence-dir "${sequence_dir}" \
  --frame-table "${output_dir}/frame_table.jsonl" \
  --v10 "${output_dir}/m3_v10.jsonl" \
  --v3 "${output_dir}/m3_v3_balanced.jsonl" \
  --eval-v10 "${output_dir}/evaluation_v10" \
  --eval-v3 "${output_dir}/evaluation_v3" \
  --eval-m1 "${output_dir}/evaluation_m1/summary.json" \
  --eval-m2-v10 "${output_dir}/evaluation_m2_v10.json" \
  --eval-m2-v3 "${output_dir}/evaluation_m2_v3.json" \
  --cloud-index "${output_dir}/depth_rgbd/index.jsonl" \
  --cloud-point-limit 5000 \
  --near-body-distance-cm 35 \
  --output-dir "${output_dir}/visuals" \
  --max-frames "${max_frames}" \
  --video-fps 10
