#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Download the CMU Panoptic/Kinoptic files used by the 10-view vs 3-view study.

Usage:
  scripts/download_cmu_panoptic_study.sh [options]

Options:
  --sequence NAME   Sequence to download (default: 160906_band1)
  --scope SCOPE     metadata | rgb | depth | all (default: metadata)
  --output DIR      Dataset root (default: data/cmu_panoptic)
  --jobs N          Parallel downloads, in batches (default: 4)
  --base-url URL    Dataset endpoint
  -h, --help        Show this help

Scopes:
  metadata  Calibration, sync tables, and official COCO19 3D skeletons.
  rgb       metadata plus all ten Kinect RGB videos.
  depth     metadata plus all ten Kinect depth streams.
  all       metadata, all ten RGB videos, and all ten depth streams.

Files are first written as *.part and are resumable. Completed files are moved
to their final names. The default endpoint is the official CMU data server.
EOF
}

sequence="160906_band1"
scope="metadata"
output_root="data/cmu_panoptic"
jobs=4
base_url="http://domedb.perception.cs.cmu.edu"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sequence)
      sequence="$2"
      shift 2
      ;;
    --scope)
      scope="$2"
      shift 2
      ;;
    --output)
      output_root="$2"
      shift 2
      ;;
    --jobs)
      jobs="$2"
      shift 2
      ;;
    --base-url)
      base_url="${2%/}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$scope" in
  metadata|rgb|depth|all) ;;
  *)
    echo "Invalid --scope: $scope" >&2
    exit 2
    ;;
esac

if ! [[ "$jobs" =~ ^[1-9][0-9]*$ ]]; then
  echo "--jobs must be a positive integer" >&2
  exit 2
fi

dataset_dir="$output_root/$sequence"
mkdir -p "$dataset_dir"

fetch() {
  local relative_path="$1"
  local url="$2"
  local destination="$dataset_dir/$relative_path"
  local partial="$destination.part"
  local attempt=1
  local max_attempts=20
  local remote_size=""
  local local_size=""

  mkdir -p "$(dirname "$destination")"
  if [[ -s "$destination" ]]; then
    echo "[skip] $relative_path"
    return 0
  fi

  echo "[get ] $relative_path"
  remote_size=$(
    curl --fail --head --location --silent --show-error "$url" |
      awk 'BEGIN { IGNORECASE=1 } /^Content-Length:/ { gsub("\r", ""); size=$2 } END { print size }'
  )

  while true; do
    if [[ -n "$remote_size" && -f "$partial" ]]; then
      local_size=$(wc -c < "$partial" | tr -d ' ')
      if [[ "$local_size" == "$remote_size" ]]; then
        break
      fi
    fi

    if curl \
      --fail \
      --location \
      --silent \
      --show-error \
      --connect-timeout 30 \
      --continue-at - \
      --output "$partial" \
      "$url"; then
      break
    fi

    if [[ "$attempt" -ge "$max_attempts" ]]; then
      echo "[fail] $relative_path after $attempt attempts" >&2
      return 1
    fi

    echo "[retry $attempt/$max_attempts] $relative_path" >&2
    attempt=$((attempt + 1))
    sleep 2
  done

  if [[ -n "$remote_size" ]]; then
    local_size=$(wc -c < "$partial" | tr -d ' ')
    if [[ "$local_size" != "$remote_size" ]]; then
      echo "[fail] size mismatch for $relative_path: local=$local_size remote=$remote_size" >&2
      return 1
    fi
  fi

  mv "$partial" "$destination"
  echo "[done] $relative_path"
}

run_batch() {
  local batch_pids=""
  local batch_count=0
  local item
  local failed=0

  for item in "$@"; do
    local relative_path="${item%%|*}"
    local url="${item#*|}"
    fetch "$relative_path" "$url" &
    batch_pids="$batch_pids $!"
    batch_count=$((batch_count + 1))

    if [[ "$batch_count" -ge "$jobs" ]]; then
      for pid in $batch_pids; do
        wait "$pid" || failed=1
      done
      batch_pids=""
      batch_count=0
    fi
  done

  if [[ -n "$batch_pids" ]]; then
    for pid in $batch_pids; do
      wait "$pid" || failed=1
    done
  fi

  if [[ "$failed" -ne 0 ]]; then
    echo "One or more downloads failed; rerun the same command to resume." >&2
    exit 1
  fi
}

dataset_base="$base_url/webdata/dataset/$sequence"
metadata_items=(
  "calibration_${sequence}.json|$dataset_base/calibration_${sequence}.json"
  "kcalibration_${sequence}.json|$dataset_base/kinect_shared_depth/kcalibration_${sequence}.json"
  "synctables_${sequence}.json|$dataset_base/kinect_shared_depth/synctables.json"
  "ksynctables_${sequence}.json|$dataset_base/kinect_shared_depth/ksynctables.json"
  "hdPose3d_stage1_coco19.tar|$dataset_base/hdPose3d_stage1_coco19.tar"
)

run_batch "${metadata_items[@]}"

pose_archive="$dataset_dir/hdPose3d_stage1_coco19.tar"
pose_directory="$dataset_dir/hdPose3d_stage1_coco19"
if [[ -f "$pose_archive" && ! -d "$pose_directory" ]]; then
  echo "[unpack] hdPose3d_stage1_coco19.tar"
  tar -xf "$pose_archive" -C "$dataset_dir"
  echo "[done] hdPose3d_stage1_coco19/"
fi

if [[ "$scope" == "rgb" || "$scope" == "all" ]]; then
  rgb_items=()
  for camera_id in $(seq 1 10); do
    relative_path=$(printf "kinectVideos/kinect_50_%02d.mp4" "$camera_id")
    url="$dataset_base/videos/kinect_shared_crf20/${sequence}_kinect${camera_id}.mp4"
    rgb_items+=("$relative_path|$url")
  done
  run_batch "${rgb_items[@]}"
fi

if [[ "$scope" == "depth" || "$scope" == "all" ]]; then
  depth_items=()
  for camera_id in $(seq 1 10); do
    relative_path=$(printf "kinect_shared_depth/KINECTNODE%d/depthdata.dat" "$camera_id")
    url="$dataset_base/$relative_path"
    depth_items+=("$relative_path|$url")
  done
  run_batch "${depth_items[@]}"
fi

echo
echo "Download scope '$scope' is complete:"
echo "  $dataset_dir"
du -sh "$dataset_dir"
for entry in "$dataset_dir"/*; do
  if [[ -d "$entry" ]]; then
    file_count=$(find "$entry" -type f ! -name '*.part' | wc -l | tr -d ' ')
    echo "  $(basename "$entry")/ ($file_count files)"
  elif [[ -f "$entry" && "$entry" != *.part ]]; then
    echo "  $(basename "$entry")"
  fi
done
