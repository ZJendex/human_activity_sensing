# human_activity_sensing

## CMU Panoptic 10-view versus 3-view skeleton study

The repository includes a runnable RGB-only multi-view pipeline for the
`160906_band1` pilot:

```bash
PANOPTIC_PYTHON=/Users/zjendex/.conda/envs/GAN/bin/python \
  scripts/run_cmu_panoptic_10v3.sh
```

It builds a universal-time frame table, caches M1 2D joints, performs calibrated
M2 association and robust M3 triangulation, compares V10 with the fixed
V3-balanced set, evaluates all 120 three-camera subsets, and exports HTML/PNG/MP4
visualizations. The default `oracle-noise` backend is a geometry-control test;
use `scripts/install_vitpose_env.sh` with `--backend vitpose` for MMPose, or
`scripts/install_hf_vitpose_env.sh` with `--backend hf-vitpose` for the
cross-platform Transformers implementation. The real 20-frame pilot also
exports evaluation-only M1/M2 diagnostics, all 120 camera-triplet rankings,
artifact checksums, and a cache-safe `--reuse-m1` workflow.

For the primary full-window V10-versus-V3 experiment on an M4 Pro:

```bash
scripts/run_cmu_panoptic_full59s_mps.sh
```

This runner uses resumable FP32 RT-DETR + ViTPose inference on full MPS with
staged fallback, then adds synchronized RGB-colored Kinect point clouds strictly
as an evaluation-only surface reference. The locked three-panel HTML and MP4
show the same cloud behind GT, V10, and V3, with a near-body default and a
full-scene toggle in HTML. The 59.1-second GT candidate window yields 575 frames
over 57.56 seconds after requiring all ten RGB views to pass the ±30 ms sync gate.
Set `PANOPTIC_MAX_FRAMES=300` to apply the same frozen protocol to a shorter
held-out sequence. The handbook records completed motion-range
(`171026_pose3`) and close-interaction (`160224_haggling1`) checks.

Full protocol: [CMU Panoptic 10 vs 3 study](docs/cmu_panoptic_10_vs_3_camera_study.md).
