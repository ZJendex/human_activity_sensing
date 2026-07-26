# Radar Human Activity Dataset Notes

This MkDocs site collects the Markdown notes in this repository so they can be rendered cleanly on GitHub Pages.

## Documents

- [Dataset Links](radar_human_activity_sensing_papers.md): radar/RF human sensing datasets with verified public dataset links and recording metadata where available.
- [Papers by Impact](radar_human_sensing_papers_by_impact.md): impact-ranked radar/RF human sensing papers, including RF-Pose, mmWave pose, mesh, activity, gesture, and gait-related work.
- [Accuracy Summary](selected_work_accuracy.md): source-verified pose and mesh results for RF-Pose, mmMesh, mRI, MM-Fi, MMVR, Argus, MVDoppler-Pose, and M4Human.
- [Model Architecture Comparison](model_architecture_comparison.md): visual, M4Human-centered explanation of how the eight works represent radar, encode space and time, fuse views, and produce 2D pose, 3D joints, or meshes.
- [Pose Metrics Reference](human_pose_metrics_reference.md): plain-language explanations and formulas for ML/CV/radar pose metrics and biomedical/clinical biomechanics metrics.
- [Three-ZED-X Human Mesh Roadmap](zedx_multiview_human_mesh_roadmap.md): indexed replay and mandatory rectification of three 3840×1200 side-by-side H.265 recordings carrying six unrectified eye views, followed by multi-view 2D/3D pose, trajectory and persistent tracking, SMPL fitting, and a stable world-coordinate mesh.
- [CMU Panoptic 10-vs-3 Camera Study](cmu_panoptic_10_vs_3_camera_study.md): controlled RGB-only ViTPose, cross-view association, and triangulation study measuring the skeleton-accuracy difference between ten Kinect color views and a pre-registered three-camera subset, with depth isolated to point-cloud evaluation.

## Local Preview

Install MkDocs and run the local server from the repository root:

```bash
pip install -r requirements.txt
mkdocs serve
```

Build the static site:

```bash
mkdocs build --strict
```

The generated HTML site will be written to `site/`.
