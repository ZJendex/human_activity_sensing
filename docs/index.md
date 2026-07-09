# Radar Human Activity Dataset Notes

This MkDocs site collects the Markdown notes in this repository so they can be rendered cleanly on GitHub Pages.

## Documents

- [Dataset Links](radar_human_activity_sensing_papers.md): radar/RF human sensing datasets with verified public dataset links and recording metadata where available.
- [Papers by Impact](radar_human_sensing_papers_by_impact.md): impact-ranked radar/RF human sensing papers, including RF-Pose, mmWave pose, mesh, activity, gesture, and gait-related work.
- [Pose Metrics Reference](human_pose_metrics_reference.md): plain-language explanations and formulas for ML/CV/radar pose metrics and biomedical/clinical biomechanics metrics.

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
