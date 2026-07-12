# Accuracy Summary: Selected Radar Pose and Mesh Works

Last updated: 2026-07-12.

## Technical Summary

There is no defensible single accuracy ranking across these works. They solve different tasks and report different metrics: 2D keypoint AP, 3D joint error, mesh vertex error, rotation error, or temporal consistency. The most useful comparison is therefore **within each paper's task and evaluation protocol**.

- **RF-Pose** reports 2D pose AP of **62.4 in visible scenes** and **58.1 through walls**, demonstrating a relatively small drop in its own through-wall test.
- **mmMesh** reports **2.47 cm average vertex error** in its basic scenario, increasing to **5.93-6.45 cm** behind tested barriers.
- **mRI, MM-Fi, MMVR, and M4Human** show that unseen-subject, unseen-environment, or unseen-action splits are materially harder than random splits.
- **MVDoppler-Pose** reports **60.96 mm MPJPE**, **93.24% PCK@150 mm**, and temporal correlation **0.53** on its subject-independent evaluation; its multi-view model remains near **58-63 mm MPJPE** across the paper's self-occluded distance bands.
- **Argus** reports **6.5 cm average vertex error** using two compact wearable radars, but it reconstructs pose in the wearer's local frame and does not evaluate global location or orientation.

!!! warning "Do not compare unlike headline numbers"
    AP is higher-better, while MPJPE, MVE, joint error, rotation error, and translation error are lower-better. Even two values with the same metric name are not automatically comparable when datasets, joints, alignment, subjects, actions, or test splits differ.

## Headline Results at a Glance

| Work | Primary output | Headline reported result | Harder evaluation evidence | What the number means |
|---|---|---|---|---|
| **RF-Pose** (CVPR 2018) | 2D multi-person keypoints | Visible: AP **62.4**, AP50 **93.3**, AP75 **70.7** | Through-wall: AP **58.1**, AP50 **85.0**, AP75 **66.1** | OKS-based keypoint detection; higher is better |
| **mmMesh** (MobiSys 2021) | 3D SMPL body mesh | Basic: vertex **2.47 cm**, joint **2.18 cm**, rotation **3.80°**, root translation **1.27 cm** | Barriers: vertex **5.93-6.45 cm**; joint **5.54-6.06 cm** | Corresponding mesh/joint errors; lower is better |
| **mRI / MARS-style baseline** (NeurIPS 2022) | 3D skeleton from mmWave point clouds | Random/all 12 movements: MPJPE **163.3 ± 9.1 mm**, PA-MPJPE **94.1 ± 3.6 mm** | Cross-subject/all movements: **186.6 ± 23.8 mm**, **97.3 ± 7.8 mm** | Root-aligned and Procrustes-aligned joint error; lower is better |
| **MM-Fi / IWR6843AOP** (NeurIPS 2023) | 3D skeleton from mmWave point clouds | Random/all actions: MPJPE **117.0 ± 3.7 mm**, PA-MPJPE **57.3 ± 1.8 mm** | Cross-environment/all actions: **161.6 ± 1.8 mm**, **73.7 ± 0.6 mm** | Root-aligned and Procrustes-aligned joint error; lower is better |
| **MMVR** (ECCV 2024) | 2D keypoints from multi-view radar heatmaps | Open single-person/random: AP **46.24**, AP50 **62.88**, AP75 **47.45** | Cluttered multi-person/cross-environment: AP **7.11**, AP50 **11.98**, AP75 **6.76** | OKS-based keypoint detection; higher is better |
| **Argus** (SenSys 2025) | Egocentric 3D SMPL mesh | Vertex **6.5 cm**, joint **5.0 cm**, rotation **6.4°** | Unseen users without fine-tuning: vertex **7.0 cm**, joint **5.5 cm**, rotation **6.9°** | Local-frame mesh, skeleton, and rotation error; lower is better |
| **MVDoppler-Pose** (CVPR 2025) | 3D walking skeleton | MPJPE **60.96 mm**, PCK@150 mm **93.24%**, temporal correlation **0.53** | Self-occluded distance bands: MPJPE **63.27 / 59.11 / 58.24 mm** | Spatial error, threshold accuracy, and temporal correlation |
| **M4Human / RT-Mesh** (CVPR 2026) | Global 3D SMPL-X mesh from raw radar tensor | Random/all actions: MVE **90.9 mm**, MJE **77.2 mm** | Cross-subject: **135.1 / 120.2 mm**; cross-action: **143.1 / 122.0 mm** | World-frame vertex/joint error without root or Procrustes alignment; lower is better |

## Generalization Is the Main Accuracy Gap

The clearest shared finding is not a winning method; it is the cost of realistic evaluation. MM-Fi's all-action mmWave MPJPE rises from **117.0 mm** on a random split to **161.6 mm** across environments. MMVR pose AP falls from **46.24** in its open, single-person random split to **7.11** in its cluttered, multi-person cross-environment split. M4Human RT-Mesh MVE rises from **90.9 mm** on the random split to **135.1 mm** across subjects and **143.1 mm** across actions.

These changes are not directly comparable percentages, but they consistently show that random splits can substantially overstate deployment readiness. For home monitoring or clinical work, cross-subject and cross-environment results should be treated as the more relevant evidence.

## Verified Results by Work

### RF-Pose — CVPR 2018

**Task and input.** RF-Pose estimates 14-keypoint 2D multi-person poses from low-power FMCW radio heatmaps. Training uses synchronized vision as cross-modal supervision; inference uses radio only.

**Reported pose results.** The paper evaluates 1,000 manually annotated visible-scene images and 1,000 through-wall examples using COCO-style OKS AP.

| Setting | AP | AP50 | AP75 |
|---|---:|---:|---:|
| Visible scenes | 62.4 | 93.3 | 70.7 |
| Through walls | 58.1 | 85.0 | 66.1 |

The paper also reports person identification from two-second RF-skeleton clips: top-1 accuracy is **83.4% visible** and **84.4% through walls** for 100 people. That is an identity-classification result, not a pose-accuracy result.

**Interpretation.** RF-Pose is foundational evidence for through-wall 2D pose. Its AP75 trails its AP50 because radio's lower spatial resolution makes strict keypoint localization harder.

Source: [CVPR paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhao_Through-Wall_Human_Pose_CVPR_2018_paper.pdf).

### mmMesh — MobiSys 2021

**Task and input.** mmMesh reconstructs a dynamic 3D SMPL mesh from TI AWR1843 mmWave point clouds, with VICON-derived meshes as ground truth.

| Scenario | Vertex error (cm) | Joint error (cm) | Rotation error (°) | Root translation (cm) |
|---|---:|---:|---:|---:|
| Basic | 2.47 | 2.18 | 3.80 | 1.27 |
| Foam-box barrier | 5.93 | 5.54 | 8.35 | 3.88 |
| Cloth-screen barrier | 6.33 | 5.87 | 8.88 | 3.88 |
| Bamboo-panel barrier | 6.45 | 6.06 | 8.67 | 4.57 |

**Interpretation.** The basic result is strong for its controlled dataset, but the barrier experiments approximately double the vertex and joint errors. These are mesh-correspondence metrics on mmMesh's own subjects, activities, and sensor geometry.

Source: [MobiSys paper](https://havocfixer.github.io/resource/21_MobiSys.pdf).

### mRI with the MARS-Style mmWave Baseline — NeurIPS 2022

**Task and input.** mRI benchmarks rehabilitation-focused 3D pose estimation from a TI IWR1443BOOST point cloud. The mmWave benchmark follows the MARS processing/model pipeline and predicts 17 joints.

| Split | Movement protocol | MPJPE (mm) | PA-MPJPE (mm) |
|---|---|---:|---:|
| Random (S1) | P1: all 12 movements | 163.3 ± 9.1 | 94.1 ± 3.6 |
| Random (S1) | P2: 10 fixed rehabilitation movements | 125.1 ± 2.4 | 74.1 ± 1.0 |
| Cross-subject (S2) | P1: all 12 movements | 186.6 ± 23.8 | 97.3 ± 7.8 |
| Cross-subject (S2) | P2: 10 fixed rehabilitation movements | 126.6 ± 11.3 | 75.0 ± 7.1 |

**Interpretation.** Free-form stretching and walking in P1 make the benchmark harder than the fixed rehabilitation set in P2. MPJPE is pelvis-aligned; PA-MPJPE additionally removes translation, rotation, and scale, so the aligned number should not be read as absolute room-localization accuracy.

Sources: [project page](https://sizhean.github.io/mri), [paper](https://arxiv.org/abs/2210.08394).

### MM-Fi with TI IWR6843AOP — NeurIPS 2023

**Hardware correction.** The paper specifies a **TI IWR6843AOP** operating at 60-64 GHz, not AWR1843AOP.

For the all-action protocol (P3), the single-mmWave baseline reports:

| Split | MPJPE (mm) | PA-MPJPE (mm) |
|---|---:|---:|
| Random (S1) | 117.0 ± 3.7 | 57.3 ± 1.8 |
| Cross-subject (S2) | 129.7 ± 2.2 | 60.0 ± 1.7 |
| Cross-environment (S3) | 161.6 ± 1.8 | 73.7 ± 0.6 |

**Interpretation.** The cross-environment split produces the largest degradation, consistent with radar multipath and background changes. MM-Fi reports mean and standard deviation across three runs and uses both root-aligned MPJPE and fully aligned PA-MPJPE.

Sources: [project page](https://ntu-aiot-lab.github.io/mm-fi), [NeurIPS paper](https://arxiv.org/pdf/2305.10345).

### MMVR — ECCV 2024

**Task and input.** MMVR benchmarks 2D pose from multi-view high-resolution radar heatmaps. P1 is an open-space single-person protocol; P2 is a multi-person protocol in cluttered rooms. S1 is random, while S2 includes an unseen environment.

| Protocol | Split | AP | AP50 | AP75 |
|---|---|---:|---:|---:|
| P1: open, single person | S1: random | 46.24 | 62.88 | 47.45 |
| P1: open, single person | S2: cross-session/environment | 29.82 | 43.03 | 30.29 |
| P2: cluttered, multiple people | S1: random | 32.13 | 44.22 | 32.58 |
| P2: cluttered, multiple people | S2: cross-session/environment | 7.11 | 11.98 | 6.76 |

**Interpretation.** MMVR's most realistic combination - clutter, multiple people, and an unseen environment - is also its weakest baseline. This is important negative evidence: the dataset contribution is stronger than the baseline's deployment readiness.

Sources: [ECCV paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10205.pdf), [dataset record](https://doi.org/10.5281/zenodo.12611978).

### Argus — SenSys 2025

**Task and input.** Argus reconstructs an egocentric SMPL mesh from a pair of wearable BGT60TR13C radars. Each radar uses one transmitter and three receivers.

| Model / evaluation | Vertex error (cm) | Joint error (cm) | Rotation error (°) |
|---|---:|---:|---:|
| Argus with KAN fusion | 6.5 | 5.0 | 6.4 |
| Unseen users, no fine-tuning | 7.0 | 5.5 | 6.9 |
| Unseen users, 30% fine-tuning data | 6.2 | 5.2 | 6.1 |

**Interpretation.** The result is notable because each wearable radar is only about 7.5 g and 0.35 W. However, the reference meshes are monocular-camera HMR 2.0 pseudo-labels rather than marker-based MoCap ground truth. Argus predicts in a coordinate system that moves with the wearer; the paper explicitly omits global-location and global-orientation error. Activities above ear level are also outside the validated scope.

Sources: [ACM DOI](https://doi.org/10.1145/3715014.3722045), [author-hosted paper](https://raphaelduan.github.io/papers/Argus_sensys25.pdf).

### MVDoppler-Pose — CVPR 2025

**Task and input.** MVDoppler-Pose fuses range/position and Doppler/motion information from two cross-view FMCW radars for 3D walking-pose estimation. The subject-independent split trains on 10 people and tests on 3 unseen people.

| Evaluation | MPJPE (mm) | PCK@150 mm (%) | Temporal correlation ρ |
|---|---:|---:|---:|
| Overall multi-view, multi-modal model | 60.96 | 93.24 | 0.53 |

For self-occluded walking, the multi-view, multi-modal model reports:

| Target distance | MPJPE (mm) | Temporal correlation ρ |
|---|---:|---:|
| Under 8.5 m | 63.27 | 0.53 |
| 8.5-11.5 m | 59.11 | 0.53 |
| Over 11.5 m | 58.24 | 0.54 |

**Interpretation.** Within this dataset, multi-view fusion removes much of the direction and distance sensitivity seen in single-view radar. The paper's limitations are equally important: evaluation is single-person, minimally cluttered, and restricted to a relatively narrow set of walking activities.

Sources: [project page](https://mvdoppler-pose.github.io/), [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Choi_MVDoppler-Pose_Multi-Modal_Multi-View_mmWave_Sensing_for_Long-Distance_Self-Occluded_Human_Walking_CVPR_2025_paper.pdf).

### M4Human — CVPR 2026

**Task and input.** M4Human benchmarks global SMPL-X mesh reconstruction across 20 subjects, 50 actions, and 661k frames. RT-Mesh operates on raw radar tensors (RT); the benchmark also includes processed radar point clouds (RPC).

| RT-Mesh, all actions | MVE (mm) | MJE (mm) | MRE (°) | Translation error (mm) |
|---|---:|---:|---:|---:|
| S1: random | 90.9 | 77.2 | 9.6 | 47.6 |
| S2: cross-subject | 135.1 | 120.2 | 14.9 | 86.1 |
| S3: cross-action | 143.1 | 122.0 | 15.6 | 76.0 |

**Interpretation.** These metrics are computed in the world frame without root or Procrustes alignment, making them stricter than aligned pose errors. Raw tensors and point clouds perform similarly on the random split, but RT-Mesh generalizes better on unseen subjects and actions. The large S2/S3 gap shows that high-fidelity unconstrained motion remains difficult.

Sources: [project page](https://fanjunqiao.github.io/M4Human-site/), [CVPR paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Fan_M4Human_A_Large-Scale_Multimodal_mmWave_Radar_Benchmark_for_Human_Mesh_CVPR_2026_paper.pdf).

## Metric and Protocol Definitions

- **AP / AP50 / AP75:** OKS-based average precision for 2D keypoints. AP averages multiple thresholds; AP50 is looser than AP75. Higher is better.
- **MPJPE / MJE:** mean 3D joint distance. Check whether the pelvis/root is aligned before comparing values. Lower is better.
- **PA-MPJPE:** MPJPE after Procrustes translation, rotation, and scale alignment. It measures relative pose shape more than absolute placement. Lower is better.
- **MVE / vertex error:** average distance between corresponding predicted and ground-truth mesh vertices. Lower is better.
- **Rotation error:** angular error between predicted and reference joint rotations. Lower is better.
- **PCK@150 mm:** percentage of joints within 150 mm of the reference. Higher is better.
- **Temporal correlation ρ:** correlation between predicted and reference joint-motion patterns over time. Higher is better.
- **Random split:** samples from the same subject/environment/action populations may occur in train and test.
- **Cross-subject/environment/action split:** the held-out test set contains a factor not represented during training and is usually more relevant to deployment.

For full formulas and interpretation, see the [Pose Metrics Reference](human_pose_metrics_reference.md).

## How This Summary Was Built

The values above were transcribed from the primary paper tables or official project pages. The summary prioritizes radar-only results and each paper's most representative harder split. Values were not normalized or re-computed because the original predictions and identical ground-truth definitions are not available across all eight works.

A cross-paper bar chart is intentionally omitted: placing AP, aligned joint error, world-frame mesh error, and local-frame mesh error on one axis would imply a common scale that does not exist.

## Limitations and Robustness Notes

- The works use different keypoint sets, mesh models, sensors, ranges, subject populations, actions, and ground-truth systems.
- RF-Pose and MMVR report 2D OKS AP; the other pose works primarily report 3D distance errors.
- mmMesh and Argus use SMPL, while M4Human uses SMPL-X and evaluates global trajectories in the world frame.
- Argus evaluates against monocular vision-derived pseudo-labels, whereas mmMesh and M4Human use marker-based motion-capture references.
- mRI and MM-Fi report aligned metrics; M4Human explicitly reports unaligned world-frame metrics.
- “Through wall,” “barrier,” “self-occluded,” and “cross-environment” are different conditions and should not be treated as one shared occlusion benchmark.
- Dataset papers report baseline performance, not an upper bound on what the dataset can support.

## Recommended Next Steps

1. For a fair model comparison, choose one public dataset and one fixed split, then re-train candidate methods under the same input representation and joint definition.
2. For home or clinical deployment, prioritize cross-subject and cross-environment results over random-split headline numbers.
3. Report both relative-pose quality and absolute room localization; an aligned skeleton can look accurate while being misplaced globally.
4. Add per-action, per-joint, and distance-stratified errors when selecting a method for gait, rehabilitation, or fall-risk work.

## Further Questions

- Which of these datasets best matches the intended home layout, radar placement, subject age, and mobility tasks?
- Is the application primarily 2D pose, 3D skeleton, global tracking, or full mesh reconstruction?
- What error is clinically tolerable for gait events, joint angles, turning, or sit-to-stand measurements?
- Can the selected method maintain accuracy over unseen rooms, clothing, assistive devices, multiple people, and slow or impaired movement?
