# Human Pose Metrics Reference

Last updated: 2026-07-09.

This note explains common human-pose metrics from two viewpoints:

1. ML / computer vision / radar pose-estimation papers.
2. Biomedical, clinical, and biomechanics studies.

The same word can mean different things across fields. In ML, a "good pose" often means the predicted keypoints are close to annotation labels. In biomechanics, a "good measurement" often means the system is valid, reliable, clinically interpretable, and close enough to a motion-capture, force-plate, or clinician reference.

## Notation

- `N`: number of samples or frames.
- `K`: number of keypoints, joints, or landmarks.
- `p[n,k]`: predicted position of joint `k` in sample `n`.
- `g[n,k]`: ground-truth/reference position.
- `m[n,k]`: visibility or validity mask, where `1` means the joint should be evaluated.
- `||x||_2`: Euclidean distance.
- `D`: coordinate dimension, usually 2 for image keypoints or 3 for 3D pose.
- `S`: normalization scale, such as person bounding-box size, head size, torso length, or subject height.

Lower-better metrics are distance or error metrics. Higher-better metrics are usually accuracy, AP, PCK, AUC, F1, ICC, or correlation.

## 1. ML / CV / Radar Pose Metrics

These metrics are used when a model predicts 2D keypoints, 3D joints, meshes, tracks, or radar-derived skeletons.

### Direct Keypoint Distance Metrics

**EPE / endpoint error**

Intuition: "How many pixels, centimeters, or millimeters away is the predicted joint from the label?"

Calculation:

<div class="math-display">
\[
\begin{aligned}
\operatorname{EPE}_{n,k} &= \lVert \mathbf{p}_{n,k} - \mathbf{g}_{n,k} \rVert_2, \\
\operatorname{mean\ EPE} &=
\frac{\sum_{n,k} m_{n,k}\operatorname{EPE}_{n,k}}
     {\sum_{n,k} m_{n,k}}.
\end{aligned}
\]
</div>

Use it for 2D image landmarks, 3D joints, radar-localized joints, and any single-point prediction. Lower is better. MMPose describes EPE as the endpoint error of keypoints.

**MAE / mean absolute error**

Intuition: "Average absolute coordinate error per axis."

Calculation:

<div class="math-display">
\[
\operatorname{MAE} = \frac{1}{D}\sum_{d=1}^{D}\left|p_d-g_d\right|
\]
</div>

Use it when axis-specific errors matter, e.g. radar range error versus azimuth error. Lower is better. It is less sensitive to outliers than RMSE.

**RMSE / root mean squared error**

Intuition: "Average error with a stronger penalty for large mistakes."

Calculation:

<div class="math-display">
\[
\operatorname{RMSE} =
\sqrt{\frac{\sum_{n,k}m_{n,k}\lVert\mathbf{p}_{n,k}-\mathbf{g}_{n,k}\rVert_2^2}
{\sum_{n,k}m_{n,k}}}
\]
</div>

Use it when large errors should count disproportionately, such as clinical motion tracking where rare large errors may be unsafe. Lower is better.

**NME / normalized mean error**

Intuition: "How large is the error relative to body size or face/head scale?"

Calculation:

<div class="math-display">
\[
\operatorname{NME} =
\frac{\sum_{n,k}m_{n,k}\dfrac{\lVert\mathbf{p}_{n,k}-\mathbf{g}_{n,k}\rVert_2}{S_n}}
{\sum_{n,k}m_{n,k}}
\]
</div>

Common normalizers include head size, bounding-box size, torso length, inter-ocular distance, or subject height. Lower is better. Normalization makes a 10-pixel error on a small person count more than a 10-pixel error on a large person.

### Threshold-Based Keypoint Metrics

**PCK / Percentage of Correct Keypoints**

Intuition: "What fraction of joints are close enough?"

Calculation:

<div class="math-display">
\[
\begin{aligned}
c_{n,k}(\tau) &= \mathbb{1}\!\left[
\frac{\lVert\mathbf{p}_{n,k}-\mathbf{g}_{n,k}\rVert_2}{S_n}\leq\tau
\right], \\
\operatorname{PCK}@\tau &=
\frac{\sum_{n,k}m_{n,k}c_{n,k}(\tau)}{\sum_{n,k}m_{n,k}}.
\end{aligned}
\]
</div>

Examples:

- `PCK@0.2`: joint is correct if normalized error is at most 20% of the chosen body scale.
- `PCKh`: head-normalized PCK, common in MPII-style 2D pose.
- `tPCK`: torso-normalized PCK.
- `3DPCK`: 3D joint is correct if Euclidean error is below a fixed distance threshold, e.g. 150 mm.

Higher is better. MMPose defines PCK as localization accuracy after normalizing keypoint distance by a scale such as bounding box, head, or torso.

**AUC of PCK**

Intuition: "How good is the pose estimator across many strict-to-loose thresholds?"

Calculation:

```text
1. Choose thresholds t_1 ... t_T.
2. Compute PCK@t for each threshold.
3. Integrate or average the PCK-threshold curve.
```

Higher is better. AUC is less arbitrary than reporting one threshold, because it rewards methods that work under both strict and loose tolerances. MMPose describes AUC as the area under the PCK accuracy curve.

**PCP / Percentage of Correct Parts**

Intuition: "Is an entire limb segment correct, not just one joint?"

Calculation:

<div class="math-display">
\[
\begin{aligned}
c_{a,b} &= \mathbb{1}\!\left[e_a < \alpha L_{a,b}\ \land\ e_b < \alpha L_{a,b}\right], \\
\operatorname{PCP} &= \frac{\text{number of correct limbs}}{\text{number of evaluated limbs}}.
\end{aligned}
\]
</div>

Higher is better. PCP is older and can favor long limbs because the tolerance grows with limb length.

### COCO-Style Detection Metrics

**OKS / Object Keypoint Similarity**

Intuition: "Pose version of IoU: how similar is a predicted skeleton to a ground-truth person after accounting for object scale and keypoint difficulty?"

Calculation, conceptually:

<div class="math-display">
\[
\operatorname{OKS} =
\frac{\sum_i v_i\exp\!\left(-\dfrac{d_i^2}{2s^2\kappa_i^2}\right)}
     {\sum_i v_i}
\]
</div>

where:

- `d_i` is the Euclidean distance for keypoint `i`.
- `s` is object/person scale, derived from area.
- `k_i` (shown as `κ_i` in the equation) is a per-keypoint falloff constant.
- invisible or unlabeled keypoints are excluded.

Higher is better. Small errors on a small person are penalized more than the same pixel error on a large person. COCO's evaluator computes squared keypoint distances, divides by keypoint-specific variances and object area, then averages `exp(-error)` terms over labeled keypoints.

**AP / Average Precision for keypoints**

Intuition: "How good is the detector over confidence thresholds and OKS thresholds?"

Calculation:

```text
1. Sort predicted poses by confidence.
2. Match predictions to ground truth using an OKS threshold.
3. Sweep the confidence threshold to get a precision-recall curve.
4. Average precision over recall.
5. COCO mAP averages AP over OKS thresholds, commonly 0.50:0.05:0.95.
```

Higher is better. AP mixes localization quality, confidence calibration, duplicate detections, missed people, and false positives. It is not just joint distance.

**AR / Average Recall**

Intuition: "How many true people/poses can the method recover if we allow a limited number of predictions?"

Calculation: recall is computed under OKS thresholds and averaged. Higher is better. AR ignores some precision behavior, so it is useful for measuring missed detections.

### 3D Skeleton Metrics

**MPJPE / Mean Per-Joint Position Error**

Intuition: "Average 3D joint distance from the ground truth."

Calculation:

<div class="math-display">
\[
\operatorname{MPJPE} =
\frac{\sum_{n,k}m_{n,k}\lVert\mathbf{p}_{n,k}-\mathbf{g}_{n,k}\rVert_2}
     {\sum_{n,k}m_{n,k}}
\]
</div>

Units are usually millimeters. Lower is better. This is the most common 3D pose metric. MMPose defines MPJPE as mean per-joint position error over keypoints.

**Root-aligned MPJPE**

Intuition: "How good is the body pose after ignoring global body location?"

Calculation:

<div class="math-display">
\[
\begin{aligned}
\widetilde{\mathbf{p}}_{n,k} &= \mathbf{p}_{n,k}-\mathbf{p}_{n,r}, \\
\widetilde{\mathbf{g}}_{n,k} &= \mathbf{g}_{n,k}-\mathbf{g}_{n,r}, \\
\operatorname{MPJPE}_{\text{root}} &=
\frac{\sum_{n,k}m_{n,k}\lVert\widetilde{\mathbf{p}}_{n,k}-\widetilde{\mathbf{g}}_{n,k}\rVert_2}
     {\sum_{n,k}m_{n,k}}.
\end{aligned}
\]
</div>

Lower is better. This removes global translation error and focuses on relative body configuration. For radar papers, report clearly whether global position is included, because radar often cares about both location and pose.

**N-MPJPE / scale-aligned MPJPE**

Intuition: "How good is the pose if the model gets the body shape up to a global scale?"

Calculation:

<div class="math-display">
\[
\begin{aligned}
a^* &= \underset{a}{\arg\min}\ \sum_k
\lVert a\mathbf{p}_k-\mathbf{g}_k\rVert_2^2, \\
\operatorname{N\text{-}MPJPE} &= \frac{1}{K}\sum_k
\lVert a^*\mathbf{p}_k-\mathbf{g}_k\rVert_2.
\end{aligned}
\]
</div>

Lower is better. This forgives one global scale error but does not forgive wrong joint angles or wrong shape.

**PA-MPJPE / P-MPJPE / Procrustes-aligned MPJPE**

Intuition: "How similar is the predicted pose shape after allowing the best rigid alignment?"

Calculation:

<div class="math-display">
\[
\begin{aligned}
(a^*,\mathbf{R}^*,\mathbf{t}^*) &=
\underset{a,\mathbf{R},\mathbf{t}}{\arg\min}\ \sum_k
\lVert a\mathbf{R}\mathbf{p}_k+\mathbf{t}-\mathbf{g}_k\rVert_2^2, \\
\operatorname{PA\text{-}MPJPE} &= \frac{1}{K}\sum_k
\lVert a^*\mathbf{R}^*\mathbf{p}_k+\mathbf{t}^*-\mathbf{g}_k\rVert_2.
\end{aligned}
\]
</div>

Lower is better. This removes global translation, rotation, and scale, so it is easier than raw MPJPE. It is useful for checking pose shape but can hide absolute localization errors.

**MRPE / relative position error**

Intuition: "How wrong is one body part relative to another?"

Calculation:

<div class="math-display">
\[
\begin{aligned}
\Delta\mathbf{p}_{a,b} &= \mathbf{p}_a-\mathbf{p}_b, &
\Delta\mathbf{g}_{a,b} &= \mathbf{g}_a-\mathbf{g}_b, \\
\operatorname{MRPE} &= \operatorname{mean}\!left(
\lVert\Delta\mathbf{p}_{a,b}-\Delta\mathbf{g}_{a,b}\rVert_2\right).
\end{aligned}
\]
</div>

Lower is better. Used in hand pose, multi-body pose, and interaction settings.

**Bone-length error**

Intuition: "Does the predicted skeleton have anatomically plausible limb lengths?"

Calculation:

<div class="math-display">
\[
\begin{aligned}
\widehat{L}_{i,j} &= \lVert\mathbf{p}_i-\mathbf{p}_j\rVert_2, &
L_{i,j} &= \lVert\mathbf{g}_i-\mathbf{g}_j\rVert_2, \\
E_{\text{bone}} &= \frac{1}{|\mathcal{E}|}\sum_{(i,j)\in\mathcal{E}}
\left|\widehat{L}_{i,j}-L_{i,j}\right|.
\end{aligned}
\]
</div>

Lower is better. It catches skeleton stretching even when joint distances look acceptable.

**Joint-angle error**

Intuition: "Are the predicted limb angles correct?"

Calculation for a joint `j` connected to neighbors `a` and `b`:

<div class="math-display">
\[
\begin{aligned}
\widehat{\mathbf{u}} &= \frac{\mathbf{p}_a-\mathbf{p}_j}{\lVert\mathbf{p}_a-\mathbf{p}_j\rVert_2}, &
\widehat{\mathbf{v}} &= \frac{\mathbf{p}_b-\mathbf{p}_j}{\lVert\mathbf{p}_b-\mathbf{p}_j\rVert_2}, \\
\widehat{\theta}_j &= \arccos\!\left(\widehat{\mathbf{u}}^{\mathsf T}\widehat{\mathbf{v}}\right), &
E_{\theta,j} &= \left|\widehat{\theta}_j-\theta_j\right|.
\end{aligned}
\]
</div>

Lower is better. Units are degrees or radians. This is more biomechanically interpretable than raw joint distance.

### Mesh and Body-Shape Metrics

**MPVPE / PVE / V2V error**

Intuition: "How far is each predicted body mesh vertex from the reference mesh vertex?"

Calculation:

<div class="math-display">
\[
\operatorname{MPVPE} = \frac{1}{V}\sum_{v=1}^{V}
\lVert\widehat{\mathbf{x}}_v-\mathbf{x}_v\rVert_2
\]
</div>

Lower is better. Usually reported in millimeters. It evaluates body surface shape, not just skeleton joints.

**PA-MPVPE / Procrustes-aligned vertex error**

Intuition: "How good is the mesh shape after best rigid alignment?"

Calculation: apply the same Procrustes alignment idea as PA-MPJPE, then average vertex distances. Lower is better.

**P2S / point-to-surface distance**

Intuition: "How far are predicted mesh points from the closest point on the reference surface?"

Calculation:

<div class="math-display">
\[
\operatorname{P2S} = \frac{1}{V}\sum_{v=1}^{V}
\min_{\mathbf{y}\in\mathcal{S}_{\mathrm{gt}}}
\lVert\widehat{\mathbf{x}}_v-\mathbf{y}\rVert_2
\]
</div>

Lower is better. Unlike vertex-to-vertex error, it can work when mesh topology or vertex indexing differs.

**Chamfer distance**

Intuition: "How close are two unordered point clouds or surfaces?"

Calculation:

<div class="math-display">
\[
\operatorname{CD}(A,B) =
\frac{1}{|A|}\sum_{\mathbf{a}\in A}\min_{\mathbf{b}\in B}\lVert\mathbf{a}-\mathbf{b}\rVert_2^2
+
\frac{1}{|B|}\sum_{\mathbf{b}\in B}\min_{\mathbf{a}\in A}\lVert\mathbf{b}-\mathbf{a}\rVert_2^2
\]
</div>

Lower is better. Useful for radar point clouds, LiDAR, depth, or meshes when one-to-one vertex correspondence is unavailable.

**SMPL pose/shape parameter error**

Intuition: "Did the model recover the underlying body model parameters?"

Calculation:

- Pose rotation error: geodesic distance between predicted and ground-truth joint rotations.
- Shape error: L2 distance between predicted and ground-truth shape coefficients.

Lower is better. Parameter error is useful for body-model fitting, but vertex and joint errors are usually easier to interpret.

### Temporal Pose Metrics

**Velocity error / MPJVE**

Intuition: "Does the predicted motion move at the right speed?"

Calculation:

<div class="math-display">
\[
\begin{aligned}
\widehat{\mathbf{v}}_{t,k} &= \mathbf{p}_{t,k}-\mathbf{p}_{t-1,k}, &
\mathbf{v}_{t,k} &= \mathbf{g}_{t,k}-\mathbf{g}_{t-1,k}, \\
\operatorname{MPJVE} &= \operatorname{mean}_{t,k}
\lVert\widehat{\mathbf{v}}_{t,k}-\mathbf{v}_{t,k}\rVert_2.
\end{aligned}
\]
</div>

Lower is better. It can reveal motion errors even if framewise pose error is low.

**Acceleration error**

Intuition: "Does the motion accelerate naturally?"

Calculation:

<div class="math-display">
\[
\begin{aligned}
\widehat{\mathbf{a}}_{t,k} &= \mathbf{p}_{t+1,k}-2\mathbf{p}_{t,k}+\mathbf{p}_{t-1,k}, \\
\mathbf{a}_{t,k} &= \mathbf{g}_{t+1,k}-2\mathbf{g}_{t,k}+\mathbf{g}_{t-1,k}, \\
E_{\mathrm{acc}} &= \operatorname{mean}_{t,k}
\lVert\widehat{\mathbf{a}}_{t,k}-\mathbf{a}_{t,k}\rVert_2.
\end{aligned}
\]
</div>

Lower is better. This is common in video pose to penalize jitter.

**Jitter / smoothness**

Intuition: "Is the predicted pose flickering?"

Calculation examples:

<div class="math-display">
\[
\begin{aligned}
\operatorname{jitter} &= \operatorname{mean}_{t,k}
\lVert\mathbf{p}_{t+1,k}-2\mathbf{p}_{t,k}+\mathbf{p}_{t-1,k}\rVert_2, \\
\operatorname{jerk} &= \operatorname{mean}_{t,k}
\lVert\mathbf{p}_{t+2,k}-3\mathbf{p}_{t+1,k}+3\mathbf{p}_{t,k}-\mathbf{p}_{t-1,k}\rVert_2.
\end{aligned}
\]
</div>

Lower is smoother, but smoother is not always more accurate. A method can look smooth by over-filtering and missing real fast motion.

### Tracking and Multi-Person Pose Metrics

**MOTA / Multiple Object Tracking Accuracy**

Intuition: "Across a video, how many misses, false positives, and identity switches happen?"

Calculation:

<div class="math-display">
\[
\operatorname{MOTA} = 1-
\frac{\operatorname{FN}+\operatorname{FP}+\operatorname{IDSW}}
     {\operatorname{GT}}
\]
</div>

Higher is better. It is useful when pose belongs to people over time, but it is dominated by detection failures and identity switches.

**MOTP / Multiple Object Tracking Precision**

Intuition: "When the tracker matches the right target, how precise is its location?"

Calculation: average localization distance or overlap for matched tracks. Higher or lower depends on the implementation; always check whether it reports similarity or error.

**IDF1**

Intuition: "How consistently does the system preserve person identity?"

Calculation: F1 score over correctly identified detections. Higher is better.

### Radar-Specific Pose and Localization Metrics

Radar pose papers often use CV metrics, but several extra measurements matter because radar observes range, angle, Doppler, and sparse point clouds.

**Range error**

Intuition: "How wrong is the estimated distance from radar to body or joint?"

Calculation:

<div class="math-display">
\[
E_{\mathrm{range}} = \left|\widehat{r}-r\right|
\]
</div>

Lower is better. Report MAE/RMSE in meters or centimeters.

**Azimuth/elevation angle error**

Intuition: "How wrong is the estimated direction of the person or joint?"

Calculation:

<div class="math-display">
\[
\begin{aligned}
E_{\mathrm{az}} &= \left|\operatorname{wrap}(\widehat{\phi}-\phi)\right|, \\
E_{\mathrm{el}} &= \left|\widehat{\theta}-\theta\right|.
\end{aligned}
\]
</div>

Lower is better. Units are degrees. Radar angle error often grows when SNR is low or people are close together.

**Cartesian localization error**

Intuition: "After converting radar range-angle to 3D position, how far is the estimate from the reference?"

Calculation:

<div class="math-display">
\[
E_{\mathrm{loc}} =
\left\lVert\widehat{\mathbf{x}}_{\mathrm{xyz}}-\mathbf{x}_{\mathrm{xyz}}\right\rVert_2
\]
</div>

Lower is better. This is often more comparable to MPJPE than raw range/angle errors.

**Point-cloud reconstruction metrics**

Intuition: "Does the radar point cloud or generated body surface occupy the right 3D space?"

Calculation: use Chamfer distance, point-to-surface distance, occupancy IoU, or average nearest-neighbor distance. Lower is better for distances; higher is better for IoU.

**Cross-subject / cross-environment split accuracy**

Intuition: "Does the pose system generalize to unseen people or rooms?"

Calculation: train on one set of subjects/environments and evaluate on held-out subjects/environments using MPJPE, PCK, AP, or classification metrics.

Interpretation: this is not a new metric formula, but it is a critical protocol label. A 30 mm MPJPE on same-subject lab data may be less convincing than 60 mm MPJPE on unseen rooms and unseen subjects.

## 2. Biomedical / Clinical / Biomechanics Metrics

Biomedical pose metrics often care less about leaderboard score and more about whether the measurement is clinically meaningful, repeatable, interpretable, and sensitive to change.

### Measurement Agreement and Validity

**Bias / mean error**

Intuition: "Does the new method systematically overestimate or underestimate the reference?"

Calculation:

<div class="math-display">
\[
e_i=\widehat{y}_i-y_i,
\qquad
\operatorname{bias}=\frac{1}{N}\sum_{i=1}^{N}e_i
\]
</div>

Lower absolute bias is better. Bias near zero means no average offset, but it does not guarantee low random error.

**MAE and RMSE against clinical reference**

Intuition: "How far is the markerless/radar/wearable measurement from gold-standard motion capture or clinician measurement?"

Calculation:

<div class="math-display">
\[
\begin{aligned}
\operatorname{MAE} &= \frac{1}{N}\sum_{i=1}^{N}\left|\widehat{y}_i-y_i\right|, \\
\operatorname{RMSE} &= \sqrt{\frac{1}{N}\sum_{i=1}^{N}\left(\widehat{y}_i-y_i\right)^2}.
\end{aligned}
\]
</div>

Lower is better. Use MAE for typical absolute error; use RMSE when large errors should be punished more.

**Correlation coefficient**

Intuition: "Do the two methods rise and fall together?"

Calculation: Pearson correlation for linear association, Spearman correlation for rank association.

Higher absolute value means stronger association, but correlation is not agreement. A method can correlate well while having a large bias.

**ICC / intraclass correlation coefficient**

Intuition: "How reliable are repeated measurements or raters?"

Calculation: ICC is a ratio of between-subject variance to total variance under a chosen ANOVA/mixed-effects model.

<div class="math-display">
\[
\operatorname{ICC}\ \approx\
\frac{\sigma_{\mathrm{between}}^2}{\sigma_{\mathrm{total}}^2}
\]
</div>

Higher is better. Always report the ICC form, e.g. one-way/two-way, consistency/absolute agreement, single/average rater. ICC is widely used in medical measurement reliability.

**Bland-Altman limits of agreement**

Intuition: "How far apart can two measurement methods be for an individual subject?"

Calculation:

<div class="math-display">
\[
\begin{aligned}
d_i &= A_i-B_i, \\
\overline{d} &= \frac{1}{N}\sum_{i=1}^{N}d_i, \\
\operatorname{LoA}_{95\%} &= \overline{d}\ \pm\ 1.96\,s_d.
\end{aligned}
\]
</div>

Narrower limits are better, if they are within a clinically acceptable range. Bland-Altman is often more informative than correlation for comparing a new pose system with motion capture.

**SEM / standard error of measurement**

Intuition: "How much noise is expected in one measured score?"

Calculation:

<div class="math-display">
\[
\operatorname{SEM}=\operatorname{SD}\sqrt{1-\operatorname{ICC}}
\]
</div>

Lower is better. SEM converts reliability into the units of the measurement, such as degrees or centimeters.

**MDC / minimal detectable change**

Intuition: "How large must a change be before we believe it is not just measurement noise?"

Calculation, common 95% version:

<div class="math-display">
\[
\operatorname{MDC}_{95}=1.96\sqrt{2}\,\operatorname{SEM}
\]
</div>

Lower is better for sensitive measurement systems. In rehabilitation, a patient change smaller than MDC95 may be noise rather than real improvement.

**MCID / minimal clinically important difference**

Intuition: "How much change matters to patients or clinicians?"

Calculation: MCID is usually estimated from clinical anchors, patient-reported outcomes, or distribution-based methods. It is not just a mathematical property of the sensor.

Interpretation: MDC asks "is the change real?" MCID asks "is the change meaningful?"

### Kinematic Pose Metrics

**Joint angle**

Intuition: "How bent or rotated is the joint?"

Calculation: define body segments, build vectors or coordinate frames, then compute relative orientation.

For a simple hinge angle:

<div class="math-display">
\[
\theta = \arccos\!\left(
\frac{\mathbf{u}^{\mathsf T}\mathbf{v}}
     {\lVert\mathbf{u}\rVert_2\lVert\mathbf{v}\rVert_2}
\right)
\]
</div>

Units are degrees. Joint angles are more clinically interpretable than raw keypoint coordinates.

**Joint angle MAE / RMSE**

Intuition: "How many degrees away is the estimated joint angle from the reference?"

Calculation:

<div class="math-display">
\[
\begin{aligned}
e_{\theta,t} &= \widehat{\theta}_t-\theta_t, \\
\operatorname{MAE}_{\theta} &= \frac{1}{T}\sum_{t=1}^{T}|e_{\theta,t}|, \\
\operatorname{RMSE}_{\theta} &= \sqrt{\frac{1}{T}\sum_{t=1}^{T}e_{\theta,t}^{2}}.
\end{aligned}
\]
</div>

Lower is better. Report the plane and convention, e.g. sagittal knee flexion or frontal hip abduction.

**Peak angle error**

Intuition: "Did the method capture clinically important peak flexion or extension?"

Calculation:

<div class="math-display">
\[
E_{\mathrm{peak}} =
\left|\max_t\widehat{\theta}_t-\max_t\theta_t\right|
\]
</div>

Lower is better. Also report timing error if peak timing matters.

**ROM / range of motion**

Intuition: "How much did the joint move?"

Calculation:

<div class="math-display">
\[
\begin{aligned}
\operatorname{ROM}(\theta) &= \max_t\theta_t-\min_t\theta_t, \\
E_{\mathrm{ROM}} &= \left|\operatorname{ROM}(\widehat{\theta})-
\operatorname{ROM}(\theta)\right|.
\end{aligned}
\]
</div>

Higher ROM is not always better clinically; interpretation depends on task and pathology.

**Segment orientation error**

Intuition: "Is the thigh, shank, trunk, pelvis, or foot orientation correct?"

Calculation with rotation matrices:

<div class="math-display">
\[
\mathbf{R}_{\mathrm{err}}=\mathbf{R}_{\mathrm{ref}}^{\mathsf T}\mathbf{R}_{\mathrm{pred}},
\qquad
E_{\theta}=\arccos\!\left(\frac{\operatorname{tr}(\mathbf{R}_{\mathrm{err}})-1}{2}\right)
\]
</div>

Lower is better. This is common for IMU and marker-based biomechanics.

**Trunk sway / postural sway**

Intuition: "How much does the body oscillate or drift?"

Calculation examples:

<div class="math-display">
\[
\begin{aligned}
\operatorname{RMS}_{\mathrm{sway}} &=
\sqrt{\frac{1}{T}\sum_{t=1}^{T}\lVert\mathbf{x}_t-\overline{\mathbf{x}}\rVert_2^2}, \\
L_{\mathrm{sway}} &= \sum_{t=2}^{T}\lVert\mathbf{x}_t-\mathbf{x}_{t-1}\rVert_2, \\
v_{\mathrm{sway}} &= \frac{L_{\mathrm{sway}}}{\text{duration}}.
\end{aligned}
\]
</div>

Lower sway can indicate better balance in many settings, but not always; some tasks require controlled movement.

### Spatiotemporal Gait Metrics

These are clinical gait measurements computed from foot contact events, foot trajectories, pressure sensors, IMUs, video pose, or motion capture. A gait cycle is commonly separated into stance, when the foot contacts the ground, and swing, when it is off the ground.

**Gait speed / walking speed**

Intuition: "How fast does the person move forward?"

Calculation:

<div class="math-display">
\[
v_{\mathrm{gait}}=\frac{\text{distance walked}}{\text{elapsed time}}
\]
</div>

Units are m/s. Faster is not always better, but very slow gait speed often indicates impairment or frailty.

**Cadence**

Intuition: "How many steps per minute?"

Calculation:

<div class="math-display">
\[
\operatorname{cadence}=\frac{\text{number of steps}}{\text{walking time in minutes}}
\]
</div>

Higher cadence means more frequent steps. Interpret with stride length and speed.

**Step length**

Intuition: "How far forward does one foot land relative to the other?"

Calculation:

<div class="math-display">
\[
L_{\mathrm{step,R}} =
x_{\mathrm{R,contact}}-x_{\mathrm{L,previous\ contact}}
\]
</div>

Units are meters or centimeters. Step length is side-specific.

**Stride length**

Intuition: "How far does the same foot travel from one contact to the next?"

Calculation:

<div class="math-display">
\[
L_{\mathrm{stride,R},i}=x_{\mathrm{R},i+1}-x_{\mathrm{R},i}
\]
</div>

In symmetric walking, stride length is roughly two step lengths.

**Step time and stride time**

Intuition: "How long does each step or full gait cycle take?"

Calculation:

<div class="math-display">
\[
\begin{aligned}
T_{\mathrm{step}} &= t_{\mathrm{current\ contact}}-t_{\mathrm{opposite\ previous\ contact}}, \\
T_{\mathrm{stride},i} &= t_{\mathrm{same\ foot},i+1}-t_{\mathrm{same\ foot},i}.
\end{aligned}
\]
</div>

Lower time means faster stepping, but interpretation depends on gait speed.

**Stance time, swing time, and double-support time**

Intuition: "How is the gait cycle divided between support and limb advancement?"

Calculation:

<div class="math-display">
\[
\begin{aligned}
T_{\mathrm{stance}} &= t_{\mathrm{toe\ off}}-t_{\mathrm{initial\ contact}}, \\
T_{\mathrm{swing}} &= t_{\mathrm{next\ contact}}-t_{\mathrm{toe\ off}}, \\
T_{\mathrm{double\ support}} &= \sum_j\left|I_j^{\mathrm{left\ contact}}
\cap I_j^{\mathrm{right\ contact}}\right|.
\end{aligned}
\]
</div>

Often reported as seconds or percent of gait cycle. Increased double support often suggests cautious or unstable gait.

**Step width / base of support**

Intuition: "How wide is the walking base?"

Calculation:

<div class="math-display">
\[
W_{\mathrm{step}}=
\left|y_{\mathrm{left\ contact}}-y_{\mathrm{right\ contact}}\right|
\]
</div>

Wider step width can reflect balance strategy, pathology, or task constraints.

**Gait variability**

Intuition: "How consistent is the walking pattern from step to step?"

Calculation:

<div class="math-display">
\[
\begin{aligned}
\operatorname{SD}_{T} &= \operatorname{sd}(T_1,\ldots,T_N), \\
\operatorname{CV}_{T} &= 100\%\times\frac{\operatorname{SD}_{T}}{\overline{T}}.
\end{aligned}
\]
</div>

Higher variability can indicate instability, fatigue, neurological impairment, or adaptation to a difficult task.

**Gait asymmetry**

Intuition: "Are the left and right sides different?"

Common calculation:

<div class="math-display">
\[
\operatorname{SI}=100\%\times
\frac{L-R}{\tfrac{1}{2}(L+R)}
\]
</div>

Other studies use ratios such as `left/right` or `affected/unaffected`. Always report the convention because sign and scale differ.

### Gait Event Metrics

**Initial-contact / heel-strike timing error**

Intuition: "How close is the detected foot-contact event to the reference event?"

Calculation:

<div class="math-display">
\[
e_{t,i}=\widehat{t}_i-t_i,
\qquad
\operatorname{MAE}_{t}=\frac{1}{N}\sum_{i=1}^{N}|e_{t,i}|
\]
</div>

Lower is better. Units are milliseconds.

**Toe-off timing error**

Intuition: "Did the system detect the start of swing at the right time?"

Calculation is the same as initial-contact timing error, but using toe-off events.

**Event detection precision, recall, and F1**

Intuition: "Did the system find the right gait events without hallucinating extra ones?"

Calculation:

<div class="math-display">
\[
\begin{aligned}
\operatorname{precision} &= \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}, &
\operatorname{recall} &= \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}, \\
F_1 &= 2\,\frac{\operatorname{precision}\cdot\operatorname{recall}}
{\operatorname{precision}+\operatorname{recall}}.
\end{aligned}
\]
</div>

A predicted event is usually a true positive if it falls within a time tolerance, such as +/- 50 ms or +/- 100 ms. Higher is better.

### Kinetic and Dynamic Metrics

**Ground reaction force error**

Intuition: "Does the estimated force match force-plate measurements?"

Calculation:

<div class="math-display">
\[
e_{F,t}=\widehat{F}_t-F_t,
\qquad
\operatorname{RMSE}_{\mathrm{GRF}}=
\sqrt{\frac{1}{T}\sum_{t=1}^{T}e_{F,t}^{2}}
\]
</div>

Forces are often normalized by body weight. Lower is better.

**Joint moment error**

Intuition: "Does the method estimate the rotational load at a joint?"

Calculation: compute inverse dynamics from segment motion, external forces, body mass/inertia, and joint centers, then compare estimated versus reference joint moments.

<div class="math-display">
\[
e_M=\widehat{M}_{\mathrm{joint}}-M_{\mathrm{joint}}
\]
</div>

Moments are often normalized by body mass, e.g. Nm/kg. Lower is better.

**Joint power error**

Intuition: "Does the joint generate or absorb mechanical power correctly?"

Calculation:

<div class="math-display">
\[
P_{\mathrm{joint}}=M_{\mathrm{joint}}\,\omega_{\mathrm{joint}},
\qquad
e_P=\widehat{P}_{\mathrm{joint}}-P_{\mathrm{joint}}
\]
</div>

Power is often normalized by body mass, e.g. W/kg. Lower error is better.

**Impulse**

Intuition: "How much total force is applied over time?"

Calculation:

<div class="math-display">
\[
J=\int_{t_{\mathrm{contact}}}^{t_{\mathrm{toe\ off}}}F(t)\,\mathrm{d}t
\]
</div>

Used for braking, propulsion, and vertical support. Interpret relative to body weight and walking speed.

### Clinical Gait Summary Indices

**GVS / Gait Variable Score**

Intuition: "How abnormal is one specific joint-angle curve?"

Calculation:

<div class="math-display">
\[
\operatorname{GVS}_{q}=
\sqrt{\frac{1}{T}\sum_{t=1}^{T}\left(q_t-\overline{q}^{\mathrm{norm}}_t\right)^2}
\]
</div>

Lower is closer to the normative reference. Usually computed over a normalized gait cycle.

**GPS / Gait Profile Score**

Intuition: "How abnormal is the overall gait kinematic pattern?"

Calculation:

<div class="math-display">
\[
\operatorname{GPS}=\sqrt{\frac{1}{Q}\sum_{q=1}^{Q}\operatorname{GVS}_{q}^{2}}
\]
</div>

Lower is better, meaning closer to normative gait. GPS is often reported in degrees and can be paired with a Movement Analysis Profile showing which variables contribute most.

**MAP / Movement Analysis Profile**

Intuition: "Which joints or planes explain the abnormal gait score?"

Calculation: report the individual GVS components that make up GPS, commonly pelvis, hip, knee, ankle, and foot progression variables.

**GDI / Gait Deviation Index**

Intuition: "Single-number summary of how far a gait pattern deviates from normal."

Calculation, conceptually:

```text
1. Extract gait features from kinematic curves.
2. Compute distance from normative gait in the feature space.
3. Transform to a score where 100 is typical normal gait and lower is more abnormal.
```

Interpretation: GDI is scaled so that each 10-point decrease is commonly interpreted as about one standard deviation farther from typical normal gait. Higher is better.

**GGI / Gillette Gait Index**

Intuition: "How abnormal is gait across a set of clinically chosen variables?"

Calculation: compute deviations from normal across selected spatiotemporal and kinematic variables and combine them into one index.

Lower is better. GGI is older and less directly interpretable than GPS/GDI, but still appears in clinical gait literature.

### Functional and Clinical Outcome Metrics Related to Pose

These are not pure geometric pose metrics, but they are often used as biomedical endpoints for pose-estimation systems.

**TUG / Timed Up and Go**

Intuition: "How quickly can the person stand, walk, turn, return, and sit?"

Calculation:

<div class="math-display">
\[
T_{\mathrm{TUG}}=t_{\mathrm{seated\ finish}}-t_{\mathrm{start}}
\]
</div>

Lower time usually indicates better mobility. Pose systems may estimate sub-events such as sit-to-stand time, turning time, and gait speed.

**6MWT / six-minute walk test**

Intuition: "How much walking capacity does the person have?"

Calculation:

<div class="math-display">
\[
D_{\mathrm{6MWT}}=\sum_j D_j
\quad\text{over a six-minute interval}
\]
</div>

Higher distance usually indicates better endurance or functional capacity.

**Berg Balance Scale or observational scores**

Intuition: "Can the person perform balance tasks safely?"

Calculation: clinician-scored items are summed. These scores are ordinal clinical outcomes, not coordinate errors. Pose metrics may support automation, but should be validated against clinician labels.

### Practical Reporting Rules

- Always say whether a metric is lower-better or higher-better.
- Always state units: pixels, mm, cm, meters, degrees, seconds, percent gait cycle, or body-weight normalized force.
- Always state alignment: raw, root-centered, scale-aligned, or Procrustes-aligned.
- Always state evaluation protocol: same-subject, cross-subject, cross-view, cross-room, cross-device, or clinical holdout.
- For biomedical validation, report both error and agreement. Correlation alone is not enough.
- For clinical relevance, report whether errors are below MDC or an accepted clinical threshold, not only whether the model beats another model.
- For radar pose, separate global localization quality from body-pose quality. A method can have good relative skeleton shape but poor absolute room position.

## Sources

- COCO keypoint OKS implementation in `pycocotools`: [cocoeval.py](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).
- MMPose metric documentation for AUC, EPE, PCK, NME, and MPJPE: [MMPose API docs](https://mmpose.readthedocs.io/en/latest/api.html).
- MMPose dataset/evaluation context: [MMPose dataset preparation docs](https://mmpose.readthedocs.io/en/latest/user_guides/prepare_datasets.html).
- Gait cycle terminology and spatiotemporal gait definitions: [Bipedal gait cycle](https://en.wikipedia.org/wiki/Bipedal_gait_cycle).
- Video-based spatiotemporal gait-analysis examples: [Transforming Gait](https://arxiv.org/abs/2203.09371).
- Gait deviation summary context for GDI/GPS/OAM: [Functional Gait Deviation Index](https://arxiv.org/abs/2310.06674).
- Bland-Altman agreement and limits of agreement: [Bland-Altman plot](https://en.wikipedia.org/wiki/Bland%E2%80%93Altman_plot).
