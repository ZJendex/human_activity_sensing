# Three-ZED-X Human 3D Skeleton, Trajectory, and SMPL Mesh Roadmap

Last updated: 2026-07-24.

## Executive recommendation

Build the research pipeline below, with one optional external benchmark:

1. **Research pipeline:** process the three custom side-by-side elementary H.265 streams and implement the modular M0a-M6 stages below. This pipeline exposes intermediate measurements, uncertainty, and failure cases, and allows individual methods to be replaced.
2. **Optional Theia3D pilot:** export six synchronized rectified eye videos and matching custom intrinsics after M0b. This is an experimental comparison, because ZED X is not a camera system verified by Theia Markerless and the three stereo pairs provide only three substantially different viewpoints.

The recommended research pipeline is:

```text
Three synchronized ZED X stereo pairs
        |
        v
M0a replay indexed H.265 samples, split left/right eyes, and align the three rigs
        |
        v
M0b rectify raw eyes, calibrate the array, and define the world frame
        |
        v
M1  person detection + reliable 2D joints in every camera
        |
        v
M2  same-person association across synchronized views
        |
        v
M3  robust triangulation -> 3D joints + covariance in world coordinates
        |
        v
M4  world-space tracking -> stable trajectory + anonymous persistent ID
        |
        v
M5  per-track SMPL pose fitting to the fused 3D skeleton
        |
        v
M6  sequence-level shape, silhouette, depth, temporal, and floor optimization
        |
        v
World-space mesh sequence + trajectory + confidence
```

Do not begin by training one end-to-end model. A calibrated, uncertainty-aware geometric baseline is easier to debug and is likely to transfer to the real three-camera layout better than a model trained for a different camera arrangement. Once the modular baseline works, benchmark learned multi-view models such as MV-SSM or TEMPO against it.

The most important engineering conclusion is that **M0 is a prerequisite milestone**. Synchronization or extrinsic-calibration errors can look like failures in M1-M6. No downstream model can consistently repair a moving, incorrectly scaled, or temporally skewed coordinate system.

### Adjustment for the current H.265 capture path

The current input has been confirmed as:

```text
ZED X 1: one 3840x1200 side-by-side elementary H.265 stream
ZED X 2: one 3840x1200 side-by-side elementary H.265 stream
ZED X 3: one 3840x1200 side-by-side elementary H.265 stream

Within every decoded frame:
  left eye  = pixels x=[0, 1920),    1920x1200
  right eye = pixels x=[1920, 3840), 1920x1200
```

Before encoding, the two unrectified 1920x1200 BGRA eye images are concatenated
horizontally without crop, padding, rotation, scaling, or rectification. The
3840x1200 image is converted to BT.709 limited-range NV12 with 2x2 chroma
averaging and then lossily H.265 encoded. Rectification is therefore a
mandatory, exactly-once operation after decoding and splitting. All calibration
lookup, rectification maps, and projection matrices must match the 1920x1200
eye geometry and the correct camera serial number.

The adjusted front end is:

```text
three 3840x1200 side-by-side elementary H.265 streams
  + rgb/meta.yaml + rgb/meta.npz + rgb/frames.idx
  -> replay using external sample_index and pts_ns
  -> decode one stereo pair per encoded frame
  -> split fixed left/right 1920x1200 regions
  -> verify no frame-index or cross-camera discontinuities
  -> rectify each raw eye with the calibration from that capture session
  -> use the three rectified left views for M1-M5
  -> reconstruct per-ZED stereo depth only as a validated optional branch
```

Existing H.265 data does not need to be discarded. Wide-baseline 3D triangulation uses the three rectified left images and remains the primary M3 method. Dense stereo depth is helpful for single-view fallback and M6 mesh refinement, but it is not a prerequisite for the first working 3D skeleton and trajectory.

#### Input status for a normally finalized recording

| Status | Input |
|---|---|
| ✅ Confirmed | Three full-size ZED X stereo cameras with nominal 120 mm baselines and three side-by-side elementary H.265 streams |
| ✅ Confirmed | Left and right images are unrectified |
| ✅ Confirmed | Each decoded frame is 3840x1200: left `x=[0,1920)`, right `x=[1920,3840)` |
| ✅ Confirmed | The three ZED X rigs are synchronized over a peer-to-peer LAN; no shared hardware exposure trigger is used |
| ✅ Available in `rgb/meta.yaml` | Raw left/right `fx`, `fy`, `cx`, `cy`, width, height, physical focal length, and 12 distortion coefficients |
| ✅ Available in `rgb/meta.yaml` | Internal stereo translation in meters and row-major 3x3 rotation under `left_to_right` |
| ✅ Available | Recording-to-camera mapping from `native_stream_info.serial_number`; eye mapping from the fixed side-by-side pixel regions |
| ✅ Available | Requested, actual, and observed FPS; `sample_index`, host UTC `time_ns`, `host_monotonic_ns`, ZED hardware `device_time_ns`, and external `pts_ns` with nanosecond time base |
| ✅ Available | No crop, padding, rotation, rescale, or rectification; BGRA side-by-side input is converted to BT.709 limited-range NV12 with 2x2 chroma averaging before lossy H.265 encoding |
| ⬜ Still required | Array extrinsics `T_world_left`, or a simultaneous calibration-target sequence. Stored `left_to_right` and `camera_imu_transform` do not locate a camera in the shared world |
| ⬜ Still required | Metric target dimensions, capture-volume measurements, and desired floor/world origin; these physical scene definitions are not in the recording metadata |

#### Existing `rgb/meta.yaml` calibration contract

Every normal ZED recording stores:

```yaml
native_stream_info:
  serial_number: ...
  raw_calibration:
    left:
      fx: ...
      fy: ...
      cx: ...
      cy: ...
      width: 1920
      height: 1200
      focal_length_mm: ...
      distortion: [d0, ..., d11]
    right:
      fx: ...
      fy: ...
      cx: ...
      cy: ...
      width: 1920
      height: 1200
      focal_length_mm: ...
      distortion: [d0, ..., d11]
    left_to_right:
      translation_m: [tx, ty, tz]
      rotation: [r11, r12, r13, r21, r22, r23, r31, r32, r33]
      # rotation is row-major 3x3
```

Convert each eye's scalar fields into:

```text
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]

D = distortion[12]
```

For a ZED radial-tangential model, verify that the stored distortion order is:

```text
[k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4]
```

The `left_to_right` rotation and translation can be assigned to the handbook's
`T_right_left` only after confirming that the metadata means:

```text
X_right = R_left_to_right * X_left + t_left_to_right
```

This is a convention check, not a missing-calibration problem. Confirm it once
with the rectified epipolar test and a known point; if the equation is reversed,
invert the rigid transform before use.

#### Finalized recording identity, timing, and pixel contract

For a normally finalized recording:

- `rgb/meta.yaml -> native_stream_info.serial_number` identifies the ZED X.
- Each H.265 sample is one complete stereo pair. `sample_index` is therefore the
  stereo-pair ID.
- The elementary H.265 stream has no container PTS. The authoritative
  presentation timestamp is the external `pts_ns` stored with the frame index
  in `rgb/meta.npz` and `rgb/frames.idx`.
- Per-sample timing includes host UTC `time_ns`, `host_monotonic_ns`, ZED
  hardware `device_time_ns`, and external `pts_ns` with nanosecond time base.
- The left eye is the decoded pixel region `x=[0,1920)` and the right eye is
  `x=[1920,3840)`.

Use the metadata fields as follows:

```text
recording identity       = native_stream_info.serial_number
stereo-pair identity     = sample_index
presentation/replay time = pts_ns
cross-host candidate     = host UTC time_ns, after P2P-LAN residual validation
local cadence check      = host_monotonic_ns
device cadence check     = ZED device_time_ns
```

Do not invent container PTS for the elementary bitstream or treat decoder
arrival time as capture time. Preserve the external index when remuxing or
transcoding, and carry `sample_index` into every derived left/right frame.

Implementation evidence supplied by the capture system:

- Serial and fixed left/right layout: `nodes/zedx/native/main.cpp:425` and
  `nodes/zedx/native_backend.py:98`.
- Per-frame timing/index contract: `docs/spec/data-format.md:341`.
- BGRA concatenation and BT.709 limited-range NV12 conversion:
  `nodes/zedx/native/convert.cu:27`.

## Scope and assumptions

This roadmap assumes:

- Three fixed ZED X stereo cameras observe a shared indoor capture volume.
- Each stereo camera retains its factory left/right calibration.
- Each ZED X is stored as one 3840x1200 side-by-side elementary H.265 stream containing two unrectified 1920x1200 eyes. Authoritative timing and frame identity come from `rgb/meta.npz` and `rgb/frames.idx`, not container timestamps.
- The deployed three-camera system uses peer-to-peer LAN synchronization and does not use a shared hardware exposure trigger.
- People are visible in at least two wide-baseline cameras for most frames.
- The desired output is an anonymous track within a recording/session, not a real-world identity.
- SMPL is sufficient for body pose and shape. Use SMPL-X only if articulated hands or facial expression are required.

Three stereo cameras provide useful redundancy, but they should not be thought of as six independent wide-baseline cameras. The left/right sensors within one ZED X have a short, fixed baseline and mainly provide local depth. The large baselines between the three camera locations provide the strongest cross-view triangulation geometry.

### Confirmed camera model

The hardware is the [Stereolabs ZED X Stereo Camera](https://www.stereolabs.com/store/products/zed-x-stereo-camera), not the ZED X Mini, Nano, or ZED X One. The official specifications list:

- Two synchronized 1920x1200 global-shutter color sensors.
- 15, 30, or 60 FPS at 1920x1200.
- A nominal 120 mm stereo baseline.
- GMSL2 transport and external multi-camera synchronization capability.

These properties are favorable for human-motion capture: the global shutter avoids rolling-shutter body deformation, and the product supports external synchronization. The current deployment does not use that external trigger; it synchronizes the rigs over the LAN. The effective per-eye focal length is already stored in `rgb/meta.yaml`; the actual frame rate, serial numbers, exposure, synchronization evidence, and metadata-file identity must still be recorded rather than inferred from the product family.

## Recommended camera and compute layout

### Physical arrangement

For a room-scale volume, begin with:

- Cameras distributed around the volume rather than placed along one wall.
- Roughly 90-120 degree azimuth separation when the room permits it.
- A modest downward view from above normal head height, while keeping feet visible.
- The complete working volume visible in at least two cameras and the central volume visible in all three.
- Static, rigid mounts with strain relief; record a calibration hash after any physical adjustment.
- Similar exposure, gain, white balance, resolution, and frame rate across cameras.

These are starting design recommendations, not universal constants. Verify the actual rig with an occlusion map, ray-intersection angles, and held-out calibration measurements.

### Capture topology

ZED X is a GMSL2 camera and requires a ZED Link capture device on an NVIDIA
Jetson platform. The two eyes inside each ZED X are synchronized internally. In
the deployed system, however, the three complete ZED X rigs do not share a
hardware exposure trigger; their capture systems synchronize over a
peer-to-peer LAN.

`P2P LAN` describes the network topology and should not automatically be
interpreted as IEEE 1588 Precision Time Protocol (`PTP`). Record whether the
actual clock protocol is PTP, NTP/chrony, or a custom peer synchronization
protocol. Network synchronization can align clocks, but it does not by itself
guarantee that three free-running cameras expose at the same instant.

Therefore:

1. Document the ZED serial, capture port, host, and network peer for every stream.
2. Record the synchronization protocol, clock master, clock domain, update interval, and reported offset for every host.
3. Timestamp each captured stereo pair as close to exposure/acquisition as possible, before H.265 encoding.
4. Measure cross-camera exposure phase with a flashing LED or electronic timer visible in all cameras.
5. Estimate both constant offset and drift over a long recording.
6. Form multi-camera bundles by corrected capture timestamps and reject bundles outside the measured skew limit.
7. Treat hardware triggering as a possible future improvement, not as a property of the current dataset.

Vendor references for the camera and a possible future hardware-trigger upgrade:

- [Stereolabs multi-camera setup and synchronization](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/multi-camera)
- [ZED Link Quad GPIO triggering details](https://docs.stereolabs.com/docs/products/embedded/zed-link-capture-card/zed-link-quad/zed-link-quad-gpio-triggering)

### Compute split

A practical deployment has three stages:

- **Current capture stage on Jetson:** acquire one raw left/right pair, concatenate the two 1920x1200 BGRA images into a 3840x1200 frame, convert it to BT.709 limited-range NV12, encode one H.265 sample, and write its identity/timing to the external metadata/index.
- **Ingest stage:** decode, restore presentation order, validate stereo/camera synchronization, rectify both eyes, and optionally compute depth.
- **Inference stage:** run batched per-view pose inference and multi-view fusion either on the same Jetson for a light model or on a workstation GPU for the research models.

For future recordings, keep the custom H.265 path and write a lossless per-frame
metadata sidecar plus the matching calibration snapshot.

### H.265 capture contract

For each encoded frame, store:

```text
session_id
camera serial number
sample_index = stereo_pair_id
host UTC time_ns
host_monotonic_ns
ZED device_time_ns
external pts_ns
requested, actual, and observed FPS
decoded layout = side_by_side
decoded size = 3840x1200
left pixel region = x:[0,1920), y:[0,1200)
right pixel region = x:[1920,3840), y:[0,1200)
source format = two unrectified 1920x1200 BGRA eyes
encoded format = BT.709 limited-range NV12, 2x2 chroma averaging
codec profile, level, bitrate/quality mode, GOP, B-frame count
exposure, gain, and white balance
calibration_snapshot_id
dropped/corrupt frame flag
```

Rules:

- Treat one `sample_index` as one indivisible stereo pair; both eyes occupy the same encoded frame.
- Read authoritative `pts_ns` from `rgb/meta.npz`/`rgb/frames.idx`; the elementary stream has no container PTS.
- Align the three ZED cameras using validated cross-host timing metadata, never H.265 decoder arrival time or file position alone.
- Assert that every decoded visible frame is exactly 3840x1200, then split it at `x=1920`.
- Assert that both resulting eyes are exactly 1920x1200 before rectification.
- Do not silently resize, crop, rotate, pad, or swap either half. Any image-coordinate change must also update the camera matrix.
- Rectify each frame exactly once; mark the resulting images explicitly as rectified.
- Preserve `sample_index` and the external timing fields when splitting, remuxing, or transcoding.
- Prefer the same encoder configuration for all three ZED X streams.

Lossy H.265 can be adequate for 2D pose at sufficient quality, but that must be measured on this data. BT.709 limited-range conversion, NV12 4:2:0 chroma averaging, and H.265 artifacts can reduce fine joint appearance and stereo correspondence accuracy. Use a higher-bitrate, short-GOP/intra, or lossless reference capture for calibration and stereo-depth validation. Stereolabs publishes H.265 streaming bitrates as transport-oriented starting points; they are not a guarantee of scientific keypoint or depth accuracy.

## M0: calibration, synchronization, and coordinate contract

### M0a: decode and stereo-pair integrity

Before spatial calibration or pose inference:

1. Load `rgb/meta.yaml`, `rgb/meta.npz`, and `rgb/frames.idx` before decoding the elementary stream.
2. Decode each stream in the order defined by the external index and authoritative `pts_ns`.
3. Build a manifest keyed by `(serial_number, sample_index)`.
4. Assert a 3840x1200 decoded frame and split the fixed left/right pixel regions.
5. Copy the same `sample_index` and timing metadata to both derived eye images.
6. Use validated P2P-LAN timing to form three-ZED bundles inside the allowed residual skew.
7. Mark missing, duplicate, corrupt, or concealed encoded samples invalid.
8. Compare decoded counts against index entries and verify monotonic `sample_index` and `pts_ns`.
9. Run a shared flashing-LED test through the complete encode/decode path.
10. Hash the H.265 stream, metadata files, and derived manifest for reproducibility.

If a recording is not normally finalized and the external index is missing, container PTS is unavailable because the input is an elementary H.265 stream. Recovery must use bitstream decode order plus a visual synchronization event and is explicitly lower confidence. Do not assume that sample `n` in the three recordings represents the same cross-camera exposure until timing and visual events confirm it.

For recordings without a session calibration snapshot:

1. Recover the factory raw calibration by camera serial number for the exact 1920x1200 mode.
2. Determine whether the capture used ZED startup self-calibration.
3. If the rig is still fixed, record a high-quality target sequence now and validate the recovered calibration through the full decode/rectify path.
4. If the rig moved and no target was present in the old session, flag stereo depth and absolute array geometry as unverified. M1 can still run, but metric M3 requires a defensible old-session array transform.

### M0b: mandatory rectification for the confirmed input

For every camera serial:

1. Replay one indexed sample, assert a 3840x1200 decoded image, split at `x=1920`, and assert two 1920x1200 eyes with the documented absence of spatial transforms.
2. Load the 1920x1200 raw left/right intrinsics, distortion coefficients, and `T_right_left` from the matching capture-session calibration.
3. Generate `R_left`, `R_right`, `P_left`, `P_right`, `Q`, and the two remap tables.
4. Remap both eyes once and save or stream them with `image_convention=rectified`.
5. Record the rectified output size, valid-pixel ROI, and matrices actually used. Do not infer them later from the input dimensions.
6. Verify that corresponding calibration-target points have near-zero vertical disparity throughout the image, including corners.

The rectified left image and `P_left` become the canonical camera observation for
M1-M5 and cross-camera array calibration. The rectified right image, `P_right`,
and `Q` are retained for the optional stereo-depth branch.

### M0 output

M0 is complete only when the system produces a versioned calibration package containing:

```text
rig_id
calibration_timestamp
ZED_SDK_version
ZED_driver_version
JetPack_version
camera serial numbers and capture ports
resolution, frame rate, exposure mode
raw K_left, distortion model and coefficients
raw K_right, distortion model and coefficients
T_right_left for each stereo pair
rectification R_left, R_right, P_left, P_right, and Q
T_world_left[serial] for each camera
floor plane in world coordinates
time-offset estimate and uncertainty for every camera
H.265 stream manifest and external-index/pts_ns replay policy
calibration quality report
coordinate-system and unit declaration
recording metadata paths/hashes for rgb/meta.yaml, rgb/meta.npz, and rgb/frames.idx
```

### Required calibration inputs

| Input | Why it is needed | Source |
|---|---|---|
| Camera model and serial number | Stable hardware identity and calibration lookup | `rgb/meta.yaml -> native_stream_info.serial_number` |
| Exact capture layout and eye resolution | Intrinsics and split/rectification maps are resolution-specific | Assert decoded 3840x1200, then fixed 1920x1200 halves |
| Raw left/right camera matrices `K` | Rectification and raw-image projection | `native_stream_info.raw_calibration.left/right` in `rgb/meta.yaml` |
| Raw distortion model and 12 coefficients | Correct unrectified sensor pixels | `native_stream_info.raw_calibration.left/right.distortion` |
| Internal stereo transform `T_right_left` | Rectification and stereo depth | `raw_calibration.left_to_right`, after verifying transform direction |
| Rectified `P_left`, `P_right`, and `Q` | Projection and disparity-to-depth | Rectification output/ZED API |
| `sample_index`, host UTC, host monotonic, ZED device time, and `pts_ns` | Restore frame identity and cross-rig timing after H.265 decoding | `rgb/meta.npz` and `rgb/frames.idx` |
| External `pts_ns` plus GOP/B-frame settings | Correct presentation order for an elementary stream with no container PTS | External frame index plus encoder metadata |
| Pre/post-encode dimensions and pixel transform | Prevent an unmodeled image-coordinate change | Confirmed fixed split; `nodes/zedx/native/convert.cu:27` |
| Exposure timing and verified residual offset | Moving-person triangulation | P2P-LAN clock log plus flashing-target test |
| Array extrinsics `T_world_left` | Common world coordinates | Target calibration |
| Metric target geometry | Array scale and target-based calibration | Measured ChArUco/AprilGrid |
| Floor plane and chosen origin | Stable world mesh and contact | Target survey/plane fit |
| Capture-volume bounds | 3D gating and learned voxel methods | Measured room/ROI |

Stereolabs factory calibration includes intrinsics, distortion, and the transform between the two sensors. For raw eye images, use `calibration_parameters_raw`; the normal `calibration_parameters` describe already rectified `VIEW::LEFT`/`VIEW::RIGHT` images. Archive the values returned in the same live session in which the images were captured, because ZED self-calibration can refine them at startup. Do not retrieve a possibly different self-calibrated snapshot from a later run and silently apply it to old video. See [ZED raw and rectified calibration](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/camera-calibration).

### Relationship to the generic OpenCV calibration tutorial

The [Camera Calibration Explained](https://medium.com/perceptron-perspectives/camera-calibration-explained-enhancing-accuracy-in-computer-vision-applications-8ad1494cc5f2) article describes the standard pinhole-camera workflow: observe a measured chessboard at multiple poses, detect corners, solve for intrinsics/extrinsics, and evaluate reprojection error. We use the same geometry, corner observations, and validation concept, but not the article's generic `cv2.calibrateCamera` call as the default source of ZED X intrinsics.

The project decision is:

1. Use the ZED X factory/session `calibration_parameters_raw` for each 1920x1200 left and right sensor.
2. Use those parameters and `T_right_left` to rectify each stereo pair.
3. Keep the ZED-derived intrinsics fixed while estimating the three-rig array extrinsics from a measured multi-corner board.
4. Use held-out board reprojection and stereo epipolar error to validate the ZED parameters.
5. Re-estimate intrinsics from a dedicated lens-calibration sequence only if the ZED parameters fail validation or the optics have been modified.

Therefore, we are using the tutorial's calibration model and quality checks, while deliberately using the camera manufacturer's per-serial intrinsics instead of replacing them with a generic tutorial estimate.

### Temporal synchronization procedure

1. Configure all cameras to the same nominal frame rate.
2. Start and monitor the peer-to-peer LAN clock-synchronization process.
3. Record the clock protocol, master/domain, offset estimate, and drift estimate.
4. Use `sample_index` as the indivisible stereo-pair ID and retain all four timing fields from the external index.
5. Place a fast flashing LED or electronic timer visible in all cameras.
6. Record several transitions and fast human motion through H.265.
7. Decode using external `pts_ns` and compare image exposure events, not decoder receive time.
8. Verify the fixed side-by-side split and shared `sample_index` before measuring cross-camera skew.
9. Estimate residual pairwise cross-camera offsets and check for drift over a long recording.
10. Save `sample_index`, all source timing fields, clock-domain information, measured residual, and a `sync_valid` flag.
11. Correct only a validated stable offset. Reject or offline-interpolate multi-camera observations outside the configured skew limit; never interpolate the left and right eyes of one stereo pair independently for depth.

At a body speed of 1 m/s, 10 ms of temporal skew creates approximately 10 mm of apparent displacement before any pose error. A full 30 FPS frame of skew creates roughly 33 mm. A reasonable initial acceptance target is therefore less than 5 ms measured residual skew for ordinary movement, with a stricter target for fast motion. LAN synchronization is accepted only after this residual is measured; synchronized host clocks alone are not sufficient evidence.

### Target-based array calibration

Use a rigid, accurately measured ChArUco board or AprilGrid large enough to be detected at the working distance.

1. **Keep factory intrinsics fixed.** Load the raw intrinsics for each serial's 1920x1200 capture mode.
2. **Decode before detecting the target.** Calibration validation must see the same H.265 decode path used by M1-M6.
3. **Rectify with the matching snapshot.** Generate maps from the session's raw left/right calibration and stereo transform, then apply them exactly once.
4. **Use one image convention downstream.** Pose and array calibration use rectified left images with `P_left` and zero residual distortion. Do not combine unrectified pixels with rectified intrinsics.
5. **Collect connected views.** Move the target throughout the entire working volume and capture it simultaneously in cameras 1-2, 2-3, and 1-3 where possible. Vary depth, tilt, height, and image location.
6. **Estimate board poses.** Detect subpixel corners and solve PnP for each camera/board observation.
7. **Bundle adjust.** Jointly optimize camera extrinsics and time-varying board poses while fixing one camera to remove gauge freedom. Use robust losses and retain metric target dimensions.
8. **Define the world frame.** Set the floor as `z=0`, choose a measured origin, and align the vertical axis with gravity/floor normal.
9. **Validate on held-out observations.** Do not report training-view reprojection alone.
10. **Export the result.** Store the canonical calibration file with its coordinate convention, target measurements, uncertainty, and validation report.

OpenCV's ChArUco calibration tools expose the camera/board pose and reprojection error; see [OpenCV calibration with ArUco and ChArUco](https://docs.opencv.org/trunk/da/d13/tutorial_aruco_calibration.html).

#### RGB-D checkerboard calibration procedure

Use the moving-checkerboard procedure from the CMU Panoptic Studio RGB-D
subsystem as a reference pattern, adapted to ZED X geometry. CMU detected the
same measured checkerboard in the color and IR images of ten Kinect v2 units,
attached depth measurements to the IR corners, initialized the depth-camera
extrinsics by Procrustes alignment, and jointly refined the camera parameters
and checkerboard geometry. The important transfer to this project is the joint
use of 2D reprojection, measured depth, and known metric board geometry—not the
Kinect-specific IR implementation.

ZED X does not have a separate IR depth camera. Its depth is reconstructed from
the two RGB eyes. Therefore, calibrate and validate the three RGB-D rigs as
follows:

1. **Freeze the sensor and image contract.** Record all three serial numbers,
   exact 1920x1200 mode, session raw intrinsics, distortion coefficients,
   `T_right_left`, rectification matrices, H.265 hashes, and timing metadata.
   Keep the factory/session intrinsics fixed in the default solve.
2. **Measure the target.** Use a rigid checkerboard, ChArUco board, or AprilGrid
   with a surveyed corner layout and metric square spacing. Record the printed
   dimensions, physical dimensions, board version, and measurement uncertainty.
3. **Capture a moving-board sequence.** Move the board for approximately three
   minutes throughout the deployed working volume. Vary range, pitch, yaw,
   roll, height, and image position. Include simultaneous observations in
   cameras 1-2, 2-3, and 1-3 so that the camera-overlap graph is connected.
   Keep some poses and regions held out for validation.
4. **Use the production pixel path.** Decode each indexed H.265 sample, split
   the fixed left/right regions, and rectify both eyes exactly once with the
   matching session snapshot before detecting corners.
5. **Detect corresponding corners.** Detect subpixel board corners in both
   rectified eyes. Match corners by board ID and corner index; never use
   nearest-neighbor image matching when the target supplies an identity.
   Reject blurred, clipped, grazing-angle, or inconsistent detections.
6. **Attach metric depth to board corners.** Triangulate each accepted left/right
   corner pair with `P_left` and `P_right`, or sample the ZED stereo depth only
   after confirming that it produces the same 3D corner within tolerance.
   Retain the left/right pixel covariance and depth uncertainty. Do not sample
   an unvalidated dense-depth pixel on a board boundary.
7. **Initialize the solve.** Use the session intrinsics and internal stereo
   transform as initial values. Fix the first rectified-left camera to remove
   gauge freedom. Initialize each board pose with PnP, then initialize the other
   rigs through shared board poses, pairwise 3D Procrustes alignment, or a
   connected pose graph.
8. **Jointly optimize.** Optimize the three left-camera array extrinsics and
   the time-varying board poses with a robust nonlinear least-squares solver.
   Keep each ZED's internal `T_right_left` fixed by default. Refining internal
   stereo geometry is a separate, explicitly versioned operation permitted only
   when the factory/session stereo calibration fails held-out epipolar and
   metric tests.
9. **Use all RGB-D constraints.** Minimize robust 2D reprojection residuals in
   every accepted eye, stereo-derived 3D/depth residuals, and metric board
   geometry residuals. A conceptual objective is:

   ```text
   E = E_left/right_reprojection
     + lambda_depth * E_stereo_3D
     + lambda_metric * E_board_geometry
   ```

   where `E_board_geometry` penalizes disagreement with the surveyed corner
   spacing and rigid planar target, and `E_stereo_3D` compares the optimized
   board corners with their stereo-triangulated 3D measurements. Use
   uncertainty weighting and a robust loss instead of treating every corner
   equally.
10. **Place the calibration in the project world.** After the relative RGB-D
    solve is stable, align it to the surveyed floor and selected origin. Derive
    all right-eye and depth/world transforms by composing the canonical
    left-camera world transforms with the declared `T_right_left` convention.
11. **Validate on unseen measurements.** Report held-out per-eye reprojection
    error, rectified vertical disparity, metric board-edge error, 3D corner
    error, depth residual versus range, and cross-rig point-cloud seam error.
    Plot these errors over image position and capture-volume position. Repeat
    or reject the calibration if a region is weakly covered.
12. **Keep timing separate from geometry.** Run the flashing-target
    synchronization gate on the same sequence. A moving board can expose
    temporal skew, but the optimizer must not absorb a clock offset into a
    biased camera translation.

The output of this procedure is only a versioned RGB-D calibration: per-eye
intrinsics and rectification, fixed internal stereo transforms, three-rig array
extrinsics, board and world-frame definitions, timing validity, covariance or
uncertainty summaries, and held-out calibration metrics. It does not depend on
human pose detections or body reconstruction.

### Theia3D compatibility assessment

[Theia3D's camera-system requirements](https://docs.theiamarkerless.com/theia3d-documentation/camera-system-requirements) allow users to bring their own synchronized cameras, but ZED X is not on Theia Markerless's verified equipment list. Treat this as an experimental pilot and validate it against the research pipeline rather than assuming vendor support.

| Theia3D requirement | Three-ZED-X assessment | Required action |
|---|---|---|
| Minimum six cameras; eight recommended | Six eye sensors meet the numerical minimum, but the eyes form three close stereo pairs and therefore only three substantially different viewpoints | Pilot is possible but geometrically borderline; do not describe it as equivalent to six cameras distributed independently around the subject |
| Synchronous video with identical starts and durations | The deployed rigs use P2P-LAN synchronization; each ZED pair has reliable `sample_index`/timing metadata, but cross-rig exposure residual still needs measurement | Measure residual visual skew, build M0a from the three indexes, and export six eye videos over the common valid interval with identical frame counts |
| Each joint visible in at least three cameras | Possible only if the three ZED X rigs surround the subject and avoid occlusion | Perform a visibility audit over the complete movement volume |
| Person roughly 500 pixels tall in typical applications | Feasible within a 1200-pixel-high image | Check this in all six rectified views at the farthest subject location |
| High-quality, low-blur images | Global shutter is favorable | Use short exposure, sufficient light, fixed focus, and inspect H.265 artifacts |
| Intrinsic and extrinsic calibration for every view | ZED provides per-eye intrinsics; Theia can load custom intrinsics and solve extrinsics from a chessboard | Generate a Theia calibration file for all six rectified eyes and record a synchronized board trial |
| Six separate MP4/AVI camera videos in a supported codec/chroma format | The current input is three side-by-side elementary streams, so it is not directly loadable as six Theia cameras; Theia also does not document H.265/HEVC as supported | Split and rectify first, then export six synchronized high-quality H.264 MP4 videos with supported YUV 4:2:0 and preserved `sample_index` mapping |
| Verified camera-system support | ZED X is not listed among Theia's verified systems | Expect self-validation and potentially no Theia support for capture-quality issues |

#### Intrinsics for the Theia3D pilot

The safest Theia export begins after M0b:

```text
three side-by-side unrectified ZED H.265 streams
  -> external-index/pts_ns replay
  -> fixed left/right split
  -> ZED rectification
  -> six rectified, synchronized videos
  -> Theia custom intrinsics generated from each rectified P matrix
  -> zero distortion coefficients
  -> Theia chessboard extrinsic calibration
```

This avoids a distortion-model mismatch. ZED raw calibration can expose more Brown-Conrady coefficients than the Theia custom-calibration schema stores, whereas rectified ZED images use a pinhole model with distortion already removed.

For each rectified eye, map:

```text
P[0,0] -> focalLengthU
P[1,1] -> focalLengthV
P[0,2] -> centerPointU
P[1,2] -> centerPointV
skew = 0
radialDistortion1-3 = 0
tengentialDistortion1-2 = 0  # spelling used by the Theia schema
sensor bounds = exact rectified output dimensions
```

Populate the Theia `focallength` field from the corresponding eye's
`raw_calibration.<eye>.focal_length_mm`; the lens focal length is therefore no
longer a missing input. Each eye still needs a unique Theia camera ID. The Theia
chessboard recording must use the same rectified resolution as the movement
videos, and overlapping groups of at least three views should see the board
throughout the trial.

#### Go/no-go pilot

Before committing the dataset to Theia3D:

1. Export a 10-20 second six-view trial plus a chessboard calibration trial.
2. Confirm that Theia loads all six files, reports identical resolution, frame rate, frame count, and zero frame offset.
3. Confirm successful custom-intrinsic import for all six eye IDs.
4. Calibrate the array and inspect reprojection and diagonal-length errors.
5. Run one simple walking trial and inspect per-view overlays, joint visibility, and 3D trajectory.
6. Compare Theia's 3D output against the M3 geometric baseline and a measured wand/board.

Pass the pilot only if there are no unexplained frame offsets, all six views load, board calibration is stable, and 3D errors meet the project acceptance gates. Otherwise, continue with the native research lane; do not add cameras or transcode the full dataset before identifying the failed requirement.

### Coordinate-frame convention

Use one canonical project convention:

```text
World: right-handed, z-up, x-forward, meters
Camera pose stored as: T_world_cameraLeft
Projection uses: T_cameraLeft_world = inverse(T_world_cameraLeft)
SMPL root translation: meters in world coordinates
Quaternion order: explicitly declared, never inferred
Timestamp: integer nanoseconds in the declared P2P-LAN-synchronized clock domain
```

For a world point `X_w`, project it into camera `c` as:

<div class="math-display">
\[
\widetilde{\mathbf{x}}_c
\sim
\mathbf{K}_c
\begin{bmatrix}
\mathbf{R}_{c\leftarrow w} & \mathbf{t}_{c\leftarrow w}
\end{bmatrix}
\widetilde{\mathbf{X}}_w.
\]
</div>

The ZED default image frame is right-handed with `x` right, `y` down, and `z`
forward. Convert the recorded calibration explicitly into the canonical project
frame and unit convention. See
[ZED coordinate frames](https://docs.stereolabs.com/docs/development/zed-sdk/modules/positional-tracking/coordinate-frames).

In the canonical calibration package, define `T_right_left` as the transform that maps a 3D point from the left optical frame into the right optical frame. This removes the common ambiguity in names such as `T_left_right`.

For fixed cameras, use the stored target-based transform rather than allowing
independent visual-inertial estimates to drift.

### M0 acceptance gates

The following are recommended project targets, not guarantees from the camera vendor:

| Test | Initial target |
|---|---:|
| Held-out target reprojection RMSE | `< 0.5 px` preferred; investigate `> 1.0 px` |
| Pairwise transform loop closure | translation `< 5 mm`, rotation `< 0.2 deg` |
| Known wand/target 3D length error in capture volume | median `< 10 mm` |
| Floor-plane error at measured checkpoints | `< 10 mm` |
| Measured inter-camera exposure skew | `< 5 ms` |
| Side-by-side split or shared-sample identity mismatch | zero accepted mismatches |
| External `pts_ns`/`sample_index` discontinuities | zero unexplained events |
| Dropped/duplicated frames in a 10-minute test | zero unexplained events |
| Static-target 3D jitter | median `< 5 mm` |
| Rectified left/right vertical epipolar residual | median `< 0.5 px` preferred |

Also generate heatmaps of reprojection and 3D error over the full capture volume. A single global mean can conceal a poorly calibrated corner. Repeat the test at the deployed H.265 setting and at a high-quality reference setting; the difference measures compression-induced geometry loss.

## Shared data contracts

Every milestone should write an immutable, versioned result so that later stages can be replayed without rerunning earlier models.

### Synchronized frame bundle

```yaml
timestamp_ns: 123456789
calibration_id: room_a_2026_07_23_v1
views:
  - serial: 12345678
    sample_index: 4172
    stereo_pair_id: 4172
    host_utc_time_ns: 123456789
    host_monotonic_ns: ...
    zed_device_time_ns: ...
    pts_ns: ...
    side_by_side_h265:
      uri: ...
      decoded_width: 3840
      decoded_height: 1200
    left_source_roi_xywh: [0, 0, 1920, 1200]
    right_source_roi_xywh: [1920, 0, 1920, 1200]
    source_pixel_convention: raw_unrectified
    source_color_encoding: bt709_limited_nv12_420
    rgb_left_rectified: ...
    rgb_right_rectified: ...
    depth_left_m: optional
    depth_source: none|zed_sdk_stream|sgbm|raft_stereo|foundation_stereo
    depth_confidence: optional
    T_world_camera_left: [4x4]
    sync_method: p2p_lan
    clock_domain: ...
    clock_offset_ms: ...
    sync_residual_ms: ...  # measured from a visual timing event
    stereo_pair_valid: true
    valid: true
```

### 2D detection

```yaml
view_serial: 12345678
local_detection_id: 9
bbox_xyxy: [...]
keypoint_layout: coco_wholebody_133
keypoints_xy: [[x, y], ...]
keypoint_score: [...]
keypoint_covariance_px2: [[[...]], ...]
visible: [...]
appearance_embedding: [...]
mask_rle: optional
```

### 3D skeleton observation

```yaml
timestamp_ns: 123456789
association_id: 12
joint_layout: project_body_v1
joints_world_m: [[x, y, z], ...]
joint_covariance_m2: [[[...]], ...]
source_views_per_joint: [[serial_a, serial_b], ...]
reprojection_rmse_px: [...]
triangulation_angle_deg: [...]
valid: [...]
```

### Track and mesh output

```yaml
track_id: session_uuid_track_0007
timestamp_ns: 123456789
track_state: confirmed
root_world_m: [x, y, z]
root_velocity_world_mps: [vx, vy, vz]
smpl_model: neutral
smpl_betas: [...]
smpl_body_pose: [...]
smpl_global_orient_world: [...]
smpl_translation_world_m: [...]
mesh_vertices_world_m: optional
quality:
  source_view_count: 3
  joint_rmse_m: 0.018
  reprojection_rmse_px: 0.72
  depth_rmse_m: 0.014
```

## M1: reliable 2D joints per camera

### Goal

For every synchronized view, detect each person and return 2D joints with confidence, covariance, visibility, and a stable joint convention.

### Recommended baseline

Use a top-down pipeline:

1. Decode and rectify the left image using M0's fixed mapping.
2. Run a person detector.
3. Crop each person at a sufficiently high resolution.
4. Run RTMPose-L/RTMW or ViTPose++.
5. Retain heatmaps or SimCC distributions to estimate uncertainty.
6. Map the model-specific skeleton into a project skeleton that includes shoulders, hips, knees, ankles, heels, and toes.

[RTMPose](https://arxiv.org/abs/2303.07399) is the recommended real-time starting point because it has strong published speed/accuracy and an established MMPose implementation. Use [RTMW](https://arxiv.org/abs/2407.08634) when a real-time 133-keypoint whole-body layout is required. [ViTPose++](https://arxiv.org/abs/2212.04246) is a useful higher-accuracy comparison.

Run M1 on each decoded and rectified left image. Retain the decoded right image
for optional stereo depth. Running a second pose model on the right image is
optional; its short baseline adds less geometric information than another
camera location and doubles pose-inference cost.

The 2026 [Sapiens2](https://arxiv.org/abs/2604.21681) pose model is a frontier, high-resolution offline benchmark. It is not the default implementation choice: it is large, recently released, and its custom license explicitly restricts surveillance and biometric uses. Review the [Sapiens2 license](https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md) against the intended application before downloading or using it.

### Reliability improvements

- Batch all three synchronized views for GPU efficiency.
- Use a hardware H.265 decoder and keep `sample_index`, external `pts_ns`, and source timing metadata outside the image tensor.
- Preserve heatmap-based uncertainty rather than thresholding everything to valid/invalid.
- Benchmark the deployed H.265 setting against a high-quality reference clip. Report keypoint EPE/AP change versus bitrate, GOP, and motion.
- Fine-tune on 500-2,000 labeled frames from the real room, emphasizing occlusion, floor poses, fast motion, and image edges.
- Include feet; missing or unstable ankles harm calibration, world placement, and contact constraints later.
- Use temporal crop propagation only to improve crop stability, not to invent a missing joint silently.
- Record the exact pixel convention: original rectified image coordinates, not crop coordinates.

### Output

Per-view person boxes, joints, confidence/covariance, visibility, optional mask, and local detection ID.

### M1 acceptance gates

- COCO-style keypoint AP/PCK on a manually labeled room-specific set.
- Per-joint pixel EPE and failure rate.
- Static-person jitter in pixels.
- Visibility-stratified accuracy: visible, self-occluded, and externally occluded.
- Multi-view epipolar residual before any triangulation.
- At least 95% full-body detection recall in the intended central volume under the target person count.

Do not promote M1 based only on aggregate AP. Plot wrists, ankles, and feet separately.

## M2: same-person association across views

### Goal

At one synchronized timestamp, group detections from different cameras that belong to the same physical person. Enforce at most one detection from each camera in a group.

### Recommended modular algorithm

Build a multipartite graph whose nodes are per-view detections. Score each cross-view edge with:

1. **Epipolar pose consistency:** symmetric point-to-epipolar-line distance over reliable torso/limb joints.
2. **Optional 3D root consistency:** if stereo depth has passed validation, transform a per-ZED pelvis/torso estimate into the world frame and compare positions; otherwise omit this term.
3. **Triangulation validity:** positive depth, reprojection residual, ray angle, and plausible bone lengths after tentative triangulation.
4. **Appearance similarity:** a person ReID embedding used as a secondary cue.
5. **Height and color cues:** only when calibrated and reliable.

Normalize every term by uncertainty. Then solve a constrained multi-way matching problem rather than applying independent pairwise Hungarian matches, which can create inconsistent triplets.

The classic [fast and robust multi-way matching method](https://arxiv.org/abs/1901.04111) combines geometry and appearance and is a strong conceptual baseline. The [4D Association Graph](https://arxiv.org/abs/2002.12625) jointly reasons across image, view, and time; it is a valuable reference if M2 and M4 later need to be coupled.

### Why appearance is secondary

In synchronized overlapping views, calibrated geometry should dominate. Clothing appearance changes with view, lighting, and occlusion; visually similar people can cross. Appearance is most helpful when only weak geometry is available or during re-entry after a gap.

### Output

A set of cross-view association groups, the members from each camera, association cost components, and a confidence score.

### M2 acceptance gates

- Cross-view association precision, recall, and F1.
- Group purity and completeness.
- False merge rate for people standing close together.
- False split rate under partial occlusion.
- Association accuracy stratified by two-view versus three-view visibility.
- Reprojection residual of the resulting tentative 3D skeletons.

Create adversarial tests with similar clothing, people crossing, and one camera temporarily missing.

## M3: 3D joints in world coordinates

### Goal

Fuse associated 2D joints into metric 3D positions in the declared world frame and attach a meaningful uncertainty to every joint.

### Recommended baseline

For each joint:

1. Select visible observations with adequate confidence.
2. Initialize with weighted linear DLT triangulation.
3. Use RANSAC over view pairs when three or more useful rays are present.
4. Refine with nonlinear reprojection-error minimization.
5. Reject solutions behind a camera or outside the capture volume.
6. Reject or down-weight narrow ray-intersection angles.
7. Estimate 3D covariance from the projection Jacobian and 2D covariance.
8. Apply a light kinematic refinement with robust bone-length and joint-limit terms.

Optimize:

<div class="math-display">
\[
\mathbf{X}^{*}_{j}
=
\underset{\mathbf{X}_{j}}{\arg\min}
\sum_{c \in \mathcal{V}_{j}}
\rho\left(
\left(\pi_c(\mathbf{X}_{j})-\mathbf{x}_{c,j}\right)^\top
\mathbf{\Sigma}_{c,j}^{-1}
\left(\pi_c(\mathbf{X}_{j})-\mathbf{x}_{c,j}\right)
\right),
\]
</div>

where `rho` is a robust loss, `pi_c` is the calibrated world-to-image projection, and `Sigma` is the 2D joint covariance.

### Stereo-depth fallback

Stereo depth is an optional subproject. Replay each indexed side-by-side sample,
split both eyes, rectify them, and estimate disparity externally. Use StereoSGBM
as a transparent baseline, [RAFT-Stereo](https://arxiv.org/abs/2109.07547) or
[IGEV-Stereo](https://arxiv.org/abs/2303.06615) as established learned
comparisons, and [FoundationStereo](https://arxiv.org/abs/2501.09898) or
[Fast-FoundationStereo](https://arxiv.org/abs/2512.11130) as current zero-shot
research candidates.

Convert valid disparity to depth with the rectified focal length and baseline, or the stored `Q` matrix. Apply left-right consistency, occlusion, disparity-range, person-mask, and confidence checks. Validate depth on measured planes and a wand at several ranges before using it in M3 or M6.

If only one wide-baseline camera observes a joint and its stereo depth is validated, sample depth near the joint using the person mask and depth confidence. Transform that point into the world frame and mark it as a **single-ZED stereo-depth fallback** with larger covariance. If depth is not validated, leave the joint missing or predict it later with the body model; do not insert unverified depth.

For two views, triangulation is possible but has no third view for outlier rejection. The geometry and M1 confidence gates must therefore be stricter.

### Learned research alternatives

- [VoxelPose](https://arxiv.org/abs/2004.06239) aggregates multi-view features in a common voxel volume and avoids explicit 2D association.
- [MvP](https://arxiv.org/abs/2111.04076) directly regresses multi-person 3D poses with projective attention.
- [MVGFormer](https://arxiv.org/abs/2311.10983) alternates learning-free geometry with learned appearance reasoning.
- [MV-SSM](https://arxiv.org/abs/2509.00649) is the strongest recent research candidate for a three-camera layout because its CVPR 2025 paper specifically reports camera-arrangement generalization and a three-camera evaluation.

These methods should be benchmarked after the geometric baseline. Confirm their skeleton definitions, camera model, training layouts, scene volume, and license before integrating them.

### Output

World-coordinate joints, covariance, source cameras, triangulation angle, reprojection residual, fallback type, and validity.

### M3 acceptance gates

- Raw world-coordinate MPJPE against a marker-based or surveyed reference.
- Root-aligned MPJPE as a separate diagnostic, not a replacement for world error.
- Per-joint reprojection RMSE.
- Bone-length variance over a track.
- 3D jitter for a stationary subject.
- Failure rate versus camera count and triangulation angle.
- Root-position error throughout a measured floor grid.

The repository's [Pose Metrics Reference](human_pose_metrics_reference.md) defines MPJPE, root alignment, PCK, and mesh metrics in more detail.

## M4: stable trajectory and persistent ID

### Goal

Produce a smooth, metric world trajectory and maintain an anonymous ID through ordinary occlusion and view changes.

### Recommended baseline

Track after 3D fusion, not independently in each image.

Use a world-space state such as:

```text
pelvis position and velocity
torso orientation
selected relative 3D joints
estimated height
appearance gallery
state covariance
```

For each frame:

1. Predict each active track with a constant-velocity Kalman filter or unscented Kalman filter.
2. Gate candidates by Mahalanobis distance in world space.
3. Score matches using root distance, articulated pose similarity, height, source-view overlap, and appearance.
4. Solve one-to-one assignment with Hungarian/Jonker-Volgenant matching.
5. Update confirmed tracks with robust observations.
6. Maintain tentative, confirmed, occluded, and terminated lifecycle states.
7. Re-identify after a gap only inside a physically reachable region and with sufficient appearance/shape evidence.
8. Run a Rauch-Tung-Striebel smoother offline; use causal filtering online.

Keep filtering and identity separate. A smooth trajectory can still have the wrong ID, and aggressive smoothing can conceal an ID switch.

### Research alternatives

- [TEMPO](https://openaccess.thecvf.com/content/ICCV2023/html/Choudhury_TEMPO_Efficient_Multi-View_Pose_Estimation_Tracking_and_Forecasting_ICCV_2023_paper.html) jointly estimates, tracks, and forecasts multi-view pose using a recurrent spatiotemporal representation.
- The 2026 preprint [Efficient Online 3D Multi-Camera Multi-Object Tracking and Pose Estimation](https://arxiv.org/abs/2604.16522) is a relevant frontier comparison because it uses only 2D boxes/poses and explicitly handles cameras disconnecting and reconnecting.
- [Deep OC-SORT](https://arxiv.org/abs/2302.11813) is a useful appearance/motion reference for per-view tracking, but the final authoritative identity should remain the world-space track.

### Identity boundary

`track_id` should be a random, session-scoped identifier. Persistent ID here means continuity within a capture session. Cross-session recognition or real-person identification is a different, privacy-sensitive problem and is not required for M4.

### Output

Track ID, filtered root/velocity, filtered 3D joints and covariance, track state, age, last observation time, and association confidence.

### M4 acceptance gates

- HOTA and IDF1.
- ID switches per minute.
- Track fragmentation and mostly-tracked percentage.
- Trajectory position/velocity error.
- Reacquisition accuracy after 0.5 s, 1 s, and 3 s occlusions.
- Latency added by causal filtering.
- Accuracy when one camera drops out.

## M5: SMPL pose fitted to the 3D skeleton

### Goal

Estimate per-frame SMPL articulation, global orientation, and translation for every confirmed 3D track. Body shape is provisional in M5 and becomes a sequence-level quantity in M6.

### Required inputs

- Tracked 3D joints and covariance from M4.
- A declared mapping from the project skeleton to the SMPL joint regressor.
- World floor plane.
- A licensed SMPL model file and differentiable implementation.
- Optional per-view 2D joints for an additional reprojection term.

SMPL represents a body with pose parameters, shape coefficients, and a skinned mesh. See the original [SMPL paper](https://virtualhumans.mpi-inf.mpg.de/papers/SMPL15/SMPL15.pdf) and the [official SMPL site](https://smpl.is.tuebingen.mpg.de/). Model files require registration and license review.

### Recommended staged optimizer

For each track:

1. Initialize translation from the pelvis.
2. Initialize yaw/global orientation from the shoulder and hip axes.
3. Optimize root translation and global orientation with torso joints.
4. Add major limbs while holding provisional shape near the mean.
5. Optimize all body-pose parameters with a pose prior.
6. Refine in a short temporal window using the previous solution as initialization.
7. Mark rather than conceal frames whose residual stays above the failure threshold.

The core objective is:

<div class="math-display">
\[
\begin{aligned}
E(\theta_t,\beta,\mathbf{R}_t,\mathbf{t}_t)
=\;&
\lambda_{3D}
\sum_j
\rho\left(
\left(\mathbf{J}_j(\theta_t,\beta,\mathbf{R}_t,\mathbf{t}_t)
-\widehat{\mathbf{J}}_{t,j}\right)^\top
\mathbf{\Sigma}_{t,j}^{-1}
\left(\mathbf{J}_j-\widehat{\mathbf{J}}_{t,j}\right)
\right) \\
&+\lambda_{2D}E_{\mathrm{reproj}}
+\lambda_{\mathrm{pose}}E_{\mathrm{pose\ prior}}
+\lambda_{\mathrm{limits}}E_{\mathrm{joint\ limits}}
+\lambda_{\mathrm{temp}}E_{\mathrm{temporal}}.
\end{aligned}
\]
</div>

Use a robust loss and cap the influence of uncertain or fallback joints. Do not optimize equally against every keypoint score.

### Practical references

- [EasyMocap](https://github.com/zju3dv/EasyMocap) provides a working multi-view SMPL/SMPL-X fitting reference.
- [XRMoCap](https://github.com/openxrlab/xrmocap) supports keypoint-based and parametric multi-view motion-capture methods.
- [SMPLify](https://arxiv.org/abs/1607.08128) and [SMPLify-X](https://arxiv.org/abs/1904.05866) define the optimization-based fitting pattern and pose priors.
- [HuMoR](https://geometry.stanford.edu/projects/humor/) is a useful motion prior for fitting pose and shape to noisy or partial 3D observations.

### Output

Per-frame `global_orient`, `body_pose`, `translation_world`, provisional `betas`, joint-fit residuals, and a mesh-fit validity flag.

### M5 acceptance gates

- SMPL-regressed joint error against the M4 skeleton.
- MPJPE/PA-MPJPE against an independent 3D reference.
- Per-view 2D reprojection error.
- Joint-angle and pose-prior violations.
- Angular velocity/acceleration discontinuity.
- Failure rate under missing wrists/ankles and self-occlusion.

## M6: stable body shape and world-coordinate mesh

### Goal

Estimate one stable body shape per track and generate temporally coherent SMPL meshes in the common world frame.

### Important observability limit

Sparse 3D joints constrain bone layout but do not fully determine body surface shape. M6 therefore needs additional observations:

- Multi-view person silhouettes.
- Optional validated stereo depth/point clouds with confidence and visibility handling.
- Diverse high-quality poses across time.
- A shape prior.
- Floor/contact evidence.

Without these, the system can produce a plausible SMPL body but cannot claim accurate body girth or surface geometry.

For the current H.265 data, begin M6 with multi-view silhouettes and a shared shape prior. Set the depth weight to zero until the decoded stereo-depth branch passes its metric validation. This preserves a clean path to a stable world mesh without making M6 depend on unverified compressed-stereo correspondence.

### Recommended sequence-level optimization

Use a shared shape vector `beta_i` for track `i`, while pose, global orientation, and translation vary per frame:

<div class="math-display">
\[
\underset{
\beta_i,\{\theta_{i,t},\mathbf{R}_{i,t},\mathbf{t}_{i,t}\}
}{\arg\min}
\sum_t
\left[
\lambda_J E_{\mathrm{3D\ joints}}
+\lambda_S E_{\mathrm{silhouette}}
+\lambda_D E_{\mathrm{depth}}
+\lambda_{2D} E_{\mathrm{reprojection}}
+\lambda_C E_{\mathrm{contact/floor}}
\right]
+\lambda_\beta E_{\mathrm{shape\ prior}}
+\lambda_T E_{\mathrm{temporal}}.
\]
</div>

Procedure:

1. Select keyframes with three-view coverage, high joint confidence, varied poses, and clean silhouettes.
2. Estimate one initial `beta` from the keyframes.
3. Jointly optimize shared `beta` and per-frame pose/root variables.
4. Use robust silhouette distance-transform loss rather than only bounding boxes.
5. If the depth branch is validated, compare visible rendered vertices with masked stereo depth using a robust point-to-plane or depth residual.
6. Exclude depth edges, occluded vertices, reflective regions, and clothing outliers.
7. Add ground non-penetration and foot-contact constraints.
8. Freeze shape after sufficient evidence; update only slowly and only from high-quality frames.
9. Refit pose with the stable shape fixed.

World vertices are:

<div class="math-display">
\[
\mathbf{V}^{world}_{i,t}
=
\mathbf{R}_{i,t}\,
\mathbf{V}_{SMPL}(\theta_{i,t},\beta_i)
+\mathbf{t}_{i,t}.
\]
</div>

### Research alternatives

- [HeatFormer](https://openaccess.thecvf.com/content/CVPR2025/html/Matsubara_HeatFormer_A_Neural_Optimizer_for_Multiview_Human_Mesh_Recovery_CVPR_2025_paper.html) is a strong 2025 multi-view HMR initializer: it iteratively refines SMPL from a variable number of views and is designed to generalize across camera configurations.
- [Towards Accurate Markerless Human Shape and Pose Estimation over Time](https://arxiv.org/abs/1707.07548) remains directly relevant because it combines multi-view keypoints, silhouettes, and temporal regularization.
- [IPMAN](https://ipman.is.tue.mpg.de/) provides differentiable floor-contact and stability ideas to reduce floating and ground penetration.
- [HuMMan](https://arxiv.org/abs/2204.13686) and [Hi4D](https://arxiv.org/abs/2303.15380) are useful evaluation/training references for multi-modal meshes and close human interaction.

HeatFormer is a benchmark/initializer, not a substitute for the shared-shape temporal optimization above. Its published formulation produces view-dependent estimates and does not by itself enforce one shape over the entire track.

### Output

- One stable SMPL `beta` vector per track.
- Per-frame world-space pose/root transform.
- World-space vertices, joints, and faces.
- Optional texture only as a separate, consented processing stage.
- Quality flags for shape maturity, view coverage, silhouette fit, and depth fit.

### M6 acceptance gates

- Shape-parameter drift after the track is mature.
- PVE/MPVPE against a scan or trusted mesh where available.
- Multi-view silhouette IoU and contour error.
- Masked visible-surface depth RMSE.
- Foot skating and floor penetration.
- Vertex acceleration/jitter for a stationary subject.
- World root/mesh translation accuracy.
- Shape consistency after clothing/view changes.

## Milestone dependency and promotion table

| Milestone | Inputs | Primary deliverable | Do not promote if |
|---|---|---|---|
| M0a | Three side-by-side H.265 streams + YAML/NPZ/index metadata | Valid split eye pairs and three-camera frame bundles | External index/replay order or cross-rig timing is ambiguous |
| M0b | Decoded pairs, calibration, target, measured room | Rectification and versioned world calibration | World points shift with view or motion |
| M1 | Decoded, synchronized, rectified left RGB | 2D joints + covariance | Feet/occluded limbs fail systematically |
| M2 | M1 + calibration + optional depth/ReID | Cross-view person groups | Close people merge or triplets are inconsistent |
| M3 | M2 + projection matrices | World 3D joints + covariance | Reprojection is good but surveyed world error is poor |
| M4 | M3 + time | Trajectory + session ID | ID switches are hidden by smoothing |
| M5 | M4 + SMPL model/prior | Per-frame SMPL pose | Joint fit is plausible only after world alignment |
| M6 | M5 + masks + optional validated depth + sequence | Stable shape + world mesh | Shape changes frame-to-frame or fits clothing noise |

## Recommended implementation order

### Phase 1: rig truth before model research

1. Inventory the three elementary H.265 streams, encoder settings, sample counts, and associated `rgb/meta.yaml`, `rgb/meta.npz`, and `rgb/frames.idx`.
2. Implement M0a indexed replay and build the `(serial_number, sample_index)` stereo/camera manifest.
3. Assert that each decoded sample is 3840x1200, split the fixed 1920x1200 eye regions, and verify the documented color conversion and absence of spatial transforms.
4. Retrieve and archive the matching raw 1920x1200 calibration snapshot for each serial number.
5. Decode and rectify a calibration/LED sequence through the actual H.265 path.
6. Verify common exposure timing, stereo vertical alignment, target reprojection, and array extrinsics.
7. Create the canonical calibration package and automated M0 tests.

Deliverable: external-index/`pts_ns`-correct H.265 replay with verified rectification and `T_world_camera`, plus a visualizer that overlays a projected 3D target into all views.

### Phase 2: modular skeleton baseline

1. Implement M1 with RTMPose and retain uncertainty.
2. Implement M2 geometry-first multi-way matching.
3. Implement weighted robust M3 triangulation.
4. Visualize every 3D skeleton reprojected into all source images.

Deliverable: offline 3D skeletons with per-joint provenance and residuals.

Do not wait for dense stereo depth to begin this phase. Use the three rectified left views.

### Phase 3: identity and trajectory

1. Add world-space Kalman filtering and lifecycle management.
2. Add pose and appearance cues only after spatial gating.
3. Build scripted crossing/occlusion/camera-drop tests.
4. Add offline smoothing as a separate export step.

Deliverable: stable session IDs and trajectories with HOTA/IDF1 evaluation.

### Phase 4: SMPL pose and stable shape

1. Fit SMPL pose to clean single-person tracks.
2. Add multi-person track batching.
3. Add silhouettes.
4. Validate custom decoded stereo depth on measured geometry.
5. Add only validated depth with occlusion-aware residuals.
6. Optimize one shape per track across selected keyframes.
7. Add floor/contact terms.

Deliverable: world-coordinate SMPL mesh sequences and fit-quality reports.

### Phase 5: research upgrades and real-time optimization

1. Benchmark ViTPose++ against RTMPose for M1.
2. Benchmark MV-SSM or TEMPO against modular M2-M4.
3. Benchmark HeatFormer as an M5/M6 initializer.
4. Fine-tune only after error attribution identifies a model bottleneck.
5. Export inference models to TensorRT where supported.
6. Move from offline playback to the live synchronized stream without changing data contracts.

## Experiment design and ground truth

Create a small, deliberately difficult in-room benchmark:

| Sequence | Purpose |
|---|---|
| One stationary person at floor-grid points | Calibration, world accuracy, jitter |
| Slow walk around perimeter and center | Coverage and trajectory |
| Fast arm/leg motion | Synchronization and M1/M3 |
| Sitting, crouching, lying, floor contact | View coverage and pose priors |
| Two people crossing | M2/M4 identity |
| Similar clothing | Geometry versus appearance |
| Partial/full occlusion | Track lifecycle and re-identification |
| One camera disabled | Graceful degradation |
| Loose clothing and reflective fabric | Depth/shape robustness |
| Calibration target/wand throughout volume | M0 truth |

For quantitative 3D ground truth, preferred options are:

1. Marker-based motion capture registered to the same world frame.
2. Surveyed static points and a tracked rigid wand for M0/root validation.
3. A small number of manually verified multi-view frames and body measurements.

Public research benchmarks do not replace the room-specific set. [CMU Panoptic Studio](https://www.cs.cmu.edu/~hanbyulj/panoptic-studio/) is valuable for multi-person multi-view pose, but its camera count and appearance differ greatly from three ZED X cameras.

## Method selection summary

| Stage | Practical first choice | Strong comparison | Frontier/research note |
|---|---|---|---|
| H.265 ingest | Hardware decode + external `sample_index`/`pts_ns` manifest + fixed split | Independent decoder/replay validation | Preserve the external index and shared stereo-sample identity |
| Rectification | Session raw calibration + fixed maps | Target-refined calibration | Never mix raw pixels and rectified intrinsics |
| M1 | RTMPose-L/RTMW | ViTPose++ | Sapiens2 is high accuracy but large and license-restricted |
| M2 | Geometry + optional depth/ReID multi-way graph | Dong et al. multi-way matching | 4D Association couples view and time |
| Stereo depth | StereoSGBM | RAFT/IGEV-Stereo | FoundationStereo/Fast-FoundationStereo after metric validation |
| M3 | Robust uncertainty-weighted triangulation of three left views | VoxelPose/MVGFormer | MV-SSM is the most relevant recent three-camera candidate |
| M4 | 3D Kalman + assignment + appearance gallery | TEMPO | 2026 Bayesian online tracker is worth reproducing |
| M5 | Differentiable SMPL fitting to M4 joints | EasyMocap/XRMoCap | HuMoR supplies a strong motion prior |
| M6 | Shared-shape sequence optimization with masks/depth | HeatFormer initialization | Add IPMAN-style contact constraints |

## Main risks and mitigations

| Risk | Symptom | Mitigation |
|---|---|---|
| External index/bitstream mismatch | Correct-looking playback but wrong sample identity or cross-camera phase | Validate decoded sample count, offsets, monotonic `sample_index`, and external `pts_ns` |
| Side-by-side split error | Left/right depth fails everywhere or eyes are swapped | Assert 3840x1200 decode and fixed ROIs before rectification |
| Accidental double rectification | Curved epipolar residual or lost field of view | Label the input unrectified and the M0b output rectified; enforce exactly one remap |
| Missing session calibration | Stereo alignment differs from the old recording | Recover by serial, validate, and flag uncertainty |
| H.265 bitrate too low | Blocky limbs and unstable stereo disparity | Bitrate/GOP ablation and high-quality calibration capture |
| Hidden resize/crop/padding | Systematic projection offset | Store coded/visible dimensions and transform intrinsics |
| Temporal skew | Limbs split in 3D only during motion | Log P2P-LAN clock offset/drift, estimate content-based residual skew, and reject invalid bundles |
| Extrinsic bias | All joints reproject consistently in one pair but world positions drift | Held-out target/wand and loop-closure tests |
| Raw/rectified mix-up | Systematic image-edge error | Encode image convention in every file |
| Coordinate inversion | Mirrored/rotated skeletons | Unit tests with known axes and round trips |
| Weak three-view coverage | Frequent single-view fallbacks | Reposition cameras or shrink declared ROI |
| Similar-looking people | ID swaps at crossings | Geometry-first association and 3D lifecycle |
| Over-smoothing | Low jitter but delayed/incorrect fast motion | Separate causal tracking from offline smoothing |
| Shape fitted from joints only | Plausible but inaccurate girth | Add silhouettes, depth, and shared sequence shape |
| Clothing fitted as body | Shape grows/shrinks with outfit | Robust depth/mask loss and strong shape prior |
| Floor penetration/skating | Visually unstable mesh | Calibrated floor plus contact constraints |
| Model/camera-layout overfit | Public benchmark good, room poor | Room-specific validation and geometric baseline |
| Licensing/privacy mismatch | Model cannot be deployed for intended use | Review model, code, dataset, and body-model licenses early |
| Theia treats paired eyes as weakly diverse views | Six files load but depth/occlusion performance resembles only three views | Keep Theia as a pilot, audit joint visibility, and compare against the native M3 baseline |
| Theia codec incompatibility | H.265 MP4 is rejected or decoded differently | Test a short remux first; otherwise export synchronized high-quality H.264 YUV 4:2:0 |

## Definition of the final system

The system is complete when, for every valid timestamp and confirmed person, it can provide:

```text
anonymous persistent track ID
world-space root position, velocity, and uncertainty
world-space 3D skeleton and per-joint uncertainty
SMPL pose and one stable per-track body shape
world-space mesh vertices/faces
source cameras and data-quality flags
reprojection, depth, and fitting residuals
```

The final output should always distinguish:

- measured versus predicted joints,
- multi-view triangulation versus single-ZED stereo-depth fallback,
- causal live estimates versus offline-smoothed results,
- a plausible SMPL body versus a metrically validated body surface.

That distinction is essential for scientific use and for diagnosing future radar-to-camera comparisons.

## Sources and implementation references

### ZED and calibration

- [Stereolabs: ZED X Stereo Camera](https://www.stereolabs.com/store/products/zed-x-stereo-camera)
- [Stereolabs: Setting Up Multiple 3D Cameras](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/multi-camera)
- [Stereolabs: Camera Calibration](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/camera-calibration)
- [Stereolabs: Coordinate Frames](https://docs.stereolabs.com/docs/development/zed-sdk/modules/positional-tracking/coordinate-frames)
- [OpenCV: ChArUco Calibration](https://docs.opencv.org/trunk/da/d13/tutorial_aruco_calibration.html)
- [Perceptron Perspectives: Camera Calibration Explained](https://medium.com/perceptron-perspectives/camera-calibration-explained-enhancing-accuracy-in-computer-vision-applications-8ad1494cc5f2)
- [CMU: Measuring Human Motion in Social Interactions, RGB-D calibration procedure](https://publications.ri.cmu.edu/storage/publications/2018/01/Measuring-Human-Motion-in-Social-Interactions.pdf)
- [CMU Panoptic Studio: spatial calibration and RGB-D alignment](https://s3-eu-west-1.amazonaws.com/pstorage-cmu-348901238291901/14343020/h_joo_robotics_2019.pdf)

### Theia3D compatibility

- [Theia3D: Camera System Requirements](https://docs.theiamarkerless.com/theia3d-documentation/camera-system-requirements)
- [Theia3D: Can I Use My Own Cameras?](https://www.theiamarkerless.com/faq/can-i-use-my-own-cameras)
- [Theia3D: Video Data Format](https://docs.theiamarkerless.com/theia3d-documentation/data-formats/video-data)
- [Theia3D: Supported H.264 Chroma Formats](https://docs.theiamarkerless.com/troubleshooting/error-messages/load-video-data-errors/invalid-video)
- [Theia3D: Calibration Files](https://docs.theiamarkerless.com/theia3d-documentation/data-formats/calibration-files)
- [Theia3D: Chessboard Calibration](https://docs.theiamarkerless.com/theia3d-documentation/theia3d-dropdown-menus/calibration-menu/chessboard-calibration)

### Stereo depth from decoded eyes

- [RAFT-Stereo](https://arxiv.org/abs/2109.07547)
- [IGEV-Stereo](https://arxiv.org/abs/2303.06615)
- [FoundationStereo](https://arxiv.org/abs/2501.09898)
- [Fast-FoundationStereo](https://arxiv.org/abs/2512.11130)

### Pose, association, and tracking

- [RTMPose](https://arxiv.org/abs/2303.07399)
- [RTMW](https://arxiv.org/abs/2407.08634)
- [ViTPose++](https://arxiv.org/abs/2212.04246)
- [Sapiens2](https://arxiv.org/abs/2604.21681)
- [Fast and Robust Multi-Person 3D Pose Estimation from Multiple Views](https://arxiv.org/abs/1901.04111)
- [4D Association Graph](https://arxiv.org/abs/2002.12625)
- [VoxelPose](https://arxiv.org/abs/2004.06239)
- [MvP](https://arxiv.org/abs/2111.04076)
- [MVGFormer](https://arxiv.org/abs/2311.10983)
- [MV-SSM](https://arxiv.org/abs/2509.00649)
- [TEMPO](https://arxiv.org/abs/2309.07910)
- [Efficient Online 3D Multi-Camera Multi-Object Tracking and Pose Estimation](https://arxiv.org/abs/2604.16522)

### SMPL and mesh fitting

- [SMPL](https://virtualhumans.mpi-inf.mpg.de/papers/SMPL15/SMPL15.pdf)
- [SMPLify](https://arxiv.org/abs/1607.08128)
- [SMPLify-X](https://arxiv.org/abs/1904.05866)
- [EasyMocap](https://github.com/zju3dv/EasyMocap)
- [XRMoCap](https://github.com/openxrlab/xrmocap)
- [HuMoR](https://arxiv.org/abs/2105.04668)
- [HeatFormer](https://arxiv.org/abs/2412.04456)
- [IPMAN](https://arxiv.org/abs/2303.18246)
- [HuMMan](https://arxiv.org/abs/2204.13686)
- [Hi4D](https://arxiv.org/abs/2303.15380)
