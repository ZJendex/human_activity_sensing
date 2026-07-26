"""Portable comparison visualizations for GT, V10, and V3 skeletons."""

from __future__ import annotations

import base64
import csv
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .constants import COCO17_EDGES, COCO17_NAMES, PERSON_COLORS
from .depth_eval import near_body_surface_mask
from .evaluate import evaluate_frame
from .geometry import project_points
from .io import load_cameras, load_gt_coco17, read_jsonl
from .io import VideoReaderPool
from .model import Reconstruction


GROUND_Y_CM = 0.0
GRID_MINOR_CM = 50
GRID_MAJOR_CM = 100
GRID_HALF_SPAN_CM = 300


def _ground_grid_segments(
    center: Sequence[float],
    floor_y_cm: float = GROUND_Y_CM,
    minor_cm: int = GRID_MINOR_CM,
    major_cm: int = GRID_MAJOR_CM,
    half_span_cm: int = GRID_HALF_SPAN_CM,
) -> List[tuple[tuple[float, float, float], tuple[float, float, float], bool, bool]]:
    """Return world-aligned X/Z ground-grid lines.

    The two booleans mark major metre lines and the world X/Z axes,
    respectively.
    """

    center_array = np.asarray(center, dtype=float)
    x_min = int(np.floor((center_array[0] - half_span_cm) / minor_cm)) * minor_cm
    x_max = int(np.ceil((center_array[0] + half_span_cm) / minor_cm)) * minor_cm
    z_min = int(np.floor((center_array[2] - half_span_cm) / minor_cm)) * minor_cm
    z_max = int(np.ceil((center_array[2] + half_span_cm) / minor_cm)) * minor_cm
    segments = []
    for x in range(x_min, x_max + 1, minor_cm):
        segments.append(
            (
                (float(x), floor_y_cm, float(z_min)),
                (float(x), floor_y_cm, float(z_max)),
                x % major_cm == 0,
                x == 0,
            )
        )
    for z in range(z_min, z_max + 1, minor_cm):
        segments.append(
            (
                (float(x_min), floor_y_cm, float(z)),
                (float(x_max), floor_y_cm, float(z)),
                z % major_cm == 0,
                z == 0,
            )
        )
    return segments


def _people_for_json(
    people: Sequence[Mapping[str, Any]],
    kind: str,
    matched_gt_ids: Optional[Mapping[int, int]] = None,
) -> List[Dict[str, Any]]:
    output = []
    for person in people:
        if kind == "gt":
            joints = np.asarray(person["joints_cm"], dtype=float)
            valid = np.asarray(person["confidence"], dtype=float) > 0.1
            person_id = int(person["id"])
        else:
            reconstruction = Reconstruction.from_json(person)
            joints = reconstruction.joints_cm
            valid = reconstruction.joint_valid
            person_id = (
                reconstruction.track_id
                if reconstruction.track_id is not None
                else reconstruction.local_id
            )
            support = reconstruction.joint_support.astype(int).tolist()
            reprojection = [
                float(value) if np.isfinite(value) else None
                for value in reconstruction.reprojection_rmse_px
            ]
        serialized = [
            [float(v) for v in joint] if is_valid and np.all(np.isfinite(joint)) else None
            for joint, is_valid in zip(joints, valid)
        ]
        result = {
            "id": person_id,
            "colorId": (
                person_id
                if kind == "gt"
                else (matched_gt_ids or {}).get(len(output), person_id)
            ),
            "joints": serialized,
        }
        if kind != "gt":
            result["support"] = support
            result["reprojectionRmsePx"] = reprojection
        output.append(result)
    return output


def build_viewer_data(
    sequence_dir: Path,
    frame_table_path: Path,
    reconstruction_v10_path: Path,
    reconstruction_v3_path: Path,
    max_frames: Optional[int] = 300,
    colored_cloud_index_path: Optional[Path] = None,
    cloud_point_limit: int = 5_000,
    near_body_distance_cm: float = 35.0,
) -> Dict[str, Any]:
    cameras = load_cameras(sequence_dir)
    v10 = {int(row["hd_index"]): row for row in read_jsonl(reconstruction_v10_path)}
    v3 = {int(row["hd_index"]): row for row in read_jsonl(reconstruction_v3_path)}
    cloud_rows = (
        {
            int(row["hd_index"]): row
            for row in read_jsonl(colored_cloud_index_path)
        }
        if colored_cloud_index_path
        else {}
    )
    frames = []
    bounds_min = np.full(3, np.inf, dtype=float)
    bounds_max = np.full(3, -np.inf, dtype=float)
    full_bounds_min = np.full(3, np.inf, dtype=float)
    full_bounds_max = np.full(3, -np.inf, dtype=float)
    for frame in read_jsonl(frame_table_path):
        hd_index = int(frame["hd_index"])
        frame_bounds_min = np.full(3, np.inf, dtype=float)
        frame_bounds_max = np.full(3, -np.inf, dtype=float)
        gt = load_gt_coco17(Path(frame["gt_path"]))
        row10 = v10.get(hd_index, {})
        row3 = v3.get(hd_index, {})
        recon10 = [Reconstruction.from_json(item) for item in row10.get("people", [])]
        recon3 = [Reconstruction.from_json(item) for item in row3.get("people", [])]
        metrics10 = evaluate_frame(recon10, gt)
        metrics3 = evaluate_frame(recon3, gt)
        match10 = {item["pred_index"]: item["gt_id"] for item in metrics10["matches"]}
        match3 = {item["pred_index"]: item["gt_id"] for item in metrics3["matches"]}
        serialized_gt = _people_for_json(gt, "gt")
        serialized_v10 = _people_for_json(
            row10.get("people", []), "prediction", match10
        )
        serialized_v3 = _people_for_json(
            row3.get("people", []), "prediction", match3
        )
        for people in (serialized_gt, serialized_v10, serialized_v3):
            joints = [
                point
                for person in people
                for point in person["joints"]
                if point is not None
            ]
            if joints:
                points = np.asarray(joints, dtype=float)
                bounds_min = np.minimum(bounds_min, np.min(points, axis=0))
                bounds_max = np.maximum(bounds_max, np.max(points, axis=0))
                frame_bounds_min = np.minimum(
                    frame_bounds_min,
                    np.min(points, axis=0),
                )
                frame_bounds_max = np.maximum(
                    frame_bounds_max,
                    np.max(points, axis=0),
                )
        cloud = None
        cloud_row = cloud_rows.get(hd_index)
        if cloud_row is not None:
            cloud_path = Path(cloud_row["cloud_path"])
            if not cloud_path.exists() and colored_cloud_index_path is not None:
                cloud_path = colored_cloud_index_path.parent / cloud_path
            with np.load(cloud_path, allow_pickle=False) as cached:
                cloud_points = np.asarray(cached["xyz_cm"], dtype=np.float32)
                cloud_rgb = np.asarray(cached["rgb"], dtype=np.uint8)
            near = near_body_surface_mask(
                cloud_points,
                gt,
                maximum_distance_cm=near_body_distance_cm,
            )
            chosen = _select_cloud_points(
                near,
                maximum_points=cloud_point_limit,
            )
            cloud_points = cloud_points[chosen]
            cloud_rgb = cloud_rgb[chosen]
            near = near[chosen]
            if len(cloud_points):
                full_bounds_min = np.minimum(
                    full_bounds_min,
                    np.min(cloud_points, axis=0),
                )
                full_bounds_max = np.maximum(
                    full_bounds_max,
                    np.max(cloud_points, axis=0),
                )
            if np.any(near):
                bounds_min = np.minimum(
                    bounds_min,
                    np.min(cloud_points[near], axis=0),
                )
                bounds_max = np.maximum(
                    bounds_max,
                    np.max(cloud_points[near], axis=0),
                )
                frame_bounds_min = np.minimum(
                    frame_bounds_min,
                    np.min(cloud_points[near], axis=0),
                )
                frame_bounds_max = np.maximum(
                    frame_bounds_max,
                    np.max(cloud_points[near], axis=0),
                )
            cloud = {
                "packed": _pack_cloud_points(cloud_points, cloud_rgb, near),
                "count": int(len(cloud_points)),
                "nearCount": int(np.count_nonzero(near)),
                "sourceCount": int(cloud_row["point_count"]),
                "acceptedNodes": int(cloud_row["accepted_nodes"]),
                "temporalSpanMs": cloud_row.get("temporal_span_ms"),
                "quantizationCm": 0.1,
                "strideBytes": 10,
                "warnings": cloud_row.get("warnings", []),
            }
        frames.append(
            {
                "hd": hd_index,
                "time": float(frame["univ_time_ms"]),
                "bounds": {
                    "center": (
                        ((frame_bounds_min + frame_bounds_max) / 2.0).tolist()
                        if np.all(np.isfinite(frame_bounds_min))
                        else [0.0, 0.0, 0.0]
                    ),
                    "extent": (
                        max(
                            float(np.max(frame_bounds_max - frame_bounds_min)),
                            100.0,
                        )
                        * 1.08
                        if np.all(np.isfinite(frame_bounds_min))
                        else 600.0
                    ),
                },
                "gt": serialized_gt,
                "v10": serialized_v10,
                "v3": serialized_v3,
                "cloud": cloud,
                "metrics": {
                    "v10MpjpeMm": metrics10["mpjpe_mm"],
                    "v3MpjpeMm": metrics3["mpjpe_mm"],
                    "v10Availability": (
                        metrics10["reconstructed_gt_joints"]
                        / metrics10["eligible_gt_joints"]
                        if metrics10["eligible_gt_joints"]
                        else None
                    ),
                    "v3Availability": (
                        metrics3["reconstructed_gt_joints"]
                        / metrics3["eligible_gt_joints"]
                        if metrics3["eligible_gt_joints"]
                        else None
                    ),
                },
            }
        )
        if max_frames is not None and len(frames) >= max_frames:
            break
    tracking_extent = max(
        (float(frame["bounds"]["extent"]) for frame in frames),
        default=600.0,
    )
    for frame in frames:
        frame["bounds"]["extent"] = tracking_extent
    if np.all(np.isfinite(bounds_min)):
        center = (bounds_min + bounds_max) / 2.0
        extent = max(float(np.max(bounds_max - bounds_min)), 100.0) * 1.08
    else:
        center = np.zeros(3, dtype=float)
        extent = 600.0
    if np.all(np.isfinite(full_bounds_min)):
        full_center = (full_bounds_min + full_bounds_max) / 2.0
        full_extent = (
            max(float(np.max(full_bounds_max - full_bounds_min)), 100.0) * 1.08
        )
    else:
        full_center, full_extent = center, extent
    return {
        "sequence": sequence_dir.name,
        "edges": list(COCO17_EDGES),
        "jointNames": list(COCO17_NAMES),
        "colors": list(PERSON_COLORS),
        "bounds": {
            "near": {
                "center": center.tolist(),
                "extent": extent,
                "trackingExtent": tracking_extent,
            },
            "full": {
                "center": full_center.tolist(),
                "extent": full_extent,
            },
        },
        "cloud": {
            "available": bool(cloud_rows),
            "nearBodyDistanceCm": near_body_distance_cm,
            "pointLimit": cloud_point_limit,
            "surfaceReferenceOnly": True,
        },
        "groundGrid": {
            "floorYcm": GROUND_Y_CM,
            "minorCm": GRID_MINOR_CM,
            "majorCm": GRID_MAJOR_CM,
            "halfSpanCm": GRID_HALF_SPAN_CM,
        },
        "cameras": [
            {
                "name": name,
                "center": cameras[name].center_world_cm.tolist(),
                "selectedV3": name in set(v3[next(iter(v3))].get("cameras", [])) if v3 else False,
            }
            for name in sorted(cameras)
        ],
        "frames": frames,
    }


def _evenly_spaced_indices(indices: np.ndarray, count: int) -> np.ndarray:
    if count <= 0 or not len(indices):
        return np.empty(0, dtype=int)
    if len(indices) <= count:
        return indices
    return indices[
        np.linspace(0, len(indices) - 1, num=count, dtype=int)
    ]


def _select_cloud_points(
    near_mask: np.ndarray,
    maximum_points: int,
) -> np.ndarray:
    """Keep a near-body majority while retaining context for full-scene mode."""

    near_indices = np.flatnonzero(near_mask)
    far_indices = np.flatnonzero(~near_mask)
    if len(near_mask) <= maximum_points:
        return np.arange(len(near_mask), dtype=int)
    near_budget = min(len(near_indices), int(round(maximum_points * 0.8)))
    far_budget = min(len(far_indices), maximum_points - near_budget)
    remaining = maximum_points - near_budget - far_budget
    near_budget = min(len(near_indices), near_budget + remaining)
    remaining = maximum_points - near_budget - far_budget
    far_budget = min(len(far_indices), far_budget + remaining)
    chosen = np.concatenate(
        (
            _evenly_spaced_indices(near_indices, near_budget),
            _evenly_spaced_indices(far_indices, far_budget),
        )
    )
    return np.sort(chosen)


def _pack_cloud_points(
    points_cm: np.ndarray,
    colors_rgb: np.ndarray,
    near_mask: np.ndarray,
) -> str:
    """Pack one cloud into 10 bytes/point for a compact self-contained HTML."""

    dtype = np.dtype(
        [
            ("x", "<i2"),
            ("y", "<i2"),
            ("z", "<i2"),
            ("r", "u1"),
            ("g", "u1"),
            ("b", "u1"),
            ("near", "u1"),
        ]
    )
    packed = np.empty(len(points_cm), dtype=dtype)
    quantized = np.clip(
        np.rint(np.asarray(points_cm) * 10.0),
        np.iinfo(np.int16).min,
        np.iinfo(np.int16).max,
    ).astype(np.int16)
    packed["x"], packed["y"], packed["z"] = quantized.T
    rgb = np.asarray(colors_rgb, dtype=np.uint8)
    packed["r"], packed["g"], packed["b"] = rgb.T
    packed["near"] = np.asarray(near_mask, dtype=np.uint8)
    return base64.b64encode(packed.tobytes()).decode("ascii")


def _unpack_cloud_points(cloud: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dtype = np.dtype(
        [
            ("x", "<i2"),
            ("y", "<i2"),
            ("z", "<i2"),
            ("r", "u1"),
            ("g", "u1"),
            ("b", "u1"),
            ("near", "u1"),
        ]
    )
    packed = np.frombuffer(base64.b64decode(cloud["packed"]), dtype=dtype)
    points = np.column_stack((packed["x"], packed["y"], packed["z"])).astype(
        np.float32
    )
    points *= float(cloud["quantizationCm"])
    colors = np.column_stack((packed["r"], packed["g"], packed["b"]))
    return points, colors.astype(np.uint8), packed["near"].astype(bool)


def write_interactive_viewer(data: Mapping[str, Any], output_path: Path) -> None:
    """Write a self-contained, dependency-free 3D skeleton comparison viewer."""

    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    title = f"{data['sequence']} · RGB-only V10 vs V3 skeleton comparison"
    document = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--ink:#18212f;--muted:#667085;--grid:#d9dee7;--blue:#2563eb;--orange:#e58b17;--paper:#f7f8fb}
*{box-sizing:border-box} body{margin:0;background:var(--paper);color:var(--ink);font:14px Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}
header{position:relative;background:white;border-bottom:1px solid #e5e7eb;padding:20px 24px 17px}
h1{font-size:20px;margin:0 0 6px}.sub{color:var(--muted);max-width:1100px;line-height:1.45}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:15px}
button,select{border:1px solid #cfd5df;background:white;border-radius:7px;padding:7px 11px;color:var(--ink)}
button{cursor:pointer} input[type=range]{width:min(520px,62vw)} .compact{width:110px!important}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
main{padding:16px;max-width:1800px;margin:auto}.grid{display:grid;grid-template-columns:repeat(3,minmax(280px,1fr));gap:12px}
.panel{background:white;border:1px solid #e2e6ed;border-radius:10px;overflow:hidden}.panel h2{font-size:14px;margin:0;padding:11px 13px;border-bottom:1px solid #edf0f4}
canvas{display:block;width:100%;height:64vh;min-height:440px;cursor:grab}canvas:active{cursor:grabbing}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin:12px 2px;color:var(--muted)}
.chip:before{content:"";display:inline-block;width:14px;height:3px;margin:0 6px 3px 0;background:var(--blue)}
.chip.pred:before{background:var(--orange)}.note{margin-left:auto}.status{min-width:330px}
@media(max-width:1000px){.grid{grid-template-columns:1fr}canvas{height:52vh}.note{margin-left:0}}
</style>
</head>
<body>
<header>
<h1>__TITLE__</h1>
<div class="sub">ViTPose and triangulation consume Kinect RGB frames only. The colored RGB-D point cloud is shown afterward as a synchronized body-surface reference; it is neither an inference input nor joint-center ground truth. All three panels share the same world-coordinate cloud, bounds, view, and HD time.</div>
<div class="controls">
<button id="play">Play</button><input id="frame" type="range" min="0" value="0" step="1">
<span class="mono status" id="status"></span>
<label>Speed <select id="speed"><option value="200">0.5×</option><option value="100" selected>1×</option><option value="50">2×</option></select></label>
<label>Surface <select id="cloudmode"><option value="near" selected>Near body</option><option value="full">Full scene</option><option value="off">Off</option></select></label>
<label>Opacity <input class="compact" id="cloudopacity" type="range" min="0.15" max="1" value="0.72" step="0.05"></label>
<label>Point size <input class="compact" id="cloudsize" type="range" min="1" max="3" value="1.4" step="0.2"></label>
<button id="reset">Reset view</button>
</div>
</header>
<main>
<div class="grid">
<section class="panel"><h2>GT skeleton + RGB-D surface reference</h2><canvas data-kind="gt"></canvas></section>
<section class="panel"><h2>V10 · ten RGB monocular views</h2><canvas data-kind="v10"></canvas></section>
<section class="panel"><h2>V3 · balanced cameras 50_06 + 50_04 + 50_02</h2><canvas data-kind="v3"></canvas></section>
</div>
<div class="legend"><span class="chip">Skeleton: person identity color</span><span class="chip pred">Camera center: orange = V3, grey = other</span><span>Ground grid: 50 cm minor · 1 m major</span><span>Point color comes from synchronized Kinect RGB.</span><span class="note">Units: Panoptic world centimeters</span></div>
</main>
<script>
const DATA=__PAYLOAD__;
const slider=document.getElementById('frame'),status=document.getElementById('status'),play=document.getElementById('play');
const cloudMode=document.getElementById('cloudmode'),cloudOpacity=document.getElementById('cloudopacity'),cloudSize=document.getElementById('cloudsize');
slider.max=Math.max(0,DATA.frames.length-1);
let view={yaw:-0.68,pitch:-0.28,zoom:1.3},timer=null,drag=null;
const canvases=[...document.querySelectorAll('canvas')];
function resize(c){const r=c.getBoundingClientRect(),d=devicePixelRatio||1;c.width=Math.round(r.width*d);c.height=Math.round(r.height*d);c.getContext('2d').setTransform(d,0,0,d,0,0)}
function rotate(p){const cy=Math.cos(view.yaw),sy=Math.sin(view.yaw),cp=Math.cos(view.pitch),sp=Math.sin(view.pitch);let x=cy*p[0]+sy*p[2],z=-sy*p[0]+cy*p[2],y=cp*p[1]-sp*z;z=sp*p[1]+cp*z;return[x,y,z]}
function bounds(frame){const selected=cloudMode.value==='full'?DATA.bounds.full:frame.bounds;return{center:selected.center,scale:selected.extent}}
function projection(p,b,w,h){const q=rotate([p[0]-b.center[0],p[1]-b.center[1],p[2]-b.center[2]]),s=Math.min(w,h)*0.74/b.scale*view.zoom;return[w/2+q[0]*s,h/2+q[1]*s,q[2]]}
function line(ctx,a,b,color,width=2,alpha=1){ctx.globalAlpha=alpha;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();ctx.globalAlpha=1}
function drawGrid(ctx,b,w,h){const g=DATA.groundGrid,half=g.halfSpanCm,snap=(v,step,up)=>(up?Math.ceil(v/step):Math.floor(v/step))*step,x0=snap(b.center[0]-half,g.minorCm,false),x1=snap(b.center[0]+half,g.minorCm,true),z0=snap(b.center[2]-half,g.minorCm,false),z1=snap(b.center[2]+half,g.minorCm,true),draw=(a,z,tick)=>{const axis=tick===0,major=tick%g.majorCm===0;line(ctx,projection(a,b,w,h),projection(z,b,w,h),axis?'#8b95a5':major?'#c3cad5':'#e2e6ed',axis?1.5:major?1.05:.65)};for(let x=x0;x<=x1;x+=g.minorCm)draw([x,g.floorYcm,z0],[x,g.floorYcm,z1],x);for(let z=z0;z<=z1;z+=g.minorCm)draw([x0,g.floorYcm,z],[x1,g.floorYcm,z],z)}
function decodeCloud(cloud){if(!cloud)return null;if(cloud.decoded)return cloud.decoded;const raw=atob(cloud.packed),bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);const viewBytes=new DataView(bytes.buffer),points=[];for(let i=0;i<cloud.count;i++){const o=i*cloud.strideBytes;points.push([viewBytes.getInt16(o,true)*cloud.quantizationCm,viewBytes.getInt16(o+2,true)*cloud.quantizationCm,viewBytes.getInt16(o+4,true)*cloud.quantizationCm,bytes[o+6],bytes[o+7],bytes[o+8],bytes[o+9]===1])}cloud.decoded=points;return points}
function drawCloud(ctx,cloud,b,w,h){if(!cloud||cloudMode.value==='off')return;const full=cloudMode.value==='full',size=+cloudSize.value,projected=[];for(const p of decodeCloud(cloud)){if(!full&&!p[6])continue;const q=projection(p,b,w,h);if(q[0]>=0&&q[0]<w&&q[1]>=0&&q[1]<h)projected.push([q[0],q[1],q[2],p[3],p[4],p[5]])}projected.sort((a,z)=>a[2]-z[2]);ctx.globalAlpha=+cloudOpacity.value;for(const p of projected){ctx.fillStyle=`rgb(${p[3]},${p[4]},${p[5]})`;ctx.fillRect(p[0]-size/2,p[1]-size/2,size,size)}ctx.globalAlpha=1}
function fmt(v,d=1){return v==null?'n/a':Number(v).toFixed(d)}
function draw(){if(!DATA.frames.length)return;const i=+slider.value,f=DATA.frames[i],b=bounds(f),span=f.cloud?f.cloud.temporalSpanMs:null,cloudText=f.cloud?` · RGB-D ${f.cloud.nearCount}/${f.cloud.count} near · ${f.cloud.acceptedNodes} nodes · span ${fmt(span)} ms${span>30?' ⚠':''}`:'';status.textContent=`frame ${i+1}/${DATA.frames.length} · HD ${f.hd} · V10 ${fmt(f.metrics.v10MpjpeMm)} mm · V3 ${fmt(f.metrics.v3MpjpeMm)} mm${cloudText}`;for(const c of canvases){resize(c);const ctx=c.getContext('2d'),w=c.clientWidth,h=c.clientHeight;ctx.fillStyle='#fff';ctx.fillRect(0,0,w,h);drawGrid(ctx,b,w,h);drawCloud(ctx,f.cloud,b,w,h);for(const cam of DATA.cameras){const p=projection(cam.center,b,w,h);ctx.fillStyle=cam.selectedV3?'#e58b17':'#a8b0bd';ctx.beginPath();ctx.arc(p[0],p[1],cam.selectedV3?4:3,0,Math.PI*2);ctx.fill()}const people=f[c.dataset.kind];for(const person of people){const color=DATA.colors[Math.abs(person.colorId)%DATA.colors.length],pts=person.joints.map(p=>p?projection(p,b,w,h):null);for(const e of DATA.edges)if(pts[e[0]]&&pts[e[1]])line(ctx,pts[e[0]],pts[e[1]],color,3);for(const p of pts)if(p){ctx.fillStyle=color;ctx.beginPath();ctx.arc(p[0],p[1],3.3,0,Math.PI*2);ctx.fill()}const root=pts[11]||pts[12]||pts[0];if(root){const used=person.support?Math.max(...person.support):null;ctx.fillStyle='#18212f';ctx.font='12px ui-monospace';ctx.fillText(`ID ${person.id}${used?` · ≤${used} views`:''}`,root[0]+6,root[1]-6)}}ctx.fillStyle='#667085';ctx.font='12px ui-monospace';ctx.fillText('ground grid · 0.5 m minor / 1 m major',13,h-32);const metric=c.dataset.kind==='v10'?f.metrics.v10MpjpeMm:c.dataset.kind==='v3'?f.metrics.v3MpjpeMm:null;ctx.fillText(metric==null?`HD ${f.hd}`:`MPJPE ${fmt(metric)} mm · HD ${f.hd}`,13,h-14)}}
function stop(){clearInterval(timer);timer=null;play.textContent='Play'}
slider.oninput=draw;play.onclick=()=>{if(timer){stop();return}play.textContent='Pause';timer=setInterval(()=>{if(+slider.value>=+slider.max){stop();return}slider.value=+slider.value+1;draw()},+document.getElementById('speed').value)};
document.getElementById('speed').onchange=()=>{if(timer){stop();play.click()}};
cloudMode.onchange=draw;cloudOpacity.oninput=draw;cloudSize.oninput=draw;
document.getElementById('reset').onclick=()=>{view={yaw:-0.68,pitch:-0.28,zoom:1.3};draw()};
for(const c of canvases){c.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY};c.setPointerCapture(e.pointerId)};c.onpointermove=e=>{if(!drag)return;view.yaw+=(e.clientX-drag.x)*.008;view.pitch=Math.max(-1.45,Math.min(1.45,view.pitch+(e.clientY-drag.y)*.008));drag={x:e.clientX,y:e.clientY};draw()};c.onpointerup=()=>drag=null;c.onwheel=e=>{e.preventDefault();view.zoom=Math.max(.35,Math.min(4,view.zoom*Math.exp(-e.deltaY*.001)));draw()}}
window.onresize=draw;draw();
</script>
</body></html>"""
    document = (
        document.replace("__TITLE__", html.escape(title))
        .replace("__TITLE__", html.escape(title))
        .replace("__PAYLOAD__", payload)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def render_summary_figure(
    evaluation_v10_dir: Path,
    evaluation_v3_dir: Path,
    output_path: Path,
) -> None:
    summary10 = json.loads((evaluation_v10_dir / "summary.json").read_text())
    summary3 = json.loads((evaluation_v3_dir / "summary.json").read_text())
    frames10 = {int(x["hd_index"]): x for x in read_jsonl(evaluation_v10_dir / "per_frame.jsonl")}
    frames3 = {int(x["hd_index"]): x for x in read_jsonl(evaluation_v3_dir / "per_frame.jsonl")}
    common = sorted(set(frames10) & set(frames3))
    times = np.asarray([frames10[i]["univ_time_ms"] for i in common])
    times = (times - times[0]) / 1000.0 if len(times) else times
    series10 = np.asarray(
        [np.nan if frames10[i]["mpjpe_mm"] is None else frames10[i]["mpjpe_mm"] for i in common]
    )
    series3 = np.asarray(
        [np.nan if frames3[i]["mpjpe_mm"] is None else frames3[i]["mpjpe_mm"] for i in common]
    )
    joints10 = [x["mpjpe_mm"] for x in summary10["joints"]]
    joints3 = [x["mpjpe_mm"] for x in summary3["joints"]]
    blue, orange, ink, grid = "#2563eb", "#e58b17", "#18212f", "#d9dee7"
    fig = plt.figure(figsize=(15, 10), facecolor="#f7f8fb")
    layout = fig.add_gridspec(2, 2, height_ratios=(0.8, 1.2), width_ratios=(0.8, 1.2))
    ax0 = fig.add_subplot(layout[0, 0])
    ax1 = fig.add_subplot(layout[0, 1])
    ax2 = fig.add_subplot(layout[1, :])
    labels = ["MPJPE", "PCK@50", "Availability"]
    values10 = [summary10["mpjpe_mm"], summary10["pck_50"] * 100, summary10["joint_availability"] * 100]
    values3 = [summary3["mpjpe_mm"], summary3["pck_50"] * 100, summary3["joint_availability"] * 100]
    units = ["mm", "%", "%"]
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 3)
    ax0.axis("off")
    ax0.set_title("Overall reconstruction metrics", loc="left", color=ink, weight="bold")
    ax0.text(0.56, 2.82, "V10", color=blue, weight="bold", ha="right")
    ax0.text(0.82, 2.82, "V3", color=orange, weight="bold", ha="right")
    for row, (label, value10, value3, unit) in enumerate(zip(labels, values10, values3, units)):
        y = 2.25 - row * 0.85
        ax0.text(0.0, y, label, color=ink, weight="bold")
        ax0.text(0.56, y, f"{value10:.1f} {unit}", color=blue, ha="right", family="monospace")
        ax0.text(0.82, y, f"{value3:.1f} {unit}", color=orange, ha="right", family="monospace")
        delta = value3 - value10
        ax0.text(0.98, y, f"Δ {delta:+.1f}", color="#667085", ha="right", family="monospace")
        ax0.hlines(y - 0.3, 0, 1, color=grid, lw=0.8)
    joint_x = np.arange(17)
    ax1.plot(joint_x, joints10, color=blue, marker="o", ms=3, label="V10")
    ax1.plot(joint_x, joints3, color=orange, marker="s", ms=3, linestyle="--", label="V3")
    ax1.set_xticks(joint_x, [name.replace("left_", "L ").replace("right_", "R ") for name in COCO17_NAMES], rotation=60, ha="right")
    ax1.set_ylabel("MPJPE (mm)")
    ax1.set_title("Joint-level error", loc="left", color=ink, weight="bold")
    ax1.legend(frameon=False)
    ax2.plot(times, series10, color=blue, lw=1.8, label="V10")
    ax2.plot(times, series3, color=orange, lw=1.5, linestyle="--", label="V3")
    ax2.set_xlabel("Elapsed synchronized time (s)")
    ax2.set_ylabel("Frame MPJPE (mm)")
    ax2.set_title("Frame-level reconstruction error", loc="left", color=ink, weight="bold")
    ax2.legend(frameon=False, ncol=2)
    for axis in (ax0, ax1, ax2):
        axis.set_facecolor("white")
        axis.grid(axis="y", color=grid, lw=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "RGB-only V10 versus V3 skeleton reconstruction",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=18,
        weight="bold",
        color=ink,
    )
    fig.text(
        0.055,
        0.948,
        "CMU Panoptic pilot · same synchronized frames · official 3D skeleton reference · no Kinect depth in inference",
        color="#667085",
    )
    fig.tight_layout(rect=(0.03, 0.03, 0.98, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_m1_m2_diagnostics(
    m1_summary_path: Path,
    m2_v10_path: Path,
    m2_v3_path: Path,
    output_path: Path,
) -> None:
    """Render stage-level diagnostics without reusing GT in inference."""

    with m1_summary_path.open("r", encoding="utf-8") as handle:
        m1 = json.load(handle)
    with m2_v10_path.open("r", encoding="utf-8") as handle:
        m2_v10 = json.load(handle)
    with m2_v3_path.open("r", encoding="utf-8") as handle:
        m2_v3 = json.load(handle)

    cameras = sorted(m1["per_camera"])
    errors = [m1["per_camera"][camera]["mean_joint_error_px"] for camera in cameras]
    recalls = [m1["per_camera"][camera]["person_recall"] * 100 for camera in cameras]
    stage_metrics = (
        ("Precision", "pairwise_precision"),
        ("Recall", "pairwise_recall"),
        ("F1", "pairwise_f1"),
        ("Purity", "cluster_purity"),
        ("Completeness", "cluster_completeness"),
    )
    failure_metrics = (
        ("Wrong merge", "wrong_person_merge_rate"),
        ("Split person", "split_person_rate"),
        ("Unclustered", "unclustered_matched_detection_rate"),
    )
    ink, blue, orange, grid = "#18212f", "#2563eb", "#e58b17", "#d9dee7"
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="#f7f8fb")
    x = np.arange(len(cameras))
    axes[0, 0].bar(x, errors, color=blue, width=0.68)
    axes[0, 0].axhline(
        m1["mean_joint_error_px"],
        color=orange,
        linestyle="--",
        label=f"overall {m1['mean_joint_error_px']:.1f} px",
    )
    axes[0, 0].set_xticks(x, cameras, rotation=45, ha="right")
    axes[0, 0].set_ylabel("Mean joint error (px)")
    axes[0, 0].set_title("M1 localization by camera", loc="left", weight="bold")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].bar(x, recalls, color="#657a2d", width=0.68)
    axes[0, 1].set_xticks(x, cameras, rotation=45, ha="right")
    axes[0, 1].set_ylim(max(0, min(recalls) - 5), 101)
    axes[0, 1].set_ylabel("Person recall (%)")
    axes[0, 1].set_title("M1 detection recall by camera", loc="left", weight="bold")

    metric_x = np.arange(len(stage_metrics))
    width = 0.34
    axes[1, 0].bar(
        metric_x - width / 2,
        [m2_v10[key] * 100 for _, key in stage_metrics],
        width,
        color=blue,
        label="V10",
    )
    axes[1, 0].bar(
        metric_x + width / 2,
        [m2_v3[key] * 100 for _, key in stage_metrics],
        width,
        color=orange,
        label="V3",
    )
    axes[1, 0].set_xticks(metric_x, [label for label, _ in stage_metrics])
    axes[1, 0].set_ylim(0, 105)
    axes[1, 0].set_ylabel("Score (%)")
    axes[1, 0].set_title("M2 association quality", loc="left", weight="bold")
    axes[1, 0].legend(frameon=False)

    failure_x = np.arange(len(failure_metrics))
    axes[1, 1].bar(
        failure_x - width / 2,
        [m2_v10[key] * 100 for _, key in failure_metrics],
        width,
        color=blue,
        label="V10",
    )
    axes[1, 1].bar(
        failure_x + width / 2,
        [m2_v3[key] * 100 for _, key in failure_metrics],
        width,
        color=orange,
        label="V3",
    )
    axes[1, 1].set_xticks(
        failure_x,
        [label for label, _ in failure_metrics],
        rotation=12,
        ha="right",
    )
    axes[1, 1].set_ylabel("Rate (%)")
    axes[1, 1].set_title("M2 failure modes", loc="left", weight="bold")
    axes[1, 1].legend(frameon=False)

    for axis in axes.flat:
        axis.set_facecolor("white")
        axis.grid(axis="y", color=grid, lw=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "M1 and M2 diagnostics for the frozen RGB predictions",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=18,
        weight="bold",
        color=ink,
    )
    fig.text(
        0.055,
        0.95,
        "Official 3D joints are projected only during evaluation · GT identity is not consumed by association or triangulation",
        color="#667085",
    )
    fig.tight_layout(rect=(0.03, 0.03, 0.98, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_triplet_geometry_figure(csv_path: Path, output_path: Path) -> Dict[str, Any]:
    with csv_path.open("r", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("mpjpe_mm")]
    if not rows:
        raise ValueError(f"No triplet metrics in {csv_path}")
    rows.sort(key=lambda item: float(item["mpjpe_mm"]))
    baseline = np.asarray([float(item["max_baseline_cm"]) for item in rows])
    error = np.asarray([float(item["mpjpe_mm"]) for item in rows])
    availability = np.asarray([float(item["joint_availability"]) for item in rows])
    balanced_index = next(
        index for index, item in enumerate(rows) if item["contains_balanced_primary"] == "True"
    )
    best_index, median_index, worst_index = 0, len(rows) // 2, len(rows) - 1
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), facecolor="#f7f8fb")
    ink, blue, orange, grid = "#18212f", "#2563eb", "#e58b17", "#d9dee7"
    scatter = axes[0].scatter(
        baseline,
        error,
        c=availability,
        cmap="Blues",
        s=42,
        edgecolors="#ffffff",
        linewidths=0.5,
    )
    colorbar = fig.colorbar(scatter, ax=axes[0], fraction=0.046, pad=0.04)
    colorbar.set_label("Joint availability")
    axes[0].scatter(
        baseline[balanced_index],
        error[balanced_index],
        marker="*",
        s=260,
        color=orange,
        edgecolor=ink,
        linewidth=0.8,
        zorder=5,
        label="Pre-registered balanced V3",
    )
    axes[0].set_xlabel("Maximum camera baseline (cm)")
    axes[0].set_ylabel("MPJPE (mm)")
    axes[0].set_title("Triplet geometry and reconstruction error", loc="left", weight="bold")
    axes[0].legend(frameon=False, loc="upper left")
    rank_x = np.arange(1, len(rows) + 1)
    axes[1].plot(rank_x, error, color=blue, lw=2)
    axes[1].scatter(
        balanced_index + 1,
        error[balanced_index],
        marker="*",
        s=260,
        color=orange,
        edgecolor=ink,
        linewidth=0.8,
        zorder=5,
    )
    for index, label in ((best_index, "best"), (median_index, "median"), (worst_index, "worst")):
        axes[1].annotate(
            f"{label}: {rows[index]['cameras']}",
            (index + 1, error[index]),
            xytext=(5, 8 if index != worst_index else -18),
            textcoords="offset points",
            fontsize=8,
            color="#667085",
        )
    axes[1].annotate(
        f"balanced rank {balanced_index + 1}/120",
        (balanced_index + 1, error[balanced_index]),
        xytext=(9, -20),
        textcoords="offset points",
        fontsize=9,
        color=orange,
    )
    axes[1].set_xlabel("Triplet rank by MPJPE")
    axes[1].set_ylabel("MPJPE (mm)")
    axes[1].set_title("All 120 three-camera subsets", loc="left", weight="bold")
    for axis in axes:
        axis.set_facecolor("white")
        axis.grid(color=grid, lw=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Three-camera placement sensitivity",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=17,
        weight="bold",
        color=ink,
    )
    fig.text(
        0.055,
        0.93,
        f"{rows[0]['frames']} synchronized frames per triplet · identical cached 2D joints · color indicates reconstructed-joint availability",
        color="#667085",
    )
    fig.tight_layout(rect=(0.03, 0.04, 0.98, 0.9))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "output": str(output_path),
        "balanced_rank": balanced_index + 1,
        "triplets": len(rows),
        "balanced_mpjpe_mm": float(error[balanced_index]),
        "best_mpjpe_mm": float(error[best_index]),
        "median_mpjpe_mm": float(error[median_index]),
        "worst_mpjpe_mm": float(error[worst_index]),
    }


def render_calibration_audit(
    sequence_dir: Path,
    frame_table_path: Path,
    output_path: Path,
    frame_offset: int = 0,
) -> Dict[str, Any]:
    """Render actual synchronized RGB frames with projected official GT joints."""

    frames = list(read_jsonl(frame_table_path))
    if not frames:
        raise ValueError("Frame table is empty")
    frame = frames[min(max(frame_offset, 0), len(frames) - 1)]
    cameras = load_cameras(sequence_dir)
    gt = load_gt_coco17(Path(frame["gt_path"]))
    tiles: List[np.ndarray] = []
    colors_bgr = [(235, 99, 37), (23, 139, 229), (45, 122, 101), (123, 68, 192)]
    with VideoReaderPool(sequence_dir) as videos:
        for camera_name in sorted(cameras):
            source_index = int(frame["cameras"][camera_name]["source_index"])
            image = videos.read(camera_name, source_index)
            for person_index, person in enumerate(gt):
                points, depth = project_points(cameras[camera_name], person["joints_cm"])
                valid = (np.asarray(person["confidence"]) > 0.1) & (depth > 0)
                color = colors_bgr[person_index % len(colors_bgr)]
                for first, second in COCO17_EDGES:
                    if valid[first] and valid[second]:
                        cv2.line(
                            image,
                            tuple(np.round(points[first]).astype(int)),
                            tuple(np.round(points[second]).astype(int)),
                            color,
                            5,
                            cv2.LINE_AA,
                        )
                for index in np.flatnonzero(valid):
                    cv2.circle(
                        image,
                        tuple(np.round(points[index]).astype(int)),
                        7,
                        color,
                        -1,
                        cv2.LINE_AA,
                    )
            label = (
                f"{camera_name} source={source_index} "
                f"skew={frame['cameras'][camera_name]['delta_ms']:+.2f}ms"
            )
            cv2.rectangle(image, (0, 0), (900, 52), (255, 255, 255), -1)
            cv2.putText(
                image,
                label,
                (14, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (24, 33, 47),
                2,
                cv2.LINE_AA,
            )
            tiles.append(cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA))
    rows = [np.hstack(tiles[index : index + 5]) for index in range(0, 10, 5)]
    mosaic = np.vstack(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), mosaic):
        raise RuntimeError(f"Could not write {output_path}")
    return {
        "output": str(output_path),
        "hd_index": int(frame["hd_index"]),
        "univ_time_ms": float(frame["univ_time_ms"]),
        "cameras": 10,
    }


def render_comparison_video(
    data: Mapping[str, Any],
    output_path: Path,
    fps: float = 10.0,
    width: int = 1920,
    height: int = 720,
) -> None:
    """Render a portable side-by-side MP4 from the same viewer payload."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {output_path}")
    panels = ("gt", "v10", "v3")
    titles = (
        "GT skeleton + RGB-D surface",
        "V10 - RGB-only inference",
        "V3 - RGB-only inference",
    )
    panel_width = width // 3
    center = np.asarray(data["bounds"]["near"]["center"], dtype=float)
    extent = float(data["bounds"]["near"]["trackingExtent"])
    yaw, pitch = -0.68, -0.28
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)

    def project(point: Sequence[float], x_offset: int) -> tuple:
        p = np.asarray(point) - center
        x, z = cy * p[0] + sy * p[2], -sy * p[0] + cy * p[2]
        y = cp * p[1] - sp * z
        scale = min(panel_width, height) * 0.68 / extent
        return int(x_offset + panel_width / 2 + x * scale), int(height / 2 + y * scale)

    try:
        for frame in data["frames"]:
            center = np.asarray(frame["bounds"]["center"], dtype=float)
            extent = float(frame["bounds"]["extent"])
            image = np.full((height, width, 3), 250, dtype=np.uint8)
            if frame.get("cloud"):
                cloud_points, cloud_rgb, near = _unpack_cloud_points(frame["cloud"])
                cloud_points = cloud_points[near]
                cloud_rgb = cloud_rgb[near]
            else:
                cloud_points = np.empty((0, 3), dtype=float)
                cloud_rgb = np.empty((0, 3), dtype=np.uint8)
            for panel_index, (key, title) in enumerate(zip(panels, titles)):
                x_offset = panel_index * panel_width
                for start, end, major, axis in _ground_grid_segments(
                    center,
                ):
                    cv2.line(
                        image,
                        project(start, x_offset),
                        project(end, x_offset),
                        (
                            (149, 143, 139)
                            if axis
                            else (213, 205, 195)
                            if major
                            else (237, 230, 226)
                        ),
                        2 if axis else 1,
                        cv2.LINE_AA,
                    )
                if len(cloud_points):
                    cloud_pixels = np.asarray(
                        [project(point, x_offset) for point in cloud_points],
                        dtype=int,
                    )
                    inside = (
                        (cloud_pixels[:, 0] >= x_offset)
                        & (cloud_pixels[:, 0] < x_offset + panel_width)
                        & (cloud_pixels[:, 1] >= 0)
                        & (cloud_pixels[:, 1] < height)
                    )
                    visible_pixels = cloud_pixels[inside]
                    visible_bgr = cloud_rgb[inside, ::-1]
                    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                        x = np.clip(
                            visible_pixels[:, 0] + dx,
                            x_offset,
                            x_offset + panel_width - 1,
                        )
                        y = np.clip(visible_pixels[:, 1] + dy, 0, height - 1)
                        image[y, x] = visible_bgr
                cv2.putText(
                    image,
                    title,
                    (x_offset + 18, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (47, 33, 24),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    "ground grid: 0.5 m minor / 1 m major",
                    (x_offset + 18, 54),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (133, 112, 102),
                    1,
                    cv2.LINE_AA,
                )
                if panel_index:
                    cv2.line(image, (x_offset, 0), (x_offset, height), (220, 224, 230), 1)
                for person in frame[key]:
                    color_hex = data["colors"][abs(int(person["colorId"])) % len(data["colors"])]
                    color = tuple(int(color_hex[i : i + 2], 16) for i in (5, 3, 1))
                    projected = [
                        project(point, x_offset) if point is not None else None
                        for point in person["joints"]
                    ]
                    for first, second in data["edges"]:
                        if projected[first] is not None and projected[second] is not None:
                            cv2.line(image, projected[first], projected[second], color, 3, cv2.LINE_AA)
                    for point in projected:
                        if point is not None:
                            cv2.circle(image, point, 4, color, -1, cv2.LINE_AA)
            cv2.putText(
                image,
                (
                    f"HD {frame['hd']}  t={frame['time']:.3f} ms  "
                    "RGB-D surface is evaluation-only"
                    + (
                        f"  nodes={frame['cloud']['acceptedNodes']}"
                        f"  span={frame['cloud']['temporalSpanMs']:.1f} ms"
                        + (
                            "  WARNING span>30ms"
                            if frame["cloud"]["temporalSpanMs"] > 30.0
                            else ""
                        )
                        if frame.get("cloud")
                        and frame["cloud"].get("temporalSpanMs") is not None
                        else ""
                    )
                ),
                (18, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (102, 112, 133),
                1,
                cv2.LINE_AA,
            )
            writer.write(image)
    finally:
        writer.release()
