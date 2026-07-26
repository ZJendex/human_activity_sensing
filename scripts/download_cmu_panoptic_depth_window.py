#!/usr/bin/env python3
"""Download only the raw depth prefixes required by a frozen frame table."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


DEPTH_FRAME_BYTES = 512 * 424 * 2
DEFAULT_BASE_URL = "http://domedb.perception.cs.cmu.edu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create depthdata.window.dat prefixes containing every Kinect depth "
            "frame required by an existing RGB study frame table."
        )
    )
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--frame-table", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=20)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--filename", default="depthdata.window.dat")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def required_prefixes(sequence_dir: Path, frame_table: Path) -> dict[int, dict]:
    sequence = sequence_dir.name
    panoptic_sync = load_json(sequence_dir / f"synctables_{sequence}.json")
    ksync = load_json(
        sequence_dir / f"ksynctables_{sequence}.json"
    )["kinect"]["depth"]
    hd_times = np.asarray(panoptic_sync["hd"]["univ_time"], dtype=float)
    frames = [
        json.loads(line)
        for line in frame_table.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    targets = [float(hd_times[int(frame["hd_index"])]) for frame in frames]
    result = {}
    for node in range(1, 11):
        key = f"KINECTNODE{node}"
        depth_times = np.asarray(ksync[key]["univ_time"], dtype=float)
        valid = np.flatnonzero(depth_times > 0)
        if valid.size == 0:
            raise ValueError(f"{key} has no valid synchronized depth times")
        positions = [
            int(valid[np.argmin(np.abs(depth_times[valid] - target))])
            for target in targets
        ]
        maximum = max(positions)
        result[node] = {
            "minimum_sync_position": min(positions),
            "maximum_sync_position": maximum,
            "required_bytes": (maximum + 1) * DEPTH_FRAME_BYTES,
        }
    return result


def copy_prefix(source: Path, destination: Path, required_bytes: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with source.open("rb") as input_stream, partial.open("wb") as output_stream:
        remaining = required_bytes
        while remaining:
            chunk = input_stream.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise EOFError(
                    f"{source} ended with {remaining} required bytes remaining"
                )
            output_stream.write(chunk)
            remaining -= len(chunk)
    os.replace(partial, destination)


def download_prefix(url: str, destination: Path, required_bytes: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    current = partial.stat().st_size if partial.exists() else 0
    if current > required_bytes:
        partial.unlink()
        current = 0
    if current == required_bytes:
        os.replace(partial, destination)
        return
    request = urllib.request.Request(
        url,
        headers={"Range": f"bytes={current}-{required_bytes - 1}"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        status = getattr(response, "status", None)
        if current and status != 206:
            raise RuntimeError(
                f"Server did not honor resume Range for {url} (HTTP {status})"
            )
        mode = "ab" if current else "wb"
        remaining = required_bytes - current
        with partial.open(mode) as output_stream:
            while remaining:
                chunk = response.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise EOFError(
                        f"{url} ended with {remaining} required bytes remaining"
                    )
                output_stream.write(chunk)
                remaining -= len(chunk)
    if partial.stat().st_size != required_bytes:
        raise RuntimeError(f"Unexpected prefix size for {partial}")
    os.replace(partial, destination)


def prepare_node(
    *,
    sequence_dir: Path,
    base_url: str,
    filename: str,
    node: int,
    required_bytes: int,
    attempts: int,
) -> dict:
    node_dir = sequence_dir / "kinect_shared_depth" / f"KINECTNODE{node}"
    destination = node_dir / filename
    if destination.exists() and destination.stat().st_size == required_bytes:
        source = "existing-window"
    else:
        full = node_dir / "depthdata.dat"
        resumable_full = node_dir / "depthdata.dat.part"
        local_source = next(
            (
                path
                for path in (full, resumable_full)
                if path.exists() and path.stat().st_size >= required_bytes
            ),
            None,
        )
        if local_source is not None:
            copy_prefix(local_source, destination, required_bytes)
            source = str(local_source)
        else:
            sequence = sequence_dir.name
            url = (
                f"{base_url.rstrip('/')}/webdata/dataset/{sequence}/"
                f"kinect_shared_depth/KINECTNODE{node}/depthdata.dat"
            )
            for attempt in range(1, attempts + 1):
                try:
                    download_prefix(url, destination, required_bytes)
                    break
                except (EOFError, OSError, RuntimeError) as error:
                    if attempt == attempts:
                        raise
                    print(
                        f"[retry {attempt}/{attempts}] node {node:02d}: {error}",
                        flush=True,
                    )
                    time.sleep(2)
            source = url
    return {
        "node": node,
        "path": str(destination),
        "bytes": required_bytes,
        "source": source,
    }


def main() -> None:
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    if args.attempts <= 0:
        raise ValueError("--attempts must be positive")
    prefixes = required_prefixes(args.sequence_dir, args.frame_table)
    completed = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                prepare_node,
                sequence_dir=args.sequence_dir,
                base_url=args.base_url,
                filename=args.filename,
                node=node,
                required_bytes=details["required_bytes"],
                attempts=args.attempts,
            ): node
            for node, details in prefixes.items()
        }
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            print(
                f"[done] node {result['node']:02d}: "
                f"{result['bytes'] / 1024**2:.1f} MiB",
                flush=True,
            )
    manifest = {
        "sequence": args.sequence_dir.name,
        "frame_table": str(args.frame_table),
        "depth_filename": args.filename,
        "depth_frame_bytes": DEPTH_FRAME_BYTES,
        "nodes": [
            {**prefixes[item["node"]], **item}
            for item in sorted(completed, key=lambda value: value["node"])
        ],
        "scope": (
            "Frame-table-bounded raw prefixes for evaluation-only RGB-D fusion; "
            "not complete sequence depth streams."
        ),
    }
    manifest_path = args.sequence_dir / "kinect_shared_depth" / (
        args.filename + ".manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    total = sum(item["bytes"] for item in completed)
    print(f"Complete: {total / 1024**3:.2f} GiB -> {manifest_path}")


if __name__ == "__main__":
    main()
