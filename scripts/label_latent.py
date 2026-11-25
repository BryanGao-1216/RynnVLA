#!/usr/bin/env python3
"""Label local videos with RynnLAM latent actions -> per-(episode, view) latent.npz.

Minimal single-node CLI. It decodes a local video with `rynnlam.video.FrameReader`, encodes
each ``(t, t+gap)`` frame pair (advancing by ``pair_stride``) with a frozen `RynnLAMEncoder`,
and writes the npz that the Stage-1 index builders consume:

    video --(this script)--> latent.npz
        -> scripts/rebuild_latent_manifest.py     (manifest jsonl)
        -> scripts/build_latent_pretrain_index.py (Stage-1 training index npz)
        -> scripts/build_latent_stats.py          (normalization stats json)
        -> rynnvla.api.train                       (Stage-1 latent-action pretraining)

The byte layout and ``meta`` match the "Latent-action protocol" section of README.md
(``schema_version 2``, ``gap 4``, ``pair_stride 4``, ``latent_action`` float16 ``[N, code]``,
``pair_indices`` int32 ``[N, 2]``, ``meta`` a JSON string). The default representation is
``ktoken_zcam`` (608 dims for the b512 recipe: 8*64 k-tokens + 64 z + 32 camera_pose), which
is what Stage-1 consumes.

This needs a trained RynnLAM motion checkpoint (``--checkpoint``); producing one is
`scripts/train_rynnlam.py`. Run from the repo root: relative ``video_path`` values resolve
against the current working directory, as they do everywhere else in this repo. Examples:

    # one video -> <out>/latents/videos/<stem>/<view>/latent.npz
    python scripts/label_latent.py --checkpoint rynnlam_motion.pt \
        --video demo.mp4 --output-dir out

    # a RynnVLA-Base episode JSON (list of {dataset, episode_id, views:{view:{video_path}}})
    # -> <out>/latents/<dataset>/<episode_id>/<view>/latent.npz
    python scripts/label_latent.py --checkpoint rynnlam_motion.pt \
        --metadata data/sample/source/SampleToy.json --output-dir out
"""
import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, nullcontext
from dataclasses import dataclass
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import zipfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rynnlam.inference import RynnLAMEncoder, file_sha256, implementation_sha256
from rynnlam.prefetch import PrefetchedBatches
from rynnlam.video import BUCKETS, FrameReader, VideoSeekError, pair_batches, resolve_source


def safe_name(value, *, nested=False):
    """Reject path-traversal in dataset/episode/view identifiers used to build output paths."""
    value = str(value)
    path = PurePosixPath(value)
    if (
        not value
        or value in {".", ".."}
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
    ):
        raise ValueError(f"Invalid output identifier: {value!r}")
    if not nested and "/" in value:
        raise ValueError(f"Expected one path component: {value!r}")
    return value


def label_video(encoder, video, protocol, *, view, batch_size, prepared=None, label_name=None):
    """Encode every (t, t+gap) pair of one video; return (latent, pair_indices, meta)."""
    try:
        return _encode_video(encoder, video, protocol, view=view, batch_size=batch_size,
                             seek=True, prepared=prepared, label_name=label_name)
    except VideoSeekError as exc:
        print(f"[decode] {video}: {exc}. Retrying the window with sequential decoding.", flush=True)
        return _encode_video(encoder, video, protocol, view=view, batch_size=batch_size,
                             seek=False, label_name=label_name)


def _video_batches(video, protocol, view, batch_size, seek, *, buffer_count=1):
    reader = FrameReader(video, view)
    batches = pair_batches(
        reader,
        gap=protocol["gap"],
        batch_size=batch_size,
        bucket=protocol["bucket"],
        start=protocol["start_frame"],
        end=protocol["end_frame"],
        pair_stride=protocol["pair_stride"],
        seek=seek,
        reuse_buffer=True,
        buffer_count=buffer_count,
    )
    return reader, batches


def _encode_video(encoder, video, protocol, *, view, batch_size, seek, prepared=None, label_name=None):
    if prepared is None:
        reader, batches = _video_batches(video, protocol, view, batch_size, seek)
    else:
        reader, prefetch = prepared
        batches = prefetch.consume()
    latents, pair_indices = [], []
    shape = bucket = native_hw = None
    with closing(batches):
        for batch_number, (images, pairs, batch_bucket, batch_hw) in enumerate(batches, 1):
            # The copy back to CPU finishes before the next iteration releases the image buffer.
            tokens = encoder(images, representation=protocol["representation"]).cpu().numpy()
            current = list(tokens.shape[1:])
            if shape is not None and current != shape:
                raise ValueError("Token shape changed within the video")
            shape = current
            encoded = tokens.reshape(len(tokens), -1).astype(np.float16)
            if not np.isfinite(encoded).all():
                raise ValueError("Non-finite values after float16 conversion")
            latents.append(encoded)
            pair_indices.append(pairs)
            bucket, native_hw = batch_bucket, batch_hw
            if label_name is not None:
                print(f"[batch] {label_name} batch={batch_number} pairs={len(pairs)}", flush=True)
    if not latents:
        raise ValueError(f"No frame pairs encoded for {video}")
    latent = np.concatenate(latents).astype(np.float16)
    pairs = np.concatenate(pair_indices).astype(np.int32)
    count = len(latent)
    num_frames = reader.num_frames
    meta = {
        "protocol": protocol,
        "view": view,
        "num_latents": count,
        "num_frames": num_frames,
        "unpaired_tail_frames": num_frames
        - ((count - 1) * protocol["pair_stride"] + protocol["gap"] + 1),
        "code_shape": shape,
        "bucket": bucket,
        "target_hw": list(BUCKETS[bucket]),
        "source_hw": list(native_hw),
        "fps": reader.fps,
    }
    return latent, pairs, meta


def write_npz(destination, latent, pairs, meta):
    """np.savez seeks on write, which FUSE mounts reject; build in memory then write bytes.

    The bytes land in a ``.partial`` sibling and are published with ``os.replace``, so this
    writer can never leave a half-written file at the final path -- not for its own skip check
    on a rerun, and not for a ``rebuild_latent_manifest`` sweep running while labeling is still
    in progress. ``read_complete_latent`` is the second layer, covering files truncated by something
    other than this function. Not imported from ``stream_extract_tar.atomic_write`` because that
    module pulls in cv2 and torch at import time, which the labeler does not otherwise need.
    """
    buffer = io.BytesIO()
    np.savez(buffer, latent_action=latent, pair_indices=pairs, meta=json.dumps(meta))
    payload = buffer.getvalue()
    pending = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        pending.write_bytes(payload)
        os.replace(pending, destination)
    finally:
        pending.unlink(missing_ok=True)


def _has_label_array(archive, member, shape, dtype):
    entry = archive.getinfo(member)
    with archive.open(entry) as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            actual_shape, _, actual_dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            actual_shape, _, actual_dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            return False
        return (actual_shape == shape and actual_dtype.newbyteorder("=") == dtype
                and entry.file_size == stream.tell() + math.prod(shape) * actual_dtype.itemsize)


def read_complete_latent(path):
    """Check the stored window and array headers without loading all latent values."""
    try:
        with zipfile.ZipFile(path) as archive:
            with archive.open("meta.npy") as stream:
                meta = json.loads(np.lib.format.read_array(stream, allow_pickle=False).item())
            count, code_shape = meta["num_latents"], meta["code_shape"]
            protocol = meta["protocol"]
            if (type(count) is not int or count < 1 or not isinstance(code_shape, list)
                    or not code_shape or any(type(dim) is not int or dim < 1 for dim in code_shape)
                    or not isinstance(protocol, dict)):
                return None
            frames = meta.get("num_frames")
            if protocol.get("end_frame") is not None:
                window_frames = protocol["end_frame"] - protocol.get("start_frame", 0)
                if frames is not None and frames != window_frames:
                    return None
                frames = window_frames
            if frames is not None:
                gap, stride = protocol.get("gap"), protocol.get("pair_stride")
                if (type(frames) is not int or type(gap) is not int or gap < 1
                        or type(stride) is not int or stride < 1
                        or count != 1 + (frames - gap - 1) // stride):
                    return None
            if (_has_label_array(archive, "latent_action.npy", (count, math.prod(code_shape)), np.float16)
                    and _has_label_array(archive, "pair_indices.npy", (count, 2), np.int32)):
                return meta
    except (FileNotFoundError, EOFError, KeyError, TypeError, ValueError, zipfile.BadZipFile):
        pass
    return None


def build_items(args):
    if args.metadata:
        items = []
        for path in args.metadata:
            with path.open(encoding="utf-8") as handle:
                entries = json.load(handle)
            if not isinstance(entries, list):
                raise SystemExit(f"{path}: --metadata must contain a JSON list of episode entries")
            items.extend(entries)
        # Relative video_path values resolve against the current working directory, matching
        # the Stage-1 consumer (latent_pretrain.py) and every committed config/manifest in
        # this repo -- so run from the repo root, as documented in README.md.
        return items, Path.cwd()
    item = {
        "dataset": "videos",
        "episode_id": args.video.stem,
        "views": {args.view: {"video_path": str(args.video.resolve())}},
    }
    return [item], Path.cwd()


def validate_latent_dim(meta, expected, path):
    if expected is not None and meta.get("code_shape") != [expected]:
        raise ValueError(f"{path}: expected latent shape [{expected}], got {meta.get('code_shape')}")


def verify_existing_latent(path, meta, protocol, expected_dim):
    validate_latent_dim(meta, expected_dim, path)
    previous = meta.get("protocol") or {}
    fields = ("checkpoint_sha256", "source", "source_size", "source_mtime_ns", "gap",
              "pair_stride", "start_frame", "end_frame", "bucket", "normalization",
              "representation", "precision", "expected_fps")
    changed = [field for field in fields if previous.get(field) != protocol.get(field)]
    if changed:
        raise ValueError(f"{path}: existing latent differs in {', '.join(changed)}. "
                         "Use --overwrite or choose a new output directory.")


@dataclass
class LabelWindow:
    name: str
    view: str
    destination: Path
    protocol: dict


def _pending_episode(item, base, args, protocol_defaults, counts):
    dataset = safe_name(item["dataset"])
    episode = safe_name(item["episode_id"], nested=True)
    windows = []
    verify = args.verify_existing or args.expected_latent_dim is not None
    for view, source_view in item["views"].items():
        safe_name(view)
        if args.views and view not in args.views:
            continue
        name = f"{dataset}/{episode}/{view}"
        destination = args.output_dir / "latents" / dataset / episode / view / "latent.npz"
        exists = not args.overwrite and destination.exists()
        existing = read_complete_latent(destination) if exists else None
        if existing is not None and not verify:
            print(f"[skip] {name}: exists (use --overwrite)", flush=True)
            counts["skipped"] += 1
            continue
        if exists and existing is None:
            print(f"[relabel] {name}: existing latent.npz is incomplete or unreadable", flush=True)
        video, source_size, source_mtime_ns = resolve_source(source_view["video_path"], base)
        start = int(source_view.get("start_frame", item.get("start_frame", 0)) or 0)
        end = source_view.get("end_frame", item.get("end_frame"))
        protocol = {
            **protocol_defaults,
            "source": str(video), "source_size": source_size, "source_mtime_ns": source_mtime_ns,
            "start_frame": start, "end_frame": int(end) if end is not None else None,
        }
        expected_fps = source_view.get("expected_fps")
        if expected_fps is not None:
            protocol["expected_fps"] = expected_fps
        if existing is not None:
            verify_existing_latent(destination, existing, protocol, args.expected_latent_dim)
            print(f"[skip] {name}: existing latent verified", flush=True)
            counts["skipped"] += 1
            continue
        windows.append(LabelWindow(name, view, destination, protocol))
    return windows


def _prepared_windows(windows, executor, workers, batch_size):
    if not workers:
        for window in windows:
            yield window, None
        return

    windows = iter(windows)
    pending = deque()
    try:
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < workers:
                window = next(windows, None)
                if window is None:
                    exhausted = True
                    break
                reader, batches = _video_batches(
                    window.protocol["source"], window.protocol, window.view, batch_size,
                    True, buffer_count=PrefetchedBatches.buffer_count,
                )
                prefetch = PrefetchedBatches()
                # Register before launching so an interrupt always cancels the worker.
                pending.append((window, (reader, prefetch)))
                prefetch.start(executor, batches)
            if pending:
                yield pending[0]
                _, prepared = pending.popleft()
                prepared[1].close()
    finally:
        # Cancel blocked producers before the executor waits for its threads to finish.
        for _, prepared in pending:
            prepared[1].close()


def _episode_jobs(items, base, args, protocol_defaults, counts):
    workers = args.prefetch_workers
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rynnlam-prep") if workers else nullcontext()
    with pool as executor:
        for item in items:
            windows = _pending_episode(item, base, args, protocol_defaults, counts)
            with closing(_prepared_windows(windows, executor, workers, args.batch_size)) as jobs:
                yield from jobs


def main(argv=None, *, show_next_steps=True):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="One RGB video -> a single latent.npz")
    source.add_argument(
        "--metadata", type=Path, action="append",
        help="JSON list of {dataset, episode_id, views:{view:{video_path,...}}} entries (repeatable)",
    )
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Trained RynnLAM motion checkpoint (from scripts/train_rynnlam.py)")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--representation",
                        choices=["ktoken", "z", "zcam", "hints_pool", "ktoken_zcam"],
                        default="ktoken_zcam")
    parser.add_argument("--view", default="head", help="View name used with --video")
    parser.add_argument("--views", nargs="+", help="Only label these view names from --metadata")
    parser.add_argument("--gap", type=int, default=4, help="RGB frame interval within each pair")
    parser.add_argument("--pair-stride", "--stride", dest="pair_stride", type=int, default=4,
                        help="Step between pair start frames; default 4 gives (0,4),(4,8),...")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bucket", choices=["auto", *BUCKETS], default="auto")
    parser.add_argument("--normalization", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=None,
                        help="Override fps in meta when the container reports none")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-label even if the destination latent.npz already exists")
    parser.add_argument("--verify-existing", action="store_true",
                        help="Check a complete latent's labeling settings before reusing it")
    parser.add_argument("--expected-latent-dim", type=int,
                        help="Require a one-dimensional latent of this width")
    parser.add_argument("--trust-checkpoint", action="store_true",
                        help="Allow Python object deserialization; only for a trusted checkpoint")
    parser.add_argument("--prefetch-workers", type=int, default=0, metavar="N",
                        help="CPU threads preparing video windows (0: prepare on the inference thread)")
    args = parser.parse_args(argv)

    if args.gap < 1 or args.pair_stride < 1 or args.batch_size < 1:
        parser.error("--gap, --pair-stride and --batch-size must be positive")
    if args.expected_latent_dim is not None and args.expected_latent_dim < 1:
        parser.error("--expected-latent-dim must be positive")
    if args.prefetch_workers < 0:
        parser.error("--prefetch-workers must be nonnegative")

    items, base = build_items(args)
    checkpoint_hash = file_sha256(args.checkpoint)
    implementation_hash = implementation_sha256()
    labeler_hash = file_sha256(Path(__file__).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    encoder = None
    counts = Counter()
    protocol_defaults = {
        "schema_version": 2, "checkpoint_sha256": checkpoint_hash,
        "implementation_sha256": implementation_hash, "labeler_sha256": labeler_hash,
        "gap": args.gap, "pair_stride": args.pair_stride, "bucket": args.bucket,
        "normalization": args.normalization, "representation": args.representation,
        "precision": args.precision, "device": str(device),
    }
    with closing(_episode_jobs(items, base, args, protocol_defaults, counts)) as jobs:
        for window, prepared in jobs:
            if encoder is None:
                encoder = RynnLAMEncoder(
                    args.checkpoint, device=args.device,
                    normalize=args.normalization == "imagenet", precision=args.precision,
                    trust_checkpoint=args.trust_checkpoint, checkpoint_sha256=checkpoint_hash,
                )
                # Equal checkpoints can produce slightly different bf16 results on different GPUs.
                device_name = (torch.cuda.get_device_name(encoder.device)
                               if encoder.device.type == "cuda" else "cpu")
            protocol = window.protocol
            protocol["device_name"] = device_name
            latent, pairs, meta = label_video(
                encoder, protocol["source"], protocol, view=window.view, batch_size=args.batch_size,
                prepared=prepared, label_name=window.name,
            )
            validate_latent_dim(meta, args.expected_latent_dim, window.destination)
            expected_fps = protocol.get("expected_fps")
            if expected_fps is not None:
                if meta["fps"] and not math.isclose(meta["fps"], expected_fps, rel_tol=1e-3):
                    raise ValueError(f"{protocol['source']}: decoded fps {meta['fps']:g} differs from "
                                     f"manifest fps {expected_fps:g}")
                meta["fps"] = expected_fps
            if args.fps is not None and not meta.get("fps"):
                meta["fps"] = args.fps
            write_npz(window.destination, latent, pairs, meta)
            counts["labeled"] += 1
            print(f"{window.name}: {meta['num_latents']} latents, "
                  f"shape={meta['code_shape']} -> {window.destination}", flush=True)

    latents = args.output_dir / "latents"
    labeled, skipped = counts["labeled"], counts["skipped"]
    print(f"\n[label] wrote {labeled} latent.npz ({skipped} skipped) under {latents}")
    if not show_next_steps:
        return
    build = args.output_dir / "build"
    manifest = build / "manifests" / "corpus.jsonl"
    # --data-root is the directory holding the <Dataset>.json episode files, which is the
    # parent of --metadata when one was given.
    data_root = args.metadata[0].parent if args.metadata else Path("<dir holding Dataset.json>")
    print("[label] next -- build the Stage-1 index from these latents:")
    print(f"    python scripts/rebuild_latent_manifest.py --data-root {data_root} \\")
    print(f"        --latent-dir {latents} --out-dir {build} --name corpus --combine")
    print(f"    python scripts/build_latent_pretrain_index.py --manifest {manifest} \\")
    print(f"        --out {build / 'index_c6.npz'} --latent-chunk 6")
    print(f"    python scripts/build_latent_stats.py --manifest {manifest} \\")
    print(f"        --out {build / 'latent_stats.json'} --latent-dim 608")


if __name__ == "__main__":
    main()
