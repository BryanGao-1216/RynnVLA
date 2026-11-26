#!/usr/bin/env python3
"""Prepare episode manifests for local LeRobot v3 videos and run the RynnLAM labeler."""

import argparse
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rynnvla.utils.lerobot_v3 import dataset_specs, episode_rows, episode_views, read_info, resolve_path
from rynnvla.utils.manifest_io import join_caption, write_json


def prepare_dataset(root, dataset, *, gap, pair_stride, latent_chunk, latent_step_seconds):
    name = dataset["name"]
    info_path = root / "meta" / "info.json"
    cameras = dataset["cameras"]
    info = read_info(root, cameras)
    fps = float(info["fps"])

    features = info["features"]
    for field in ("state_key", "action_key"):
        key = dataset.get(field)
        if key is not None and key not in features:
            raise ValueError(f"{info_path}: {field}={key!r} is missing from features")

    items = []
    skipped = usable = 0
    latent_stride = max(1, round(fps * latent_step_seconds / pair_stride))
    min_stage1_frames = (latent_chunk - 1) * latent_stride * pair_stride + gap + 1
    for row, context in episode_rows(root, cameras, dataset.get("episode_indices")):
        if row["length"] <= gap:
            skipped += 1
            continue
        caption = join_caption(row["tasks"], max_chars=512)
        if not caption:
            raise ValueError(f"{context}: no task instruction in episode metadata")
        views = episode_views(root, row, cameras, info["video_path"], fps, context)
        items.append({"dataset": name, "episode_id": row["episode_index"], "views": views, "caption": caption})
        usable += row["length"] >= min_stage1_frames

    if not items:
        raise ValueError(f"{name}: no episodes longer than the frame-pair gap ({gap})")
    print(f"[{name}] {len(items)} episodes × {len(cameras)} cameras at {fps:g} fps. "
          f"{usable} fit a Stage 1 chunk (minimum {min_stage1_frames} frames). "
          f"{skipped} skipped because length <= gap.", flush=True)
    return sorted(items, key=lambda item: item["episode_id"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/config.json")
    parser.add_argument("--prepare-only", action="store_true", help="Write manifests without loading RynnLAM")
    parser.add_argument("--overwrite", action="store_true", help="Re-encode existing latent files")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_dir = config_path.parent
    output = resolve_path(config_dir, config["output_dir"])
    label_options = config["rynnlam"]
    async_prefetch = label_options.get("async_prefetch", False)
    prefetch_workers = label_options.get("prefetch_workers", 1)
    if type(async_prefetch) is not bool:
        parser.error("rynnlam.async_prefetch must be true or false")
    if type(prefetch_workers) is not int or prefetch_workers < 1:
        parser.error("rynnlam.prefetch_workers must be a positive integer")
    checkpoint = resolve_path(config_dir, label_options["checkpoint"])
    if not args.prepare_only and not checkpoint.is_file():
        parser.error(f"RynnLAM checkpoint not found: {checkpoint}")

    settings = config["rynnvla"]
    stage1 = settings.get("stage1", settings)
    gap = label_options["gap"]
    pair_stride = label_options["pair_stride"]
    latent_dim = settings["latent_action_dim"]
    latent_chunk = stage1["latent_chunk"]
    step_seconds = stage1["latent_step_seconds"]
    if any(type(value) is not int or value <= 0 for value in (gap, pair_stride, latent_dim, latent_chunk)):
        parser.error("gap, pair_stride, latent_action_dim and latent_chunk must be positive integers")
    if not math.isfinite(step_seconds) or step_seconds <= 0:
        parser.error("latent_step_seconds must be finite and positive")

    manifests = {}
    for root, dataset in dataset_specs(config, config_dir):
        manifests[dataset["name"]] = prepare_dataset(
            root, dataset, gap=gap, pair_stride=pair_stride, latent_chunk=latent_chunk,
            latent_step_seconds=step_seconds,
        )

    source_dir = output / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    label_args = ["--checkpoint", str(checkpoint), "--output-dir", str(output),
                  "--expected-latent-dim", str(latent_dim), "--verify-existing"]
    for name, items in manifests.items():
        path = source_dir / f"{name}.json"
        write_json(path, items)
        label_args.extend(("--metadata", str(path)))
    print(f"Wrote episode manifests to {source_dir}", flush=True)
    if args.prepare_only:
        return

    from label_latent import main as label

    for option in ("device", "precision", "batch_size", "representation", "gap", "pair_stride",
                   "normalization", "bucket"):
        label_args.extend(("--" + option.replace("_", "-"), str(label_options[option])))
    if args.overwrite:
        label_args.append("--overwrite")
    if async_prefetch:
        label_args.extend(("--prefetch-workers", str(prefetch_workers)))
    label(label_args, show_next_steps=False)


if __name__ == "__main__":
    try:
        main()
    except (KeyError, TypeError, ValueError, OSError, ImportError, RuntimeError) as exc:
        sys.exit(f"[label_lerobot_v3] {exc}")
