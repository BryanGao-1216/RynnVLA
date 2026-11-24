#!/usr/bin/env python3
"""Prepare episode manifests for local LeRobot v3 videos and run the RynnLAM labeler."""

import argparse
import json
import math
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rynnvla.constants import VIEW_NAME_TO_ROLE
from rynnvla.utils.manifest_io import join_caption

MANIFEST_VIEW = {role: name for name, role in VIEW_NAME_TO_ROLE.items()}
VIDEO_POINTER_FIELDS = ("chunk_index", "file_index", "from_timestamp", "to_timestamp")


def resolve_path(config_dir, value):
    return (config_dir / Path(value).expanduser()).resolve()


def episode_views(root, row, cameras, video_template, fps, context):
    views = {}
    for key, role in cameras.items():
        prefix = f"videos/{key}/"
        chunk, shard, from_ts, to_ts = (row[prefix + field] for field in VIDEO_POINTER_FIELDS)
        if (chunk is None or shard is None or from_ts is None or to_ts is None
                or not math.isfinite(from_ts) or not math.isfinite(to_ts)
                or chunk < 0 or shard < 0 or from_ts < 0 or to_ts <= from_ts):
            raise ValueError(f"{context}: invalid video pointer for {key}")
        start = round(from_ts * fps)
        end = round(to_ts * fps)
        if end - start != row["length"]:
            raise ValueError(
                f"{context}: {key} has {end - start} frames at {fps:g} fps, "
                f"episode length is {row['length']}"
            )
        video_path = root / video_template.format(video_key=key, chunk_index=chunk, file_index=shard)
        if not video_path.is_file():
            raise FileNotFoundError(f"{context}: missing video {video_path}")
        views[MANIFEST_VIEW[role]] = {
            "video_path": str(video_path), "start_frame": start, "end_frame": end, "expected_fps": fps,
        }
    return views


def prepare_dataset(root, dataset, *, gap, pair_stride, latent_chunk, latent_step_seconds):
    import pyarrow.parquet as pq

    name = dataset["name"]
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"{info_path}: expected codebase_version='v3.0'")
    fps = float(info["fps"])
    video_template = info.get("video_path")
    if not math.isfinite(fps) or fps <= 0 or not video_template:
        raise ValueError(f"{info_path}: a positive fps and video_path template are required")

    features = info["features"]
    for field in ("state_key", "action_key"):
        key = dataset.get(field)
        if key is not None and key not in features:
            raise ValueError(f"{info_path}: {field}={key!r} is missing from features")
    cameras = dataset["cameras"]
    if not isinstance(cameras, dict) or not cameras:
        raise ValueError(f"{name}: cameras must map LeRobot feature keys to RynnVLA roles")
    if any(role not in MANIFEST_VIEW for role in cameras.values()):
        raise ValueError(f"{name}: supported camera roles are {', '.join(MANIFEST_VIEW)}")
    if len(set(cameras.values())) != len(cameras):
        raise ValueError(f"{name}: each selected camera must have a distinct RynnVLA role")
    for key in cameras:
        if features.get(key, {}).get("dtype") != "video":
            raise ValueError(f"{info_path}: camera {key!r} must be a video feature")

    selection = dataset.get("episode_indices")
    if selection is not None and (not isinstance(selection, list) or not selection
                                  or any(type(index) is not int or index < 0 for index in selection)):
        raise ValueError(f"{name}: episode_indices must be a nonempty list of nonnegative integers")
    selected = set(selection) if selection is not None else None
    shards = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not shards:
        raise FileNotFoundError(f"{name}: no episode Parquet shards under {root / 'meta/episodes'}")
    columns = ["episode_index", "length", "tasks"]
    columns.extend(f"videos/{key}/{field}" for key in cameras for field in VIDEO_POINTER_FIELDS)

    items = []
    seen = set()
    skipped = usable = 0
    latent_stride = max(1, round(fps * latent_step_seconds / pair_stride))
    min_stage1_frames = (latent_chunk - 1) * latent_stride * pair_stride + gap + 1
    for shard in shards:
        for row in pq.read_table(shard, columns=columns).to_pylist():
            episode = row["episode_index"]
            if selected is not None and episode not in selected:
                continue
            context = f"{name} episode {episode} ({shard})"
            if episode in seen:
                raise ValueError(f"{context}: duplicate episode_index")
            seen.add(episode)
            if row["length"] <= gap:
                skipped += 1
                continue
            caption = join_caption(row["tasks"], max_chars=512)
            if not caption:
                raise ValueError(f"{context}: no task instruction in episode metadata")
            views = episode_views(root, row, cameras, video_template, fps, context)
            items.append({"dataset": name, "episode_id": episode, "views": views, "caption": caption})
            usable += row["length"] >= min_stage1_frames

    if selected is not None and selected - seen:
        raise ValueError(f"{name}: missing episode indices {sorted(selected - seen)}")
    if not items:
        raise ValueError(f"{name}: no episodes longer than the frame-pair gap ({gap})")
    print(f"[{name}] {len(items)} episodes × {len(cameras)} cameras at {fps:g} fps. "
          f"{usable} fit a Stage 1 chunk (minimum {min_stage1_frames} frames). "
          f"{skipped} skipped because length <= gap.", flush=True)
    return sorted(items, key=lambda item: item["episode_id"])


def write_manifest(path, items):
    handle = NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(items, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


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
    checkpoint = resolve_path(config_dir, label_options["checkpoint"])
    if not args.prepare_only and not checkpoint.is_file():
        parser.error(f"RynnLAM checkpoint not found: {checkpoint}")

    stage1 = config["rynnvla"]
    gap = label_options["gap"]
    pair_stride = label_options["pair_stride"]
    latent_dim = stage1["latent_action_dim"]
    latent_chunk = stage1["latent_chunk"]
    step_seconds = stage1["latent_step_seconds"]
    if any(type(value) is not int or value <= 0 for value in (gap, pair_stride, latent_dim, latent_chunk)):
        parser.error("gap, pair_stride, latent_action_dim and latent_chunk must be positive integers")
    if not math.isfinite(step_seconds) or step_seconds <= 0:
        parser.error("latent_step_seconds must be finite and positive")

    datasets = config["datasets"]
    if not isinstance(datasets, list) or not datasets:
        parser.error("datasets must be a nonempty list")
    manifests = {}
    for source in datasets:
        dataset = {**config.get("dataset_defaults", {}), **source}
        name = dataset["name"]
        if (not isinstance(name, str) or name in {"", ".", ".."}
                or "/" in name or "\\" in name or "," in name or name in manifests):
            parser.error(f"Each dataset needs a unique name that is a path component: {name!r}")
        root = resolve_path(config_dir, dataset["path"])
        manifests[name] = prepare_dataset(root, dataset, gap=gap, pair_stride=pair_stride,
                                         latent_chunk=latent_chunk, latent_step_seconds=step_seconds)

    source_dir = output / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    label_args = ["--checkpoint", str(checkpoint), "--output-dir", str(output),
                  "--expected-latent-dim", str(latent_dim), "--verify-existing"]
    for name, items in manifests.items():
        path = source_dir / f"{name}.json"
        write_manifest(path, items)
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
    label(label_args, show_next_steps=False)


if __name__ == "__main__":
    try:
        main()
    except (KeyError, TypeError, ValueError, OSError, ImportError, RuntimeError) as exc:
        sys.exit(f"[label_lerobot_v3] {exc}")
