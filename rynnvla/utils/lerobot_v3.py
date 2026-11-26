"""Read the shared dataset and video metadata for LeRobot v3 training and labeling."""

import json
import math
from pathlib import Path

from ..constants import ACTION_DIM, VIEW_NAME_TO_ROLE


MANIFEST_VIEW = {role: name for name, role in VIEW_NAME_TO_ROLE.items()}
VIDEO_POINTER_FIELDS = ("chunk_index", "file_index", "from_timestamp", "to_timestamp")


def resolve_path(config_dir, value):
    return (config_dir / Path(value).expanduser()).resolve()


def dataset_specs(config, config_dir):
    sources = config["datasets"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("datasets must be a nonempty list")
    seen = set()
    for source in sources:
        dataset = {**config.get("dataset_defaults", {}), **source}
        name = dataset["name"]
        if (not isinstance(name, str) or name in {"", ".", ".."}
                or "/" in name or "\\" in name or "," in name or name in seen):
            raise ValueError(f"Each dataset needs a unique name that is a path component: {name!r}")
        seen.add(name)
        yield resolve_path(config_dir, dataset["path"]), dataset


def read_info(root, cameras):
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"{info_path}: expected codebase_version='v3.0'")
    fps = float(info["fps"])
    if not math.isfinite(fps) or fps <= 0 or not info.get("video_path"):
        raise ValueError(f"{info_path}: a positive fps and video_path template are required")
    if not isinstance(cameras, dict) or not cameras:
        raise ValueError(f"{info_path}: cameras must map LeRobot feature keys to RynnVLA roles")
    if any(role not in MANIFEST_VIEW for role in cameras.values()):
        raise ValueError(f"{info_path}: supported camera roles are {', '.join(MANIFEST_VIEW)}")
    if len(set(cameras.values())) != len(cameras):
        raise ValueError(f"{info_path}: each selected camera must have a distinct RynnVLA role")
    for key in cameras:
        if info["features"].get(key, {}).get("dtype") != "video":
            raise ValueError(f"{info_path}: camera {key!r} must be a video feature")
    return info


def vector_size(info, key, root):
    feature = info["features"].get(key) or {}
    shape = feature.get("shape")
    if (not shape or len(shape) != 1 or type(shape[0]) is not int or not 1 <= shape[0] <= ACTION_DIM
            or feature.get("dtype") not in {"float32", "float64", "float16", "int32", "int64"}):
        raise ValueError(f"{root}: expected a numeric vector of width 1–{ACTION_DIM} for {key!r}, got {feature}")
    return shape[0]


def episode_rows(root, cameras, episode_indices=None):
    import pyarrow.parquet as pq

    if episode_indices is not None and (
        not isinstance(episode_indices, list) or not episode_indices
        or any(type(index) is not int or index < 0 for index in episode_indices)
    ):
        raise ValueError(f"{root}: episode_indices must be a nonempty list of nonnegative integers")
    selected = set(episode_indices) if episode_indices is not None else None
    shards = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No episode Parquet shards under {root / 'meta/episodes'}")
    columns = ["episode_index", "length", "tasks"]
    columns.extend(f"videos/{key}/{field}" for key in cameras for field in VIDEO_POINTER_FIELDS)

    seen = set()
    for shard in shards:
        for row in pq.read_table(shard, columns=columns).to_pylist():
            episode = row["episode_index"]
            if selected is not None and episode not in selected:
                continue
            context = f"{root.name} episode {episode} ({shard})"
            if episode in seen:
                raise ValueError(f"{context}: duplicate episode_index")
            seen.add(episode)
            yield row, context
    if selected is not None and selected - seen:
        raise ValueError(f"{root}: missing episode indices {sorted(selected - seen)}")


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
