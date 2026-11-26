#!/usr/bin/env python3
"""Prepare or launch RynnVLA 4B training with the same config as LeRobot v3 labeling."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import sys
from tempfile import mkdtemp
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rynnvla.api import launch
from rynnvla.constants import RobotType
from rynnvla.utils.lerobot_v3 import dataset_specs, read_info, resolve_path, vector_size
from rynnvla.utils.manifest_io import write_json


def fingerprint(payload):
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def stage1_mix(config, datasets, config_dir, work, *, rebuild, resume):
    from build_latent_stats import main as build_stats
    from label_lerobot_v3 import prepare_dataset
    from rebuild_latent_manifest import Options, rebuild_episode
    from rynnvla.datasets.vla_datasets.latent_loader import load_latent_stats
    from rynnvla.datasets.vla_datasets.latent_pretrain import build_index

    settings, label = config["rynnvla"], config["rynnlam"]
    stage1 = settings.get("stage1", settings)
    dim = settings["latent_action_dim"]
    chunk, seconds = stage1["latent_chunk"], stage1["latent_step_seconds"]
    gap, stride = label["gap"], label["pair_stride"]
    preset = json.loads((REPO_ROOT / "rynnvla/configs" / launch.PRESETS["stage1"]).read_text())
    if (dim, chunk) != (preset["latent_action_dim"], preset["config_overrides"]["latent_action_chunk_size"]):
        raise ValueError("latent_action_dim and latent_chunk must match the RynnVLA-Latent-4B preset")
    if (any(type(value) is not int or value < 1 for value in (gap, stride))
            or not math.isfinite(seconds) or seconds <= 0):
        raise ValueError("gap, pair_stride and latent_step_seconds must be positive")
    label_root = resolve_path(config_dir, config["output_dir"])
    stats_source = stage1.get("latent_stats_path")
    stats_path = resolve_path(config_dir, stats_source) if stats_source else None
    preparation = {
        "datasets": [{**spec, "path": str(root)} for root, spec in datasets],
        "latents": str(label_root / "latents"),
        "gap": gap, "pair_stride": stride, "representation": label["representation"],
        "latent_action_dim": dim, "latent_chunk": chunk, "latent_step_seconds": seconds,
        "latent_stats_path": str(stats_path) if stats_path else None,
    }
    pointer = work / f"stage1-{fingerprint(preparation)}.json"
    if not rebuild and pointer.is_file():
        mixture = Path(json.loads(pointer.read_text(encoding="utf-8"))["mixture"])
        if all((mixture.parent / name).is_file() for name in
               ("mixture.json", "manifest.jsonl", "index.npz", "latent_stats.json")):
            return mixture
    if resume:
        raise ValueError("The prepared Stage 1 inputs for this config are missing; resume needs the original data")
    if not (label_root / "latents").is_dir():
        raise FileNotFoundError(f"Missing RynnLAM labels under {label_root / 'latents'}")

    directory = Path(mkdtemp(dir=work, prefix="stage1-"))
    manifest, index, stats = (directory / name for name in ("manifest.jsonl", "index.npz", "latent_stats.json"))
    options = Options(SimpleNamespace(
        latent_dir=str(label_root / "latents"), tail_bytes=1 << 16, max_caption_chars=512,
        expected_latent_dim=dim, expected_pair_stride=stride,
        expected_representation=label["representation"], default_fps=30.0, source_gap=gap,
    ))
    try:
        with manifest.open("w", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=16) as pool:
            for root, dataset in datasets:
                items = prepare_dataset(root, dataset, gap=gap, pair_stride=stride, latent_chunk=chunk,
                                        latent_step_seconds=seconds)
                kept, dropped = 0, Counter()
                for entry, reasons in pool.map(partial(rebuild_episode, opt=options), items):
                    if entry is None:
                        dropped[reasons[-1]] += 1
                        continue
                    fps = entry["views"][0]["fps"]
                    latent_stride = max(1, round(fps * seconds / stride))
                    if min(view["num_latents"] for view in entry["views"]) < (chunk - 1) * latent_stride + 1:
                        dropped["short_latent_chunk"] += 1
                        continue
                    print(json.dumps(entry, ensure_ascii=False), file=handle)
                    kept += 1
                print(f"[stage1] {dataset['name']}: {kept}/{len(items)} episodes, dropped={dict(dropped)}", flush=True)
                if kept == 0:
                    raise ValueError(f"{dataset['name']}: no complete labeled episodes can supply a Stage 1 chunk")
        build_index(str(manifest), str(index), chunk, seconds)
        if stats_path is None:
            build_stats(["--manifest", str(manifest), "--out", str(stats), "--latent-dim", str(dim)])
        else:
            shutil.copyfile(stats_path, stats)
        load_latent_stats(str(stats), dim)
        mixture = directory / "mixture.json"
        write_json(mixture, [{
            "data_type": "LatentPretrainDataset", "data_path": str(index), "latent_stats_path": str(stats),
            "latent_chunk": chunk, "latent_step_seconds": seconds,
        }])
        write_json(pointer, {"mixture": str(mixture)})
        return mixture
    except BaseException:
        shutil.rmtree(directory)
        raise


def stage2_mix(datasets, work):
    from rynnvla.datasets.vla_datasets import LeRobotV3Dataset

    sources, widths, names = [], {}, {}
    for root, dataset in datasets:
        cameras = dataset["cameras"]
        info = read_info(root, cameras)
        robot = RobotType(dataset.get("robot_type", "default")).value
        for field in ("state_key", "action_key"):
            key = dataset[field]
            width = vector_size(info, key, root)
            feature_names = info["features"][key].get("names")
            signature = (robot, field)
            if widths.setdefault(signature, width) != width:
                raise ValueError(f"{root}: {field}={key!r} has inconsistent width across {robot} subsets")
            if feature_names and names.setdefault(signature, feature_names) != feature_names:
                raise ValueError(f"{root}: {field}={key!r} has inconsistent feature order across {robot} subsets")
        source = {
            "data_path": str(root), "cameras": cameras,
            "robot_type": robot, "state_key": dataset["state_key"], "action_key": dataset["action_key"],
            "episode_indices": dataset.get("episode_indices"),
        }
        print(f"[stage2] {dataset['name']}: {robot}, {len(cameras)} cameras, "
              f"state={widths[robot, 'state_key']} action={widths[robot, 'action_key']}, fps={info['fps']}", flush=True)
        prepared = LeRobotV3Dataset(**source, action_chunk_size=1).prepare_frames(work)
        sources.append({"data_type": "LeRobotV3Dataset", **source, "frame_cache": str(prepared)})
    mixture = work / f"stage2-{fingerprint(sources)}.json"
    write_json(mixture, sources)
    return mixture


def validate_initialization(model):
    from rynnvla.api.export_checkpoint import checkpoint_metadata

    checkpoint_metadata(model, stage="stage1")
    weights = ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin")
    if not any((model / name).is_file() and (model / name).stat().st_size for name in weights):
        raise ValueError(f"No RynnVLA model weights or weight index in {model}")


def stage_output_path(settings, stage):
    output_dir = settings.get(stage, {}).get("output_dir")
    if output_dir is not None:
        return Path(output_dir)
    return Path(settings["training_output_dir"]) / stage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("stage", choices=("stage1", "stage2"))
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/config.json")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare the stage's data mixture without launching")
    parser.add_argument("--rebuild-data", action="store_true", help="Build a new Stage 1 data snapshot")
    parser.add_argument("--resume", action="store_true", help="Resume the same stage's training directory")
    args, overrides = parser.parse_known_args(argv)
    if args.rebuild_data and (args.stage != "stage1" or args.resume):
        parser.error("--rebuild-data is only for a fresh Stage 1 preparation")
    for token in launch.normalize_options(overrides):
        if token.split("=", 1)[0] in {"--stage", "--model_path", "--data_mixture", "--output_dir"}:
            parser.error("stage, model, data mixture and output directory are selected by this config")

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_dir = config_path.parent
    settings = config["rynnvla"]
    output_dir = resolve_path(config_dir, stage_output_path(settings, args.stage))
    stage_args = []
    if args.stage == "stage1":
        model_path = settings.get("stage1", settings)["pretrained_model_path"]
    else:
        stage2 = settings.get("stage2", {})
        model_path = stage2.get("pretrained_model_path")
        if model_path is None:
            model_path = stage_output_path(settings, "stage1") / "ema"
        if "action_chunk_size" in stage2:
            chunk_size = stage2["action_chunk_size"]
            if type(chunk_size) is not int or chunk_size < 1:
                parser.error("rynnvla.stage2.action_chunk_size must be a positive integer")
            stage_args.extend(("--action-chunk-size", str(chunk_size)))
    model = resolve_path(config_dir, model_path)
    if not args.prepare_only and not args.resume:
        if not model.is_dir():
            raise FileNotFoundError(f"{args.stage} pretrained model directory is missing: {model}")
        validate_initialization(model)
    datasets = list(dataset_specs(config, config_dir))
    prepared_data_dir = settings.get("prepared_data_dir")
    if prepared_data_dir is None:
        prepared_data_dir = Path(settings["training_output_dir"]) / "data"
    work = resolve_path(config_dir, prepared_data_dir)
    work.mkdir(parents=True, exist_ok=True)
    mixture = (stage1_mix(config, datasets, config_dir, work, rebuild=args.rebuild_data, resume=args.resume)
               if args.stage == "stage1" else stage2_mix(datasets, work))
    print(f"[{args.stage}] data mixture: {mixture}", flush=True)
    if args.prepare_only:
        return 0
    return launch.main([
        "--stage", args.stage, "--model-path", str(model), "--data-mixture", str(mixture),
        "--output-dir", str(output_dir), *(["--resume"] if args.resume else []),
        *stage_args, *overrides,
    ])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        sys.exit(f"[train_lerobot_v3] {exc}")
