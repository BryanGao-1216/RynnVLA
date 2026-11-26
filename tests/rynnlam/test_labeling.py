"""RynnLAM labeling primitives: frame pairing, sparse intervals and identifier safety.

These cover the shared machinery ``scripts/label_latent.py`` is built on --
``rynnlam.video.pair_batches`` / ``FrameReader`` / ``resolve_source`` and the labeler's own
``safe_name`` path-traversal guard -- not the CLI's end-to-end behaviour, which
``scripts/smoke.sh`` exercises against the bundled sample.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
import sys
from threading import Event, current_thread, main_thread

import numpy as np
import pytest
import torch

from _scripts import load_script

labeling = load_script("label_latent")
from rynnlam.prefetch import PrefetchedBatches
import rynnlam.video as video_module
from rynnlam.video import FrameReader, pair_batches


def write_video(path, n=9, *, timestamps=None):
    import av

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = "yuv420p"
        if timestamps is not None:
            stream.gop_size = 6
            stream.codec_context.max_b_frames = 2
        for i in range(n):
            image = np.full((48, 64, 3), (i * 20) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            if timestamps is not None:
                frame.pts = timestamps[i]
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class FakeEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, images, representation="ktoken"):
        values = torch.from_numpy(images).mean((1, 2, 3, 4))
        return values[:, None, None].expand(-1, 2, 7)


def test_frame_pair_batches(tmp_path):
    video = tmp_path / "video.mp4"
    write_video(video)
    batches = list(pair_batches(FrameReader(video), gap=3, batch_size=4))
    assert [len(b[0]) for b in batches] == [4, 2]
    pairs = np.concatenate([b[1] for b in batches])
    np.testing.assert_array_equal(
        pairs, np.stack([np.arange(6), np.arange(6) + 3], axis=-1)
    )
    assert batches[0][2] == "4x3"
    assert batches[0][0].shape == (4, 2, 238, 322, 3)
    with pytest.raises(ValueError, match="incomplete interval"):
        list(pair_batches(FrameReader(video), end=20))
    with pytest.raises(ValueError, match="gap"):
        list(pair_batches(FrameReader(video), gap=9))


def test_unsafe_identifiers():
    for value in ["../escape", "/absolute", "", "..", "bad\\path"]:
        with pytest.raises(ValueError):
            labeling.safe_name(value, nested=True)


@pytest.mark.parametrize("workers", [0, 2])
def test_lerobot_v3_shared_video_windows_and_label_reuse(tmp_path, monkeypatch, capsys, workers):
    import pyarrow as pa
    import pyarrow.parquet as pq

    adapter = load_script("label_lerobot_v3")
    monkeypatch.setitem(sys.modules, "label_latent", labeling)

    class VectorEncoder(FakeEncoder):
        calls = 0
        loads = 0
        device = torch.device("cpu")

        def __init__(self, *args, **kwargs):
            self.__class__.loads += 1

        def __call__(self, images, representation):
            assert current_thread() is main_thread()
            self.__class__.calls += 1
            return torch.from_numpy(images).mean((1, 2, 3, 4))[:, None].expand(-1, 608)

    monkeypatch.setattr(labeling, "RynnLAMEncoder", VectorEncoder)
    root = tmp_path / "dataset"
    cameras = {"observation.images.front": "front_third", "observation.images.wrist": "left_wrist"}
    template = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    rows = [{"episode_index": i, "length": 5, "tasks": [f"Task {i}"]} for i in range(2)]
    for offset, key in enumerate(cameras):
        path = root / template.format(video_key=key, chunk_index=1, file_index=2)
        path.parent.mkdir(parents=True)
        write_video(path, n=11)
        for index, row in enumerate(rows):
            start = index * 5 + offset
            prefix = f"videos/{key}/"
            row.update({prefix + "chunk_index": 1, prefix + "file_index": 2,
                        prefix + "from_timestamp": start / 30, prefix + "to_timestamp": (start + 5) / 30})
    episodes = root / "meta" / "episodes" / "chunk-000"
    episodes.mkdir(parents=True)
    for index, row in enumerate(rows):
        pq.write_table(pa.Table.from_pylist([row]), episodes / f"file-{index:03d}.parquet")
    (root / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0", "fps": 30, "video_path": template,
        "features": {**{key: {"dtype": "video"} for key in cameras},
                     "custom.state": {"dtype": "float32"}, "custom.action": {"dtype": "float32"}},
    }))

    checkpoint = tmp_path / "RynnLAM.pt"
    checkpoint.write_bytes(b"fixture")
    config = {
        "output_dir": "out",
        "dataset_defaults": {"cameras": cameras, "state_key": "custom.state", "action_key": "custom.action"},
        "datasets": [{"name": name, "path": "dataset", "episode_indices": [1]} for name in ("first", "second")],
        "rynnlam": {"checkpoint": checkpoint.name, "device": "cpu", "precision": "fp32", "batch_size": 4,
                    "representation": "ktoken_zcam", "gap": 2, "pair_stride": 2,
                    "normalization": "imagenet", "bucket": "auto"},
        "rynnvla": {"latent_action_dim": 608, "latent_chunk": 2, "latent_step_seconds": 2 / 30},
    }
    if workers:
        config["rynnlam"].update(async_prefetch=True, prefetch_workers=workers)
        config["rynnvla"] = {"latent_action_dim": 608, "stage1": {"latent_chunk": 2, "latent_step_seconds": 2 / 30}}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    args = ["--config", str(config_path)]
    adapter.main(args + ["--prepare-only"])
    out = tmp_path / "out"
    item, = json.loads((out / "source/first.json").read_text())
    assert item["caption"] == "Task 1"
    assert item["views"]["global"]["start_frame"] == 5
    assert item["views"]["wrist_left"]["start_frame"] == 6
    assert VectorEncoder.calls == 0
    assert "Wrote episode manifests" in capsys.readouterr().out

    adapter.main(args)
    report = capsys.readouterr().out
    assert "[label] wrote 4 latent.npz (0 skipped)" in report
    assert "[timing:" not in report
    labeled_views = ("first/1/global", "first/1/wrist_left", "second/1/global", "second/1/wrist_left")
    assert [line for line in report.splitlines() if line.startswith("[batch]")] == [
        f"[batch] {view} batch=1 pairs=2" for view in labeled_views
    ]
    for view in labeled_views:
        assert f"{view}: 2 latents, shape=[608] -> {out / 'latents' / view / 'latent.npz'}" in report
    assert len(list((out / "latents").rglob("latent.npz"))) == 4
    with np.load(out / "latents/second/1/wrist_left/latent.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["pair_indices"], [[0, 2], [2, 4]])
        assert data["latent_action"].shape == (2, 608)
        assert data["latent_action"][0, 0] == pytest.approx(140 / 255, abs=0.02)
        meta = json.loads(data["meta"].item())
        assert meta["protocol"]["end_frame"] == 11
        assert meta["num_frames"] == 5
    assert VectorEncoder.calls == 4
    adapter.main(args)
    assert VectorEncoder.calls == 4
    assert VectorEncoder.loads == 1
    report = capsys.readouterr().out
    assert "[label] wrote 0 latent.npz (4 skipped)" in report
    assert [line for line in report.splitlines() if line.startswith("[skip]")] == [
        f"[skip] {view}: existing latent verified" for view in labeled_views
    ]
    assert "[batch]" not in report

    config["rynnlam"]["gap"] = 1
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="existing latent differs in gap"):
        adapter.main(args)
    adapter.main(args + ["--overwrite"])
    assert VectorEncoder.calls == 8
    if workers:
        config["rynnlam"]["async_prefetch"] = False
        config["output_dir"] = "serial"
        config_path.write_text(json.dumps(config))
        adapter.main(args)
        for path in (out / "latents").rglob("latent.npz"):
            serial = tmp_path / "serial" / path.relative_to(out)
            with np.load(path, allow_pickle=False) as actual, np.load(serial, allow_pickle=False) as expected:
                for field in ("latent_action", "pair_indices", "meta"):
                    np.testing.assert_array_equal(actual[field], expected[field])


def test_resume_completes_current_episode_before_next_prefetch(tmp_path, monkeypatch, capsys):
    class ResumeEncoder(FakeEncoder):
        device = torch.device("cpu")

    monkeypatch.setattr(labeling, "RynnLAMEncoder", ResumeEncoder)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture")
    views = ("wrist_left", "global", "wrist_right", "head")
    items = []
    for episode in (160, 161):
        video = tmp_path / f"{episode}.mp4"
        write_video(video)
        items.append({"dataset": "task_01", "episode_id": episode,
                      "views": {view: {"video_path": str(video)} for view in views}})
    full, first = tmp_path / "full.json", tmp_path / "first.json"
    full.write_text(json.dumps(items))
    first.write_text(json.dumps(items[:1]))
    output = tmp_path / "out"
    options = ["--checkpoint", str(checkpoint), "--output-dir", str(output), "--verify-existing",
               "--device", "cpu", "--precision", "fp32", "--batch-size", "2"]
    labeling.main(options + ["--metadata", str(first), "--views", "wrist_left", "wrist_right", "head"])
    capsys.readouterr()

    episode_dir = output / "latents/task_01/160"
    preserved = {}
    for view in ("wrist_left", "head"):
        path = episode_dir / view / "latent.npz"
        preserved[path] = path.read_bytes(), path.stat().st_mtime_ns
    incomplete = episode_dir / "wrist_right/latent.npz"
    with np.load(incomplete, allow_pickle=False) as archive:
        labels, indices, metadata = archive["latent_action"], archive["pair_indices"], archive["meta"]
    np.savez(incomplete, latent_action=labels, pair_indices=indices[:-1], meta=metadata)
    assert labeling.read_complete_latent(incomplete) is None

    video_batches = labeling._video_batches
    scheduled = []

    def check_before_prefetch(video, protocol, view, *args, **kwargs):
        episode = Path(video).stem
        if episode == "161":
            assert all(labeling.read_complete_latent(episode_dir / name / "latent.npz") for name in views)
        scheduled.append((episode, view))
        return video_batches(video, protocol, view, *args, **kwargs)

    monkeypatch.setattr(labeling, "_video_batches", check_before_prefetch)
    monkeypatch.setattr(labeling, "implementation_sha256", lambda: "new-code-revision")
    labeling.main(options + ["--metadata", str(full), "--prefetch-workers", "4"])
    assert scheduled == [("160", "global"), ("160", "wrist_right"), *(("161", view) for view in views)]
    for path, (payload, mtime) in preserved.items():
        assert path.read_bytes() == payload
        assert path.stat().st_mtime_ns == mtime
    report = capsys.readouterr().out
    assert "[skip] task_01/160/wrist_left: existing latent verified" in report
    assert "[skip] task_01/160/head: existing latent verified" in report
    assert "[relabel] task_01/160/wrist_right" in report
    assert "[label] wrote 6 latent.npz (2 skipped)" in report


def test_wetop_config_prepares_both_vla_stages(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from rynnvla.constants import RobotType
    from rynnvla.datasets.vla_datasets import LatentPretrainDataset, LeRobotV3Dataset
    from rynnvla.models.rynn_brain_vla.processing_rynn_brain_vla import RynnBrainVLAProcessor
    from rynnvla.utils.lerobot_v3 import MANIFEST_VIEW

    runner = load_script("train_lerobot_v3")
    root = tmp_path / "task"
    cameras = dict(zip(("video.top", "video.left", "video.right", "video.overlook"),
                       ("head", "left_wrist", "right_wrist", "front_third")))
    template = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    row = {"episode_index": 2, "length": 40, "tasks": ["Transfer liquid"]}
    for offset, (key, role) in enumerate(cameras.items(), 1):
        video = root / template.format(video_key=key, chunk_index=0, file_index=0)
        video.parent.mkdir(parents=True)
        write_video(video, n=45)
        row.update({f"videos/{key}/{field}": value for field, value in zip(
            ("chunk_index", "file_index", "from_timestamp", "to_timestamp"),
            (0, 0, offset / 30, (offset + 40) / 30),
        )})
        view = MANIFEST_VIEW[role]
        destination = tmp_path / "labels" / "latents" / "task" / "2" / view / "latent.npz"
        starts = np.arange(0, 34, 6, dtype=np.int32)
        label = np.broadcast_to(np.arange(6, dtype=np.float16)[:, None], (6, 608))
        labeling.write_npz(destination, label, np.stack((starts, starts + 6), axis=-1), {
            "num_latents": 6, "num_frames": 40, "fps": 30.0, "code_shape": [608],
            "protocol": {"gap": 6, "pair_stride": 6, "representation": "ktoken_zcam",
                         "start_frame": offset, "end_frame": offset + 40, "source": str(video)},
        })
    episodes = root / "meta/episodes/chunk-000"
    episodes.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([row]), episodes / "file-000.parquet")
    (root / "meta/info.json").write_text(json.dumps({
        "codebase_version": "v3.0", "fps": 30, "video_path": template,
        "features": {**{key: {"dtype": "video"} for key in cameras},
                     "state": {"dtype": "float32", "shape": [2]},
                     "action": {"dtype": "float32", "shape": [3]}},
    }))
    frame_dir = root / "data/chunk-000"
    frame_dir.mkdir(parents=True)
    # The physical row order and shard boundary must not change the action timeline.
    for index, frames in enumerate((range(39, 19, -1), range(19, -1, -1))):
        pq.write_table(pa.table({
            "episode_index": [2] * 20, "frame_index": list(frames),
            "state": pa.array([[frame + 0.5, frame + 0.75] for frame in frames], type=pa.list_(pa.float32(), 2)),
            "action": [[frame, frame + 10, frame + 20] for frame in frames],
        }), frame_dir / f"file-{index:03d}.parquet", row_group_size=10)
    config_path = tmp_path / "config.json"
    config = {
        "output_dir": "labels",
        "datasets": [{"name": "task", "path": "task", "episode_indices": [2]}],
        "dataset_defaults": {"cameras": cameras, "state_key": "state", "action_key": "action", "robot_type": "lift2"},
        "rynnlam": {"gap": 6, "pair_stride": 6, "representation": "ktoken_zcam"},
        "rynnvla": {
            "latent_action_dim": 608, "prepared_data_dir": "prepared",
            "stage1": {"latent_chunk": 6, "latent_step_seconds": 0.2,
                       "pretrained_model_path": "pretrained", "output_dir": "runs/latent"},
            "stage2": {"action_chunk_size": 12,
                       "pretrained_model_path": "exports/stage1-ema", "output_dir": "runs/actions"},
        },
    }
    config_path.write_text(json.dumps(config))
    options = ["--config", str(config_path)]
    for stage in ("stage1", "stage2"):
        assert runner.main([stage, *options, "--prepare-only"]) == 0
    work = tmp_path / "prepared"
    pointer, = work.glob("stage1-*.json")
    stage1_mixture = Path(json.loads(pointer.read_text())["mixture"])
    stage2_mixture, = work.glob("stage2-*.json")
    stage1_data, = json.loads(stage1_mixture.read_text())
    stage2_data, = json.loads(stage2_mixture.read_text())
    assert stage1_data.pop("data_type") == "LatentPretrainDataset"
    assert stage2_data.pop("data_type") == "LeRobotV3Dataset"
    common = {"use_delta_action": False, "num_view_slots": 6}
    latent = LatentPretrainDataset(**stage1_data, **common, latent_action_dim=608, action_chunk_size=30)[0]
    assert latent["latent_targets"].shape == (6, 6, 608)
    assert latent["slot_mask"].tolist() == [True, True, True, True, False, False]
    assert {view: int(image[0, 0, 0]) for view, image in latent["images"].items()} == pytest.approx(
        {"view0": 20, "view1": 40, "view2": 60, "view3": 80}, abs=4
    )
    action_chunk_size = config["rynnvla"]["stage2"]["action_chunk_size"]
    dataset = LeRobotV3Dataset(**stage2_data, **common, action_chunk_size=action_chunk_size, chunk_overlap_ratio=0.5)
    sample = dataset[1]
    assert sample["robot_type"] == RobotType.LIFT2
    assert sample["camera_slot_map"] == latent["camera_slot_map"] == {"view0": 0, "view1": 1, "view2": 2, "view3": 3}
    torch.testing.assert_close(sample["state"].native_vector.data, torch.tensor([[6.5, 6.75]]))
    torch.testing.assert_close(sample["action"].native_vector.data[0], torch.tensor([6., 16., 26.]))
    assert int(sample["images"]["view2"][0, 0, 0]) == pytest.approx(180, abs=4)
    schema = dataset.get_schema(num_workers=0)
    processor = object.__new__(RynnBrainVLAProcessor)
    processor.schema, processor.action_norm_type = schema, "min_max_sym"
    encoded, mask = processor._process_action(sample["action"], schema["action"]["lift2"])
    assert encoded.shape == mask.shape == (action_chunk_size, 81)
    assert mask.sum().item() == 3 * action_chunk_size
    torch.testing.assert_close(mask, processor.get_action_mask(RobotType.LIFT2, action_chunk_size))
    decoded = processor.post_process(encoded, RobotType.LIFT2)
    torch.testing.assert_close(decoded.native_vector.data, sample["action"].native_vector.data)

    calls = []
    monkeypatch.setattr(runner, "validate_initialization", lambda path: calls.append(path))
    monkeypatch.setattr(runner.launch, "main", lambda args: calls.append(args) or 0)
    (tmp_path / "pretrained").mkdir()
    assert runner.main(["stage1", *options, "--nproc-per-node", "2"]) == 0
    assert calls[0] == tmp_path / "pretrained"
    assert calls[1][calls[1].index("--output-dir") + 1] == str(tmp_path / "runs/latent")
    assert str(stage1_mixture) in calls[1]
    assert "2" in calls[1]
    assert "--action-chunk-size" not in calls[1]
    with pytest.raises(FileNotFoundError, match="stage2 pretrained model directory is missing.*exports/stage1-ema"):
        runner.main(["stage2", *options])
    ema = tmp_path / "exports/stage1-ema"
    ema.mkdir(parents=True)
    assert runner.main(["stage2", *options]) == 0
    assert calls[2] == ema
    assert calls[3][calls[3].index("--model-path") + 1] == str(ema)
    assert calls[3][calls[3].index("--output-dir") + 1] == str(tmp_path / "runs/actions")
    assert str(stage2_mixture) in calls[3]
    command, _ = runner.launch.build_command(calls[3])
    assert command[command.index("--action_chunk_size") + 1] == str(action_chunk_size)


class DummyReader:
    """Distinct constant RGB frames keep temporal checks exact and inexpensive."""

    def __init__(self, n):
        self.n = n
        self.num_frames = None
        self.fps = 30.0

    def frames(self, start=0, end=None):
        stop = self.n if end is None else min(end, self.n)
        count = 0
        for index in range(start, stop):
            count += 1
            yield np.full((3, 4, 3), index, dtype=np.uint8)
        self.num_frames = count
        if not count or (end is not None and count != end - start):
            raise ValueError("incomplete interval")


@pytest.fixture
def tiny_buckets(monkeypatch):
    monkeypatch.setitem(video_module.BUCKETS, "4x3", (3, 4))


@pytest.mark.parametrize(
    "n,expected_count", [(4, 0), (5, 1), (8, 1), (9, 2), (10, 2), (97, 24)]
)
def test_sparse_pair_lengths(n, expected_count, tiny_buckets):
    reader = DummyReader(n)
    iterator = pair_batches(reader, gap=4, pair_stride=4, batch_size=5)
    if expected_count == 0:
        with pytest.raises(ValueError, match="gap"):
            list(iterator)
        assert reader.num_frames == n
        return
    batches = list(iterator)
    pairs = np.concatenate([batch[1] for batch in batches])
    starts = np.arange(expected_count) * 4
    np.testing.assert_array_equal(pairs, np.column_stack((starts, starts + 4)))
    images = np.concatenate([batch[0] for batch in batches])
    np.testing.assert_allclose(images[:, :, 0, 0, 0], pairs / 255.0)
    assert reader.num_frames == n
    assert pairs[-1, 1] < n
    assert all(len(batch[0]) == 5 for batch in batches[:-1])
    assert len(batches[-1][0]) == (expected_count - 1) % 5 + 1


@pytest.mark.parametrize("end", [None, 97])
def test_batch_buffer_reuse_keeps_input_values_and_independent_indices(end, tiny_buckets):
    options = dict(gap=4, pair_stride=4, batch_size=10, end=end)
    expected = list(pair_batches(DummyReader(97), **options))
    assert [len(batch[0]) for batch in expected] == [10, 10, 4]
    assert not np.shares_memory(expected[0][0], expected[1][0])

    reusable_images = []
    saved_indices = []
    actual = pair_batches(DummyReader(97), **options, reuse_buffer=True)
    for (images, indices, bucket, hw), (expected_images, expected_indices, *geometry) in zip(
        actual, expected, strict=True,
    ):
        np.testing.assert_array_equal(images, expected_images)
        np.testing.assert_array_equal(indices, expected_indices)
        assert (images.dtype, indices.dtype) == (np.float32, np.int32)
        assert images.flags.c_contiguous
        assert [bucket, hw] == geometry
        reusable_images.append(images)
        saved_indices.append(indices)

    assert all(np.shares_memory(reusable_images[0], images) for images in reusable_images[1:])
    for indices, batch in zip(saved_indices, expected, strict=True):
        np.testing.assert_array_equal(indices, batch[1])


def test_prefetch_overlaps_with_consumer_without_overwriting_held_batch(tiny_buckets):
    second_ready = Event()
    worker_closed = Event()
    opts = dict(gap=4, pair_stride=4, batch_size=2, end=30)
    expected = list(pair_batches(DummyReader(30), **opts))

    def batches():
        try:
            for batch_index, batch in enumerate(pair_batches(
                    DummyReader(30), **opts, reuse_buffer=True, buffer_count=PrefetchedBatches.buffer_count)):
                assert current_thread() is not main_thread()
                if batch_index == 1:
                    second_ready.set()
                yield batch
        finally:
            worker_closed.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        with closing(PrefetchedBatches()) as prepared, closing(prepared.consume()) as stream:
            prepared.start(executor, batches())
            first = next(stream)
            assert second_ready.wait(timeout=3)
            np.testing.assert_array_equal(first[0], expected[0][0])
            second = next(stream)
            np.testing.assert_array_equal(second[0], expected[1][0])
            third = next(stream)
            np.testing.assert_array_equal(third[0], expected[2][0])
            assert np.shares_memory(first[0], third[0])
            assert not np.shares_memory(second[0], third[0])
        assert worker_closed.wait(timeout=3)


@pytest.mark.parametrize("gap", [4, 5])
def test_sparse_interval_indices_and_batch_boundaries(gap, tiny_buckets):
    reader = DummyReader(30)
    batches = list(
        pair_batches(reader, gap=gap, pair_stride=4, batch_size=2, start=3, end=21)
    )
    starts = np.arange(0, 18 - gap, 4)
    pairs = np.concatenate([batch[1] for batch in batches])
    np.testing.assert_array_equal(pairs, np.column_stack((starts, starts + gap)))
    images = np.concatenate([batch[0] for batch in batches])
    np.testing.assert_allclose(images[:, :, 0, 0, 0], (pairs + 3) / 255.0)
    assert [len(batch[0]) for batch in batches] == [2, 2]
    assert reader.num_frames == 18


@pytest.mark.parametrize("gap", [4, 5])
def test_sparse_tokens_match_dense_subset(gap, tiny_buckets):
    encoder = FakeEncoder()

    def encode(stride, batch_size):
        batches = list(
            pair_batches(
                DummyReader(23), gap=gap, pair_stride=stride, batch_size=batch_size
            )
        )
        return (
            np.concatenate([encoder(batch[0]).numpy() for batch in batches]),
            np.concatenate([batch[1] for batch in batches]),
        )

    dense_tokens, dense_pairs = encode(1, 3)
    sparse_tokens, sparse_pairs = encode(4, 2)
    np.testing.assert_array_equal(sparse_pairs, dense_pairs[::4])
    np.testing.assert_array_equal(sparse_tokens, dense_tokens[::4])
    assert np.all(
        sparse_tokens[0] > 0
    ), "The first label must encode (0, gap), not padding"


@pytest.mark.parametrize("gap,n", [(4, 10), (4, 97), (5, 10)])
def test_sparse_resizes_only_needed_endpoints(gap, n, tiny_buckets, monkeypatch):
    resized = []
    original = video_module.resize_crop

    def track(frame, target_hw):
        resized.append(int(frame[0, 0, 0]))
        return original(frame, target_hw)

    monkeypatch.setattr(video_module, "resize_crop", track)
    batches = list(pair_batches(DummyReader(n), gap=gap, pair_stride=4, batch_size=2))
    pairs = np.concatenate([batch[1] for batch in batches])
    expected = sorted(set(pairs.ravel().tolist()))
    assert sorted(resized) == expected


@pytest.mark.parametrize("start,end,expected", [(0, None, 10), (3, None, 7), (3, 9, 6)])
def test_frame_reader_reports_exact_interval_length(tmp_path, start, end, expected):
    video = tmp_path / "video.mp4"
    write_video(video, n=10)
    reader = FrameReader(video)
    frames = list(reader.frames(start=start, end=end))
    assert len(frames) == expected
    assert reader.num_frames == expected


@pytest.mark.parametrize("constant_rate", [True, False])
def test_sparse_mp4_window_matches_sequential_rgb(tmp_path, monkeypatch, constant_rate):
    video = tmp_path / "shared.mp4"
    timestamps = range(9, 39) if constant_rate else [i + 2 * (i >= 10) for i in range(30)]
    write_video(video, n=30, timestamps=timestamps)
    start, end = 12, 27
    expected = list(FrameReader(video).frames(start, end))

    first_indices = []
    seek_frames = video_module._seek_video_frames

    def track_first(*args):
        frames = seek_frames(*args)
        first = next(frames)
        first_indices.append(first[0])
        yield first
        yield from frames

    monkeypatch.setattr(video_module, "_seek_video_frames", track_first)
    reader = FrameReader(video)
    frames = list(reader.frames(start, end, rgb_filter=lambda index: index % 3 == 0, seek=True))
    assert reader.num_frames == len(frames) == end - start
    assert len(first_indices) == 1
    assert (first_indices[0] > 0) == constant_rate
    for index, frame in enumerate(frames):
        if index % 3:
            assert frame is None
        else:
            np.testing.assert_array_equal(frame, expected[index])

    gap, stride = 2, 3
    batches = list(pair_batches(FrameReader(video), gap=gap, pair_stride=stride,
                                start=start, end=end, batch_size=2, seek=True))
    indices = np.concatenate([batch[1] for batch in batches])
    pair_starts = np.arange(0, end - start - gap, stride)
    np.testing.assert_array_equal(indices, np.column_stack((pair_starts, pair_starts + gap)))
    images = np.concatenate([batch[0] for batch in batches])
    expected_pairs = np.asarray([[video_module.resize_crop(expected[index], (238, 322))
                                  for index in pair] for pair in indices], dtype=np.float32) / 255.0
    np.testing.assert_array_equal(images, expected_pairs)


@pytest.mark.parametrize("with_prefetch", [False, True])
def test_labeler_retries_entire_window_after_seek_error(tmp_path, monkeypatch, capsys, with_prefetch):
    video = tmp_path / "shared.mp4"
    write_video(video, n=30, timestamps=range(30))
    protocol = {"gap": 2, "pair_stride": 2, "bucket": "4x3",
                "start_frame": 12, "end_frame": 27, "representation": "ktoken"}
    encoder = FakeEncoder()
    expected = labeling._encode_video(encoder, video, protocol, view="head", batch_size=1, seek=False)
    seek_frames = video_module._seek_video_frames

    def fail_after_one_pair(*args):
        for index, frame in seek_frames(*args):
            if index == 15:
                raise video_module.VideoSeekError("fixture timestamp error")
            yield index, frame

    monkeypatch.setattr(video_module, "_seek_video_frames", fail_after_one_pair)
    if with_prefetch:
        reader, batches = labeling._video_batches(video, protocol, "head", 1, True,
                                                 buffer_count=PrefetchedBatches.buffer_count)
        with ThreadPoolExecutor(max_workers=1) as executor:
            with closing(PrefetchedBatches()) as prefetch:
                prefetch.start(executor, batches)
                latent, pairs, meta = labeling.label_video(
                    encoder, video, protocol, view="head", batch_size=1, prepared=(reader, prefetch),
                )
    else:
        latent, pairs, meta = labeling.label_video(encoder, video, protocol, view="head", batch_size=1)
    np.testing.assert_array_equal(latent, expected[0])
    np.testing.assert_array_equal(pairs, expected[1])
    assert meta == expected[2]
    assert "Retrying the window" in capsys.readouterr().out


@pytest.mark.parametrize("stride", [0, -1])
def test_pair_batches_reject_invalid_stride(stride):
    with pytest.raises(ValueError, match="stride"):
        list(pair_batches(DummyReader(10), gap=4, pair_stride=stride))


@pytest.mark.parametrize("option", ["--pair-stride", "--stride"])
def test_cli_rejects_zero_stride(tmp_path, capsys, option):
    with pytest.raises(SystemExit) as error:
        labeling.main(
            [
                "--video",
                str(tmp_path / "missing.mp4"),
                "--checkpoint",
                str(tmp_path / "missing.pt"),
                "--output-dir",
                str(tmp_path / "labels"),
                option,
                "0",
            ]
        )
    assert error.value.code == 2
    message = capsys.readouterr().err.lower()
    assert "stride" in message
    assert "unrecognized arguments" not in message
    assert not (tmp_path / "labels").exists()


