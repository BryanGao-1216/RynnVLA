"""RynnLAM labeling primitives: frame pairing, sparse intervals and identifier safety.

These cover the shared machinery ``scripts/label_latent.py`` is built on --
``rynnlam.video.pair_batches`` / ``FrameReader`` / ``resolve_source`` and the labeler's own
``safe_name`` path-traversal guard -- not the CLI's end-to-end behaviour, which
``scripts/smoke.sh`` exercises against the bundled sample.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from _scripts import load_script

labeling = load_script("label_latent")
import rynnlam.video as video_module
from rynnlam.video import FrameReader, pair_batches


def write_video(path, n=9):
    import av

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = "yuv420p"
        for i in range(n):
            image = np.full((48, 64, 3), i * 20, dtype=np.uint8)
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(image, format="rgb24")
            ):
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


def test_lerobot_v3_shared_video_windows_and_label_reuse(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    adapter = load_script("label_lerobot_v3")
    monkeypatch.setitem(sys.modules, "label_latent", labeling)

    class VectorEncoder(FakeEncoder):
        calls = 0
        device = torch.device("cpu")

        def __call__(self, images, representation):
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

    adapter.main(args)
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

    config["rynnlam"]["gap"] = 1
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="existing latent differs in gap"):
        adapter.main(args)
    adapter.main(args + ["--overwrite"])
    assert VectorEncoder.calls == 8


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


