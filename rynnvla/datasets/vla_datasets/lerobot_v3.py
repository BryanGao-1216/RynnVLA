"""Read LeRobot v3 trajectories from shared Parquet shards and per-view video windows."""

from hashlib import sha256
import json
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import torch

from .base import BaseVLADataset, _robot_schema
from .latent_pretrain import _read_frame
from ...constants import RobotType, VIEW_NAME_TO_ROLE, VIEW_ROLE_TO_ID
from ...registry import DATASET_REGISTRY
from ...utils.lerobot_v3 import episode_rows, episode_views, read_info, vector_size
from ...utils.manifest_io import join_caption
from ...utils.robot import Position, RobotAction, RobotState


@DATASET_REGISTRY.register()
class LeRobotV3Dataset(BaseVLADataset):
    """Train on the native action/state columns without reordering their components."""

    def __init__(self, data_path, *, cameras, state_key, action_key, episode_indices=None,
                 robot_type="default", frame_cache=None, use_delta_action=False, target_fps=None, **kwargs):
        if use_delta_action or target_fps is not None:
            raise ValueError("Native LeRobot v3 vectors require use_delta_action=false and target_fps=null")
        if state_key == action_key or {state_key, action_key} & {"episode_index", "frame_index"}:
            raise ValueError("state_key and action_key must be distinct data vector columns")
        self.robot_type = RobotType(robot_type)
        self.root = Path(data_path)
        info = read_info(self.root, cameras)
        self._fps = float(info["fps"])
        self._widths = {key: vector_size(info, key, self.root) for key in (action_key, state_key)}
        self.state_key, self.action_key = state_key, action_key
        action_width = self._widths[action_key]
        self._column_slices = {action_key: slice(0, action_width), state_key: slice(action_width, None)}
        slots = {
            name: VIEW_ROLE_TO_ID[role] for name, role in VIEW_NAME_TO_ROLE.items()
            if role in cameras.values()
        }
        # Match Stage 1's camera token order as well as its role IDs.
        self.camera_slot_map = {f"view{slot}": slot for slot in slots.values()}
        self._episodes = []
        for row, context in episode_rows(self.root, cameras, episode_indices):
            instruction = join_caption(row["tasks"], max_chars=512)
            if not instruction or row["length"] < 1:
                raise ValueError(f"{context}: missing task instruction or positive episode length")
            views = episode_views(self.root, row, cameras, info["video_path"], self._fps, context)
            self._episodes.append((row["episode_index"], row["length"], instruction,
                                   {f"view{slots[name]}": view for name, view in views.items()}))
        self._episodes.sort(key=lambda episode: episode[0])
        if not self._episodes:
            raise ValueError(f"{self.root}: no selected LeRobot episodes")

        self._parquet = sorted((self.root / "data").glob("chunk-*/*.parquet"))
        if not self._parquet:
            raise FileNotFoundError(f"No frame Parquet shards under {self.root / 'data'}")
        self._offsets = np.cumsum([0, *self.episode_lengths], dtype=np.int64)
        self._frame_shape = (int(self._offsets[-1]), sum(self._widths.values()))
        self._frame_cache = Path(frame_cache) if frame_cache is not None else None
        if self._frame_cache is not None:
            if self._frame_cache.name != self._cache_name():
                raise ValueError(f"{self.root}: prepared Stage 2 data no longer matches the Parquet files or selection")
            self._check_frame_cache(self._frame_cache)
        self._frames = None
        super().__init__(data_path=data_path, use_delta_action=False, target_fps=None, **kwargs)

    def __getstate__(self):
        return {**self.__dict__, "_frames": None}

    @property
    def episode_lengths(self):
        return [length for _, length, _, _ in self._episodes]

    def _index_lengths(self):
        return self._chunk_start_counts()

    def get_fps(self, episode_index):
        return self._fps

    def get_robot_type(self, episode_index):
        return self.robot_type

    def _cache_name(self):
        files = []
        for path in self._parquet:
            stat = path.stat()
            files.append((str(path), stat.st_size, stat.st_mtime_ns))
        signature = [files, [(i, n) for i, n, _, _ in self._episodes], list(self._widths.items())]
        digest = sha256(json.dumps(signature).encode()).hexdigest()[:20]
        return f"frames-{digest}.npy"

    def _check_frame_cache(self, path):
        frames = np.load(path, mmap_mode="r", allow_pickle=False)
        if frames.shape != self._frame_shape or frames.dtype != np.float32:
            raise ValueError(f"{path}: expected float32 frame arrays of shape {self._frame_shape}")

    def prepare_frames(self, directory):
        """Scan the chosen columns once and publish a memory-mappable training array."""
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / self._cache_name()
        if not target.exists():
            # mmap writes need a local filesystem even when the training root is on FUSE.
            with TemporaryDirectory() as local:
                local_path = Path(local) / target.name
                frames = np.lib.format.open_memmap(local_path, mode="w+", dtype=np.float32, shape=self._frame_shape)
                self._collect_frames(frames)
                frames.flush()
                del frames
                with NamedTemporaryFile(dir=directory, delete=False) as temporary:
                    staging = Path(temporary.name)
                try:
                    shutil.copyfile(local_path, staging)
                    staging.replace(target)
                finally:
                    staging.unlink(missing_ok=True)
        self._check_frame_cache(target)
        return target

    def _collect_frames(self, frames):
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        episode_ids = np.array([episode for episode, _, _, _ in self._episodes], dtype=np.int64)
        lengths = np.diff(self._offsets)
        written = np.zeros(len(frames), dtype=bool)
        columns = ["episode_index", "frame_index", *self._widths]
        for path in self._parquet:
            with pq.ParquetFile(path) as parquet:
                missing = set(columns) - set(parquet.schema_arrow.names)
                if missing:
                    raise ValueError(f"{path}: missing Parquet columns {sorted(missing)}")
                for batch in parquet.iter_batches(batch_size=32768, columns=columns):
                    self._copy_frame_batch(batch, pc, episode_ids, lengths, frames, written, path)
        if not written.all():
            raise ValueError(f"{self.root}: {int((~written).sum())} selected episode frames are missing from Parquet")

    def _copy_frame_batch(self, batch, pc, episode_ids, lengths, frames, written, path):
        episodes, indices = (batch[name] for name in ("episode_index", "frame_index"))
        if episodes.null_count or indices.null_count:
            raise ValueError(f"{path}: null episode_index or frame_index")
        episode_column = episodes.to_numpy()
        slots = np.searchsorted(episode_ids, episode_column).clip(max=len(episode_ids) - 1)
        selected = episode_ids[slots] == episode_column
        slots, indices = slots[selected], indices.to_numpy()[selected]
        if not len(slots):
            return
        if np.any(indices < 0) or np.any(indices >= lengths[slots]):
            raise ValueError(f"{path}: frame_index out of range for a selected episode")
        rows = self._offsets[slots] + indices
        if len(np.unique(rows)) != len(rows) or written[rows].any():
            raise ValueError(f"{path}: duplicate (episode_index, frame_index) in selected data")
        for key, width in self._widths.items():
            column = batch[key].filter(selected)
            if column.null_count or not pc.all(pc.equal(pc.list_value_length(column), width)).as_py():
                raise ValueError(f"{path}: {key!r} contains null or non-{width}-dimensional rows")
            values = np.asarray(column.flatten().to_numpy(zero_copy_only=False), dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: {key!r} contains non-finite values")
            frames[rows, self._column_slices[key]] = values.reshape(len(rows), width)
        written[rows] = True

    def _frame_values(self, episode_index, frame_index, key):
        if self._frames is None:
            if self._frame_cache is None:
                raise ValueError("Prepare the Stage 2 frame cache before loading samples")
            self._frames = np.load(self._frame_cache, mmap_mode="r", allow_pickle=False)
        start, end = self._offsets[episode_index:episode_index + 2]
        rows = self._frames[start:end, self._column_slices[key]]
        return torch.from_numpy(rows[frame_index].copy())

    def load_action(self, episode_index, frame_index):
        values = self._frame_values(episode_index, frame_index, self.action_key)
        return RobotAction(native_vector=Position(values, allow_relative=False))

    def load_state(self, episode_index, frame_index):
        values = self._frame_values(episode_index, frame_index, self.state_key)
        return RobotState(native_vector=Position(values, allow_relative=False))

    def load_images(self, episode_index, frame_index):
        _, length, _, views = self._episodes[episode_index]
        indices = self._indices(frame_index, length)
        return {
            name: torch.stack([
                _read_frame(view["video_path"], view["start_frame"] + index)
                for index in indices
            ])
            for name, view in views.items()
        }

    def load_instruction(self, episode_index, frame_index):
        _, length, instruction, _ = self._episodes[episode_index]
        return [instruction] * len(self._indices(frame_index, length))

    @staticmethod
    def _indices(frame_index, length):
        if isinstance(frame_index, slice):
            return range(*frame_index.indices(length))
        indices = [frame_index] if isinstance(frame_index, int) else frame_index
        if any(index < 0 or index >= length for index in indices):
            raise IndexError(f"Frame outside episode length {length}: {indices}")
        return indices

    def load_episode(self, episode_index, action_only):
        full = slice(None)
        sample = {"action": self.load_action(episode_index, full), "state": self.load_state(episode_index, full)}
        if not action_only:
            sample.update(text=self.load_instruction(episode_index, full), images=self.load_images(episode_index, full))
        return sample

    def _get_episode_schemas(self, episode_index):
        # Native actions are used as-is; counting each recorded row once is enough for statistics.
        episode = self.load_episode(episode_index, action_only=True)
        return self.robot_type, _robot_schema(episode["action"]), _robot_schema(episode["state"])

    def _schema_cache_key(self):
        return {
            **super()._schema_cache_key(),
            "robot_type": self.robot_type.value,
            "frame_cache": str(self._frame_cache),
        }
