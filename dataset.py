from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def resolve_split_csv(path_or_none: str | None, name: str) -> str:
    if path_or_none:
        return path_or_none

    candidates = [
        Path("/workspace/irobot/training/splits") / name,
        Path("/workspace/irobot/extraction/splits") / name,
        Path("/workspace/irobot/splits") / name,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return str(candidates[0])


def load_mhr_record(path: str) -> dict[str, Any]:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        if arr.shape == ():
            rec = arr.item()
        else:
            rec = arr[0]
        if isinstance(rec, dict):
            return rec
    raise ValueError(f"Unexpected MHR format: {path}")


def get_feature_array(rec: dict[str, Any], feature_key: str) -> np.ndarray:
    if feature_key not in rec:
        raise KeyError(f"Feature key '{feature_key}' not found. Available keys: {list(rec.keys())}")
    x = np.asarray(rec[feature_key], dtype=np.float32)
    if x.ndim < 2:
        raise ValueError(f"Feature '{feature_key}' must have temporal dimension. Got shape={x.shape}")
    t = x.shape[0]
    return x.reshape(t, -1)


class LRURecordCache:
    def __init__(self, max_size: int = 64):
        self.max_size = max(1, max_size)
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def get(self, path: str) -> dict[str, Any]:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        rec = load_mhr_record(path)
        self._cache[path] = rec
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
        return rec


def build_chunk_starts(length: int, window_size: int, stride: int, cover_tail: bool = True) -> list[int]:
    if length <= 0:
        return []
    if length <= window_size:
        return [0]
    starts = list(range(0, length - window_size + 1, stride))
    if cover_tail:
        tail = length - window_size
        if starts[-1] != tail:
            starts.append(tail)
    return starts


@dataclass
class SampleIndex:
    file_path: str
    label: int
    start: int
    row_id: int
    source: str
    class_name: str
    model: str
    action: str
    filename: str


class MotionChunkDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        feature_key: str = "body_pose_params",
        window_size: int = 32,
        stride: int = 16,
        cache_size: int = 64,
        max_chunks: int | None = None,
        seed: int = 42,
        return_index: bool = False,
    ):
        self.df = pd.read_csv(csv_path)
        self.feature_key = feature_key
        self.window_size = window_size
        self.stride = stride
        self.cache = LRURecordCache(cache_size)
        self.return_index = return_index

        required = {"file_path", "label_is_human", "frame_length"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

        self.samples: list[SampleIndex] = []
        for row_id, row in self.df.reset_index(drop=True).iterrows():
            path = str(row["file_path"])
            label = int(row["label_is_human"])
            length = int(row["frame_length"])
            starts = build_chunk_starts(length, window_size, stride, cover_tail=True)
            for s in starts:
                self.samples.append(
                    SampleIndex(
                        file_path=path,
                        label=label,
                        start=s,
                        row_id=int(row_id),
                        source=str(row.get("source", "unknown")),
                        class_name=str(row.get("class_name", "unknown")),
                        model=str(row.get("model", "unknown")),
                        action=str(row.get("action", "unknown")),
                        filename=str(row.get("filename", Path(path).name)),
                    )
                )

        if not self.samples:
            raise RuntimeError(f"No chunks built from {csv_path}")

        if max_chunks is not None and max_chunks < len(self.samples):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(self.samples), size=max_chunks, replace=False)
            self.samples = [self.samples[i] for i in idx]

        self._input_dim = self._infer_input_dim()

    def _infer_input_dim(self) -> int:
        sample = self.samples[0]
        rec = self.cache.get(sample.file_path)
        x = get_feature_array(rec, self.feature_key)
        return int(x.shape[1])

    @property
    def input_dim(self) -> int:
        return self._input_dim

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        rec = self.cache.get(sample.file_path)
        x = get_feature_array(rec, self.feature_key)  # [T, F]

        start = sample.start
        end = min(start + self.window_size, x.shape[0])
        clip = x[start:end]

        if clip.shape[0] < self.window_size:
            pad_len = self.window_size - clip.shape[0]
            if clip.shape[0] == 0:
                pad = np.zeros((pad_len, x.shape[1]), dtype=np.float32)
            else:
                pad = np.repeat(clip[-1:, :], pad_len, axis=0)
            clip = np.concatenate([clip, pad], axis=0)

        y = np.float32(sample.label)
        x_tensor = torch.from_numpy(clip)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        if self.return_index:
            return x_tensor, y_tensor, torch.tensor(idx, dtype=torch.long)
        return x_tensor, y_tensor


class PrebuiltChunkDataset(Dataset):
    """
    preprocess_chunks.py가 만든 memmap DB를 읽는 Dataset.

    구조:
        db_root/
          labels.npy          ← 모든 키에서 공유
          meta.csv            ← 모든 키에서 공유
          {feature_key}/
            chunks.npy        ← (N, T, F) float32
            info.json

    Args:
        db_root:     chunk_db/train  같은 split 루트
        feature_key: pred_joint_coords 등 MHR 키
    """

    def __init__(self, db_root: str, feature_key: str = "pred_joint_coords",
                 return_index: bool = False):
        db_root  = Path(db_root)
        key_dir  = db_root / feature_key
        info     = json.loads((key_dir / "info.json").read_text())
        N, W, F  = info["N"], info["window_size"], info["F"]

        self._chunks    = np.memmap(key_dir / "chunks.npy", dtype=np.float32, mode="r", shape=(N, W, F))
        self._labels    = np.load(db_root / "labels.npy")
        self._input_dim = F
        self.feature_key   = feature_key
        self.original_shape = info.get("original_shape", [F])  # e.g. [127, 3]
        self.return_index  = return_index

    @classmethod
    def available_keys(cls, db_root: str) -> list[str]:
        """db_root 아래에 이미 빌드된 feature key 목록 반환."""
        root = Path(db_root)
        return sorted(
            d.name for d in root.iterdir()
            if d.is_dir() and (d / "info.json").exists()
        )

    @property
    def input_dim(self) -> int:
        return self._input_dim

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self._chunks[idx].copy())   # (T, F)
        y = torch.tensor(float(self._labels[idx]), dtype=torch.float32)
        if self.return_index:
            return x, y, torch.tensor(idx, dtype=torch.long)
        return x, y
