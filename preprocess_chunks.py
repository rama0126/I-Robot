#!/usr/bin/env python3
"""
MHR feature를 고정 윈도우로 잘라 memmap DB로 저장.

구조:
    out-dir/
      labels.npy              (N,) int8  - 모든 키에서 공유
      meta.csv                chunk 메타데이터 - 모든 키에서 공유
      {feature_key}/
        chunks.npy            (N, window_size, F) float32
        info.json             {feature_key, window_size, F, N, original_shape}

Usage:
    # 단일 키
    python preprocess_chunks.py --csv splits/all_train.csv --out-dir chunk_db/train \
        --feature-key pred_joint_coords

    # 복수 키
    python preprocess_chunks.py --csv splits/all_train.csv --out-dir chunk_db/train \
        --feature-key pred_joint_coords body_pose_params pred_keypoints_3d

    # 사용 가능한 모든 키 자동 추출
    python preprocess_chunks.py --csv splits/all_train.csv --out-dir chunk_db/train \
        --feature-key all

    # 파일에 있는 키 목록만 확인
    python preprocess_chunks.py --csv splits/all_train.csv --list-keys
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import get_feature_array, load_mhr_record

# 너무 커서 실용적이지 않은 키 (flatten 시 수만 차원)
_SKIP_KEYS = {"pred_vertices", "frame_index"}

# 기본으로 추출할 키 목록 (--feature-key all 일 때)
_DEFAULT_KEYS = [
    "pred_joint_coords",
    "pred_keypoints_3d",
    "pred_keypoints_2d",
    "body_pose_params",
    "hand_pose_params",
    "pred_pose_raw",
    "shape_params",
    "expr_params",
]


def detect_temporal_keys(rec: dict) -> list[str]:
    """첫 번째 dim이 T인 array 키만 반환."""
    keys = []
    T = None
    for k, v in rec.items():
        if k in _SKIP_KEYS:
            continue
        try:
            a = np.asarray(v)
            if a.ndim < 1:
                continue
            if T is None:
                T = a.shape[0]
            if a.shape[0] == T and a.ndim >= 2:
                keys.append(k)
        except Exception:
            continue
    return keys


def build_meta(df: pd.DataFrame, window_size: int) -> list[dict]:
    """CSV에서 청크 메타데이터 생성 (키에 독립적)."""
    stride = window_size
    rows = []
    for _, row in df.iterrows():
        length = int(row["frame_length"])
        starts = list(range(0, max(length - window_size + 1, 1), stride))
        for s in starts:
            rows.append({
                "file_path":  str(row["file_path"]),
                "label":      int(row["label_is_human"]),
                "start":      s,
                "source":     str(row.get("source", "")),
                "class_name": str(row.get("class_name", "")),
                "action":     str(row.get("action", "")),
                "robot_model":str(row.get("model", "")),
            })
    return rows


def build_key_db(
    meta_rows: list[dict],
    feature_key: str,
    window_size: int,
    key_dir: Path,
    force: bool = False,
) -> bool:
    """단일 feature_key에 대한 chunks.npy를 생성. 이미 있으면 건너뜀."""
    info_path = key_dir / "info.json"
    if not force and info_path.exists():
        info = json.loads(info_path.read_text())
        print(f"  [skip] '{feature_key}' already built  "
              f"(N={info['N']}, F={info['F']}, W={info['window_size']})")
        return False

    key_dir.mkdir(parents=True, exist_ok=True)
    W = window_size
    N = len(meta_rows)

    # feature dim 추론
    first_path = meta_rows[0]["file_path"]
    try:
        first_rec  = load_mhr_record(first_path)
        first_feat = get_feature_array(first_rec, feature_key)
    except KeyError as e:
        print(f"  [skip] {e}")
        return False

    F              = first_feat.shape[1]
    original_shape = list(np.asarray(first_rec[feature_key]).shape[1:])  # e.g. [127, 3]

    print(f"  '{feature_key}'  original_shape=T×{original_shape}  →  flat F={F}  N={N}")

    chunks_mm = np.memmap(key_dir / "chunks.npy", dtype=np.float32, mode="w+", shape=(N, W, F))

    prev_path = None
    feat      = None
    for i, meta in enumerate(meta_rows):
        path = meta["file_path"]
        if path != prev_path:
            rec       = load_mhr_record(path)
            feat      = get_feature_array(rec, feature_key)
            prev_path = path

        s    = meta["start"]
        clip = feat[s : s + W]
        if clip.shape[0] < W:
            pad  = np.repeat(clip[-1:], W - clip.shape[0], axis=0)
            clip = np.concatenate([clip, pad], axis=0)

        chunks_mm[i] = clip
        if (i + 1) % 10000 == 0:
            print(f"    {i+1:,} / {N:,}")

    chunks_mm.flush()

    info = {
        "feature_key":    feature_key,
        "window_size":    W,
        "F":              F,
        "N":              N,
        "original_shape": original_shape,
    }
    info_path.write_text(json.dumps(info, indent=2))
    sz_gb = (key_dir / "chunks.npy").stat().st_size / 1e9
    print(f"    saved → {key_dir}  ({sz_gb:.2f} GB)")
    return True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",          required=True,  help="split CSV (all_train.csv / all_val.csv)")
    p.add_argument("--out-dir",      default=None,   help="출력 루트 (e.g. chunk_db/train)")
    p.add_argument("--feature-key",  nargs="+",      default=["pred_joint_coords", "pred_keypoints_3d", "pred_pose_raw"],
                   help="추출할 MHR 키. 'all'이면 자동 감지된 모든 키")
    p.add_argument("--window-size",  type=int, default=32)
    p.add_argument("--force",        action="store_true", help="이미 있어도 덮어쓰기")
    p.add_argument("--list-keys",    action="store_true", help="사용 가능한 키 목록만 출력하고 종료")
    return p.parse_args()


def main():
    args = parse_args()
    df   = pd.read_csv(args.csv)
    first_rec = load_mhr_record(str(df.iloc[0]["file_path"]))

    if args.list_keys:
        print(f"\nAvailable temporal keys in: {df.iloc[0]['file_path']}\n")
        for k, v in first_rec.items():
            if k in _SKIP_KEYS:
                continue
            try:
                a = np.asarray(v)
                flat_f = int(np.prod(a.shape[1:])) if a.ndim >= 2 else None
                print(f"  {k:40s} shape={str(a.shape):25s} flat_F={flat_f}")
            except Exception:
                pass
        return

    if args.out_dir is None:
        print("--out-dir 를 지정해주세요.")
        return

    out_dir = Path(args.out_dir)

    # 추출 키 결정
    if args.feature_key == ["all"]:
        feature_keys = detect_temporal_keys(first_rec)
        # _DEFAULT_KEYS 우선 순서 유지
        ordered = [k for k in _DEFAULT_KEYS if k in feature_keys]
        rest    = [k for k in feature_keys  if k not in ordered]
        feature_keys = ordered + rest
        print(f"Auto-detected {len(feature_keys)} keys: {feature_keys}")
    else:
        feature_keys = args.feature_key

    W = args.window_size

    # ── 메타/레이블 빌드 (공유, 한 번만) ─────────────────────────────────────
    meta_path   = out_dir / "meta.csv"
    labels_path = out_dir / "labels.npy"

    if not args.force and meta_path.exists():
        print(f"[meta] loading existing meta.csv  ({meta_path})")
        meta_rows = pd.read_csv(meta_path).to_dict("records")
    else:
        print(f"[meta] building chunk metadata  (window={W}, stride={W})")
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_rows = build_meta(df, W)
        pd.DataFrame(meta_rows).to_csv(meta_path, index=False)
        np.save(labels_path, np.array([r["label"] for r in meta_rows], dtype=np.int8))
        print(f"  total chunks: {len(meta_rows):,}")

    # ── 키별 chunks.npy 빌드 ──────────────────────────────────────────────────
    print(f"\n[keys] building {len(feature_keys)} feature key(s)...")
    for key in feature_keys:
        print(f"\n--- {key} ---")
        build_key_db(meta_rows, key, W, out_dir / key, force=args.force)

    print(f"\n=== Done. DB root: {out_dir} ===")
    print("Key directories:")
    for key in feature_keys:
        info_p = out_dir / key / "info.json"
        if info_p.exists():
            info = json.loads(info_p.read_text())
            print(f"  {key:40s} F={info['F']:5d}  N={info['N']:,}")


if __name__ == "__main__":
    main()
