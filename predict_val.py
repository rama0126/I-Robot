#!/usr/bin/env python3
"""
Validation set prediction CSV generator.

Usage:
    python predict_val.py --run-dir outputs_binary_cls/irobot_pool_conv_run
    python predict_val.py --run-dir outputs_binary_cls/irobot_pool_conv_run --out predictions_val.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import PrebuiltChunkDataset
from model.init_model import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir",   required=True, help="model run directory (contains checkpoint_best.pt)")
    p.add_argument("--ckpt",      default="checkpoint_best.pt")
    p.add_argument("--out",       default=None,  help="output CSV path (default: <run-dir>/predictions_val.csv)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / args.ckpt
    out_path  = Path(args.out) if args.out else run_dir / "predictions_val.csv"

    # ── load checkpoint ───────────────────────────────────────────────────────
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg  = ckpt["args"]
    print(f"  model={cfg['model']}  epoch={ckpt['epoch']}  val_metrics={ckpt.get('val_metrics')}")

    # ── dataset ───────────────────────────────────────────────────────────────
    val_db  = cfg["val_db_dir"]
    feat_key = cfg["feature_key"]
    print(f"Loading val DB: {val_db}  feature={feat_key}")

    ds = PrebuiltChunkDataset(val_db, feature_key=feat_key, return_index=True)
    meta = pd.read_csv(Path(val_db) / "meta.csv")
    assert len(meta) == len(ds), f"meta rows ({len(meta)}) != dataset size ({len(ds)})"

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # ── build model ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(
        model_name=cfg["model"],
        input_dim=ckpt["input_dim"],
        hidden_dim=cfg.get("hidden_dim", 256),
        dropout=cfg.get("dropout", 0.2),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

    # ── inference ─────────────────────────────────────────────────────────────
    all_probs  = []
    all_preds  = []
    all_labels = []
    all_idxs   = []

    print("Running inference...")
    with torch.no_grad():
        for x, y, idx in loader:
            x = x.to(device)
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            prob = torch.sigmoid(logits).cpu().numpy().ravel()
            all_probs.append(prob)
            all_preds.append((prob >= args.threshold).astype(int))
            all_labels.append(y.numpy().ravel().astype(int))
            all_idxs.append(idx.numpy().ravel())

    probs  = np.concatenate(all_probs)
    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    idxs   = np.concatenate(all_idxs)

    # ── assemble CSV ──────────────────────────────────────────────────────────
    # Training used label_is_human (0=robot, 1=human).
    # Flip so that robot=1 (positive class), human=0.
    df = meta.iloc[idxs].reset_index(drop=True).copy()
    df["label"]     = 1 - labels          # robot→1, human→0
    df["prob"]      = 1.0 - probs         # prob of being robot
    df["pred"]      = 1 - preds           # 1=predicted robot, 0=predicted human
    df["correct"]   = (df["pred"] == df["label"]).astype(int)

    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows → {out_path}")
    print("  (label/pred/prob flipped: robot=1 positive class, human=0)")

    # ── quick summary ─────────────────────────────────────────────────────────
    acc  = df["correct"].mean()
    tp   = int(((df["pred"] == 1) & (df["label"] == 1)).sum())
    tn   = int(((df["pred"] == 0) & (df["label"] == 0)).sum())
    fp   = int(((df["pred"] == 1) & (df["label"] == 0)).sum())
    fn   = int(((df["pred"] == 0) & (df["label"] == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-8)
    print(f"  ACC={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
    print(f"  TP={tp}(robot)  TN={tn}(human)  FP={fp}  FN={fn}")


if __name__ == "__main__":
    main()
