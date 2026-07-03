#!/usr/bin/env python3
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Limit to single GPU for simplicity

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import MotionChunkDataset, PrebuiltChunkDataset, resolve_split_csv
from loss import build_loss, compute_pos_weight
from model.init_model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return float("nan")
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float, loss: float) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).to(torch.int64)
    y = labels.to(torch.int64)

    tp = int(((preds == 1) & (y == 1)).sum().item())
    tn = int(((preds == 0) & (y == 0)).sum().item())
    fp = int(((preds == 1) & (y == 0)).sum().item())
    fn = int(((preds == 0) & (y == 1)).sum().item())

    total = max(tp + tn + fp + fn, 1)
    acc = (tp + tn) / total

    # Macro-averaged precision / recall / F1 (average over both classes)
    prec1 = tp / max(tp + fp, 1);  rec1 = tp / max(tp + fn, 1)
    prec0 = tn / max(tn + fn, 1);  rec0 = tn / max(tn + fp, 1)
    f1_1  = 2 * prec1 * rec1 / max(prec1 + rec1, 1e-8)
    f1_0  = 2 * prec0 * rec0 / max(prec0 + rec0, 1e-8)
    precision = (prec1 + prec0) / 2
    recall    = (rec1  + rec0)  / 2
    f1        = (f1_1  + f1_0)  / 2

    y_np = y.detach().cpu().numpy()
    p_np = probs.detach().cpu().numpy()
    auc = safe_roc_auc(y_np, p_np)

    return {
        "loss": float(loss),
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def run_one_epoch(model, loader, optimizer, criterion, device, threshold: float, train_mode: bool, clip_grad: float):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_count = 0
    all_logits = []
    all_labels = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            logits = model(x)
            loss = criterion(logits, y)
            if train_mode:
                loss.backward()
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()

        b = x.size(0)
        total_loss += float(loss.item()) * b
        total_count += b
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    avg_loss = total_loss / max(total_count, 1)
    return classification_metrics(logits, labels, threshold=threshold, loss=avg_loss)


def save_curves(history_df: pd.DataFrame, out_dir: Path) -> None:
    plot_code_path = out_dir / "plot_curves.py"
    script = r'''import pandas as pd
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("matplotlib not installed. Please run: pip install matplotlib")
    raise SystemExit(0)

hist = pd.read_csv(Path(__file__).resolve().parent / "train_log.csv")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(hist["epoch"], hist["train_loss"], label="train")
axes[0, 0].plot(hist["epoch"], hist["val_loss"], label="val")
axes[0, 0].set_title("Loss")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].legend()

axes[0, 1].plot(hist["epoch"], hist["train_acc"], label="train")
axes[0, 1].plot(hist["epoch"], hist["val_acc"], label="val")
axes[0, 1].set_title("Accuracy")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("ACC")
axes[0, 1].legend()

axes[1, 0].plot(hist["epoch"], hist["train_f1"], label="train")
axes[1, 0].plot(hist["epoch"], hist["val_f1"], label="val")
axes[1, 0].set_title("F1 Score")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("F1")
axes[1, 0].legend()

axes[1, 1].plot(hist["epoch"], hist["train_auc"], label="train")
axes[1, 1].plot(hist["epoch"], hist["val_auc"], label="val")
axes[1, 1].set_title("AUC")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("AUC")
axes[1, 1].legend()

plt.tight_layout()
out_png = Path(__file__).resolve().parent / "training_curves.png"
plt.savefig(out_png, dpi=220, bbox_inches="tight")
print("Saved:", out_png)
'''
    plot_code_path.write_text(script, encoding="utf-8")

    # Save CSV history always
    hist_csv = out_dir / "train_log.csv"
    history_df.to_csv(hist_csv, index=False)
    print(f"Saved training history: {hist_csv}")

    # Try generating curves now if matplotlib is available
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed in current env. "
              f"Run later: python {plot_code_path} to draw loss/accuracy curves.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history_df["epoch"], history_df["train_loss"], label="train")
    axes[0, 0].plot(history_df["epoch"], history_df["val_loss"], label="val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(history_df["epoch"], history_df["train_acc"], label="train")
    axes[0, 1].plot(history_df["epoch"], history_df["val_acc"], label="val")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("ACC")
    axes[0, 1].legend()

    axes[1, 0].plot(history_df["epoch"], history_df["train_f1"], label="train")
    axes[1, 0].plot(history_df["epoch"], history_df["val_f1"], label="val")
    axes[1, 0].set_title("F1 Score")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("F1")
    axes[1, 0].legend()

    axes[1, 1].plot(history_df["epoch"], history_df["train_auc"], label="train")
    axes[1, 1].plot(history_df["epoch"], history_df["val_auc"], label="val")
    axes[1, 1].set_title("AUC")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("AUC")
    axes[1, 1].legend()

    plt.tight_layout()
    out_png = out_dir / "training_curves.png"
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    print(f"Saved curves: {out_png}")


def parse_args():
    p = argparse.ArgumentParser(description="Train the iRobot binary classifier (human vs humanoid) on MHR chunks")
    p.add_argument("--train-csv", type=str, default=None, help="Optional custom train CSV path")
    p.add_argument("--val-csv", type=str, default=None, help="Optional custom val CSV path")
    p.add_argument(
        "--feature-key",
        type=str,
        default="pred_joint_coords",
        help=(
            "MHR key to use as temporal feature. "
            "Examples: body_pose_params, pred_joint_coords, pred_keypoints_3d, pred_pose_raw"
        ),
    )
    p.add_argument(
        "--model",
        type=str,
        default="irobot_pool_conv",
        help="iRobot variant: irobot_raw | irobot_dual | irobot_pool_conv | irobot_pool_conv_diff",
    )

    p.add_argument("--window-size", type=int, default=32)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cache-size", type=int, default=64)
    p.add_argument("--max-train-chunks", type=int, default=None)
    p.add_argument("--max-val-chunks", type=int, default=None)

    p.add_argument("--loss-name", type=str, default="bce", choices=["bce", "focal"])
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=0.5)

    p.add_argument("--hidden-dim", type=int, default=256)

    p.add_argument("--train-db-dir", type=str, default="/workspace/irobot/training/chunk_db/train")
    p.add_argument("--val-db-dir", type=str, default="/workspace/irobot/training/chunk_db/val")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--run-analysis", action="store_true", help="Run post-hoc analysis/plots after training")
    p.add_argument("--analysis-out-dir", type=str, default=None, help="Optional analysis output directory")
    return p.parse_args()


def main():
    args = parse_args()
    args.train_csv = resolve_split_csv(args.train_csv, "all_train.csv")
    args.val_csv = resolve_split_csv(args.val_csv, "all_val.csv")

    if args.save_dir is None:
        args.save_dir = f"/workspace/irobot/training/outputs_binary_cls/{args.model}_run"

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    # iRobot consumes flat (B, T, F) chunks.
    if args.train_db_dir:
        train_ds = PrebuiltChunkDataset(args.train_db_dir, feature_key=args.feature_key)
        val_ds   = PrebuiltChunkDataset(args.val_db_dir or args.train_db_dir, feature_key=args.feature_key)
    else:
        train_ds = MotionChunkDataset(
            csv_path=args.train_csv,
            feature_key=args.feature_key,
            window_size=args.window_size,
            stride=args.stride,
            cache_size=args.cache_size,
            max_chunks=args.max_train_chunks,
            seed=args.seed,
        )
        val_ds = MotionChunkDataset(
            csv_path=args.val_csv,
            feature_key=args.feature_key,
            window_size=args.window_size,
            stride=args.stride,
            cache_size=args.cache_size,
            max_chunks=args.max_val_chunks,
            seed=args.seed,
        )

    print(f"Train chunks: {len(train_ds):,}")
    print(f"Val chunks:   {len(val_ds):,}")
    print(f"Input dim:    {train_ds.input_dim}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(
        model_name=args.model,
        input_dim=train_ds.input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    train_labels = pd.read_csv(args.train_csv)["label_is_human"].astype(int).values \
        if args.train_csv and Path(args.train_csv).exists() \
        else np.load(Path(args.train_db_dir) / "labels.npy").astype(int)
    pos_weight = compute_pos_weight(train_labels)
    criterion = build_loss(args.loss_name, pos_weight=pos_weight, device=device, focal_gamma=args.focal_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    print(f"Using loss={args.loss_name}, pos_weight={pos_weight:.4f}")

    history = []
    best_f1 = -1.0
    best_auc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_m = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            threshold=args.threshold,
            train_mode=True,
            clip_grad=args.clip_grad,
        )
        val_m = run_one_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            criterion=criterion,
            device=device,
            threshold=args.threshold,
            train_mode=False,
            clip_grad=0.0,
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"val_{k}": v for k, v in val_m.items()},
        }
        history.append(row)

        print(
            f"[{epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_m['loss']:.4f} train_acc={train_m['acc']:.4f} train_f1={train_m['f1']:.4f} train_auc={train_m['auc']:.4f} | "
            f"val_loss={val_m['loss']:.4f} val_acc={val_m['acc']:.4f} val_f1={val_m['f1']:.4f} val_auc={val_m['auc']:.4f}"
        )

        better = (val_m["f1"] > best_f1) or (abs(val_m["f1"] - best_f1) < 1e-9 and val_m["auc"] > best_auc)
        if better:
            best_f1 = val_m["f1"]
            best_auc = val_m["auc"]
            ckpt = {
                "epoch": epoch,
                "args": vars(args),
                "model_state": model.state_dict(),
                "input_dim": train_ds.input_dim,
                "val_metrics": val_m,
            }
            best_path = out_dir / "checkpoint_best.pt"
            torch.save(ckpt, best_path)
            print(f"  -> saved best checkpoint: {best_path}")

    final_path = out_dir / "checkpoint_final.pt"
    torch.save(
        {
            "epoch": args.epochs,
            "args": vars(args),
            "model_state": model.state_dict(),
            "input_dim": train_ds.input_dim,
        },
        final_path,
    )
    print(f"Saved final checkpoint: {final_path}")

    hist_df = pd.DataFrame(history)
    save_curves(hist_df, out_dir)

    # Final summary for quick read
    best_row = hist_df.iloc[hist_df["val_f1"].idxmax()]
    print(
        f"Best epoch={int(best_row['epoch'])} | "
        f"val_acc={best_row['val_acc']:.4f}, val_f1={best_row['val_f1']:.4f}, val_auc={best_row['val_auc']:.4f}"
    )

    if args.run_analysis:
        best_ckpt = out_dir / "checkpoint_best.pt"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "analyze_and_plot.py"),
            "--checkpoint",
            str(best_ckpt),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            "0",
            "--threshold",
            str(args.threshold),
        ]
        if args.analysis_out_dir:
            cmd.extend(["--out-dir", args.analysis_out_dir])
        print("Running analysis:", " ".join(cmd))
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
