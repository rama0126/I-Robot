# I-Robot

Official Implementation for 

I-Robot: Identifying Robotic and Human Motion in Humanoids, ICML 2026 Workshop [Paper](https://openreview.net/pdf?id=KID7LITvcW)



---




Binary classifier: **human (1)** vs **humanoid robot (0)**, trained on MHR motion sequences.





You can get the **HumanVsHumanoid (HvH) dataset** in here [Google Drive](https://drive.google.com/drive/folders/1p4sKvaXCuHjcWTHF1UZKNTEWX3F4RbQa)
It includes the raw YouTube URLs used for data collection, human-annotated bounding boxes for humanoid robots, and the extracted MHR pose sequences used for model training and evaluation.
```
/workspace/irobot/datasets/
  human/   # *_mhr.npy   (175 human sequences)
  robot/   # *_mhr.npy   (422 robot sequences)
```


---

## 1. Pipeline overview

```
datasets/{human,robot}/*.npy
        │  (Step 1) build split CSVs
        ▼
splits/all_train.csv, all_val.csv
        │  (Step 2) preprocess_chunks.py
        ▼
chunk_db/{train,val}/  (memmap)
        │  (Step 3) train.py
        ▼
outputs_binary_cls/<model>_run/checkpoint_best.pt
        │  (Step 4) predict_val.py
        ▼
predictions_val.csv
```

| File | Role |
|------|------|
| `preprocess_chunks.py` | Cut MHR sequences into fixed windows → memmap DB (`chunk_db`) |
| `dataset.py` | Training datasets (`PrebuiltChunkDataset`, `MotionChunkDataset`) |
| `model/init_model.py` | iRobot model definition + `build_model` factory |
| `loss.py` | BCE / Focal loss |
| `train.py` | Training entry point |
| `predict_val.py` | Generate validation prediction CSV from a checkpoint |

---

## 2. Data format

Each `*_mhr.npy` is a pickled dict (MHR record). The iRobot model uses the
`pred_joint_coords` key by default:

- `pred_joint_coords` : shape `(T, 127, 3)` → flattened to **381** features per frame
- `T` (frame count) = `shape[0]` → this is the `frame_length` used for windowing

The label comes from the folder: `human/` → `label_is_human = 1`, `robot/` → `0`.

---

## 3. iRobot model variants

Set with `--model`. All are Dual-branch BiGRU (raw + 1st-order difference); only the
ablation flags differ.

| `--model` | Configuration |
|-----------|---------------|
| `irobot_raw` | raw branch only (no difference) |
| `irobot_dual` | raw + diff, concat fusion |
| `irobot_pool_conv` | raw + diff, ConvAttn fusion **(default, recommended)** |
| `irobot_pool_conv_diff` | diff branch only, ConvAttn fusion |

---

## 4. Step 1 — Build split CSVs

There is **no split file yet**, and `preprocess_chunks.py` requires a CSV with the
columns `file_path`, `label_is_human`, `frame_length`. Generate train/val splits from
the two dataset folders with the snippet below (save as `make_splits.py` and run once):

```python
# make_splits.py
import glob, os
import numpy as np
import pandas as pd

DATA_ROOT = "/workspace/irobot/datasets"
OUT_DIR   = "/workspace/irobot/irobot_src/splits"
VAL_RATIO = 0.2
SEED      = 42

rows = []
for folder, label in [("human", 1), ("robot", 0)]:
    for path in sorted(glob.glob(os.path.join(DATA_ROOT, folder, "*_mhr.npy"))):
        rec = np.load(path, allow_pickle=True)
        rec = rec.item() if rec.shape == () else rec[0]
        T = int(np.asarray(rec["pred_joint_coords"]).shape[0])   # frame length
        rows.append({
            "file_path":      path,
            "label_is_human": label,
            "frame_length":   T,
            "source":         folder,                 # optional metadata
            "class_name":     folder,
            "filename":       os.path.basename(path),
        })

df = pd.DataFrame(rows)

# Stratified train/val split (keep human/robot ratio in both)
rng = np.random.default_rng(SEED)
train_parts, val_parts = [], []
for label, grp in df.groupby("label_is_human"):
    grp = grp.sample(frac=1.0, random_state=SEED)     # shuffle
    n_val = int(len(grp) * VAL_RATIO)
    val_parts.append(grp.iloc[:n_val])
    train_parts.append(grp.iloc[n_val:])

os.makedirs(OUT_DIR, exist_ok=True)
pd.concat(train_parts).to_csv(os.path.join(OUT_DIR, "all_train.csv"), index=False)
pd.concat(val_parts).to_csv(os.path.join(OUT_DIR, "all_val.csv"),   index=False)
print(f"train={sum(len(p) for p in train_parts)}  val={sum(len(p) for p in val_parts)}")
```

```bash
cd /workspace/irobot/irobot_src
python make_splits.py
# -> splits/all_train.csv, splits/all_val.csv
```

> `dataset.py` also auto-searches `irobot_src/splits/` (and `training/splits/`,
> `extraction/splits/`) when `--train-csv`/`--val-csv` are omitted.

---

## 5. Step 2 — Preprocess into chunk DB

Run once for train and once for val. iRobot's default feature is `pred_joint_coords`
(127 joints × 3 = 381 dims). `--window-size` fixes the temporal window (default 32).

```bash
cd /workspace/irobot/irobot_src

python preprocess_chunks.py \
    --csv        splits/all_train.csv \
    --out-dir    chunk_db/train \
    --feature-key pred_joint_coords \
    --window-size 32

python preprocess_chunks.py \
    --csv        splits/all_val.csv \
    --out-dir    chunk_db/val \
    --feature-key pred_joint_coords \
    --window-size 32
```

Result:
```
chunk_db/train/
  labels.npy
  meta.csv
  pred_joint_coords/
    chunks.npy      # (N, 32, 381) float32 memmap
    info.json       # {feature_key, window_size, F, N, original_shape}
```

Helpers:
- `--list-keys` — print available MHR keys in a file, then exit
- `--force` — rebuild even if it already exists
- `--feature-key all` — auto-extract every temporal key

---

## 6. Step 3 — Train

```bash
python train.py \
    --model         irobot_pool_conv \
    --feature-key   pred_joint_coords \
    --train-db-dir  chunk_db/train \
    --val-db-dir    chunk_db/val \
    --epochs        100 \
    --batch-size    32 \
    --lr            1e-3
```

### Key arguments (defaults)

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `irobot_pool_conv` | iRobot variant (see §3) |
| `--feature-key` | `pred_joint_coords` | MHR key to use |
| `--train-db-dir` | `/workspace/irobot/training/chunk_db/train` | Train chunk DB |
| `--val-db-dir` | `/workspace/irobot/training/chunk_db/val` | Val chunk DB |
| `--epochs` | `100` | Number of epochs |
| `--batch-size` | `32` | Batch size |
| `--lr` | `1e-3` | Learning rate (AdamW + CosineAnnealing) |
| `--weight-decay` | `1e-4` | Weight decay |
| `--dropout` | `0.2` | Dropout |
| `--hidden-dim` | `256` | GRU hidden size |
| `--loss-name` | `bce` | `bce` or `focal` |
| `--focal-gamma` | `2.0` | Focal loss γ |
| `--clip-grad` | `1.0` | Gradient clipping |
| `--threshold` | `0.5` | Classification threshold |
| `--seed` | `42` | Random seed |
| `--save-dir` | `outputs_binary_cls/<model>_run` | Output directory |

> **GPU selection:** `train.py` hardcodes `os.environ["CUDA_VISIBLE_DEVICES"] = "3"`
> at the top. Edit this to use a different GPU.
>
> **Class imbalance** (422 robot vs 175 human) is handled automatically: `pos_weight`
> is computed from the training labels and passed to the loss.

### Outputs (in `--save-dir`)
- `checkpoint_best.pt` — best val-F1 checkpoint
- `checkpoint_final.pt` — last-epoch checkpoint
- `config.json` — run configuration
- `train_log.csv` — per-epoch loss/acc/f1/auc
- `training_curves.png` — training curves (if matplotlib is installed)

Per-epoch log line:
```
[001/100] train_loss=... train_acc=... train_f1=... train_auc=... | val_loss=... val_acc=... val_f1=... val_auc=...
```

---

## 7. Step 4 — Predict

Generate a validation prediction CSV from a finished run directory.

```bash
python predict_val.py --run-dir outputs_binary_cls/irobot_pool_conv_run
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--run-dir` | (required) | Folder containing `checkpoint_best.pt` |
| `--ckpt` | `checkpoint_best.pt` | Checkpoint filename |
| `--out` | `<run-dir>/predictions_val.csv` | Output CSV path |
| `--threshold` | `0.5` | Classification threshold |

> In the output CSV, labels/probs are **flipped to robot=1 (positive), human=0**
> (robot-detection view). ACC / Precision / Recall / F1 are printed at the end.

---

## 8. End-to-end example

```bash
cd /workspace/irobot/irobot_src

# 1) Build split CSVs from datasets/{human,robot}
python make_splits.py

# 2) Preprocess
python preprocess_chunks.py --csv splits/all_train.csv --out-dir chunk_db/train --feature-key pred_joint_coords --window-size 32
python preprocess_chunks.py --csv splits/all_val.csv   --out-dir chunk_db/val   --feature-key pred_joint_coords --window-size 32

# 3) Train
python train.py --model irobot_pool_conv --train-db-dir chunk_db/train --val-db-dir chunk_db/val --epochs 100

# 4) Predict
python predict_val.py --run-dir outputs_binary_cls/irobot_pool_conv_run
```

### License

The dataset annotations, labels, metadata, train/validation splits, and extracted pose parameters are licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.

This means that the dataset may be shared and adapted for non-commercial purposes, provided that proper attribution is given.

The original YouTube videos are not distributed by this repository and are not covered by this license. All rights to the original videos remain with their respective copyright holders.

Code in this repository is licensed separately. Please see the code license file for details.
