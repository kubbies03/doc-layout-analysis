"""
DocLayNet × YOLOv11 — Complete Training Pipeline for Kaggle
============================================================
Run end-to-end on Kaggle notebook (Accelerator: GPU T4 ×2 or ×1).
Includes:
  1. COCO JSON → YOLO txt conversion
  2. Balanced training set (oversampling rare classes)
  3. Dataset YAML generation
  4. Training with config calibrated from profiling results
  5. Evaluation + per-class AP + macro AP logging

Integrated profiling findings:
  - Imbalance ratio 97:1 → oversample Title 3×, Footnote 3×, Caption 2×
  - 47.4% tiny bbox → imgsz=1024 required
  - 74.2% bbox AR > 5:1 → rect=True
  - P95 objects/page = 28 → max_det=100 sufficient
  - Split has minor variance but acceptable (keep pre-defined split)
"""

# =================================================================
# CELL 1: Install & Imports
# =================================================================
# !pip install ultralytics>=8.3 -q

import json
import os
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# =================================================================
# CELL 2: Config — EDIT HERE
# =================================================================

# --- Paths ---
DOCLAYNET_ROOT = Path("/kaggle/input/doclaynet/DocLayNet_core")
COCO_DIR = DOCLAYNET_ROOT / "COCO"
PNG_DIR = DOCLAYNET_ROOT / "PNG"

WORK_DIR = Path("/kaggle/working/doclaynet_yolo")
WORK_DIR.mkdir(parents=True, exist_ok=True)

# --- Class mapping (from profiling) ---
# DocLayNet COCO JSON category_id → YOLO class index
# Verify category_id in JSON; mapping below assumes 1-indexed
CLASS_NAMES = [
    "Caption",         # 0
    "Footnote",        # 1
    "Formula",         # 2
    "List-item",       # 3
    "Page-footer",     # 4
    "Page-header",     # 5
    "Picture",         # 6
    "Section-header",  # 7
    "Table",           # 8
    "Text",            # 9
    "Title",           # 10
]

# --- Oversampling config (from profiling: imbalance 97:1) ---
# Key = class index, value = number of times to repeat images containing this class
# Only oversample classes with count < 50% of median
OVERSAMPLE_MAP = {
    1: 3,    # Footnote:  5,619 → ×3
    10: 3,   # Title:     4,437 → ×3
    0: 2,    # Caption:  19,218 → ×2
}

# --- Training run ---
# Change RUN_NAME for each ablation
RUN_NAME = "run_B_1024_baseline"

# =================================================================
# CELL 3: COCO JSON → YOLO Format Conversion
# =================================================================

def convert_coco_to_yolo(coco_json_path, images_src_dir, output_images_dir, output_labels_dir):
    """
    Convert COCO JSON annotations → YOLO txt format.
    Copy/symlink images into YOLO-compatible directory.
    Returns: dict mapping image_id → set of class indices present in that image.
    """
    output_images_dir = Path(output_images_dir)
    output_labels_dir = Path(output_labels_dir)
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    # Build category mapping: coco_cat_id → 0-indexed YOLO class
    categories = sorted(coco["categories"], key=lambda x: x["id"])
    cat_id_to_yolo = {cat["id"]: idx for idx, cat in enumerate(categories)}

    # Verify mapping matches our CLASS_NAMES
    for cat in categories:
        yolo_idx = cat_id_to_yolo[cat["id"]]
        expected_name = CLASS_NAMES[yolo_idx]
        actual_name = cat["name"]
        if expected_name.lower().replace("-", "").replace("_", "") != actual_name.lower().replace("-", "").replace("_", ""):
            print(f"  ⚠ Category mismatch: YOLO idx {yolo_idx} = '{expected_name}' but JSON has '{actual_name}'")
            print(f"    → Adjust CLASS_NAMES or cat_id_to_yolo mapping!")

    # Image info
    img_dict = {img["id"]: img for img in coco["images"]}

    # Group annotations by image
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    # Track which classes appear in each image (for oversampling)
    img_classes = {}  # image_filename → set of class indices

    converted = 0
    skipped = 0

    for img_id, img_info in img_dict.items():
        w = img_info["width"]
        h = img_info["height"]
        filename = img_info["file_name"]
        stem = Path(filename).stem

        # Source image
        src_img = Path(images_src_dir) / filename
        if not src_img.exists():
            # Try without subdirectory
            src_img = Path(images_src_dir) / Path(filename).name
        if not src_img.exists():
            skipped += 1
            continue

        # Symlink image (avoid copying 30GB)
        dst_img = output_images_dir / Path(filename).name
        if not dst_img.exists():
            try:
                os.symlink(src_img, dst_img)
            except OSError:
                shutil.copy2(src_img, dst_img)

        # Convert annotations
        classes_in_img = set()
        lines = []
        for ann in anns_by_img.get(img_id, []):
            bx, by, bw, bh = ann["bbox"]  # COCO: x_min, y_min, width, height
            yolo_cls = cat_id_to_yolo[ann["category_id"]]
            classes_in_img.add(yolo_cls)

            # YOLO: class cx cy w h (normalized)
            cx = (bx + bw / 2) / w
            cy = (by + bh / 2) / h
            nw = bw / w
            nh = bh / h

            # Clamp to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        # Write label file
        txt_path = output_labels_dir / f"{stem}.txt"
        with open(txt_path, "w") as f:
            f.write("\n".join(lines))

        img_classes[Path(filename).name] = classes_in_img
        converted += 1

    print(f"  Converted: {converted:,} images, Skipped: {skipped}")
    return img_classes


print("=" * 60)
print("STEP 1: Converting COCO JSON → YOLO format")
print("=" * 60)

split_img_classes = {}
for split in ["train", "val", "test"]:
    json_path = COCO_DIR / f"{split}.json"
    if not json_path.exists():
        print(f"  ✗ {json_path} not found, skipping {split}")
        continue

    print(f"\n  Converting {split}...")
    img_dir = PNG_DIR / split if (PNG_DIR / split).exists() else PNG_DIR
    out_imgs = WORK_DIR / "images" / split
    out_lbls = WORK_DIR / "labels" / split

    img_classes = convert_coco_to_yolo(json_path, img_dir, out_imgs, out_lbls)
    split_img_classes[split] = img_classes

# Verify
for split in ["train", "val", "test"]:
    img_count = len(list((WORK_DIR / "images" / split).glob("*"))) if (WORK_DIR / "images" / split).exists() else 0
    lbl_count = len(list((WORK_DIR / "labels" / split).glob("*.txt"))) if (WORK_DIR / "labels" / split).exists() else 0
    print(f"  {split}: {img_count:,} images, {lbl_count:,} labels")


# =================================================================
# CELL 4: Create Balanced Training Set (Oversampling)
# =================================================================

print("\n" + "=" * 60)
print("STEP 2: Creating balanced training set")
print("=" * 60)

def create_balanced_train_list(img_classes, images_dir, output_txt_path, oversample_map):
    """
    Create text file listing image paths, duplicating images containing rare classes.
    YOLO reads this file instead of scanning the directory.
    """
    images_dir = Path(images_dir)
    all_images = sorted(img_classes.keys())

    lines = []
    base_count = 0
    extra_count = 0

    for img_name in all_images:
        img_path = str(images_dir / img_name)
        lines.append(img_path)
        base_count += 1

        # Check if image contains any rare class
        classes = img_classes[img_name]
        max_repeat = 1
        for cls_idx, repeat in oversample_map.items():
            if cls_idx in classes:
                max_repeat = max(max_repeat, repeat)

        # Add extra copies (repeat - 1 since already added once above)
        for _ in range(max_repeat - 1):
            lines.append(img_path)
            extra_count += 1

    with open(output_txt_path, "w") as f:
        f.write("\n".join(lines))

    print(f"  Base images:  {base_count:,}")
    print(f"  Extra copies: {extra_count:,}")
    print(f"  Total lines:  {len(lines):,} ({extra_count/base_count*100:.1f}% increase)")
    return len(lines)


# Balanced train list
balanced_txt = WORK_DIR / "train_balanced.txt"
if "train" in split_img_classes:
    total_balanced = create_balanced_train_list(
        split_img_classes["train"],
        WORK_DIR / "images" / "train",
        balanced_txt,
        OVERSAMPLE_MAP,
    )

# Normal train list (for baseline comparison)
normal_txt = WORK_DIR / "train_normal.txt"
if "train" in split_img_classes:
    normal_images = sorted(split_img_classes["train"].keys())
    with open(normal_txt, "w") as f:
        f.write("\n".join(str(WORK_DIR / "images" / "train" / img) for img in normal_images))
    print(f"  Normal train list: {len(normal_images):,} images")


# =================================================================
# CELL 5: Dataset YAML files
# =================================================================

print("\n" + "=" * 60)
print("STEP 3: Writing dataset YAML files")
print("=" * 60)

yaml_template = """# DocLayNet — YOLO format
# Auto-generated {timestamp}

path: {work_dir}
train: {train_source}
val: images/val
test: images/test

nc: 11
names:
  0: Caption
  1: Footnote
  2: Formula
  3: List-item
  4: Page-footer
  5: Page-header
  6: Picture
  7: Section-header
  8: Table
  9: Text
  10: Title
"""

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

# Baseline YAML (no oversampling)
yaml_normal = yaml_template.format(
    timestamp=timestamp,
    work_dir=WORK_DIR,
    train_source="images/train",
)
yaml_normal_path = WORK_DIR / "doclaynet.yaml"
with open(yaml_normal_path, "w") as f:
    f.write(yaml_normal)
print(f"  ✓ {yaml_normal_path}")

# Balanced YAML (oversampling)
yaml_balanced = yaml_template.format(
    timestamp=timestamp,
    work_dir=WORK_DIR,
    train_source=str(balanced_txt),
)
yaml_balanced_path = WORK_DIR / "doclaynet_balanced.yaml"
with open(yaml_balanced_path, "w") as f:
    f.write(yaml_balanced)
print(f"  ✓ {yaml_balanced_path}")


# =================================================================
# CELL 6: Training
# =================================================================
# ⚠ RUN THIS CELL WITH GPU ENABLED
# ⚠ Each run estimated 8-12h on T4 at imgsz=1024
# ⚠ Change RUN_NAME and config for each ablation

print("\n" + "=" * 60)
print(f"STEP 4: Training — {RUN_NAME}")
print("=" * 60)

from ultralytics import YOLO

# -------------------------------------------------------------------
# ABLATION CONFIGS
# Uncomment exactly 1 block corresponding to the run you want.
# Settings not listed = keep COMMON_CONFIG defaults below.
# -------------------------------------------------------------------

# === COMMON CONFIG (unchanged across runs) ===
COMMON_CONFIG = dict(
    # Optimizer
    optimizer="SGD",
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    # Warmup
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    # Scheduler
    cos_lr=True,
    # Training
    epochs=100,
    patience=20,
    amp=True,
    nbs=64,
    cache="disk",
    workers=2,
    pretrained=True,
    deterministic=True,
    seed=42,
    verbose=True,
    # NMS / eval
    conf=0.001,
    iou=0.6,
    max_det=100,
    # Output
    project=str(WORK_DIR / "runs"),
    exist_ok=True,
)

# === DOCUMENT-SAFE AUGMENTATION DEFAULTS ===
DOC_AUG_CONSERVATIVE = dict(
    hsv_h=0.01,
    hsv_s=0.3,
    hsv_v=0.3,
    degrees=0.0,
    translate=0.1,
    scale=0.3,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.0,      # Disabled for safety
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
    close_mosaic=15,
)

DOC_AUG_TUNED = dict(
    hsv_h=0.015,
    hsv_s=0.4,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.15,
    scale=0.4,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,      # Enabled — DocLayNet is mostly LTR, flip is acceptable
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
    close_mosaic=20,  # Disable mosaic earlier → learn real spatial relations
)

# -------------------------------------------------------------------
# RUN A: Baseline 640
# Goal: Lower bound, fast training (~4h)
# -------------------------------------------------------------------
# RUN_NAME = "run_A_640_baseline"
# run_config = dict(
#     data=str(yaml_normal_path),
#     imgsz=640,
#     batch=16,
#     rect=False,
#     multi_scale=False,
#     **DOC_AUG_CONSERVATIVE,
# )

# -------------------------------------------------------------------
# RUN B: Resolution 1024 (RECOMMENDED FIRST RUN)
# Goal: Confirm imgsz=1024 helps tiny bbox (47.4%)
# Profiling: rect=True because 74.2% bbox AR > 5:1
# -------------------------------------------------------------------
RUN_NAME = "run_B_1024_baseline"
run_config = dict(
    data=str(yaml_normal_path),
    imgsz=1024,
    batch=4,
    rect=True,          # ENABLED — 74.2% bbox has AR > 5:1
    multi_scale=False,
    **DOC_AUG_CONSERVATIVE,
)

# -------------------------------------------------------------------
# RUN C: 640 + Class Balancing (oversampling)
# Goal: Improve rare classes (Title, Footnote, Caption)
# -------------------------------------------------------------------
# RUN_NAME = "run_C_640_balanced"
# run_config = dict(
#     data=str(yaml_balanced_path),   # ← balanced dataset
#     imgsz=640,
#     batch=24,
#     rect=True,
#     multi_scale=False,
#     **DOC_AUG_CONSERVATIVE,
# )

# -------------------------------------------------------------------
# RUN D: 640 + Tuned Augmentation
# Goal: Better regularization, reduce overfitting
# -------------------------------------------------------------------
# RUN_NAME = "run_D_1024_aug"
# run_config = dict(
#     data=str(yaml_normal_path),
#     imgsz=640,
#     batch=24,
#     rect=True,
#     multi_scale=False,
#     **DOC_AUG_TUNED,              # ← tuned augmentation
# )

# -------------------------------------------------------------------
# RUN E: 1024 + Balanced + Tuned Aug (combined)
# Goal: Best overall, run last
# -------------------------------------------------------------------
# RUN_NAME = "run_E_1024_combined"
# run_config = dict(
#     data=str(yaml_balanced_path),  # ← balanced dataset
#     imgsz=1024,
#     batch=4,
#     rect=True,
#     multi_scale=False,
#     **DOC_AUG_TUNED,              # ← tuned augmentation
# )

# -------------------------------------------------------------------
# TRAIN
# -------------------------------------------------------------------
model = YOLO("yolo11s.pt")

results = model.train(
    name=RUN_NAME,
    **COMMON_CONFIG,
    **run_config,
)


# =================================================================
# CELL 7: Evaluation — run after training completes
# =================================================================

print("\n" + "=" * 60)
print("STEP 5: Evaluation")
print("=" * 60)

import torch

# Load best weights
best_weights = WORK_DIR / "runs" / RUN_NAME / "weights" / "best.pt"
if not best_weights.exists():
    # Fallback: search in runs directory
    candidates = list((WORK_DIR / "runs").rglob("best.pt"))
    if candidates:
        best_weights = candidates[-1]
        print(f"  Using weights: {best_weights}")
    else:
        print("  ✗ No best.pt found!")

model_eval = YOLO(str(best_weights))

# --- Validate on VAL set ---
print("\n  Evaluating on VAL set...")
val_metrics = model_eval.val(
    data=str(yaml_normal_path),
    imgsz=run_config.get("imgsz", 1024),
    batch=run_config.get("batch", 4),
    conf=0.001,
    iou=0.6,
    max_det=100,
    split="val",
    rect=True,
)

# --- Validate on TEST set ---
print("\n  Evaluating on TEST set...")
test_metrics = model_eval.val(
    data=str(yaml_normal_path),
    imgsz=run_config.get("imgsz", 1024),
    batch=run_config.get("batch", 4),
    conf=0.001,
    iou=0.6,
    max_det=100,
    split="test",
    rect=True,
)

# --- Log results ---
def log_metrics(metrics, split_name, run_name, run_cfg):
    """Print and return structured results dict."""
    per_class_ap50 = metrics.box.ap50  # array of shape (n_classes,)
    macro_ap50 = np.mean(per_class_ap50)

    print(f"\n  {'='*50}")
    print(f"  {run_name} — {split_name}")
    print(f"  {'='*50}")
    print(f"  mAP50:      {metrics.box.map50:.4f}")
    print(f"  mAP50-95:   {metrics.box.map:.4f}")
    print(f"  Macro AP50: {macro_ap50:.4f}")
    print(f"  Gap (mAP50 - Macro): {metrics.box.map50 - macro_ap50:+.4f}")
    print()

    for i, name in enumerate(CLASS_NAMES):
        flag = " ← rare" if i in OVERSAMPLE_MAP else ""
        print(f"    {name:<18s} AP50={per_class_ap50[i]:.4f}{flag}")

    # VRAM
    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"\n  Peak VRAM: {peak_vram:.2f} GB")

    result = {
        "run": run_name,
        "split": split_name,
        "imgsz": run_cfg.get("imgsz", "?"),
        "batch": run_cfg.get("batch", "?"),
        "rect": run_cfg.get("rect", False),
        "mAP50": round(metrics.box.map50, 4),
        "mAP50_95": round(metrics.box.map, 4),
        "macro_AP50": round(macro_ap50, 4),
        "peak_vram_gb": round(peak_vram, 2),
    }
    for i, name in enumerate(CLASS_NAMES):
        result[f"AP50_{name}"] = round(float(per_class_ap50[i]), 4)

    return result


val_result = log_metrics(val_metrics, "val", RUN_NAME, run_config)
test_result = log_metrics(test_metrics, "test", RUN_NAME, run_config)

# Save results as JSON for easy comparison
import json as json_mod

results_path = WORK_DIR / "runs" / f"{RUN_NAME}_results.json"
with open(results_path, "w") as f:
    json_mod.dump({"val": val_result, "test": test_result}, f, indent=2)
print(f"\n  ✓ Results saved: {results_path}")


# =================================================================
# CELL 8: Quick Visual Inspection — run after eval
# =================================================================

print("\n" + "=" * 60)
print("STEP 6: Visual Inspection")
print("=" * 60)

import matplotlib.pyplot as plt
from glob import glob
import random

val_images = sorted(glob(str(WORK_DIR / "images" / "val" / "*.png")))

if val_images:
    # Get 12 images: 4 random + 4 most objects + 4 with rare class
    random.seed(42)
    sample_random = random.sample(val_images, min(4, len(val_images)))

    # Most objects (need to count from label files)
    val_labels = WORK_DIR / "labels" / "val"
    obj_counts = {}
    for lbl in val_labels.glob("*.txt"):
        with open(lbl) as f:
            obj_counts[lbl.stem] = len(f.readlines())

    # Top 4 by object count
    top_dense = sorted(obj_counts.items(), key=lambda x: -x[1])[:4]
    sample_dense = [str(WORK_DIR / "images" / "val" / f"{stem}.png") for stem, _ in top_dense]

    # Images with rare classes (Title=10, Footnote=1)
    rare_images = []
    for lbl in val_labels.glob("*.txt"):
        with open(lbl) as f:
            classes = set(int(line.split()[0]) for line in f.readlines() if line.strip())
        if classes & {1, 10}:  # Footnote or Title
            rare_images.append(str(WORK_DIR / "images" / "val" / f"{lbl.stem}.png"))
    sample_rare = rare_images[:4] if rare_images else []

    all_samples = list(dict.fromkeys(sample_random + sample_dense + sample_rare))[:12]

    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    axes = axes.flatten()

    for i, img_path in enumerate(all_samples):
        if i >= len(axes):
            break
        results = model_eval.predict(img_path, conf=0.25, imgsz=run_config.get("imgsz", 1024), verbose=False)
        plotted = results[0].plot()
        axes[i].imshow(plotted)
        n_det = len(results[0].boxes)
        axes[i].set_title(f"{Path(img_path).stem}\n{n_det} detections", fontsize=8)
        axes[i].axis("off")

    # Hide unused axes
    for j in range(len(all_samples), len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"{RUN_NAME} — Sample Predictions (conf≥0.25)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(WORK_DIR / "runs" / f"{RUN_NAME}_samples.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  ✓ Saved: {RUN_NAME}_samples.png")
else:
    print("  ✗ No val images found for inspection")


# =================================================================
# CELL 9: Compare All Runs — run after completing ≥2 runs
# =================================================================

print("\n" + "=" * 60)
print("STEP 7: Ablation Comparison")
print("=" * 60)

all_results = []
for rfile in sorted((WORK_DIR / "runs").glob("*_results.json")):
    with open(rfile) as f:
        data = json_mod.load(f)
    if "val" in data:
        all_results.append(data["val"])

if len(all_results) >= 2:
    import pandas as pd

    df = pd.DataFrame(all_results)
    cols_order = ["run", "imgsz", "batch", "rect", "mAP50", "mAP50_95", "macro_AP50"]
    cols_order += [c for c in df.columns if c.startswith("AP50_")]
    cols_order += ["peak_vram_gb"]
    df = df[[c for c in cols_order if c in df.columns]]

    print("\n  Ablation Results (val set):")
    print(df.to_string(index=False))

    # Highlight best macro AP
    best_idx = df["macro_AP50"].idxmax()
    print(f"\n  ★ Best Macro AP50: {df.loc[best_idx, 'run']} ({df.loc[best_idx, 'macro_AP50']:.4f})")
else:
    print(f"  Only {len(all_results)} run(s) found. Complete ≥2 runs to compare.")
    if all_results:
        print(f"  Current: {all_results[0]['run']} → mAP50={all_results[0]['mAP50']}, Macro={all_results[0]['macro_AP50']}")
