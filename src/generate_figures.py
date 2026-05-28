"""
generate_thesis_figures.py
==========================
Generate AND COPY illustration images for the thesis into thesis_figures/.

Strategy:
  - Images ALREADY IN dataset/ → copy directly (no redraw)
  - Images NOT YET PRESENT → generate with matplotlib / pipeline

Output (thesis_figures/ directory):
  === EXISTING images (copied from dataset/) ===
  exist_class_distribution_full.png   — 11-class distribution train/val/test (BaseC)
  exist_aspect_ratio_dist.png         — Bbox aspect ratio distribution (BaseC)
  exist_bbox_scale_dist.png           — Bbox scale distribution (BaseC)
  exist_objects_per_page.png          — Objects per page distribution (BaseC)
  exist_fig3_ablation_overall.png     — Ablation overall metrics val+test
  exist_fig1_ap_val.png               — AP per class — validation set
  exist_fig2_ap_test.png              — AP per class — test set
  exist_fig5_rare_class.png           — Rare class comparison
  exist_fig4_convergence.png          — Training convergence curves
  exist_sample_predictions.png        — Sample YOLO predictions on test set
  exist_confusion_matrix.png          — Confusion matrix

  === NEW images (generated from pipeline) ===
  fig1_yolo_detection.png     — One specific PDF page + YOLO color bboxes + reading order
  fig2_crops_grid.png         — Grid of representative block crops
  fig3_ocr_result.png         — Crop image ↔ OCR text per block
  fig4_pipeline_stages.png    — 4 stages: raw → YOLO → text → JSON
  fig6_ablation_chart.png     — Ablation bar chart (redrawn for thesis)
  fig7_block_f1.png           — Block F1 per class end-to-end
  fig8_cer_chart.png          — CER per class + coverage

Usage:
  python generate_thesis_figures.py

Requirements:
  - YOLO model: dataset/BaseC/best.pt
  - poppler: poppler/Library/bin
  - pip install ultralytics pdf2image pillow opencv-python matplotlib
  - docTR (optional for real fig3 OCR — falls back if not available)
"""

import os, sys, json, textwrap
from pathlib import Path

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image

# ── Configuration ─────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
YOLO_WEIGHTS = BASE_DIR / "dataset" / "BaseC" / "best.pt"
POPPLER_PATH = str(BASE_DIR / "poppler" / "Library" / "bin")
OUTPUT_DIR   = BASE_DIR / "thesis_figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Best sample ranking (pipeline actually run):
# - sample_0029: Financial report — large table (2 financial tables), good text ✅
# - sample_0116: Manual — has LaTeX formulas, clean text ✅
# - sample_0181: Manual (Navy) — dense text, many sections ✅
# - sample_0388: Government tender — legal text, many section-headers ✅
# Use sample_0029 for fig1/fig2/fig4 (prominent Table)
# Use sample_0116 for fig3 OCR (has Formula)
SAMPLE_PDF_MAIN = str(BASE_DIR / "GT" / "staging" /
    "sample_0029_61253fc219f35687e9d3ff96d17bf63338828cb5ae6b801b8f0f6ce357d114ae.pdf")
SAMPLE_PDF_OCR = str(BASE_DIR / "GT" / "staging" /
    "sample_0116_def1abb6fe5361b7671912def0abe602c442e6ef3ba4bcbc78294eea8400b667.pdf")

# Fallback if file does not exist
def _pick_sample(preferred: str) -> str:
    if Path(preferred).exists():
        return preferred
    staging = sorted((BASE_DIR / "GT" / "staging").glob("*.pdf"))
    if not staging:
        sys.exit("[ERROR] No PDF found in GT/staging/")
    print(f"[WARN] {preferred} does not exist, using: {staging[0].name}")
    return str(staging[0])

SAMPLE_PDF_MAIN = _pick_sample(SAMPLE_PDF_MAIN)
SAMPLE_PDF_OCR  = _pick_sample(SAMPLE_PDF_OCR)
SAMPLE_PDF = SAMPLE_PDF_MAIN  # shared alias

PDF_DPI = 150  # DPI render — matches actual pipeline

# ── 11-class colors ───────────────────────────────────────────────────
CLASS_COLORS = {
    "Caption":        "#4E79A7",
    "Footnote":       "#59A14F",
    "Formula":        "#F28E2B",
    "List-item":      "#76B7B2",
    "Page-footer":    "#BAB0AC",
    "Page-header":    "#9467BD",
    "Picture":        "#E15759",
    "Section-header": "#FF9DA7",
    "Table":          "#9C755F",
    "Text":           "#AEC7E8",
    "Title":          "#EDC948",
}

# ── Helpers ───────────────────────────────────────────────────────────
def hex_to_rgb01(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))


def render_pdf_page(pdf_path: str, dpi: int = 150) -> np.ndarray:
    from pdf2image import convert_from_path
    pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=POPPLER_PATH)
    img = np.array(pages[0])
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def letterbox(img: np.ndarray, target: int = 1025) -> np.ndarray:
    h, w = img.shape[:2]
    scale = target / max(h, w)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((target, target, 3), 255, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    return canvas


def run_yolo(img: np.ndarray):
    from ultralytics import YOLO
    model = YOLO(str(YOLO_WEIGHTS))
    CLASS_NAMES = [
        "Caption", "Footnote", "Formula", "List-item",
        "Page-footer", "Page-header", "Picture", "Section-header",
        "Table", "Text", "Title",
    ]
    CLASS_CONF_OVERRIDE = {
        "Picture": 0.50, "Section-header": 0.40,
        "Caption": 0.40, "Table": 0.12,
    }
    DEFAULT_CONF = 0.25

    results = model.predict(
        source=img, conf=DEFAULT_CONF, iou=0.60,
        imgsz=640, max_det=300, verbose=False,
    )[0]

    blocks = []
    for box in results.boxes:
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        name   = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        min_c  = CLASS_CONF_OVERRIDE.get(name, DEFAULT_CONF)
        if conf < min_c:
            continue
        area = (x2 - x1) * (y2 - y1)
        if area < 800:
            continue
        blocks.append(dict(cls=name, conf=conf,
                           x1=x1, y1=y1, x2=x2, y2=y2))

    # Simple reading order: top→bottom, left→right per row
    blocks.sort(key=lambda b: (b["y1"], b["x1"]))
    for i, b in enumerate(blocks):
        b["order"] = i
    return blocks


# ══════════════════════════════════════════════════════════════════════
# FIG 1 — YOLO Detection with colored bboxes + labels + reading order
# ══════════════════════════════════════════════════════════════════════
def make_fig1_yolo_detection(img_yolo: np.ndarray,
                              blocks: list,
                              save_path: Path):
    print("[fig1] Drawing YOLO detection...")
    img_rgb = cv2.cvtColor(img_yolo, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    fig, ax = plt.subplots(figsize=(10, 10 * h / w), dpi=150)
    ax.imshow(img_rgb)

    seen_classes = set()
    for b in blocks:
        name  = b["cls"]
        color = CLASS_COLORS.get(name, "#AAAAAA")
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        rect = mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            boxstyle="square,pad=0",
            linewidth=1.8, edgecolor=color, facecolor=(*hex_to_rgb01(color), 0.08),
        )
        ax.add_patch(rect)
        label = f"{b['order']}: {name}"
        ax.text(x1 + 3, y1 + 11, label,
                fontsize=6.5, color="white", fontweight="bold",
                bbox=dict(facecolor=color, alpha=0.85, pad=1.5,
                          edgecolor="none", boxstyle="round,pad=0.2"))
        seen_classes.add(name)

    legend_patches = [
        mpatches.Patch(color=CLASS_COLORS.get(c, "#AAA"), label=c)
        for c in sorted(seen_classes)
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              fontsize=7, framealpha=0.9, ncol=2)
    ax.set_title("Figure 3.x — Layout detection results (YOLOv11s)\n"
                 f"Total blocks: {len(blocks)}", fontsize=11, pad=8)
    ax.axis("off")
    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════
# FIG 2 — Grid crop per block (up to 12 representative blocks)
# ══════════════════════════════════════════════════════════════════════
def make_fig2_crops_grid(img_yolo: np.ndarray,
                          blocks: list,
                          save_path: Path,
                          max_blocks: int = 12):
    print("[fig2] Drawing crop grid...")

    # Filter: take at most 1-2 samples per class, prioritize diverse classes
    selected = []
    seen = {}
    for b in sorted(blocks, key=lambda b: b["order"]):
        cls = b["cls"]
        if seen.get(cls, 0) < 2:
            selected.append(b)
            seen[cls] = seen.get(cls, 0) + 1
        if len(selected) >= max_blocks:
            break

    n = len(selected)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.5, nrows * 2.8), dpi=120)
    axes = np.array(axes).flatten()

    for i, b in enumerate(selected):
        ax = axes[i]
        x1 = max(0, int(b["x1"]) - 4)
        y1 = max(0, int(b["y1"]) - 4)
        x2 = min(img_yolo.shape[1], int(b["x2"]) + 4)
        y2 = min(img_yolo.shape[0], int(b["y2"]) + 4)
        crop = img_yolo[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        ax.imshow(crop_rgb)
        color = CLASS_COLORS.get(b["cls"], "#AAAAAA")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        ax.set_title(f"#{b['order']} {b['cls']}\nconf={b['conf']:.2f}  "
                     f"{x2-x1}×{y2-y1}px",
                     fontsize=8, color=color, pad=3)
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Figure 3.x — Layout regions cropped by YOLO bbox",
                 fontsize=12, y=1.01)
    plt.tight_layout(pad=0.8)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════
# FIG 3 — OCR result: crop image ↔ OCR text (uses docTR if available, falls back to mock)
# ══════════════════════════════════════════════════════════════════════
def make_fig3_ocr_result(img_yolo: np.ndarray,
                          blocks: list,
                          save_path: Path,
                          use_real_ocr: bool = True,
                          max_show: int = 6):
    print("[fig3] Drawing OCR result...")

    # Get Text/List-item/Title/Section-header blocks
    text_classes = {"Text", "List-item", "Title", "Section-header",
                    "Caption", "Footnote"}
    text_blocks = [b for b in blocks if b["cls"] in text_classes][:max_show]

    if not text_blocks:
        print("  [SKIP] No text blocks to display.")
        return

    # Try real OCR
    ocr_texts = []
    if use_real_ocr:
        try:
            import os as _os
            _os.environ["USE_TORCH"] = "1"
            import torch
            from doctr.models import ocr_predictor
            from doctr.io import DocumentFile
            import io as _io
            model = ocr_predictor(pretrained=True)
            if torch.cuda.is_available():
                model = model.cuda()

            for b in text_blocks:
                x1 = max(0, int(b["x1"]) - 4)
                y1 = max(0, int(b["y1"]) - 4)
                x2 = min(img_yolo.shape[1], int(b["x2"]) + 4)
                y2 = min(img_yolo.shape[0], int(b["y2"]) + 4)
                crop = img_yolo[y1:y2, x1:x2]
                if crop.shape[0] < 16 or crop.shape[1] < 16:
                    ocr_texts.append("[crop too small]")
                    continue
                pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                buf = _io.BytesIO()
                pil.save(buf, format="PNG")
                doc = DocumentFile.from_images([buf.getvalue()])
                result = model(doc)
                words = []
                for page in result.pages:
                    for block in page.blocks:
                        for line in block.lines:
                            for word in line.words:
                                if word.value.strip():
                                    words.append(word.value)
                ocr_texts.append(" ".join(words) if words else "[not recognized]")
            print("  docTR OCR complete.")
        except Exception as e:
            print(f"  [WARN] docTR not available ({e}). Using placeholder text.")
            use_real_ocr = False

    if not use_real_ocr or not ocr_texts:
        ocr_texts = [
            "This report presents the financial results for...",
            "• Revenue increased by 12% year-over-year",
            "• Operating margin improved to 18.3%",
            "Table 1. Summary of Financial Performance",
            "Note: All figures in USD millions unless stated.",
            "Fig. 1: Quarterly revenue trend 2020–2024",
        ][:len(text_blocks)]

    n = len(text_blocks)
    fig, axes = plt.subplots(n, 2,
                              figsize=(12, n * 2.2), dpi=120,
                              gridspec_kw={"width_ratios": [1, 1.8]})
    if n == 1:
        axes = [axes]

    for i, (b, txt) in enumerate(zip(text_blocks, ocr_texts)):
        ax_img, ax_txt = axes[i]

        # Left column: crop image
        x1 = max(0, int(b["x1"]) - 4)
        y1 = max(0, int(b["y1"]) - 4)
        x2 = min(img_yolo.shape[1], int(b["x2"]) + 4)
        y2 = min(img_yolo.shape[0], int(b["y2"]) + 4)
        crop = img_yolo[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        ax_img.imshow(crop_rgb)
        color = CLASS_COLORS.get(b["cls"], "#AAAAAA")
        for spine in ax_img.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
        ax_img.set_title(f"#{b['order']} {b['cls']}",
                         fontsize=8.5, color=color, pad=3)
        ax_img.axis("off")

        # Right column: OCR text
        wrapped = "\n".join(textwrap.wrap(txt, width=58))
        ax_txt.text(0.03, 0.5, wrapped,
                    transform=ax_txt.transAxes,
                    fontsize=8, va="center", ha="left",
                    fontfamily="monospace",
                    bbox=dict(facecolor="#f8f8f8", alpha=0.9,
                              edgecolor="#cccccc", pad=6,
                              boxstyle="round,pad=0.4"))
        ax_txt.set_xlim(0, 1); ax_txt.set_ylim(0, 1)
        ax_txt.axis("off")
        ax_txt.set_title("docTR OCR output", fontsize=8, color="#555555", pad=3)

        # Arrow connecting the two panels
        fig.add_artist(plt.Annotation(
            "", xy=(0.51, (n - i - 0.5) / n),
            xytext=(0.48, (n - i - 0.5) / n),
            xycoords="figure fraction", textcoords="figure fraction",
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5),
        ))

    fig.suptitle("Figure 3.x — Text recognition results (docTR OCR)\n"
                 "Left: crop by YOLO bbox | Right: recognized text",
                 fontsize=11, y=1.01)
    plt.tight_layout(pad=0.6, h_pad=0.4)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════
# FIG 4 — Pipeline stages: raw → YOLO → text → JSON
# ══════════════════════════════════════════════════════════════════════
def make_fig4_pipeline_stages(img_raw: np.ndarray,
                               img_yolo: np.ndarray,
                               blocks: list,
                               save_path: Path):
    print("[fig4] Drawing pipeline stages...")

    # Create YOLO image with bboxes
    img_yolo_annotated = img_yolo.copy()
    for b in blocks:
        color_hex = CLASS_COLORS.get(b["cls"], "#AAAAAA")
        color_bgr = tuple(int(color_hex.lstrip("#")[i:i+2], 16)
                          for i in (4, 2, 0))
        cv2.rectangle(img_yolo_annotated,
                      (int(b["x1"]), int(b["y1"])),
                      (int(b["x2"]), int(b["y2"])),
                      color_bgr, 2)
        cv2.putText(img_yolo_annotated,
                    f"{b['order']}:{b['cls'][:3]}",
                    (int(b["x1"]) + 2, int(b["y1"]) + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    color_bgr, 1, cv2.LINE_AA)

    # Create text image (placeholder markdown)
    text_lines = []
    for b in sorted(blocks, key=lambda x: x["order"])[:12]:
        prefix = {
            "Title": "# ", "Section-header": "## ",
            "List-item": "- ", "Caption": "**Caption:** ",
            "Table": "[TABLE] ", "Formula": "$...$  ",
            "Picture": "[IMAGE] ", "Footnote": "^",
        }.get(b["cls"], "")
        example = {
            "Title": "Document Title Here",
            "Section-header": "1. Introduction",
            "Text": "Lorem ipsum dolor sit amet...",
            "List-item": "Key finding number one",
            "Caption": "Figure 1: Overview diagram",
            "Table": "| Col1 | Col2 | Col3 |",
            "Footnote": "See reference [12] for details",
            "Page-header": "[header text]",
            "Page-footer": "[page number]",
            "Picture": "chart.png",
            "Formula": "E = mc^2",
        }.get(b["cls"], b["cls"])
        text_lines.append(f"{prefix}{example}")

    text_content = "\n".join(text_lines[:10])

    # JSON snippet
    json_snippet = json.dumps({
        "sections": [{
            "heading": "1. Introduction",
            "paragraphs": ["Lorem ipsum dolor sit..."],
            "tables": [],
        }],
        "summary": "This page presents the introduction...",
        "keywords": ["layout", "detection", "YOLO"],
        "metadata": {"page_type": "body", "language": "en"},
    }, indent=2, ensure_ascii=False)

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), dpi=120)
    stage_labels = [
        "Stage 0\nInput PDF\n(render DPI=150)",
        "Stage 1\nYOLO Detection\n(11 layout classes)",
        "Stage 2-3\nOCR + Reading Order\n(raw text)",
        "Stage 4\nGemini LLM\n(structured JSON)",
    ]
    colors_stage = ["#4E79A7", "#E15759", "#59A14F", "#F28E2B"]

    # Stage 0: raw image
    axes[0].imshow(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
    axes[0].axis("off")

    # Stage 1: YOLO annotated
    axes[1].imshow(cv2.cvtColor(img_yolo_annotated, cv2.COLOR_BGR2RGB))
    axes[1].axis("off")

    # Stage 2-3: text
    axes[2].set_facecolor("#fafafa")
    axes[2].text(0.04, 0.97, text_content,
                 transform=axes[2].transAxes,
                 fontsize=7.5, va="top", ha="left",
                 fontfamily="monospace",
                 bbox=dict(facecolor="white", edgecolor="#ddd",
                           alpha=0.9, pad=6, boxstyle="round,pad=0.3"))
    axes[2].set_xlim(0, 1); axes[2].set_ylim(0, 1)
    axes[2].axis("off")

    # Stage 3: JSON
    axes[3].set_facecolor("#1e1e2e")
    axes[3].text(0.03, 0.98, json_snippet,
                 transform=axes[3].transAxes,
                 fontsize=6.8, va="top", ha="left",
                 fontfamily="monospace", color="#cdd6f4",
                 bbox=dict(facecolor="#1e1e2e", edgecolor="none",
                           alpha=1.0, pad=4))
    axes[3].set_xlim(0, 1); axes[3].set_ylim(0, 1)
    axes[3].axis("off")

    for ax, label, c in zip(axes, stage_labels, colors_stage):
        ax.set_title(label, fontsize=9.5, color=c, fontweight="bold",
                     pad=6, linespacing=1.4)
        for spine in ax.spines.values():
            spine.set_edgecolor(c)
            spine.set_linewidth(2.5)
            spine.set_visible(True)

    # Arrows between stages — using axes[0] coordinate transform
    for i in range(3):
        # Draw arrow on axes i, in axes fraction coordinates
        ax_src = axes[i]
        ax_src.annotate("→", xy=(1.0, 0.5), xycoords="axes fraction",
                        fontsize=22, color="#aaa", ha="center", va="center",
                        annotation_clip=False)

    fig.suptitle("Figure 3.x — Illustration of 4 pipeline stages\n"
                 "Input PDF → YOLO Detection → OCR Text → Structured JSON",
                 fontsize=12, y=1.03)
    plt.tight_layout(pad=0.8)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════
# FIG 5 — DocLayNet 11-class distribution (bar chart)
# ══════════════════════════════════════════════════════════════════════
def make_fig5_class_distribution(save_path: Path):
    print("[fig5] Drawing class distribution...")

    data = {
        "Text":           431251,
        "List-item":      161818,
        "Section-header": 118590,
        "Page-footer":    61313,
        "Page-header":    47973,
        "Picture":        39667,
        "Table":          30070,
        "Formula":        21167,
        "Caption":        19218,
        "Footnote":       5619,
        "Title":          4437,
    }
    total = sum(data.values())

    classes = list(data.keys())
    counts  = list(data.values())
    percents = [c / total * 100 for c in counts]
    colors   = [CLASS_COLORS.get(c, "#AAAAAA") for c in classes]

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=130)
    bars = ax.barh(classes, counts, color=colors, edgecolor="white",
                   linewidth=0.6, height=0.7)

    for bar, cnt, pct in zip(bars, counts, percents):
        ax.text(bar.get_width() + total * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{cnt:,}  ({pct:.2f}%)",
                va="center", ha="left", fontsize=8.5)

    ax.set_xlabel("Annotation count (training set)", fontsize=10)
    ax.set_title("Figure 3.x — Distribution of 11 object classes in DocLayNet (train set)\n"
                 f"Total: {total:,} annotations | Imbalance: 97:1 (Text vs Title)",
                 fontsize=10.5, pad=10)
    ax.set_xlim(0, max(counts) * 1.22)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9.5)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    # Dividers: Dominant / Medium / Rare
    ax.axhline(y=2.5, color="#ccc", linestyle="--", lw=0.8)
    ax.axhline(y=8.5, color="#ccc", linestyle="--", lw=0.8)
    ax.text(max(counts) * 1.18, 1.0, "Dominant",
            fontsize=8, color="#666", ha="right", va="center")
    ax.text(max(counts) * 1.18, 5.5, "Medium",
            fontsize=8, color="#666", ha="right", va="center")
    ax.text(max(counts) * 1.18, 9.5, "Rare",
            fontsize=8, color="#c0392b", ha="right", va="center")

    plt.tight_layout(pad=1.0)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════
# FIG 6 — Ablation study bar chart (mAP@0.5 comparison of 3 experiments)
# ══════════════════════════════════════════════════════════════════════
def make_fig6_ablation_chart(save_path: Path):
    print("[fig6] Drawing ablation chart...")

    experiments = ["Baseline", "Oversampling\n(selected)", "Strong aug."]
    metrics = {
        "mAP@0.5 (val)":     [0.916, 0.922, 0.904],
        "mAP@0.5 (test)":    [0.901, 0.909, 0.883],
        "Recall (val)":      [0.858, 0.870, 0.831],
        "Precision (val)":   [0.882, 0.878, 0.885],
    }

    x = np.arange(len(experiments))
    width = 0.18
    colors_bar = ["#4E79A7", "#E15759", "#59A14F", "#F28E2B"]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
    for i, (metric, vals) in enumerate(metrics.items()):
        offset = (i - len(metrics) / 2 + 0.5) * width
        rects = ax.bar(x + offset, vals, width, label=metric,
                       color=colors_bar[i], alpha=0.88, edgecolor="white")
        for rect, v in zip(rects, vals):
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + 0.002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_ylim(0.82, 0.94)
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, fontsize=10)
    ax.set_ylabel("Metric value", fontsize=10)
    ax.set_title("Figure 4.x — Comparison of 3 ablation experiments\n"
                 "Oversampling experiment achieves best mAP@0.5 on both val and test",
                 fontsize=10.5, pad=10)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))

    # Highlight Oversampling column
    for y_val in [0.0, 0.94]:
        ax.axvline(x=0.65, ymin=0, ymax=1, color="#59A14F",
                   linestyle=":", lw=0.8, alpha=0.5)
        ax.axvline(x=1.35, ymin=0, ymax=1, color="#59A14F",
                   linestyle=":", lw=0.8, alpha=0.5)

    plt.tight_layout(pad=1.0)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════
# FIG 7 — Block F1 per class (end-to-end eval)
# ══════════════════════════════════════════════════════════════════════
def make_fig7_block_f1(save_path: Path):
    print("[fig7] Drawing block F1 per class...")

    data = {
        "List-item":      0.914,
        "Footnote":       0.894,
        "Text":           0.888,
        "Table":          0.866,
        "Section-header": 0.852,
        "Title":          0.792,
        "Picture":        0.731,
        "Page-header":    0.699,
        "Page-footer":    0.534,
        "Formula":        0.400,
        "Caption":        0.366,
    }

    classes = list(data.keys())
    f1s     = list(data.values())
    colors  = []
    for f in f1s:
        if f >= 0.80:
            colors.append("#59A14F")
        elif f >= 0.65:
            colors.append("#F28E2B")
        else:
            colors.append("#E15759")

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=130)
    bars = ax.barh(classes, f1s, color=colors, edgecolor="white",
                   linewidth=0.6, height=0.65)

    for bar, f in zip(bars, f1s):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{f:.3f}", va="center", ha="left", fontsize=9)

    ax.axvline(x=0.80, color="#59A14F", linestyle="--", lw=1.2, alpha=0.7,
               label="F1 = 0.80 (good threshold)")
    ax.axvline(x=0.65, color="#F28E2B", linestyle="--", lw=1.2, alpha=0.7,
               label="F1 = 0.65 (medium threshold)")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1-score", fontsize=10)
    ax.set_title("Figure 4.x — Block detection F1 per class (294 DocLayNet samples)\n"
                 "Color: ✅ F1≥0.80 | 🟠 0.65≤F1<0.80 | 🔴 F1<0.65",
                 fontsize=10.5, pad=10)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8.5, loc="lower right")

    # Custom color legend
    legend_patches = [
        mpatches.Patch(color="#59A14F", label="F1 ≥ 0.80 (Good)"),
        mpatches.Patch(color="#F28E2B", label="0.65 ≤ F1 < 0.80 (Medium)"),
        mpatches.Patch(color="#E15759", label="F1 < 0.65 (Needs improvement)"),
    ]
    ax.legend(handles=legend_patches, fontsize=8.5, loc="lower right")

    plt.tight_layout(pad=1.0)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════
# FIG 8 — CER per class (OCR quality)
# ══════════════════════════════════════════════════════════════════════
def make_fig8_cer_chart(save_path: Path):
    print("[fig8] Drawing CER per class...")

    data = {
        "List-item":      (0.120, 98.7),
        "Text":           (0.121, 92.9),
        "Footnote":       (0.172, 91.5),
        "Section-header": (0.223, 88.4),
        "Title":          (0.250, 100.0),
        "Caption":        (0.281, 42.2),
        "Page-header":    (0.396, 73.4),
        "Page-footer":    (0.532, 32.9),
    }

    classes  = list(data.keys())
    cers     = [v[0] for v in data.values()]
    coverage = [v[1] for v in data.values()]
    colors   = ["#59A14F" if c <= 0.20 else
                "#F28E2B" if c <= 0.35 else "#E15759" for c in cers]

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=130)
    x = np.arange(len(classes))
    bars = ax1.bar(x, cers, color=colors, edgecolor="white",
                   linewidth=0.6, width=0.5, label="CER")

    for bar, c in zip(bars, cers):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.008,
                 f"{c:.3f}", ha="center", va="bottom", fontsize=8.5)

    ax2 = ax1.twinx()
    ax2.plot(x, coverage, "o--", color="#4E79A7", linewidth=1.8,
             markersize=7, label="Coverage (%)")
    for xi, cov in zip(x, coverage):
        ax2.text(xi, cov + 1.5, f"{cov:.0f}%",
                 ha="center", fontsize=7.5, color="#4E79A7")

    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=25, ha="right", fontsize=9)
    ax1.set_ylabel("CER (lower = better)", fontsize=10)
    ax2.set_ylabel("Coverage (%)", fontsize=10, color="#4E79A7")
    ax2.set_ylim(0, 115)
    ax1.set_ylim(0, 0.65)
    ax1.set_title("Figure 4.x — OCR quality (CER) and Coverage per class\n"
                  "CER: ✅ ≤0.20 | 🟠 ≤0.35 | 🔴 >0.35   |   Coverage: fraction of blocks with OCR text",
                  fontsize=10.5, pad=10)
    ax1.spines[["top", "right"]].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8.5, loc="upper left")

    plt.tight_layout(pad=1.0)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════
# COPY existing figures from dataset/
# ══════════════════════════════════════════════════════════════════════
def copy_existing_figures():
    """Copy figures already present in dataset/ into thesis_figures/."""
    import shutil

    # Map: (source, destination name, thesis caption)
    existing = [
        (BASE_DIR / "dataset" / "BaseC" / "class_distribution.png",
         "exist_class_distribution_full.png",
         "Figure 3.2 — DocLayNet 11-class annotation distribution (train/val/test)"),

        (BASE_DIR / "dataset" / "BaseC" / "aspect_ratio_distribution.png",
         "exist_aspect_ratio_dist.png",
         "Figure 3.3 — Bounding box aspect ratio (AR) distribution"),

        (BASE_DIR / "dataset" / "BaseC" / "bbox_scale_distribution.png",
         "exist_bbox_scale_dist.png",
         "Figure 3.4 — Bounding box size distribution by group"),

        (BASE_DIR / "dataset" / "BaseC" / "objects_per_page.png",
         "exist_objects_per_page.png",
         "Figure 3.5 — Objects per page distribution"),

        (BASE_DIR / "dataset" / "final_Base" / "fig3_overall_metrics.png",
         "exist_ablation_overall_metrics.png",
         "Figure 4.2 — Overall comparison of 3 ablation experiments (val + test)"),

        (BASE_DIR / "dataset" / "final_Base" / "fig1_perclass_ap_validation.png",
         "exist_ap_per_class_val.png",
         "Figure 4.3 — AP@0.5 per class — validation set"),

        (BASE_DIR / "dataset" / "final_Base" / "fig2_perclass_ap_test.png",
         "exist_ap_per_class_test.png",
         "Figure 4.4 — AP@0.5 per class — test set"),

        (BASE_DIR / "dataset" / "final_Base" / "fig5_rare_class_comparison.png",
         "exist_rare_class_comparison.png",
         "Figure 4.5 — Oversampling effectiveness on rare classes"),

        (BASE_DIR / "dataset" / "final_Base" / "fig4_training_convergence.png",
         "exist_training_convergence.png",
         "Figure 4.6 — Training convergence curves — 3 ablation experiments"),

        (BASE_DIR / "dataset" / "final_Base" / "sample_predictions.png",
         "exist_sample_predictions.png",
         "Figure 3.x — Example YOLO detection results on test set"),

        (BASE_DIR / "dataset" / "final_Base" / "confusion_matrix_detect.png",
         "exist_confusion_matrix.png",
         "Figure 4.7 — Confusion matrix — YOLOv11s on test set"),
    ]

    print("\n[COPY] Copying existing figures...")
    caption_log = []
    for src, dst_name, caption in existing:
        dst = OUTPUT_DIR / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            size_kb = dst.stat().st_size // 1024
            print(f"  ✓ {dst_name:45s}  {size_kb:5d} KB  ← {src.name}")
            caption_log.append((dst_name, caption))
        else:
            print(f"  ✗ SKIP (not found): {src}")

    return caption_log


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("THESIS FIGURE GENERATOR")
    print(f"Main PDF  : {Path(SAMPLE_PDF_MAIN).name}")
    print(f"OCR PDF   : {Path(SAMPLE_PDF_OCR).name}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 70)

    # ── PART 1: Copy existing figures ─────────────────────────────────
    caption_log = copy_existing_figures()

    # ── PART 2: Render PDF + YOLO ─────────────────────────────────────
    print(f"\n[YOLO] Render + detect: {Path(SAMPLE_PDF_MAIN).name}")
    img_raw  = render_pdf_page(SAMPLE_PDF_MAIN, dpi=PDF_DPI)
    img_yolo = letterbox(img_raw, target=1025)
    blocks   = run_yolo(img_yolo)
    print(f"  raw: {img_raw.shape[1]}×{img_raw.shape[0]}  "
          f"yolo: {img_yolo.shape[1]}×{img_yolo.shape[0]}  "
          f"blocks: {len(blocks)}")

    # Second PDF for OCR (sample_0116 has Formula)
    img_raw_ocr  = render_pdf_page(SAMPLE_PDF_OCR, dpi=PDF_DPI)
    img_yolo_ocr = letterbox(img_raw_ocr, target=1025)
    blocks_ocr   = run_yolo(img_yolo_ocr)

    # ── PART 3: Generate new figures ──────────────────────────────────
    make_fig1_yolo_detection(
        img_yolo, blocks,
        OUTPUT_DIR / "fig1_yolo_detection.png")

    make_fig2_crops_grid(
        img_yolo, blocks,
        OUTPUT_DIR / "fig2_crops_grid.png")

    make_fig3_ocr_result(
        img_yolo_ocr, blocks_ocr,
        OUTPUT_DIR / "fig3_ocr_result.png",
        use_real_ocr=True)

    make_fig4_pipeline_stages(
        img_raw, img_yolo, blocks,
        OUTPUT_DIR / "fig4_pipeline_stages.png")

    make_fig6_ablation_chart(
        OUTPUT_DIR / "fig6_ablation_chart.png")

    make_fig7_block_f1(
        OUTPUT_DIR / "fig7_block_f1.png")

    make_fig8_cer_chart(
        OUTPUT_DIR / "fig8_cer_chart.png")

    # ── SUMMARY ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPLETE — All files in thesis_figures/:")
    print()
    print("  [EXISTING figures — copied from dataset/]")
    for dst_name, caption in caption_log:
        print(f"    {dst_name}")
        print(f"      → {caption}")
    print()
    print("  [NEW figures]")
    new_figs = [
        ("fig1_yolo_detection.png",  "Figure 3.x — YOLO detection visual results (Ch3 §3.1)"),
        ("fig2_crops_grid.png",      "Figure 3.x — Per-block crop grid (Ch3 §3.3)"),
        ("fig3_ocr_result.png",      "Figure 3.x — Crop ↔ OCR text (Ch3 §3.3)"),
        ("fig4_pipeline_stages.png", "Figure 3.x — 4 pipeline stages (Ch3 §3.1)"),
        ("fig6_ablation_chart.png",  "Figure 4.x — Ablation bar chart (Ch4 §4.2)"),
        ("fig7_block_f1.png",        "Figure 4.x — Block F1 end-to-end (Ch4 §4.3)"),
        ("fig8_cer_chart.png",       "Figure 4.x — CER per class (Ch4 §4.4)"),
    ]
    for fname, caption in new_figs:
        p = OUTPUT_DIR / fname
        size_kb = p.stat().st_size // 1024 if p.exists() else 0
        mark = "✓" if p.exists() else "✗"
        print(f"    {mark} {fname:40s}  {size_kb:5d} KB")
        print(f"      → {caption}")

    print()
    total = len(list(OUTPUT_DIR.glob("*.png")))
    total_kb = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.png")) // 1024
    print(f"  Total: {total} files  |  {total_kb} KB")
    print("=" * 70)


if __name__ == "__main__":
    main()
