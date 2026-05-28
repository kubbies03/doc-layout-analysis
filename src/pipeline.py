"""
Benchmark script — Latency, Cost, and sample outputs for the YOLO + docTR + LLM pipeline.

Measures:
  - Per-module latency: preprocess, YOLO, docTR OCR, XY-Cut, LLM API call
  - Total pipeline latency (ms/page)
  - Gemini API cost: input/output tokens, USD/page
  - Saves the 5 best output JSONs as thesis examples

Usage:
  python benchmark_pipeline.py [--limit 50] [--seed 42] [--save-outputs 5]

Output:
  benchmark_results/benchmark_summary.json
  benchmark_results/benchmark_per_sample.csv
  benchmark_results/sample_outputs/*.json
"""

import os, io, sys, json, csv, time, re, argparse, traceback
import random as _random
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field

# ── Config ────────────────────────────────────────────────────────────────────
STAGING_DIR   = Path(r"C:\Users\HP\Desktop\Kubbies\KhoaLuan\GT\staging")
RESULTS_DIR   = Path(r"C:\Users\HP\Desktop\Kubbies\KhoaLuan\benchmark_results")
YOLO_WEIGHTS  = r"C:\Users\HP\Desktop\Kubbies\KhoaLuan\dataset\BaseC\best.pt"
POPPLER_PATH  = r"C:\Users\HP\Desktop\Kubbies\KhoaLuan\poppler\Library\bin"

PDF_DPI    = 150
GT_IMGSZ   = 1025
MIN_AREA   = 800.0
YOLO_CONF  = 0.20
YOLO_IOU   = 0.60
YOLO_IMGSZ = 640
CLASS_CONF_OVERRIDE = {
    "Picture": 0.50, "Section-header": 0.40, "Caption": 0.40, "Table": 0.12,
    "Formula": 0.45, "Footnote": 0.35,
}
CLASS_NAMES = [
    "Caption", "Footnote", "Formula", "List-item", "Page-footer",
    "Page-header", "Picture", "Section-header", "Table", "Text", "Title"
]
TEXT_CLASSES  = {"Caption", "Footnote", "List-item", "Page-footer",
                 "Page-header", "Section-header", "Text", "Title"}
TABLE_CLASSES = {"Table"}

GEMINI_MODEL   = "gemini-2.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBIyAcp-f5BW6cAUbJoxFr9Wj71gIuVzKE")
MAX_RETRIES    = 3

INPUT_PRICE  = 0.30 / 1_000_000
OUTPUT_PRICE = 2.50 / 1_000_000

# ── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--limit",        type=int, default=50)
ap.add_argument("--seed",         type=int, default=42)
ap.add_argument("--save-outputs", type=int, default=5, dest="save_outputs")
args = ap.parse_args()

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "sample_outputs").mkdir(exist_ok=True)

# ── Model singletons ──────────────────────────────────────────────────────────
_yolo   = None
_doctr  = None
_client = None

def get_yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        _yolo = YOLO(YOLO_WEIGHTS)
        print("  [load] YOLO ready.")
    return _yolo

def unload_yolo():
    global _yolo
    import torch
    if _yolo is not None:
        del _yolo
        _yolo = None
        torch.cuda.empty_cache()
        import gc; gc.collect()
        print("  [unload] YOLO freed.")

def get_doctr():
    global _doctr
    if _doctr is None:
        os.environ["USE_TORCH"] = "1"
        from doctr.models import ocr_predictor
        import torch
        _doctr = ocr_predictor(pretrained=True)
        if torch.cuda.is_available():
            _doctr = _doctr.cuda()
        print(f"  [load] docTR ready (cuda={torch.cuda.is_available()}).")
    return _doctr

def get_client():
    global _client
    if _client is None:
        import google.genai as genai
        _client = genai.Client(api_key=GEMINI_API_KEY)
        print("  [load] Gemini client ready.")
    return _client

# ── Image utilities ───────────────────────────────────────────────────────────
def pdf_to_pil(pdf_path):
    from pdf2image import convert_from_path
    return convert_from_path(str(pdf_path), dpi=PDF_DPI, poppler_path=POPPLER_PATH)

def letterbox(img, target=GT_IMGSZ):
    import cv2, numpy as np
    h, w = img.shape[:2]
    scale = target / max(h, w)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((target, target, 3), 255, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    return canvas, nh, nw

# ── Stage 1: YOLO detection ───────────────────────────────────────────────────
NMS_IOU_THR = 0.35  # IoU threshold for cross-class NMS post-processing

def _iou(a, b):
    """IoU between two bboxes in [x1,y1,w,h] format."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0

def _nms_blocks(blocks, iou_thr=NMS_IOU_THR):
    """Cross-class NMS: removes duplicate blocks across classes, keeps highest conf."""
    blocks = sorted(blocks, key=lambda b: b["conf"], reverse=True)
    kept = []
    for b in blocks:
        suppress = False
        for k in kept:
            if _iou(b["bbox_coco"], k["bbox_coco"]) >= iou_thr:
                suppress = True
                break
        if not suppress:
            kept.append(b)
    return kept

def detect_blocks(canvas):
    model = get_yolo()
    res = model.predict(
        source=canvas, conf=YOLO_CONF, iou=YOLO_IOU,
        imgsz=YOLO_IMGSZ, max_det=300, device="0", verbose=False
    )[0]
    blocks = []
    for box in res.boxes:
        cls_id = int(box.cls[0])
        label  = CLASS_NAMES[cls_id]
        conf   = float(box.conf[0])
        thr    = CLASS_CONF_OVERRIDE.get(label, YOLO_CONF)
        if conf < thr:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        w, h = x2 - x1, y2 - y1
        if w * h < MIN_AREA:
            continue
        blocks.append({
            "label_name": label,
            "bbox_coco": [x1, y1, w, h],
            "conf": conf,
            "text": "",
        })
    return _nms_blocks(blocks)

# ── Stage 2: docTR OCR ────────────────────────────────────────────────────────
OCR_PADDING   = 6    # px padding around crop before OCR
OCR_MIN_SIDE  = 32   # upscale if either side is smaller than this (px)
OCR_TARGET_H  = 96   # target height (px) — scale up if crop is shorter
OCR_MAX_SCALE = 3.0  # upper scale limit to avoid wasting VRAM on large blocks

def ocr_crops_batch(img_bgr, bboxes):
    """OCR all text crop bboxes in one docTR forward pass."""
    import numpy as np
    from PIL import Image as PILImage
    from doctr.io import DocumentFile
    import cv2

    H, W = img_bgr.shape[:2]

    def crop_png(x, y, w, h):
        # Add padding and clamp to image boundary
        x1 = max(0, int(x) - OCR_PADDING)
        y1 = max(0, int(y) - OCR_PADDING)
        x2 = min(W, int(x + w) + OCR_PADDING)
        y2 = min(H, int(y + h) + OCR_PADDING)
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        # Adaptive scale: reach OCR_TARGET_H but do not exceed OCR_MAX_SCALE
        ch, cw = crop.shape[:2]
        min_side_scale = max(OCR_MIN_SIDE / cw, OCR_MIN_SIDE / ch)
        target_scale   = OCR_TARGET_H / ch if ch < OCR_TARGET_H else 1.0
        scale = min(max(min_side_scale, target_scale), OCR_MAX_SCALE)
        crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)),
                          interpolation=cv2.INTER_CUBIC)
        pil = PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()

    png_list = [crop_png(*bb) for bb in bboxes]
    valid = [(i, p) for i, p in enumerate(png_list) if p]
    texts = [""] * len(bboxes)
    if not valid:
        return texts

    model = get_doctr()
    doc = DocumentFile.from_images([p for _, p in valid])
    result = model(doc)
    for page_i, (src_i, _) in enumerate(valid):
        words = []
        for blk in result.pages[page_i].blocks:
            for line in blk.lines:
                for word in line.words:
                    if word.value.strip():
                        words.append(word.value)
        texts[src_i] = " ".join(words)
    return texts

# ── Stage 3a: XY-Cut reading order ───────────────────────────────────────────
@dataclass
class BBox:
    x1: float; y1: float; x2: float; y2: float

    @property
    def cx(self): return (self.x1 + self.x2) / 2
    @property
    def cy(self): return (self.y1 + self.y2) / 2

@dataclass
class LayoutBlock:
    block_id:   int
    class_name: str
    conf:       float
    bbox:       BBox
    reading_order: int = -1
    raw_text:   str = ""

def _is_full_width(b, img_width: int) -> bool:
    """Block is full-width if it satisfies ONE of two conditions:
    1. True span: starts_left AND ends_right (x1<=15%, x2>=85%) — regardless of width%.
    2. Clearly wide: width >= 45% AND starts_left (x1<=15%) — catches Title/Table/Caption
       but excludes the widest single-column blocks (~26-27% in standard 2-col layout).
    """
    block_w     = b.bbox.x2 - b.bbox.x1
    width_ratio = block_w / img_width
    starts_left = b.bbox.x1 <= img_width * 0.15
    ends_right  = b.bbox.x2 >= img_width * 0.85
    return starts_left and (ends_right or width_ratio >= 0.45)


def _find_column_boundary(blocks: list, img_width: int) -> float:
    """Find column boundary via X-projection in the 25%-75% horizontal zone.
    Finds all gaps (regions with no block coverage) and picks the one closest to mid.
    Falls back to mid if no gap is found.
    """
    mid = img_width / 2
    lo  = int(img_width * 0.25)
    hi  = int(img_width * 0.75)

    # Coverage array: 1 if at least one block covers pixel x
    coverage = [0] * img_width
    for b in blocks:
        x1 = max(0, int(b.bbox.x1))
        x2 = min(img_width, int(b.bbox.x2))
        for x in range(x1, x2):
            coverage[x] = 1

    # Collect all gaps in [lo, hi]
    gaps = []
    gap_start = None
    for x in range(lo, hi):
        if coverage[x] == 0:
            if gap_start is None:
                gap_start = x
        else:
            if gap_start is not None:
                gaps.append((gap_start, x))
                gap_start = None
    if gap_start is not None:
        gaps.append((gap_start, hi))

    if not gaps:
        return mid

    # Pick the gap closest to mid (prefer center-of-page gap)
    best_gap_x = min(gaps, key=lambda g: abs((g[0] + g[1]) / 2 - mid))
    return (best_gap_x[0] + best_gap_x[1]) / 2


def assign_reading_order(blocks: list, img_width: int) -> list:
    """Custom XY-Cut — 4 steps as described in thesis section 3.5.2.

    Step 1: Detect column count using cx vs actual column boundary, min/max ratio > 0.35.
    Step 2a (2-col): sort left column (y1,x1), sort right column (y1,x1),
                     read entire left column then right column.
    Step 2b (1-col): row-grouping by vertical overlap, within each row sort by x1.
    Step 3: Assign reading_order.
    Extra: separate full-width blocks and re-insert by y1.
    """
    if not blocks:
        return blocks

    mid = img_width / 2

    # Separate full-width blocks (page-spanning) from the columnar flow
    full_width = [b for b in blocks if _is_full_width(b, img_width)]
    columnar   = [b for b in blocks if not _is_full_width(b, img_width)]

    # Step 1 — Detect column count using actual column boundary
    col_boundary = _find_column_boundary(columnar, img_width)
    left_col  = [b for b in columnar if b.bbox.cx <  col_boundary]
    right_col = [b for b in columnar if b.bbox.cx >= col_boundary]
    if left_col and right_col:
        ratio = min(len(left_col), len(right_col)) / max(len(left_col), len(right_col))
        # Use both block-count ratio AND real gap: 2-col only when gap is clear
        # (col_boundary differs from mid by >2%) AND ratio is high enough (>=0.25, i.e. 1:4)
        has_real_gap = abs(col_boundary - img_width / 2) > img_width * 0.02
        two_col = has_real_gap and ratio >= 0.25
    else:
        two_col = False

    if two_col:
        # Step 2a — 2-col: read entire left column then right column
        left_sorted  = sorted(left_col,  key=lambda b: (b.bbox.y1, b.bbox.x1))
        right_sorted = sorted(right_col, key=lambda b: (b.bbox.y1, b.bbox.x1))
        col_blocks = left_sorted + right_sorted
    else:
        # Step 2b — 1-col: row-grouping by vertical overlap, within row sort by x1
        sorted_y = sorted(columnar, key=lambda b: b.bbox.y1)
        rows: list = []
        for b in sorted_y:
            placed = False
            for row in rows:
                row_y1 = min(r.bbox.y1 for r in row)
                row_y2 = max(r.bbox.y2 for r in row)
                if b.bbox.y1 < row_y2 and b.bbox.y2 > row_y1:
                    row.append(b)
                    placed = True
                    break
            if not placed:
                rows.append([b])
        col_blocks = []
        for row in rows:
            col_blocks.extend(sorted(row, key=lambda b: b.bbox.x1))

    # Step 3 — Insert full-width blocks at correct y position then assign reading_order
    fw_sorted = sorted(full_width, key=lambda b: b.bbox.y1)
    ordered: list = []
    fi = ci = 0
    while fi < len(fw_sorted) and ci < len(col_blocks):
        if fw_sorted[fi].bbox.y1 <= col_blocks[ci].bbox.y1:
            ordered.append(fw_sorted[fi]); fi += 1
        else:
            ordered.append(col_blocks[ci]); ci += 1
    ordered.extend(fw_sorted[fi:])
    ordered.extend(col_blocks[ci:])

    for i, b in enumerate(ordered):
        b.reading_order = i
    return ordered

# ── Stage 3b: LLM call with usage tracking ───────────────────────────────────

# ── Gemini prompt ─────────────────────────────────────────────────────────────
COMBINED_SCHEMA_PROMPT = """\
You are a structured document analysis assistant.
Below are text blocks extracted from a single document page (in reading order).
{layout_note}{table_note}
Text blocks:
---
{blocks_text}
---

CRITICAL RULES:
1. Every [Section-header] block ALWAYS starts a NEW section, regardless of whether it has a number or not.
2. [Footnote] blocks belong to the section that physically contains them — copy them VERBATIM into footnotes[] of that section, NOT into paragraphs[].
3. [Page-header] and [Page-footer] blocks are page-level metadata — copy them VERBATIM into page_header[] and page_footer[] at the top level, NOT inside any section.
4. Copy each [Text], [Title], [List-item], [Caption] block VERBATIM into paragraphs[]. Do NOT summarize, paraphrase, shorten, or omit any block. Only the "summary" field should be a summary.

Return a single JSON object with this exact structure:
{{
  "structured_json": {{
    "title": "<string | null>",
    "page_header": ["<copy each Page-header block verbatim>"],
    "page_footer": ["<copy each Page-footer block verbatim>"],
    "sections": [
      {{
        "heading": "<string | null — copy verbatim from Section-header block>",
        "paragraphs": ["<copy each Text/List-item/Caption block verbatim, one string per block>"],
        "footnotes": ["<copy each Footnote block verbatim, one string per footnote>"],
        "tables": [{{"headers": [], "rows": [[]]}}],
        "formulas": ["<LaTeX>"],
        "figures": ["<description>"]
      }}
    ]
  }},
  "summary": "<3-5 sentence summary of the page>",
  "keywords": ["kw1", "kw2"],
  "page_type": "<title_page|table_of_contents|body|figure_page|reference|other>",
  "language": "<ISO 639-1>"
}}

Return ONLY valid JSON. No markdown fences. No explanation.\
"""

TABLE_NOTE_TEMPLATE = (
    "The {n} image(s) appended after this prompt are table crops from the page (in order). "
    "Extract each table into the tables[] field of the appropriate section.\n"
)

def _cv2_to_pil(img):
    from PIL import Image as PILImage
    import cv2
    return PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def _call_gemini_with_usage(prompt: str, images: list = None) -> tuple[str, dict]:
    """Gemini API call with usage tracking."""
    from google.genai import types as genai_types
    client = get_client()
    contents = [prompt]
    if images:
        contents.extend(_cv2_to_pil(img) for img in images)
    cfg = genai_types.GenerateContentConfig(
        thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
    )
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL, contents=contents, config=cfg,
            )
            usage = resp.usage_metadata
            usage_dict = {
                "input_tokens":    getattr(usage, "prompt_token_count", 0) or 0,
                "output_tokens":   getattr(usage, "candidates_token_count", 0) or 0,
                "thinking_tokens": getattr(usage, "thoughts_token_count", 0) or 0,
            }
            return resp.text.strip(), usage_dict
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Gemini failed after {MAX_RETRIES} attempts: {last_err}")


def assemble_and_summarize(blocks: list, table_crops: list = None, img_width: int = 0):
    """Returns: (structured_json, is_valid, usage_dict, raw_text)"""
    empty_usage = {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}

    def _build_blocks_text(blk_list, width):
        """Build text with 2-column separator when layout is 2-col."""
        ordered = sorted(blk_list, key=lambda x: x.reading_order)
        if not ordered:
            return "", False

        two_col = False
        separator_idx = -1
        if width > 0:
            full_w = [b for b in ordered if _is_full_width(b, width)]
            col_b  = [b for b in ordered if not _is_full_width(b, width)]
            if col_b:
                col_bnd = _find_column_boundary(col_b, width)
                left_col  = [b for b in col_b if b.bbox.cx <  col_bnd]
                right_col = [b for b in col_b if b.bbox.cx >= col_bnd]
                if left_col and right_col:
                    ratio = min(len(left_col), len(right_col)) / max(len(left_col), len(right_col))
                    two_col = ratio > 0.35
                if two_col:
                    right_ids = {id(b) for b in right_col}
                    for i, b in enumerate(ordered):
                        if id(b) in right_ids:
                            separator_idx = i
                            break

        parts = []
        for i, b in enumerate(ordered):
            if two_col and i == separator_idx:
                parts.append("--- RIGHT COLUMN START ---")
            parts.append(f"[{b.class_name}] {b.raw_text}")
        return "\n".join(parts), two_col

    txt, two_col = _build_blocks_text(blocks, img_width)
    crops = [c for c in (table_crops or []) if c is not None and c.size > 0]

    if not txt.strip() and not crops:
        return {}, True, empty_usage, ""

    layout_note = (
        "LAYOUT NOTE: This page has a 2-COLUMN layout. Blocks are listed left column first, "
        "then right column (marked by '--- RIGHT COLUMN START ---'). "
        "Each [Section-header] starts a new section. Footnotes belong to the preceding section.\n"
        if two_col else ""
    )
    table_note = TABLE_NOTE_TEMPLATE.format(n=len(crops)) if crops else ""
    prompt = COMBINED_SCHEMA_PROMPT.format(blocks_text=txt, table_note=table_note, layout_note=layout_note)
    for attempt in range(MAX_RETRIES):
        try:
            raw, usage = _call_gemini_with_usage(prompt, images=crops if crops else None)
            cleaned = re.sub(r"^```[a-z]*\n?", "", raw)
            cleaned = re.sub(r"```$", "", cleaned).strip()
            data = json.loads(cleaned)
            is_valid = isinstance(data.get("structured_json", {}).get("sections"), list)
            return data, is_valid, usage, raw
        except Exception as e:
            print(f"  [LLM-Gemini] attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2.0)
    return {"error": "parse_failed"}, False, empty_usage, ""

# ── Main benchmark loop ───────────────────────────────────────────────────────
def collect_stems() -> list[str]:
    """
    Select top-50 stems by combining 2 sources:
      - eval_results/per_sample.csv      : F1 detection, CER OCR
      - eval_results_llm/per_sample_llm.csv : schema_ok, content_recall

    Composite score = f1_detection * 0.4 + (1 - cer_text) * 0.2 + schema_ok * 0.2 + content_recall * 0.2

    Stem formats:
      - LLM stems:  'sample_0029_612...' (long)
      - DET stems:  '0029_612'           (short, 7 chars)
      - PDF stems:  'sample_0029_612...' (long, matches LLM)
      - DET→PDF mapping: DET stem is a substring of PDF stem
    """
    import csv as _csv

    llm_csv = Path(r"C:\Users\HP\Desktop\Kubbies\KhoaLuan\eval_results_llm\per_sample_llm.csv")
    det_csv = Path(r"C:\Users\HP\Desktop\Kubbies\KhoaLuan\eval_results\per_sample.csv")

    # Stem formats:
    #   PDF files (staging): 'sample_0029_61253fc219f35687e9d3ff9...' (76 chars)
    #   LLM eval stems:      'sample_0029_61253fc2'                   (20 chars)
    #   DET eval stems:      '0029_612'                               (8 chars)
    #
    # Mapping: pdf_stem.startswith(llm_stem) and pdf_stem[7:].startswith(det_stem)
    pdf_stems = {p.stem for p in STAGING_DIR.glob("*.pdf")}

    # llm_short → full_pdf_stem
    llm_to_pdf: dict[str, str] = {}
    # det_short  → full_pdf_stem
    det_to_pdf: dict[str, str] = {}
    for pstem in pdf_stems:
        after = pstem[len("sample_"):]          # '0029_61253fc2...'
        llm_to_pdf[pstem[:20]] = pstem          # key = first 20 chars = LLM stem format
        det_to_pdf[after[:8]]  = pstem          # key = first 8 chars of after = DET stem format

    # Load LLM results: short stem → metrics, map to full PDF stem
    llm_data: dict[str, dict] = {}   # full_pdf_stem → llm row
    if llm_csv.exists():
        with open(llm_csv, encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                full = llm_to_pdf.get(row["stem"])
                if full:
                    llm_data[full] = row

    # Load DET results: short stem → metrics, map to full PDF stem
    det_data: dict[str, dict] = {}   # full_pdf_stem → det row
    if det_csv.exists():
        with open(det_csv, encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                full = det_to_pdf.get(row["stem"])
                if full:
                    det_data[full] = row

    # Score each available PDF stem
    scored = []
    for pstem in pdf_stems:
        llm = llm_data.get(pstem, {})
        det = det_data.get(pstem, {})

        # Detection F1 (overall)
        try:    f1_det = float(det.get("f1") or 0)
        except: f1_det = 0.0

        # OCR CER on text (lower = better → convert to quality)
        try:    cer_text = float(det.get("cer_text") or 1)
        except: cer_text = 1.0
        ocr_quality = 1.0 - min(cer_text, 1.0)

        # LLM schema + recall
        schema_ok = 1.0 if llm.get("schema_ok") == "1" else 0.0
        try:    recall = float(llm.get("content_recall") or 0)
        except: recall = 0.0

        has_error = bool(llm.get("error", "").strip())

        # Skip samples with LLM error or schema failure
        if has_error or schema_ok == 0.0:
            continue

        # Composite score (weights tuned to prioritize complete pipeline)
        score = (f1_det * 0.4 + ocr_quality * 0.2 + schema_ok * 0.2 + recall * 0.2)
        scored.append((score, f1_det, recall, pstem))

    if scored:
        scored.sort(key=lambda x: -x[0])
        top = scored[:args.limit]
        stems = [x[3] for x in top]
        avg_f1     = sum(x[1] for x in top) / len(top)
        avg_recall = sum(x[2] for x in top) / len(top)
        print(f"  [collect] Top-{len(stems)} stems (combined DET+LLM score) "
              f"avg_f1={avg_f1:.3f}  avg_recall={avg_recall:.3f}")
        return stems

    # Fallback if no eval data available
    print("  [collect] Fallback: random shuffle")
    stems = sorted(pdf_stems)
    rng = _random.Random(args.seed)
    rng.shuffle(stems)
    return stems[:args.limit]

def run_benchmark():
    import cv2
    import numpy as np

    stems = collect_stems()
    print(f"\n{'='*60}")
    print(f"  Benchmark: {len(stems)} samples | model={GEMINI_MODEL}")
    print(f"{'='*60}\n")

    # Pre-load models, measure load time separately
    print("[Model loading]")
    t0 = time.perf_counter()
    get_yolo()
    t_yolo_load = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    get_doctr()
    t_doctr_load = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    get_client()
    t_llm_load = (time.perf_counter() - t0) * 1000
    print(f"  YOLO load:   {t_yolo_load:.0f} ms")
    print(f"  docTR load:  {t_doctr_load:.0f} ms")
    print(f"  Gemini init: {t_llm_load:.0f} ms\n")

    # CSV writer
    csv_path = RESULTS_DIR / "benchmark_per_sample.csv"
    csv_fields = [
        "stem",
        "t_preprocess_ms", "t_yolo_ms", "t_ocr_ms", "t_xycut_ms",
        "t_llm_ms", "t_total_ms",
        "n_blocks", "n_text_blocks", "n_table_crops",
        "input_tokens", "output_tokens", "thinking_tokens", "cost_usd",
        "schema_ok", "n_sections",
    ]
    csv_file  = open(csv_path, "w", newline="", encoding="utf-8")
    writer    = csv.DictWriter(csv_file, fieldnames=csv_fields)
    writer.writeheader()

    rows = []
    saved_outputs = []

    for idx, stem in enumerate(stems):
        pdf_path = STAGING_DIR / f"{stem}.pdf"
        if not pdf_path.exists():
            print(f"  [{idx+1}/{len(stems)}] SKIP {stem} — PDF not found")
            continue

        print(f"  [{idx+1:>2}/{len(stems)}] {stem}", end="", flush=True)

        try:
            t_page_start = time.perf_counter()

            # ── Preprocess ─────────────────────────────────────────────────
            t0 = time.perf_counter()
            pages = pdf_to_pil(pdf_path)
            pil = pages[0]
            import numpy as np
            canvas_rgb = np.array(pil)
            canvas_bgr = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR)
            canvas, actual_h, actual_w = letterbox(canvas_bgr)
            t_preprocess = (time.perf_counter() - t0) * 1000

            # ── YOLO ───────────────────────────────────────────────────────
            t0 = time.perf_counter()
            raw_blocks = detect_blocks(canvas)
            t_yolo = (time.perf_counter() - t0) * 1000

            n_blocks = len(raw_blocks)

            # ── docTR OCR ──────────────────────────────────────────────────
            text_bbs  = [b["bbox_coco"] for b in raw_blocks if b["label_name"] in TEXT_CLASSES]
            text_idxs = [i for i, b in enumerate(raw_blocks) if b["label_name"] in TEXT_CLASSES]
            table_idxs = [i for i, b in enumerate(raw_blocks) if b["label_name"] in TABLE_CLASSES]

            t0 = time.perf_counter()
            if text_bbs:
                ocr_texts = ocr_crops_batch(canvas, text_bbs)
                for ti, src_i in enumerate(text_idxs):
                    raw_blocks[src_i]["text"] = ocr_texts[ti]
            t_ocr = (time.perf_counter() - t0) * 1000

            n_text_blocks = len(text_bbs)

            # Table crops for LLM
            table_crops = []
            for ti in table_idxs:
                x, y, w, h = [int(v) for v in raw_blocks[ti]["bbox_coco"]]
                crop = canvas[y:y+h, x:x+w]
                if crop.size > 0:
                    table_crops.append(crop)
            n_table_crops = len(table_crops)

            # ── XY-Cut ─────────────────────────────────────────────────────
            t0 = time.perf_counter()
            layout_blocks = []
            for i, b in enumerate(raw_blocks):
                x, y, w, h = b["bbox_coco"]
                lb = LayoutBlock(
                    block_id=i,
                    class_name=b["label_name"],
                    conf=b["conf"],
                    bbox=BBox(x1=x, y1=y, x2=x+w, y2=y+h),
                    raw_text=b["text"],
                )
                layout_blocks.append(lb)
            ordered = assign_reading_order(layout_blocks, actual_w)
            t_xycut = (time.perf_counter() - t0) * 1000

            # ── LLM ────────────────────────────────────────────────────────
            t0 = time.perf_counter()
            data, is_valid, usage, raw_llm = assemble_and_summarize(ordered, table_crops)
            t_llm = (time.perf_counter() - t0) * 1000

            t_total = (time.perf_counter() - t_page_start) * 1000

            # Cost
            cost_usd = (
                usage["input_tokens"]  * INPUT_PRICE +
                usage["output_tokens"] * OUTPUT_PRICE
            )

            n_sections = len(data.get("structured_json", {}).get("sections", []))

            print(f"  → total={t_total:.0f}ms  "
                  f"yolo={t_yolo:.0f}ms  ocr={t_ocr:.0f}ms  "
                  f"llm={t_llm:.0f}ms  "
                  f"tok={usage['input_tokens']}+{usage['output_tokens']}  "
                  f"${cost_usd:.5f}")

            row = {
                "stem":            stem,
                "t_preprocess_ms": round(t_preprocess, 1),
                "t_yolo_ms":       round(t_yolo, 1),
                "t_ocr_ms":        round(t_ocr, 1),
                "t_xycut_ms":      round(t_xycut, 1),
                "t_llm_ms":        round(t_llm, 1),
                "t_total_ms":      round(t_total, 1),
                "n_blocks":        n_blocks,
                "n_text_blocks":   n_text_blocks,
                "n_table_crops":   n_table_crops,
                "input_tokens":    usage["input_tokens"],
                "output_tokens":   usage["output_tokens"],
                "thinking_tokens": usage["thinking_tokens"],
                "cost_usd":        round(cost_usd, 6),
                "schema_ok":       int(is_valid),
                "n_sections":      n_sections,
            }
            writer.writerow(row)
            csv_file.flush()
            rows.append(row)

            # Save output if qualifies
            if (is_valid and n_sections >= 2
                    and any(t.get("tables") for s in data.get("structured_json", {}).get("sections", []) for t in [s])
                    and len(saved_outputs) < args.save_outputs):
                out = {
                    "stem":           stem,
                    "pipeline_output": data,
                    "usage":          usage,
                    "cost_usd":       cost_usd,
                    "latency_ms": {
                        "preprocess": round(t_preprocess, 1),
                        "yolo":       round(t_yolo, 1),
                        "ocr":        round(t_ocr, 1),
                        "xycut":      round(t_xycut, 1),
                        "llm":        round(t_llm, 1),
                        "total":      round(t_total, 1),
                    },
                }
                out_path = RESULTS_DIR / "sample_outputs" / f"{stem}_output.json"
                out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
                saved_outputs.append(stem)
                print(f"    ✓ saved sample output ({len(saved_outputs)}/{args.save_outputs})")

        except Exception:
            print(f"  ERROR")
            traceback.print_exc()

    csv_file.close()

    if not rows:
        print("\nNo samples processed.")
        return

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    def std(vals):
        if len(vals) < 2:
            return 0.0
        m = mean(vals)
        return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5

    def col(key):
        return [r[key] for r in rows if isinstance(r[key], (int, float))]

    summary = {
        "n_samples":    len(rows),
        "model":        GEMINI_MODEL,
        "latency_ms": {
            "preprocess_mean":  round(mean(col("t_preprocess_ms")), 1),
            "preprocess_std":   round(std(col("t_preprocess_ms")), 1),
            "yolo_mean":        round(mean(col("t_yolo_ms")), 1),
            "yolo_std":         round(std(col("t_yolo_ms")), 1),
            "ocr_mean":         round(mean(col("t_ocr_ms")), 1),
            "ocr_std":          round(std(col("t_ocr_ms")), 1),
            "xycut_mean":       round(mean(col("t_xycut_ms")), 1),
            "xycut_std":        round(std(col("t_xycut_ms")), 1),
            "llm_mean":         round(mean(col("t_llm_ms")), 1),
            "llm_std":          round(std(col("t_llm_ms")), 1),
            "total_mean":       round(mean(col("t_total_ms")), 1),
            "total_std":        round(std(col("t_total_ms")), 1),
        },
        "blocks_per_page": {
            "total_mean":  round(mean(col("n_blocks")), 1),
            "text_mean":   round(mean(col("n_text_blocks")), 1),
            "table_mean":  round(mean(col("n_table_crops")), 1),
        },
        "cost": {
            "input_tokens_mean":    round(mean(col("input_tokens")), 0),
            "output_tokens_mean":   round(mean(col("output_tokens")), 0),
            "thinking_tokens_mean": round(mean(col("thinking_tokens")), 0),
            "cost_usd_mean":        round(mean(col("cost_usd")), 6),
            "cost_usd_std":         round(std(col("cost_usd")), 6),
            "cost_usd_total":       round(sum(col("cost_usd")), 4),
            "cost_usd_per_1000_pages": round(mean(col("cost_usd")) * 1000, 3),
        },
        "schema_parse_rate": round(sum(col("schema_ok")) / len(rows), 4),
        "sections_mean":     round(mean(col("n_sections")), 1),
        "sample_outputs_saved": len(saved_outputs),
        "model_load_ms": {
            "yolo":  round(t_yolo_load, 0),
            "doctr": round(t_doctr_load, 0),
        },
    }

    summary_path = RESULTS_DIR / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Print report ───────────────────────────────────────────────────────────
    lat = summary["latency_ms"]
    cst = summary["cost"]

    print(f"\n{'='*60}")
    print(f"  BENCHMARK RESULTS — {len(rows)} samples")
    print(f"{'='*60}")
    print(f"\n  Latency (mean ± std) per page:")
    print(f"    Preprocess (pdf2image + letterbox) : {lat['preprocess_mean']:>7.1f} ± {lat['preprocess_std']:.1f} ms")
    print(f"    YOLO inference                     : {lat['yolo_mean']:>7.1f} ± {lat['yolo_std']:.1f} ms")
    print(f"    docTR OCR (all text blocks)        : {lat['ocr_mean']:>7.1f} ± {lat['ocr_std']:.1f} ms")
    print(f"    XY-Cut reading order               : {lat['xycut_mean']:>7.1f} ± {lat['xycut_std']:.1f} ms")
    print(f"    LLM API call (network + inference) : {lat['llm_mean']:>7.1f} ± {lat['llm_std']:.1f} ms")
    print(f"    ─────────────────────────────────────────────")
    print(f"    TOTAL pipeline                     : {lat['total_mean']:>7.1f} ± {lat['total_std']:.1f} ms")
    print(f"\n  Blocks per page: {summary['blocks_per_page']['total_mean']:.1f} total "
          f"({summary['blocks_per_page']['text_mean']:.1f} text, "
          f"{summary['blocks_per_page']['table_mean']:.1f} table)")
    print(f"\n  Gemini API cost per page:")
    print(f"    Input tokens  : {cst['input_tokens_mean']:.0f}")
    print(f"    Output tokens : {cst['output_tokens_mean']:.0f}")
    print(f"    Cost (mean)   : ${cst['cost_usd_mean']:.5f}")
    print(f"    Cost (total {len(rows)} pages): ${cst['cost_usd_total']:.4f}")
    print(f"    Cost/1000 pages: ${cst['cost_usd_per_1000_pages']:.3f}")
    print(f"\n  Schema parse rate : {summary['schema_parse_rate']:.1%}")
    print(f"  Sections/page     : {summary['sections_mean']:.1f}")
    print(f"  Sample outputs    : {len(saved_outputs)} saved")
    print(f"\n  Output files:")
    print(f"    {csv_path}")
    print(f"    {summary_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_benchmark()
