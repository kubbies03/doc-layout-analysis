"""
Evaluation script for LLM stage of Pipeline C.

Metrics:
  1. Schema parse rate  — % pages where LLM returns valid JSON with sections[]
  2. Table token F1     — set-overlap between LLM table text and GT table text

Run:
  python eval_llm_stage.py [--limit N] [--random] [--seed 42]

Output:
  eval_results_llm/per_sample_llm.csv
  eval_results_llm/summary_llm.json
  eval_results_llm/progress_llm.txt
"""

import os, io, sys, json, csv, time, re, argparse, traceback
import random as _random
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field

# ── Config ────────────────────────────────────────────────────────────────────
STAGING_DIR  = Path(r"C:\Users\HP\Desktop\Kubbies\KhoaLuan\GT\staging")
RESULTS_DIR  = Path(r"C:\Users\HP\Desktop\Kubbies\KhoaLuan\eval_results_llm")
YOLO_WEIGHTS = r"C:\Users\HP\Desktop\Kubbies\KhoaLuan\dataset\BaseC\best.pt"
POPPLER_PATH = r"C:\Users\HP\Desktop\Kubbies\KhoaLuan\poppler\Library\bin"

PDF_DPI   = 150
GT_IMGSZ  = 1025
MIN_AREA  = 800.0
YOLO_CONF = 0.20
YOLO_IOU  = 0.60
YOLO_IMGSZ = 640
CLASS_CONF_OVERRIDE = {"Picture": 0.50, "Section-header": 0.40, "Caption": 0.40, "Table": 0.12}
CLASS_NAMES = ["Caption","Footnote","Formula","List-item","Page-footer",
               "Page-header","Picture","Section-header","Table","Text","Title"]
TEXT_CLASSES  = {"Caption","Footnote","List-item","Page-footer",
                 "Page-header","Section-header","Text","Title"}
TABLE_CLASSES = {"Table"}

GEMINI_MODEL   = "gemini-2.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBIyAcp-f5BW6cAUbJoxFr9Wj71gIuVzKE")
MAX_RETRIES    = 3
CHECKPOINT_EVERY = 50

# ── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--limit",  type=int, default=None)
ap.add_argument("--random", action="store_true")
ap.add_argument("--seed",   type=int, default=42)
ap.add_argument("--stems",  type=str, default=None, help="Override stems file path")
args, _ = ap.parse_known_args()

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── LayoutBlock bridge ────────────────────────────────────────────────────────
@dataclass
class BBox:
    x1: float; y1: float; x2: float; y2: float

    @property
    def cx(self): return (self.x1 + self.x2) / 2
    @property
    def cy(self): return (self.y1 + self.y2) / 2

@dataclass
class LayoutBlock:
    block_id: int
    class_name: str
    conf: float
    bbox: BBox
    reading_order: int = -1
    raw_text: str = ""
    ocr_conf: float = 1.0
    ocr_method: str = "doctr"
    processed_content: Any = None

def dict_to_layoutblock(d: dict, idx: int) -> LayoutBlock:
    x, y, w, h = d["bbox_coco"]
    return LayoutBlock(
        block_id=idx,
        class_name=d["label_name"],
        conf=d.get("conf", 1.0),
        bbox=BBox(x1=x, y1=y, x2=x+w, y2=y+h),
        raw_text=d.get("text", ""),
        processed_content=d.get("text", "") or None,
    )

# ── LLM client ────────────────────────────────────────────────────────────────
_genai_client = None

def get_genai_client():
    global _genai_client
    if _genai_client is None:
        import google.genai as genai
        _genai_client = genai.Client(api_key=GEMINI_API_KEY)
    return _genai_client

def _cv2_to_pil(img):
    from PIL import Image as PILImage
    import cv2
    return PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def _call_gemini(prompt: str, images: list = None, retries: int = MAX_RETRIES, delay: float = 2.0) -> str:
    """Call Gemini with optional image crops. Thinking disabled (budget=0)."""
    from google.genai import types as genai_types
    client = get_genai_client()
    contents = [prompt]
    if images:
        contents.extend(_cv2_to_pil(img) for img in images)
    cfg = genai_types.GenerateContentConfig(
        thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
    )
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL, contents=contents, config=cfg,
            )
            return resp.text.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                raise RuntimeError(f"Gemini failed after {retries} attempts: {e}")

def crop_block_bgr(canvas_bgr, bbox_coco):
    """Crop a block from canvas using [x, y, w, h] COCO bbox."""
    x, y, w, h = [int(v) for v in bbox_coco]
    return canvas_bgr[y:y+h, x:x+w]

# ── LLM prompt + schema ───────────────────────────────────────────────────────
COMBINED_SCHEMA_SUMMARY_PROMPT = """\
You are a structured document analysis assistant.
Below are text blocks extracted from a single document page (in reading order).
{table_note}
Text blocks:
---
{blocks_text}
---

CRITICAL RULE: Copy each text block's content VERBATIM into paragraphs[]. Do NOT summarize, paraphrase, shorten, or omit any text block. Every [Text], [Title], [List-item], [Caption], [Footnote], [Section-header] block must appear as a separate paragraph string with its full original content preserved exactly. Only the "summary" field should be a summary.

Return a single JSON object with this exact structure:
{{
  "structured_json": {{
    "title": "<string | null>",
    "sections": [
      {{
        "heading": "<string | null — copy verbatim from Section-header block>",
        "paragraphs": ["<copy each text block verbatim, one string per block>"],
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

def _validate_schema(data: dict) -> bool:
    structured = data.get("structured_json", {})
    return isinstance(structured.get("sections"), list)

def _blocks_to_text(blocks: list) -> str:
    parts = []
    for b in sorted(blocks, key=lambda x: x.reading_order):
        content = b.processed_content or b.raw_text
        parts.append(f"[{b.class_name}] {content}")
    return "\n".join(parts)

def assemble_and_summarize(blocks: list, table_crops: list = None) -> tuple:
    """LLM call: text blocks + table crops → (structured_json, is_valid, summary, keywords, metadata)."""
    blocks_text = _blocks_to_text(blocks)
    if not blocks_text.strip() and not table_crops:
        return {}, True, "Empty page.", [], {}

    crops = [c for c in (table_crops or []) if c is not None and c.size > 0]
    table_note = TABLE_NOTE_TEMPLATE.format(n=len(crops)) if crops else ""
    prompt = COMBINED_SCHEMA_SUMMARY_PROMPT.format(
        blocks_text=blocks_text, table_note=table_note,
    )
    for attempt in range(MAX_RETRIES):
        try:
            raw = _call_gemini(prompt, images=crops if crops else None)
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"```$", "", raw).strip()
            data = json.loads(raw)
            structured = data.get("structured_json", {})
            is_valid = _validate_schema(data)
            summary  = data.get("summary", "")
            keywords = data.get("keywords", [])
            meta     = {k: data[k] for k in ("summary","keywords","page_type","language") if k in data}
            return structured, is_valid, summary, keywords, meta
        except Exception as e:
            print(f"  [LLM-Gemini] Attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
    print("  [LLM-Gemini] All attempts failed — returning empty.")
    return {"error": "parse_failed"}, False, "", [], {}

# ── Reading order (XY-cut) ────────────────────────────────────────────────────
def detect_columns(blocks: list, img_width: int) -> int:
    if not blocks:
        return 1
    mid = img_width / 2
    left  = sum(1 for b in blocks if b.bbox.cx < mid)
    right = sum(1 for b in blocks if b.bbox.cx >= mid)
    if left > 0 and right > 0 and min(left, right) / max(left, right) > 0.35:
        return 2
    return 1

def assign_reading_order(blocks: list, img_width: int) -> list:
    n_cols = detect_columns(blocks, img_width)
    if n_cols == 2:
        mid = img_width / 2
        left_blocks  = sorted([b for b in blocks if b.bbox.cx <  mid], key=lambda b: (b.bbox.y1, b.bbox.x1))
        right_blocks = sorted([b for b in blocks if b.bbox.cx >= mid], key=lambda b: (b.bbox.y1, b.bbox.x1))
        ordered = left_blocks + right_blocks
    else:
        sorted_y = sorted(blocks, key=lambda b: b.bbox.y1)
        rows: list = []
        for b in sorted_y:
            placed = False
            for row in rows:
                row_y1 = min(r.bbox.y1 for r in row)
                row_y2 = max(r.bbox.y2 for r in row)
                if b.bbox.y1 < row_y2 and b.bbox.y2 > row_y1:
                    row.append(b); placed = True; break
            if not placed:
                rows.append([b])
        ordered = []
        for row in rows:
            ordered.extend(sorted(row, key=lambda b: b.bbox.x1))
    for i, b in enumerate(ordered):
        b.reading_order = i
    return ordered

# ── YOLO + Doctr pipeline (reused from eval_pipeline_c) ──────────────────────
_yolo = None
def get_yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        _yolo = YOLO(YOLO_WEIGHTS)
        print("  YOLO loaded.")
    return _yolo

_doctr = None
def get_doctr():
    global _doctr
    if _doctr is None:
        os.environ["USE_TORCH"] = "1"
        from doctr.models import ocr_predictor
        import torch
        _doctr = ocr_predictor(pretrained=True)
        if torch.cuda.is_available():
            _doctr = _doctr.cuda()
        print("  Doctr loaded.")
    return _doctr

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

def detect_blocks(canvas):
    import numpy as np
    model = get_yolo()
    res = model.predict(source=canvas, conf=YOLO_CONF, iou=YOLO_IOU,
                        imgsz=YOLO_IMGSZ, max_det=300, device="0",
                        verbose=False)[0]
    blocks = []
    for box in res.boxes:
        cls_id = int(box.cls[0])
        label  = CLASS_NAMES[cls_id]
        conf   = float(box.conf[0])
        thr    = CLASS_CONF_OVERRIDE.get(label, YOLO_CONF)
        if conf < thr: continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        w, h = x2 - x1, y2 - y1
        if w * h < MIN_AREA: continue
        blocks.append({"label_name": label, "bbox_coco": [x1,y1,w,h], "conf": conf, "text": ""})
    return blocks

def ocr_crops_batch(img_bgr, bboxes):
    import numpy as np
    from PIL import Image as PILImage
    from doctr.io import DocumentFile

    def crop_png(x, y, w, h):
        import cv2
        x1,y1,x2,y2 = int(x),int(y),int(x+w),int(y+h)
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0: return None
        pil = PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO(); pil.save(buf, format="PNG")
        return buf.getvalue()

    png_list = [crop_png(*bb) for bb in bboxes]
    valid = [(i, p) for i, p in enumerate(png_list) if p]
    texts = [""] * len(bboxes)
    if not valid: return texts

    model = get_doctr()
    doc = DocumentFile.from_images([p for _, p in valid])
    result = model(doc)
    for page_i, (src_i, _) in enumerate(valid):
        words = []
        for blk in result.pages[page_i].blocks:
            for line in blk.lines:
                for word in line.words:
                    if word.value.strip(): words.append(word.value)
        texts[src_i] = " ".join(words)
    return texts

def run_pipeline_c(pdf_path):
    """YOLO+Doctr detect+OCR.
    - Text/Title/Caption/... → Doctr OCR
    - Table                  → crop collected, sent in single Gemini Vision call later
    - Picture/Formula        → skipped (no text)
    Returns (block_dicts, (actual_h, canvas_w), joined_text, table_crops_bgr).
    """
    import cv2, numpy as np
    pil_pages = pdf_to_pil(pdf_path)
    all_blocks = []
    actual_h = GT_IMGSZ
    text_parts = []
    table_crops = []  # list of bgr ndarray, one per Table block

    for pil in pil_pages:
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        canvas, actual_h, actual_w = letterbox(img)
        blocks = detect_blocks(canvas)

        # Doctr OCR for text-bearing blocks
        doctr_idx = [i for i, b in enumerate(blocks)
                     if b["label_name"] in TEXT_CLASSES]
        bboxes = [blocks[i]["bbox_coco"] for i in doctr_idx]
        texts  = ocr_crops_batch(canvas, bboxes) if bboxes else []
        for k, i in enumerate(doctr_idx):
            blocks[i]["text"] = texts[k]
            if texts[k]:
                text_parts.append(texts[k])

        # Collect Table crops (no Gemini call here)
        for b in blocks:
            if b["label_name"] in TABLE_CLASSES:
                crop = crop_block_bgr(canvas, b["bbox_coco"])
                if crop.size > 0:
                    table_crops.append(crop)

        all_blocks.extend(blocks)

    # GT_IMGSZ as canvas width — bboxes live in canvas coordinate space
    return all_blocks, (actual_h, GT_IMGSZ), " ".join(text_parts), table_crops

# ── Metric helpers ────────────────────────────────────────────────────────────
from collections import Counter

def _normalize_tokens(text: str) -> Counter:
    """Tokenize into Counter, stripping markdown table syntax and punctuation noise."""
    text = re.sub(r"[|:\-]{1,}", " ", text)
    text = re.sub(r"[¥$€£()\[\]{}]", " ", text)
    tokens = text.lower().split()
    return Counter(t for t in tokens if any(c.isalnum() for c in t))

def token_f1(ref: str, hyp: str) -> float:
    """Token-level F1 using multiset (Counter) overlap — respects duplicate tokens."""
    ref_c = _normalize_tokens(ref)
    hyp_c = _normalize_tokens(hyp)
    n_ref = sum(ref_c.values())
    n_hyp = sum(hyp_c.values())
    if n_ref == 0:
        return 1.0 if n_hyp == 0 else 0.0
    overlap = sum((ref_c & hyp_c).values())
    precision = overlap / n_hyp if n_hyp else 0.0
    recall    = overlap / n_ref
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def flatten_llm_tables(structured_json: dict) -> str:
    """Flatten all LLM-extracted table content into a single text."""
    parts = []
    for section in structured_json.get("sections", []):
        for tbl in section.get("tables", []):
            parts.extend(str(h) for h in tbl.get("headers", []))
            for row in tbl.get("rows", []):
                parts.extend(str(c) for c in row)
    return " ".join(parts)

def count_llm_tables(structured_json: dict) -> int:
    n = 0
    for section in structured_json.get("sections", []):
        n += len(section.get("tables", []))
    return n

def get_gt_table_text(gt_data: dict) -> str:
    """Concatenate all GT Table block texts from the JSON."""
    parts = []
    for page in gt_data.get("pages", []):
        for ann in page.get("layout_gt", []):
            if ann.get("label_name") == "Table" and ann.get("text"):
                parts.append(ann["text"])
    return " ".join(parts)

def count_gt_tables(gt_data: dict) -> int:
    n = 0
    for page in gt_data.get("pages", []):
        n += sum(1 for ann in page.get("layout_gt", []) if ann.get("label_name") == "Table")
    return n

def get_gt_full_text(gt_data: dict) -> str:
    """All GT block texts concatenated (for content recall)."""
    parts = []
    for page in gt_data.get("pages", []):
        for ann in page.get("layout_gt", []):
            if ann.get("text"):
                parts.append(ann["text"])
    return " ".join(parts)

def flatten_llm_text(structured_json: dict) -> str:
    """Flatten all LLM paragraphs + table content into one string."""
    parts = []
    for sec in structured_json.get("sections", []):
        parts.extend(sec.get("paragraphs", []))
        for tbl in sec.get("tables", []):
            parts.extend(str(h) for h in tbl.get("headers", []))
            for row in tbl.get("rows", []):
                parts.extend(str(c) for c in row)
        parts.extend(sec.get("formulas", []))
    return " ".join(parts)

def content_recall(gt_text: str, llm_text: str) -> float:
    """Token recall: what fraction of GT tokens appear in LLM output."""
    gt_c  = _normalize_tokens(gt_text)
    llm_c = _normalize_tokens(llm_text)
    n_gt  = sum(gt_c.values())
    if n_gt == 0:
        return 1.0
    return sum((gt_c & llm_c).values()) / n_gt

# ── CSV / progress setup ──────────────────────────────────────────────────────
SAMPLE_FIELDS = ["stem","schema_ok","sections_ok","table_extracted_ok",
                 "table_f1","content_recall",
                 "has_gt_table","n_gt_tables","llm_sections","n_llm_tables",
                 "elapsed","error"]

PROGRESS_FILE = RESULTS_DIR / "progress_llm.txt"
SAMPLE_CSV    = RESULTS_DIR / "per_sample_llm.csv"
SUMMARY_JSON  = RESULTS_DIR / "summary_llm.json"

done_stems: set = set()
if PROGRESS_FILE.exists():
    done_stems = set(PROGRESS_FILE.read_text(encoding="utf-8").splitlines())

csv_is_new = not SAMPLE_CSV.exists()
csv_fh = open(SAMPLE_CSV, "a", newline="", encoding="utf-8")
csv_w  = csv.DictWriter(csv_fh, fieldnames=SAMPLE_FIELDS)
if csv_is_new:
    csv_w.writeheader()

prog_fh = open(PROGRESS_FILE, "a", encoding="utf-8")

# ── Accumulators (reload from CSV on resume) ──────────────────────────────────
agg_schema_ok:      list[bool]  = []
agg_sections_ok:    list[bool]  = []  # llm_sections > 0
agg_tbl_extract_ok: list[bool]  = []  # n_llm_tables > 0, only when has_gt_table
agg_table_f1:       list[float] = []  # only when has_gt_table
agg_content_recall: list[float] = []  # all samples

if SAMPLE_CSV.exists():
    with open(SAMPLE_CSV, newline="", encoding="utf-8") as _f:
        for _row in csv.DictReader(_f):
            if _row.get("schema_ok") != "":
                agg_schema_ok.append(_row["schema_ok"] == "1")
            if _row.get("sections_ok") != "":
                agg_sections_ok.append(_row["sections_ok"] == "1")
            if _row.get("table_extracted_ok") != "":
                agg_tbl_extract_ok.append(_row["table_extracted_ok"] == "1")
            if _row.get("table_f1") != "":
                agg_table_f1.append(float(_row["table_f1"]))
            if _row.get("content_recall") != "":
                agg_content_recall.append(float(_row["content_recall"]))
    if agg_schema_ok:
        print(f"  Resumed: {len(agg_schema_ok)} samples already in CSV")

# ── Sample list ───────────────────────────────────────────────────────────────
STEMS_FILE = Path(args.stems) if args.stems else None
if STEMS_FILE and STEMS_FILE.exists():
    allowed_stems = set(STEMS_FILE.read_text(encoding="utf-8").splitlines())
    all_pdfs = sorted(p for p in STAGING_DIR.glob("*.pdf") if p.stem in allowed_stems)
    print(f"Using curated stems file ({len(all_pdfs)} PDFs found)")
else:
    all_pdfs = sorted(STAGING_DIR.glob("*.pdf"))
    print(f"Using all PDFs in staging ({len(all_pdfs)} found)")

if args.limit:
    if args.random:
        rng = _random.Random(args.seed)
        all_pdfs = sorted(rng.sample(all_pdfs, min(args.limit, len(all_pdfs))))
    else:
        all_pdfs = all_pdfs[:args.limit]

print(f"\nLLM Stage Eval -- {len(all_pdfs)} samples -> {RESULTS_DIR}\n")

# ── Checkpoint printer ────────────────────────────────────────────────────────
def print_checkpoint(n_done: int):
    n = len(agg_schema_ok)
    if n == 0:
        return
    parse_rate = sum(agg_schema_ok) / n
    tbl_n = len(agg_table_f1)
    tbl_mean = sum(agg_table_f1) / tbl_n if tbl_n else float("nan")

    rec_n    = len(agg_content_recall)
    rec_mean = sum(agg_content_recall) / rec_n if rec_n else float("nan")
    ext_n    = len(agg_tbl_extract_ok)
    ext_rate = sum(agg_tbl_extract_ok) / ext_n if ext_n else float("nan")

    print(f"\n{'='*60}")
    print(f"  CHECKPOINT @ {n_done} samples")
    print(f"{'='*60}")
    print(f"  Schema parse rate  : {parse_rate:.1%}  ({sum(agg_schema_ok)}/{n})")
    print(f"  Sections coverage  : {sum(agg_sections_ok)}/{n} non-empty")
    print(f"  Content recall     : {rec_mean:.3f}  (n={rec_n})")
    if ext_n > 0:
        print(f"  Table extract rate : {ext_rate:.1%}  ({sum(agg_tbl_extract_ok)}/{ext_n} GT-table samples)")
    if tbl_n > 0:
        print(f"  Table token F1     : {tbl_mean:.3f}  (n={tbl_n})")
    else:
        print(f"  Table token F1     : N/A")
    print(f"{'='*60}\n")

# ── Main loop ─────────────────────────────────────────────────────────────────
n_processed = 0
for idx, pdf_path in enumerate(all_pdfs):
    stem = pdf_path.stem
    if stem in done_stems:
        print(f"[{idx+1:3}/{len(all_pdfs)}] {stem[:20]} SKIP")
        continue

    gt_path = STAGING_DIR / f"{stem}_v12_merged.json"
    if not gt_path.exists():
        print(f"[{idx+1:3}/{len(all_pdfs)}] {stem[:20]} NO_GT")
        continue

    try:
        gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[{idx+1:3}/{len(all_pdfs)}] {stem[:20]} GT_ERR {e}")
        continue

    n_gt_tables  = count_gt_tables(gt_data)
    has_gt_table = n_gt_tables > 0
    gt_tbl_text  = get_gt_table_text(gt_data) if has_gt_table else ""
    gt_full_text = get_gt_full_text(gt_data)

    err_msg = ""
    schema_ok         = False
    sections_ok       = False
    table_extracted_ok = False
    tbl_f1            = float("nan")
    recall            = float("nan")
    llm_sections      = 0
    n_llm_tables      = 0
    elapsed           = 0.0

    try:
        t0 = time.time()

        # Step 1: YOLO + Doctr + collect table crops
        block_dicts, (actual_h, actual_w), _, table_crops = run_pipeline_c(pdf_path)

        # Step 2: Convert to LayoutBlock + assign reading order
        layout_blocks = [dict_to_layoutblock(d, i) for i, d in enumerate(block_dicts)]
        layout_blocks = assign_reading_order(layout_blocks, actual_w)

        # Step 3: single Gemini call — text blocks + all table images
        structured_json, is_valid, summary, keywords, meta = assemble_and_summarize(
            layout_blocks, table_crops=table_crops
        )

        elapsed = time.time() - t0

        # Metric 1: schema parse rate
        schema_ok    = is_valid
        llm_sections = len(structured_json.get("sections", []))
        n_llm_tables = count_llm_tables(structured_json)

        # Metric 2a: sections_ok — LLM produced at least 1 section
        sections_ok = llm_sections > 0

        # Metric 2b: table_extracted_ok — LLM extracted table when GT has one
        if has_gt_table:
            table_extracted_ok = n_llm_tables > 0
            agg_tbl_extract_ok.append(table_extracted_ok)

        # Metric 3: table token F1
        if has_gt_table:
            llm_tbl_text = flatten_llm_tables(structured_json)
            tbl_f1 = token_f1(gt_tbl_text, llm_tbl_text)
            agg_table_f1.append(tbl_f1)

        # Metric 4: content recall (all samples)
        llm_full_text = flatten_llm_text(structured_json)
        recall = content_recall(gt_full_text, llm_full_text)
        agg_content_recall.append(recall)

        agg_schema_ok.append(schema_ok)
        agg_sections_ok.append(sections_ok)

    except Exception as e:
        err_msg = str(e)[:120]
        traceback.print_exc()
        agg_schema_ok.append(False)
        agg_sections_ok.append(False)
        elapsed = time.time() - t0

    # Write CSV row
    row = {
        "stem":               stem[:20],
        "schema_ok":          int(schema_ok),
        "sections_ok":        int(sections_ok),
        "table_extracted_ok": int(table_extracted_ok) if has_gt_table else "",
        "table_f1":           round(tbl_f1, 4) if tbl_f1 == tbl_f1 else "",
        "content_recall":     round(recall, 4) if recall == recall else "",
        "has_gt_table":       int(has_gt_table),
        "n_gt_tables":        n_gt_tables,
        "llm_sections":       llm_sections,
        "n_llm_tables":       n_llm_tables,
        "elapsed":            round(elapsed, 1),
        "error":              err_msg,
    }
    csv_w.writerow(row); csv_fh.flush()
    prog_fh.write(stem + "\n"); prog_fh.flush()

    n_processed += 1
    tbl_f1_str  = f"{tbl_f1:.3f}"  if tbl_f1  == tbl_f1  else "  N/A"
    recall_str  = f"{recall:.3f}"  if recall   == recall   else "  N/A"
    print(f"[{idx+1:3}/{len(all_pdfs)}] {stem[:20]} schema={'OK' if schema_ok else 'FAIL'}"
          f"  recall={recall_str}  tbl_f1={tbl_f1_str}  sec={llm_sections}  {elapsed:.1f}s")

    if n_processed % CHECKPOINT_EVERY == 0:
        print_checkpoint(n_processed)

# ── Final summary ─────────────────────────────────────────────────────────────
csv_fh.close()
prog_fh.close()

n_total    = len(agg_schema_ok)
parse_rate = sum(agg_schema_ok) / n_total if n_total else 0.0
sec_rate   = sum(agg_sections_ok) / n_total if n_total else 0.0

tbl_n    = len(agg_table_f1)
tbl_mean = sum(agg_table_f1) / tbl_n if tbl_n else float("nan")
tbl_std  = (sum((x - tbl_mean)**2 for x in agg_table_f1) / tbl_n) ** 0.5 if tbl_n > 1 else float("nan")

ext_n    = len(agg_tbl_extract_ok)
ext_rate = sum(agg_tbl_extract_ok) / ext_n if ext_n else float("nan")

rec_n    = len(agg_content_recall)
rec_mean = sum(agg_content_recall) / rec_n if rec_n else float("nan")
rec_std  = (sum((x - rec_mean)**2 for x in agg_content_recall) / rec_n) ** 0.5 if rec_n > 1 else float("nan")

summary = {
    "n_samples":                     n_total,
    "schema_parse_rate":             round(parse_rate, 4),
    "sections_coverage_rate":        round(sec_rate, 4),
    "content_recall_mean":           round(rec_mean, 4) if rec_mean == rec_mean else None,
    "content_recall_std":            round(rec_std,  4) if rec_std  == rec_std  else None,
    "table_extract_rate":            round(ext_rate, 4) if ext_rate == ext_rate else None,
    "n_samples_with_gt_table":       ext_n,
    "table_token_f1_mean":           round(tbl_mean, 4) if tbl_mean == tbl_mean else None,
    "table_token_f1_std":            round(tbl_std,  4) if tbl_std  == tbl_std  else None,
}
SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

print("\n" + "="*60)
print(f"  FINAL RESULTS — {n_total} samples")
print("="*60)
print(f"  Schema parse rate  : {parse_rate:.1%}")
print(f"  Sections coverage  : {sec_rate:.1%}")
print(f"  Content recall     : {rec_mean:.3f} ± {rec_std:.3f}  (n={rec_n})" if rec_n > 1 else f"  Content recall     : {rec_mean:.3f}")
if ext_n > 0:
    print(f"  Table extract rate : {ext_rate:.1%}  ({int(sum(agg_tbl_extract_ok))}/{ext_n})")
if tbl_n > 0:
    print(f"  Table token F1     : {tbl_mean:.3f} ± {tbl_std:.3f}  (n={tbl_n})" if tbl_n > 1 else f"  Table token F1     : {tbl_mean:.3f}")
else:
    print(f"  Table token F1     : N/A")
print(f"\n  Results saved to: {RESULTS_DIR}")
print("="*60)
