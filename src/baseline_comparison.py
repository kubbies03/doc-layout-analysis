"""
Baseline Comparison Benchmark
==============================
Compare 4 document content extraction methods on 100 high-quality samples:

  B1: PyMuPDF  — extract text directly from PDF layer
  B2: docTR    — full-page OCR (no layout detection)
  B3: Gemini Vision — send full page image, request JSON schema response
  P:  Proposed pipeline — use cached results from per_sample_llm.csv + per_sample.csv

Metrics:
  - content_recall: fuzzy token matching between extracted text and GT text
  - table_f1:       token F1 on table content (149 samples with tables, ~22 in top-100)
  - cost_usd_per_page: 0 for B1/B2, measured for B3
  - latency_s_per_page: measured for B1/B2/B3

Results saved to:
  baseline_results/baseline_summary.json
  baseline_results/baseline_per_sample.csv
  baseline_results/baseline_comparison_table.txt

Usage:
  python baseline_comparison.py [--n 100] [--skip-gemini] [--skip-doctr]
"""

import os, io, sys, json, csv, time, re, argparse, traceback, random
from pathlib import Path
from typing import Optional

# Fix Windows console encoding for Vietnamese/special chars
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ── Config ────────────────────────────────────────────────────────────────────
STAGING_DIR    = Path(r"C:\Users\HP\Desktop\Kubbies\KhoaLuan\GT\staging")
RESULTS_DIR    = Path(r"C:\Users\HP\Desktop\Kubbies\KhoaLuan\baseline_results")
LLM_CSV        = Path(r"C:\Users\HP\Desktop\Kubbies\KhoaLuan\eval_results_llm\per_sample_llm.csv")
DET_CSV        = Path(r"C:\Users\HP\Desktop\Kubbies\KhoaLuan\eval_results\per_sample.csv")
POPPLER_PATH   = r"C:\Users\HP\Desktop\Kubbies\KhoaLuan\poppler\Library\bin"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBIyAcp-f5BW6cAUbJoxFr9Wj71gIuVzKE")
GEMINI_MODEL   = "gemini-2.5-flash"
INPUT_PRICE    = 0.30 / 1_000_000
OUTPUT_PRICE   = 2.50 / 1_000_000
PDF_DPI        = 150

# ── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--n",              type=int, default=100, help="Number of benchmark samples")
ap.add_argument("--skip-gemini",   action="store_true",   help="Skip B3 Gemini Vision")
ap.add_argument("--skip-doctr",    action="store_true",   help="Skip B2 docTR full-page")
ap.add_argument("--skip-pipeline", action="store_true",   help="Skip proposed pipeline (live run)")
ap.add_argument("--resume",        action="store_true",   help="Resume from CSV checkpoint, skip stems already completed")
ap.add_argument("--seed",          type=int, default=42)
args = ap.parse_args()

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Fuzzy token matching (same logic as eval pipeline) ───────────────────────
def _tokenize(text: str) -> list:
    return re.findall(r'\w+', text.lower())

def fuzzy_token_recall(pred_text: str, gt_text: str) -> float:
    """Fraction of GT tokens present in pred (recall, not precision)."""
    gt_tokens = _tokenize(gt_text)
    if not gt_tokens:
        return 1.0
    pred_tokens = set(_tokenize(pred_text))
    return sum(1 for t in gt_tokens if t in pred_tokens) / len(gt_tokens)

def token_f1(pred_text: str, gt_text: str) -> float:
    """Token-level F1 between pred and gt."""
    pred_tokens = _tokenize(pred_text)
    gt_tokens   = _tokenize(gt_text)
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    pred_set, gt_set = set(pred_tokens), set(gt_tokens)
    tp = sum(1 for t in pred_tokens if t in gt_set)
    prec = tp / len(pred_tokens)
    rec  = sum(1 for t in gt_tokens if t in pred_set) / len(gt_tokens)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

# ── GT extraction from staging JSON ──────────────────────────────────────────
def load_gt_text(gt_path: Path) -> tuple[str, str]:
    """
    Returns (gt_full_text, gt_table_text) from staging JSON.
    gt_full_text: all text with is_relevant_for_extraction=True
    gt_table_text: text from Table blocks only
    """
    try:
        data = json.load(open(gt_path, encoding='utf-8', errors='replace'))
    except Exception:
        return "", ""

    full_parts  = []
    table_parts = []
    for page in data.get('pages', []):
        for block in page.get('layout_gt', []):
            if not block.get('is_relevant_for_extraction', True):
                continue
            text = block.get('text', '').strip()
            if not text:
                continue
            full_parts.append(text)
            if block.get('label_name') == 'Table':
                table_parts.append(text)

    return ' '.join(full_parts), ' '.join(table_parts)

# ── Sample selection ──────────────────────────────────────────────────────────
def select_top_samples(n: int) -> list[dict]:
    """Select top-n highest-quality samples based on composite score."""
    import statistics

    # Load GT file mapping
    gt_files = {f.split('_v12_merged')[0]: f
                for f in os.listdir(STAGING_DIR)
                if f.endswith('_v12_merged.json')}

    # Build stem map: llm_stem -> det_stem
    stem_map = {}
    gt_stem_map = {}  # llm_stem -> full GT filename stem
    for fname, json_fname in gt_files.items():
        try:
            s = json.load(open(STAGING_DIR / json_fname, encoding='utf-8', errors='replace'))
            did = s.get('doc_id', '')
            parts = did.split('_')
            if len(parts) >= 3:
                llm_stem = 'sample_' + parts[1] + '_' + parts[2][:8]
                det_stem = parts[1] + '_' + parts[2][:3]
                stem_map[llm_stem] = det_stem
                gt_stem_map[llm_stem] = fname  # e.g. sample_0004_d3b8ea12...
        except Exception:
            pass

    # Load eval results
    with open(LLM_CSV, encoding='utf-8') as f:
        llm_rows = {r['stem']: r for r in csv.DictReader(f)}
    with open(DET_CSV, encoding='utf-8') as f:
        det_rows = {r['stem']: r for r in csv.DictReader(f)}

    # Merge and score
    candidates = []
    for ls, lrow in llm_rows.items():
        ds = stem_map.get(ls)
        if not ds or ds not in det_rows:
            continue
        dr = det_rows[ds]
        cr      = float(lrow['content_recall']) if lrow['content_recall'] else 0.0
        det_f1  = float(dr['f1']) if dr['f1'] else 0.0
        schema  = lrow['schema_ok'] == '1'
        cer     = float(dr.get('cer_text') or 0.5)
        has_tbl = lrow['has_gt_table'] == '1'
        tf1     = float(lrow['table_f1']) if lrow['table_f1'] and has_tbl else None

        if not schema or cr < 0.75 or det_f1 < 0.75:
            continue

        score = cr * 0.5 + det_f1 * 0.3 + max(0, 1 - cer) * 0.2
        gt_fname = gt_stem_map.get(ls, '')

        # Find PDF file
        pdf_candidates = list(STAGING_DIR.glob(f"{gt_fname}*.pdf")) if gt_fname else []
        pdf_path = pdf_candidates[0] if pdf_candidates else None

        candidates.append({
            'llm_stem':       ls,
            'det_stem':       ds,
            'gt_fname':       gt_fname,
            'pdf_path':       pdf_path,
            'score':          score,
            'pipeline_recall': cr,
            'pipeline_det_f1': det_f1,
            'pipeline_tf1':   tf1,
            'has_table':      has_tbl,
        })

    candidates.sort(key=lambda x: -x['score'])
    selected = candidates[:n]
    print(f"[select] {len(selected)}/{len(candidates)} candidates -> top {n}")
    print(f"  Avg pipeline recall={sum(s['pipeline_recall'] for s in selected)/len(selected):.3f}  "
          f"det_f1={sum(s['pipeline_det_f1'] for s in selected)/len(selected):.3f}")
    print(f"  Samples with table: {sum(1 for s in selected if s['has_table'])}")
    return selected

# ── B1: PyMuPDF ───────────────────────────────────────────────────────────────
def extract_pymupdf(pdf_path: Path) -> tuple[str, float]:
    """Extract text from PDF layer via PyMuPDF. Returns (text, latency_s)."""
    import fitz
    t0 = time.perf_counter()
    doc = fitz.open(str(pdf_path))
    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
    doc.close()
    return ' '.join(parts), time.perf_counter() - t0

# ── B2: docTR full-page OCR ───────────────────────────────────────────────────
_doctr_model = None

def get_doctr():
    global _doctr_model
    if _doctr_model is None:
        os.environ["USE_TORCH"] = "1"
        from doctr.models import ocr_predictor
        import torch
        _doctr_model = ocr_predictor(pretrained=True)
        if torch.cuda.is_available():
            _doctr_model = _doctr_model.cuda()
        print(f"  [load] docTR ready (cuda={torch.cuda.is_available()})")
    return _doctr_model

def extract_doctr(pdf_path: Path) -> tuple[str, float]:
    """Full-page OCR with docTR. Returns (text, latency_s)."""
    from doctr.io import DocumentFile
    t0 = time.perf_counter()
    doc_obj = DocumentFile.from_pdf(str(pdf_path))
    result  = get_doctr()(doc_obj)
    parts   = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                parts.append(' '.join(w.value for w in line.words))
    return ' '.join(parts), time.perf_counter() - t0

# ── Pipeline live run (YOLO + docTR crop OCR + XY-Cut + Gemini) ──────────────
def _load_pipeline_modules():
    """
    Import required functions from benchmark_pipeline.py.
    benchmark_pipeline has argparse at module level so we use importlib to avoid
    conflicts with this script's argparse.
    """
    import importlib.util, sys as _sys
    spec = importlib.util.spec_from_file_location(
        "bpipe",
        str(Path(__file__).parent / "benchmark_pipeline.py")
    )
    # Temporarily patch sys.argv so benchmark_pipeline's argparse is not triggered
    _orig_argv = _sys.argv
    _sys.argv = [str(Path(__file__).parent / "benchmark_pipeline.py")]
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _sys.argv = _orig_argv
    return mod


_pipe_mod = None

def get_pipe():
    global _pipe_mod
    if _pipe_mod is None:
        print("[init] Loading pipeline modules from benchmark_pipeline.py ...")
        _pipe_mod = _load_pipeline_modules()
        # Pre-load models
        _pipe_mod.get_yolo()
        _pipe_mod.get_doctr()
        _pipe_mod.get_client()
        print("[init] Pipeline models ready.")
    return _pipe_mod


def _extract_text_from_structured_json(sj: dict) -> tuple[str, str]:
    """Extract full_text and table_text from pipeline's structured_json."""
    parts, table_parts = [], []
    title = sj.get("title") or ""
    if title:
        parts.append(title)
    for h in sj.get("page_header", []):
        parts.append(h)
    for sec in sj.get("sections", []):
        hdg = sec.get("heading") or ""
        if hdg:
            parts.append(hdg)
        parts.extend(sec.get("paragraphs", []))
        parts.extend(sec.get("footnotes", []))
        for tbl in sec.get("tables", []):
            cell_texts = []
            for row in tbl.get("rows", []):
                cell_texts.extend(str(c) for c in row)
            if cell_texts:
                tbl_str = " | ".join(cell_texts)
                table_parts.append(tbl_str)
                parts.append(tbl_str)
    for f in sj.get("page_footer", []):
        parts.append(f)
    return " ".join(parts), " ".join(table_parts)


def run_pipeline_live(pdf_path: Path) -> tuple[str, str, float, float, dict]:
    """
    Run proposed pipeline end-to-end on pdf_path.
    Returns (full_text, table_text, latency_s, cost_usd, usage).
    """
    import cv2, numpy as np

    mod = get_pipe()
    t0  = time.perf_counter()

    # Preprocess
    pages      = mod.pdf_to_pil(pdf_path)
    pil        = pages[0]
    canvas_rgb = np.array(pil)
    canvas_bgr = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR)
    canvas, actual_h, actual_w = mod.letterbox(canvas_bgr)

    # YOLO detection
    raw_blocks = mod.detect_blocks(canvas)

    # docTR crop OCR
    TEXT_CLASSES  = mod.TEXT_CLASSES
    TABLE_CLASSES = mod.TABLE_CLASSES
    text_bbs   = [b["bbox_coco"] for b in raw_blocks if b["label_name"] in TEXT_CLASSES]
    text_idxs  = [i for i, b in enumerate(raw_blocks) if b["label_name"] in TEXT_CLASSES]
    table_idxs = [i for i, b in enumerate(raw_blocks) if b["label_name"] in TABLE_CLASSES]

    if text_bbs:
        ocr_texts = mod.ocr_crops_batch(canvas, text_bbs)
        for ti, src_i in enumerate(text_idxs):
            raw_blocks[src_i]["text"] = ocr_texts[ti]

    # Table crops
    table_crops = []
    for ti in table_idxs:
        x, y, w, h = [int(v) for v in raw_blocks[ti]["bbox_coco"]]
        crop = canvas[y:y+h, x:x+w]
        if crop.size > 0:
            table_crops.append(crop)

    # XY-Cut reading order
    layout_blocks = []
    for i, b in enumerate(raw_blocks):
        x, y, w, h = b["bbox_coco"]
        lb = mod.LayoutBlock(
            block_id=i, class_name=b["label_name"], conf=b["conf"],
            bbox=mod.BBox(x1=x, y1=y, x2=x+w, y2=y+h),
            raw_text=b["text"],
        )
        layout_blocks.append(lb)
    ordered = mod.assign_reading_order(layout_blocks, actual_w)

    # LLM
    data, is_valid, usage, _ = mod.assemble_and_summarize(ordered, table_crops)
    latency = time.perf_counter() - t0

    cost = (usage["input_tokens"] * INPUT_PRICE + usage["output_tokens"] * OUTPUT_PRICE)

    sj = data.get("structured_json", {}) if is_valid else {}
    full_text, table_text = _extract_text_from_structured_json(sj)

    return full_text, table_text, latency, cost, usage


# ── B3: Gemini Vision full-page ───────────────────────────────────────────────
_gemini_client = None

VISION_PROMPT = """\
You are a document content extractor. I will give you a page image.
Extract ALL text content from the page into a single JSON object with this structure:
{
  "full_text": "<all text on the page, in reading order, separated by newlines>",
  "tables": ["<table 1 content as pipe-separated text>", "<table 2 content>"]
}
Rules:
- Copy text VERBATIM, do NOT paraphrase or summarize
- Preserve reading order (top to bottom, left to right, then right column)
- Include headers, footers, captions, footnotes
- For tables: extract cell content row by row, cells separated by |
Return ONLY valid JSON. No markdown fences."""

def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        import google.genai as genai
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("  [load] Gemini Vision client ready")
    return _gemini_client

def extract_gemini_vision(pdf_path: Path) -> tuple[str, str, float, dict]:
    """
    Send page image to Gemini Vision, returns (full_text, table_text, latency_s, usage).
    """
    from pdf2image import convert_from_path
    from google.genai import types as genai_types

    client = get_gemini_client()
    t0 = time.perf_counter()

    pages = convert_from_path(str(pdf_path), dpi=PDF_DPI, poppler_path=POPPLER_PATH)
    all_text, all_tables = [], []
    total_usage = {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}

    cfg = genai_types.GenerateContentConfig(
        thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
    )

    for page_img in pages:
        # Convert PIL to bytes
        buf = io.BytesIO()
        page_img.save(buf, format='JPEG', quality=85)
        img_bytes = buf.getvalue()

        img_part = genai_types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")

        last_err = None
        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[VISION_PROMPT, img_part],
                    config=cfg,
                )
                raw = resp.text.strip()
                # Parse JSON
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    # Try to extract JSON from markdown fences
                    m = re.search(r'\{[\s\S]*\}', raw)
                    obj = json.loads(m.group()) if m else {"full_text": raw, "tables": []}

                all_text.append(obj.get("full_text", ""))
                all_tables.extend(obj.get("tables", []))

                usage = resp.usage_metadata
                total_usage["input_tokens"]    += getattr(usage, "prompt_token_count", 0) or 0
                total_usage["output_tokens"]   += getattr(usage, "candidates_token_count", 0) or 0
                total_usage["thinking_tokens"] += getattr(usage, "thoughts_token_count", 0) or 0
                break
            except Exception as e:
                last_err = e
                if attempt < 2:
                    time.sleep(2.0 * (attempt + 1))

        if last_err and not all_text:
            raise RuntimeError(f"Gemini Vision failed: {last_err}")

    latency = time.perf_counter() - t0
    return ' '.join(all_text), ' '.join(all_tables), latency, total_usage

# ── Main benchmark loop ───────────────────────────────────────────────────────
CHECKPOINT_EVERY = 20  # save intermediate summary every N samples

def _load_existing_csv() -> dict:
    """Load existing CSV results keyed by stem, for --resume mode."""
    csv_path = RESULTS_DIR / "baseline_per_sample.csv"
    if not csv_path.exists():
        return {}
    with open(csv_path, encoding='utf-8') as f:
        return {r['stem']: r for r in csv.DictReader(f)}


def run_benchmark(samples: list[dict], skip_gemini: bool, skip_doctr: bool, skip_pipeline: bool):
    # Resume: load existing rows keyed by stem
    existing = _load_existing_csv() if args.resume else {}
    if existing:
        print(f"[resume] Found {len(existing)} existing rows in CSV")

    rows = []

    # Preload models once
    if not skip_doctr:
        print("[init] Loading docTR (full-page) ...")
        get_doctr()
    if not skip_pipeline:
        print("[init] Loading pipeline models ...")
        get_pipe()

    total = len(samples)
    for idx, s in enumerate(samples):
        pdf_path = s.get('pdf_path')
        stem     = s['llm_stem']
        gt_fname = s['gt_fname']

        print(f"\n[{idx+1}/{total}] {stem}")

        gt_json = STAGING_DIR / f"{gt_fname}_v12_merged.json"
        gt_full, gt_table = load_gt_text(gt_json)
        if not gt_full:
            print(f"  [skip] No GT text")
            continue

        # Resume: check if this stem already has all requested methods done
        if args.resume and stem in existing:
            ex = existing[stem]
            need_p  = not skip_pipeline and ex.get('p_recall', '') == ''
            need_b1 = ex.get('b1_recall', '') == ''
            need_b2 = not skip_doctr   and ex.get('b2_recall', '') == ''
            need_b3 = not skip_gemini  and ex.get('b3_recall', '') == ''
            if not (need_p or need_b1 or need_b2 or need_b3):
                # Convert types back from CSV strings
                ex_row = dict(ex)
                for k in ('has_table',):
                    try: ex_row[k] = ex_row[k] == 'True'
                    except: pass
                rows.append(ex_row)
                print(f"  [resume] all methods done, skipping")
                continue

        row = {
            'stem':      stem,
            'has_table': s['has_table'],
            # cached results from eval CSVs (stages 1-3 run previously)
            'cached_recall': s['pipeline_recall'],
            'cached_tf1':    s.get('pipeline_tf1', ''),
            # live pipeline run
            'p_recall': '', 'p_tf1': '', 'p_latency': '', 'p_cost': '',
            # baselines
            'b1_recall': '', 'b1_tf1': '', 'b1_latency': '',
            'b2_recall': '', 'b2_tf1': '', 'b2_latency': '',
            'b3_recall': '', 'b3_tf1': '', 'b3_latency': '', 'b3_cost': '',
            'error': '',
        }

        if pdf_path is None or not pdf_path.exists():
            print(f"  [warn] PDF not found")
            row['error'] = 'pdf_missing'
            rows.append(row)
            continue

        # Pipeline live
        if not skip_pipeline:
            try:
                p_text, p_table, p_lat, p_cost, _ = run_pipeline_live(pdf_path)
                row['p_recall']  = round(fuzzy_token_recall(p_text, gt_full), 4)
                row['p_tf1']     = round(token_f1(p_table, gt_table), 4) if s['has_table'] and gt_table else ''
                row['p_latency'] = round(p_lat, 3)
                row['p_cost']    = round(p_cost, 6)
                print(f"  Pipeline: recall={row['p_recall']:.3f}  lat={p_lat:.1f}s  cost=${p_cost:.5f}")
            except Exception as e:
                print(f"  Pipeline ERROR: {e}")
                traceback.print_exc()
                row['error'] += f"|pipe:{str(e)[:80]}"

        # B1: PyMuPDF
        try:
            b1_text, b1_lat = extract_pymupdf(pdf_path)
            row['b1_recall']  = round(fuzzy_token_recall(b1_text, gt_full), 4)
            row['b1_tf1']     = round(token_f1(b1_text, gt_table), 4) if s['has_table'] and gt_table else ''
            row['b1_latency'] = round(b1_lat, 3)
            print(f"  B1 PyMuPDF: recall={row['b1_recall']:.3f}  lat={b1_lat:.2f}s")
        except Exception as e:
            print(f"  B1 ERROR: {e}")
            row['error'] += f"|b1:{str(e)[:60]}"

        # B2: docTR full-page
        if not skip_doctr:
            try:
                b2_text, b2_lat = extract_doctr(pdf_path)
                row['b2_recall']  = round(fuzzy_token_recall(b2_text, gt_full), 4)
                row['b2_tf1']     = round(token_f1(b2_text, gt_table), 4) if s['has_table'] and gt_table else ''
                row['b2_latency'] = round(b2_lat, 3)
                print(f"  B2 docTR:  recall={row['b2_recall']:.3f}  lat={b2_lat:.2f}s")
            except Exception as e:
                print(f"  B2 ERROR: {e}")
                row['error'] += f"|b2:{str(e)[:60]}"

        # B3: Gemini Vision
        if not skip_gemini:
            try:
                b3_text, b3_table, b3_lat, b3_usage = extract_gemini_vision(pdf_path)
                b3_cost = (b3_usage['input_tokens'] * INPUT_PRICE +
                           b3_usage['output_tokens'] * OUTPUT_PRICE)
                row['b3_recall']  = round(fuzzy_token_recall(b3_text, gt_full), 4)
                row['b3_tf1']     = round(token_f1(b3_table, gt_table), 4) if s['has_table'] and gt_table else ''
                row['b3_latency'] = round(b3_lat, 3)
                row['b3_cost']    = round(b3_cost, 6)
                print(f"  B3 Gemini: recall={row['b3_recall']:.3f}  lat={b3_lat:.2f}s  cost=${b3_cost:.4f}")
            except Exception as e:
                print(f"  B3 ERROR: {e}")
                row['error'] += f"|b3:{str(e)[:60]}"

        rows.append(row)

        # Checkpoint: save every CHECKPOINT_EVERY samples
        is_checkpoint = ((idx + 1) % CHECKPOINT_EVERY == 0)
        is_last       = (idx + 1 == total)
        if is_checkpoint or is_last:
            label = f"checkpoint-{idx+1}" if not is_last else "final"
            print(f"\n  [checkpoint {label}] saving intermediate results ...")
            _save_results(rows, final=is_last)

    return rows

# ── Save results ──────────────────────────────────────────────────────────────
def _safe_mean(vals):
    vals = [v for v in vals if v != '' and v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None

def _save_results(rows: list[dict], final: bool = True):
    # CSV
    csv_path = RESULTS_DIR / "baseline_per_sample.csv"
    fieldnames = ['stem', 'has_table',
                  'cached_recall', 'cached_tf1',
                  'p_recall', 'p_tf1', 'p_latency', 'p_cost',
                  'b1_recall', 'b1_tf1', 'b1_latency',
                  'b2_recall', 'b2_tf1', 'b2_latency',
                  'b3_recall', 'b3_tf1', 'b3_latency', 'b3_cost',
                  'error']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    if not final:
        return

    # Summary JSON
    def _col(key):
        return [r[key] for r in rows if r[key] != '' and r[key] is not None]

    tbl_rows = [r for r in rows if r['has_table']]

    # ── Cost analysis ─────────────────────────────────────────────────────────
    n = len(rows)
    p_costs   = [float(v) for v in _col('p_cost')]
    b3_costs  = [float(v) for v in _col('b3_cost')]

    p_cost_mean  = _safe_mean(p_costs)  or 0.0
    b3_cost_mean = _safe_mean(b3_costs) or 0.0

    p_cost_total  = round(sum(p_costs),  4)
    b3_cost_total = round(sum(b3_costs), 4) if b3_costs else 0.0

    # Commercial reference (USD/page, source: official pricing pages 2024-2025)
    COMMERCIAL_REF = {
        "Adobe PDF Extract API":  0.0150,   # $0.015/page (Pay-as-you-go)
        "AWS Textract (forms)":   0.0650,   # $0.065/page (Forms + Tables)
        "Google Doc AI":          0.0300,   # ~$0.03/page (Document OCR)
        "Azure Form Recognizer":  0.0100,   # $0.01/page (Read model)
    }

    def _scale(cost_per_page, pages):
        return round(cost_per_page * pages, 2) if cost_per_page is not None else None

    def _savings_vs(pipeline_cost, ref_cost, pages):
        if pipeline_cost is None or ref_cost is None:
            return None
        return round((ref_cost - pipeline_cost) * pages, 2)

    # Build scale table for pipeline
    scales = [1_000, 10_000, 100_000]
    pipeline_scale = {str(s): _scale(p_cost_mean, s) for s in scales}
    b3_scale       = {str(s): _scale(b3_cost_mean, s) for s in scales} if b3_cost_mean else {}

    commercial_scale = {}
    for name, ref in COMMERCIAL_REF.items():
        commercial_scale[name] = {
            "cost_per_page_usd": ref,
            **{str(s): _scale(ref, s) for s in scales},
            **{f"savings_vs_pipeline_{s}": _savings_vs(p_cost_mean, ref, s) for s in scales},
            f"pipeline_cheaper_by_x": round(ref / p_cost_mean, 1) if p_cost_mean else None,
        }

    cost_summary = {
        "pipeline_live": {
            "cost_per_page_usd_mean":  round(p_cost_mean, 6),
            "cost_per_page_usd_min":   round(min(p_costs), 6) if p_costs else None,
            "cost_per_page_usd_max":   round(max(p_costs), 6) if p_costs else None,
            "cost_per_page_usd_std":   round(
                (sum((x - p_cost_mean)**2 for x in p_costs) / len(p_costs))**0.5, 6
            ) if len(p_costs) > 1 else None,
            "scale": pipeline_scale,
            "total_spent_this_run_usd": p_cost_total,
            "n_samples_measured": len(p_costs),
        },
        "b3_gemini_vision": {
            "cost_per_page_usd_mean":  round(b3_cost_mean, 6) if b3_cost_mean else None,
            "scale": b3_scale,
            "total_spent_this_run_usd": b3_cost_total,
            "n_samples_measured": len(b3_costs),
            "vs_pipeline_multiplier": round(b3_cost_mean / p_cost_mean, 2) if p_cost_mean and b3_cost_mean else None,
        },
        "b1_pymupdf": {"cost_per_page_usd": 0.0, "scale": {str(s): 0.0 for s in scales}},
        "b2_doctr":   {"cost_per_page_usd": 0.0, "scale": {str(s): 0.0 for s in scales}},
        "commercial_reference": commercial_scale,
        "total_benchmark_spend_usd": round(p_cost_total + b3_cost_total, 4),
    }

    summary = {
        "n_samples": n,
        "n_with_table": sum(1 for r in rows if r['has_table']),
        "pipeline_cached": {
            "content_recall_mean": _safe_mean(_col('cached_recall')),
            "table_f1_mean":       _safe_mean([r['cached_tf1'] for r in tbl_rows if r.get('cached_tf1') not in ('', None)]),
            "cost_per_page_usd":   0.0038,
            "note":                "cached from eval CSVs (485-sample run)",
        },
        "pipeline_live": {
            "content_recall_mean": _safe_mean(_col('p_recall')),
            "table_f1_mean":       _safe_mean([r['p_tf1'] for r in tbl_rows if r.get('p_tf1') not in ('', None)]),
            "latency_s_mean":      _safe_mean(_col('p_latency')),
            "cost_per_page_usd":   p_cost_mean,
        },
        "b1_pymupdf": {
            "content_recall_mean": _safe_mean(_col('b1_recall')),
            "table_f1_mean":       _safe_mean([r['b1_tf1'] for r in tbl_rows if r['b1_tf1'] != '']),
            "latency_s_mean":      _safe_mean(_col('b1_latency')),
            "cost_per_page_usd":   0.0,
        },
        "b2_doctr": {
            "content_recall_mean": _safe_mean(_col('b2_recall')),
            "table_f1_mean":       _safe_mean([r['b2_tf1'] for r in tbl_rows if r['b2_tf1'] != '']),
            "latency_s_mean":      _safe_mean(_col('b2_latency')),
            "cost_per_page_usd":   0.0,
        },
        "b3_gemini_vision": {
            "content_recall_mean": _safe_mean(_col('b3_recall')),
            "table_f1_mean":       _safe_mean([r['b3_tf1'] for r in tbl_rows if r['b3_tf1'] != '']),
            "latency_s_mean":      _safe_mean(_col('b3_latency')),
            "cost_per_page_usd":   b3_cost_mean or None,
        },
        "cost_analysis": cost_summary,
    }

    json_path = RESULTS_DIR / "baseline_summary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Human-readable comparison table
    _write_comparison_table(summary)

    print(f"\n[save] {csv_path}")
    print(f"[save] {json_path}")
    print(f"[save] {RESULTS_DIR / 'baseline_comparison_table.txt'}")

def _write_comparison_table(summary: dict):
    lines = [
        "=" * 80,
        "  BASELINE COMPARISON — Content Recall & Table F1",
        f"  n={summary['n_samples']} samples  |  n_table={summary['n_with_table']} samples with table",
        "=" * 80,
        f"{'Method':<25} {'Content Recall':>15} {'Table F1':>10} {'Cost/page':>12} {'Latency(s)':>12}",
        "-" * 80,
    ]

    def fmt(d, lat_key='latency_s_mean'):
        cr  = f"{d['content_recall_mean']:.3f}" if d.get('content_recall_mean') is not None else "N/A"
        tf1 = f"{d['table_f1_mean']:.3f}"       if d.get('table_f1_mean')       is not None else "N/A"
        cost= f"${d['cost_per_page_usd']:.4f}"  if d.get('cost_per_page_usd')   is not None else "N/A"
        lat = f"{d[lat_key]:.2f}s"              if d.get(lat_key)               is not None else "—"
        return cr, tf1, cost, lat

    pc = summary.get('pipeline_cached', {})
    pc_cr  = f"{pc['content_recall_mean']:.3f}" if pc.get('content_recall_mean') is not None else "N/A"
    pc_tf  = f"{pc['table_f1_mean']:.3f}"       if pc.get('table_f1_mean')       is not None else "N/A"
    lines.append(f"{'Pipeline (cached)':<25} {pc_cr:>15} {pc_tf:>10} {'$0.0038':>12} {'—':>12}")

    pl = summary.get('pipeline_live', {})
    pl_cr  = f"{pl['content_recall_mean']:.3f}" if pl.get('content_recall_mean') is not None else "N/A"
    pl_tf  = f"{pl['table_f1_mean']:.3f}"       if pl.get('table_f1_mean')       is not None else "N/A"
    pl_lat = f"{pl['latency_s_mean']:.1f}s"     if pl.get('latency_s_mean')      is not None else "—"
    pl_c   = f"${pl['cost_per_page_usd']:.4f}"  if pl.get('cost_per_page_usd')   is not None else "—"
    lines.append(f"{'Pipeline (live)':<25} {pl_cr:>15} {pl_tf:>10} {pl_c:>12} {pl_lat:>12}")

    b1 = fmt(summary['b1_pymupdf'])
    lines.append(f"{'PyMuPDF/pdfplumber':<25} {b1[0]:>15} {b1[1]:>10} {b1[2]:>12} {b1[3]:>12}")

    b2 = fmt(summary['b2_doctr'])
    lines.append(f"{'docTR full-page OCR':<25} {b2[0]:>15} {b2[1]:>10} {b2[2]:>12} {b2[3]:>12}")

    b3 = fmt(summary['b3_gemini_vision'])
    lines.append(f"{'Gemini Vision full-page':<25} {b3[0]:>15} {b3[1]:>10} {b3[2]:>12} {b3[3]:>12}")

    # ── Cost detail section ────────────────────────────────────────────────────
    ca  = summary.get("cost_analysis", {})
    pl  = ca.get("pipeline_live", {})
    b3  = ca.get("b3_gemini_vision", {})
    com = ca.get("commercial_reference", {})

    def _u(v, fmt=".6f"): return f"${v:{fmt}}" if v is not None else "N/A"
    def _u2(v):           return f"${v:.2f}"   if v is not None else "N/A"
    def _u4(v):           return f"${v:.4f}"   if v is not None else "N/A"

    lines += ["=" * 80, "", "COST ANALYSIS:", "-" * 80]

    # Per-page stats
    lines.append(f"  {'Method':<30} {'Cost/page':>12} {'Min':>12} {'Max':>12} {'Std':>12}")
    lines.append(f"  {'-'*70}")
    lines.append(f"  {'Pipeline (live)':<30} {_u4(pl.get('cost_per_page_usd_mean')):>12} "
                 f"{_u4(pl.get('cost_per_page_usd_min')):>12} "
                 f"{_u4(pl.get('cost_per_page_usd_max')):>12} "
                 f"{_u4(pl.get('cost_per_page_usd_std')):>12}")
    if b3.get('cost_per_page_usd_mean'):
        lines.append(f"  {'Gemini Vision':<30} {_u4(b3.get('cost_per_page_usd_mean')):>12} "
                     f"{'—':>12} {'—':>12} {'—':>12}")
    lines.append(f"  {'PyMuPDF / docTR':<30} {'$0.0000':>12} {'$0.0000':>12} {'$0.0000':>12} {'$0.0000':>12}")

    # Scale table
    lines += ["", f"  Cost at scale (USD):"]
    lines.append(f"  {'Method':<30} {'1,000 pages':>14} {'10,000 pages':>14} {'100,000 pages':>15}")
    lines.append(f"  {'-'*75}")
    p_sc = pl.get('scale', {})
    b3_sc = b3.get('scale', {})
    lines.append(f"  {'Pipeline (live)':<30} {_u2(p_sc.get('1000')):>14} {_u2(p_sc.get('10000')):>14} {_u2(p_sc.get('100000')):>15}")
    if b3_sc:
        lines.append(f"  {'Gemini Vision':<30} {_u2(b3_sc.get('1000')):>14} {_u2(b3_sc.get('10000')):>14} {_u2(b3_sc.get('100000')):>15}")
    lines.append(f"  {'PyMuPDF / docTR':<30} {'$0.00':>14} {'$0.00':>14} {'$0.00':>15}")

    # Commercial comparison
    if com:
        lines += ["", f"  Comparison with commercial services (pipeline cheaper by X):"]
        lines.append(f"  {'Service':<30} {'Cost/page':>12} {'1K pages':>12} {'10K pages':>12} {'100K pages':>13} {'Cheaper':>8}")
        lines.append(f"  {'-'*90}")
        for svc, d in com.items():
            mult = d.get('pipeline_cheaper_by_x')
            mult_str = f"{mult:.1f}x" if mult else "N/A"
            lines.append(f"  {svc:<30} {_u4(d.get('cost_per_page_usd')):>12} "
                         f"{_u2(d.get('1000')):>12} {_u2(d.get('10000')):>12} "
                         f"{_u2(d.get('100000')):>13} {mult_str:>8}")
        lines.append(f"  {'Pipeline (live)':<30} {_u4(pl.get('cost_per_page_usd_mean')):>12} "
                     f"{_u2(p_sc.get('1000')):>12} {_u2(p_sc.get('10000')):>12} "
                     f"{_u2(p_sc.get('100000')):>13} {'(baseline)':>8}")

    # Total spent
    total_spent = ca.get('total_benchmark_spend_usd', 0)
    lines += [
        "",
        f"  Total benchmark spend:  Pipeline={_u2(pl.get('total_spent_this_run_usd'))}  "
        f"Gemini Vision={_u2(b3.get('total_spent_this_run_usd'))}  "
        f"TOTAL={_u2(total_spent)}",
        f"  (n_pipeline={pl.get('n_samples_measured',0)}  n_gemini_vision={b3.get('n_samples_measured',0)})",
    ]

    lines += [
        "",
        "=" * 80,
        "",
        "Notes:",
        "  Content Recall = fraction of GT tokens present in output (fuzzy matching)",
        "  Table F1        = token F1 on table content",
        "  Pipeline cost   = input_tokens * $0.30/M + output_tokens * $2.50/M (Gemini 2.5 Flash)",
        "  Commercial ref  = Adobe PDF Extract, AWS Textract, Google Doc AI, Azure Form Recognizer",
        "  UPPER BOUND NOTE: Pipeline recall/F1 has circular dependency (GT = Gemini GT)",
        "    -> actual results vs human annotation may be lower",
    ]

    txt_path = RESULTS_DIR / "baseline_comparison_table.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print('\n' + '\n'.join(lines))

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print(f"[config] n={args.n}  skip_pipeline={args.skip_pipeline}  "
          f"skip_gemini={args.skip_gemini}  skip_doctr={args.skip_doctr}")
    print(f"[config] checkpoint every {CHECKPOINT_EVERY} samples")
    print(f"[config] output -> {RESULTS_DIR}")

    samples = select_top_samples(args.n)
    rows    = run_benchmark(samples, args.skip_gemini, args.skip_doctr, args.skip_pipeline)
    _save_results(rows, final=True)

    print(f"\nDone. {len(rows)} samples processed.")

if __name__ == '__main__':
    main()
