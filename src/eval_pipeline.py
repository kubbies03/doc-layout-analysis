"""
Evaluation script for Pipeline C (YOLO + Doctr).

Metrics:
  1. mAP@0.5 per class     — from training results (printed from ablation CSV)
  2. Block P/R/F1          — per sample, per class: how many GT blocks detected
  3. Per-class CER         — OCR quality per block type (IoU-overlap matched)
  4. Table structure score — GT Table blocks: was the region detected at all?

Run:
  python eval_pipeline_c.py [--limit N] [--random] [--seed 42]

Output:
  eval_results/per_sample.csv     — one row per sample
  eval_results/per_class.csv      — Block F1 + CER per class (aggregated)
  eval_results/summary.json       — overall numbers
  eval_results/progress.txt       — resume support
"""

import os, sys, io, json, csv, time, argparse, traceback
import random as _random
from pathlib import Path
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────
STAGING_DIR  = Path(r"C:\Users\HP\Desktop\Kubbies\KhoaLuan\GT\staging")
RESULTS_DIR  = Path(r"C:\Users\HP\Desktop\Kubbies\KhoaLuan\eval_results")
ABLATION_CSV = Path(r"C:\Users\HP\Desktop\Kubbies\KhoaLuan\dataset\final_Base\ablation_full_results.csv")
YOLO_WEIGHTS = r"C:\Users\HP\Desktop\Kubbies\KhoaLuan\dataset\BaseC\best.pt"
POPPLER_PATH = r"C:\Users\HP\Desktop\Kubbies\KhoaLuan\poppler\Library\bin"

PDF_DPI  = 150
GT_IMGSZ = 1025
MIN_AREA = 800.0
IOU_MATCH     = 0.5   # for Block F1
OVERLAP_MIN   = 0.1   # for per-class CER (recall-oriented)

CLASS_NAMES = ['Caption','Footnote','Formula','List-item','Page-footer',
               'Page-header','Picture','Section-header','Table','Text','Title']
TEXT_CLASSES = {'Caption','Footnote','List-item','Page-footer',
                'Page-header','Section-header','Text','Title'}
YOLO_CONF = 0.20
YOLO_IOU  = 0.60
YOLO_IMGSZ = 640
CLASS_CONF_OVERRIDE = {'Picture': 0.50, 'Section-header': 0.40, 'Caption': 0.40}

# ── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--limit",   type=int, default=None)
ap.add_argument("--random",  action="store_true")
ap.add_argument("--seed",    type=int, default=42)
args, _ = ap.parse_known_args()

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── mAP summary from ablation CSV ────────────────────────────────────────────
def print_map_summary():
    if not ABLATION_CSV.exists():
        print("[mAP] ablation_full_results.csv not found, skipping.")
        return
    import csv as _csv
    rows = list(_csv.DictReader(open(ABLATION_CSV, encoding='utf-8')))
    # Find Exp C, Test split
    target = None
    for r in rows:
        if r.get('Run','').strip() == 'C' and r.get('Split','').strip() == 'Test':
            target = r; break
    if not target:
        target = rows[-1]
    print("\n" + "="*60)
    print("  Metric 1 -- mAP@0.5 per class (Exp C, Test set)")
    print("="*60)
    print(f"  {'mAP@0.5':<20} {target.get('mAP@0.5','')}")
    print(f"  {'mAP@0.5:0.95':<20} {target.get('mAP@0.5:0.95','')}")
    print(f"  {'Precision':<20} {target.get('Precision','')}")
    print(f"  {'Recall':<20} {target.get('Recall','')}")
    print()
    print(f"  {'Class':<18} {'AP@0.5':>8}")
    print(f"  {'-'*28}")
    for c in CLASS_NAMES:
        print(f"  {c:<18} {target.get(c,'N/A'):>8}")
    print()

# ── Helpers ───────────────────────────────────────────────────────────────────
def _edit_distance(a, b):
    m, n = len(a), len(b)
    dp = list(range(n+1))
    for i in range(1, m+1):
        prev, dp[0] = dp[0], i
        for j in range(1, n+1):
            temp = dp[j]
            dp[j] = prev if a[i-1]==b[j-1] else 1+min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]

def cer(ref, hyp):
    ref, hyp = ref.strip(), hyp.strip()
    if not ref: return 0.0 if not hyp else 1.0
    return min(_edit_distance(ref, hyp)/len(ref), 1.0)

def iou(a, b):
    ax1,ay1,ax2,ay2 = a[0],a[1],a[0]+a[2],a[1]+a[3]
    bx1,by1,bx2,by2 = b[0],b[1],b[0]+b[2],b[1]+b[3]
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter/union if union>0 else 0.0

def overlap_ratio(gt_bb, pipe_bb):
    """Fraction of GT bbox covered by pipe_bb (recall-oriented)."""
    ax1,ay1 = gt_bb[0],gt_bb[1]
    ax2,ay2 = ax1+gt_bb[2], ay1+gt_bb[3]
    bx1,by1 = pipe_bb[0],pipe_bb[1]
    bx2,by2 = bx1+pipe_bb[2], by1+pipe_bb[3]
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    gt_area = gt_bb[2]*gt_bb[3]
    return inter/gt_area if gt_area>0 else 0.0

def pdf_to_pil(pdf_path):
    from pdf2image import convert_from_path
    return convert_from_path(str(pdf_path), dpi=PDF_DPI, poppler_path=POPPLER_PATH)

def letterbox(img, target=GT_IMGSZ):
    import cv2, numpy as np
    h,w = img.shape[:2]
    scale = target/max(h,w)
    nw,nh = int(w*scale), int(h*scale)
    resized = cv2.resize(img,(nw,nh),interpolation=cv2.INTER_AREA)
    canvas = np.full((target,target,3),255,dtype=np.uint8)
    canvas[:nh,:nw] = resized
    return canvas, nh, nw

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
        print(f"  Doctr loaded (gpu=True).")
    return _doctr

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
        x1,y1,x2,y2 = box.xyxy[0].tolist()
        w,h = x2-x1, y2-y1
        if w*h < MIN_AREA: continue
        blocks.append({"label_name": label, "bbox_coco": [x1,y1,w,h], "conf": conf, "text": ""})
    return blocks

def ocr_crops_batch(img_bgr, bboxes):
    """OCR list of (x,y,w,h) crops in one Doctr forward pass."""
    import numpy as np
    from PIL import Image as PILImage
    from doctr.io import DocumentFile

    def crop_png(x,y,w,h):
        x1,y1,x2,y2 = int(x),int(y),int(x+w),int(y+h)
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size==0: return None
        import cv2
        pil = PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO(); pil.save(buf,format="PNG")
        return buf.getvalue()

    png_list = [crop_png(*bb) for bb in bboxes]
    valid = [(i,p) for i,p in enumerate(png_list) if p]
    texts = [""]*len(bboxes)
    if not valid: return texts

    model = get_doctr()
    doc = DocumentFile.from_images([p for _,p in valid])
    result = model(doc)
    for page_i, (src_i,_) in enumerate(valid):
        words = []
        for blk in result.pages[page_i].blocks:
            for line in blk.lines:
                for word in line.words:
                    if word.value.strip(): words.append(word.value)
        texts[src_i] = " ".join(words)
    return texts

# ── Scale GT coords 1025→canvas ───────────────────────────────────────────────
def scale_gt_bb(gt_bb, actual_h, actual_w):
    """Scale GT bbox (1025×1025 space) to letterboxed canvas coords."""
    sx = actual_w / GT_IMGSZ
    sy = actual_h / GT_IMGSZ
    x,y,w,h = gt_bb
    return [x*sx, y*sy, w*sx, h*sy]

# ── Metric 2: Block P/R/F1 per class ─────────────────────────────────────────
def eval_block_metrics(gt_data, pipe_blocks, actual_h, actual_w):
    """
    Returns per-class {TP, FP, FN} and overall P/R/F1.
    GT coords are in 1025 space; pipe_blocks are in canvas (letterboxed) space.
    """
    per_class = {c: {"tp":0,"fp":0,"fn":0} for c in CLASS_NAMES}

    for page in gt_data.get("pages",[]):
        gt_anns = page.get("layout_gt",[])
        # scale GT bboxes to canvas coords
        gt_scaled = []
        for ann in gt_anns:
            bb = scale_gt_bb(ann.get("bbox_coco",[0,0,0,0]), actual_h, actual_w)
            gt_scaled.append({"label": ann.get("label_name","?"), "bbox": bb, "matched": False})

        page_pipes = [b for b in pipe_blocks]  # all pipe blocks (single-page assumption)
        matched_pipe = set()

        for gi, gt in enumerate(gt_scaled):
            best_iou, best_pi = 0.0, -1
            for pi, pb in enumerate(page_pipes):
                if pb["label_name"] != gt["label"]: continue
                if pi in matched_pipe: continue
                v = iou(gt["bbox"], pb["bbox_coco"])
                if v > best_iou:
                    best_iou, best_pi = v, pi
            if best_iou >= IOU_MATCH:
                per_class[gt["label"]]["tp"] += 1
                gt_scaled[gi]["matched"] = True
                matched_pipe.add(best_pi)
            else:
                per_class[gt["label"]]["fn"] += 1

        for pi, pb in enumerate(page_pipes):
            if pi not in matched_pipe:
                per_class[pb["label_name"]]["fp"] += 1

    # Overall
    total_tp = sum(v["tp"] for v in per_class.values())
    total_fp = sum(v["fp"] for v in per_class.values())
    total_fn = sum(v["fn"] for v in per_class.values())
    prec = total_tp/(total_tp+total_fp) if (total_tp+total_fp)>0 else 0.0
    rec  = total_tp/(total_tp+total_fn) if (total_tp+total_fn)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return per_class, prec, rec, f1

# ── Metric 3: Per-class CER ───────────────────────────────────────────────────
def eval_per_class_cer(gt_data, pipe_blocks, actual_h, actual_w):
    """
    For each GT block with text, find pipe block with best spatial overlap.
    Returns {class: {cer_sum, count, missed}}.
    """
    results = {c: {"cer_sum":0.0,"count":0,"missed":0} for c in CLASS_NAMES}

    for page in gt_data.get("pages",[]):
        for ann in page.get("layout_gt",[]):
            label    = ann.get("label_name","?")
            gt_text  = ann.get("text","").strip()
            if not gt_text: continue
            gt_bb    = scale_gt_bb(ann.get("bbox_coco",[0,0,0,0]), actual_h, actual_w)

            best_ratio, best_text = 0.0, ""
            for pb in pipe_blocks:
                r = overlap_ratio(gt_bb, pb["bbox_coco"])
                if r > best_ratio:
                    best_ratio, best_text = r, pb.get("text","")

            if best_ratio >= OVERLAP_MIN and best_text.strip():
                results[label]["cer_sum"] += cer(gt_text, best_text.strip())
                results[label]["count"]   += 1
            else:
                results[label]["missed"]  += 1
    return results

# ── Metric 4: Table detection rate ───────────────────────────────────────────
def eval_table_detection(gt_data, pipe_blocks, actual_h, actual_w):
    """
    For each GT Table block, check if any pipe block overlaps ≥ 0.3.
    Returns (detected, total).
    """
    detected, total = 0, 0
    for page in gt_data.get("pages",[]):
        for ann in page.get("layout_gt",[]):
            if ann.get("label_name") != "Table": continue
            total += 1
            gt_bb = scale_gt_bb(ann.get("bbox_coco",[0,0,0,0]), actual_h, actual_w)
            pipe_tables = [b for b in pipe_blocks if b["label_name"]=="Table"]
            best = max((overlap_ratio(gt_bb, b["bbox_coco"]) for b in pipe_tables), default=0.0)
            if best >= 0.3:
                detected += 1
    return detected, total

# ── Pipeline C: run one PDF ───────────────────────────────────────────────────
def run_pipeline_c(pdf_path):
    import cv2, numpy as np
    pil_pages = pdf_to_pil(pdf_path)
    all_blocks = []
    actual_hw  = (GT_IMGSZ, GT_IMGSZ)  # default if single page
    text_parts = []

    for pil in pil_pages:
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        canvas, actual_h, actual_w = letterbox(img)
        actual_hw = (actual_h, actual_w)
        blocks = detect_blocks(canvas)

        text_idx = [i for i,b in enumerate(blocks) if b["label_name"] in TEXT_CLASSES]
        bboxes   = [blocks[i]["bbox_coco"] for i in text_idx]
        texts    = ocr_crops_batch(canvas, bboxes) if bboxes else []
        for k, i in enumerate(text_idx):
            blocks[i]["text"] = texts[k]
            if texts[k]: text_parts.append(texts[k])
        all_blocks.extend(blocks)

    return all_blocks, actual_hw, " ".join(text_parts)

# ── CSV setup ─────────────────────────────────────────────────────────────────
SAMPLE_FIELDS = (
    ["stem", "n_gt_blocks", "n_pipe_blocks", "prec", "rec", "f1",
     "table_detected", "table_total", "elapsed"]
    + [f"f1_{c.lower().replace('-','_').replace(' ','_')}" for c in CLASS_NAMES]
    + [f"cer_{c.lower().replace('-','_').replace(' ','_')}" for c in CLASS_NAMES]
    + [f"cov_{c.lower().replace('-','_').replace(' ','_')}" for c in CLASS_NAMES]
)

PROGRESS_FILE = RESULTS_DIR / "progress.txt"
SAMPLE_CSV    = RESULTS_DIR / "per_sample.csv"
PERCLASS_CSV  = RESULTS_DIR / "per_class.csv"
SUMMARY_JSON  = RESULTS_DIR / "summary.json"

done_stems = set()
if PROGRESS_FILE.exists():
    done_stems = set(PROGRESS_FILE.read_text(encoding="utf-8").splitlines())

csv_is_new = not SAMPLE_CSV.exists()
csv_fh  = open(SAMPLE_CSV,  "a", newline="", encoding="utf-8")
csv_w   = csv.DictWriter(csv_fh, fieldnames=SAMPLE_FIELDS)
if csv_is_new: csv_w.writeheader()

prog_fh = open(PROGRESS_FILE, "a", encoding="utf-8")

# ── Accumulators ──────────────────────────────────────────────────────────────
agg_prec, agg_rec, agg_f1 = [], [], []
agg_table_det, agg_table_tot = 0, 0
agg_per_class  = {c: {"tp":0,"fp":0,"fn":0} for c in CLASS_NAMES}
agg_cer        = {c: {"cer_sum":0.0,"count":0,"missed":0} for c in CLASS_NAMES}

# ── Sample list ───────────────────────────────────────────────────────────────
all_pdfs = sorted(STAGING_DIR.glob("*.pdf"))
if args.limit:
    if args.random:
        rng = _random.Random(args.seed)
        all_pdfs = sorted(rng.sample(all_pdfs, min(args.limit, len(all_pdfs))))
    else:
        all_pdfs = all_pdfs[:args.limit]

# ── Print Metric 1 (mAP from ablation) ───────────────────────────────────────
print_map_summary()

print(f"\nMetrics 2-4 — running Pipeline C on {len(all_pdfs)} samples")
print(f"Results -> {RESULTS_DIR}\n")

CHECKPOINT_EVERY = 50

def print_checkpoint(n_done):
    avg = lambda lst: sum(lst)/len(lst) if lst else float("nan")
    overall_f1 = avg(agg_f1)
    overall_p  = avg(agg_prec)
    overall_r  = avg(agg_rec)
    tbl_rate   = agg_table_det/agg_table_tot if agg_table_tot>0 else float("nan")

    print(f"\n{'='*72}")
    print(f"  CHECKPOINT @ {n_done} samples")
    print(f"{'='*72}")
    print(f"  Block Detection  Prec={overall_p:.3f}  Rec={overall_r:.3f}  F1={overall_f1:.3f}")
    print(f"  Table Detection  {agg_table_det}/{agg_table_tot} = {tbl_rate:.1%}" if agg_table_tot>0 else "  Table Detection  N/A")
    print()
    print(f"  {'Class':<18} {'F1':>6}  {'CER':>7}  {'Coverage':>9}")
    print(f"  {'-'*48}")
    for c in CLASS_NAMES:
        v  = agg_per_class[c]
        tp,fp,fn = v["tp"],v["fp"],v["fn"]
        p_ = tp/(tp+fp) if (tp+fp)>0 else float("nan")
        r_ = tp/(tp+fn) if (tp+fn)>0 else float("nan")
        f_ = 2*p_*r_/(p_+r_) if (p_+r_)>0 else float("nan")
        cv = agg_cer[c]
        cer_avg = cv["cer_sum"]/cv["count"] if cv["count"]>0 else float("nan")
        cov = cv["count"]/(cv["count"]+cv["missed"]) if (cv["count"]+cv["missed"])>0 else float("nan")
        f_str   = f"{f_:6.3f}" if f_==f_ else "   N/A"
        cer_str = f"{cer_avg:7.4f}" if cer_avg==cer_avg else "    N/A"
        cov_str = f"{cov:8.1%}" if cov==cov else "     N/A"
        if v["tp"]+v["fp"]+v["fn"]+cv["count"]+cv["missed"] == 0:
            continue
        print(f"  {c:<18} {f_str}  {cer_str}  {cov_str}")
    print(f"{'='*72}\n")

# ── Main loop ─────────────────────────────────────────────────────────────────
n_processed = 0
for idx, pdf_path in enumerate(all_pdfs):
    stem = pdf_path.stem
    if stem in done_stems:
        print(f"[{idx+1:3}/{len(all_pdfs)}] {stem[7:15]} SKIP")
        continue

    gt_path = STAGING_DIR / f"{stem}_v12_merged.json"
    if not gt_path.exists():
        print(f"[{idx+1:3}/{len(all_pdfs)}] {stem[7:15]} NO GT")
        continue

    try:
        gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[{idx+1:3}/{len(all_pdfs)}] {stem[7:15]} GT_ERR {e}")
        continue
    n_gt = sum(len(p.get("layout_gt",[])) for p in gt_data.get("pages",[]))

    try:
        t0 = time.time()
        pipe_blocks, (actual_h, actual_w), _ = run_pipeline_c(pdf_path)
        elapsed = time.time() - t0

        # Metric 2: Block P/R/F1
        per_cls_cnt, prec, rec, f1 = eval_block_metrics(gt_data, pipe_blocks, actual_h, actual_w)

        # Metric 3: Per-class CER
        per_cls_cer = eval_per_class_cer(gt_data, pipe_blocks, actual_h, actual_w)

        # Metric 4: Table detection
        tbl_det, tbl_tot = eval_table_detection(gt_data, pipe_blocks, actual_h, actual_w)

        # Accumulate
        agg_prec.append(prec); agg_rec.append(rec); agg_f1.append(f1)
        agg_table_det += tbl_det; agg_table_tot += tbl_tot
        for c in CLASS_NAMES:
            for k in ("tp","fp","fn"):
                agg_per_class[c][k] += per_cls_cnt[c][k]
            agg_cer[c]["cer_sum"] += per_cls_cer[c]["cer_sum"]
            agg_cer[c]["count"]   += per_cls_cer[c]["count"]
            agg_cer[c]["missed"]  += per_cls_cer[c]["missed"]

        # CSV row
        row = {"stem": stem[7:15], "n_gt_blocks": n_gt,
               "n_pipe_blocks": len(pipe_blocks),
               "prec": round(prec,4), "rec": round(rec,4), "f1": round(f1,4),
               "table_detected": tbl_det, "table_total": tbl_tot,
               "elapsed": round(elapsed,1)}
        for c in CLASS_NAMES:
            key = c.lower().replace('-','_').replace(' ','_')
            cnt = per_cls_cnt[c]
            p2  = cnt["tp"]/(cnt["tp"]+cnt["fp"]) if (cnt["tp"]+cnt["fp"])>0 else float("nan")
            r2  = cnt["tp"]/(cnt["tp"]+cnt["fn"]) if (cnt["tp"]+cnt["fn"])>0 else float("nan")
            f2  = 2*p2*r2/(p2+r2) if (p2+r2)>0 else float("nan")
            row[f"f1_{key}"] = round(f2,4) if f2==f2 else ""
            cc  = per_cls_cer[c]
            avg_cer = cc["cer_sum"]/cc["count"] if cc["count"]>0 else float("nan")
            cov = cc["count"]/(cc["count"]+cc["missed"]) if (cc["count"]+cc["missed"])>0 else float("nan")
            row[f"cer_{key}"] = round(avg_cer,4) if avg_cer==avg_cer else ""
            row[f"cov_{key}"] = round(cov,4)     if cov==cov         else ""

        csv_w.writerow(row); csv_fh.flush()
        prog_fh.write(stem+"\n"); prog_fh.flush()
        n_processed += 1

        print(f"[{idx+1:3}/{len(all_pdfs)}] {stem[7:15]}  "
              f"GT={n_gt:3d} Pipe={len(pipe_blocks):3d}  "
              f"P={prec:.2f} R={rec:.2f} F1={f1:.2f}  "
              f"Tbl={tbl_det}/{tbl_tot}  {elapsed:.1f}s")

        if n_processed % CHECKPOINT_EVERY == 0:
            print_checkpoint(n_processed)

    except Exception as e:
        print(f"[{idx+1:3}/{len(all_pdfs)}] {stem[7:15]} ERROR: {e}")
        traceback.print_exc()

csv_fh.close(); prog_fh.close()

# ── Per-class summary ─────────────────────────────────────────────────────────
avg = lambda lst: sum(lst)/len(lst) if lst else float("nan")

print(f"\n{'='*72}")
print(f"  Metric 2 — Block Detection F1 per class")
print(f"{'='*72}")
print(f"{'Class':<18} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>6} {'Rec':>6} {'F1':>6}")
print(f"{'-'*72}")
for c in CLASS_NAMES:
    v  = agg_per_class[c]
    tp,fp,fn = v["tp"],v["fp"],v["fn"]
    p  = tp/(tp+fp) if (tp+fp)>0 else float("nan")
    r  = tp/(tp+fn) if (tp+fn)>0 else float("nan")
    f  = 2*p*r/(p+r) if (p+r)>0 else float("nan")
    def fmt(x): return f"{x:6.3f}" if x==x else "   N/A"
    print(f"{c:<18} {tp:>5} {fp:>5} {fn:>5} {fmt(p)} {fmt(r)} {fmt(f)}")

overall_p = avg(agg_prec); overall_r = avg(agg_rec); overall_f1 = avg(agg_f1)
print(f"\n  Overall:  Prec={overall_p:.3f}  Rec={overall_r:.3f}  F1={overall_f1:.3f}")

print(f"\n{'='*72}")
print(f"  Metric 3 — Per-class CER (OCR quality per block type)")
print(f"{'='*72}")
print(f"{'Class':<18} {'Avg CER':>8} {'N matched':>10} {'N missed':>9} {'Coverage':>9}")
print(f"{'-'*72}")
for c in CLASS_NAMES:
    v = agg_cer[c]
    if v["count"]+v["missed"] == 0: continue
    avg_cer = v["cer_sum"]/v["count"] if v["count"]>0 else float("nan")
    cov = v["count"]/(v["count"]+v["missed"])
    def fmt(x): return f"{x:8.4f}" if x==x else "     N/A"
    print(f"{c:<18} {fmt(avg_cer)} {v['count']:>10} {v['missed']:>9} {cov:>8.1%}")

print(f"\n{'='*72}")
tbl_rate = agg_table_det/agg_table_tot if agg_table_tot>0 else float("nan")
print(f"  Metric 4 — Table Detection Rate: {agg_table_det}/{agg_table_tot} = {tbl_rate:.1%}")
print(f"{'='*72}\n")

# ── Save per_class.csv ────────────────────────────────────────────────────────
with open(PERCLASS_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["class","tp","fp","fn","prec","rec","f1",
                                       "avg_cer","n_matched","n_missed","coverage"])
    w.writeheader()
    for c in CLASS_NAMES:
        v  = agg_per_class[c]
        tp,fp,fn = v["tp"],v["fp"],v["fn"]
        p  = tp/(tp+fp) if (tp+fp)>0 else None
        r  = tp/(tp+fn) if (tp+fn)>0 else None
        f  = 2*p*r/(p+r) if (p and r and p+r>0) else None
        cv = agg_cer[c]
        avg_cer = cv["cer_sum"]/cv["count"] if cv["count"]>0 else None
        cov = cv["count"]/(cv["count"]+cv["missed"]) if (cv["count"]+cv["missed"])>0 else None
        w.writerow({"class":c,"tp":tp,"fp":fp,"fn":fn,
                    "prec": round(p,4) if p else "",
                    "rec":  round(r,4) if r else "",
                    "f1":   round(f,4) if f else "",
                    "avg_cer": round(avg_cer,4) if avg_cer is not None else "",
                    "n_matched": cv["count"], "n_missed": cv["missed"],
                    "coverage": round(cov,4) if cov else ""})

# ── Save summary.json ─────────────────────────────────────────────────────────
summary = {
    "n_samples": len(agg_prec),
    "overall_prec": round(overall_p,4),
    "overall_rec":  round(overall_r,4),
    "overall_f1":   round(overall_f1,4),
    "table_detection_rate": round(tbl_rate,4) if tbl_rate==tbl_rate else None,
    "table_detected": agg_table_det,
    "table_total":    agg_table_tot,
}
SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(f"Saved: {PERCLASS_CSV}")
print(f"Saved: {SAMPLE_CSV}")
print(f"Saved: {SUMMARY_JSON}")
