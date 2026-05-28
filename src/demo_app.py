import os
import sys
import time
import tempfile
import json

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image as PILImage

# Add project root to path for benchmark_pipeline import
sys.path.insert(0, os.path.dirname(__file__))

from benchmark_pipeline import (
    get_yolo, get_doctr, get_client,
    pdf_to_pil, letterbox, detect_blocks,
    ocr_crops_batch, assign_reading_order,
    assemble_and_summarize,
    CLASS_NAMES, TEXT_CLASSES, TABLE_CLASSES,
    INPUT_PRICE, OUTPUT_PRICE,
    YOLO_WEIGHTS, GEMINI_MODEL,
    BBox, LayoutBlock,
)

LLM_DISPLAY = f"Gemini 2.5 Flash ({GEMINI_MODEL})"
PRICING_DISPLAY = "$0.30/1M in · $2.50/1M out"

# ── Constants ─────────────────────────────────────────────────────────────────

CLASS_COLORS_BGR = {
    "Text":           (180, 130,  70),
    "Title":          ( 60,  20, 220),
    "Section-header": (  0, 140, 255),
    "List-item":      (113, 179,  60),
    "Caption":        (211,  85, 186),
    "Footnote":       (169, 169, 169),
    "Formula":        (  0, 215, 255),
    "Table":          (  0,  69, 255),
    "Picture":        (255, 191,   0),
    "Page-header":    (100, 220, 100),
    "Page-footer":    (193, 182, 255),
}
DEFAULT_COLOR = (128, 128, 128)

# ── Model loading (cached across reruns) ──────────────────────────────────────

@st.cache_resource(show_spinner="Loading YOLO + docTR models...")
def load_models():
    get_yolo()
    get_doctr()
    get_client()

# ── Annotation drawing ────────────────────────────────────────────────────────

def draw_annotations(canvas_bgr: np.ndarray, blocks: list) -> PILImage.Image:
    vis = canvas_bgr.copy()
    for b in blocks:
        x, y, w, h = b["bbox_coco"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        label = b["label_name"]
        conf  = b.get("conf", 0.0)
        color = CLASS_COLORS_BGR.get(label, DEFAULT_COLOR)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        text  = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(y1 - 4, th + 2)
        cv2.rectangle(vis, (x1, ty - th - 2), (x1 + tw + 2, ty + 2), color, -1)
        cv2.putText(vis, text, (x1 + 1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)

    return PILImage.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

# ── Pipeline runner ────────────────────────────────────────────────────────────

def run_demo_pipeline(pdf_path: str) -> dict:
    # Stage 0: preprocess
    t0 = time.perf_counter()
    pages = pdf_to_pil(pdf_path)
    n_pages = len(pages)
    page_pil = pages[0]

    img_np = np.array(page_pil.convert("RGB"))
    canvas_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    canvas_lb, pad_top, pad_left = letterbox(canvas_bgr)
    t_pre = (time.perf_counter() - t0) * 1000

    # Stage 1: YOLO
    t1 = time.perf_counter()
    raw_blocks = detect_blocks(canvas_lb)
    t_yolo = (time.perf_counter() - t1) * 1000

    # Stage 2: OCR
    t2 = time.perf_counter()
    text_bboxes = [b["bbox_coco"] for b in raw_blocks if b["label_name"] in TEXT_CLASSES]
    if text_bboxes:
        texts = ocr_crops_batch(canvas_lb, text_bboxes)
        ti = 0
        for b in raw_blocks:
            if b["label_name"] in TEXT_CLASSES:
                b["text"] = texts[ti]
                ti += 1
    t_ocr = (time.perf_counter() - t2) * 1000

    # Build LayoutBlock list for XY-Cut + LLM
    img_w = canvas_lb.shape[1]
    layout_blocks = []
    table_crops = []
    for i, b in enumerate(raw_blocks):
        x, y, w, h = b["bbox_coco"]
        lb = LayoutBlock(
            block_id=i,
            class_name=b["label_name"],
            conf=b.get("conf", 0.0),
            bbox=BBox(x1=x, y1=y, x2=x + w, y2=y + h),
            raw_text=b.get("text", ""),
        )
        layout_blocks.append(lb)
        if b["label_name"] in TABLE_CLASSES:
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            crop = canvas_lb[max(0, y1):y2, max(0, x1):x2]
            if crop.size > 0:
                table_crops.append(crop)

    # Stage 3: XY-Cut reading order
    t3 = time.perf_counter()
    layout_blocks = assign_reading_order(layout_blocks, img_w)
    t_xycut = (time.perf_counter() - t3) * 1000

    # Stage 4: LLM
    t4 = time.perf_counter()
    data, is_valid, usage, raw_text = assemble_and_summarize(layout_blocks, table_crops, img_width=img_w)
    t_llm = (time.perf_counter() - t4) * 1000

    t_total = t_pre + t_yolo + t_ocr + t_xycut + t_llm
    cost_usd = usage["input_tokens"] * INPUT_PRICE + usage["output_tokens"] * OUTPUT_PRICE

    structured_json = data.get("structured_json", data)
    summary  = data.get("summary", "")
    keywords = data.get("keywords", [])
    page_type = data.get("page_type", "")
    language  = data.get("language", "")

    annotated_pil = draw_annotations(canvas_lb, raw_blocks)

    return {
        "page_img_pil":    PILImage.fromarray(cv2.cvtColor(canvas_lb, cv2.COLOR_BGR2RGB)),
        "annotated_pil":   annotated_pil,
        "blocks":          raw_blocks,
        "structured_json": structured_json,
        "is_valid":        is_valid,
        "raw_text":        raw_text,
        "summary":         summary,
        "keywords":        keywords,
        "page_type":       page_type,
        "language":        language,
        "usage":           usage,
        "cost_usd":        cost_usd,
        "n_pages":         n_pages,
        "latency": {
            "preprocess_ms": round(t_pre, 1),
            "yolo_ms":       round(t_yolo, 1),
            "ocr_ms":        round(t_ocr, 1),
            "xycut_ms":      round(t_xycut, 1),
            "llm_ms":        round(t_llm, 1),
            "total_ms":      round(t_total, 1),
        },
    }

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Document Layout Analysis Demo",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Document Layout Analysis Pipeline")
st.caption(f"YOLO v11s · docTR · XY-Cut · {LLM_DISPLAY}")

# Load models once
load_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Select PDF file", type=["pdf"])

    run_btn = st.button("▶ Run Pipeline", type="primary", disabled=(uploaded is None))

    st.divider()
    st.caption(f"**YOLO:** `{os.path.basename(YOLO_WEIGHTS)}`")
    st.caption(f"**LLM:** `{LLM_DISPLAY}`")
    st.caption(f"**Pricing:** {PRICING_DISPLAY}")

# ── Run pipeline ──────────────────────────────────────────────────────────────

if run_btn and uploaded is not None:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("Running pipeline..."):
        try:
            result = run_demo_pipeline(tmp_path)
            st.session_state["result"] = result
            st.session_state["filename"] = uploaded.name
        except Exception as e:
            st.exception(e)
    try:
        os.unlink(tmp_path)
    except Exception:
        pass

# ── Display results ───────────────────────────────────────────────────────────

if "result" in st.session_state:
    r   = st.session_state["result"]
    lat = r["latency"]
    use = r["usage"]

    if r["n_pages"] > 1:
        st.info(f"PDF has {r['n_pages']} pages — showing page 1.")

    # ── Section 1: Layout detection ──────────────────────────────────────────
    st.subheader("Layout Detection")

    col_orig, col_ann = st.columns(2)
    with col_orig:
        st.caption("Original page (DPI=150, letterboxed)")
        st.image(r["page_img_pil"], use_container_width=True)
    with col_ann:
        st.caption(f"YOLO annotated — {len(r['blocks'])} detected regions")
        st.image(r["annotated_pil"], use_container_width=True)

    if not r["blocks"]:
        st.warning("No layout regions detected.")
    else:
        # Block summary table
        from collections import Counter
        cnt = Counter(b["label_name"] for b in r["blocks"])
        avg_conf = {}
        for b in r["blocks"]:
            avg_conf.setdefault(b["label_name"], []).append(b.get("conf", 0))
        rows = [
            {"Class": cls, "Block count": cnt[cls],
             "Avg conf": f"{sum(avg_conf[cls])/len(avg_conf[cls]):.2f}"}
            for cls in sorted(cnt)
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.divider()

    # ── Section 2: JSON output ────────────────────────────────────────────────
    st.subheader(f"JSON Output ({LLM_DISPLAY})")

    if r["is_valid"]:
        st.json(r["structured_json"], expanded=2)
    else:
        st.warning("JSON parse failed — showing raw response.")
        st.code(r["raw_text"], language="json")

    st.divider()

    # ── Section 3: Summary & Keywords ────────────────────────────────────────
    st.subheader("Summary & Keywords")

    meta_cols = st.columns(2)
    meta_cols[0].caption(f"**page_type:** {r['page_type'] or '—'}")
    meta_cols[1].caption(f"**language:** {r['language'] or '—'}")

    if r["summary"]:
        st.info(r["summary"])
    else:
        st.caption("_(no summary)_")

    if r["keywords"]:
        st.caption("**Keywords:** " + " · ".join(f"`{k}`" for k in r["keywords"]))

    st.divider()

    # ── Section 4: Latency & Cost ─────────────────────────────────────────────
    st.subheader("Performance & Cost")

    total_ms = lat["total_ms"]
    latency_rows = [
        {"Module": "Preprocess (pdf2image + letterbox)", "ms": lat["preprocess_ms"],
         "%": f"{lat['preprocess_ms']/total_ms*100:.1f}%"},
        {"Module": "YOLO inference",                     "ms": lat["yolo_ms"],
         "%": f"{lat['yolo_ms']/total_ms*100:.1f}%"},
        {"Module": "docTR OCR",                          "ms": lat["ocr_ms"],
         "%": f"{lat['ocr_ms']/total_ms*100:.1f}%"},
        {"Module": "XY-Cut reading order",               "ms": lat["xycut_ms"],
         "%": f"{lat['xycut_ms']/total_ms*100:.1f}%"},
        {"Module": LLM_DISPLAY,                           "ms": lat["llm_ms"],
         "%": f"{lat['llm_ms']/total_ms*100:.1f}%"},
        {"Module": "TOTAL",                              "ms": total_ms, "%": "100%"},
    ]
    st.dataframe(pd.DataFrame(latency_rows), hide_index=True, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total time",      f"{total_ms:.0f} ms")
    c2.metric("Input tokens",    use["input_tokens"])
    c3.metric("Output tokens",   use["output_tokens"])
    c4.metric("Cost",            f"${r['cost_usd']:.5f}")

else:
    st.info("Upload a PDF file and click **▶ Run Pipeline** to start.")
