#!/usr/bin/env python3
"""GPU OCR conversion using ONNX Runtime with ROCm ExecutionProvider.

Runs PP-OCRv5 (det → cls → rec) pipeline on AMD GPU via onnxruntime-rocm.
Produces document.json and document.md matching ppocr_cpu_convert.py output format.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
import onnxruntime as ort
import yaml


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
ONNX_DIR = PROJECT_DIR / "onnx_models"

DET_MODEL = ONNX_DIR / "det.onnx"
REC_MODEL = ONNX_DIR / "rec.onnx"
CLS_MODEL = ONNX_DIR / "cls.onnx"

REC_YML = Path.home() / ".paddlex/official_models/korean_PP-OCRv5_mobile_rec/inference.yml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_character_dict(yml_path: Path) -> list[str]:
    """Load character dictionary from inference.yml."""
    with open(yml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    chars = cfg["PostProcess"]["character_dict"]
    # PaddleOCR convention: index 0 = blank, last = <space>
    vocab = ["blank"] + chars + [" "]
    return vocab


def create_session(model_path: Path, use_gpu: bool = True) -> ort.InferenceSession:
    """Create ONNX Runtime session with ROCm EP if available."""
    providers = []
    if use_gpu:
        providers.append("ROCMExecutionProvider")
    providers.append("CPUExecutionProvider")
    sess = ort.InferenceSession(str(model_path), providers=providers)
    actual = sess.get_providers()
    return sess


def render_pdf_pages(pdf_path: Path, dpi: int, max_pages: int | None) -> list[np.ndarray]:
    """Render PDF pages to BGR numpy arrays."""
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pages = []
    total = len(doc)
    limit = total if max_pages is None else min(total, max_pages)
    for i in range(limit):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        # Convert RGB to BGR for OpenCV compatibility
        img = img[:, :, ::-1].copy()
        pages.append(img)
    return pages


# ---------------------------------------------------------------------------
# Detection pre/post processing
# ---------------------------------------------------------------------------
def det_preprocess(img: np.ndarray, target_size: int = 960) -> tuple[np.ndarray, float]:
    """Resize and normalize image for detection model."""
    h, w = img.shape[:2]
    ratio = target_size / max(h, w)
    new_h = int(h * ratio)
    new_w = int(w * ratio)
    # Pad to multiple of 32
    new_h = max(32, math.ceil(new_h / 32) * 32)
    new_w = max(32, math.ceil(new_w / 32) * 32)
    resized = cv2.resize(img, (new_w, new_h))
    # Normalize: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (resized.astype(np.float32) / 255.0 - mean) / std
    # HWC -> CHW, add batch
    tensor = normalized.transpose(2, 0, 1)[np.newaxis]
    return tensor, ratio


def det_postprocess(output: np.ndarray, orig_h: int, orig_w: int, ratio: float,
                    thresh: float = 0.3, box_thresh: float = 0.6,
                    min_size: int = 3) -> list[np.ndarray]:
    """Extract text bounding boxes from detection heatmap using DB postprocess."""
    pred = output[0, 0]  # (H, W) probability map
    bitmap = (pred > thresh).astype(np.uint8)
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        if len(contour) < 4:
            continue
        # Score inside contour
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape(-1, 2)
        if len(points) < 4:
            continue

        rect = cv2.minAreaRect(contour)
        w_r, h_r = rect[1]
        if min(w_r, h_r) < min_size:
            continue

        # Mean score
        mask = np.zeros(pred.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [points], 1)
        score = cv2.mean(pred, mask)[0]
        if score < box_thresh:
            continue

        box = cv2.boxPoints(rect)
        box = box / ratio  # Scale back to original
        box[:, 0] = np.clip(box[:, 0], 0, orig_w)
        box[:, 1] = np.clip(box[:, 1], 0, orig_h)
        boxes.append(order_points(box))

    # Sort boxes top-to-bottom, left-to-right
    if boxes:
        boxes.sort(key=lambda b: (b[0][1], b[0][0]))
    return boxes


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


# ---------------------------------------------------------------------------
# Classification pre/post processing
# ---------------------------------------------------------------------------
def cls_preprocess(img: np.ndarray) -> np.ndarray:
    """Preprocess crop for text direction classification."""
    h, w = img.shape[:2]
    target_h, target_w = 80, 160
    ratio = w / float(h)
    resized_w = min(int(target_h * ratio), target_w)
    resized = cv2.resize(img, (resized_w, target_h))
    padded = np.zeros((target_h, target_w, 3), dtype=np.float32)
    padded[:, :resized_w, :] = resized.astype(np.float32)
    normalized = (padded / 255.0 - 0.5) / 0.5
    return normalized.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


def cls_postprocess(output: np.ndarray) -> tuple[int, float]:
    """Return (label, confidence) for text direction."""
    probs = output[0]
    label = int(np.argmax(probs))
    conf = float(probs[label])
    return label, conf


# ---------------------------------------------------------------------------
# Recognition pre/post processing
# ---------------------------------------------------------------------------
def rec_preprocess(img: np.ndarray, target_h: int = 48, max_w: int = 320) -> np.ndarray:
    """Preprocess crop for text recognition."""
    h, w = img.shape[:2]
    ratio = target_h / h
    target_w = min(int(w * ratio), max_w)
    target_w = max(target_w, 1)
    resized = cv2.resize(img, (target_w, target_h))
    padded = np.zeros((target_h, max_w, 3), dtype=np.float32)
    padded[:, :target_w, :] = resized.astype(np.float32)
    normalized = (padded / 255.0 - 0.5) / 0.5
    return normalized.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


def rec_postprocess(output: np.ndarray, vocab: list[str]) -> tuple[str, float]:
    """CTC decode: greedy, collapse repeats, remove blank."""
    preds = output[0]  # (T, vocab_size)
    indices = np.argmax(preds, axis=1)
    # CTC greedy decode
    chars = []
    confs = []
    prev = -1
    for t, idx in enumerate(indices):
        if idx != 0 and idx != prev:  # 0 = blank
            if idx < len(vocab):
                chars.append(vocab[idx])
                confs.append(float(preds[t, idx]))
        prev = idx
    text = "".join(chars)
    avg_conf = float(np.mean(confs)) if confs else 0.0
    return text, avg_conf


# ---------------------------------------------------------------------------
# Crop from detection box
# ---------------------------------------------------------------------------
def crop_from_box(img: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Perspective-crop text region from image."""
    box = box.astype(np.float32)
    w1 = np.linalg.norm(box[0] - box[1])
    w2 = np.linalg.norm(box[3] - box[2])
    h1 = np.linalg.norm(box[0] - box[3])
    h2 = np.linalg.norm(box[1] - box[2])
    width = int(max(w1, w2))
    height = int(max(h1, h2))
    if width < 1 or height < 1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(box, dst)
    crop = cv2.warpPerspective(img, M, (width, height))
    return crop


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_ocr(img: np.ndarray, det_sess: ort.InferenceSession,
            cls_sess: ort.InferenceSession, rec_sess: ort.InferenceSession,
            vocab: list[str], det_target_size: int = 960) -> list[dict]:
    """Run full OCR pipeline on a single image. Returns list of text results."""
    h, w = img.shape[:2]

    # 1) Detection
    det_input, ratio = det_preprocess(img, det_target_size)
    det_out = det_sess.run(None, {"x": det_input})[0]
    boxes = det_postprocess(det_out, h, w, ratio)

    results = []
    for box in boxes:
        crop = crop_from_box(img, box)
        if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            continue

        # 2) Classification (text direction)
        cls_input = cls_preprocess(crop)
        cls_out = cls_sess.run(None, {"x": cls_input})[0]
        label, cls_conf = cls_postprocess(cls_out)
        if label == 1 and cls_conf > 0.9:
            crop = cv2.rotate(crop, cv2.ROTATE_180)

        # 3) Recognition
        rec_input = rec_preprocess(crop)
        rec_out = rec_sess.run(None, {"x": rec_input})[0]
        text, rec_conf = rec_postprocess(rec_out, vocab)

        if text.strip():
            results.append({
                "text": text,
                "confidence": round(rec_conf, 4),
                "box": box.tolist(),
            })

    return results


def build_markdown(pages_results: list[list[dict]]) -> str:
    """Build markdown document from OCR results."""
    lines = []
    for page_idx, results in enumerate(pages_results, start=1):
        lines.append(f"\n\n# Page {page_idx}\n")
        for r in results:
            lines.append(r["text"])
    return "\n\n".join(lines).strip() + "\n"


def main():
    parser = argparse.ArgumentParser(description="GPU OCR with ONNX Runtime ROCm EP")
    parser.add_argument("pdf", type=str, help="Input PDF path")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    parser.add_argument("--dpi", type=int, default=200, help="Render DPI")
    parser.add_argument("--max-pages", type=int, default=None, help="Max pages to process")
    parser.add_argument("--det-size", type=int, default=960, help="Detection target size")
    parser.add_argument("--cpu", action="store_true", help="Force CPU-only mode")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    out_dir = Path(args.out) if args.out else pdf_path.parent / "outputs" / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    use_gpu = not args.cpu
    print(f"Loading ONNX models (GPU={use_gpu})...")
    t0 = time.time()

    det_sess = create_session(DET_MODEL, use_gpu)
    cls_sess = create_session(CLS_MODEL, use_gpu)
    rec_sess = create_session(REC_MODEL, use_gpu)
    vocab = load_character_dict(REC_YML)

    active_provider = det_sess.get_providers()[0]
    print(f"Active provider: {active_provider}")
    print(f"Models loaded in {time.time() - t0:.1f}s")

    # Render PDF
    print(f"Rendering PDF at {args.dpi} DPI...")
    pages = render_pdf_pages(pdf_path, dpi=args.dpi, max_pages=args.max_pages)
    print(f"  {len(pages)} page(s) rendered")

    # Run OCR per page
    all_results = []
    for idx, img in enumerate(pages):
        t1 = time.time()
        results = run_ocr(img, det_sess, cls_sess, rec_sess, vocab, args.det_size)
        elapsed = time.time() - t1
        print(f"  Page {idx + 1}: {len(results)} text regions ({elapsed:.1f}s)")
        all_results.append(results)

        # Save per-page JSON
        with open(out_dir / f"page_{idx + 1:03d}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # Save full document JSON
    full = {
        "source": str(pdf_path),
        "dpi": args.dpi,
        "provider": active_provider,
        "pages": all_results,
    }
    with open(out_dir / "document.json", "w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, indent=2)

    # Save markdown
    md = build_markdown(all_results)
    with open(out_dir / "document.md", "w", encoding="utf-8") as f:
        f.write(md)

    total_time = time.time() - t0
    total_texts = sum(len(r) for r in all_results)
    print(f"\nDone: {total_texts} text regions from {len(pages)} pages in {total_time:.1f}s")
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
