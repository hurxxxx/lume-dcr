#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional

import fitz  # PyMuPDF
import numpy as np


PRUNE_KEYS = {
    "input_img",
    "input_img_ori",
    "binary",
    "thresh",
    "heatmap",
    "mask",
    "bitmap",
}


def _list_shape(value, max_depth=3):
    shape = []
    cur = value
    for _ in range(max_depth):
        if isinstance(cur, list) and cur:
            shape.append(len(cur))
            cur = cur[0]
        else:
            break
    return shape


def _is_large_image_like(value):
    if not isinstance(value, list) or not value:
        return False
    shape = _list_shape(value, max_depth=3)
    if len(shape) >= 2 and shape[0] >= 200 and shape[1] >= 200:
        return True
    return False


def _summarize_array(value, kind):
    if hasattr(value, "shape"):
        shape = list(value.shape)
        dtype = getattr(value, "dtype", None)
        return {"__omitted__": kind, "shape": shape, "dtype": str(dtype) if dtype is not None else None}
    if isinstance(value, list):
        return {"__omitted__": kind, "shape": _list_shape(value, max_depth=4), "len": len(value)}
    return {"__omitted__": kind}


def to_builtin(obj: Any):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return _summarize_array(obj, "ndarray")
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in PRUNE_KEYS:
                out[k] = _summarize_array(v, f"omitted:{k}")
                continue
            if isinstance(v, np.ndarray):
                out[k] = _summarize_array(v, f"ndarray:{k}")
                continue
            if _is_large_image_like(v):
                out[k] = _summarize_array(v, f"image_like:{k}")
                continue
            out[k] = to_builtin(v)
        return out
    if isinstance(obj, list):
        if _is_large_image_like(obj) or len(obj) > 5000:
            return _summarize_array(obj, "list")
        return [to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_builtin(v) for v in obj]
    # Fallback for non-serializable objects (e.g., fitz.Font)
    return str(obj)


def render_pdf_pages(pdf_path: Path, dpi: int, max_pages: Optional[int]):
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
        # Paddle expects BGR
        img = img[:, :, ::-1]
        pages.append(img)
    return pages


def build_markdown(pages_results):
    lines = []
    for page_idx, items in enumerate(pages_results, start=1):
        lines.append(f"\n\n# Page {page_idx}\n")
        # Some pipelines wrap a page result dict inside a list
        if isinstance(items, list) and len(items) == 1 and isinstance(items[0], dict):
            if "parsing_res_list" in items[0]:
                items = items[0]["parsing_res_list"]
        for item in items:
            if not isinstance(item, dict):
                if isinstance(item, str):
                    label = None
                    content = None
                    for raw in item.splitlines():
                        line = raw.strip()
                        if line.startswith("label:"):
                            label = line.split(":", 1)[1].strip()
                        elif line.startswith("content:"):
                            content = line.split(":", 1)[1].strip()
                    if label == "table" and content:
                        lines.append(content)
                    elif label in ("title", "heading") and content:
                        lines.append(f"## {content}")
                    elif content:
                        lines.append(content)
                    else:
                        lines.append(item.strip() or "[unknown]")
                else:
                    lines.append(str(item))
                continue
            item_type = item.get("type") or item.get("label") or "unknown"
            res = item.get("res")
            text = None
            if isinstance(res, dict) and "text" in res:
                text = res.get("text")
            elif "text" in item:
                text = item.get("text")
            elif isinstance(res, list):
                texts = []
                for r in res:
                    if isinstance(r, dict) and "text" in r:
                        texts.append(r.get("text"))
                if texts:
                    text = " ".join(texts)

            if item_type in ("title", "heading") and text:
                lines.append(f"## {text}")
            elif item_type == "table":
                html = None
                if isinstance(res, dict):
                    html = res.get("html") or res.get("html_table")
                if html:
                    lines.append(html)
                else:
                    lines.append("[table]")
            elif text:
                lines.append(text)
            else:
                lines.append(f"[{item_type}]")
    return "\n\n".join(lines).strip() + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=str)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-pages", type=int, default=None)
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    out_dir = Path(args.out) if args.out else pdf_path.parent / "outputs" / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure PaddleX cache stays in workspace
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(Path(__file__).resolve().parent.parent / ".paddlex_cache"))

    # Lazy import so env is set first
    from paddleocr import PPStructureV3

    engine = PPStructureV3(lang="korean", ocr_version="PP-OCRv5")

    pages = render_pdf_pages(pdf_path, dpi=args.dpi, max_pages=args.max_pages)
    results = []
    for idx, img in enumerate(pages):
        pred = engine.predict(img)
        # Some pipelines return a generator
        if not isinstance(pred, (list, dict)) and hasattr(pred, "__iter__"):
            pred = list(pred)
        results.append(to_builtin(pred))
        with open(out_dir / f"page_{idx+1:03d}.json", "w", encoding="utf-8") as f:
            json.dump(results[-1], f, ensure_ascii=False, indent=2)

    full = {
        "source": str(pdf_path),
        "dpi": args.dpi,
        "pages": results,
    }
    with open(out_dir / "document.json", "w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, indent=2)

    md = build_markdown(results)
    with open(out_dir / "document.md", "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
