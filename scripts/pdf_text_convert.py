#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF


def extract_page_blocks(page: fitz.Page) -> List[Dict[str, Any]]:
    data = page.get_text("dict")
    blocks = []
    for b in data.get("blocks", []):
        block_type = b.get("type", 0)
        bbox = b.get("bbox")
        if block_type == 0:
            # text block
            texts = []
            for line in b.get("lines", []):
                line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                if line_text:
                    texts.append(line_text)
            text = "\n".join(texts).strip()
            blocks.append({"type": "text", "bbox": bbox, "text": text})
        elif block_type == 1:
            blocks.append({"type": "image", "bbox": bbox})
        else:
            blocks.append({"type": f"type_{block_type}", "bbox": bbox})
    # sort by top then left to stabilize reading order
    blocks.sort(key=lambda x: (x.get("bbox", [0, 0, 0, 0])[1], x.get("bbox", [0, 0, 0, 0])[0]))
    return blocks


def build_markdown(pages: List[Dict[str, Any]]) -> str:
    lines = []
    for page in pages:
        lines.append(f"\n\n# Page {page['page_index']}\n")
        for block in page.get("blocks", []):
            if block.get("type") == "text":
                text = block.get("text", "").strip()
                if text:
                    lines.append(text)
            elif block.get("type") == "image":
                lines.append("[image]")
    return "\n\n".join(lines).strip() + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=str)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    out_dir = Path(args.out) if args.out else pdf_path.parent / "outputs" / f"{pdf_path.stem}_pdftext"
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    total = len(doc)
    limit = total if args.max_pages is None else min(total, args.max_pages)

    pages = []
    for i in range(limit):
        page = doc.load_page(i)
        text = page.get_text("text")
        blocks = extract_page_blocks(page)
        pages.append({
            "page_index": i + 1,
            "text": text,
            "blocks": blocks,
        })
        with open(out_dir / f"page_{i+1:03d}.json", "w", encoding="utf-8") as f:
            json.dump(pages[-1], f, ensure_ascii=False, indent=2)

    full = {
        "source": str(pdf_path),
        "pages": pages,
    }
    with open(out_dir / "document.json", "w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, indent=2)

    md = build_markdown(pages)
    with open(out_dir / "document.md", "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
