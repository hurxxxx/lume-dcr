# Lume DCR

High-performance document OCR parser focused on structured output (Markdown/HTML/JSON) with layout preservation and optional LLM-assisted post-processing.

## Goals (from the research brief)
- Preserve document structure: headings, paragraphs, lists, tables, figures, captions, and reading order.
- Support complex content: tables, equations, and mixed-language documents.
- Provide structured outputs suitable for downstream parsing and automation.
- Offer enterprise-friendly distribution and licensing (offline/air-gapped friendly).

## Decisions (confirmed)
- SDK only (no REST API).
- Output formats: Markdown and JSON.
- Target runtime: GPU servers.
- Korean OCR accuracy is a primary requirement.

## Constraints
- Python-first implementation.
- Distribution avoids public PyPI and Docker (offline delivery expected).
- Licensing should support online and offline validation paths.

## Quick Start (CPU-only PoC)
This repo currently provides a CPU-only pipeline for macOS to validate quality and output structure.

### Setup
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install paddlepaddle paddleocr pymupdf \"paddlex[ocr]==3.3.13\"
```

### Run (single PDF)
```
PADDLE_PDX_CACHE_HOME=.paddlex_cache \\
DISABLE_MODEL_SOURCE_CHECK=True \\
python scripts/ppocr_cpu_convert.py sampledocs/20251202190315023.pdf \\
  --out outputs/20251202190315023
```

Outputs:
- `outputs/<name>/document.json` (structured result with large arrays pruned)
- `outputs/<name>/document.md` (Markdown with tables embedded as HTML)

### Notes
- macOS (Apple Silicon) currently runs PaddleOCR in CPU mode only.
- Model files are cached under `.paddlex_cache/` (ignored by git).
- `sampledocs/` and `outputs/` are intentionally excluded from version control.

## Status
See `TASKS.md` for current work items and priorities.
