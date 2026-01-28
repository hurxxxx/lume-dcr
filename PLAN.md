# Lume DCR - Plan (2026-01-28)

## Goal
Improve OCR/document parsing quality without VLLM by prioritizing text-layer extraction, high-quality OCR for scanned pages, and robust structured output.

## Scope (this iteration)
- Add a hybrid pipeline: text-layer extraction when available, OCR fallback otherwise.
- Tune PP-StructureV3 for quality (higher DPI, orientation/deskew, table/formula on).
- Normalize outputs (Markdown + JSON) and keep files compact.
- Build a minimal comparison harness for 20251202190315023.pdf.

## Proposed Steps
1) Detect text-layer presence per page (PyMuPDF):
   - If text length > threshold, use PDF text extraction.
   - Else render page image and run PP-StructureV3 OCR.
2) Add OCR quality settings:
   - DPI 300/350 option.
   - Enable doc orientation, textline orientation, table and formula recognition.
3) Unify output:
   - JSON schema: page_index, source_type (pdf_text/ocr), blocks (type, bbox, text/html).
   - Markdown: ordered blocks, tables embedded as HTML.
4) Evaluation on 20251202190315023.pdf:
   - Compare PDF-text baseline vs OCR vs hybrid (SequenceMatcher/CER).
   - Record sample diffs and highlight failure modes.
5) Documentation:
   - Update README with hybrid mode usage.

## Files to Change
- scripts/ppocr_cpu_convert.py (quality flags + output normalization)
- scripts/pdf_text_convert.py (reuse for hybrid path)
- scripts/compare_pipelines.py (add CER/WER or improved metrics)
- (new) scripts/hybrid_convert.py
- README.md (usage)

## Validation
- Run page-1 test on 20251202190315023.pdf.
- If improved, run full document and compare metrics.
