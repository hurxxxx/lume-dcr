# Tasks (updated 2026-01-29)

## Next
- Validate a Korean-strong baseline OCR engine + layout/table stack (candidate: PaddleOCR + PP-Structure) with a small PoC.
- Define Markdown/JSON output schema (including reading order, bounding boxes, and element types).
- Decide LLM role (post-correction vs end-to-end extraction) and model constraints.
- Confirm GPU/CUDA target and dependency strategy for offline delivery.
- Sketch SDK surface (init, parse, output adapters, error model).

## Backlog
- Implement OCR adapter interface and at least one engine integration.
- Implement layout analysis and reading-order reconstruction.
- Implement table/figure/equation extraction pipeline.
- Build structured output formatter (Markdown/HTML/JSON).
- Add LLM post-processing layer and prompt templates.
- Create evaluation harness with benchmark datasets and metrics.
- Package and obfuscate for offline distribution; add license enforcement.
- Provide optional REST wrapper for non-Python consumers.

## Done
- Summarized the research PDF and captured initial constraints.
- Confirmed MVP scope: SDK only; outputs Markdown/JSON; GPU runtime; Korean OCR priority.
- Added lume-dcr to PROJECTS.md.
- Checked ROCm environment: ROCm 7.2 present, gfx1151 (Ryzen AI MAX+ 395) detected, python3 available but torch missing.
- Installed ROCm PyTorch + vLLM (rocm700 wheel) in /tmp and ran vLLM GPT-2 smoke test; generation succeeded, noted numpy<2 pin + opencv conflict warning.
- Ran DeepSeek-OCR-2 inference on ROCm (transformers 4.46.3, tokenizers 0.20.3) using a PDF-derived image; output saved at /tmp/dsocr2/output/result.mmd and result_with_boxes.jpg.
- Ran DeepSeek-OCR-2 on sampledocs/2026_enterprise_ai_strategy_report_20260116021914.pptx; outputs under outputs/deepseek-ocr2-batch/... with timing summary.json.
- Ran PaddleOCR PP-StructureV3 on sampledocs/2026_enterprise_ai_strategy_report_20260116021914.pptx (LibreOffice→PDF→render); outputs under outputs/paddleocr/2026_enterprise_ai_strategy_report_20260116021914 (document.json, document.md, per-page JSON).
- Compared OCR outputs on sampledocs/20251202190315023.pdf across DeepSeek-OCR-2, PaddleOCR, Docling, Marker, PyMuPDF4LLM, and Tesseract; results under /tmp/ocr-compare with summary.json and keywords.json.
- Expanded open-source OCR comparisons on sampledocs/20251202190315023.pdf (added EasyOCR, docTR, OCRmyPDF, Kraken) with Docling base64-stripped output and new evaluation reports under /tmp/ocr-compare-open-source.
- Ran full open-source OCR comparison on sampledocs/2026_enterprise_ai_strategy_report_20260116021914.pptx (DeepSeek-OCR-2, Docling, Marker, PyMuPDF4LLM, Tesseract, EasyOCR, docTR, OCRmyPDF, Kraken) and generated markdown outputs + evaluation reports under /tmp/ocr-compare-open-source-pptx.
- Installed Kraken in a separate venv and ran OCR using a downloaded model (catmus-print-fondue-large) via binarize/segment/ocr pipeline.
- PaddleOCR PP-StructureV3 failed on the PPTX PDF with Paddle new-executor error (ConvertPirAttribute2RuntimeAttribute); results not produced.
- Re-ran DeepSeek-OCR-2 on the PPTX with recommended settings (base_size=1024, image_size=768, crop_mode=True); metrics did not improve and line coverage/num recall worsened vs the prior report.
- Documented composite score table in outputs/ocr-compare-open-source-pptx/score_table.md and appended to report.md.
