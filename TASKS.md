# Tasks (updated 2026-01-27)

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
