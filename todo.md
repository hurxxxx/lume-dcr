# TODO

## Phase 1: Hybrid PDF Pipeline
- Define quality criteria for text-layer vs OCR decision
- Implement page-level selection and merge into a unified schema
- Validate selection ratio and error patterns on sample PDFs

## Phase 2: Office Support + Image Extraction
- Define routing policy by document type
- Establish parser-first policy with OCR fallback for image-heavy pages
- Define image extraction policy per document type
- Validate text/image completeness on office samples

## Phase 3: OCR Post-Processing (LLM)
- Define reference-text-based correction workflow
- Lock structure/order/bbox and allow text-only fixes
- Validate CER/WER improvement and no-structure-regression

## Phase 4: RAG Optimization
- Define table dual-output policy (structured + searchable)
- Define metadata schema (section path, anchors, block IDs)
- Define semantic chunking strategy per document type

## Phase 5: Evaluation Framework
- Define test coverage per document type
- Define quality metrics for text, tables, and images
- Decide baseline comparison policy
