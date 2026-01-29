# OCR Test Master Report (lume-dcr)

이 문서는 lume-dcr에서 수행한 OCR 비교/추론 실험 결과를 한 곳에 모아 관리합니다.

## Datasets
- **20251202190315023.pdf** (수입신고서)
  - Source: `lume-dcr/sampledocs/20251202190315023.pdf`
- **2026_enterprise_ai_strategy_report_20260116021914.pptx** (기업 AI 전략 보고서)
  - Source: `lume-dcr/sampledocs/2026_enterprise_ai_strategy_report_20260116021914.pptx`
- **[현대해상] 상해 보험 약관, 갤럭시케어팩.docx** (보험 약관 문서)
  - Source: `lume-dcr/sampledocs/[현대해상] 상해 보험 약관, 갤럭시케어팩.docx`

---

## A. 오픈소스 OCR 비교 (로컬)
### 1) PDF (20251202190315023.pdf)
- **리포트**: `lume-dcr/outputs/ocr-compare-open-source/report.md`
- **요약/평가지표**: `lume-dcr/outputs/ocr-compare-open-source/summary.json`, `lume-dcr/outputs/ocr-compare-open-source/summary_eval.json`
- **주요 후보**: deepseek-ocr2, paddleocr, docling(clean), marker, tesseract, easyocr, doctr, ocrmypdf, kraken, pymupdf4llm

### 2) PPTX (2026_enterprise_ai_strategy_report_20260116021914.pptx)
- **리포트**: `lume-dcr/outputs/ocr-compare-open-source-pptx/report.md`
- **점수표**: `lume-dcr/outputs/ocr-compare-open-source-pptx/score_table.md`
- **요약/평가지표**: `lume-dcr/outputs/ocr-compare-open-source-pptx/summary_eval.json`
- **주요 후보**: deepseek-ocr2, paddleocr, docling, marker, tesseract, easyocr, doctr, ocrmypdf, kraken, pymupdf4llm

---

## B. DeepSeek-OCR-2 (로컬)
- **PPTX 배치 요약**: `lume-dcr/outputs/deepseek-ocr2-batch/2026_enterprise_ai_strategy_report_20260116021914/summary.json`
- **PPTX 비교용 요약**: `lume-dcr/outputs/ocr-compare-open-source-pptx/deepseek-ocr2/2026_enterprise_ai_strategy_report_20260116021914/summary.json`
- **품질 기준 파이프라인 결과 (참조용)**:
  - `lume-dcr/outputs/ocr-compare-open-source/deepseek-ocr2/document.md`

---

## C. OpenRouter Qwen OCR 비교 (API)
- **리포트**: `lume-dcr/outputs/openrouter-qwen-compare/report.md`
- **요약/평가지표**: `lume-dcr/outputs/openrouter-qwen-compare/summary.json`
- **DeepSeek 로컬 요약**: `lume-dcr/outputs/openrouter-qwen-compare/summary_deepseek_local.json`

---

## D. Qwen3-VL-32B-GGUF (로컬, ROCm)
### 1) PDF (20251202190315023.pdf)
- **출력**: `lume-dcr/outputs/openrouter-qwen-compare/qwen3-vl-32b-gguf-q4_k_m/20251202190315023/document.md`
- **최신 파라미터 기록**: `lume-dcr/outputs/openrouter-qwen-compare/qwen3-vl-32b-gguf-q4_k_m/20251202190315023/params.md`
- **시도 로그**: `.../20251202190315023/_attempts/`

### 2) PPTX (2026_enterprise_ai_strategy_report_20260116021914.pptx)
- **출력**: `lume-dcr/outputs/openrouter-qwen-compare/qwen3-vl-32b-gguf-q4_k_m/2026_enterprise_ai_strategy_report_20260116021914/document.md`
- **최신 파라미터 기록**: `lume-dcr/outputs/openrouter-qwen-compare/qwen3-vl-32b-gguf-q4_k_m/2026_enterprise_ai_strategy_report_20260116021914/params.md`
- **시도 로그**: `.../2026_enterprise_ai_strategy_report_20260116021914/_attempts/attempt_01_c6144_n2048_imgmin1024_imgmax2048/`

---

## E. DeepSeek vs Qwen 비교 (기존 작업)
- **리포트**: `lume-dcr/outputs/deepseek-vs-qwen3vl/report.md`
- **요약**: `lume-dcr/outputs/deepseek-vs-qwen3vl/summary.json`

---

## F. DOCX 빠른 OCR (Tesseract)
- **입력**: `lume-dcr/sampledocs/[현대해상] 상해 보험 약관, 갤럭시케어팩.docx`
- **파이프라인**: DOCX → PDF(soffice) → 이미지(pdftoppm, 200 DPI) → Tesseract(kor+eng, oem=1, psm=6)
- **출력**: `lume-dcr/outputs/docx-fast-ocr-tesseract/[현대해상] 상해 보험 약관, 갤럭시케어팩/document.md`
- **페이지 결과**: `.../page_###.txt`
- **평가**: 품질이 매우 낮아 실사용 불가. (비추천)

---

## G. DOCX DeepSeek-OCR-2 (10p 샘플)
- **입력**: `lume-dcr/sampledocs/[현대해상] 상해 보험 약관, 갤럭시케어팩.docx`
- **설정**: DPI 200, `--deepseek-crop-mode`, 최대 10페이지
- **출력**: `lume-dcr/outputs/deepseek-ocr2-docx/deepseek-ocr2/[현대해상] 상해 보험 약관, 갤럭시케어팩/document.md`
- **요약/리포트**: `lume-dcr/outputs/deepseek-ocr2-docx/summary.json`, `lume-dcr/outputs/deepseek-ocr2-docx/report.md`
- **실측 시간( Strix Halo 기준 )**: 10p 합계 114.3초 (평균 11.43초/페이지) → 53p 기준 약 10.1분 추정

---

## DOCX 운영 기준
- **텍스트 위주 문서**: DeepSeek-OCR-2 사용 권장
- **이미지 내 텍스트가 중요**: Qwen3‑VL‑32B로 이미지 OCR 보완 필요
- **Qwen3 Docx OCR 속도( Strix Halo 기준 )**: 10p기준 약 450초 정도 소요

---

## 참고 메모
- PPTX는 `soffice`로 PDF 변환 후 `pdftoppm`으로 이미지 렌더링해 OCR 수행.
- Qwen3-VL GGUF는 이미지 토큰 상한(`--image-max-tokens`)과 컨텍스트(`-c`) 조합이 품질/안정성에 큰 영향.
- 상세 설정/로그는 각 `_attempts` 디렉토리에 저장.
