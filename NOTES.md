# Engineering Notes & Edge Case Log

## System Constraints & Principles
- **Decision Support:** System is purely advisory. No auto-rejections.
- **Privacy & Compliance:** Name, gender, age, photo, and address are explicitly excluded from feature extraction and scoring.
- **Determinism:** Fixed random seed and consistent deterministic transformations.
- **Local Execution:** 100% on-premises CPU processing.

## Known Failures and Edge Cases Logged
- **OCR Engine:** Tesseract executable is not present on standard Windows path. System falls back gracefully to `pdfplumber` and `pymupdf`, flagging scanned-only PDFs with an advisory flag `OCR_ENGINE_UNAVAILABLE` or `LOW_QUALITY_EXTRACTION`.
- **Date Parsing:** Ambiguous date formats (e.g. 05/06) will be normalized using explicit date patterns before dateparser. "Present", "Current", and open intervals capped at current timestamp.
