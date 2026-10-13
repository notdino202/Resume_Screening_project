# Engineering Notes & Edge Case Log

## System Constraints & Principles
- **Decision Support:** System is purely advisory. No auto-rejections.
- **Privacy & Compliance:** Name, gender, age, photo, and address are explicitly excluded from feature extraction and scoring.
- **Determinism:** Fixed random seed and consistent deterministic transformations.
- **Local Execution:** 100% on-premises CPU processing.

## Known Failures and Edge Cases Logged
- **OCR Engine:** Tesseract executable is not present on standard Windows path. System falls back gracefully to `pdfplumber` and `pymupdf`, flagging scanned-only PDFs with an advisory flag `OCR_ENGINE_UNAVAILABLE` or `LOW_QUALITY_EXTRACTION`.
- **Date Parsing:** Ambiguous date formats (e.g. 05/06) will be normalized using explicit date patterns before dateparser. "Present", "Current", and open intervals capped at current timestamp.
- **Education Regex Collision:** "High School Diploma" matched level 2 "Diploma" due to standard degree ordering. Resolved using fixed-width negative lookbehind `(?<!high school )diploma`.
- **Experience Overlap Merging:** Sorted start-date intervals are merged into contiguous non-overlapping spans so concurrent/overlapping jobs (e.g. moonlight/consulting) calculate true elapsed career years rather than duplicate time.
- **JD Experience Qualifiers:** JDs often chain qualifiers (e.g., "4+ years of relevant industry experience"). Regex was updated with `(?:\s+(?:relevant|industry|professional|work|hands-on|direct|practical))*` to handle multi-word modifiers.
- **JD Education Scoping:** Preferred/Nice-to-have qualifications frequently mention higher degrees (e.g. Ph.D. preferred, Master's required). Required degree extraction scopes strictly to the must-have / minimum qualifications section first to prevent requirement inflation.

