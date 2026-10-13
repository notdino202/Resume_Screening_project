"""Unit and integration tests for document ingestion, cleaning, and extraction."""

from pathlib import Path
import pytest

from src.ingestion.extractor import DocumentExtractor, ExtractionError
from src.ingestion.cleaner import TextCleaner
from src.ingestion.hasher import compute_bytes_hash, compute_file_hash

SAMPLE_DIR = Path("data/sample_resumes")


@pytest.fixture
def extractor():
    return DocumentExtractor()


@pytest.fixture
def cleaner():
    return TextCleaner()


def test_hasher_deterministic(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("Deterministic content for hashing test.", encoding="utf-8")
    h1 = compute_file_hash(test_file)
    h2 = compute_file_hash(test_file)
    assert h1 == h2
    assert len(h1) == 64
    assert compute_bytes_hash(b"test") == compute_bytes_hash(b"test")


def test_cleaner_normalizes_text(cleaner):
    raw = "Senior ML Engineer \u2022 Python \u201cPyTorch\u201d \u2014 NLP\r\n\r\n\r\n\r\nExperience:\n - Built LLM"
    result = cleaner.clean(raw)
    assert "\r" not in result.cleaned_text
    assert '"PyTorch"' in result.cleaned_text
    assert "-" in result.cleaned_text
    assert "\n\n\n" not in result.cleaned_text
    assert result.char_count > 0
    assert result.alphanumeric_ratio > 0.5


def test_cleaner_flags_low_quality(cleaner):
    # Very short / low density text
    raw_junk = "??? !!! ### $$"
    res = cleaner.clean(raw_junk)
    assert res.is_low_quality is True
    assert any("LOW_ALPHANUMERIC_DENSITY" in f or "low-quality" in f for f in res.quality_flags)


def test_extract_all_10_sample_resumes(extractor, capsys):
    sample_files = [
        "resume_01_ml_senior.pdf",
        "resume_02_data_scientist.docx",
        "resume_03_fullstack_dev.txt",
        "resume_04_nlp_researcher.pdf",
        "resume_05_devops_cloud.docx",
        "resume_06_backend_python.pdf",
        "resume_07_junior_qa.docx",
        "resume_08_data_analyst.txt",
        "resume_09_cloud_architect.pdf",
        "resume_10_frontend_angular.docx",
    ]

    for fname in sample_files:
        fpath = SAMPLE_DIR / fname
        assert fpath.exists(), f"Sample file not found: {fpath}"
        result = extractor.extract(fpath)

        assert result.is_success is True
        assert len(result.clean_text) > 100, f"Expected text for {fname}"
        assert result.file_hash is not None
        assert len(result.file_hash) == 64
        assert result.cleaned_result.quality_score >= 0.60
        assert "low-quality extraction" not in result.advisory_flags

        # Print preview to verify clean text output for all 10 resumes
        print(f"\n--- Extracted: {fname} [{result.file_type} via {result.method_used}] ---")
        print(f"Length: {len(result.clean_text)} chars | Quality Score: {result.cleaned_result.quality_score}")
        print(f"Preview: {result.clean_text[:120]}...\n")


def test_empty_files_fail_with_clear_message(extractor):
    # Empty TXT
    empty_txt = SAMPLE_DIR / "edge_empty_resume.txt"
    with pytest.raises(ExtractionError) as exc_info_txt:
        extractor.extract(empty_txt)
    assert "Empty file" in str(exc_info_txt.value) or "empty" in str(exc_info_txt.value).lower()

    # Empty PDF
    empty_pdf = SAMPLE_DIR / "edge_empty_resume.pdf"
    with pytest.raises(ExtractionError) as exc_info_pdf:
        extractor.extract(empty_pdf)
    assert "Empty file" in str(exc_info_pdf.value) or "empty" in str(exc_info_pdf.value).lower()


def test_corrupt_files_fail_with_clear_message(extractor):
    # Corrupt PDF
    corrupt_pdf = SAMPLE_DIR / "edge_corrupt_resume.pdf"
    with pytest.raises(ExtractionError) as exc_pdf:
        extractor.extract(corrupt_pdf)
    assert "Corrupted or invalid PDF" in str(exc_pdf.value)

    # Corrupt DOCX
    corrupt_docx = SAMPLE_DIR / "edge_corrupt_resume.docx"
    with pytest.raises(ExtractionError) as exc_docx:
        extractor.extract(corrupt_docx)
    assert "Corrupted or invalid DOCX" in str(exc_docx.value)


def test_scanned_or_near_empty_pdf_flags_low_quality(extractor):
    scanned_pdf = SAMPLE_DIR / "edge_scanned_resume.pdf"
    res = extractor.extract(scanned_pdf)
    assert res.ocr_attempted is True
    assert "low-quality extraction" in res.advisory_flags
