"""Unit tests for Phase 3: Education leveling, date parsing, interval merging, and gap scoring."""

from datetime import datetime
import pytest

from src.extraction.education_parser import EducationParser
from src.extraction.experience_parser import ExperienceParser, DateInterval


@pytest.fixture
def edu_parser():
    return EducationParser()


@pytest.fixture
def exp_parser():
    # Fixed reference date for deterministic testing of "Present"
    ref_date = datetime(2025, 1, 1)
    return ExperienceParser(reference_date=ref_date)


def test_education_degree_levels(edu_parser):
    # Test doctorate (level 5)
    doc_res = edu_parser.parse_education("Ph.D. in Computer Science, MIT (2020)")
    assert doc_res.highest_level == 5
    assert doc_res.highest_degree_name == "Doctorate"

    # Test master's (level 4)
    ms_res = edu_parser.parse_education("Master of Science in Data Science | Stanford University")
    assert ms_res.highest_level == 4
    assert ms_res.highest_degree_name == "Master's"

    # Test bachelor's (level 3)
    bs_res = edu_parser.parse_education("Bachelor of Technology in Information Technology, 2019")
    assert bs_res.highest_level == 3
    assert bs_res.highest_degree_name == "Bachelor's"

    # Test associate / diploma (level 2)
    assoc_res = edu_parser.parse_education("Associate Degree in Computer Systems (2021)")
    assert assoc_res.highest_level == 2
    assert assoc_res.highest_degree_name == "Diploma/Associate"

    # Test high school (level 1)
    hs_res = edu_parser.parse_education("High School Diploma, Central High (2017)")
    assert hs_res.highest_level == 1
    assert hs_res.highest_degree_name == "High School"


def test_education_gap_scoring(edu_parser):
    # SPEC: gap >= 0 scores 1.0; gap == -1 scores 0.5; gap < -1 scores 0.0

    # Gap >= 0: Master's (4) for Bachelor's (3) requirement -> 1.0
    s1, gap1, flags1 = edu_parser.score_education(candidate_level=4, required_level=3)
    assert s1 == 1.0
    assert gap1 == 1
    assert len(flags1) == 0

    # Gap = 0: Bachelor's (3) for Bachelor's (3) requirement -> 1.0
    s2, gap2, flags2 = edu_parser.score_education(candidate_level=3, required_level=3)
    assert s2 == 1.0
    assert gap2 == 0
    assert len(flags2) == 0

    # Gap = -1: Diploma (2) for Bachelor's (3) requirement -> 0.5
    s3, gap3, flags3 = edu_parser.score_education(candidate_level=2, required_level=3)
    assert s3 == 0.5
    assert gap3 == -1
    assert "education below requirement" in flags3

    # Gap < -1: High School (1) for Master's (4) requirement -> 0.0
    s4, gap4, flags4 = edu_parser.score_education(candidate_level=1, required_level=4)
    assert s4 == 0.0
    assert gap4 == -3
    assert "education below requirement" in flags4


def test_experience_interval_overlaps(exp_parser):
    # SPEC: Overlapping jobs must be merged before summing to prevent double-counting
    # Job 1: Jan 2020 - Dec 2021 (2 years)
    # Job 2: Jun 2021 - Jun 2022 (1 year, overlaps with Job 1 between Jun 2021 and Dec 2021)
    # Merged interval: Jan 2020 to Jun 2022 = 2.5 years (approx 29 months)
    intervals = [
        DateInterval(start_date=datetime(2020, 1, 1), end_date=datetime(2021, 12, 31), raw_text="Job 1"),
        DateInterval(start_date=datetime(2021, 6, 1), end_date=datetime(2022, 6, 30), raw_text="Job 2"),
    ]
    merged = exp_parser.merge_overlapping_intervals(intervals)
    assert len(merged) == 1
    assert merged[0].start_date == datetime(2020, 1, 1)
    assert merged[0].end_date == datetime(2022, 6, 30)

    # 2.5 years instead of 2.0 + 1.0 = 3.0 years
    total_years = sum(i.duration_years for i in merged)
    assert 2.45 <= total_years <= 2.55


def test_experience_interval_subsumed(exp_parser):
    # Job B is fully within Job A
    intervals = [
        DateInterval(start_date=datetime(2018, 1, 1), end_date=datetime(2023, 1, 1), raw_text="Parent Job"),
        DateInterval(start_date=datetime(2019, 6, 1), end_date=datetime(2021, 6, 1), raw_text="Sub Job"),
    ]
    merged = exp_parser.merge_overlapping_intervals(intervals)
    assert len(merged) == 1
    assert merged[0].start_date == datetime(2018, 1, 1)
    assert merged[0].end_date == datetime(2023, 1, 1)
    total_years = sum(i.duration_years for i in merged)
    assert 4.9 <= total_years <= 5.1


def test_experience_interval_gaps(exp_parser):
    # Job 1: Jan 2018 - Dec 2019 (2 years)
    # GAP: 2020 (no employment)
    # Job 2: Jan 2021 - Dec 2022 (2 years)
    intervals = [
        DateInterval(start_date=datetime(2018, 1, 1), end_date=datetime(2019, 12, 31), raw_text="Job 1"),
        DateInterval(start_date=datetime(2021, 1, 1), end_date=datetime(2022, 12, 31), raw_text="Job 2"),
    ]
    merged = exp_parser.merge_overlapping_intervals(intervals)
    assert len(merged) == 2
    total_years = sum(i.duration_years for i in merged)
    # Total should be approx 4 years (gap year excluded)
    assert 3.9 <= total_years <= 4.1


def test_experience_date_parsing_with_present(exp_parser):
    # Reference date is 2025-01-01
    # Jan 2021 to Present = 4 years
    text = """
    Software Engineer | TechCorp | Jan 2021 - Present
    Junior Developer | Startup | Jun 2018 - Dec 2020
    """
    res = exp_parser.parse_experience(text)
    assert len(res.raw_intervals) == 2
    # Jan 2021 to Jan 2025 = 4.0 years; Jun 2018 to Dec 2020 = 2.5 years; Total = 6.5 years
    assert 6.4 <= res.total_years <= 6.6


def test_experience_scoring_rules(exp_parser):
    # SPEC: min(candidate_years / required_years, 1.0)
    # effective_years = 0.7 * relevant + 0.3 * total
    # If no requirement, score 1.0

    # No requirement stated
    score_no_req, flags_no_req = exp_parser.score_experience(candidate_years=3.0, required_years=None)
    assert score_no_req == 1.0
    assert len(flags_no_req) == 0

    # Required 5 years, candidate has 5 years relevant and 5 years total
    score_met, flags_met = exp_parser.score_experience(
        candidate_years=5.0, required_years=5.0, relevant_years=5.0
    )
    assert score_met == 1.0
    assert len(flags_met) == 0

    # Required 5 years, candidate has 6 years -> capped at 1.0
    score_over, _ = exp_parser.score_experience(
        candidate_years=6.0, required_years=5.0, relevant_years=6.0
    )
    assert score_over == 1.0

    # Required 5 years, candidate has 4 total, 2 relevant:
    # effective = 0.7 * 2 + 0.3 * 4 = 1.4 + 1.2 = 2.6 years
    # score = 2.6 / 5.0 = 0.52
    score_partial, _ = exp_parser.score_experience(
        candidate_years=4.0, required_years=5.0, relevant_years=2.0
    )
    assert score_partial == 0.52

    # Low experience flag: effective < 0.5 * required
    # Required 5 years, effective = 1.0 -> flag "low experience"
    score_low, flags_low = exp_parser.score_experience(
        candidate_years=1.0, required_years=5.0, relevant_years=1.0
    )
    assert score_low == 0.20
    assert "low experience" in flags_low
