"""Interactive Streamlit Decision-Support Web Application for AI Resume Screening."""

from __future__ import annotations
import io
from pathlib import Path
import sqlite3
import pandas as pd
import streamlit as st

from src.ingestion.extractor import DocumentExtractor
from src.extraction.section_detector import SectionDetector
from src.extraction.skills_matcher import SkillsMatcher
from src.extraction.education_parser import EducationParser
from src.extraction.experience_parser import ExperienceParser
from src.jd.models import JobDescription
from src.jd.parser import JobDescriptionParser
from src.scoring.embeddings import EmbeddingManager
from src.scoring.scorer import HybridScorer, CandidateScoreResult
from src.scoring.explainer import ExplanationGenerator
from src.storage.db import ScreeningDatabase

# Page Configuration
st.set_page_config(
    page_title="TalentRank | AI Resume Screening & Decision Support",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich, polished aesthetics
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e2638 0%, #151b28 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .tier-badge-strong {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .tier-badge-moderate {
        background-color: #78350f;
        color: #fbbf24;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .tier-badge-weak {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .flag-tag {
        background-color: #374151;
        color: #f3f4f6;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    """Initializes and caches system pipelines and models."""
    embedder = EmbeddingManager()
    extractor = DocumentExtractor()
    detector = SectionDetector()
    matcher = SkillsMatcher()
    edu_parser = EducationParser()
    exp_parser = ExperienceParser()
    jd_parser = JobDescriptionParser(skills_matcher=matcher, education_parser=edu_parser)
    scorer = HybridScorer(
        embedder=embedder,
        edu_parser=edu_parser,
        exp_parser=exp_parser,
    )
    db = ScreeningDatabase()
    return {
        "extractor": extractor,
        "detector": detector,
        "matcher": matcher,
        "edu_parser": edu_parser,
        "exp_parser": exp_parser,
        "jd_parser": jd_parser,
        "scorer": scorer,
        "db": db,
    }

pipe = get_pipeline()

# Title and Disclaimer
st.title("🎯 TalentRank: AI Resume Screening Decision Support")
st.caption("🔒 100% Local Processing • PII-Blind Scoring • Decision Support Only (Recruiter Decides, Zero Auto-Rejection)")

# Sidebar: Job Description & Weight Configuration
st.sidebar.header("📋 1. Job Description & Scoring")

sample_jd_files = list(Path("data/sample_jds").glob("*.txt"))
sample_jd_options = ["Custom Input"] + [f.name for f in sample_jd_files]
selected_jd_option = st.sidebar.selectbox("Choose Sample JD or Paste Custom", sample_jd_options)

default_jd_text = ""
if selected_jd_option != "Custom Input":
    selected_path = Path("data/sample_jds") / selected_jd_option
    if selected_path.exists():
        default_jd_text = selected_path.read_text(encoding="utf-8")

raw_jd_text = st.sidebar.text_area("Job Description Text", value=default_jd_text, height=220)

if "parsed_jd" not in st.session_state or st.sidebar.button("🔄 Parse / Reload JD"):
    if raw_jd_text.strip():
        parsed = pipe["jd_parser"].parse(raw_jd_text)
        st.session_state["parsed_jd"] = parsed

if "parsed_jd" in st.session_state:
    p_jd: JobDescription = st.session_state["parsed_jd"]

    st.sidebar.subheader("⚙️ Editable JD Parameters")
    edit_title = st.sidebar.text_input("Role Title", value=p_jd.title)
    edit_years = st.sidebar.number_input(
        "Required Experience (Years)", min_value=0.0, max_value=20.0,
        value=float(p_jd.required_years or 0.0), step=0.5
    )
    edu_options = {1: "1 - High School", 2: "2 - Diploma/Associate", 3: "3 - Bachelor's", 4: "4 - Master's", 5: "5 - Doctorate"}
    edit_edu_level = st.sidebar.selectbox(
        "Required Education Level",
        options=list(edu_options.keys()),
        index=p_jd.required_education_level - 1,
        format_func=lambda x: edu_options[x]
    )

    must_skills_str = st.sidebar.text_area(
        "Must-Have Skills (comma-separated)",
        value=", ".join(p_jd.must_have_skills),
        help="Core non-negotiable requirements"
    )
    nice_skills_str = st.sidebar.text_area(
        "Nice-To-Have Skills (comma-separated)",
        value=", ".join(p_jd.nice_to_have_skills),
        help="Bonus skills"
    )

    # Scoring Weights Sliders
    st.sidebar.subheader("⚖️ Hybrid Scoring Weights")
    st.sidebar.caption("SPEC standard: Skill (40%), Semantic (30%), Experience (20%), Education (10%)")
    w_skill = st.sidebar.slider("Skill Weight", 0.0, 1.0, 0.40, 0.05)
    w_sem = st.sidebar.slider("Semantic Weight", 0.0, 1.0, 0.30, 0.05)
    w_exp = st.sidebar.slider("Experience Weight", 0.0, 1.0, 0.20, 0.05)
    w_edu = st.sidebar.slider("Education Weight", 0.0, 1.0, 0.10, 0.05)

    weight_sum = round(w_skill + w_sem + w_exp + w_edu, 2)
    if weight_sum != 1.0:
        st.sidebar.error(f"⚠️ Weights sum to {weight_sum:.2f}! Must equal 1.00.")
    else:
        st.sidebar.success("✅ Weights validated: 1.00")

    # Update session JD
    p_jd.title = edit_title
    p_jd.required_years = edit_years if edit_years > 0 else None
    p_jd.required_education_level = edit_edu_level
    p_jd.required_education_name = edu_options[edit_edu_level].split(" - ")[1]
    p_jd.must_have_skills = [s.strip() for s in must_skills_str.split(",") if s.strip()]
    p_jd.nice_to_have_skills = [s.strip() for s in nice_skills_str.split(",") if s.strip()]
    p_jd.weights_override = {
        "skill": w_skill,
        "semantic": w_sem,
        "experience": w_exp,
        "education": w_edu,
    }


# Main Content Area: Resumes & Screening
st.subheader("📂 2. Candidate Resumes")
col_upload, col_sample = st.columns([3, 1])

with col_upload:
    uploaded_files = st.file_uploader(
        "Upload Resumes (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

with col_sample:
    st.write("Or use preloaded samples:")
    use_sample_resumes = st.button("📁 Load 10 Sample Resumes")

if use_sample_resumes:
    st.session_state["use_samples"] = True
    st.success("Loaded 10 diverse sample candidate profiles.")

# Run Screening Button
run_screen = st.button("🚀 Screen & Rank Candidates", type="primary", use_container_width=True)

if run_screen:
    if "parsed_jd" not in st.session_state:
        st.error("Please configure or parse a Job Description first.")
    else:
        target_jd = st.session_state["parsed_jd"]
        if round(sum(target_jd.weights_override.values()), 2) != 1.0:
            st.error("Weights must sum to 1.0 before running.")
        else:
            candidates_to_process = []

            # Determine sources
            if uploaded_files:
                for uf in uploaded_files:
                    candidates_to_process.append((uf.name, uf.getvalue()))
            elif st.session_state.get("use_samples", False) or True:  # Default to sample resumes if none uploaded
                sample_files = list(Path("data/sample_resumes").glob("resume_*.*"))
                for sf in sample_files:
                    candidates_to_process.append((sf.name, sf.read_bytes()))

            progress_bar = st.progress(0)
            status_text = st.empty()
            screened_results: list[CandidateScoreResult] = []
            file_hashes: dict[str, str] = {}

            for idx, (fname, raw_bytes) in enumerate(candidates_to_process):
                status_text.text(f"Extracting & evaluating {fname} ({idx+1}/{len(candidates_to_process)})...")

                # Ingestion & cleaning
                ext_res = pipe["extractor"].extract(raw_bytes, file_name=fname)
                file_hashes[fname] = ext_res.file_hash

                # Sections & Skills
                sections = pipe["detector"].detect_sections(ext_res.clean_text)
                skills = pipe["matcher"].extract_skills_from_sections(sections)
                edu = pipe["edu_parser"].parse_education(sections.education_text or ext_res.clean_text)
                exp = pipe["exp_parser"].parse_experience(sections.experience_text or ext_res.clean_text)

                # Cache in SQLite
                pipe["db"].cache_parsed_resume(
                    file_hash=ext_res.file_hash,
                    file_name=fname,
                    file_type=ext_res.file_type,
                    clean_text=ext_res.clean_text,
                    sections_dict={s.value: sections.get_text(s) for s in sections.section_texts},
                    skills_dict={s: skills[s].evidence_factor for s in skills},
                    education_dict={"level": edu.highest_level, "degree": edu.highest_degree_name},
                    experience_dict={"total": exp.total_years, "relevant": exp.relevant_years},
                    quality_flags=ext_res.advisory_flags,
                )

                # Score
                score_res = pipe["scorer"].score_candidate(
                    candidate_id=fname.rsplit(".", 1)[0],
                    file_name=fname,
                    clean_text=ext_res.clean_text,
                    extracted_skills=skills,
                    candidate_edu=edu,
                    candidate_exp=exp,
                    jd=target_jd,
                    extraction_flags=ext_res.advisory_flags,
                )
                score_res.explanation = ExplanationGenerator.generate_explanation(score_res)
                screened_results.append(score_res)
                progress_bar.progress((idx + 1) / len(candidates_to_process))

            # Ranking with tie-breaking
            ranked = pipe["scorer"].rank_candidates(screened_results)
            for c in ranked:
                c.explanation = ExplanationGenerator.generate_explanation(c)

            # Save results to SQLite
            job_key = f"job_{abs(hash(target_jd.title))%100000}"
            pipe["db"].save_job_description(job_key, target_jd)
            pipe["db"].save_screening_results(job_key, ranked, file_hashes)

            st.session_state["screening_results"] = ranked
            progress_bar.empty()
            status_text.empty()
            st.success(f"✅ Successfully screened and ranked {len(ranked)} candidates!")


# Display Leaderboard and Details if available
if "screening_results" in st.session_state:
    results: list[CandidateScoreResult] = st.session_state["screening_results"]

    st.markdown("---")
    st.subheader("📊 3. Screening Leaderboard & Decision Dashboard")

    # Metrics Summary
    strong_cnt = sum(1 for r in results if r.tier == "Strong")
    mod_cnt = sum(1 for r in results if r.tier == "Moderate")
    weak_cnt = sum(1 for r in results if r.tier == "Weak")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Candidates", len(results))
    m2.metric("Strong Matches (>=75)", strong_cnt)
    m3.metric("Moderate Matches (50-74)", mod_cnt)
    m4.metric("Weak Matches (<50)", weak_cnt)

    # Tier Filter
    tier_filter = st.selectbox("Filter by Tier", ["All Tiers", "Strong", "Moderate", "Weak"])
    filtered_results = [r for r in results if tier_filter == "All Tiers" or r.tier == tier_filter]

    # Build Display Table
    table_rows = []
    for r in filtered_results:
        table_rows.append({
            "Rank": f"#{r.rank}",
            "Candidate File": r.file_name,
            "Final Score": r.final_score,
            "Tier": r.tier,
            "Skill Score": f"{r.skill_score * 100:.1f}%",
            "Semantic Score": f"{r.semantic_score * 100:.1f}%",
            "Exp Score": f"{r.experience_score * 100:.1f}%",
            "Edu Score": f"{r.education_score * 100:.1f}%",
            "Must-Have Cov": f"{r.must_have_coverage * 100:.1f}%",
            "Advisory Flags": ", ".join(r.advisory_flags) if r.advisory_flags else "None",
        })

    df_display = pd.DataFrame(table_rows)
    st.dataframe(df_display, use_container_width=True)

    # CSV Download Button
    csv_data = df_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Ranked Results to CSV",
        data=csv_data,
        file_name="screening_results.csv",
        mime="text/csv",
    )

    # 4. Candidate Detail Deep-Dive
    st.markdown("---")
    st.subheader("🔍 4. Candidate Detail Deep-Dive")

    cand_options = [f"#{r.rank} - {r.file_name} ({r.final_score:.1f} pts, {r.tier})" for r in results]
    selected_cand_idx = st.selectbox("Select Candidate to Inspect", range(len(results)), format_func=lambda i: cand_options[i])
    selected_cand: CandidateScoreResult = results[selected_cand_idx]

    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.markdown(f"### Assessment for **{selected_cand.file_name}**")
        tier_class = f"tier-badge-{selected_cand.tier.lower()}"
        st.markdown(f"**Tier:** <span class='{tier_class}'>{selected_cand.tier.upper()}</span> | **Final Score:** `{selected_cand.final_score:.1f} / 100`", unsafe_allow_html=True)
        st.write("")

        # Component breakdown chart
        st.markdown("#### Component Breakdown")
        comp_df = pd.DataFrame({
            "Component": ["Skill", "Semantic", "Experience", "Education"],
            "Score (%)": [
                selected_cand.skill_score * 100,
                selected_cand.semantic_score * 100,
                selected_cand.experience_score * 100,
                selected_cand.education_score * 100,
            ]
        })
        st.bar_chart(comp_df.set_index("Component"))

        # Flags
        if selected_cand.advisory_flags:
            st.warning("⚠️ **Advisory Flags:** " + ", ".join(selected_cand.advisory_flags))
        else:
            st.success("✅ **No Advisory Flags:** All baseline requirements satisfied.")

    with c_right:
        st.markdown("#### Skills Alignment")
        matched_m = [d for d in selected_cand.must_have_details if d.final_skill_score > 0]
        missing_m = [d for d in selected_cand.must_have_details if d.final_skill_score == 0]

        st.markdown(f"**Must-Have Skills Coverage:** `{selected_cand.must_have_coverage * 100:.1f}%`")
        if matched_m:
            st.markdown("✅ **Matched Must-Haves:**")
            for m in matched_m:
                ev_label = f"evidence={m.evidence_factor} in {', '.join(m.sections_found) or 'text'}"
                if m.match_type == "exact":
                    st.markdown(f"- **{m.jd_skill}** ({ev_label})")
                else:
                    st.markdown(f"- **{m.jd_skill}** (semantically matched via *{m.matched_candidate_skill}*, sim={m.similarity:.2f})")

        if missing_m:
            st.markdown("❌ **Missing Must-Haves:**")
            for miss in missing_m:
                st.markdown(f"- <span style='color:#f87171;'>{miss.jd_skill}</span>", unsafe_allow_html=True)

        matched_n = [d for d in selected_cand.nice_to_have_details if d.final_skill_score > 0]
        if matched_n:
            st.markdown("🌟 **Bonus Skills:** " + ", ".join(d.jd_skill for d in matched_n))

    # Full Explanation Expander
    with st.expander("📄 View Full Decision Support Explanation"):
        st.markdown(selected_cand.explanation)
