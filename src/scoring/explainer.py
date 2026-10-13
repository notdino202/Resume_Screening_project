"""Plain-language explanation generator for decision support."""

from __future__ import annotations
from typing import Optional
from .scorer import CandidateScoreResult


class ExplanationGenerator:
    """Generates plain-language, transparent recruiter summaries explaining scoring rationale."""

    @staticmethod
    def generate_explanation(result: CandidateScoreResult) -> str:
        """Constructs human-readable markdown explanation for a candidate."""
        lines: list[str] = []

        # Header & Tier summary
        lines.append(f"### Screening Assessment: {result.file_name}")
        lines.append(
            f"**Recommendation Tier:** `{result.tier.upper()}` | **Final Score:** `{result.final_score:.1f} / 100` (Rank #{result.rank})"
        )
        lines.append(
            "> *Advisory note: This system assists decisions by scoring alignment against the JD. Final recruitment decisions rest entirely with the hiring team.*"
        )
        lines.append("")

        # 1. Component breakdown
        w = result.weights_used
        lines.append("#### Component Score Breakdown")
        lines.append(
            f"- **Skills ({w.get('skill', 0.4)*100:.0f}% weight):** `{result.skill_score * 100:.1f}%` "
            f"(Must-have coverage: `{result.must_have_coverage * 100:.1f}%`, Nice-to-have: `{result.nice_to_have_coverage * 100:.1f}%`)"
        )
        lines.append(
            f"- **Semantic Alignment ({w.get('semantic', 0.3)*100:.0f}% weight):** `{result.semantic_score * 100:.1f}%` "
            f"(Average raw cosine similarity: `{result.semantic_match_result.raw_average_similarity:.2f}` if available)"
            if result.semantic_match_result
            else f"- **Semantic Alignment:** `{result.semantic_score * 100:.1f}%`"
        )
        req_str = f" / {result.required_years} yrs required" if result.required_years else " (No min requirement)"
        lines.append(
            f"- **Experience ({w.get('experience', 0.2)*100:.0f}% weight):** `{result.experience_score * 100:.1f}%` "
            f"({result.relevant_experience_years:.1f} relevant yrs, {result.total_experience_years:.1f} total yrs{req_str}, overlaps merged)"
        )
        lines.append(
            f"- **Education ({w.get('education', 0.1)*100:.0f}% weight):** `{result.education_score * 100:.1f}%` "
            f"(Candidate: {result.candidate_education_name} [Level {result.candidate_education_level}], Required: Level {result.required_education_level})"
        )
        lines.append("")

        # 2. Skill details
        lines.append("#### Skills Analysis")
        matched_must = [d for d in result.must_have_details if d.final_skill_score > 0]
        missing_must = [d for d in result.must_have_details if d.final_skill_score == 0]

        if matched_must:
            m_items = []
            for d in matched_must:
                sec_str = f"in {', '.join(d.sections_found)}" if d.sections_found else "extracted"
                if d.match_type == "exact":
                    m_items.append(f"**{d.jd_skill}** ({sec_str}, evidence={d.evidence_factor})")
                else:
                    m_items.append(f"**{d.jd_skill}** (via related skill *{d.matched_candidate_skill}*, sim={d.similarity:.2f})")
            lines.append(f"- **Matched Must-Have Skills ({len(matched_must)}):** " + ", ".join(m_items))

        if missing_must:
            lines.append(
                f"- **Missing Must-Have Skills ({len(missing_must)}):** "
                + ", ".join(f"`{d.jd_skill}`" for d in missing_must)
            )

        matched_nice = [d for d in result.nice_to_have_details if d.final_skill_score > 0]
        if matched_nice:
            lines.append(
                f"- **Bonus / Preferred Skills ({len(matched_nice)}):** "
                + ", ".join(f"{d.jd_skill}" for d in matched_nice)
            )
        lines.append("")

        # 3. Semantic requirement evidence
        if result.semantic_match_result and result.semantic_match_result.requirement_details:
            lines.append("#### Top Requirement Evidence (Semantic Match)")
            # Show top 2-3 requirements with best matching excerpts
            top_reqs = sorted(
                result.semantic_match_result.requirement_details,
                key=lambda x: x.cosine_similarity,
                reverse=True,
            )[:3]
            for tr in top_reqs:
                short_req = tr.jd_requirement[:80] + ("..." if len(tr.jd_requirement) > 80 else "")
                short_chunk = tr.best_chunk[:130].replace("\n", " ") + "..."
                lines.append(f"- **JD Requirement:** *\"{short_req}\"*")
                lines.append(f"  - **Resume Evidence (Sim {tr.cosine_similarity:.2f}):** *\"{short_chunk}\"*")
            lines.append("")

        # 4. Advisory flags
        if result.advisory_flags:
            lines.append("#### Advisory Flags (Review Needed)")
            for f in result.advisory_flags:
                lines.append(f"- `FLAG`: **{f}**")
        else:
            lines.append("#### Advisory Flags")
            lines.append("- None (all minimum requirements and quality checks satisfied).")

        return "\n".join(lines)
