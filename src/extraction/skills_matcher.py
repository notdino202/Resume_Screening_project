"""Skill extraction module using spaCy PhraseMatcher with evidence levels and alias normalization."""

from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Optional

import spacy
from spacy.matcher import PhraseMatcher
from spacy.util import filter_spans

from .section_detector import ResumeSections, SectionType


@dataclass
class ExtractedSkill:
    """Represents an extracted skill normalized to its canonical taxonomy name."""
    skill_name: str
    category: str
    matched_text: str
    evidence_factor: float
    sections_found: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)


class SkillsMatcher:
    """Extracts technical skills using spaCy PhraseMatcher with section-aware evidence weighting."""

    EVIDENCE_WEIGHTS = {
        SectionType.EXPERIENCE: 1.0,
        SectionType.PROJECTS: 1.0,
        SectionType.CERTIFICATIONS: 0.9,
        SectionType.SKILLS: 0.7,
        SectionType.SUMMARY: 0.5,
        SectionType.OTHER: 0.5,
    }

    def __init__(self, taxonomy_path: Optional[str | Path] = None):
        """Initializes blank English NLP pipeline and compiles taxonomy patterns."""
        self.nlp = spacy.blank("en")
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")

        if taxonomy_path is None:
            taxonomy_path = Path(__file__).resolve().parent.parent.parent / "data" / "skills_taxonomy.json"
        self.taxonomy_path = Path(taxonomy_path)
        self.taxonomy: dict[str, dict] = self._load_taxonomy()
        self._build_matcher()

    def _load_taxonomy(self) -> dict[str, dict]:
        """Loads canonical skills and aliases from taxonomy JSON."""
        if not self.taxonomy_path.exists():
            raise FileNotFoundError(f"Skills taxonomy file not found: {self.taxonomy_path}")
        with open(self.taxonomy_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_matcher(self) -> None:
        """Compiles phrase patterns for all canonical names and aliases."""
        for canonical_name, meta in self.taxonomy.items():
            aliases = meta.get("aliases", [])
            patterns = set([canonical_name.lower()] + [a.lower() for a in aliases if a.strip()])
            docs = [self.nlp.make_doc(p) for p in patterns]
            self.matcher.add(canonical_name, docs)

    def extract_skills_from_text(self, text: str, default_section: SectionType = SectionType.OTHER) -> dict[str, ExtractedSkill]:
        """Extracts skills from a single block of raw text.

        Args:
            text: Input string.
            default_section: Fallback section type for evidence weighting.

        Returns:
            Dictionary of canonical skill name -> ExtractedSkill.
        """
        if not text or not text.strip():
            return {}

        doc = self.nlp(text)
        matches = self.matcher(doc, as_spans=True)
        filtered = filter_spans(matches)

        evidence_factor = self.EVIDENCE_WEIGHTS.get(default_section, 0.5)
        extracted: dict[str, ExtractedSkill] = {}

        for span in filtered:
            canonical = span.label_
            category = self.taxonomy.get(canonical, {}).get("category", "General")
            matched_text = span.text

            # Sentence / snippet context window
            start_ctx = max(0, span.start - 5)
            end_ctx = min(len(doc), span.end + 5)
            context_snip = doc[start_ctx:end_ctx].text

            if canonical not in extracted:
                extracted[canonical] = ExtractedSkill(
                    skill_name=canonical,
                    category=category,
                    matched_text=matched_text,
                    evidence_factor=evidence_factor,
                    sections_found=[default_section.value],
                    contexts=[context_snip],
                )
            else:
                curr = extracted[canonical]
                curr.evidence_factor = max(curr.evidence_factor, evidence_factor)
                if default_section.value not in curr.sections_found:
                    curr.sections_found.append(default_section.value)
                if len(curr.contexts) < 3:
                    curr.contexts.append(context_snip)

        return extracted

    def extract_skills_from_sections(self, resume_sections: ResumeSections) -> dict[str, ExtractedSkill]:
        """Extracts skills section-by-section and aggregates with maximum evidence weighting.

        Evidence factor rule:
        - Experience / Projects: 1.0
        - Certifications: 0.9
        - Skills section only: 0.7
        - Elsewhere: 0.5

        Args:
            resume_sections: Segmented ResumeSections instance.

        Returns:
            Dictionary mapping canonical skill name -> ExtractedSkill.
        """
        consolidated: dict[str, ExtractedSkill] = {}

        for sec_type in SectionType:
            sec_text = resume_sections.get_text(sec_type)
            if not sec_text:
                continue

            sec_skills = self.extract_skills_from_text(sec_text, default_section=sec_type)
            weight = self.EVIDENCE_WEIGHTS.get(sec_type, 0.5)

            for canonical, skill_obj in sec_skills.items():
                if canonical not in consolidated:
                    consolidated[canonical] = ExtractedSkill(
                        skill_name=canonical,
                        category=skill_obj.category,
                        matched_text=skill_obj.matched_text,
                        evidence_factor=weight,
                        sections_found=[sec_type.value],
                        contexts=skill_obj.contexts,
                    )
                else:
                    curr = consolidated[canonical]
                    curr.evidence_factor = max(curr.evidence_factor, weight)
                    if sec_type.value not in curr.sections_found:
                        curr.sections_found.append(sec_type.value)
                    for ctx in skill_obj.contexts:
                        if len(curr.contexts) < 4:
                            curr.contexts.append(ctx)

        return consolidated
