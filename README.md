# TalentRank: AI Resume Screening Decision-Support System

[![Tests: 44 Passing](https://img.shields.io/badge/Tests-44%20Passing-brightgreen.svg)]()
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()
[![NLP: spaCy%20%2B%20SentenceTransformers](https://img.shields.io/badge/NLP-spaCy%20%7C%20all--MiniLM--L6--v2-orange.svg)]()
[![Privacy: 100%20Local%20CPU](https://img.shields.io/badge/Privacy-100%25%20Local%20CPU-success.svg)]()
[![Ethics: Zero%20Auto--Reject](https://img.shields.io/badge/Ethics-Decision%20Support%20Only-purple.svg)]()

---

## 1. Executive Summary & Problem Statement

Modern talent acquisition teams face overwhelming applicant volume, leading to recruiter burnout and superficial keyword searches that overlook promising non-traditional candidates. Existing automated applicant tracking systems (ATS) often rely on crude keyword filters or opaque "black-box" models that introduce algorithmic bias and risk unlawful auto-rejection.

**TalentRank** is a 100% local, deterministic, PII-blind decision-support screening engine that accepts a Job Description (JD) and a batch of candidate resumes (PDF, DOCX, TXT), extracts verifiable qualifications (skills, experience, education), computes a multi-faceted hybrid score, and delivers an auditable, ranked leaderboard with plain-language explanations.

### Core Operating Principles
- **Decision-Support Only (Zero Auto-Rejection):** TalentRank recommends and ranks; human recruiters retain exclusive hiring authority. No candidate is ever dropped automatically.
- **Strict PII-Blind Scoring:** Candidate names, genders, contact information, photos, and addresses are systematically scrubbed prior to semantic embedding and scoring.
- **100% Local Processing:** Runs on standard CPU using lightweight, optimized models (`all-MiniLM-L6-v2` and `spaCy`). Zero applicant data leaves the local machine or perimeter network.
- **Deterministic & Auditable:** Transparent scoring rules, reproducible results, and sub-second re-scoring when weights are adjusted.

---

## 2. Hybrid Scoring Architecture

Candidates are evaluated using a four-pillar normalized scoring formula:

$$\text{Final Score} = 100 \times (0.40 \times S_{\text{skill}} + 0.30 \times S_{\text{semantic}} + 0.20 \times S_{\text{experience}} + 0.10 \times S_{\text{education}})$$

All four component scores are normalized to $[0.0, 1.0]$. The default weights reside in `config.yaml` and can be interactively adjusted or overridden per Job Description, validated to sum to $1.00$.

```
+----------------------------------------------------------------------------------+
|                             HYBRID SCORING PIPELINE                              |
+----------------------------------------------------------------------------------+
|                                                                                  |
|  [Resume PDF/DOCX/TXT] ──> [PII Scrubbing & Cleaning] ──> [Section Detection]    |
|                                                                     │            |
|       ┌──────────────────────┬──────────────────────┬───────────────┴────────┐   |
|       ▼                      ▼                      ▼                        ▼   |
| [Skills Matcher]     [Semantic Matcher]    [Experience Parser]     [Edu Parser]  |
| - Boundary matcher   - Requirements chunk  - Date ranges & gaps    - Level 1-5   |
| - Must/Nice split    - Bi-encoder cosine   - Interval merging      - Step-down   |
| - Evidence weight    - Mean of max matches - Seniority bonus       - Field match |
|   (Weight: 40%)          (Weight: 30%)         (Weight: 20%)       (Weight: 10%) |
|       │                      │                      │                        │   |
|       └──────────────────────┴──────────┬───────────┴────────────────────────┘   |
|                                         ▼                                        |
|                          [Hybrid Weighted Aggregation]                           |
|                                         │                                        |
|                                         ▼                                        |
|               [Tier Assignment, Advisory Flags & Explanation Engine]             |
|                                         │                                        |
|                                         ▼                                        |
|                     [Streamlit Dashboard & Leaderboard Export]                   |
+----------------------------------------------------------------------------------+
```

### Component Details

#### 1. Skill Matcher ($S_{\text{skill}}$, Weight: 0.40)
- **Coverage Split:** $0.75 \times \text{Must-Have Coverage} + 0.25 \times \text{Nice-To-Have Coverage}$.
- **Match Types:**
  - *Exact & Alias Match:* $1.0$ (via token-boundary preserved `PhraseMatcher` to avoid substring collisions such as "Java" inside "JavaScript").
  - *Semantically Related Skill Match:* If embedding cosine similarity $\ge 0.75$, matched with weight capped at $0.80$.
  - *Missing Skill:* $0.0$.
- **Evidence Hierarchy Weighting:** Skills proven in work history receive higher confidence than uncontextualized keyword lists:
  - Work Experience / Projects: $\times 1.0$
  - Certifications: $\times 0.9$
  - Skills List Section Only: $\times 0.7$
  - Other Sections: $\times 0.5$

#### 2. Semantic Matcher ($S_{\text{semantic}}$, Weight: 0.30)
- Uses `sentence-transformers/all-MiniLM-L6-v2` running locally on CPU with SHA-256 embedding caching.
- Chunks the Job Description into individual bullet requirements.
- Computes cosine similarity against candidate resume sections and takes the mean of the top match per requirement.

#### 3. Experience Parser ($S_{\text{experience}}$, Weight: 0.20)
- Extracts chronological date ranges (handles `YYYY-MM`, `MM/YYYY`, `Month YYYY`, `Present`, `Current`).
- **Interval Merging:** Merges concurrent/overlapping job dates on the calendar timeline to prevent double-counting moonlight or concurrent consulting roles.
- **Scoring:** Full points ($1.0$) if total years $\ge$ required years; linear penalty below requirement ($\text{years} / \text{required}$); small capped bonus ($+0.10$) for relevant senior experience.

#### 4. Education Parser ($S_{\text{education}}$, Weight: 0.10)
- Hierarchical degree mapping: High School ($1$), Associate/Diploma ($2$), Bachelor's ($3$), Master's ($4$), Doctorate/PhD ($5$).
- Negative lookbehind regex (`(?<!high school )diploma`) prevents "High School Diploma" from misclassifying as Level 2 Associate/Diploma.
- Degree scoring: Exact or higher $= 1.0$, one level below $= 0.7$, two levels below $= 0.4$, otherwise $0.0$.
- Field of study alignment bonus/penalty ($\pm 0.10$).
- Scopes strictly to minimum/must-have qualifications to prevent degree inflation from preferred criteria.

---

## 3. Match Tiers, Advisory Flags & Explanations

### Candidate Match Tiers
- 🟢 **Strong Match ($\ge 75.0$):** High alignment across core skills, semantic requirements, and career trajectory.
- 🟡 **Moderate Match ($50.0 - 74.9$):** Solid foundational background; may have minor skill gaps or experience deltas easily bridged by onboarding.
- 🔴 **Weak Match ($< 50.0$):** Significant divergence from must-have skills or core job scope.

### Transparent Advisory Flags (Never Auto-Reject)
- `MISSING_CRITICAL_SKILL`: Flagged when candidate lacks one or more non-negotiable must-have competencies.
- `OVERQUALIFIED`: Flagged when candidate has $\ge 2\times$ required experience or exceeds senior scope.
- `CAREER_GAP`: Flagged when contiguous employment gaps exceed $9$ months.
- `CREDENTIAL_MISMATCH`: Flagged when highest degree level is below required baseline.
- `LOW_CONFIDENCE_EXTRACTION`: Advisory flag when resume text is extremely sparse or parsed via degraded OCR scan.

### Plain-Language Explanations
Every candidate receives an auditable breakdown detailing:
- Exact and semantic skills matched with section provenance.
- List of missing must-have and nice-to-have skills.
- Top semantic requirement alignment and rationale.
- Experience delta (years observed vs. years required) and timeline gap analysis.
- Education degree tier alignment.

---

## 4. Evaluation & Ablation Study

Evaluated on 100 labeled pairs across 5 distinct Job Descriptions and 20 multi-format candidate resumes (PDF, DOCX, TXT):

### Extraction Quality
- **Macro Skill F1:** `0.5446` (measured against multi-domain ground truth annotations).
- **Experience MAE:** `2.33 years` (RMSE: `2.50 years`), with automated date normalization and concurrent overlap merging.

### Ablation Study Results
| Variant / Model | Mean NDCG@10 | Mean Precision@5 | Key Observations |
|:---|:---:|:---:|:---|
| **TF-IDF only** | 0.9597 | 0.8000 | Baseline bag-of-words; susceptible to keyword stuffing and lexical mismatches. |
| **Semantic only** | 0.9438 | 0.7600 | Captures conceptual similarity well, but misses strict hard technical skill gates. |
| **Skills only** | 0.9834 | 0.8800 | High precision on explicit requirements; blind to contextual experience depth. |
| **Skills + Semantic** | 0.9826 | 0.8800 | Balances hard requirements and conceptual scope. |
| **Full Hybrid (Proposed)** | **0.9830** | **0.8800** | Combines skills, semantic context, career duration, and educational credentials. |

### Train / Held-Out Generalization
- **Training Split (JDs 1–3):** Mean NDCG@10 = `0.9796` (Optimal weights: Skill 0.40, Semantic 0.30, Experience 0.20, Education 0.10).
- **Held-Out Test Split (JDs 4–5):** Mean NDCG@10 = `0.9882`, Mean Precision@5 = `0.8000`.

---

## 5. Fairness, Privacy & Robustness Verification

TalentRank implements proactive algorithmic fairness and defensive text preprocessing:

1. **Demographic Invariance (Name & Pronoun Swapping):**
   - Candidate headers (names, emails, phone numbers, addresses) are scrubbed before computing embeddings.
   - Pronouns (*he/him*, *she/her*, *they/them*) are normalized to neutral tokens.
   - **Verification:** Unit tests confirm $100\%$ score equality ($\Delta = 0.000$) across demographic swaps.
2. **Adversarial Keyword-Stuffing Defense:**
   - Skills detected solely in isolated lists receive an evidence weighting cap ($0.70$) vs. experiential proof ($1.00$).
   - Requirement-level semantic chunking requires holistic context, neutralizing hidden white-text keyword padding.
3. **Format & Scan Robustness:**
   - Multi-format ingestion supporting PDF (`pdfplumber` + `pymupdf`), DOCX (`python-docx`), and plain text.
   - OCR fallback with automated advisory quality flags when text extraction confidence is degraded.

---

## 6. Project Structure

```
Resume_Screening_project/
├── config.yaml                     # Global weights, model config, and threshold settings
├── pytest.ini                      # Pytest runner configuration
├── NOTES.md                        # Architecture decisions & edge case log
├── data/
│   ├── skills_taxonomy.json        # Curated taxonomy (~120 skills with aliases & categories)
│   ├── sample_jds/                 # 5 benchmark job descriptions
│   └── sample_resumes/             # 20 diverse test resumes (PDF, DOCX, TXT)
├── evaluation/
│   ├── evaluator.py                # NDCG@10, Precision@5, F1, MAE computation
│   ├── labeled_pairs.csv           # 100 labeled candidate-JD evaluation pairs
│   └── ablation_study_results.csv  # 5-way model ablation performance metrics
├── scripts/
│   ├── run_evaluation_pipeline.py  # End-to-end evaluation & weight tuning CLI
│   └── generate_evaluation_dataset.py # Synthetic sample resumes generator
├── src/
│   ├── ingestion/                  # Document parsing, hashing, text cleaning
│   │   ├── extractor.py            # PDF/DOCX/TXT multi-engine parser & OCR fallback
│   │   ├── cleaner.py              # Text normalization & encoding cleanup
│   │   └── hasher.py               # SHA-256 fingerprinting for caching
│   ├── preprocessing/              # PII scrubber & demographic anonymizer
│   │   └── pii_scrubber.py         # Name/email/phone stripping & pronoun normalization
│   ├── extraction/                 # Information extraction components
│   │   ├── section_detector.py     # Header & section boundary segmenter
│   │   ├── skills_matcher.py       # PhraseMatcher + evidence location weighting
│   │   ├── experience_parser.py    # Date ranges, interval overlap merging, gaps
│   │   └── education_parser.py     # Degree levels 1-5, negative lookbehind, field match
│   ├── jd/                         # Job description modeling & requirement chunker
│   │   ├── models.py               # Pydantic/dataclass JD schema & weight overrides
│   │   └── parser.py               # Must-have / nice-to-have & qualification extractor
│   ├── matching/                   # Baselines
│   │   └── baselines.py            # TF-IDF cosine & exact skill overlap matchers
│   ├── scoring/                    # Hybrid scoring engine
│   │   ├── embeddings.py           # all-MiniLM-L6-v2 embedding manager with cache
│   │   ├── semantic_matcher.py     # Requirement-chunked cosine similarity
│   │   ├── scorer.py               # Hybrid weighted scoring formula & tiering
│   │   └── explainer.py            # Natural language audit trail generator
│   ├── storage/                    # SQLite persistent store
│   │   └── db.py                   # Caching parsed resumes, runs & JD metadata
│   └── ui/                         # Streamlit web application
│       └── app.py                  # Recruiter interface, interactive weights & CSV export
└── tests/                          # 44 unit & integration tests
    ├── test_ingestion.py           # 7 tests
    ├── test_extraction.py          # 5 tests
    ├── test_education_experience.py# 7 tests
    ├── test_jd_parser.py           # 6 tests
    ├── test_baselines.py           # 3 tests
    ├── test_hybrid_scorer.py       # 5 tests
    ├── test_storage.py             # 3 tests
    ├── test_fairness_robustness.py # 4 tests
    └── test_evaluation.py          # 4 tests
```

---

## 7. Installation & Quickstart

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Git

### 2. Clone and Setup Environment
```bash
git clone https://github.com/your-org/Resume_Screening_project.git
cd Resume_Screening_project

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Run the Automated Test Suite (44 Tests)
```bash
pytest
```

### 4. Run the Evaluation & Ablation Pipeline
```bash
python scripts/run_evaluation_pipeline.py
```

### 5. Launch the Streamlit Decision-Support Web UI
```bash
streamlit run src/ui/app.py
```
*Open [http://localhost:8501](http://localhost:8501) in your browser.*

---

## 8. Streamlit Recruiter Interface Guide

1. **Job Description Configuration (Sidebar):**
   - Select a pre-loaded sample JD (e.g. Senior ML Engineer, DevOps, Fullstack) or paste custom text.
   - Fine-tune required experience years, education level, and comma-separated must-have/nice-to-have skills.
   - Adjust the **Hybrid Scoring Weights** sliders (must sum to $1.00$).
2. **Resume Ingestion:**
   - Upload candidate resumes (PDF, DOCX, TXT) via drag-and-drop or click **"Load 10 Sample Resumes"**.
3. **Screening & Ranking:**
   - Click **"🚀 Screen & Rank Candidates"**. Parsed profiles are cached in SQLite for instant weight re-tuning.
4. **Leaderboard & Candidate Deep-Dive:**
   - View candidates ranked by Final Score with color-coded tier badges and warning tags.
   - Click any candidate to expand the 4-component score breakdown, matched/missing skill chips, date gap timeline, and plain-language explanation.
5. **Export:**
   - Download the complete ranked leaderboard as an auditable CSV report.

---

## 9. Known Limitations & Production Roadmap

- **OCR Engine Availability:** If Tesseract is not installed on the system path, scanned image PDFs fall back to direct text stream parsing and generate an advisory `LOW_QUALITY_EXTRACTION` flag.
- **Multilingual Support:** Current taxonomy and section detection are optimized for English-language resumes; future iterations will integrate multi-lingual sentence models (e.g. `paraphrase-multilingual-MiniLM-L12-v2`).
- **Domain-Specific Taxonomies:** The current taxonomy contains ~120 software/data/cloud skills. Expanding via ESCO / O*NET taxonomies will broaden coverage to healthcare, finance, and legal domains.

---

## 10. License & Ethical Disclaimer
This software is provided for decision-support and applicant triage assistance only. It is designed to assist human recruiters and should not be used as an automated decision-making tool in hiring processes without human oversight.
