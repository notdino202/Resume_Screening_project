"""SQLite persistent storage layer for Job Descriptions, Parsed Resumes, and Screening Results."""

from __future__ import annotations
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Optional, Any

from src.jd.models import JobDescription
from src.scoring.scorer import CandidateScoreResult


DEFAULT_DB_PATH = Path("data/screening_database.sqlite")


class ScreeningDatabase:
    """Manages SQLite storage and caching for resumes, jobs, and screening results."""

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        """Initializes normalized database tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Job Descriptions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS job_descriptions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    raw_text TEXT NOT NULL,
                    must_have_skills TEXT NOT NULL,
                    nice_to_have_skills TEXT NOT NULL,
                    required_years REAL,
                    required_education_level INTEGER,
                    required_education_name TEXT,
                    requirements_json TEXT,
                    weights_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 2. Parsed Resumes cache table (keyed by SHA256 file_hash)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS parsed_resumes (
                    file_hash TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    clean_text TEXT NOT NULL,
                    sections_json TEXT,
                    skills_json TEXT,
                    education_json TEXT,
                    experience_json TEXT,
                    quality_flags_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 3. Screening Results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS screening_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    final_score REAL NOT NULL,
                    tier TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    skill_score REAL NOT NULL,
                    semantic_score REAL NOT NULL,
                    experience_score REAL NOT NULL,
                    education_score REAL NOT NULL,
                    must_have_coverage REAL NOT NULL,
                    nice_to_have_coverage REAL NOT NULL,
                    must_details_json TEXT,
                    nice_details_json TEXT,
                    flags_json TEXT,
                    explanation TEXT,
                    screened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(job_id) REFERENCES job_descriptions(id)
                );
            """)
            conn.commit()

    def save_job_description(self, job_id: str, jd: JobDescription) -> None:
        """Persists or updates a Job Description record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO job_descriptions (
                    id, title, raw_text, must_have_skills, nice_to_have_skills,
                    required_years, required_education_level, required_education_name,
                    requirements_json, weights_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job_id,
                jd.title,
                jd.raw_text,
                json.dumps(jd.must_have_skills),
                json.dumps(jd.nice_to_have_skills),
                jd.required_years,
                jd.required_education_level,
                jd.required_education_name,
                json.dumps(jd.requirements),
                json.dumps(jd.weights_override) if jd.weights_override else None,
                datetime.now().isoformat(),
            ))
            conn.commit()

    def get_job_description(self, job_id: str) -> Optional[JobDescription]:
        """Retrieves a Job Description by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM job_descriptions WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return JobDescription(
                title=row["title"],
                raw_text=row["raw_text"],
                must_have_skills=json.loads(row["must_have_skills"]),
                nice_to_have_skills=json.loads(row["nice_to_have_skills"]),
                required_years=row["required_years"],
                required_education_level=row["required_education_level"],
                required_education_name=row["required_education_name"],
                requirements=json.loads(row["requirements_json"] or "[]"),
                weights_override=json.loads(row["weights_json"]) if row["weights_json"] else None,
            )

    def cache_parsed_resume(
        self,
        file_hash: str,
        file_name: str,
        file_type: str,
        clean_text: str,
        sections_dict: dict,
        skills_dict: dict,
        education_dict: dict,
        experience_dict: dict,
        quality_flags: list[str],
    ) -> None:
        """Caches parsed resume data indexed by file hash."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO parsed_resumes (
                    file_hash, file_name, file_type, clean_text,
                    sections_json, skills_json, education_json, experience_json,
                    quality_flags_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                file_hash,
                file_name,
                file_type,
                clean_text,
                json.dumps(sections_dict),
                json.dumps(skills_dict),
                json.dumps(education_dict),
                json.dumps(experience_dict),
                json.dumps(quality_flags),
                datetime.now().isoformat(),
            ))
            conn.commit()

    def get_cached_resume(self, file_hash: str) -> Optional[dict]:
        """Retrieves cached resume parse output by hash if present."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM parsed_resumes WHERE file_hash = ?", (file_hash,))
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "file_hash": row["file_hash"],
                "file_name": row["file_name"],
                "file_type": row["file_type"],
                "clean_text": row["clean_text"],
                "sections": json.loads(row["sections_json"] or "{}"),
                "skills": json.loads(row["skills_json"] or "{}"),
                "education": json.loads(row["education_json"] or "{}"),
                "experience": json.loads(row["experience_json"] or "{}"),
                "quality_flags": json.loads(row["quality_flags_json"] or "[]"),
            }

    def save_screening_results(self, job_id: str, results: list[CandidateScoreResult], file_hashes: dict[str, str]) -> None:
        """Stores batch screening run outputs."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for r in results:
                f_hash = file_hashes.get(r.file_name, r.candidate_id)
                must_details = [asdict(d) for d in r.must_have_details]
                nice_details = [asdict(d) for d in r.nice_to_have_details]
                cursor.execute("""
                    INSERT INTO screening_results (
                        job_id, file_hash, file_name, final_score, tier, rank,
                        skill_score, semantic_score, experience_score, education_score,
                        must_have_coverage, nice_to_have_coverage, must_details_json,
                        nice_details_json, flags_json, explanation, screened_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    job_id,
                    f_hash,
                    r.file_name,
                    r.final_score,
                    r.tier,
                    r.rank,
                    r.skill_score,
                    r.semantic_score,
                    r.experience_score,
                    r.education_score,
                    r.must_have_coverage,
                    r.nice_to_have_coverage,
                    json.dumps(must_details),
                    json.dumps(nice_details),
                    json.dumps(r.advisory_flags),
                    r.explanation,
                    datetime.now().isoformat(),
                ))
            conn.commit()

    def get_screening_results_for_job(self, job_id: str) -> list[dict]:
        """Loads historical screening results for a given job."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM screening_results WHERE job_id = ? ORDER BY rank ASC", (job_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
