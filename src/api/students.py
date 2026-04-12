"""
students.py
-----------
Layer 2: Postgres inserts and student write operations.

Sits between main.py (HTTP layer) and recommendation_agent.py (AI layer).
Responsible for all DB inserts — main.py never writes to Postgres directly.

Flow:
    main.py  →  register_student()  →  Postgres INSERT
                                    →  generate_recommendation()  →  AI pipeline
"""

import os
import logging
import bcrypt
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def get_db():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "34.23.27.68"),
        port=int(os.getenv("DB_PORT", 5432)),
        dbname=os.getenv("DB_NAME", "courseweave"),
        user=os.getenv("DB_USER", "courseweave_user"),
        password=os.getenv("DB_PASSWORD", ""),
    )
    conn.autocommit = True
    return conn


def register_student(
    name: str,
    email: str,
    password: str,
    program_code: str,
    target_career: str,
) -> dict:
    """
    Insert new student into Postgres, then trigger the recommendation pipeline.

    Returns:
        {
            "student":        { id, name, email, program_code, target_career },
            "recommendation": full generate_recommendation() response dict
        }

    Raises:
        psycopg2.errors.UniqueViolation  if email already exists
        Exception                        for any other DB error
    """
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    # ── Layer 2: Postgres INSERT ─────────────────────────────────────────────
    logger.info("Registering student: %s (%s)", email, program_code)
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        INSERT INTO students (name, email, program_code, target_career, password_hash)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id, name, email, program_code, target_career
        """,
        (name, email, program_code, target_career, hashed),
    )
    student = dict(cur.fetchone())
    conn.close()
    logger.info("Student %d inserted into Postgres — triggering AI pipeline", student["id"])

    # ── Layer 3: AI pipeline (called after DB commit is confirmed) ────────────
    try:
        from src.agents.recommendation_agent import generate_recommendation
        recommendation = generate_recommendation(student_id=student["id"])
    except Exception as e:
        logger.error("Recommendation pipeline failed for new student %d: %s", student["id"], e)
        recommendation = None

    return {
        "student": student,
        "recommendation": recommendation,
    }