"""
students.py
-----------
Handles student registration and course insertion into PostgreSQL.

Flow:
    1. Check if email already exists → return existing student if so
    2. Hash the password
    3. Insert into students table → get student_id
    4. Validate each course_code exists in courses table
    5. Insert valid courses into student_courses
    6. Call generate_recommendation(student_id)
    7. Return student_id + recommendation to frontend
"""

import logging
from datetime import date
from typing import Optional

import bcrypt
import psycopg2
import psycopg2.extras

from src.models.postgres_filter import get_connection
from src.agents.recommendation_agent import generate_recommendation

logger = logging.getLogger(__name__)


# ── Password hashing ──────────────────────────────────────────────────────────

def hash_password(plain_password: str) -> str:
    """Hash a plain text password using bcrypt."""
    return bcrypt.hashpw(
        plain_password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")


def verify_password(plain_password: str, hashed: str) -> bool:
    """Verify a plain text password against a stored hash."""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed.encode("utf-8")
    )


# ── Course validation ─────────────────────────────────────────────────────────

def get_valid_course_codes(course_codes: list[str], conn) -> set[str]:
    """
    Check which course codes actually exist in the courses table.
    Returns only the valid ones — invalid codes are silently skipped.
    Prevents foreign key violations if frontend sends a bad course code.
    """
    if not course_codes:
        return set()

    with conn.cursor() as cur:
        cur.execute("""
            SELECT course_code FROM courses
            WHERE course_code = ANY(%s)
        """, (course_codes,))
        return {row[0] for row in cur.fetchall()}


# ── Core registration logic ───────────────────────────────────────────────────

def register_student(
    name: str,
    email: str,
    program_code: str,
    target_career: str,
    password: str,
    completed_courses: list[dict]
) -> dict:
    """
    Full registration flow:
    1. Check if email already exists
    2. Hash password and insert student
    3. Insert completed courses (skipping invalid course codes)
    4. Run recommendation pipeline
    5. Return structured response

    completed_courses format:
    [
        {"course_code": "IE6400", "completed_at": "2025-05-15", "grade": "A"},
        {"course_code": "IE6700", "grade": "A-"}  # completed_at optional
    ]

    Returns:
    {
        "student_id":    101,
        "name":          "Aisha Patel",
        "program_code":  "MS_DAE",
        "target_career": "Data Engineer",
        "is_new":        True,
        "courses_inserted": 2,
        "courses_skipped":  0,
        "degree_audit":  {...},
        "action":        "recommend" | "ask_path" | "complete",
        "recommendation": "..."
    }
    """
    conn = get_connection()

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

            # ── Step 1: Check if email already exists ─────────────────────────
            cur.execute(
                "SELECT id, name, program_code, target_career FROM students WHERE email = %s",
                (email,)
            )
            existing = cur.fetchone()

            if existing:
                logger.info("Student already exists with email %s — returning existing", email)
                student_id = existing["id"]
                is_new = False
            else:
                # ── Step 2: Hash password and insert student ──────────────────
                password_hash = hash_password(password)

                cur.execute("""
                    INSERT INTO students (name, email, program_code, target_career, password_hash)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (name, email, program_code, target_career, password_hash))

                student_id = cur.fetchone()["id"]
                is_new = True
                logger.info("Inserted new student: %s (id=%d)", name, student_id)

            # ── Step 3: Insert completed courses ──────────────────────────────
            courses_inserted = 0
            courses_skipped  = 0

            if completed_courses and is_new:
                # Validate course codes against courses table
                incoming_codes = [c["course_code"] for c in completed_courses]
                valid_codes    = get_valid_course_codes(incoming_codes, conn)

                for course in completed_courses:
                    code = course["course_code"]

                    if code not in valid_codes:
                        logger.warning(
                            "Skipping unknown course_code: %s for student %d",
                            code, student_id
                        )
                        courses_skipped += 1
                        continue

                    # Use provided date or default to today
                    completed_at = course.get("completed_at") or date.today().isoformat()
                    grade        = course.get("grade")

                    cur.execute("""
                        INSERT INTO student_courses (student_id, course_code, completed_at, grade)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (student_id, course_code) DO NOTHING
                    """, (student_id, code, completed_at, grade))

                    courses_inserted += 1

                logger.info(
                    "Courses inserted: %d, skipped: %d for student %d",
                    courses_inserted, courses_skipped, student_id
                )

        conn.commit()

        # ── Step 4: Run recommendation pipeline ───────────────────────────────
        # This runs AFTER the commit so postgres_filter sees the new data
        logger.info("Running recommendation pipeline for student %d", student_id)
        result = generate_recommendation(student_id=student_id)

        if "error" in result:
            logger.error("Recommendation failed: %s", result["error"])
            return {
                "student_id":       student_id,
                "name":             name,
                "program_code":     program_code,
                "target_career":    target_career,
                "is_new":           is_new,
                "courses_inserted": courses_inserted,
                "courses_skipped":  courses_skipped,
                "error":            result["error"]
            }

        # ── Step 5: Return structured response ────────────────────────────────
        return {
            "student_id":       student_id,
            "name":             result["student"]["name"],
            "program_code":     result["student"]["program_code"],
            "target_career":    result["career_goal"],
            "is_new":           is_new,
            "courses_inserted": courses_inserted,
            "courses_skipped":  courses_skipped,
            "degree_audit":     result["degree_audit"],
            "action":           result["action"],
            "recommendation":   result["recommendation"],
            "courses":          result["courses"],
        }

    except Exception as e:
        conn.rollback()
        logger.error("Registration failed for %s: %s", email, e)
        raise
    finally:
        conn.close()


# ── Course catalog for frontend dropdown ──────────────────────────────────────

def get_courses_for_program(program_code: str) -> list[dict]:
    """
    Returns all active courses for a program.
    Used by the frontend to populate the completed courses dropdown.
    Returns core courses first, then electives — alphabetically within each group.
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT course_code, course_name, course_type, credits
                FROM courses
                WHERE program_code = %s
                AND is_active = TRUE
                AND course_type NOT IN ('Project', 'Thesis')
                ORDER BY
                    CASE WHEN course_type = 'Core' THEN 0 ELSE 1 END,
                    course_code
            """, (program_code,))
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def get_all_programs() -> list[dict]:
    """
    Returns all programs with their requirements.
    Used by frontend to populate the program dropdown.
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT program_code, program_name, total_credits
                FROM program_requirements
                ORDER BY program_code
            """)
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


# ── Test block ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("\n=== Testing register_student ===\n")

    result = register_student(
        name             = "Test Student",
        email            = "test.student@northeastern.edu",
        program_code     = "MS_DAE",
        target_career    = "Data Engineer",
        password         = "testpassword123",
        completed_courses = [
            {"course_code": "IE6400", "completed_at": "2025-05-15", "grade": "A"},
            {"course_code": "IE6700", "completed_at": "2025-05-15", "grade": "A-"},
            {"course_code": "INVALID999", "grade": "A"},  # should be skipped
        ]
    )

    print(f"Student ID:        {result['student_id']}")
    print(f"Is new:            {result['is_new']}")
    print(f"Courses inserted:  {result['courses_inserted']}")
    print(f"Courses skipped:   {result['courses_skipped']}")
    print(f"Action:            {result['action']}")
    print(f"Recommendation:\n{result.get('recommendation', 'N/A')}")

    print("\n=== Testing get_courses_for_program ===\n")
    courses = get_courses_for_program("MS_DAE")
    for c in courses:
        print(f"  {c['course_code']} — {c['course_name']} ({c['course_type']})")