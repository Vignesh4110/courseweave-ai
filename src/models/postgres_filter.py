"""
postgres_filter.py
------------------
Fetches student context from PostgreSQL.
This is the FIRST step in the recommendation pipeline.

Answers two questions with certainty before any vector search:
1. What has the student already completed?
2. What courses are they eligible to take next?

Eligibility rules:
- Must be in student's program
- Must not already be completed
- Must be active
- Core incomplete courses returned FIRST (academic policy)
"""

import os
import logging
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def get_connection():
    """Create and return a PostgreSQL connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", 5432),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )


def get_student_context(student_id: int) -> dict:
    """
    Fetch full student context from PostgreSQL.

    Returns:
    {
        "student_id":        1,
        "name":              "Aisha Patel",
        "email":             "patel.ai@northeastern.edu",
        "program_code":      "MS_DAE",
        "target_career":     "Data Engineer",
        "completed_courses": ["IE6400", "IE6700", "IE6200"],
        "eligible_courses":  ["IE7275", "IE6600", ...],  # core first
        "core_remaining":    ["IE7275", "IE6600"],       # incomplete core
        "electives_available": ["IE7615", "IE7500", ...]
    }

    Returns None if student not found.
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

            # Step 1: Get student profile
            cur.execute("""
                SELECT id, name, email, program_code, target_career
                FROM students
                WHERE id = %s
            """, (student_id,))

            student = cur.fetchone()
            if not student:
                logger.warning("Student %d not found.", student_id)
                return None

            student = dict(student)

            # Step 2: Get completed courses
            cur.execute("""
                SELECT course_code
                FROM student_courses
                WHERE student_id = %s
            """, (student_id,))

            completed = [row["course_code"] for row in cur.fetchall()]

            # Step 3: Get all active courses for student's program
            # excluding already completed ones
            # Core courses come FIRST — academic policy
            cur.execute("""
                SELECT course_code, course_name, course_type, requires_petition
                FROM (
                    SELECT course_code, course_name, course_type, FALSE as requires_petition
                    FROM courses
                    WHERE program_code = %s
                    AND is_active = TRUE
                    AND course_code NOT IN (
                        SELECT course_code FROM student_courses
                        WHERE student_id = %s
                    )

                    UNION

                    SELECT ec.course_code, ec.course_name, 'Elective' as course_type,
                        TRUE as requires_petition
                    FROM elective_courses ec
                    JOIN program_elective_departments ped
                    ON ec.dept_code = ped.dept_code
                    AND ped.program_code = %s
                    WHERE ec.course_code NOT IN (
                        SELECT course_code FROM student_courses
                        WHERE student_id = %s
                    )
                    AND ec.course_code NOT IN (
                        SELECT course_code FROM courses
                        WHERE program_code = %s
                    )
                ) combined
                ORDER BY
                    CASE WHEN course_type = 'Core' THEN 0 ELSE 1 END,
                    course_code
            """, (student["program_code"], student_id,
                student["program_code"], student_id,
                student["program_code"]))

            eligible_rows = cur.fetchall()
            eligible      = [row["course_code"] for row in eligible_rows]
            core_remaining = [
                row["course_code"] for row in eligible_rows
                if row["course_type"] == "Core"
            ]
            electives_available = [
                row["course_code"] for row in eligible_rows
                if row["course_type"] == "Elective"
            ]
            # Track which courses require petition
            petition_required = [
                row["course_code"] for row in eligible_rows
                if row["requires_petition"]
            ]
            logger.info("Eligible courses: %d total (%d require petition)", len(eligible), len(petition_required))

            # Step 4: Get prerequisite graph for eligible courses
            # Used later for reordering recommendations
            if eligible:
                cur.execute("""
                    SELECT course_code, required_course_code
                    FROM prerequisites
                    WHERE course_code = ANY(%s)
                """, (eligible,))

                prereq_rows = cur.fetchall()
                prereq_map  = {}
                for row in prereq_rows:
                    course   = row["course_code"]
                    required = row["required_course_code"]
                    if course not in prereq_map:
                        prereq_map[course] = []
                    prereq_map[course].append(required)
            else:
                prereq_map = {}

            return {
                "student_id":          student["id"],
                "name":                student["name"],
                "email":               student["email"],
                "program_code":        student["program_code"],
                "target_career":       student["target_career"],
                "completed_courses":   completed,
                "eligible_courses":    eligible,       # full list, core first
                "core_remaining":      core_remaining,
                "electives_available": electives_available,
                "prereq_map":          prereq_map,     # course → [required courses]
                "petition_required":   petition_required,  # NEW
            }

    except Exception as e:
        logger.error("Failed to fetch student context for %d: %s", student_id, e)
        raise
    finally:
        conn.close()


def check_prerequisites_satisfied(
    course_code: str,
    completed_courses: list[str],
    prereq_map: dict
) -> tuple[bool, list[str]]:
    """
    Check if a student has satisfied prerequisites for a course.

    Returns:
        (True, [])                    — all prereqs satisfied
        (False, ["IE6400", "IE6700"]) — list of missing prereqs
    """
    required = prereq_map.get(course_code, [])
    missing  = [r for r in required if r not in completed_courses]

    if missing:
        return False, missing
    return True, []


def reorder_by_prerequisites(
    recommended_courses: list[str],
    completed_courses: list[str],
    prereq_map: dict
) -> list[dict]:
    """
    Takes a list of recommended course codes.
    Returns them reordered so prerequisites come before dependents.
    Flags courses that require a prerequisite not yet completed.

    Returns list of dicts:
    [
        {
            "course_code": "IE7275",
            "prereqs_satisfied": True,
            "missing_prereqs": []
        },
        {
            "course_code": "IE7615",
            "prereqs_satisfied": False,
            "missing_prereqs": ["IE7275"]
        }
    ]
    """
    result = []

    for course in recommended_courses:
        satisfied, missing = check_prerequisites_satisfied(
            course, completed_courses, prereq_map
        )
        result.append({
            "course_code":       course,
            "prereqs_satisfied": satisfied,
            "missing_prereqs":   missing
        })

    # Sort: courses with satisfied prereqs first
    result.sort(key=lambda x: (0 if x["prereqs_satisfied"] else 1))

    return result

"""
ADDITION TO postgres_filter.py
-------------------------------
Add these two functions to the bottom of your existing postgres_filter.py.
They handle degree audit logic — credit tracking, path selection, remaining courses.
"""


def get_degree_audit(student_id: int) -> dict:
    """
    Full degree audit for a student.
    Computes credit progress, remaining requirements,
    and what the chatbot should ask/suggest next.

    Returns:
    {
        "student_id":           1,
        "program_code":         "MS_DAE",
        "degree_path":          "undecided",
        "total_credits":        32,
        "credits_completed":    12,
        "credits_remaining":    20,

        "core_credits_required":   20,
        "core_credits_completed":  12,
        "core_credits_remaining":  8,
        "core_courses_remaining":  ["IE6600", "IE7275"],

        "elective_credits_required":  12,  # depends on path
        "elective_credits_completed": 0,
        "elective_credits_remaining": 12,
        "electives_completed":        [],

        "project_available":    True,
        "needs_path_selection": True,   # True if path is undecided

        "on_track":             True,
        "next_action":          "ask_path" | "take_core" | "take_elective" | "complete"
    }
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

            # Get student profile including degree_path
            cur.execute("""
                SELECT id, name, program_code, degree_path, target_career
                FROM students
                WHERE id = %s
            """, (student_id,))

            student = cur.fetchone()
            if not student:
                return {"error": f"Student {student_id} not found."}
            student = dict(student)

            # Get program requirements
            cur.execute("""
                SELECT * FROM program_requirements
                WHERE program_code = %s
            """, (student["program_code"],))

            req = cur.fetchone()
            if not req:
                return {"error": f"Program requirements not found for {student['program_code']}."}
            req = dict(req)

            # Get all completed courses with their types
            cur.execute("""
                SELECT sc.course_code, c.course_type, c.credits
                FROM student_courses sc
                JOIN courses c ON sc.course_code = c.course_code
                WHERE sc.student_id = %s
            """, (student_id,))

            completed_rows  = cur.fetchall()
            completed_core  = [r["course_code"] for r in completed_rows if r["course_type"] == "Core"]
            completed_elec  = [r["course_code"] for r in completed_rows if r["course_type"] == "Elective"]
            credits_completed = sum(r["credits"] for r in completed_rows)

            # Get remaining core courses
            cur.execute("""
                SELECT course_code FROM courses
                WHERE program_code = %s
                  AND course_type = 'Core'
                  AND is_active = TRUE
                  AND course_code NOT IN (
                      SELECT course_code FROM student_courses
                      WHERE student_id = %s
                  )
                ORDER BY course_code
            """, (student["program_code"], student_id))

            core_remaining = [r["course_code"] for r in cur.fetchall()]

            # Compute core credit stats
            core_credits_completed = len(completed_core) * 4
            core_credits_remaining = req["core_credits"] - core_credits_completed

            # Determine elective requirements based on chosen path
            degree_path = student["degree_path"] or "undecided"

            if degree_path == "project":
                elective_credits_required = req["project_elective_credits"]
            elif degree_path == "thesis":
                elective_credits_required = req["thesis_elective_credits"]
            else:
                # coursework or undecided — use full elective requirement
                elective_credits_required = req["elective_credits"]

            elective_credits_completed = len(completed_elec) * 4
            elective_credits_remaining = max(
                0, elective_credits_required - elective_credits_completed
            )

            credits_remaining = req["total_credits"] - credits_completed

            # Determine next action for chatbot
            needs_path_selection = (
                degree_path == "undecided" and
                req["project_available"] and
                core_credits_remaining <= 8  # ask path when nearing end of core
            )

            if credits_remaining <= 0:
                next_action = "complete"
            elif needs_path_selection:
                next_action = "ask_path"
            elif core_credits_remaining > 0:
                next_action = "take_core"
            elif elective_credits_remaining > 0:
                next_action = "take_elective"
            else:
                next_action = "complete"

            return {
                "student_id":               student["id"],
                "name":                     student["name"],
                "program_code":             student["program_code"],
                "program_name":             req["program_name"],
                "degree_path":              degree_path,
                "target_career":            student["target_career"],

                "total_credits":            req["total_credits"],
                "credits_completed":        credits_completed,
                "credits_remaining":        credits_remaining,

                "core_credits_required":    req["core_credits"],
                "core_credits_completed":   core_credits_completed,
                "core_credits_remaining":   max(0, core_credits_remaining),
                "core_courses_remaining":   core_remaining,

                "elective_credits_required":  elective_credits_required,
                "elective_credits_completed": elective_credits_completed,
                "elective_credits_remaining": elective_credits_remaining,
                "electives_completed":        completed_elec,

                "project_available":        req["project_available"],
                "project_credits":          req["project_credits"],
                "needs_path_selection":     needs_path_selection,

                "on_track":                 credits_remaining > 0,
                "next_action":              next_action,
            }

    except Exception as e:
        logger.error("Degree audit failed for student %d: %s", student_id, e)
        raise
    finally:
        conn.close()


def update_degree_path(student_id: int, path: str) -> bool:
    """
    Save student's chosen degree path to DB.
    Called when student tells chatbot which path they want.

    Args:
        path: 'coursework', 'project', or 'thesis'

    Returns True if successful.
    """
    valid_paths = {"coursework", "project", "thesis"}
    if path not in valid_paths:
        logger.error("Invalid degree path: %s", path)
        return False

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE students
                SET degree_path = %s,
                    path_selected_at = NOW()
                WHERE id = %s
            """, (path, student_id))
        conn.commit()
        logger.info("Updated degree path for student %d to %s", student_id, path)
        return True
    except Exception as e:
        conn.rollback()
        logger.error("Failed to update degree path: %s", e)
        return False
    finally:
        conn.close()


# ── Test block ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # ── Test degree audit ────────────────────────────────────────────────────
    print("\n=== Testing degree audit ===\n")

    for student_id in [1, 2, 5]:
        audit = get_degree_audit(student_id)
        if "error" not in audit:
            print(f"Student:              {audit['name']}")
            print(f"Program:              {audit['program_code']}")
            print(f"Path:                 {audit['degree_path']}")
            print(f"Credits completed:    {audit['credits_completed']}/{audit['total_credits']}")
            print(f"Core remaining:       {audit['core_courses_remaining']}")
            print(f"Elective cr needed:   {audit['elective_credits_remaining']}")
            print(f"Needs path select:    {audit['needs_path_selection']}")
            print(f"Next action:          {audit['next_action']}")
            print()

    # ── Test student context ─────────────────────────────────────────────────
    print("\n=== Testing postgres_filter.py ===\n")

    context = get_student_context(1)

    if context:
        print(f"Student:          {context['name']}")
        print(f"Program:          {context['program_code']}")
        print(f"Target career:    {context['target_career']}")
        print(f"Completed:        {context['completed_courses']}")
        print(f"Core remaining:   {context['core_remaining']}")
        print(f"Electives avail:  {context['electives_available']}")
        print(f"Prereq map:       {context['prereq_map']}")

        print("\n--- Prereq check for eligible courses ---")
        for course in context["eligible_courses"][:5]:
            satisfied, missing = check_prerequisites_satisfied(
                course,
                context["completed_courses"],
                context["prereq_map"]
            )
            status = "✅" if satisfied else f"❌ needs {missing}"
            print(f"  {course}: {status}")

        print("\n--- Reorder test ---")
        sample = context["eligible_courses"][:5]
        reordered = reorder_by_prerequisites(
            sample,
            context["completed_courses"],
            context["prereq_map"]
        )
        for r in reordered:
            print(f"  {r}")