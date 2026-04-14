"""
recommendation_agent.py
-----------------------
Wires the full recommendation pipeline together.

Flow:
    1. postgres_filter  → student context (completed, eligible, prereqs)
    2. degree_audit     → credit progress, path selection status
    3. Path check       → if undecided and near end of core, ask path first
    4. query_builder    → enriched skill query from careers.json
    5. retriever        → top courses from Pinecone
    6. prereq reorder   → flag missing prerequisites
    7. Gemini 2.5 Flash → conversational explanation
    8. Log interaction   → save full trace to interaction_logs table

Gemini's role:
    - Receives: student profile + degree audit + retrieved courses + prereq status
    - Returns:  friendly, conversational recommendation
    - CANNOT:   invent courses outside the retrieved set
    - CANNOT:   recommend completed courses
"""

import os
import json
import logging
import psycopg2
from dotenv import load_dotenv
from google import genai

from src.models.postgres_filter import (
    get_student_context,
    get_degree_audit,
    update_degree_path,
    reorder_by_prerequisites
)
from src.models.query_builder import build_query
from src.models.retriever import get_relevant_courses

import time

load_dotenv()

logger = logging.getLogger(__name__)

_gemini_client = None

_gemini_client = None

def _get_gemini_client():
    """Lazily initialize Gemini client — avoids startup crash if Vertex AI is unreachable at import time."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(
            vertexai=True,
            project=os.getenv("GCP_PROJECT_ID"),
            location=os.getenv("GCP_LOCATION", "us-central1"),
        )
    return _gemini_client


def gemini_generate(prompt: str, max_retries: int = 4) -> str:
    """
    Call Gemini 2.5 Flash with exponential backoff retry.
    Handles 429 RESOURCE_EXHAUSTED gracefully — retries up to max_retries times.
    This ensures MLflow evaluation runs never fail due to rate limits.

    Backoff: 7s → 14s → 28s → 56s
    Raises exception only after all retries exhausted.
    """
    delays = [7, 14, 28, 56]

    for attempt in range(max_retries):
        try:
            response = _get_gemini_client().models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text.strip()

        except Exception as e:
            err = str(e)
            is_rate_limit = "429" in err or "RESOURCE_EXHAUSTED" in err

            if is_rate_limit and attempt < max_retries - 1:
                wait = delays[attempt]
                logger.warning(
                    "Gemini rate limit hit (attempt %d/%d) — retrying in %ds...",
                    attempt + 1, max_retries, wait
                )
                time.sleep(wait)
                continue

            # Non-rate-limit error or final attempt — re-raise
            raise


def format_courses_for_prompt(
    courses: list[dict],
    prereq_status: list[dict],
    student_context: dict = None
) -> str:
    """Format retrieved courses + prereq status for the Gemini prompt."""
    lines      = []
    prereq_map = {p["course_code"]: p for p in prereq_status}

    for i, course in enumerate(courses, 1):
        code   = course["course_code"]
        name   = course["course_name"]
        text   = course["text"][:300]
        score  = course["score"]
        status = prereq_map.get(code, {})

        prereq_note = (
            "✅ Prerequisites satisfied"
            if status.get("prereqs_satisfied", True)
            else f"⚠️  Requires completing first: {', '.join(status.get('missing_prereqs', []))}"
        )

        petition_required = (student_context or {}).get("petition_required", [])
        petition_note = (
            "📋 Outside your home department — petition required before enrolling"
            if code in petition_required else ""
        )

        lines.append(
            f"{i}. {code} — {name}\n"
            f"   Relevance score: {score}\n"
            f"   Description: {text}...\n"
            f"   {prereq_note}"
            + (f"\n   {petition_note}" if petition_note else "")
        )

    return "\n\n".join(lines)


def build_recommendation_prompt(
    student_context: dict,
    degree_audit: dict,
    courses: list[dict],
    prereq_status: list[dict],
    career_goal: str,
    career_skills: dict
) -> str:
    """
    Build the Gemini prompt with full degree audit context.
    Gemini now knows the student's credit progress and path.
    """
    courses_text = format_courses_for_prompt(courses, prereq_status, student_context)
    core_skills  = career_skills.get("core_skills", [])
    tools        = career_skills.get("tools", [])

    # Degree progress summary for prompt
    path         = degree_audit.get("degree_path", "undecided")
    path_display = path.title() if path != "undecided" else "Not yet selected"
    credits_done = degree_audit.get("credits_completed", 0)
    credits_left = degree_audit.get("credits_remaining", 0)
    core_left    = degree_audit.get("core_courses_remaining", [])
    elec_cr_left = degree_audit.get("elective_credits_remaining", 0)

    prompt = f"""You are CourseWeave, a friendly academic advisor at Northeastern University.
You help graduate students pick the right courses for their career goals.

STUDENT PROFILE:
- Name: {student_context['name']}
- Program: {degree_audit.get('program_name', student_context['program_code'])}
- Career goal: {career_goal}
- Degree path: {path_display}

DEGREE PROGRESS:
- Credits completed: {credits_done} / {degree_audit.get('total_credits', 32)}
- Credits remaining: {credits_left}
- Core courses still needed: {', '.join(core_left) if core_left else 'All core completed ✅'}
- Elective credits still needed: {elec_cr_left}
- Courses completed: {', '.join(student_context['completed_courses'])}

KEY SKILLS NEEDED FOR {career_goal.upper()} (from real job market data):
- Core skills: {', '.join(core_skills)}
- Tools: {', '.join(tools)}

RETRIEVED COURSES (you may ONLY recommend from this list):
{courses_text}

YOUR TASK:
Write a friendly, conversational recommendation for {student_context['name']}.
- Organize the courses into a realistic semester plan based on their {credits_left} credits remaining
  (assume ~8 credits / semester = 2 courses per semester, semesters labeled "Semester 1", "Semester 2", etc.)
- Start with core courses if any remain — these MUST come first
- Then fill remaining semesters with relevant electives from the retrieved list
- Explain why each course fits their {career_goal} goal specifically
- If a course has unmet prerequisites, slot it AFTER its prerequisite in the plan
- Be warm and encouraging — like a real advisor who knows their full situation
- If a course is marked 📋 (petition required), tell the student they need to 
  petition that department before enrolling — mention it naturally, not as a warning

STRICT RULES:
- Only recommend courses from the retrieved list above
- Never invent course names or codes
- Keep response under 400 words
- Do not repeat all the profile data back verbatim
- Focus on WHY each course matters for their specific career goal and degree progress
"""
    return prompt


def build_path_selection_prompt(
    student_context: dict,
    degree_audit: dict,
    career_goal: str
) -> str:
    """
    Build Gemini prompt when student needs to choose a degree path.
    Called when next_action == 'ask_path'.
    """
    program   = degree_audit.get("program_name", student_context["program_code"])
    credits   = degree_audit.get("credits_completed", 0)
    total     = degree_audit.get("total_credits", 32)
    core_left = degree_audit.get("core_courses_remaining", [])

    proj_cr   = degree_audit.get("project_credits", 4)
    proj_elec = degree_audit.get("elective_credits_required", 12)

    prompt = f"""You are CourseWeave, a friendly academic advisor at Northeastern University.

STUDENT PROFILE:
- Name: {student_context['name']}
- Program: {program}
- Career goal: {career_goal}
- Credits completed: {credits} / {total}
- Core courses still needed: {', '.join(core_left) if core_left else 'All done!'}

The student has not yet selected their degree path. They are near the end of their
core requirements and need to decide before planning electives.

DEGREE PATH OPTIONS for {program}:

1. Coursework Path
   - Complete all remaining core courses
   - Then take electives to reach {total} total credits
   - Best for: students focused on breadth of knowledge

2. Project Path
   - Complete all remaining core courses
   - Take IE 7945 Master's Project ({proj_cr} credits) with a faculty advisor
   - Fewer electives needed ({proj_elec} credits worth)
   - Best for: students who want hands-on research or industry project experience

3. Thesis Path (if available)
   - Complete all remaining core courses
   - Conduct original research and write a thesis
   - Best for: students considering PhD or deep research roles

YOUR TASK:
Write a warm, conversational message to {student_context['name']} explaining
the path options and asking which they'd like to pursue.
Relate each path to their {career_goal} career goal.
Keep it under 200 words. End with a clear question asking which path they prefer.
Do NOT make the choice for them — just inform and ask.
"""
    return prompt


# ── Interaction Logging ──────────────────────────────────────────────────────

def log_interaction(
    student_id, student_name, career_goal, degree_audit,
    skill_query, retrieved_courses, prereq_status,
    gemini_prompt, final_recommendation, action_taken, response_time_sec
):
    """
    Log every user interaction to the interaction_logs table in PostgreSQL.
    Captures the full trace of what happened at each pipeline layer.
    Fails silently — logging should never break the recommendation flow.
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO interaction_logs (
                student_id, student_name, career_goal, degree_audit_summary,
                skill_query, retrieved_courses, prereq_flags,
                gemini_prompt, final_recommendation, action_taken, response_time_sec
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            student_id,
            student_name,
            career_goal,
            json.dumps(degree_audit),
            skill_query,
            json.dumps([
                {
                    "course_code": c["course_code"],
                    "course_name": c["course_name"],
                    "score": c.get("score", 0)
                } for c in retrieved_courses
            ]) if retrieved_courses else "[]",
            json.dumps([
                {
                    "course_code": p["course_code"],
                    "prereqs_satisfied": p.get("prereqs_satisfied", True),
                    "missing": p.get("missing_prereqs", [])
                } for p in prereq_status
            ]) if prereq_status else "[]",
            gemini_prompt,
            final_recommendation,
            action_taken,
            response_time_sec
        ))
        conn.commit()
        cur.close()
        conn.close()
        logger.info("Interaction logged to database for student %d", student_id)
    except Exception as e:
        logger.error("Failed to log interaction: %s", e)


# ── Main Recommendation Function ─────────────────────────────────────────────

def generate_recommendation(
    student_id: int,
    career_goal: str = None,
    top_k: int = 3,
    degree_path: str = None
) -> dict:
    """
    Master recommendation function.
    Orchestrates the full pipeline for a student.

    Args:
        student_id:   Postgres student ID
        career_goal:  Override career goal (uses DB value if None)
        top_k:        Number of courses to recommend
        degree_path:  If provided, saves this path to DB before recommending
                      Values: 'coursework', 'project', 'thesis'

    Returns:
    {
        "student":         { name, program, completed, ... },
        "degree_audit":    { credits, path, remaining, ... },
        "career_goal":     "Data Engineer",
        "action":          "recommend" | "ask_path" | "complete",
        "courses":         [ retrieved course dicts ],
        "prereq_status":   [ prereq check results ],
        "recommendation":  "Gemini's conversational response",
        "career_skills":   { from careers.json }
    }
    """
    start_time = time.time()

    # ── Step 1: Postgres context ─────────────────────────────────────────────
    logger.info("Step 1: Fetching student context for ID %d", student_id)
    student_context = get_student_context(student_id)
    if not student_context:
        return {"error": f"Student {student_id} not found."}

    # ── Step 2: Degree audit ─────────────────────────────────────────────────
    logger.info("Step 2: Running degree audit")
    degree_audit = get_degree_audit(student_id)
    if "error" in degree_audit:
        return {"error": degree_audit["error"]}

    # ── Step 2b: Save path if student just chose one ─────────────────────────
    if degree_path:
        success = update_degree_path(student_id, degree_path)
        if success:
            degree_audit = get_degree_audit(student_id)
            logger.info("Degree path updated to: %s", degree_path)

    # Use DB career goal if not overridden
    if not career_goal:
        career_goal = student_context.get("target_career", "Data Engineer")

    # ── Step 3: Check if student needs to pick a path first ──────────────────
    next_action = degree_audit.get("next_action", "take_core")

    if next_action == "complete":
        recommendation = (
            f"Congratulations {student_context['name']}! "
            f"You've completed all {degree_audit['total_credits']} credits "
            f"for your {degree_audit['program_name']}. "
            f"You're ready to graduate! 🎓"
        )

        # Log the interaction
        log_interaction(
            student_id=student_id,
            student_name=student_context["name"],
            career_goal=career_goal,
            degree_audit=degree_audit,
            skill_query="",
            retrieved_courses=[],
            prereq_status=[],
            gemini_prompt="",
            final_recommendation=recommendation,
            action_taken="complete",
            response_time_sec=round(time.time() - start_time, 2)
        )
        return {
            "student":        student_context,
            "degree_audit":   degree_audit,
            "career_goal":    career_goal,
            "action":         "complete",
            "courses":        [],
            "prereq_status":  [],
            "recommendation": recommendation,
            "career_skills":  {},
        }

    if next_action == "ask_path" and not degree_path:
        logger.info("Step 3: Student needs path selection")
        prompt = build_path_selection_prompt(student_context, degree_audit, career_goal)
        try:
            path_question = gemini_generate(prompt)
        except Exception as e:
            logger.error("Gemini path prompt failed after retries: %s", e)
            path_question = (
                f"Hi {student_context['name']}! Before I suggest your next courses, "
                f"I need to know which degree path you'd like to take: "
                f"Coursework (all electives), Project (IE 7945 + fewer electives), "
                f"or Thesis (research + fewer electives). Which do you prefer?"
            )

        # Log the interaction
        log_interaction(
            student_id=student_id,
            student_name=student_context["name"],
            career_goal=career_goal,
            degree_audit=degree_audit,
            skill_query="",
            retrieved_courses=[],
            prereq_status=[],
            gemini_prompt=prompt,
            final_recommendation=path_question,
            action_taken="ask_path",
            response_time_sec=round(time.time() - start_time, 2)
        )

        return {
            "student":        student_context,
            "degree_audit":   degree_audit,
            "career_goal":    career_goal,
            "action":         "ask_path",
            "courses":        [],
            "prereq_status":  [],
            "recommendation": path_question,
            "career_skills":  {},
        }

    # ── Step 4: Build enriched query ─────────────────────────────────────────
    logger.info("Step 4: Building skill query for '%s'", career_goal)
    query_result  = build_query(career_goal)
    query         = query_result["skill_query"]
    career_skills = query_result["career_skills"]

    # ── Step 4b: Dynamic top_k based on credits remaining ────────────────────
    credits_remaining = degree_audit.get("credits_remaining", 32)
    top_k = max(3, min(credits_remaining // 4, 6))
    logger.info("Dynamic top_k=%d based on %d credits remaining", top_k, credits_remaining)

    # ── Step 5: Retrieve from Pinecone ───────────────────────────────────────
    logger.info("Step 5: Retrieving courses from Pinecone")
    courses = get_relevant_courses(query, student_context, top_k=top_k, career_goal=career_goal)

    if not courses:
        recommendation = (
            f"I couldn't find relevant courses right now for "
            f"{student_context['name']}. Please try again shortly."
        )

        # Log the interaction
        log_interaction(
            student_id=student_id,
            student_name=student_context["name"],
            career_goal=career_goal,
            degree_audit=degree_audit,
            skill_query=query,
            retrieved_courses=[],
            prereq_status=[],
            gemini_prompt="",
            final_recommendation=recommendation,
            action_taken="recommend",
            response_time_sec=round(time.time() - start_time, 2)
        )

        return {
            "student":        student_context,
            "degree_audit":   degree_audit,
            "career_goal":    career_goal,
            "action":         "recommend",
            "courses":        [],
            "prereq_status":  [],
            "recommendation": recommendation,
            "career_skills":  career_skills,
        }

    # ── Step 6: Prereq reorder ───────────────────────────────────────────────
    logger.info("Step 6: Checking prerequisites")
    course_codes  = [c["course_code"] for c in courses]
    prereq_status = reorder_by_prerequisites(
        course_codes,
        student_context["completed_courses"],
        student_context["prereq_map"]
    )
    prereq_order    = [p["course_code"] for p in prereq_status]
    courses_ordered = sorted(
        courses,
        key=lambda c: prereq_order.index(c["course_code"])
        if c["course_code"] in prereq_order else 999
    )

    # ── Step 7: Gemini explanation ───────────────────────────────────────────
    logger.info("Step 7: Generating Gemini recommendation")
    prompt = build_recommendation_prompt(
        student_context, degree_audit, courses_ordered,
        prereq_status, career_goal, career_skills
    )
    try:
        recommendation = gemini_generate(prompt)
    except Exception as e:
        logger.error("Gemini generation failed after retries: %s", e)
        recommendation = (
            f"Here are your top {top_k} recommended courses: "
            + ", ".join([c["course_name"] for c in courses_ordered])
        )

    # ── Step 8: Log interaction to database ──────────────────────────────────
    logger.info("Step 8: Logging interaction to database")
    log_interaction(
        student_id=student_id,
        student_name=student_context["name"],
        career_goal=career_goal,
        degree_audit=degree_audit,
        skill_query=query,
        retrieved_courses=courses_ordered,
        prereq_status=prereq_status,
        gemini_prompt=prompt,
        final_recommendation=recommendation,
        action_taken="recommend",
        response_time_sec=round(time.time() - start_time, 2)
    )

    return {
        "student":        student_context,
        "degree_audit":   degree_audit,
        "career_goal":    career_goal,
        "action":         "recommend",
        "courses":        courses_ordered,
        "prereq_status":  prereq_status,
        "recommendation": recommendation,
        "career_skills":  career_skills,
    }


def generate_followup(
    student_id: int,
    session_context: dict,
    conversation_history: list,
) -> dict:
    """
    Handle follow-up questions within an existing session.
    Re-fetches student/degree data from DB but skips Pinecone entirely.
    Uses conversation history for context-aware Gemini responses.
    """
    logger.info("Follow-up turn for student %d (%d messages in history)", student_id, len(conversation_history))

    student_context = get_student_context(student_id)
    if not student_context:
        return {"error": f"Student {student_id} not found."}

    degree_audit  = get_degree_audit(student_id)
    courses       = session_context.get("courses", [])
    prereq_status = session_context.get("prereq_status", [])
    career_goal   = session_context.get("career_goal", student_context.get("target_career", ""))
    career_skills = session_context.get("career_skills", {})

    courses_text  = format_courses_for_prompt(courses, prereq_status, student_context)

    history_lines = []
    for msg in conversation_history:
        speaker = student_context["name"] if msg["role"] == "user" else "CourseWeave"
        history_lines.append(f"{speaker}: {msg['text']}")
    history_text = "\n\n".join(history_lines)

    core_skills = career_skills.get("core_skills", [])
    tools       = career_skills.get("tools", [])
    prompt = f"""You are CourseWeave, a friendly academic advisor at Northeastern University.
Continue this advising conversation with {student_context['name']}.

STUDENT PROFILE:
- Program: {degree_audit.get('program_name', student_context['program_code'])}
- Career goal: {career_goal}
- Credits completed: {degree_audit.get('credits_completed', 0)} / {degree_audit.get('total_credits', 32)}
- Completed courses: {', '.join(student_context['completed_courses']) or 'None yet'}

KEY SKILLS NEEDED FOR {career_goal.upper()} (from real job market data):
- Core skills: {', '.join(core_skills)}
- Tools: {', '.join(tools)}

COURSES ALREADY RECOMMENDED:
{courses_text}

CONVERSATION HISTORY:
{history_text}

Respond naturally and helpfully to the student's latest message.
If they ask about tools or skills, reference the job market data above.
Only reference courses from the list above — never invent new courses or codes.
Keep your response under 200 words."""

    try:
        logger.info("Follow-up prompt includes career skills: core=%d tools=%d", len(core_skills), len(tools))
        response = gemini_generate(prompt)
    except Exception as e:
        logger.error("Gemini follow-up failed: %s", e)
        response = "I'm having trouble connecting right now. Please try again in a moment."

    return {
        "student":        student_context,
        "degree_audit":   degree_audit,
        "career_goal":    career_goal,
        "action":         "followup",
        "courses":        [],
        "prereq_status":  [],
        "recommendation": response,
        "career_skills":  career_skills,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n=== Testing recommendation_agent.py with degree audit ===\n")

    print("--- Test 1: Aisha Patel (path undecided, needs path selection) ---\n")
    result = generate_recommendation(student_id=1)
    print(f"Student:     {result['student']['name']}")
    print(f"Action:      {result['action']}")
    print(f"Credits:     {result['degree_audit']['credits_completed']}/{result['degree_audit']['total_credits']}")
    print(f"Core left:   {result['degree_audit']['core_courses_remaining']}")
    print(f"\n{'='*60}")
    print("GEMINI RESPONSE:")
    print("="*60)
    print(result["recommendation"])

    print("\n\n--- Test 2: Aisha chooses coursework path → recommendations ---\n")
    result2 = generate_recommendation(student_id=1, degree_path="coursework")
    print(f"Student:     {result2['student']['name']}")
    print(f"Action:      {result2['action']}")
    print(f"Path saved:  {result2['degree_audit']['degree_path']}")
    print("\nRetrieved courses:")
    for c in result2.get("courses", []):
        print(f"  {c['course_code']} — {c['course_name']} (score: {c['score']})")
    print(f"\n{'='*60}")
    print("GEMINI RECOMMENDATION:")
    print("="*60)
    print(result2["recommendation"])

    print("\n\n--- Test 3: Carlos Mendez (Data Scientist) ---\n")
    result3 = generate_recommendation(student_id=2, degree_path="project")
    print(f"Student:     {result3['student']['name']}")
    print(f"Action:      {result3['action']}")
    print(f"Path:        {result3['degree_audit']['degree_path']}")
    print(f"\n{'='*60}")
    print("GEMINI RECOMMENDATION:")
    print("="*60)
    print(result3["recommendation"])
