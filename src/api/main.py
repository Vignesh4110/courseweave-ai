from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional, List
import psycopg2
import psycopg2.extras
import jwt
import bcrypt
import os
import json
import logging
import traceback
from datetime import datetime, timedelta
from dotenv import load_dotenv
from src.models.retriever import get_relevant_courses

logging.getLogger("src.models.retriever").setLevel(logging.INFO)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="CourseWeave AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

JWT_SECRET = os.getenv("JWT_SECRET", "courseweave-secret-key-change-in-prod")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

security = HTTPBearer() 

PROGRAMS = ["MS_DAE", "MS_DS", "MS_CS", "MS_DA", "MS_IS"]
CAREERS = ["Data Engineer", "Data Scientist", "Data Analyst", "Business Analyst", "Software Engineer", "ML Engineer"]


from psycopg2 import pool as pg_pool

_db_pool = None

def _get_pool():
    global _db_pool
    if _db_pool is None:
        _db_pool = pg_pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            host=os.getenv("DB_HOST", "34.23.27.68"),
            port=int(os.getenv("DB_PORT", 5432)),
            dbname=os.getenv("DB_NAME", "courseweave"),
            user=os.getenv("DB_USER", "courseweave_user"),
            password=os.getenv("DB_PASSWORD", ""),
        )
    return _db_pool

def get_db():
    conn = _get_pool().getconn()
    conn.autocommit = True
    return conn

def release_db(conn):
    _get_pool().putconn(conn)


def create_token(student_id: int, email: str) -> str:
    payload = {
        "sub": str(student_id),
        "email": email,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return {"student_id": int(payload["sub"]), "email": payload["email"]}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ── Pydantic models ──────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    program_code: str
    target_career: str
    degree_path: Optional[str] = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class RecommendRequest(BaseModel):
    career_goal: Optional[str] = None
    degree_path: Optional[str] = None
    conversation_id: Optional[int] = None
    user_message: Optional[str] = None

class AddCourseRequest(BaseModel):
    course_code: str
    grade: Optional[str] = None
    completed_at: Optional[str] = None

class StudentProfileRequest(BaseModel):
    intake_month: Optional[str] = None
    intake_year: Optional[int] = None
    grad_month: Optional[str] = None
    grad_year: Optional[int] = None
    current_term: Optional[str] = None
    planning_semester: Optional[str] = None
    manual_gpa: Optional[float] = None

class BatchCourseItem(BaseModel):
    course_code: str
    grade: str
    semester: str
    course_name: Optional[str] = None
    credits: Optional[int] = 4

class BatchCoursesRequest(BaseModel):
    courses: List[BatchCourseItem]


# ── DB migration on startup ──────────────────────────────────────────────────

@app.on_event("startup")
def run_migrations():
    try:
        conn = get_db()
        cur = conn.cursor()
        for sql in [
            "ALTER TABLE students ADD COLUMN IF NOT EXISTS intake_month VARCHAR(5)",
            "ALTER TABLE students ADD COLUMN IF NOT EXISTS intake_year INT",
            "ALTER TABLE students ADD COLUMN IF NOT EXISTS grad_month VARCHAR(10)",
            "ALTER TABLE students ADD COLUMN IF NOT EXISTS grad_year INT",
            "ALTER TABLE students ADD COLUMN IF NOT EXISTS current_term VARCHAR(20)",
            "ALTER TABLE students ADD COLUMN IF NOT EXISTS planning_semester VARCHAR(20)",
            "ALTER TABLE students ADD COLUMN IF NOT EXISTS manual_gpa FLOAT",
            "ALTER TABLE student_courses ADD COLUMN IF NOT EXISTS semester VARCHAR(20)",
        ]:
            cur.execute(sql)
        conn.commit()
        release_db(conn)
        logger.info("DB migrations complete")
    except Exception as e:
        logger.error("Migration error: %s", e)


# ── Auth endpoints ───────────────────────────────────────────────────────────

@app.post("/auth/signup")
def signup(req: SignupRequest, background_tasks: BackgroundTasks):
    if req.program_code not in PROGRAMS:
        raise HTTPException(400, detail=f"Invalid program. Choose from {PROGRAMS}")
    if req.target_career not in CAREERS:
        raise HTTPException(400, detail=f"Please select a career from the supported options: {CAREERS}")

    try:
        from src.api.students import create_student, warm_up_recommendation
        student = create_student(
            name=req.name,
            email=req.email,
            password=req.password,
            program_code=req.program_code,
            target_career=req.target_career,
            degree_path=req.degree_path or None,
        )
        token = create_token(student["id"], student["email"])
        # Trigger AI pipeline in the background — signup never blocks on Gemini/Pinecone
        background_tasks.add_task(warm_up_recommendation, student["id"])
        return {
            "token": token,
            "student": student,
            "initial_recommendation": None,
        }
    except psycopg2.errors.UniqueViolation:
        raise HTTPException(409, detail="Email already registered")
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.post("/auth/login")
def login(req: LoginRequest):
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM students WHERE email = %s", (req.email,))
        row = cur.fetchone()
        release_db(conn)
        if not row:
            raise HTTPException(401, detail="Invalid credentials")
        student = dict(row)
        pw_hash = student.get("password_hash", "")
        if not pw_hash or not bcrypt.checkpw(req.password.encode(), pw_hash.encode()):
            raise HTTPException(401, detail="Invalid credentials")
        token = create_token(student["id"], student["email"])
        student.pop("password_hash", None)
        return {"token": token, "student": student}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/auth/me")
def me(user=Depends(verify_token)):
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT id, name, email, program_code, target_career, degree_path, created_at FROM students WHERE id = %s",
            (user["student_id"],),
        )
        student = dict(cur.fetchone())
        release_db(conn)
        return student
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ── Student dashboard ────────────────────────────────────────────────────────

@app.get("/student/dashboard")
def dashboard(user=Depends(verify_token)):
    sid = user["student_id"]
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute(
            """SELECT s.id, s.name, s.email, s.program_code, s.target_career,
                      s.intake_month, s.intake_year, s.grad_month, s.grad_year,
                      s.current_term, s.planning_semester, s.manual_gpa
               FROM students s WHERE s.id = %s""",
            (sid,),
        )
        student = dict(cur.fetchone())

        cur.execute(
            """SELECT sc.course_code, c.course_name, c.credits, c.course_type,
                      sc.grade, sc.completed_at, sc.semester
               FROM student_courses sc
               JOIN courses c ON c.course_code = sc.course_code
               WHERE sc.student_id = %s
               ORDER BY sc.completed_at DESC""",
            (sid,),
        )
        completed = [dict(r) for r in cur.fetchall()]

        cur.execute(
            """SELECT c.course_code, c.course_name, c.credits, c.course_type, c.program_code
               FROM courses c
               WHERE c.program_code = %s AND c.is_active = TRUE
               AND c.course_code NOT IN (
                   SELECT course_code FROM student_courses WHERE student_id = %s
               )""",
            (student["program_code"], sid),
        )
        remaining = [dict(r) for r in cur.fetchall()]

        prog_map = {
            "MS_DAE": 32, "MS_DS": 32, "MS_CS": 32, "MS_DA": 32, "MS_IS": 32
        }
        total_required = prog_map.get(student["program_code"], 32)
        credits_done = sum(c["credits"] for c in completed)
        credits_remaining = total_required - credits_done

        core_done = sum(1 for c in completed if c["course_type"] == "Core")
        elective_done = sum(1 for c in completed if c["course_type"] == "Elective")

        gpa_map = {"A": 4.0, "A-": 3.7, "B+": 3.3, "B": 3.0, "B-": 2.7, "C+": 2.3, "C": 2.0}
        grades = [gpa_map.get(c["grade"], 0) for c in completed if c["grade"]]
        gpa = round(sum(grades) / len(grades), 2) if grades else 0.0

        release_db(conn)
        return {
            "student": student,
            "stats": {
                "credits_completed": credits_done,
                "credits_remaining": credits_remaining,
                "total_required": total_required,
                "progress_pct": round((credits_done / total_required) * 100),
                "courses_completed": len(completed),
                "core_completed": core_done,
                "electives_completed": elective_done,
                "gpa": gpa,
            },
            "completed_courses": completed,
            "remaining_courses": remaining,
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ── Courses catalog ──────────────────────────────────────────────────────────

@app.get("/courses")
def get_courses(program: Optional[str] = None, course_type: Optional[str] = None, user=Depends(verify_token)):
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        query = "SELECT * FROM courses WHERE is_active = TRUE"
        params = []
        if program:
            query += " AND program_code = %s"
            params.append(program)
        if course_type:
            query += " AND course_type = %s"
            params.append(course_type)
        query += " ORDER BY program_code, course_type, course_code"
        cur.execute(query, params)
        courses = [dict(r) for r in cur.fetchall()]
        release_db(conn)
        return courses
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/courses/{course_code}/prerequisites")
def get_prerequisites(course_code: str, user=Depends(verify_token)):
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """SELECT p.required_course_code, c.course_name, c.credits
               FROM prerequisites p
               JOIN courses c ON c.course_code = p.required_course_code
               WHERE p.course_code = %s""",
            (course_code,),
        )
        prereqs = [dict(r) for r in cur.fetchall()]
        release_db(conn)
        return prereqs
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ── Student courses ──────────────────────────────────────────────────────────

@app.get("/student/courses")
def student_courses(user=Depends(verify_token)):
    sid = user["student_id"]
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """SELECT sc.id, sc.course_code, c.course_name, c.credits, c.course_type,
                      sc.grade, sc.completed_at
               FROM student_courses sc
               JOIN courses c ON c.course_code = sc.course_code
               WHERE sc.student_id = %s
               ORDER BY sc.completed_at DESC""",
            (sid,),
        )
        courses = [dict(r) for r in cur.fetchall()]
        release_db(conn)
        return courses
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.delete("/student/courses/{course_code}")
def remove_course(course_code: str, user=Depends(verify_token)):
    sid = user["student_id"]
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM student_courses WHERE student_id = %s AND course_code = %s",
            (sid, course_code)
        )
        cur.execute(
            "DELETE FROM student_courses_roadmap_temp_addition WHERE student_id = %s AND course_code = %s",
            (sid, course_code)
        )
        release_db(conn)
        return {"deleted": True}
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.post("/student/courses")
def add_course(req: AddCourseRequest, user=Depends(verify_token)):
    sid = user["student_id"]
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        if req.completed_at:
            # Course has a completion date → goes into student_courses (affects recommendations)
            cur.execute(
                "INSERT INTO student_courses (student_id, course_code, grade, completed_at) VALUES (%s,%s,%s,%s) RETURNING *",
                (sid, req.course_code, req.grade or None, req.completed_at),
            )
            # Remove from temp table if it was planned there
            cur.execute(
                "DELETE FROM student_courses_roadmap_temp_addition WHERE student_id = %s AND course_code = %s",
                (sid, req.course_code),
            )
        else:
            # No date → roadmap planning only, isolated from recommendation engine
            cur.execute(
                "INSERT INTO student_courses_roadmap_temp_addition (student_id, course_code) VALUES (%s,%s) RETURNING *",
                (sid, req.course_code),
            )

        row = dict(cur.fetchone())
        release_db(conn)
        return row
    except psycopg2.errors.UniqueViolation:
        raise HTTPException(409, detail="Course already added")
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ── Student profile update ───────────────────────────────────────────────────

@app.put("/student/profile")
def update_profile(req: StudentProfileRequest, user=Depends(verify_token)):
    sid = user["student_id"]
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            """UPDATE students SET
               intake_month = COALESCE(%s, intake_month),
               intake_year = COALESCE(%s, intake_year),
               grad_month = COALESCE(%s, grad_month),
               grad_year = COALESCE(%s, grad_year),
               current_term = COALESCE(%s, current_term),
               planning_semester = COALESCE(%s, planning_semester),
               manual_gpa = COALESCE(%s, manual_gpa)
               WHERE id = %s""",
            (req.intake_month, req.intake_year, req.grad_month, req.grad_year,
             req.current_term, req.planning_semester, req.manual_gpa, sid),
        )
        conn.commit()
        release_db(conn)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/student/profile")
def get_profile(user=Depends(verify_token)):
    sid = user["student_id"]
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """SELECT intake_month, intake_year, grad_month, grad_year,
                      current_term, planning_semester, manual_gpa
               FROM students WHERE id = %s""",
            (sid,),
        )
        row = cur.fetchone()
        release_db(conn)
        return dict(row) if row else {}
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ── Batch course add ─────────────────────────────────────────────────────────

def _semester_to_date(semester: str) -> str:
    parts = semester.split(" ")
    sem_type, year = parts[0], parts[1] if len(parts) > 1 else "2024"
    if sem_type == "Spring":
        return f"{year}-05-15"
    if sem_type == "Summer":
        return f"{year}-08-15"
    return f"{year}-12-15"

@app.post("/student/courses/batch")
def add_courses_batch(req: BatchCoursesRequest, user=Depends(verify_token)):
    sid = user["student_id"]
    added, skipped = 0, 0
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        for item in req.courses:
            # Ensure course exists in courses table (for manual entries)
            cur.execute("SELECT course_code FROM courses WHERE course_code = %s", (item.course_code,))
            if not cur.fetchone():
                if not item.course_name:
                    skipped += 1
                    continue
                cur.execute(
                    "INSERT INTO courses (course_code, course_name, credits, program_code, course_type) "
                    "SELECT %s, %s, %s, program_code, 'Elective' FROM students WHERE id = %s "
                    "ON CONFLICT (course_code) DO NOTHING",
                    (item.course_code, item.course_name, item.credits or 4, sid),
                )

            completed_at = _semester_to_date(item.semester)
            cur.execute(
                """INSERT INTO student_courses (student_id, course_code, grade, completed_at, semester)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (student_id, course_code) DO UPDATE
                   SET grade = EXCLUDED.grade, semester = EXCLUDED.semester""",
                (sid, item.course_code, item.grade, completed_at, item.semester),
            )
            added += 1

        conn.commit()
        release_db(conn)
        return {"added": added, "skipped": skipped}
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ── Prerequisites checker ────────────────────────────────────────────────────

@app.get("/student/prerequisites")
def check_prerequisites(user=Depends(verify_token)):
    sid = user["student_id"]
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("SELECT course_code FROM student_courses WHERE student_id = %s", (sid,))
        completed_codes = {r["course_code"] for r in cur.fetchall()}

        cur.execute("SELECT id, course_code FROM students WHERE id = %s", (sid,))
        cur.fetchone()

        cur.execute(
            "SELECT course_code, course_name FROM courses WHERE program_code = (SELECT program_code FROM students WHERE id = %s) AND is_active = TRUE",
            (sid,),
        )
        all_courses = [dict(r) for r in cur.fetchall()]

        result = []
        for course in all_courses:
            code = course["course_code"]
            cur.execute(
                """SELECT p.required_course_code, c.course_name
                   FROM prerequisites p JOIN courses c ON c.course_code = p.required_course_code
                   WHERE p.course_code = %s""",
                (code,),
            )
            prereqs = [dict(r) for r in cur.fetchall()]
            if not prereqs:
                continue
            missing = [p for p in prereqs if p["required_course_code"] not in completed_codes]
            result.append({
                "course_code": code,
                "course_name": course["course_name"],
                "prerequisites": prereqs,
                "missing_prerequisites": missing,
                "eligible": len(missing) == 0,
                "completed": code in completed_codes,
            })

        release_db(conn)
        return result
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ── Course details ────────────────────────────────────────────────────────────

@app.get("/courses/{course_code}/details")
def get_course_details(course_code: str, user=Depends(verify_token)):
    """Get detailed information about a specific course including syllabus from Pinecone."""
    try:
        from src.models.retriever import fetch_parent_chunk
        from groq import Groq
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Get course basic info from DB
        cur.execute(
            """SELECT course_code, course_name, credits, course_type, description,
                      learning_outcomes, prerequisites
               FROM courses WHERE course_code = %s""",
            (course_code,),
        )
        course = cur.fetchone()

        if not course:
            raise HTTPException(404, detail="Course not found")

        course_dict = dict(course)

        # Fetch syllabus content from Pinecone
        syllabus_text = fetch_parent_chunk(course_code)

        # Generate AI summary using Gemini
        if syllabus_text:
            prompt = f"""Based on this course syllabus for {course_code} - {course_dict['course_name']},
provide a concise, well-formatted summary covering:

1. Course Overview (2-3 sentences about what this course teaches)
2. Key Topics Covered (bullet points)
3. Learning Outcomes (what students will be able to do)
4. Prerequisites (if any are mentioned)

Syllabus content:
{syllabus_text[:3000]}

Format your response with clear headings and bullet points for easy reading."""

            try:
                response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3,
                )   
                ai_summary = response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"Gemini generation failed: {e}")
                ai_summary = "AI summary currently unavailable."
        else:
            ai_summary = "Detailed syllabus information not available for this course."
            syllabus_text = ""

        release_db(conn)

        return {
            **course_dict,
            "syllabus_text": syllabus_text[:2000] if syllabus_text else "",
            "ai_summary": ai_summary
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching course details: {e}")
        raise HTTPException(500, detail=str(e))


# ── Conversations ─────────────────────────────────────────────────────────────

@app.get("/conversations")
def list_conversations(user=Depends(verify_token)):
    sid = user["student_id"]
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT c.id, c.title, c.updated_at,
                   COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN conversation_messages m ON m.conversation_id = c.id
            WHERE c.student_id = %s
            GROUP BY c.id, c.title, c.updated_at
            ORDER BY c.updated_at DESC
        """, (sid,))
        convs = [dict(r) for r in cur.fetchall()]
        release_db(conn)
        return convs
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/conversations/{conv_id}")
def get_conversation(conv_id: int, user=Depends(verify_token)):
    sid = user["student_id"]
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT id, title, session_context FROM conversations WHERE id = %s AND student_id = %s",
            (conv_id, sid)
        )
        conv = cur.fetchone()
        if not conv:
            raise HTTPException(404, detail="Conversation not found")
        cur.execute("""
            SELECT role, text, courses, action
            FROM conversation_messages
            WHERE conversation_id = %s
            ORDER BY created_at ASC
        """, (conv_id,))
        messages = [dict(r) for r in cur.fetchall()]
        release_db(conn)
        return {**dict(conv), "messages": messages}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: int, user=Depends(verify_token)):
    sid = user["student_id"]
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("DELETE FROM conversations WHERE id = %s AND student_id = %s", (conv_id, sid))
        release_db(conn)
        return {"deleted": True}
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ── Recommendations ──────────────────────────────────────────────────────────

@app.post("/recommend")
def recommend(req: RecommendRequest, user=Depends(verify_token)):
    sid = user["student_id"]
    try:
        from src.agents.recommendation_agent import generate_recommendation, generate_followup

        conn = get_db()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        conv_id      = req.conversation_id
        user_message = req.user_message or ""

        if conv_id and not req.degree_path:
            # ── Follow-up: load history from DB, skip Pinecone ───────────────
            cur.execute(
                "SELECT session_context FROM conversations WHERE id = %s AND student_id = %s",
                (conv_id, sid)
            )
            conv = cur.fetchone()
            if not conv:
                raise HTTPException(404, detail="Conversation not found")

            # Save user message before loading history so it's included
            cur.execute(
                "INSERT INTO conversation_messages (conversation_id, role, text) VALUES (%s, %s, %s)",
                (conv_id, "user", user_message)
            )
            cur.execute(
                "SELECT role, text FROM conversation_messages WHERE conversation_id = %s ORDER BY created_at",
                (conv_id,)
            )
            history = [{"role": r["role"], "text": r["text"]} for r in cur.fetchall()]

            result = generate_followup(
                student_id=sid,
                session_context=conv["session_context"] or {},
                conversation_history=history,
            )
        else:
            # ── First turn or path selection: full RAG pipeline ──────────────
            result = generate_recommendation(
                student_id=sid,
                career_goal=req.career_goal or None,
                degree_path=req.degree_path or None,
            )

            if "error" not in result:
                session_ctx = None
                if result.get("action") == "recommend":
                    session_ctx = {
                        "courses":       result.get("courses", []),
                        "prereq_status": result.get("prereq_status", []),
                        "career_goal":   result.get("career_goal", ""),
                        "career_skills": result.get("career_skills", {}),
                    }

                if conv_id:
                    # Path selection on existing conversation — update session_context
                    if session_ctx:
                        cur.execute(
                            "UPDATE conversations SET session_context = %s, updated_at = NOW() WHERE id = %s",
                            (json.dumps(session_ctx), conv_id)
                        )
                    if user_message:
                        cur.execute(
                            "INSERT INTO conversation_messages (conversation_id, role, text) VALUES (%s, %s, %s)",
                            (conv_id, "user", user_message)
                        )
                else:
                    # Brand new conversation
                    title = (user_message[:50] + "…") if len(user_message) > 50 else user_message or f"{req.career_goal or 'Course'} recommendations"
                    cur.execute(
                        "INSERT INTO conversations (student_id, title, session_context) VALUES (%s, %s, %s) RETURNING id",
                        (sid, title, json.dumps(session_ctx) if session_ctx else None)
                    )
                    conv_id = cur.fetchone()["id"]
                    if user_message:
                        cur.execute(
                            "INSERT INTO conversation_messages (conversation_id, role, text) VALUES (%s, %s, %s)",
                            (conv_id, "user", user_message)
                        )

        if "error" in result:
            raise HTTPException(500, detail=result["error"])

        # Enrich courses with credits and course_type from DB
        if result.get("courses"):
            course_codes = [c["course_code"] for c in result["courses"]]
            if course_codes:
                placeholders = ",".join(["%s"] * len(course_codes))
                cur.execute(
                    f"SELECT course_code, credits, course_type FROM courses WHERE course_code IN ({placeholders})",
                    course_codes
                )
                course_info = {row["course_code"]: {"credits": row["credits"], "course_type": row["course_type"]} for row in cur.fetchall()}

                # Also check elective_courses for cross-dept courses not in courses table
                missing_codes = [c for c in course_codes if c not in course_info]
                if missing_codes:
                    placeholders2 = ",".join(["%s"] * len(missing_codes))
                    cur.execute(
                        f"SELECT course_code, credits, 'Elective' as course_type FROM elective_courses WHERE course_code IN ({placeholders2})",
                        missing_codes
                    )
                    for row in cur.fetchall():
                        course_info[row["course_code"]] = {"credits": row["credits"], "course_type": row["course_type"]}
                    logger.info("Enriched %d cross-dept courses from elective_courses", len(missing_codes))

                for course in result["courses"]:
                    info = course_info.get(course["course_code"], {})
                    course["credits"] = info.get("credits", 4)
                    course["course_type"] = info.get("course_type", "Elective")

        # Save bot response
        if conv_id:
            cur.execute(
                "INSERT INTO conversation_messages (conversation_id, role, text, courses, action) VALUES (%s, %s, %s, %s, %s)",
                (conv_id, "model", result["recommendation"], json.dumps(result.get("courses", [])), result.get("action"))
            )
            cur.execute("UPDATE conversations SET updated_at = NOW() WHERE id = %s", (conv_id,))

        conn.commit()
        release_db(conn)
        result["conversation_id"] = conv_id
        return result

    except ImportError:
        logger.error("RAG import failed — falling back: %s", traceback.format_exc())
        return _fallback_recommend(sid)
    except HTTPException:
        raise
    except Exception:
        logger.error("RAG pipeline error — falling back: %s", traceback.format_exc())
        return _fallback_recommend(sid)


def _fallback_recommend(student_id: int):
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """SELECT c.course_code, c.course_name, c.credits, c.course_type
               FROM courses c
               WHERE c.program_code = (SELECT program_code FROM students WHERE id = %s)
               AND c.is_active = TRUE
               AND c.course_code NOT IN (SELECT course_code FROM student_courses WHERE student_id = %s)
               LIMIT 5""",
            (student_id, student_id),
        )
        courses = [dict(r) for r in cur.fetchall()]
        release_db(conn)
        return {
            "recommendations": [
                {**c, "reason": "Recommended based on your program and remaining requirements"}
                for c in courses
            ],
            "source": "fallback",
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ── Roadmap ──────────────────────────────────────────────────────────────────

@app.get("/student/roadmap")
def roadmap(user=Depends(verify_token)):
    sid = user["student_id"]
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Student profile
        cur.execute("SELECT program_code, degree_path FROM students WHERE id = %s", (sid,))
        student = dict(cur.fetchone())
        program_code = student["program_code"]
        degree_path  = student.get("degree_path")

        # Program requirements
        cur.execute("SELECT * FROM program_requirements WHERE program_code = %s", (program_code,))
        req = dict(cur.fetchone())
        total_credits         = req["total_credits"]
        core_credits_required = req["core_credits"]

        # Elective credits depend on chosen degree path
        if degree_path == "project":
            elective_credits_required = req["project_elective_credits"]
        elif degree_path == "thesis":
            elective_credits_required = req["thesis_elective_credits"]
        else:
            elective_credits_required = req["elective_credits"]

        # Completed courses — from student_courses (drives recommendation engine)
        cur.execute(
            """SELECT sc.course_code, c.course_name, c.credits, c.course_type,
                      sc.grade, sc.completed_at
               FROM student_courses sc JOIN courses c ON c.course_code = sc.course_code
               WHERE sc.student_id = %s ORDER BY sc.completed_at""",
            (sid,),
        )
        completed = [dict(r) for r in cur.fetchall()]

        # Remove temp entries that now exist in student_courses (e.g. inserted by pipeline)
        cur.execute(
            """DELETE FROM student_courses_roadmap_temp_addition
               WHERE student_id = %s
               AND course_code IN (
                   SELECT course_code FROM student_courses WHERE student_id = %s
               )""",
            (sid, sid),
        )

        # Planned courses — from isolated temp table, invisible to recommendation engine
        cur.execute(
            """SELECT t.course_code, c.course_name, c.credits, c.course_type,
                      NULL AS grade, NULL AS completed_at
               FROM student_courses_roadmap_temp_addition t
               JOIN courses c ON c.course_code = t.course_code
               WHERE t.student_id = %s ORDER BY t.created_at""",
            (sid,),
        )
        student_planned = [dict(r) for r in cur.fetchall()]

        completed_core_credits     = sum(c["credits"] for c in completed if c["course_type"] == "Core")
        completed_elective_credits = sum(c["credits"] for c in completed if c["course_type"] == "Elective")
        remaining_core_credits     = max(0, core_credits_required - completed_core_credits)
        remaining_elective_credits = max(0, elective_credits_required - completed_elective_credits)

        # All catalog courses not yet completed
        cur.execute(
            """SELECT c.course_code, c.course_name, c.credits, c.course_type
               FROM courses c
               WHERE c.program_code = %s AND c.is_active = TRUE
               AND c.course_code NOT IN (
                   SELECT course_code FROM student_courses WHERE student_id = %s
               )
               ORDER BY c.course_type, c.course_code""",
            (program_code, sid),
        )
        all_remaining = [dict(r) for r in cur.fetchall()]
        release_db(conn)

        # Only include courses needed to satisfy remaining degree requirements
        planned = []
        added_core = added_elective = 0

        for c in all_remaining:
            if c["course_type"] == "Core" and added_core < remaining_core_credits:
                planned.append(c)
                added_core += c["credits"]

        for c in all_remaining:
            if c["course_type"] == "Elective" and added_elective < remaining_elective_credits:
                planned.append(c)
                added_elective += c["credits"]

        # Build semester blocks from completed courses (grouped by completion date)
        semesters = []
        if completed:
            dates = {}
            for c in completed:
                key = str(c["completed_at"])
                dates.setdefault(key, []).append(c)
            for i, (_, courses) in enumerate(sorted(dates.items()), 1):
                semesters.append({"label": f"Semester {i}", "status": "completed", "courses": courses})

        # Current semester: student-added courses without a completion date
        sem_num = len(semesters) + 1
        semesters.append({
            "label":   f"Semester {sem_num}",
            "status":  "current",
            "courses": student_planned,   # may be empty — frontend shows empty + slots
        })

        # Future planned semesters (empty slots for remaining requirements)
        chunk = 3
        remaining_slots = len(planned)   # catalog courses still needed
        for i in range(0, remaining_slots, chunk):
            sem_num = len(semesters) + 1
            semesters.append({
                "label":   f"Semester {sem_num}",
                "status":  "planned",
                "courses": planned[i:i + chunk],
            })

        credits_completed    = sum(c["credits"] for c in completed)
        credits_in_progress  = sum(c["credits"] for c in student_planned)
        credits_total_used   = credits_completed + credits_in_progress
        credits_remaining    = max(0, total_credits - credits_total_used)

        return {
            "semesters": semesters,
            "summary": {
                "total_credits":              total_credits,
                "credits_completed":          credits_completed,
                "credits_in_progress":        credits_in_progress,
                "credits_total_used":         credits_total_used,
                "credits_remaining":          credits_remaining,
                "core_credits_required":      core_credits_required,
                "core_credits_completed":     completed_core_credits,
                "core_credits_remaining":     remaining_core_credits,
                "elective_credits_required":  elective_credits_required,
                "elective_credits_completed": completed_elective_credits,
                "elective_credits_remaining": remaining_elective_credits,
            },
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "service": "CourseWeave AI API"}