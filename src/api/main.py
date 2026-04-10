"""
main.py
-------
FastAPI application for CourseWeave AI.

Endpoints:
    POST /register        → register student + get recommendation
    POST /recommend       → get recommendation for existing student
    GET  /courses         → get courses for a program (for frontend dropdown)
    GET  /programs        → get all programs (for frontend dropdown)
    GET  /student/{id}    → get student profile + degree audit
    GET  /health          → health check

Run locally:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

Run on GCP VM:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.api.students import (
    register_student,
    get_courses_for_program,
    get_all_programs,
)
from src.agents.recommendation_agent import generate_recommendation
from src.models.postgres_filter import get_student_context, get_degree_audit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CourseWeave AI",
    description="AI-powered course recommendation system for Northeastern University",
    version="1.0.0"
)

# ── CORS — allows React/Next.js frontend to call this API ─────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────

class CompletedCourse(BaseModel):
    course_code:  str
    completed_at: Optional[str] = None   # defaults to today if not provided
    grade:        Optional[str] = None


class RegisterRequest(BaseModel):
    name:              str
    email:             str
    program_code:      str
    target_career:     str
    password:          str
    completed_courses: Optional[list[CompletedCourse]] = []


class RecommendRequest(BaseModel):
    student_id:  int
    degree_path: Optional[str] = None   # coursework, project, thesis


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Simple health check — confirms API is running."""
    return {"status": "ok", "service": "CourseWeave AI"}


@app.get("/programs")
def get_programs():
    """
    Returns all available programs.
    Used by frontend to populate the program dropdown.
    """
    try:
        programs = get_all_programs()
        return {"programs": programs}
    except Exception as e:
        logger.error("Failed to fetch programs: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/courses")
def get_courses(program_code: str):
    """
    Returns all active courses for a program.
    Used by frontend to populate the completed courses multi-select.

    Example: GET /courses?program_code=MS_DAE
    """
    if not program_code:
        raise HTTPException(status_code=400, detail="program_code is required")

    try:
        courses = get_courses_for_program(program_code)
        return {"program_code": program_code, "courses": courses}
    except Exception as e:
        logger.error("Failed to fetch courses for %s: %s", program_code, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/register")
def register(data: RegisterRequest):
    """
    Register a new student and return their first recommendation.

    If the email already exists, returns recommendation for existing student.

    Request body:
    {
        "name": "Aisha Patel",
        "email": "patel.ai@northeastern.edu",
        "program_code": "MS_DAE",
        "target_career": "Data Engineer",
        "password": "somepassword123",
        "completed_courses": [
            {"course_code": "IE6400", "completed_at": "2025-05-15", "grade": "A"},
            {"course_code": "IE6700", "grade": "A-"}
        ]
    }
    """
    try:
        result = register_student(
            name              = data.name,
            email             = data.email,
            program_code      = data.program_code,
            target_career     = data.target_career,
            password          = data.password,
            completed_courses = [c.dict() for c in data.completed_courses]
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Registration failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend")
def recommend(data: RecommendRequest):
    """
    Get a fresh recommendation for an existing student.
    Also used when student selects their degree path.

    Request body:
    {
        "student_id": 101,
        "degree_path": "coursework"   // optional
    }
    """
    try:
        result = generate_recommendation(
            student_id  = data.student_id,
            degree_path = data.degree_path
        )

        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Recommendation failed for student %d: %s", data.student_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/student/{student_id}")
def get_student(student_id: int):
    """
    Returns student profile + degree audit for an existing student.
    Used by frontend to show student dashboard.
    """
    try:
        context = get_student_context(student_id)
        if not context:
            raise HTTPException(status_code=404, detail=f"Student {student_id} not found")

        audit = get_degree_audit(student_id)

        return {
            "student":      context,
            "degree_audit": audit
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to fetch student %d: %s", student_id, e)
        raise HTTPException(status_code=500, detail=str(e))