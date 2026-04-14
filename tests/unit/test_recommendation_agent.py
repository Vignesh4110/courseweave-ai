import sys
import pytest
from unittest.mock import MagicMock

# Mock heavy external modules BEFORE any project imports
sys.modules["src.models.retriever"] = MagicMock()
sys.modules["google"] = MagicMock()
sys.modules["google.genai"] = MagicMock()

from src.agents.recommendation_agent import (  # noqa: E402
    format_courses_for_prompt,
    build_recommendation_prompt,
    build_path_selection_prompt,
    gemini_generate,
    generate_recommendation,
)

# ── Shared test data ──────────────────────────────────────────────────────────

SAMPLE_STUDENT = {
    "student_id": 1,
    "name": "Aisha Patel",
    "email": "patel.ai@northeastern.edu",
    "program_code": "MS_DAE",
    "target_career": "Data Engineer",
    "completed_courses": ["IE6400", "IE6700"],
    "eligible_courses": ["IE7275", "IE6600", "IE7615"],
    "core_remaining": ["IE7275", "IE6600"],
    "electives_available": ["IE7615"],
    "prereq_map": {"IE7615": ["IE7275"]},
}

SAMPLE_AUDIT = {
    "student_id": 1,
    "name": "Aisha Patel",
    "program_code": "MS_DAE",
    "program_name": "MS Data Analytics Engineering",
    "degree_path": "coursework",
    "target_career": "Data Engineer",
    "total_credits": 32,
    "credits_completed": 8,
    "credits_remaining": 24,
    "core_credits_required": 20,
    "core_credits_completed": 8,
    "core_credits_remaining": 12,
    "core_courses_remaining": ["IE7275", "IE6600"],
    "elective_credits_required": 12,
    "elective_credits_completed": 0,
    "elective_credits_remaining": 12,
    "electives_completed": [],
    "project_available": True,
    "project_credits": 4,
    "needs_path_selection": False,
    "on_track": True,
    "next_action": "take_core",
}

SAMPLE_COURSES = [
    {
        "course_code": "IE7275",
        "course_name": "Data Mining",
        "text": "Learn data mining and machine learning techniques for large datasets",
        "score": 0.92,
    },
    {
        "course_code": "IE6600",
        "course_name": "Computation and Algorithms",
        "text": "Algorithms, data structures, and computational thinking for engineers",
        "score": 0.85,
    },
]

SAMPLE_PREREQ_STATUS = [
    {"course_code": "IE7275", "prereqs_satisfied": True, "missing_prereqs": []},
    {"course_code": "IE6600", "prereqs_satisfied": True, "missing_prereqs": []},
]

SAMPLE_CAREER_SKILLS = {
    "core_skills": ["Python", "SQL", "ETL"],
    "tools": ["Airflow", "Spark"],
    "nice_to_have": ["Terraform"],
}

SAMPLE_QUERY_RESULT = {
    "skill_query": "Python SQL ETL data pipelines data warehousing",
    "career_skills": SAMPLE_CAREER_SKILLS,
}


# ── format_courses_for_prompt ─────────────────────────────────────────────────

def test_format_courses_basic():
    result = format_courses_for_prompt(
        [{"course_code": "IE6400", "course_name": "Data Mining",
          "text": "Learn data mining techniques", "score": 0.9}],
        [{"course_code": "IE6400", "prereqs_satisfied": True, "missing_prereqs": []}],
    )
    assert "IE6400" in result
    assert "Data Mining" in result


def test_format_courses_prereq_satisfied_label():
    result = format_courses_for_prompt(
        [{"course_code": "IE7275", "course_name": "Data Mining",
          "text": "desc", "score": 0.9}],
        [{"course_code": "IE7275", "prereqs_satisfied": True, "missing_prereqs": []}],
    )
    assert "Prerequisites satisfied" in result


def test_format_courses_missing_prereq_warning():
    result = format_courses_for_prompt(
        [{"course_code": "IE7615", "course_name": "Advanced ML",
          "text": "Advanced machine learning", "score": 0.8}],
        [{"course_code": "IE7615", "prereqs_satisfied": False,
          "missing_prereqs": ["IE6400"]}],
    )
    assert "IE6400" in result
    assert "Requires completing first" in result


def test_format_courses_multiple_numbered():
    result = format_courses_for_prompt(SAMPLE_COURSES, SAMPLE_PREREQ_STATUS)
    assert "1." in result
    assert "2." in result
    assert "IE7275" in result
    assert "IE6600" in result


def test_format_courses_empty_returns_empty_string():
    result = format_courses_for_prompt([], [])
    assert result == ""


def test_format_courses_text_truncated_at_300():
    long_text = "x" * 500
    result = format_courses_for_prompt(
        [{"course_code": "IE6400", "course_name": "Test",
          "text": long_text, "score": 0.5}],
        [{"course_code": "IE6400", "prereqs_satisfied": True, "missing_prereqs": []}],
    )
    assert "x" * 300 in result
    assert "x" * 301 not in result


def test_format_courses_score_shown():
    result = format_courses_for_prompt(
        [{"course_code": "IE6400", "course_name": "Data Mining",
          "text": "desc", "score": 0.91}],
        [{"course_code": "IE6400", "prereqs_satisfied": True, "missing_prereqs": []}],
    )
    assert "0.91" in result


def test_format_courses_course_not_in_prereq_map_defaults_satisfied():
    result = format_courses_for_prompt(
        [{"course_code": "IE9999", "course_name": "Unknown",
          "text": "desc", "score": 0.5}],
        [],
    )
    assert "Prerequisites satisfied" in result


# ── build_recommendation_prompt ───────────────────────────────────────────────

def test_build_recommendation_prompt_contains_student_name():
    result = build_recommendation_prompt(
        SAMPLE_STUDENT, SAMPLE_AUDIT, SAMPLE_COURSES,
        SAMPLE_PREREQ_STATUS, "Data Engineer", SAMPLE_CAREER_SKILLS,
    )
    assert "Aisha Patel" in result


def test_build_recommendation_prompt_contains_career_goal():
    result = build_recommendation_prompt(
        SAMPLE_STUDENT, SAMPLE_AUDIT, SAMPLE_COURSES,
        SAMPLE_PREREQ_STATUS, "Data Engineer", SAMPLE_CAREER_SKILLS,
    )
    assert "Data Engineer" in result


def test_build_recommendation_prompt_contains_credit_counts():
    result = build_recommendation_prompt(
        SAMPLE_STUDENT, SAMPLE_AUDIT, SAMPLE_COURSES,
        SAMPLE_PREREQ_STATUS, "Data Engineer", SAMPLE_CAREER_SKILLS,
    )
    assert "8" in result
    assert "32" in result


def test_build_recommendation_prompt_contains_course_codes():
    result = build_recommendation_prompt(
        SAMPLE_STUDENT, SAMPLE_AUDIT, SAMPLE_COURSES,
        SAMPLE_PREREQ_STATUS, "Data Engineer", SAMPLE_CAREER_SKILLS,
    )
    assert "IE7275" in result
    assert "IE6600" in result


def test_build_recommendation_prompt_all_core_completed_message():
    audit = {**SAMPLE_AUDIT, "core_courses_remaining": []}
    result = build_recommendation_prompt(
        SAMPLE_STUDENT, audit, SAMPLE_COURSES,
        SAMPLE_PREREQ_STATUS, "Data Engineer", SAMPLE_CAREER_SKILLS,
    )
    assert "All core completed" in result


def test_build_recommendation_prompt_undecided_path_label():
    audit = {**SAMPLE_AUDIT, "degree_path": "undecided"}
    result = build_recommendation_prompt(
        SAMPLE_STUDENT, audit, SAMPLE_COURSES,
        SAMPLE_PREREQ_STATUS, "Data Engineer", SAMPLE_CAREER_SKILLS,
    )
    assert "Not yet selected" in result


def test_build_recommendation_prompt_returns_string():
    result = build_recommendation_prompt(
        SAMPLE_STUDENT, SAMPLE_AUDIT, SAMPLE_COURSES,
        SAMPLE_PREREQ_STATUS, "Data Engineer", SAMPLE_CAREER_SKILLS,
    )
    assert isinstance(result, str)
    assert len(result) > 0


# ── build_path_selection_prompt ───────────────────────────────────────────────

def test_build_path_selection_prompt_contains_student_name():
    result = build_path_selection_prompt(SAMPLE_STUDENT, SAMPLE_AUDIT, "Data Engineer")
    assert "Aisha Patel" in result


def test_build_path_selection_prompt_contains_program_name():
    result = build_path_selection_prompt(SAMPLE_STUDENT, SAMPLE_AUDIT, "Data Engineer")
    assert "MS Data Analytics Engineering" in result


def test_build_path_selection_prompt_lists_all_path_options():
    result = build_path_selection_prompt(SAMPLE_STUDENT, SAMPLE_AUDIT, "Data Engineer")
    assert "Coursework" in result
    assert "Project" in result
    assert "Thesis" in result


def test_build_path_selection_prompt_contains_career_goal():
    result = build_path_selection_prompt(SAMPLE_STUDENT, SAMPLE_AUDIT, "Data Scientist")
    assert "Data Scientist" in result


def test_build_path_selection_prompt_returns_string():
    result = build_path_selection_prompt(SAMPLE_STUDENT, SAMPLE_AUDIT, "Data Engineer")
    assert isinstance(result, str)
    assert len(result) > 0


# ── gemini_generate ───────────────────────────────────────────────────────────

def test_gemini_generate_success(monkeypatch):
    mock_response = MagicMock()
    mock_response.text = "  Great recommendation!  "
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    import src.agents.recommendation_agent as ra
    monkeypatch.setattr(ra, "_gemini_client", mock_client)

    result = gemini_generate("test prompt")
    assert result == "Great recommendation!"
    mock_client.models.generate_content.assert_called_once()


def test_gemini_generate_retries_on_429(monkeypatch):
    mock_response = MagicMock()
    mock_response.text = "Success after retry"
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [
        Exception("429 RESOURCE_EXHAUSTED quota exceeded"),
        mock_response,
    ]

    import src.agents.recommendation_agent as ra
    monkeypatch.setattr(ra, "_gemini_client", mock_client)
    monkeypatch.setattr(ra, "time", MagicMock())

    result = gemini_generate("test prompt", max_retries=4)
    assert result == "Success after retry"
    assert mock_client.models.generate_content.call_count == 2


def test_gemini_generate_retries_on_resource_exhausted(monkeypatch):
    mock_response = MagicMock()
    mock_response.text = "Eventually works"
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [
        Exception("RESOURCE_EXHAUSTED: quota limit"),
        Exception("RESOURCE_EXHAUSTED: quota limit"),
        mock_response,
    ]

    import src.agents.recommendation_agent as ra
    monkeypatch.setattr(ra, "_gemini_client", mock_client)
    monkeypatch.setattr(ra, "time", MagicMock())

    result = gemini_generate("test prompt", max_retries=4)
    assert result == "Eventually works"
    assert mock_client.models.generate_content.call_count == 3


def test_gemini_generate_raises_after_all_retries_exhausted(monkeypatch):
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("429 quota exceeded")

    import src.agents.recommendation_agent as ra
    monkeypatch.setattr(ra, "_gemini_client", mock_client)
    monkeypatch.setattr(ra, "time", MagicMock())

    with pytest.raises(Exception, match="429"):
        gemini_generate("test prompt", max_retries=2)

    assert mock_client.models.generate_content.call_count == 2


def test_gemini_generate_non_rate_limit_raises_immediately(monkeypatch):
    """Non-rate-limit errors should not trigger retries."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = ValueError("Invalid model name")

    import src.agents.recommendation_agent as ra
    monkeypatch.setattr(ra, "_gemini_client", mock_client)

    with pytest.raises(ValueError):
        gemini_generate("test prompt", max_retries=4)

    assert mock_client.models.generate_content.call_count == 1


# ── generate_recommendation ───────────────────────────────────────────────────

def test_generate_recommendation_student_not_found(monkeypatch):
    import src.agents.recommendation_agent as ra
    monkeypatch.setattr(ra, "get_student_context", lambda sid: None)

    result = generate_recommendation(student_id=999)
    assert "error" in result
    assert "999" in result["error"]


def test_generate_recommendation_degree_audit_error(monkeypatch):
    import src.agents.recommendation_agent as ra
    monkeypatch.setattr(ra, "get_student_context", lambda sid: SAMPLE_STUDENT)
    monkeypatch.setattr(ra, "get_degree_audit", lambda sid: {"error": "Program not found"})

    result = generate_recommendation(student_id=1)
    assert "error" in result


def test_generate_recommendation_complete_action(monkeypatch):
    audit = {**SAMPLE_AUDIT, "next_action": "complete", "credits_remaining": 0,
             "total_credits": 32, "program_name": "MS Data Analytics Engineering"}
    import src.agents.recommendation_agent as ra
    monkeypatch.setattr(ra, "get_student_context", lambda sid: SAMPLE_STUDENT)
    monkeypatch.setattr(ra, "get_degree_audit", lambda sid: audit)

    result = generate_recommendation(student_id=1)
    assert result["action"] == "complete"
    assert "Congratulations" in result["recommendation"]
    assert result["courses"] == []


def test_generate_recommendation_ask_path_action(monkeypatch):
    audit = {**SAMPLE_AUDIT, "next_action": "ask_path"}
    import src.agents.recommendation_agent as ra
    monkeypatch.setattr(ra, "get_student_context", lambda sid: SAMPLE_STUDENT)
    monkeypatch.setattr(ra, "get_degree_audit", lambda sid: audit)
    monkeypatch.setattr(ra, "gemini_generate", lambda prompt, **kw: "Which path do you prefer?")

    result = generate_recommendation(student_id=1)
    assert result["action"] == "ask_path"
    assert result["courses"] == []
    assert "Which path" in result["recommendation"]


def test_generate_recommendation_ask_path_gemini_fallback(monkeypatch):
    """Falls back to hardcoded message when Gemini fails during path selection."""
    audit = {**SAMPLE_AUDIT, "next_action": "ask_path"}
    import src.agents.recommendation_agent as ra
    monkeypatch.setattr(ra, "get_student_context", lambda sid: SAMPLE_STUDENT)
    monkeypatch.setattr(ra, "get_degree_audit", lambda sid: audit)
    monkeypatch.setattr(ra, "gemini_generate", MagicMock(side_effect=Exception("Gemini down")))

    result = generate_recommendation(student_id=1)
    assert result["action"] == "ask_path"
    assert result["recommendation"] != ""


def test_generate_recommendation_updates_degree_path(monkeypatch):
    """update_degree_path is called when degree_path arg is provided."""
    import src.agents.recommendation_agent as ra
    update_mock = MagicMock(return_value=True)

    monkeypatch.setattr(ra, "get_student_context", lambda sid: SAMPLE_STUDENT)
    monkeypatch.setattr(ra, "get_degree_audit", lambda sid: SAMPLE_AUDIT)
    monkeypatch.setattr(ra, "update_degree_path", update_mock)
    monkeypatch.setattr(ra, "build_query", lambda goal: SAMPLE_QUERY_RESULT)
    monkeypatch.setattr(ra, "get_relevant_courses", lambda q, ctx, top_k=3, career_goal=None: SAMPLE_COURSES)
    monkeypatch.setattr(ra, "reorder_by_prerequisites",
                        lambda codes, completed, pmap: SAMPLE_PREREQ_STATUS)
    monkeypatch.setattr(ra, "gemini_generate", lambda prompt, **kw: "Your courses")

    generate_recommendation(student_id=1, degree_path="coursework")
    update_mock.assert_called_once_with(1, "coursework")


def test_generate_recommendation_uses_db_career_goal(monkeypatch):
    """Uses target_career from DB when career_goal is not supplied."""
    import src.agents.recommendation_agent as ra
    query_mock = MagicMock(return_value=SAMPLE_QUERY_RESULT)

    monkeypatch.setattr(ra, "get_student_context", lambda sid: SAMPLE_STUDENT)
    monkeypatch.setattr(ra, "get_degree_audit", lambda sid: SAMPLE_AUDIT)
    monkeypatch.setattr(ra, "build_query", query_mock)
    monkeypatch.setattr(ra, "get_relevant_courses", lambda q, ctx, top_k=3, career_goal=None: SAMPLE_COURSES)
    monkeypatch.setattr(ra, "reorder_by_prerequisites",
                        lambda codes, completed, pmap: SAMPLE_PREREQ_STATUS)
    monkeypatch.setattr(ra, "gemini_generate", lambda prompt, **kw: "Recommendation")

    result = generate_recommendation(student_id=1)
    assert result["career_goal"] == "Data Engineer"
    query_mock.assert_called_once_with("Data Engineer")


def test_generate_recommendation_no_courses_found(monkeypatch):
    import src.agents.recommendation_agent as ra
    monkeypatch.setattr(ra, "get_relevant_courses", lambda q, ctx, top_k=3, career_goal=None: [])
    monkeypatch.setattr(ra, "get_degree_audit", lambda sid: SAMPLE_AUDIT)
    monkeypatch.setattr(ra, "build_query", lambda goal: SAMPLE_QUERY_RESULT)
    monkeypatch.setattr(ra, "get_relevant_courses", lambda q, ctx, top_k=3, career_goal=None: SAMPLE_COURSES)

    result = generate_recommendation(student_id=1, career_goal="Data Engineer")
    assert result["action"] == "recommend"
    assert result["courses"] == []
    assert result["recommendation"] != ""


def test_generate_recommendation_full_pipeline(monkeypatch):
    """Happy path returns expected structure with courses and recommendation."""
    import src.agents.recommendation_agent as ra
    monkeypatch.setattr(ra, "get_student_context", lambda sid: SAMPLE_STUDENT)
    monkeypatch.setattr(ra, "get_degree_audit", lambda sid: SAMPLE_AUDIT)
    monkeypatch.setattr(ra, "build_query", lambda goal: SAMPLE_QUERY_RESULT)
    monkeypatch.setattr(ra, "get_relevant_courses", lambda q, ctx, top_k=3, career_goal=None: SAMPLE_COURSES)
    monkeypatch.setattr(ra, "reorder_by_prerequisites",
                        lambda codes, completed, pmap: SAMPLE_PREREQ_STATUS)
    monkeypatch.setattr(ra, "gemini_generate", lambda prompt, **kw: "Take IE7275 first!")

    result = generate_recommendation(student_id=1, career_goal="Data Engineer")

    assert result["action"] == "recommend"
    assert result["career_goal"] == "Data Engineer"
    assert len(result["courses"]) == 2
    assert result["recommendation"] == "Take IE7275 first!"
    assert "student" in result
    assert "degree_audit" in result
    assert "career_skills" in result
    assert "prereq_status" in result


def test_generate_recommendation_gemini_failure_falls_back_to_course_names(monkeypatch):
    """Falls back to listing course names when Gemini fails."""
    import src.agents.recommendation_agent as ra
    monkeypatch.setattr(ra, "get_student_context", lambda sid: SAMPLE_STUDENT)
    monkeypatch.setattr(ra, "get_degree_audit", lambda sid: SAMPLE_AUDIT)
    monkeypatch.setattr(ra, "build_query", lambda goal: SAMPLE_QUERY_RESULT)
    monkeypatch.setattr(ra, "get_relevant_courses", lambda q, ctx, top_k=3, career_goal=None: SAMPLE_COURSES)
    monkeypatch.setattr(ra, "reorder_by_prerequisites",
                        lambda codes, completed, pmap: SAMPLE_PREREQ_STATUS)
    monkeypatch.setattr(ra, "gemini_generate",
                        MagicMock(side_effect=Exception("Gemini unavailable")))

    result = generate_recommendation(student_id=1, career_goal="Data Engineer")
    assert result["action"] == "recommend"
    assert "Data Mining" in result["recommendation"] or \
           "Computation" in result["recommendation"]
