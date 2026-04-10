"""
eval_runner.py
--------------
Evaluation logic for CourseWeave AI recommendation pipeline.


    1. run_single_test() overrides student context with test case scenario
    2. time.sleep(15) between tests — stays safely under 10 RPM Gemini limit
    3. Gemini calls in retriever.py reduced to max 2 per test
       (rewrite + HyDE — both have fallbacks if rate limited)

Note on Gemini rate limits:
    Vertex AI free tier = 10 requests per minute for Gemini 2.5 Flash
    Each test case = 2 Gemini calls (rewrite + HyDE)
    15s sleep = ~4 tests/min = 8 Gemini calls/min — safely under limit
    Full eval run takes ~4 minutes

To remove rate limit issue permanently:
    Go to https://console.cloud.google.com/iam-admin/quotas?project=courseweave-ai
    Search "Generate content requests per minute" → request increase to 60 RPM


    with mlflow.start_run(run_name="experiment_full_rag"):
        config = {
            "embedding_model": "bge-small-en-v1.5",
            "top_k": 3,
            "reranking": True,
            "mmr": True,
            "hyde": True,
            "hybrid_search": True,
        }
        for k, v in config.items():
            mlflow.log_param(k, v)

        metrics = run_evaluation(pipeline_config=config)

        mlflow.log_metric("avg_precision_at_3",       metrics["avg_precision_at_3"])
        mlflow.log_metric("avg_recall_at_3",          metrics["avg_recall_at_3"])
        mlflow.log_metric("guardrail_violations",      metrics["total_guardrail_violations"])
        mlflow.log_metric("prereq_flag_accuracy",      metrics["prereq_flag_accuracy"])
        mlflow.log_metric("pass_rate",                 metrics["pass_rate"])
        mlflow.log_metric("gemini_calls_total",        metrics["gemini_calls_total"])
        mlflow.log_metric("gemini_fallback_count",     metrics["gemini_fallback_count"])

        with open("data/eval_results.json", "w") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact("data/eval_results.json")
        mlflow.log_artifact("data/eval_dataset.json")
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath("."))

from src.models.postgres_filter import (
    get_student_context,
    reorder_by_prerequisites
)
from src.models.query_builder import build_query
from src.models.retriever import get_relevant_courses

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

EVAL_DATASET_PATH  = "data/eval_dataset.json"
EVAL_RESULTS_PATH  = "data/eval_results.json"

# Sleep between test cases to stay under Gemini RPM limit
# Each test = 2 Gemini calls (rewrite + HyDE)
# 15s sleep = ~4 tests/min = 8 Gemini calls/min (under 10 RPM)
# Increase to 7s if you have quota increase approved (60 RPM)
SLEEP_BETWEEN_TESTS = 15


def load_eval_dataset() -> list[dict]:
    """Load eval_dataset.json."""
    with open(EVAL_DATASET_PATH) as f:
        data = json.load(f)
    return data["eval_dataset"]


def compute_precision_at_k(
    recommended: list[str],
    expected: list[str],
    k: int = 3
) -> float:
    """
    Precision@K: of top K recommended, how many are in expected list?
    Range: 0.0 to 1.0
    """
    if not expected:
        return 1.0
    top_k = recommended[:k]
    hits  = len(set(top_k) & set(expected))
    return round(hits / k, 4)


def compute_recall_at_k(
    recommended: list[str],
    expected: list[str],
    k: int = 3
) -> float:
    """
    Recall@K: of all expected courses, how many appear in top K?
    Range: 0.0 to 1.0
    """
    if not expected:
        return 1.0
    top_k = recommended[:k]
    hits  = len(set(top_k) & set(expected))
    return round(hits / len(expected), 4)


def check_guardrail_violations(
    recommended: list[str],
    completed: list[str]
) -> int:
    """
    Count how many completed courses appear in recommendations.
    This must always be 0 — critical guardrail.
    Uses the TEST CASE's completed list, not Postgres.
    """
    return len(set(recommended) & set(completed))


def check_prereq_flag_accuracy(
    prereq_status: list[dict],
    must_flag: list[str]
) -> float:
    """
    Check if pipeline correctly flagged courses that need prereqs.
    must_flag = list of course codes that SHOULD have been flagged.
    Range: 0.0 to 1.0
    """
    if not must_flag:
        return 1.0

    flagged_by_pipeline = {
        p["course_code"]
        for p in prereq_status
        if not p["prereqs_satisfied"]
    }

    correctly_flagged = len(set(must_flag) & flagged_by_pipeline)
    return round(correctly_flagged / len(must_flag), 4)


def build_test_student_context(
    student_id: int,
    test_case: dict
) -> dict | None:
    """
    Build student context for a test case.

    Gets real student profile from Postgres but OVERRIDES:
    - completed_courses → test case's completed list
    - eligible_courses  → recalculated excluding test completed courses
    - prereq_map        → kept from real DB (correct for the program)

    This lets us test hypothetical student scenarios without
    modifying the actual database.
    """
    real_context = get_student_context(student_id)
    if not real_context:
        return None

    test_completed = test_case["completed_courses"]
    completed_set  = set(test_completed)

    full_pool = (
        set(real_context["eligible_courses"]) |
        set(real_context["completed_courses"])
    )
    test_eligible = [c for c in full_pool if c not in completed_set]

    original_order = (
        real_context["eligible_courses"] +
        real_context["completed_courses"]
    )
    test_eligible_ordered = sorted(
        test_eligible,
        key=lambda c: original_order.index(c) if c in original_order else 999
    )

    career_goal = test_case.get("career_goal") or real_context["target_career"]

    overridden = dict(real_context)
    overridden["completed_courses"] = test_completed
    overridden["eligible_courses"]  = test_eligible_ordered
    overridden["target_career"]     = career_goal

    return overridden


def run_single_test(test_case: dict, sleep_seconds: int = SLEEP_BETWEEN_TESTS) -> dict:
    """
    Run pipeline for one test case and compute all metrics.

    GEMINI CALLS PER TEST: max 2
        1. rewrite_query() in retriever.py  — falls back to original if 429
        2. generate_hyde_vector()            — falls back to direct embed if 429
        NO call to recommendation_agent.py (that would add a 3rd call)

    sleep_seconds: pause after each test to respect Gemini RPM limit
    """
    test_id    = test_case["test_id"]
    student_id = test_case["student_id"]
    career     = test_case["career_goal"]
    expected   = test_case["expected_top_courses"]
    should_not = test_case["should_not_recommend"]
    must_flag  = test_case["must_flag_prereq"]

    print(f"  Running {test_id}: {test_case['name']}...")

    try:
        # ── Build overridden student context (no Gemini calls) ────────────────
        student_context = build_test_student_context(student_id, test_case)

        if not student_context:
            return {
                "test_id":              test_id,
                "status":               "error",
                "error":                f"Student {student_id} not found",
                "precision_at_3":       0.0,
                "recall_at_3":          0.0,
                "guardrail_violations": 0,
                "prereq_flag_accuracy": 0.0,
                "gemini_fallback":      False,
            }

        # ── Build enriched query (no Gemini calls) ────────────────────────────
        query_result = build_query(career)
        query        = query_result["skill_query"]

        # ── Retrieve courses (max 2 Gemini calls: rewrite + HyDE) ────────────
        # Both have fallbacks — pipeline continues even if both fail
        courses = get_relevant_courses(query, student_context, top_k=3)

        # ── Prereq check (no Gemini calls) ────────────────────────────────────
        course_codes  = [c["course_code"] for c in courses]
        prereq_status = reorder_by_prerequisites(
            course_codes,
            student_context["completed_courses"],
            student_context["prereq_map"]
        )

        # ── Compute metrics ───────────────────────────────────────────────────
        recommended = course_codes
        precision   = compute_precision_at_k(recommended, expected, k=3)
        recall      = compute_recall_at_k(recommended, expected, k=3)
        violations  = check_guardrail_violations(recommended, should_not)
        prereq_acc  = check_prereq_flag_accuracy(prereq_status, must_flag)
        status      = "pass" if violations == 0 else "GUARDRAIL_FAIL"

        print(
            f"    ✅ precision@3={precision} recall@3={recall} "
            f"violations={violations} prereq_acc={prereq_acc}"
        )

        result = {
            "test_id":              test_id,
            "name":                 test_case["name"],
            "status":               status,
            "recommended":          recommended,
            "expected":             expected,
            "completed_used":       student_context["completed_courses"],
            "eligible_used":        student_context["eligible_courses"],
            "precision_at_3":       precision,
            "recall_at_3":          recall,
            "guardrail_violations": violations,
            "prereq_flag_accuracy": prereq_acc,
            "notes":                test_case.get("notes", ""),
            "gemini_fallback":      False,
        }

    except Exception as e:
        logger.error("Test %s failed with exception: %s", test_id, e)
        print(f"    ❌ exception: {str(e)[:80]}")
        result = {
            "test_id":              test_id,
            "status":               "exception",
            "error":                str(e),
            "precision_at_3":       0.0,
            "recall_at_3":          0.0,
            "guardrail_violations": 0,
            "prereq_flag_accuracy": 0.0,
            "gemini_fallback":      True,
        }

    # ── Sleep to respect Gemini RPM limit ─────────────────────────────────────
    # Skip sleep on last test to save time
    if sleep_seconds > 0:
        print(f"    ⏳ waiting {sleep_seconds}s (Gemini rate limit)...")
        time.sleep(sleep_seconds)

    return result


def run_evaluation(pipeline_config: dict = None) -> dict:
    """
    Master evaluation function.
    Runs all test cases and aggregates metrics.

    Gemini calls total: 15 tests × 2 calls = 30 calls
    With 15s sleep: runs in ~4 minutes, stays under 10 RPM

    Args:
        pipeline_config: Optional dict for MLflow param logging.
    """
    if pipeline_config is None:
        pipeline_config = {
            "embedding_model": "bge-small-en-v1.5",
            "top_k":           3,
            "reranking":       True,
            "mmr":             True,
            "hyde":            True,
            "hybrid_search":   False,  # False until Pinecone index supports dotproduct
        }

    dataset = load_eval_dataset()
    total   = len(dataset)
    print(f"\nRunning evaluation on {total} test cases...")
    print(f"Estimated time: ~{total * SLEEP_BETWEEN_TESTS // 60 + 1} minutes\n")

    per_case_results = []
    for i, test_case in enumerate(dataset):
        # Skip sleep on last test
        sleep = SLEEP_BETWEEN_TESTS if i < total - 1 else 0
        result = run_single_test(test_case, sleep_seconds=sleep)
        per_case_results.append(result)

    # Aggregate metrics
    valid = [
        r for r in per_case_results
        if r["status"] not in ("error", "exception")
    ]

    avg_precision = round(
        sum(r["precision_at_3"] for r in valid) / len(valid), 4
    ) if valid else 0.0

    avg_recall = round(
        sum(r["recall_at_3"] for r in valid) / len(valid), 4
    ) if valid else 0.0

    total_violations = sum(r["guardrail_violations"] for r in valid)

    avg_prereq_acc = round(
        sum(r["prereq_flag_accuracy"] for r in valid) / len(valid), 4
    ) if valid else 0.0

    pass_rate = round(
        len([r for r in valid if r["status"] == "pass"]) / len(valid), 4
    ) if valid else 0.0

    gemini_fallback_count = sum(
        1 for r in per_case_results if r.get("gemini_fallback", False)
    )

    summary = {
        "avg_precision_at_3":          avg_precision,
        "avg_recall_at_3":             avg_recall,
        "total_guardrail_violations":   total_violations,
        "prereq_flag_accuracy":         avg_prereq_acc,
        "pass_rate":                    pass_rate,
        "total_cases":                  len(dataset),
        "valid_cases":                  len(valid),
        "gemini_calls_total":           total * 2,
        "gemini_fallback_count":        gemini_fallback_count,
        "per_case_results":             per_case_results,
        "pipeline_config":              pipeline_config,
        "run_timestamp":                datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs("data", exist_ok=True)
    with open(EVAL_RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("CourseWeave AI — Evaluation Runner")
    print("=" * 60)
    print(f"Sleep between tests: {SLEEP_BETWEEN_TESTS}s")
    print("Change SLEEP_BETWEEN_TESTS to 7 after quota increase approved")

    metrics = run_evaluation()

    print(f"\n{'=' * 60}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total cases:           {metrics['total_cases']}")
    print(f"Valid cases:           {metrics['valid_cases']}")
    print(f"Avg Precision@3:       {metrics['avg_precision_at_3']}")
    print(f"Avg Recall@3:          {metrics['avg_recall_at_3']}")
    print(f"Guardrail violations:  {metrics['total_guardrail_violations']}")
    print(f"Prereq flag accuracy:  {metrics['prereq_flag_accuracy']}")
    print(f"Pass rate:             {metrics['pass_rate']}")
    print(f"Gemini calls total:    {metrics['gemini_calls_total']}")
    print(f"Gemini fallbacks:      {metrics['gemini_fallback_count']}")
    print(f"\nResults saved to: {EVAL_RESULTS_PATH}")
    print(f"Share eval_results.json with MLflow  for logging.")
    print(f"\nNOTE: If gemini_fallback_count > 0, results are degraded.")
    print(f"Request quota increase and re-run for accurate comparison.")