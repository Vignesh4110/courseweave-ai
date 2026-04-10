"""
query_builder.py
----------------
Builds an enriched semantic search query for Pinecone
using careers.json (Adzuna + Gemini skill profile).

responsibility:
- Read careers.json
- Build a rich skill-based query from real job market data
- Pass that query to retriever.py

- HyDE (hypothetical document embedding)
- Hybrid search (dense + sparse BM25)
- Cross-encoder reranking
- MMR diversity

Interface: build_query() returns a rich skill string
that plugs directly into retriever.py as the search query.
"""

import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

CAREERS_JSON_PATH = "data/careers.json"


def load_careers() -> dict:
    """Load careers.json from local file."""
    if not os.path.exists(CAREERS_JSON_PATH):
        raise FileNotFoundError(
            f"careers.json not found at {CAREERS_JSON_PATH}. "
            "Run careers_builder.py first."
        )
    with open(CAREERS_JSON_PATH) as f:
        return json.load(f)


def get_career_skills(career_goal: str) -> dict:
    """
    Look up skill profile for a career goal from careers.json.
    Handles fuzzy matching — "Data Engineer" → "data_engineer"

    Returns skill dict or empty dict if career not found.
    """
    careers_data = load_careers()
    careers      = careers_data.get("careers", {})

    # Normalize career goal to key format
    normalized = career_goal.lower().strip().replace(" ", "_")

    # Direct match
    if normalized in careers:
        return careers[normalized]

    # Fuzzy match — find closest key
    for key in careers:
        if key in normalized or normalized in key:
            logger.info(
                "Fuzzy matched career '%s' → '%s'", career_goal, key
            )
            return careers[key]

    logger.warning("Career '%s' not found in careers.json", career_goal)
    return {}


def build_skill_query(career_goal: str) -> str:
    """
    Build a rich skill-based query string from careers.json.
    Replaces a vague career title with specific skills.

    "Data Engineer" →
    "SQL Python ETL Data Pipelines Data Warehousing CI/CD Git
     Data Modeling Linux/Bash Scripting SQL Python ETL ...
     AWS Azure Databricks Snowflake Spark Terraform dbt Airflow Docker"

    Core skills are weighted more by repeating them.
    """
    skills = get_career_skills(career_goal)

    if not skills:
        logger.warning(
            "No skills found for '%s' — using raw career title", career_goal
        )
        return career_goal

    core    = skills.get("core_skills", [])
    tools   = skills.get("tools", [])
    nicehas = skills.get("nice_to_have", [])

    # Weight core skills more by repeating them twice
    all_skills = core + core + tools + nicehas

    query = " ".join(all_skills)
    logger.info(
        "Built skill query for '%s': %s...", career_goal, query[:80]
    )
    return query


def build_query(career_goal: str) -> dict:
    """
    Master query builder — called by retriever.py.

    Returns:
    {
        "skill_query":   "SQL Python ETL data pipelines...",
        "career_skills": { full skills dict from careers.json }
    }

    retriever.py uses skill_query as the Pinecone search input.
    Teammate's RAG pipeline can further enrich this with HyDE.
    """
    skill_query   = build_skill_query(career_goal)
    career_skills = get_career_skills(career_goal)

    return {
        "skill_query":   skill_query,
        "career_skills": career_skills,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n=== Testing query_builder.py ===\n")

    for career in ["Data Engineer", "Data Scientist", "ML Engineer", "Data Analyst"]:
        print(f"--- {career} ---")
        result = build_query(career)
        print(f"Skill query:  {result['skill_query'][:120]}...")
        print(f"Core skills:  {result['career_skills'].get('core_skills', [])}")
        print(f"Tools:        {result['career_skills'].get('tools', [])}")
        print()