"""
retriever.py
------------
CourseWeave.ai — Full RAG Retrieval Pipeline

Changes vs previous version:
    1. REMOVED rewrite_query()
       Input is always a rich skill string from query_builder.py
       (e.g. "SQL Python ETL data pipelines..."). Rewriting a structured
       skill list with Gemini adds noise, not signal, and cost 1 Gemini
       call per request for zero retrieval improvement.

    2. CACHED HyDE at server startup (_build_hyde_cache)
       Only 6 career goals are supported. HyDE generates a hypothetical
       course description per career — the output is deterministic because
       the input (skill query) never changes between requests. We generate
       all 6 vectors once at startup and serve from memory forever.
       Cost: 6 Gemini calls once at startup.
       Saving: 1 Gemini call per every recommendation request, forever.

Gemini calls per user action after these changes:
     retriever.py at startup  :  6 Groq calls  (once, builds HyDE cache)
    retriever.py per request :  0 calls  (cache hit every time)
    recommendation_agent.py  :  1 call   (the actual recommendation text)
    Total per request        :  1 call

Previously retriever.py alone was 2 calls per request (rewrite + HyDE).

Contract (unchanged):
    Input:  query (str), student_context (dict), top_k (int), career_goal (str)
    Output: list of dicts — course_code, course_name, score, source, text, metadata

Pipeline:
    1. HyDE lookup from startup cache (0 Gemini calls)
    2. Metadata pre-filter
    3. Hybrid retrieval (dense + sparse, Pinecone native fusion)
    4. Cross-encoder re-ranking
    5. Context assembly + MMR diversity
    6. Guardrails (eligible + completed check)
"""

import os
import logging
import numpy as np
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from sentence_transformers import CrossEncoder
from groq import Groq

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GCP_PROJECT_ID      = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION        = os.getenv("GCP_LOCATION", "us-central1")

EMBEDDING_MODEL_NAME   = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_SCORE_THRESHOLD = -10.0  # ms-marco produces logits, not probabilities
                                # -10.0 keeps all reasonable candidates
CANDIDATE_POOL         = 20

# The 6 supported career goals — HyDE vectors pre-built for all of these at startup.
# Must match the CAREERS list in main.py exactly.
SUPPORTED_CAREERS = [
    "Data Engineer",
    "Data Scientist",
    "ML Engineer",
    "Data Analyst",
    "Business Analyst",
    "Software Engineer",
]

logger = logging.getLogger(__name__)


# ============================================================
# MODEL & CLIENT INITIALIZATION
# ============================================================

def _init_embedding_model():
    logger.info("Loading BGE embedding model...")
    model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    logger.info("BGE embedding model loaded")
    return model


def _init_pinecone():
    logger.info("Connecting to Pinecone index: %s", PINECONE_INDEX_NAME)
    pc  = Pinecone(api_key=PINECONE_API_KEY)
    idx = pc.Index(PINECONE_INDEX_NAME)
    logger.info("Pinecone connected")
    return pc, idx


def _init_bm25(index):
    """
    Fetch corpus from Pinecone and fit BM25 encoder.
    Runs once at module load. Falls back gracefully if index
    does not support dotproduct (sparse) queries.
    """
    encoder = BM25Encoder()
    logger.info("Fetching corpus from Pinecone for BM25 fitting...")
    corpus_texts = []

    try:
        for id_batch in index.list(limit=99):
            fetch_response = index.fetch(ids=id_batch)
            for vid, vec in fetch_response.vectors.items():
                text = vec.metadata.get("text", "") if vec.metadata else ""
                if text and isinstance(text, str) and text.strip():
                    corpus_texts.append(text.strip())

        if not corpus_texts:
            raise ValueError("Corpus is empty — no text found in Pinecone metadata.")

        logger.info("Fetched %d documents — fitting BM25...", len(corpus_texts))
        encoder.fit(corpus_texts)
        logger.info("BM25 encoder fitted successfully")

    except Exception as e:
        logger.error("BM25 init failed: %s", e)
        raise

    return encoder


def _init_cross_encoder():
    logger.info("Loading cross-encoder reranker...")
    model = CrossEncoder(CROSS_ENCODER_MODEL)
    logger.info("Cross-encoder loaded")
    return model


def _init_groq():
    logger.info("Initializing Groq client (Llama 3.3 70B) for HyDE generation...")
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    logger.info("Groq client ready")
    return client

def _build_hyde_cache(careers: list, embedding_model, groq_client) -> dict:
    """
    Pre-generate HyDE vectors for all supported career goals at startup.

    Makes exactly len(careers) Gemini calls — once, when the server starts.
    Every recommendation request for the lifetime of the process is served
    from this dict — zero Gemini calls in retriever.py per request.

    Cache structure:
        {
            "data engineer":   { "vector": [...384 floats...], "text": "This course covers..." },
            "data scientist":  { "vector": [...], "text": "..." },
            ...
        }

    Key is lowercase career name for case-insensitive lookup.
    Falls back to direct BGE embedding if Gemini fails for any career —
    the pipeline still works, just with marginally lower retrieval quality
    for that one career.
    """
    cache = {}
    logger.info("Building HyDE cache for %d career goals...", len(careers))

    for career in careers:
        key    = career.lower()
        prompt = (
            f"Write a generic university graduate course description (2-3 sentences) "
            f"that would be highly relevant to a student pursuing a career as a {career}. "
            f"Return only the course description text, nothing else."
        )

        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3,
            )
            hypothesis = response.choices[0].message.content.strip()
            vector     = embedding_model.embed_query(hypothesis)
            cache[key] = {"vector": vector, "text": hypothesis}
            logger.info("HyDE cached for '%s' (%d chars)", career, len(hypothesis))

        except Exception as e:
            logger.warning(
                "HyDE generation failed for '%s', falling back to direct BGE embed: %s",
                career, e
            )
            fallback_query = f"graduate course {career} skills programming data"
            vector         = embedding_model.embed_query(fallback_query)
            cache[key]     = {"vector": vector, "text": fallback_query}

    logger.info("HyDE cache built — %d/%d careers ready", len(cache), len(careers))
    return cache


# ── Lazy initialization — models are loaded on first use, not at import time ──
# This keeps `import src.models.retriever` side-effect-free (safe for CI, tests,
# and any code that imports the module without needing live connections).
# _ensure_initialized() is called automatically by get_relevant_courses() and
# get_hyde_output() on first invocation.

embedding_model = None
pc              = None
index           = None
bm25_encoder    = None
cross_encoder   = None
groq_client     = None
hyde_cache: dict = {}
_initialized    = False


def _ensure_initialized():
    """Initialize all models and clients on first use."""
    global _initialized, embedding_model, pc, index, bm25_encoder, cross_encoder, groq_client, hyde_cache
    if _initialized:
        return
    embedding_model = _init_embedding_model()
    pc, index       = _init_pinecone()
    bm25_encoder    = _init_bm25(index)
    cross_encoder   = _init_cross_encoder()
    groq_client     = _init_groq()
    # 6 Groq calls here once — zero per request after this
    hyde_cache      = _build_hyde_cache(SUPPORTED_CAREERS, embedding_model, groq_client)
    _initialized    = True
    logger.info("All models, clients, and HyDE cache initialized — pipeline ready")


# ============================================================
# STEP 1 — QUERY LAYER
# ============================================================

def get_hyde_output(query: str, career_goal: str = None) -> dict:
    """
    Returns HyDE vector and hypothesis text for a query.

    For all 6 supported careers: returns from startup cache — 0 Gemini calls.
    If career_goal is missing or unrecognised (should not happen given
    validation in main.py): falls back to direct BGE embedding of the
    skill query — still 0 Gemini calls, marginal quality difference.

    rewrite_query() has been removed. The input is always a structured
    skill string from query_builder.py — rewriting it added no retrieval
    improvement and cost 1 Gemini call per request.

    Args:
        query:       Skill query string from query_builder.build_query()
        career_goal: Career name passed from recommendation_agent.py.
                     Used as cache key. Always pass this.
    """
    _ensure_initialized()

    if career_goal:
        key = career_goal.lower().strip()
        if key in hyde_cache:
            cached = hyde_cache[key]
            logger.info("HyDE cache hit: '%s'", career_goal)
            return {
                "original_query" : query,
                "rewritten_query": query,
                "hyde_vector"    : cached["vector"],
                "hyde_text"      : cached["text"],
                "cache_hit"      : True,
            }

    # Fallback — should only trigger if a career slips past main.py validation
    logger.warning(
        "HyDE cache miss for career '%s' — falling back to direct BGE embed. "
        "Check that SUPPORTED_CAREERS in retriever.py matches CAREERS in main.py.",
        career_goal
    )
    vector = embedding_model.embed_query(query)
    return {
        "original_query" : query,
        "rewritten_query": query,
        "hyde_vector"    : vector,
        "hyde_text"      : query,
        "cache_hit"      : False,
    }


# ============================================================
# STEP 2 — METADATA PRE-FILTER
# ============================================================

def build_pinecone_filter(student_context: dict) -> dict:
    """
    Department pre-filter removed — students can take electives across departments.
    Pinecone searches all indexed departments (IE, CS, DS, AI).
    Eligibility and guardrail filtering handled post-retrieval.
    """
    logger.info("Pre-filter: none — cross-department search enabled")
    return {}


# ============================================================
# STEPS 3 & 4 — NATIVE PINECONE HYBRID RETRIEVAL
# ============================================================
# courseweave-hybrid index uses dotproduct metric — supports
# sparse_vector natively. Pinecone fuses dense + sparse internally.
# Confirmed working:
#   Dense:  web_IE_1990 score 0.69
#   Sparse: web_IE_1990 score 0.62
#   Hybrid: web_IE_1990 score 1.31
# ============================================================

def run_hybrid_retrieval(
    query_output: dict,
    pinecone_filter: dict,
    candidate_pool: int = CANDIDATE_POOL
) -> list:
    """
    Native Pinecone hybrid search — dense + sparse in a single query.
    Falls back to dense-only if BM25 encoding fails.
    """
    hyde_vector     = query_output["hyde_vector"]
    rewritten_query = query_output["rewritten_query"]

    sparse_vector = None
    try:
        sparse_vector = bm25_encoder.encode_queries(rewritten_query)
    except Exception as e:
        logger.warning("BM25 encoding failed — running dense-only: %s", e)

    query_params = {
        "vector"          : hyde_vector,
        "top_k"           : candidate_pool,
        "include_metadata": True
    }
    if sparse_vector is not None:
        query_params["sparse_vector"] = sparse_vector
    if pinecone_filter:
        query_params["filter"] = pinecone_filter

    try:
        response = index.query(**query_params)
        matches  = response.get("matches", [])

        mode = "hybrid" if sparse_vector is not None else "dense-only"
        logger.info("Pinecone %s search: %d results", mode, len(matches))

        return [
            {
                "id"       : m["id"],
                "score"    : m["score"],
                "rrf_score": m["score"],
                "metadata" : m.get("metadata", {}) or {}
            }
            for m in matches
        ]

    except Exception as e:
        # If sparse vector is not supported by the index, retry with dense-only
        if sparse_vector is not None and "sparse" in str(e).lower():
            logger.warning("Index does not support sparse vectors — retrying dense-only: %s", e)
            dense_params = {k: v for k, v in query_params.items() if k != "sparse_vector"}
            try:
                response = index.query(**dense_params)
                matches  = response.get("matches", [])
                logger.info("Pinecone dense-only search: %d results", len(matches))
                return [
                    {
                        "id"       : m["id"],
                        "score"    : m["score"],
                        "rrf_score": m["score"],
                        "metadata" : m.get("metadata", {}) or {}
                    }
                    for m in matches
                ]
            except Exception as e2:
                logger.error("Dense-only retrieval also failed: %s", e2)
                return []
        logger.error("Hybrid retrieval failed: %s", e)
        return []


# ============================================================
# STEP 5 — CROSS-ENCODER RE-RANKING
# ============================================================

def rerank_candidates(
    hyde_text: str,
    candidates: list,
    score_threshold: float = RERANK_SCORE_THRESHOLD
) -> list:
    """
    Re-rank candidates using cross-encoder against the HyDE hypothesis text.
    Falls back to retrieval score order if reranking fails.

    Uses hyde_text (natural language course description) rather than the
    raw skill query — natural language scores much better against course
    descriptions in the cross-encoder.
    """
    if not candidates:
        return []

    pairs = [
        [hyde_text, c.get("metadata", {}).get("text", "") or ""]
        for c in candidates
    ]

    try:
        scores = cross_encoder.predict(pairs)
    except Exception as e:
        logger.error("Cross-encoder scoring failed, using retrieval order: %s", e)
        return candidates

    scored = []
    for candidate, score in zip(candidates, scores):
        if score >= score_threshold:
            c = candidate.copy()
            c["rerank_score"] = round(float(score), 4)
            scored.append(c)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    logger.info(
        "Re-ranking: %d → %d passed threshold (%.2f)",
        len(candidates), len(scored), score_threshold
    )
    return scored


# ============================================================
# STEP 6 — CONTEXT ASSEMBLY + MMR DIVERSITY
# ============================================================

def fetch_parent_chunk(course_code: str) -> str:
    """For PDF chunks, fetch sibling chunks and merge for richer context."""
    try:
        parent_ids = [
            f"pdf_{course_code.replace(' ', '_')}_chunk_{i}"
            for i in range(3)
        ]
        fetch_response = index.fetch(ids=parent_ids)
        vectors = fetch_response.vectors if hasattr(fetch_response, "vectors") else {}

        if not vectors:
            return ""

        chunks = sorted(
            vectors.values(),
            key=lambda v: v.get("metadata", {}).get("chunk_index", 0)
        )
        merged = "\n\n".join(
            c.get("metadata", {}).get("text", "")
            for c in chunks
            if c.get("metadata", {}).get("text")
        )
        return merged
    except Exception as e:
        logger.warning("Parent chunk fetch failed for %s: %s", course_code, e)
        return ""


def deduplicate_candidates(candidates: list) -> list:
    """Keep highest-scoring chunk per course_code."""
    seen = {}
    for c in candidates:
        code = c.get("metadata", {}).get("course_code", c["id"])
        if code not in seen:
            seen[code] = c
    return list(seen.values())


def compute_text_similarity(text1: str, text2: str) -> float:
    """Jaccard token overlap similarity — lightweight, no extra model needed."""
    set1 = set((text1 or "").lower().split())
    set2 = set((text2 or "").lower().split())
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def mmr_diversity(
    candidates: list,
    completed_courses: list,
    top_k: int,
    lambda_param: float = 0.7
) -> list:
    """
    Maximal Marginal Relevance diversity filtering.
    Balances relevance vs diversity, penalising overlap with
    already-selected results AND completed courses from Postgres.
    """
    if not candidates:
        return []

    completed_texts = []
    for code in completed_courses:
        norm = code.replace(" ", "").upper()
        for c in candidates:
            c_code = c.get("metadata", {}).get("course_code", "").replace(" ", "").upper()
            if c_code == norm:
                completed_texts.append(c.get("metadata", {}).get("text", "") or "")

    selected  = []
    remaining = candidates.copy()

    while remaining and len(selected) < top_k:
        best_score = -np.inf
        best_idx   = 0

        for i, candidate in enumerate(remaining):
            cand_text = candidate.get("metadata", {}).get("text", "") or ""
            relevance = candidate.get("rerank_score", 0.0)

            sim_selected = max(
                (compute_text_similarity(cand_text, s.get("metadata", {}).get("text", "") or "")
                 for s in selected),
                default=0.0
            )
            sim_completed = max(
                (compute_text_similarity(cand_text, t) for t in completed_texts),
                default=0.0
            )

            mmr_score = (
                (lambda_param * relevance) -
                ((1 - lambda_param) * max(sim_selected, sim_completed))
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx   = i

        selected.append(remaining.pop(best_idx))

    logger.info("MMR selected %d diverse candidates from %d", len(selected), len(candidates))
    return selected


def run_context_assembly(reranked: list, student_context: dict, top_k: int) -> list:
    """
    Full context assembly:
    1. Enrich PDF chunks with parent context
    2. Deduplicate (one chunk per course)
    3. MMR diversity filtering

    Falls back to deduped order if MMR returns empty.
    """
    completed_courses = student_context.get("completed_courses", [])

    enriched = []
    for c in reranked:
        source      = c.get("metadata", {}).get("source", "")
        course_code = c.get("metadata", {}).get("course_code", "")
        c = c.copy()

        if source == "pdf" and course_code:
            parent_text = fetch_parent_chunk(course_code)
            if parent_text:
                c["metadata"] = c.get("metadata", {}).copy()
                c["metadata"]["text"] = parent_text

        enriched.append(c)

    deduped = deduplicate_candidates(enriched)
    logger.info("After dedup: %d candidates", len(deduped))

    diverse = mmr_diversity(deduped, completed_courses, top_k=top_k)

    if not diverse and deduped:
        logger.warning("MMR returned empty — falling back to deduped order")
        diverse = deduped[:top_k]

    return diverse


# ============================================================
# STEP 7 — GUARDRAILS
# ============================================================

def normalize_course_code(code: str) -> str:
    return code.replace(" ", "").upper().strip()


def apply_guardrails(
    assembled: list,
    student_context: dict,
    top_k: int
) -> list:
    """
    Final guardrails before returning results.

    1. Course must be in eligible_courses from Postgres
    2. Course must NOT be in completed_courses (hard stop)
    """
    eligible_courses  = student_context.get("eligible_courses", [])
    completed_courses = student_context.get("completed_courses", [])

    eligible_normalized  = {normalize_course_code(c) for c in eligible_courses}
    completed_normalized = {normalize_course_code(c) for c in completed_courses}

    final_results = []

    for item in assembled:
        meta      = item.get("metadata", {})
        raw_code  = meta.get("course_code", "")
        norm_code = normalize_course_code(raw_code)

        if norm_code not in eligible_normalized:
            logger.debug("Dropping %s — not in eligible courses", norm_code)
            continue

        if norm_code in completed_normalized:
            logger.warning(
                "Guardrail triggered: %s is completed but appeared in results",
                norm_code
            )
            continue

        final_results.append({
            "course_code": norm_code,
            "course_name": meta.get("title", ""),
            "score"      : round(item.get("rerank_score", item.get("rrf_score", 0.0)), 4),
            "source"     : meta.get("source", ""),
            "text"       : meta.get("text", "") or "",
            "metadata"   : meta
        })

    final_results = final_results[:top_k]

    if final_results and final_results[0]["score"] < -5.0:
        logger.warning(
            "Top result score is very low (%.4f) — retrieval quality may be poor.",
            final_results[0]["score"]
        )

    if not final_results:
        logger.warning("No results passed guardrails — returning empty list.")

    logger.info("Guardrails: %d results passed for top_k=%d", len(final_results), top_k)
    return final_results


# ============================================================
# MAIN INTEGRATION FUNCTION
# ============================================================

def get_relevant_courses(
    query:           str,
    student_context: dict,
    top_k:           int = 3,
    career_goal:     str = None,
) -> list[dict]:
    """
    Full RAG retrieval pipeline.

    Args:
        query:           Skill query string from query_builder.build_query()
        student_context: Student context dict from postgres_filter.get_student_context()
        top_k:           Number of courses to return
        career_goal:     Career name — used for HyDE cache lookup.
                         Always pass this from recommendation_agent.py.

    Returns list of dicts: course_code, course_name, score, source, text, metadata
    """
    _ensure_initialized()

    eligible_courses = student_context.get("eligible_courses", [])

    if not eligible_courses:
        logger.warning("No eligible courses in student_context — returning empty.")
        return []

    try:
        query_output     = get_hyde_output(query, career_goal=career_goal)
        pinecone_filter  = build_pinecone_filter(student_context)
        fused_candidates = run_hybrid_retrieval(
            query_output, pinecone_filter, candidate_pool=CANDIDATE_POOL
        )

        if not fused_candidates:
            logger.warning("No candidates returned from retrieval.")
            return []

        reranked  = rerank_candidates(query_output["hyde_text"], fused_candidates)
        assembled = run_context_assembly(reranked, student_context, top_k=top_k * 4)
        final     = apply_guardrails(assembled, student_context, top_k=top_k)

        logger.info(
            "Pipeline complete — %d results for '%s' (cache_hit=%s)",
            len(final), career_goal or query[:40], query_output.get("cache_hit")
        )
        return final

    except Exception as e:
        import traceback
        logger.error(
            "Pipeline failed for query '%s': %s\n%s",
            query[:50], e, traceback.format_exc()
        )
        return []


# ============================================================
# TEST BLOCK
# ============================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    logging.basicConfig(level=logging.INFO)

    from src.models.postgres_filter import get_student_context
    from src.models.query_builder import build_query

    print("\n=== Testing retriever.py (HyDE-cached pipeline) ===\n")
    _ensure_initialized()
    print(f"HyDE cache built for: {list(hyde_cache.keys())}\n")

    context = get_student_context(1)
    print(f"Student:    {context['name']}")
    print(f"Career:     {context['target_career']}")
    print(f"Completed:  {context['completed_courses']}")
    print(f"Eligible:   {context['eligible_courses']}")

    query_result = build_query(context["target_career"])
    query        = query_result["skill_query"]
    print(f"\nQuery: {query[:100]}...\n")

    results = get_relevant_courses(
        query, context, top_k=3, career_goal=context["target_career"]
    )

    if results:
        print(f"\n--- Top {len(results)} Results ---")
        for i, course in enumerate(results, 1):
            print(f"\n{i}. {course['course_code']} — {course['course_name']}")
            print(f"   Score:  {course['score']}")
            print(f"   Source: {course['source']}")
            print(f"   Text:   {course['text'][:120]}...")
    else:
        print("No results returned — check logs above.")