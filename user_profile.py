"""
User Profile module — builds and queries a personalisation index from click history.

Architecture
------------
1. Each click is logged in ``click_log.jsonl`` (already done by app.py).
2. ``sync_history()`` reads the log, fetches the source documents from the main
   wiki index, and copies them into a per-user Elasticsearch index called
   ``user_history``.  The user_history index uses the same BM25 similarity
   so that relevance scores are comparable.
3. ``personalise()`` takes a user query, runs it through BM25 on the
   user_history index, and returns a mapping {doc_id → personalisation_score}.
   Documents in the main results that also appear in (or are topically similar
   to) the user's history receive a boost.
4. ``get_interest_terms()`` extracts the top-N terms from the user's clicked
   documents using Elasticsearch's term-vector API, giving a concise "interest
   profile" that can be displayed in the UI.
"""

import json
from pathlib import Path

from elasticsearch import Elasticsearch, NotFoundError, helpers

from contextual_bandit import bandit_rerank

CLICK_LOG = Path("click_log.jsonl")
MAIN_INDEX = "wiki_index"
HISTORY_INDEX = "user_history"

# BM25 parameters — keep in sync with the main index
BM25_K1 = 1.2
BM25_B = 0.75

# How much weight the personalisation score gets vs. the original BM25 score
# final_score = (1 - ALPHA) * original_bm25 + ALPHA * personalisation_bm25
ALPHA = 0.3

HISTORY_SETTINGS = {
    "settings": {
        "index": {
            "similarity": {
                "custom_bm25": {
                    "type": "BM25",
                    "k1": BM25_K1,
                    "b": BM25_B,
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "title": {
                "type": "text",
                "similarity": "custom_bm25",
            },
            "content": {
                "type": "text",
                "similarity": "custom_bm25",
            },
            "content_semantic": {
                "type": "semantic_text",
            },
            "query": {
                "type": "text",
                "similarity": "custom_bm25",
            },
            "click_count": {
                "type": "integer",
            },
            "last_clicked": {
                "type": "date",
            },
            "rank_at_click": {
                "type": "integer",
            },
        }
    },
}


def _ensure_history_index(es: Elasticsearch) -> None:
    """Create the user_history index if it doesn't exist."""
    if not es.indices.exists(index=HISTORY_INDEX):
        es.indices.create(index=HISTORY_INDEX, body=HISTORY_SETTINGS)


def read_click_log() -> list[dict]:
    """Return all click-log entries as a list of dicts."""
    if not CLICK_LOG.exists():
        return []
    entries = []
    with CLICK_LOG.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def sync_history(es: Elasticsearch) -> dict:
    """
    Synchronise the click log → user_history ES index.

    For each clicked document:
    - fetch the source from the main wiki index
    - upsert it into user_history, incrementing click_count
    - store the query that led to the click

    Returns a summary dict  {synced: int, errors: int, total_in_history: int}.
    """
    _ensure_history_index(es)

    clicks = read_click_log()
    if not clicks:
        return {"synced": 0, "errors": 0, "total_in_history": 0}

    # Group clicks by doc_id so we can batch
    doc_clicks: dict[str, list[dict]] = {}
    for click in clicks:
        did = click["doc_id"]
        doc_clicks.setdefault(did, []).append(click)

    synced = 0
    errors = 0

    for doc_id, click_entries in doc_clicks.items():
        try:
            # Check if already in history
            try:
                existing = es.get(index=HISTORY_INDEX, id=doc_id)
                current_count = existing["_source"].get("click_count", 0)
                current_queries = existing["_source"].get("query", "")
            except NotFoundError:
                current_count = 0
                current_queries = ""

            # Fetch original document
            try:
                source_doc = es.get(index=MAIN_INDEX, id=doc_id)
                source = source_doc["_source"]
            except NotFoundError:
                errors += 1
                continue

            # Collect all queries associated with clicks on this doc
            new_queries = {c["query"] for c in click_entries if c.get("query")}
            all_queries_set = set(current_queries.split(" | ")) if current_queries else set()
            all_queries_set.update(new_queries)
            all_queries_set.discard("")

            # Best (lowest) rank the user clicked at
            best_rank = min(
                (c.get("rank", 999) for c in click_entries), default=999
            )

            # Latest click timestamp
            latest_ts = max(c.get("timestamp", "") for c in click_entries)

            history_doc = {
                "title": source.get("title", ""),
                "content": source.get("content", ""),
                "content_semantic": source.get("content", ""),
                "query": " | ".join(sorted(all_queries_set)),
                "click_count": current_count + len(click_entries),
                "last_clicked": latest_ts,
                "rank_at_click": best_rank,
            }

            es.index(index=HISTORY_INDEX, id=doc_id, body=history_doc)
            synced += 1

        except Exception:
            errors += 1

    es.indices.refresh(index=HISTORY_INDEX)

    # Total docs in history
    count_resp = es.count(index=HISTORY_INDEX)
    total = count_resp.get("count", 0)

    return {"synced": synced, "errors": errors, "total_in_history": total}


def personalise(es: Elasticsearch, query: str, main_results: list[dict],
                alpha: float = ALPHA, use_bandit: bool = False) -> list[dict]:
    """
    Re-rank main_results by blending the original BM25 score with a
    personalisation score from the user_history index.

    Steps:
    1. Run the same query on user_history using BM25.
    2. For each document in main_results, check if it (or a topically related
       document) appears in the history results.
    3. Compute: final = (1 - alpha) * orig_score + alpha * personal_score
    4. Re-sort and return.

    Each result dict gets these extra keys:
        _original_score   – the raw BM25 score from the main index
        _personal_score   – the BM25 score from the user_history query
        _final_score      – the blended score
        _personal_rank    – rank in the user_history results (None if absent)
        _boosted          – bool, whether this result got a boost
    """
    if not main_results:
        return main_results

    # Check if history index exists
    if not es.indices.exists(index=HISTORY_INDEX):
        # No history — return results unchanged with metadata
        for hit in main_results:
            hit["_original_score"] = hit["_score"]
            hit["_personal_score"] = 0.0
            hit["_final_score"] = hit["_score"]
            hit["_personal_rank"] = None
            hit["_boosted"] = False
        return main_results

    # Query user history with BM25
    try:
        history_body = {
            "query": {
                "bool": {
                    "should": [
                        {"match": {"title": {"query": query, "boost": 2.0}}},
                        {"match": {"content": {"query": query}}},
                        {"match": {"query": {"query": query, "boost": 1.5}}},
                        {"semantic": {"field": "content_semantic", "query": query}},
                    ]
                }
            },
            "size": 50,
        }
        history_resp = es.search(index=HISTORY_INDEX, body=history_body)
        history_hits = history_resp["hits"]["hits"]
    except Exception:
        history_hits = []

    # Build lookup: doc_id → (score, rank)
    history_scores: dict[str, tuple[float, int]] = {}
    for rank, h in enumerate(history_hits, 1):
        history_scores[h["_id"]] = (h["_score"], rank)

    # Normalise scores to [0, 1] range for fair blending
    max_orig = max((h["_score"] for h in main_results), default=1.0) or 1.0
    max_hist = max((s for s, _ in history_scores.values()), default=1.0) or 1.0

    for hit in main_results:
        orig = hit["_score"]
        norm_orig = orig / max_orig

        doc_id = hit["_id"]
        if doc_id in history_scores:
            hist_score, hist_rank = history_scores[doc_id]
            norm_hist = hist_score / max_hist
            hit["_personal_score"] = hist_score
            hit["_personal_rank"] = hist_rank
            hit["_boosted"] = True
        else:
            norm_hist = 0.0
            hit["_personal_score"] = 0.0
            hit["_personal_rank"] = None
            hit["_boosted"] = False

        hit["_original_score"] = orig
        blended = (1 - alpha) * norm_orig + alpha * norm_hist
        # Scale back to original-score magnitude so values stay intuitive
        hit["_final_score"] = blended * max_orig

    # ── contextual bandit re-ranking ─────────────────────────────────
    if use_bandit:
        click_entries = read_click_log()
        main_results = bandit_rerank(
            es, query, main_results, history_scores, click_entries
        )
    else:
        # Re-sort by final score descending
        main_results.sort(key=lambda h: h["_final_score"], reverse=True)
    return main_results


def get_interest_terms(es: Elasticsearch, max_terms: int = 20) -> list[dict]:
    """
    Extract the most significant terms from the user's history index using
    ES's significant_terms aggregation, giving a concise interest profile.

    Returns a list of dicts: [{term, doc_count, score}, ...]
    """
    if not es.indices.exists(index=HISTORY_INDEX):
        return []

    try:
        # Use a match_all + significant_terms agg
        body = {
            "size": 0,
            "aggs": {
                "interesting_content": {
                    "significant_terms": {
                        "field": "content",
                        "size": max_terms,
                    }
                },
                "interesting_title": {
                    "significant_terms": {
                        "field": "title",
                        "size": max_terms,
                    }
                },
            },
        }
        resp = es.search(index=HISTORY_INDEX, body=body)
        aggs = resp.get("aggregations", {})

        # Merge both aggregations, dedup by term
        terms_map: dict[str, dict] = {}
        for agg_name in ("interesting_content", "interesting_title"):
            buckets = aggs.get(agg_name, {}).get("buckets", [])
            for b in buckets:
                t = b["key"]
                if t not in terms_map or b.get("score", 0) > terms_map[t].get("score", 0):
                    terms_map[t] = {
                        "term": t,
                        "doc_count": b.get("doc_count", 0),
                        "score": round(b.get("score", 0), 4),
                    }

        # Sort by score descending
        result = sorted(terms_map.values(), key=lambda x: x["score"], reverse=True)
        return result[:max_terms]

    except Exception:
        return []


def get_history_stats(es: Elasticsearch) -> dict:
    """
    Return summary statistics about the user's history.
    """
    if not es.indices.exists(index=HISTORY_INDEX):
        return {"total_docs": 0, "total_clicks": 0, "unique_queries": 0}

    try:
        count = es.count(index=HISTORY_INDEX).get("count", 0)

        # Get all docs to compute click totals and unique queries
        body = {"size": 100, "_source": ["click_count", "query"]}
        resp = es.search(index=HISTORY_INDEX, body=body)
        hits = resp["hits"]["hits"]

        total_clicks = sum(h["_source"].get("click_count", 0) for h in hits)
        all_queries = set()
        for h in hits:
            q = h["_source"].get("query", "")
            for part in q.split(" | "):
                part = part.strip()
                if part:
                    all_queries.add(part)

        return {
            "total_docs": count,
            "total_clicks": total_clicks,
            "unique_queries": len(all_queries),
        }
    except Exception:
        return {"total_docs": 0, "total_clicks": 0, "unique_queries": 0}
