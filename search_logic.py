import re
from datetime import datetime, timedelta, timezone

from config import (
    BM25_B,
    BM25_K1,
    BIO_TERM_WEIGHT,
    CLICK_QUERY_TERM_WEIGHT,
    INDEX_NAME,
    MAX_CLICK_HISTORY,
    MAX_PROFILE_TERMS,
    PERSONALIZATION_CONTENT_BOOST,
    PERSONALIZATION_TITLE_BOOST,
    PROFILE_WINDOW_DAYS,
    STOPWORDS,
    es,
)
from data_store import user_activity


def get_bm25_params() -> dict:
    try:
        settings = es.indices.get_settings(index=INDEX_NAME)
        sim = (
            settings[INDEX_NAME]["settings"]["index"]
            .get("similarity", {})
            .get("custom_bm25", {})
        )
        return {
            "k1": float(sim.get("k1", BM25_K1)),
            "b": float(sim.get("b", BM25_B)),
        }
    except Exception:
        return {"k1": BM25_K1, "b": BM25_B}


def extract_terms(text: str, limit: int = 12) -> list[str]:
    terms = []
    seen = set()

    for raw_term in re.findall(r"[a-zA-Z]{3,}", text.lower()):
        if raw_term in STOPWORDS or raw_term in seen:
            continue
        seen.add(raw_term)
        terms.append(raw_term)
        if len(terms) >= limit:
            break

    return terms


def build_search_body(query_text: str, mode: str) -> dict:
    if mode == "simple":
        return {
            "query": {
                "multi_match": {
                    "query": query_text,
                    "fields": ["title", "content"],
                }
            },
            "size": 10,
        }

    return {
        "query": {
            "bool": {
                "should": [
                    {
                        "match": {
                            "title": {
                                "query": query_text,
                                "boost": 2.0,
                            }
                        }
                    },
                    {
                        "match": {
                            "content": {
                                "query": query_text,
                            }
                        }
                    },
                ]
            }
        },
        "size": 10,
    }


def build_user_profile(user: dict | None, limit: int = MAX_PROFILE_TERMS) -> list[dict]:
    if not user:
        return []

    weighted_terms = {}

    for term in extract_terms(user.get("bio", ""), limit=limit):
        weighted_terms[term] = weighted_terms.get(term, 0.0) + BIO_TERM_WEIGHT

    _, clicks = user_activity(user["user_id"])
    cutoff = datetime.now(timezone.utc) - timedelta(days=PROFILE_WINDOW_DAYS)
    recent_clicks = []
    for click in clicks:
        timestamp = click.get("timestamp")
        if not timestamp:
            continue
        try:
            clicked_at = datetime.fromisoformat(timestamp)
        except ValueError:
            continue
        if clicked_at >= cutoff:
            recent_clicks.append(click)
        if len(recent_clicks) >= MAX_CLICK_HISTORY:
            break

    for click in recent_clicks:
        for term in extract_terms(click.get("query", ""), limit=limit):
            weighted_terms[term] = weighted_terms.get(term, 0.0) + CLICK_QUERY_TERM_WEIGHT

    ranked_terms = sorted(
        weighted_terms.items(),
        key=lambda item: (-item[1], item[0]),
    )[:limit]
    return [{"term": term, "weight": weight} for term, weight in ranked_terms]


def user_profile_terms(user: dict | None, limit: int = MAX_PROFILE_TERMS) -> list[str]:
    return [row["term"] for row in build_user_profile(user, limit=limit)]


def personalized_search_body(query_text: str, mode: str, profile: list[dict]) -> dict:
    body = build_search_body(query_text, mode)
    if not profile:
        return body

    personalization_should = []
    for row in profile:
        term = row["term"]
        weight = row["weight"]
        personalization_should.append(
            {
                "match": {
                    "title": {
                        "query": term,
                        "boost": PERSONALIZATION_TITLE_BOOST * weight,
                    }
                }
            }
        )
        personalization_should.append(
            {
                "match": {
                    "content": {
                        "query": term,
                        "boost": PERSONALIZATION_CONTENT_BOOST * weight,
                    }
                }
            }
        )

    body["query"] = {
        "bool": {
            "must": [body["query"]],
            "should": personalization_should,
        }
    }
    return body
