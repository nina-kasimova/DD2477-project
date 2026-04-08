"""
Flask search UI backed by Elasticsearch with explicit BM25 scoring
and personalised re-ranking from user click history.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from elasticsearch import Elasticsearch, NotFoundError
from flask import Flask, abort, jsonify, redirect, render_template, request, url_for
from markupsafe import Markup

from user_profile import (
    ALPHA,
    get_history_stats,
    get_interest_terms,
    personalise,
    sync_history,
)

app = Flask(__name__)
es = Elasticsearch(
    "https://my-elasticsearch-project-a21c67.es.us-central1.gcp.elastic.cloud:443",
    api_key="b3M5VmJKMEJtYkNiV0x5WExmcno6VkRXMVZmcUhuS2oxd1JZSkI0eS1OZw=="
)


def render_explanation(node, depth=0):
    """Recursively render an Elasticsearch _explain node as HTML."""
    if not node or not isinstance(node, dict):
        return ""

    value = node.get("value", 0)
    description = node.get("description", "")
    details = node.get("details", [])

    html = f'<div class="explain-node">'
    html += f'<div class="explain-header">'
    html += f'<span class="explain-score">{value:.6f}</span>'
    html += f'<span class="explain-desc">{description}</span>'
    html += f'</div>'

    if details:
        html += f'<div class="explain-children">'
        for child in details:
            html += render_explanation(child, depth + 1)
        html += f'</div>'

    html += f'</div>'
    return Markup(html)


app.jinja_env.globals["render_explanation"] = render_explanation

INDEX_NAME = "wiki_index"
CLICK_LOG = Path("click_log.jsonl")

# ── BM25 parameters (must match what the index was created with) ─────────────
BM25_K1 = 1.2
BM25_B = 0.75


# ── helpers ──────────────────────────────────────────────────────────────────
def _get_bm25_params() -> dict:
    """Read the actual BM25 settings from the live index."""
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


# ── routes ───────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def search():
    query_text = request.args.get("q", "").strip()
    personalised = request.args.get("personalised", "1") == "1"
    results = []
    total_hits = 0
    search_time_ms = 0
    error = None

    if query_text:
        try:
            body = {
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

            response = es.search(index=INDEX_NAME, body=body)
            results = response["hits"]["hits"]
            total_hits = response["hits"]["total"]["value"]
            search_time_ms = response.get("took", 0)

            # ── personalise results ──────────────────────────────
            if personalised and results:
                results = personalise(es, query_text, results)

        except Exception as exc:
            error = str(exc)

    bm25 = _get_bm25_params()

    # Get user profile stats for the sidebar
    history_stats = get_history_stats(es)
    interest_terms = get_interest_terms(es, max_terms=15)

    return render_template(
        "search.html",
        query=query_text,
        results=results,
        total_hits=total_hits,
        search_time_ms=search_time_ms,
        error=error,
        index_name=INDEX_NAME,
        bm25=bm25,
        personalised=personalised,
        alpha=ALPHA,
        history_stats=history_stats,
        interest_terms=interest_terms,
    )


@app.route("/click/<doc_id>", methods=["GET"])
def track_click(doc_id):
    """Log a click and redirect to the document page."""
    query_text = request.args.get("q", "").strip()
    rank = request.args.get("rank", type=int)

    with CLICK_LOG.open("a", encoding="utf-8") as log_file:
        log_file.write(
            json.dumps(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "doc_id": doc_id,
                    "query": query_text,
                    "rank": rank,
                }
            )
            + "\n"
        )

    # Auto-sync history after each click
    try:
        sync_history(es)
    except Exception:
        pass  # Non-critical — don't block the user

    return redirect(url_for("document", doc_id=doc_id, q=query_text))


@app.route("/doc/<doc_id>", methods=["GET"])
def document(doc_id):
    """Show a single document."""
    query_text = request.args.get("q", "").strip()

    try:
        response = es.get(index=INDEX_NAME, id=doc_id)
    except NotFoundError:
        abort(404)

    return render_template(
        "document.html",
        doc=response["_source"],
        doc_id=doc_id,
        query=query_text,
    )


@app.route("/explain/<doc_id>", methods=["GET"])
def explain(doc_id):
    """Show the BM25 scoring explanation for a document/query pair."""
    query_text = request.args.get("q", "").strip()
    explanation = None
    doc = None
    error = None

    if query_text:
        try:
            body = {
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
                }
            }
            resp = es.explain(index=INDEX_NAME, id=doc_id, body=body)
            explanation = resp.get("explanation", {})

            doc_resp = es.get(index=INDEX_NAME, id=doc_id)
            doc = doc_resp["_source"]
        except Exception as exc:
            error = str(exc)

    bm25 = _get_bm25_params()

    return render_template(
        "explain.html",
        query=query_text,
        doc_id=doc_id,
        doc=doc,
        explanation=explanation,
        error=error,
        bm25=bm25,
    )


# ── API routes for profile management ───────────────────────────────────────
@app.route("/api/sync-history", methods=["POST"])
def api_sync_history():
    """Manually trigger a sync of click_log → user_history index."""
    result = sync_history(es)
    return jsonify(result)


@app.route("/api/profile", methods=["GET"])
def api_profile():
    """Return the user's interest profile as JSON."""
    stats = get_history_stats(es)
    terms = get_interest_terms(es)
    return jsonify({"stats": stats, "interest_terms": terms})


@app.route("/profile", methods=["GET"])
def profile_page():
    """Full-page user profile dashboard."""
    stats = get_history_stats(es)
    terms = get_interest_terms(es)
    clicks = []
    try:
        from user_profile import read_click_log
        clicks = read_click_log()
        # Reverse to show most recent first
        clicks.reverse()
    except Exception:
        pass

    return render_template(
        "profile.html",
        stats=stats,
        interest_terms=terms,
        clicks=clicks,
    )


if __name__ == "__main__":
    # Sync history on startup
    try:
        sync_history(es)
        print("✓ User history synced on startup")
    except Exception as exc:
        print(f"✗ Could not sync history: {exc}")

    app.run(debug=True)
