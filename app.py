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
    read_click_log,
    get_click_log,
    get_history_index,
)
from contextual_bandit import get_model as get_bandit_model, record_impressions_and_click

app = Flask(__name__)
es = Elasticsearch(
    "https://my-elasticsearch-project-b7af22.es.us-central1.gcp.elastic.cloud:443",
    api_key="RDdDRmdwMEI0dlNRLXQyRWhtRVY6UmpFeDNmRGV0dXB4Wk4yb0Fwc25TZw=="
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

from user_profile import get_click_log
import glob

INDEX_NAME = "wiki_index"

def get_manual_interests(profile_id: str) -> list:
    p = Path(f"interests_{profile_id}.txt")
    if p.exists():
        return list(set(line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()))
    return []

def get_all_profiles():
    profiles = set(["default"])
    for f in glob.glob("click_log_*.jsonl"):
        name = Path(f).stem.replace("click_log_", "")
        if name:
            profiles.add(name)
    if Path("click_log.jsonl").exists():
        profiles.add("default")
    return sorted(list(profiles))

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
    use_bandit = request.args.get("bandit", "0") == "1"
    profile_id = request.args.get("profile_id", "default").strip()
    user_interest = request.args.get("user_interest", "").strip()
    
    try:
        result_size = int(request.args.get("size", 20))
    except ValueError:
        result_size = 20
    
    # Persist the profile simply by ensuring its click log file exists
    if profile_id:
        log_path = get_click_log(profile_id)
        if not log_path.exists():
            log_path.touch()
            
    results = []
    total_hits = 0
    search_time_ms = 0
    error = None

    if query_text:
        try:
            body = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query_text,
                                    "fields": ["title^2.0", "content"]
                                }
                            }
                        ],
                        "should": []
                    }
                },
                "size": result_size,
            }

            all_interests = get_manual_interests(profile_id)
            if user_interest and user_interest not in all_interests:
                all_interests.append(user_interest)

            for interest in all_interests:
                body["query"]["bool"]["should"].append(
                    {"multi_match": {"query": interest, "fields": ["title^1.2", "content^1.5"]}}
                )

            response = es.search(index=INDEX_NAME, body=body)
            results = response["hits"]["hits"]
            total_hits = response["hits"]["total"]["value"]
            search_time_ms = response.get("took", 0)

            # ── personalise results ───────────────────────────────
            if personalised and results:
                results = personalise(es, query_text, results,
                                      use_bandit=use_bandit, profile_id=profile_id)

        except Exception as exc:
            error = str(exc)

    bm25 = _get_bm25_params()

    # Get user profile stats for the sidebar
    history_stats = get_history_stats(es, profile_id=profile_id)
    interest_terms = get_interest_terms(es, max_terms=15, profile_id=profile_id)

    # Get bandit model stats
    bandit_model = get_bandit_model()

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
        use_bandit=use_bandit,
        alpha=ALPHA,
        history_stats=history_stats,
        interest_terms=interest_terms,
        bandit_arms=bandit_model.num_arms,
        profile_id=profile_id,
        user_interest=user_interest,
        available_profiles=get_all_profiles(),
        result_size=result_size,
    )


@app.route("/click/<doc_id>", methods=["GET"])
def track_click(doc_id):
    """Log a click and redirect to the document page."""
    query_text = request.args.get("q", "").strip()
    rank = request.args.get("rank", type=int)
    profile_id = request.args.get("profile_id", "default").strip()
    
    click_log_path = get_click_log(profile_id)
    with click_log_path.open("a", encoding="utf-8") as log_file:
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

    # Auto-sync history after each click in a background thread so it doesn't block the UI
    def background_sync(es_client, pid):
        try:
            sync_history(es_client, profile_id=pid)
        except Exception:
            pass
            
    import threading
    threading.Thread(target=background_sync, args=(es, profile_id)).start()

    return redirect(url_for("document", doc_id=doc_id, q=query_text, profile_id=profile_id))


@app.route("/api/bandit-feedback", methods=["POST"])
def api_bandit_feedback():
    """
    Record bandit feedback when a user clicks a result.
    Expects JSON: {query, clicked_doc_id, shown_doc_ids, shown_scores}
    """
    try:
        data = request.get_json(force=True)
        query = data.get("query", "")
        clicked_doc_id = data.get("clicked_doc_id", "")
        shown_doc_ids = data.get("shown_doc_ids", [])
        profile_id = data.get("profile_id", "default")

        # Build minimal result dicts for the bandit
        shown_results = []
        for i, did in enumerate(shown_doc_ids):
            shown_results.append({
                "_id": did,
                "_score": data.get("shown_scores", [1.0] * len(shown_doc_ids))[i],
            })

        # Get history scores for context
        from user_profile import get_history_index
        history_scores = {}
        history_idx = get_history_index(profile_id)
        try:
            if es.indices.exists(index=history_idx):
                history_body = {
                    "query": {
                        "bool": {
                            "should": [
                                {"match": {"title": {"query": query, "boost": 2.0}}},
                                {"match": {"content": {"query": query}}},
                            ]
                        }
                    },
                    "size": 50,
                }
                history_resp = es.search(index=history_idx, body=history_body)
                for rank, h in enumerate(history_resp["hits"]["hits"], 1):
                    history_scores[h["_id"]] = (h["_score"], rank)
        except Exception:
            pass

        click_entries = read_click_log(profile_id)
        record_impressions_and_click(
            query, shown_results, clicked_doc_id,
            history_scores, click_entries,
        )
        return jsonify({"status": "ok", "arms": get_bandit_model().num_arms})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/doc/<doc_id>", methods=["GET"])
def document(doc_id):
    """Show a single document."""
    query_text = request.args.get("q", "").strip()
    profile_id = request.args.get("profile_id", "default").strip()

    try:
        response = es.get(index=INDEX_NAME, id=doc_id)
    except NotFoundError:
        abort(404)

    return render_template(
        "document.html",
        doc=response["_source"],
        doc_id=doc_id,
        query=query_text,
        profile_id=profile_id,
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
                            }
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
    data = request.get_json(force=True, silent=True) or {}
    profile_id = data.get("profile_id", "default")
    result = sync_history(es, profile_id=profile_id)
    return jsonify(result)


@app.route("/users", methods=["GET"])
def users_list():
    """Directory of all users in the system."""
    profiles = get_all_profiles()
    users_data = []
    for p in profiles:
        stats = get_history_stats(es, profile_id=p)
        users_data.append({
            "id": p,
            "clicks": stats.get("total_clicks", 0),
            "docs": stats.get("total_docs", 0),
        })
    return render_template("users.html", users=users_data)


@app.route("/api/profile", methods=["GET"])
def api_profile():
    """Return the user's interest profile as JSON."""
    profile_id = request.args.get("profile_id", "default").strip()
    stats = get_history_stats(es, profile_id=profile_id)
    terms = get_interest_terms(es, profile_id=profile_id)
    return jsonify({"stats": stats, "interest_terms": terms})


@app.route("/profile", methods=["GET"])
def profile_page():
    """Full-page user profile dashboard."""
    profile_id = request.args.get("profile_id", "default").strip()
    stats = get_history_stats(es, profile_id=profile_id)
    terms = get_interest_terms(es, max_terms=50, profile_id=profile_id)
    manual_interests = get_manual_interests(profile_id)
    clicks = []
    try:
        from user_profile import read_click_log
        clicks = read_click_log(profile_id=profile_id)
        # Reverse to show most recent first
        clicks.reverse()
    except Exception:
        pass

    return render_template(
        "profile.html",
        stats=stats,
        interest_terms=terms,
        clicks=clicks,
        profile_id=profile_id,
        manual_interests=manual_interests,
    )

@app.route("/api/add-interest", methods=["POST"])
def add_interest():
    profile_id = request.form.get("profile_id", "default").strip()
    interest = request.form.get("interest", "").strip()
    if interest:
        with open(f"interests_{profile_id}.txt", "a", encoding="utf-8") as f:
            f.write(interest + "\n")
    return redirect(url_for("profile_page", profile_id=profile_id))

@app.route("/api/remove-interest", methods=["POST"])
def remove_interest():
    profile_id = request.form.get("profile_id", "default").strip()
    interest = request.form.get("interest", "").strip()
    p = Path(f"interests_{profile_id}.txt")
    if p.exists():
        lines = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip() and line.strip() != interest]
        p.write_text("\n".join(lines), encoding="utf-8")
    return redirect(url_for("profile_page", profile_id=profile_id))

@app.route("/api/delete-user", methods=["POST"])
def delete_user():
    """Wipe a user from disk and Elasticsearch indices."""
    profile_id = request.form.get("profile_id", "").strip()
    if profile_id:
        # Delete JSONL logs
        log_path = get_click_log(profile_id)
        if log_path.exists():
            log_path.unlink()
            
        # Delete manual interests
        int_path = Path(f"interests_{profile_id}.txt")
        if int_path.exists():
            int_path.unlink()
            
        # Delete Elasticsearch history index
        idx = get_history_index(profile_id)
        try:
            if es.indices.exists(index=idx):
                es.indices.delete(index=idx)
        except Exception:
            pass

    return redirect(url_for('users_list'))


if __name__ == "__main__":
    # Sync history on startup
    try:
        sync_history(es)
        print("✓ User history synced on startup")
    except Exception as exc:
        print(f"✗ Could not sync history: {exc}")

    app.run(debug=True)
