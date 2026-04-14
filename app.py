import json
from datetime import datetime, UTC
from pathlib import Path

from flask import Flask, abort, redirect, render_template, request, session, url_for
from elasticsearch import Elasticsearch, NotFoundError


app = Flask(__name__)
app.secret_key = "toy-search-secret"
es = Elasticsearch("http://localhost:9200")
INDEX_NAME = "wiki_index"
CLICK_LOG = Path("click_log.jsonl")
SEARCH_LOG = Path("search_log.jsonl")
USER_PROFILES = Path("user_profiles.json")

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


def _build_search_body(query_text: str, mode: str) -> dict:
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


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _load_profiles() -> dict:
    if not USER_PROFILES.exists():
        return {}

    with USER_PROFILES.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_profiles(profiles: dict) -> None:
    with USER_PROFILES.open("w", encoding="utf-8") as handle:
        json.dump(profiles, handle, indent=2)


def _all_users() -> list[dict]:
    profiles = _load_profiles()
    users = []
    for user_id in sorted(profiles):
        users.append(
            {
                "user_id": user_id,
                "bio": profiles[user_id].get("bio", ""),
            }
        )
    return users


def _current_user() -> dict | None:
    user_id = session.get("user_id")
    if not user_id:
        return None

    profiles = _load_profiles()
    profile = profiles.get(user_id, {})
    return {
        "user_id": user_id,
        "bio": profile.get("bio", ""),
    }


def _log_search(user_id: str, query_text: str, mode: str) -> None:
    _append_jsonl(
        SEARCH_LOG,
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "user_id": user_id,
            "query": query_text,
            "mode": mode,
        },
    )


def _user_activity(user_id: str) -> tuple[list[dict], list[dict]]:
    searches = [row for row in _read_jsonl(SEARCH_LOG) if row.get("user_id") == user_id]
    clicks = [row for row in _read_jsonl(CLICK_LOG) if row.get("user_id") == user_id]
    searches.reverse()
    clicks.reverse()
    return searches, clicks


def _user_profile_terms(user: dict | None, limit: int = 8) -> list[str]:
    if not user:
        return []

    terms = []
    seen = set()

    bio = user.get("bio", "")
    for raw_term in bio.replace(",", " ").split():
        term = raw_term.strip().lower()
        if len(term) > 2 and term not in seen:
            seen.add(term)
            terms.append(term)

    _, clicks = _user_activity(user["user_id"])
    for click in clicks[:10]:
        for raw_term in click.get("query", "").replace(",", " ").split():
            term = raw_term.strip().lower()
            if len(term) > 2 and term not in seen:
                seen.add(term)
                terms.append(term)
            if len(terms) >= limit:
                return terms

    return terms[:limit]


def _personalized_search_body(query_text: str, mode: str, profile_terms: list[str]) -> dict:
    body = _build_search_body(query_text, mode)
    if not profile_terms:
        return body

    interest_text = " ".join(profile_terms)
    body["query"] = {
        "bool": {
            "must": [body["query"]],
            "should": [
                {"match": {"title": {"query": interest_text, "boost": 0.8}}},
                {"match": {"content": {"query": interest_text, "boost": 0.4}}},
            ],
        }
    }
    return body


@app.route("/set-user", methods=["POST"])
def set_user():
    user_id = request.form.get("user_id", "").strip()
    bio = request.form.get("bio", "").strip()
    mode = request.form.get("mode", "boosted")
    query_text = request.form.get("q", "").strip()

    if user_id:
        session["user_id"] = user_id
        profiles = _load_profiles()
        profiles[user_id] = {"bio": bio}
        _save_profiles(profiles)
    else:
        session.pop("user_id", None)

    return redirect(url_for("search", q=query_text, mode=mode))


@app.route("/profile", methods=["GET"])
def profile():
    user = _current_user()
    if not user:
        return redirect(url_for("search"))

    searches, clicks = _user_activity(user["user_id"])
    return render_template(
        "profile.html",
        user=user,
        searches=searches[:20],
        clicks=clicks[:20],
        all_users=_all_users(),
    )


@app.route("/profile/update", methods=["POST"])
def update_profile():
    user = _current_user()
    if not user:
        return redirect(url_for("search"))

    bio = request.form.get("bio", "").strip()
    profiles = _load_profiles()
    profiles[user["user_id"]] = {"bio": bio}
    _save_profiles(profiles)
    return redirect(url_for("profile"))


@app.route("/profile/clear-clicks", methods=["POST"])
def clear_clicks():
    user = _current_user()
    if not user:
        return redirect(url_for("search"))

    rows = [row for row in _read_jsonl(CLICK_LOG) if row.get("user_id") != user["user_id"]]
    _write_jsonl(CLICK_LOG, rows)
    return redirect(url_for("profile"))


@app.route("/", methods=["GET"])
def search():
    query_text = request.args.get("q", "").strip()
    mode = request.args.get("mode", "boosted")
    user = _current_user()
    profile_terms = _user_profile_terms(user)
    results = []
    total_hits = 0
    search_time_ms = 0
    error = None

    if query_text:
        try:
            body = _personalized_search_body(query_text, mode, profile_terms)
            response = es.search(index=INDEX_NAME, body=body)
            results = response["hits"]["hits"]
            total_hits = response["hits"]["total"]["value"]
            search_time_ms = response.get("took", 0)
            if user:
                _log_search(user["user_id"], query_text, mode)

        except Exception as exc:
            error = str(exc)

    bm25 = _get_bm25_params()

    return render_template(
        "search.html",
        query=query_text,
        results=results,
        total_hits=total_hits,
        search_time_ms=search_time_ms,
        error=error,
        index_name=INDEX_NAME,
        bm25=bm25,
        mode=mode,
        user=user,
        all_users=_all_users(),
        profile_terms=profile_terms,
    )


# TODO saved user ID and keep track of users ?
@app.route("/click/<doc_id>", methods=["GET"])
def track_click(doc_id):
    query_text = request.args.get("q", "").strip()
    rank = request.args.get("rank", type=int)
    mode = request.args.get("mode", "boosted")
    user = _current_user()

    _append_jsonl(
        CLICK_LOG,
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "user_id": user["user_id"] if user else None,
            "doc_id": doc_id,
            "query": query_text,
            "rank": rank,
            "mode": mode,
        },
    )

    return redirect(url_for("document", doc_id=doc_id, q=query_text, mode=mode))


@app.route("/doc/<doc_id>", methods=["GET"])
def document(doc_id):
    query_text = request.args.get("q", "").strip()
    mode = request.args.get("mode", "boosted")
    user = _current_user()

    try:
        response = es.get(index=INDEX_NAME, id=doc_id)
    except NotFoundError:
        abort(404)

    return render_template(
        "document.html",
        doc=response["_source"],
        doc_id=doc_id,
        query=query_text,
        mode=mode,
        user=user,
    )


if __name__ == "__main__":
    app.run(debug=True)
