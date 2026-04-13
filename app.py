import json
from datetime import datetime, UTC
from pathlib import Path
import uuid

from flask import Flask, abort, redirect, render_template, request, url_for, flash, session
from elasticsearch import Elasticsearch


app = Flask(__name__)
app.secret_key = "supersecret"
es = Elasticsearch("http://localhost:9200")
INDEX_NAME = "wiki_index"
CLICK_LOG = Path("click_log.jsonl")
USERS_DIR = Path("users")
USERS_DIR.mkdir(exist_ok=True)  # creates folder if it doesn't exist

@app.route("/create_user", methods=["GET"])
def create_user():
    user_id = str(uuid.uuid4())
    user_profile = {
        "user_id": user_id,
        "created_at": datetime.now().isoformat(),
        "search_history": [],
        "preferences": {}
    }

    user_file = USERS_DIR / f"{user_id}.json"  # pathlib way to join paths
    user_file.write_text(json.dumps(user_profile, indent=2))  # write JSON

    flash(f"User created! Your ID is: {user_id}")
    return redirect(url_for("search"))  # back to search page

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":
        user_id = request.form.get("user_id", "").strip()
        user_file = USERS_DIR / f"{user_id}.json"

        if user_file.exists():
            session["user_id"] = user_id
            flash(f"Logged in as {user_id}")
            return redirect(url_for("search"))
        else:
            flash("User ID not found. Please try again.")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user_id", None)
    flash("Logged out.")
    return redirect(url_for("search"))

@app.route("/", methods=["GET"])
def search():

    query_text = request.args.get("q", "").strip()
    results = []
    error = None
    user_id, user_profile = load_user_profile()

    if user_profile is None:
        flash("Your user profile was deleted. Please log in again.")
        return redirect(url_for("login"))
    
    if user_id is None:
        # allow search but no personalization
        user_profile = {}

    # Build Elasticsearch "should" clauses
    should_clauses = []

    if query_text:
        should_clauses.append({"multi_match": {"query": query_text, "fields": ["title^2","content"]}})

        # Self-set interests
        for interest in user_profile.get("self_interests", []):
            should_clauses.append({"multi_match": {"query": interest, "fields": ["title^2","content"], "boost": 2}})

        # History-based interests
        for interest in user_profile.get("history_interests", []):
            should_clauses.append({"multi_match": {"query": interest, "fields": ["title^2","content"], "boost": 1.8}})

        # Recent queries
        for recent in user_profile.get("recent_queries", [])[-5:]:
            should_clauses.append({"multi_match": {"query": recent["query"], "fields": ["title","content"], "boost": 1.5}})

        # Recent clicked documents (boost by time spent)
        for click in user_profile.get("recent_clicks", [])[-5:]:
            try:
                doc_resp = es.get(index=INDEX_NAME, id=click["doc_id"])
                doc_terms = doc_resp["_source"]["title"] + " " + doc_resp["_source"]["content"][:200]  # top content snippet
                boost_val = 1 + click.get("time_spent", 1)/10
                should_clauses.append({"multi_match": {"query": doc_terms, "fields": ["title","content"], "boost": boost_val}})
            except Exception:
                continue

        # Final query
        es_query = {"bool": {"should": should_clauses}}

        try:
            response = es.search(index=INDEX_NAME, query=es_query, size=10)
            results = response["hits"]["hits"]
        except Exception as exc:
            error = str(exc)

    return render_template(
        "search.html",
        query=query_text,
        results=results,
        error=error,
        index_name=INDEX_NAME,
    )

@app.route("/click/<doc_id>", methods=["GET"])
def track_click(doc_id):
    query_text = request.args.get("q", "").strip()
    rank = request.args.get("rank", type=int)

    user_id, user_profile = load_user_profile()

    # Deleted user
    if user_profile is None:
        flash("Your user profile was deleted. Please log in again.")
        return redirect(url_for("login"))

    # Not logged in → skip personalization
    if user_id is None:
        return redirect(url_for("document", doc_id=doc_id, q=query_text))

    user_file = USERS_DIR / f"{user_id}.json"

    click_entry = {
        "timestamp": datetime.now().isoformat(),
        "doc_id": doc_id,
        "query": query_text,
        "rank": rank,
        "time_spent": 1
    }

    user_profile.setdefault("search_history", []).append(click_entry)
    user_profile.setdefault("recent_clicks", []).append(click_entry)
    user_profile.setdefault("recent_queries", []).append({
        "query": query_text,
        "timestamp": datetime.now().isoformat()
    })

    user_profile["recent_clicks"] = user_profile["recent_clicks"][-5:]
    user_profile["recent_queries"] = user_profile["recent_queries"][-5:]

    # Update interests
    clicked_doc_ids = [c["doc_id"] for c in user_profile["search_history"][-20:]]

    if clicked_doc_ids:
        try:
            agg_query = {
                "size": 0,
                "query": {"ids": {"values": clicked_doc_ids}},
                "aggs": {
                    "top_terms": {
                        "terms": {"field": "content", "size": 10}
                    }
                }
            }
            resp = es.search(index=INDEX_NAME, body=agg_query)
            user_profile["history_interests"] = [
                b["key"] for b in resp["aggregations"]["top_terms"]["buckets"]
            ]
        except Exception:
            pass

    user_file.write_text(json.dumps(user_profile, indent=2))

    return redirect(url_for("document", doc_id=doc_id, q=query_text))

@app.route("/doc/<doc_id>", methods=["GET"])
def document(doc_id):
    query_text = request.args.get("q", "").strip()

    try:
        response = es.get(index=INDEX_NAME, id=doc_id)
    except Exception:
        abort(404)

    return render_template(
        "document.html",
        doc=response["_source"],
        doc_id=doc_id,
        query=query_text,
    )

def load_user_profile():
    user_id = session.get("user_id")

    if not user_id:
        return None, {}

    user_file = USERS_DIR / f"{user_id}.json"

    if not user_file.exists():
        session.pop("user_id", None)
        return None, None  # signal invalid user

    user_profile = json.loads(user_file.read_text())
    return user_id, user_profile


if __name__ == "__main__":
    app.run(debug=True)
