from flask import Flask, abort, redirect, render_template, request, session, url_for
from elasticsearch import NotFoundError

from config import INDEX_NAME, es
from data_store import all_users, clear_clicks_for_user, current_user, log_click, log_search, set_user_profile, update_bio, user_activity
from search_logic import build_user_profile, get_bm25_params, personalized_search_body


app = Flask(__name__)
app.secret_key = "toy-search-secret"


@app.route("/set-user", methods=["POST"])
def set_user():
    user_id = request.form.get("user_id", "").strip()
    bio = request.form.get("bio", "").strip()
    mode = request.form.get("mode", "boosted")
    personalize = request.form.get("personalize", "1")
    query_text = request.form.get("q", "").strip()
    set_user_profile(session, user_id, bio)

    return redirect(url_for("search", q=query_text, mode=mode, personalize=personalize))


@app.route("/profile", methods=["GET"])
def profile():
    user = current_user(session)
    if not user:
        return redirect(url_for("search"))

    searches, clicks = user_activity(user["user_id"])
    return render_template(
        "profile.html",
        user=user,
        searches=searches[:20],
        clicks=clicks[:20],
        all_users=all_users(),
    )


@app.route("/profile/update", methods=["POST"])
def update_profile():
    user = current_user(session)
    if not user:
        return redirect(url_for("search"))

    bio = request.form.get("bio", "").strip()
    update_bio(user["user_id"], bio)
    return redirect(url_for("profile"))


@app.route("/profile/clear-clicks", methods=["POST"])
def clear_clicks():
    user = current_user(session)
    if not user:
        return redirect(url_for("search"))

    clear_clicks_for_user(user["user_id"])
    return redirect(url_for("profile"))


@app.route("/", methods=["GET"])
def search():
    query_text = request.args.get("q", "").strip()
    mode = request.args.get("mode", "boosted")
    personalize = request.args.get("personalize", "1") == "1"
    user = current_user(session)
    profile = build_user_profile(user) if personalize else []
    results = []
    total_hits = 0
    error = None

    if query_text:
        try:
            body = personalized_search_body(query_text, mode, profile)
            response = es.search(index=INDEX_NAME, body=body)
            results = response["hits"]["hits"]
            total_hits = response["hits"]["total"]["value"]
            if user:
                log_search(user["user_id"], query_text, mode)

        except Exception as exc:
            error = str(exc)

    bm25 = get_bm25_params()

    return render_template(
        "search.html",
        query=query_text,
        results=results,
        total_hits=total_hits,
        error=error,
        index_name=INDEX_NAME,
        bm25=bm25,
        mode=mode,
        personalize=personalize,
        user=user,
        all_users=all_users(),
        profile=profile,
    )


@app.route("/browse", methods=["GET"])
def browse():
    user = current_user(session)
    error = None
    articles = []

    try:
        response = es.search(
            index=INDEX_NAME,
            body={
                "query": {
                    "function_score": {
                        "query": {"match_all": {}},
                        "random_score": {},
                    }
                },
                "size": 50,
            },
        )
        articles = response["hits"]["hits"]
    except Exception as exc:
        error = str(exc)

    return render_template(
        "browse.html",
        user=user,
        articles=articles,
        error=error,
    )


# TODO saved user ID and keep track of users ?
@app.route("/click/<doc_id>", methods=["GET"])
def track_click(doc_id):
    query_text = request.args.get("q", "").strip()
    rank = request.args.get("rank", type=int)
    mode = request.args.get("mode", "boosted")
    personalize = request.args.get("personalize", "1")
    user = current_user(session)
    log_click(
        user["user_id"] if user else None,
        doc_id,
        query_text,
        rank,
        mode,
    )

    return redirect(url_for("document", doc_id=doc_id, q=query_text, mode=mode, personalize=personalize))


@app.route("/doc/<doc_id>", methods=["GET"])
def document(doc_id):
    query_text = request.args.get("q", "").strip()
    mode = request.args.get("mode", "boosted")
    personalize = request.args.get("personalize", "1")
    user = current_user(session)

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
        personalize=personalize,
        user=user,
    )


if __name__ == "__main__":
    app.run(debug=True)
