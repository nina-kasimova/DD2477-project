import json
from datetime import datetime, UTC
from pathlib import Path

from flask import Flask, abort, redirect, render_template, request, url_for
from elasticsearch import Elasticsearch


app = Flask(__name__)
es = Elasticsearch("http://localhost:9200")
INDEX_NAME = "wiki_index"
CLICK_LOG = Path("click_log.jsonl")


@app.route("/", methods=["GET"])
def search():
    query_text = request.args.get("q", "").strip()
    results = []
    error = None

    if query_text:
        try:
            # make title more important but need to add proper logic
            response = es.search(
                index=INDEX_NAME,
                query={
                    "multi_match": {
                        "query": query_text,
                        "fields": ["title^2", "content"],
                    }
                },
                size=10,
            )
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


# TODO saved user ID and keep track of users ?
@app.route("/click/<doc_id>", methods=["GET"])
def track_click(doc_id):
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


if __name__ == "__main__":
    app.run(debug=True)
