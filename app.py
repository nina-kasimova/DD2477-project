import json
from datetime import datetime, UTC
from pathlib import Path

from flask import Flask, abort, redirect, render_template, request, url_for
from elasticsearch import Elasticsearch, NotFoundError


app = Flask(__name__)
es = Elasticsearch("http://localhost:9200")
INDEX_NAME = "wiki_index"
CLICK_LOG = Path("click_log.jsonl")

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


@app.route("/", methods=["GET"])
def search():
    query_text = request.args.get("q", "").strip()
    mode = request.args.get("mode", "boosted")
    results = []
    total_hits = 0
    search_time_ms = 0
    error = None

    if query_text:
        try:
            body = _build_search_body(query_text, mode)
            response = es.search(index=INDEX_NAME, body=body)
            results = response["hits"]["hits"]
            total_hits = response["hits"]["total"]["value"]
            search_time_ms = response.get("took", 0)

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
    except NotFoundError:
        abort(404)

    return render_template(
        "document.html",
        doc=response["_source"],
        doc_id=doc_id,
        query=query_text,
    )


if __name__ == "__main__":
    app.run(debug=True)
