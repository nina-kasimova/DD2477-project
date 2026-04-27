"""
Index the Simple Wikipedia parquet dataset into Elasticsearch
with explicit BM25 similarity settings.

Usage:
    python test-elastic.py                  # index 1000 docs (default)
    python test-elastic.py --num-docs 5000  # index 5000 docs
    python test-elastic.py --num-docs 0     # index ALL docs
"""

import argparse
import sys

import pandas as pd
from elasticsearch import Elasticsearch, helpers

es = Elasticsearch(
    "https://my-elasticsearch-project-add782.es.us-central1.gcp.elastic.cloud:443",
    api_key="QWhXaXpwMEJOSEV0eXRtOG55VXM6a0FqdjJ6ODRIN3lxLXAwcXVtMmFvUQ=="
)

INDEX_NAME = "wiki_index"

# ── BM25 index configuration ────────────────────────────────────────────────
# k1 : controls term-frequency saturation (higher → tf matters more)
# b  : controls document-length normalization (0 = none, 1 = full)
BM25_K1 = 1.2
BM25_B = 0.75

INDEX_SETTINGS = {
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
            "category": {
                "type": "keyword",
            },
        }
    },
}


# ── document generator ──────────────────────────────────────────────────────
def doc_generator(df, index_name):
    for i, (_, row) in enumerate(df.iterrows()):
        full_text = row["text"]

        # split by the first newline to find a title,
        # or just take the first 60 characters if not
        parts = full_text.split("\n", 1)
        if len(parts) > 1:
            title = parts[0].strip()
            content = parts[1].strip()
        else:
            title = full_text[:60] + "..."
            content = full_text

        if i % 500 == 0:
            print(f"  prepared {i} documents …", file=sys.stderr)

        yield {
            "_index": index_name,
            "_source": {
                "title": title,
                "content": content,
                "content_semantic": content,
                "category": "Wikipedia Article",
            },
        }


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Index wiki data into Elasticsearch")
    parser.add_argument(
        "--num-docs",
        type=int,
        default=1000,
        help="Number of documents to index (0 = all). Default: 1000",
    )
    args = parser.parse_args()

    # ── load dataset ─────────────────────────────────────────────────────────
    df = pd.read_parquet("wiki_local.parquet")
    print(f"Loaded parquet: {len(df)} rows, columns: {df.columns.tolist()}")

    if args.num_docs > 0:
        df = df.head(args.num_docs)
        print(f"Using first {args.num_docs} documents")
    else:
        print(f"Using ALL {len(df)} documents")

    # ── (re)create index with explicit BM25 settings ─────────────────────────
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
        print(f"Deleted existing index '{INDEX_NAME}'")

    es.indices.create(index=INDEX_NAME, body=INDEX_SETTINGS)
    print(
        f"Created index '{INDEX_NAME}' with BM25 (k1={BM25_K1}, b={BM25_B})"
    )

    # ── bulk index ───────────────────────────────────────────────────────────
    print("Starting indexing …")
    success, errors = helpers.bulk(es, doc_generator(df, INDEX_NAME), stats_only=True)
    print(f"Done — indexed {success} documents, {errors} errors")

    # ── verify with a test search ────────────────────────────────────────────
    es.indices.refresh(index=INDEX_NAME)

    response = es.search(
        index=INDEX_NAME,
        query={"match": {"content": "science"}},
        size=5,
    )

    print(f"\n--- Test search for 'science' ({response['hits']['total']['value']} hits) ---")
    for hit in response["hits"]["hits"]:
        print(f"  {hit['_score']:.4f}  {hit['_source']['title'][:80]}")

    # ── print index settings to verify BM25 ──────────────────────────────────
    settings = es.indices.get_settings(index=INDEX_NAME)
    similarity = (
        settings[INDEX_NAME]["settings"]["index"]
        .get("similarity", {})
        .get("custom_bm25", {})
    )
    print(f"\nIndex similarity settings: {similarity}")


if __name__ == "__main__":
    main()