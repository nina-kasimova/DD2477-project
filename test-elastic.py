import pandas as pd
from elasticsearch import Elasticsearch, helpers

es = Elasticsearch("http://localhost:9200")
INDEX_NAME = "wiki_index"
NUM_DOCS = 1000

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
            "category": {
                "type": "keyword",
            },
        }
    },
}


def doc_generator(df, index_name):
    for _, row in df.iterrows():
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

        yield {
            "_index": index_name,
            "_source": {
                "title": title,
                "content": content,
                "category": "Wikipedia Article",
            },
        }
df = pd.read_parquet("wiki_local.parquet")
print(f"Loaded parquet: {len(df)} rows, columns: {df.columns.tolist()}")

if NUM_DOCS > 0:
    df = df.head(NUM_DOCS)
    print(f"Using first {NUM_DOCS} documents")
else:
    print(f"Using ALL {len(df)} documents")

if es.indices.exists(index=INDEX_NAME):
    es.indices.delete(index=INDEX_NAME)
    print(f"Deleted existing index '{INDEX_NAME}'")

es.indices.create(index=INDEX_NAME, body=INDEX_SETTINGS)
print(f"Created index '{INDEX_NAME}' with BM25 (k1={BM25_K1}, b={BM25_B})")

print("Starting indexing...")
success, errors = helpers.bulk(es, doc_generator(df, INDEX_NAME), stats_only=True)
print(f"Done - indexed {success} documents, {errors} errors")

es.indices.refresh(index=INDEX_NAME)

response = es.search(
    index=INDEX_NAME,
    query={"match": {"content": "science"}},
    size=5,
)

print(f"\n--- Test search for 'science' ({response['hits']['total']['value']} hits) ---")
for hit in response["hits"]["hits"]:
    print(f"  {hit['_score']:.4f}  {hit['_source']['title'][:80]}")

settings = es.indices.get_settings(index=INDEX_NAME)
similarity = (
    settings[INDEX_NAME]["settings"]["index"]
    .get("similarity", {})
    .get("custom_bm25", {})
)
print(f"\nIndex similarity settings: {similarity}")
