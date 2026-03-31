import pandas as pd
from elasticsearch import Elasticsearch, helpers

es = Elasticsearch("http://localhost:9200")


def doc_generator(df, index_name):
    for _, row in df.iterrows():
        full_text = row['text']

        # split by the first newline to find a title,
        # or just take the first 60 characters if not
        parts = full_text.split('\n', 1)
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
                "category": "Wikipedia Article"
            }
        }

# the original dataset as a test
# df = pd.read_parquet("hf://datasets/rahular/simple-wikipedia/data/train-00000-of-00001-090b52ccb189d47a.parquet")
# df.to_parquet("wiki_local.parquet")

df = pd.read_parquet("wiki_local.parquet")

print("columns in file ", df.columns.tolist())


# only index 1000 rows so it's instant
df_sample = df.head(1000)

INDEX_NAME = "wiki_index"

# avoid duplciating index
if es.indices.exists(index=INDEX_NAME):
    es.indices.delete(index=INDEX_NAME)

es.indices.create(index=INDEX_NAME)

print("Starting indexing...")
helpers.bulk(es, doc_generator(df_sample, INDEX_NAME))
print("Done")


# ---------------------------------
#test search for the word 'april' (just cos it shows up in the dataset)
query = {
    "match": {
        "content": "science"
    }
}

response = es.search(index="wiki_index", query=query, size=5)

print("\n--- Search Results for 'april' ---")
for hit in response['hits']['hits']:
    print(f"Title: {hit['_source']['title']} (Score: {hit['_score']})")