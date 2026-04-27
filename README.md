# Search Project

## Setup

Create and activate a Python environment:

```bash
conda create -n search_project python=3.11 -y
conda activate search_project
```

Install the packages used by the project:

```bash
pip install flask elasticsearch pandas pyarrow numpy
```

Project files required for running:

- `wiki_local.parquet`
- `templates/`

## Elastic Cloud

This project currently connects to Elastic Cloud with hardcoded credentials.

Connection values are defined in:

- [app.py](/Users/nina/Desktop/uni/search-engines/project/app.py:22)
- [test-elastic.py](/Users/nina/Desktop/uni/search-engines/project/test-elastic.py:17)
- [elasticAPI.py](/Users/nina/Desktop/uni/search-engines/project/elasticAPI.py:2)

Each file contains:

- the Cloud endpoint URL
- the API key

If the key expires:

1. Go to Elastic Cloud and open your deployment.
2. Copy the current Elasticsearch endpoint.
3. Create a new API key.
4. Replace the old endpoint and API key in the files above.
5. Run the indexing script again, ```python test-elastic.py --num-docs 100000```(to index 100,000 documents)
6. 
If you get auth or connection errors, check:

- wrong or expired API key
- wrong Cloud endpoint
- deployment is paused or deleted

## Indexing

Index a small test set:

```bash
python test-elastic.py --num-docs 1000
```

Index 10,000 documents:

```bash
python test-elastic.py --num-docs 10000
```

Index 100,000 documents:

```bash
python test-elastic.py --num-docs 100000
```

Index the full parquet file:

```bash
python test-elastic.py --num-docs 0
```

What the script does:

- loads `wiki_local.parquet`
- recreates the `wiki_index` index
- indexes the selected documents into Elasticsearch
- runs a small test search at the end

## Run The App

Start the Flask UI:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:43210
```

## Optional Endpoints

Manual history sync:

```bash
curl -X POST "http://127.0.0.1:43210/api/sync-history" \
  -H "Content-Type: application/json" \
  -d '{"profile_id":"default"}'
```

User pages:

- `/users`
- `/profile?profile_id=default`

## Notes

- `test-elastic.py` deletes and recreates `wiki_index` before indexing.
- `app.py` expects the main index name to be `wiki_index`.
- `docker-compose.yml` exists, but the current Python files are wired to Elastic Cloud, not to the local Docker Elasticsearch service.
