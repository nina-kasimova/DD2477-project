from pathlib import Path

from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

INDEX_NAME = "wiki_index"
CLICK_LOG = Path("click_log.jsonl")
SEARCH_LOG = Path("search_log.jsonl")
USER_PROFILES = Path("user_profiles.json")

BM25_K1 = 1.2
BM25_B = 0.75
PROFILE_WINDOW_DAYS = 7
MAX_CLICK_HISTORY = 10
MAX_PROFILE_TERMS = 8
BIO_TERM_WEIGHT = 3.0
CLICK_QUERY_TERM_WEIGHT = 1.0
PERSONALIZATION_TITLE_BOOST = 0.6
PERSONALIZATION_CONTENT_BOOST = 0.25
STOPWORDS = {
    "about", "after", "also", "and", "are", "because", "been", "between", "but",
    "for", "from", "had", "has", "have", "into", "its", "later", "more", "most",
    "not", "one", "only", "onto", "other", "over", "same", "some", "such", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "those", "through", "time", "under", "until", "very", "was", "were", "what",
    "when", "where", "which", "while", "with", "would", "your",
}
