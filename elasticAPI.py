from elasticsearch import Elasticsearch, helpers
client = Elasticsearch(
    "https://my-elasticsearch-project-b7af22.es.us-central1.gcp.elastic.cloud:443",
    api_key="RDdDRmdwMEI0dlNRLXQyRWhtRVY6UmpFeDNmRGV0dXB4Wk4yb0Fwc25TZw=="
)
# -----------------------------
# Define the index name and create index (if it doesn't exist)
# -----------------------------
index_name = "national-parks"
if not client.indices.exists(index=index_name):
    create_response = client.indices.create(index=index_name)
    print("Index created:", create_response)
else:
    print(f"Index '{index_name}' already exists.")
# -----------------------------
# Add or update mappings for the index
# -----------------------------
mappings = {
    "properties": {
        "text": {
            "type": "semantic_text"
        }
    }
}
mapping_response = client.indices.put_mapping(
    index=index_name,
    body=mappings
)
print("Mappings updated:", mapping_response)
# -----------------------------
# Sample documents to ingest
# -----------------------------
docs = [
    {
        "text": "Yellowstone National Park is one of the largest national parks in the United States. It ranges from the Wyoming to Montana and Idaho, and contains an area of 2,219,791 acres across three different states. Its most famous for hosting the geyser Old Faithful and is centered on the Yellowstone Caldera, the largest super volcano on the American continent. Yellowstone is host to hundreds of species of animal, many of which are endangered or threatened. Most notably, it contains free-ranging herds of bison and elk, alongside bears, cougars and wolves. The national park receives over 4.5 million visitors annually and is a UNESCO World Heritage Site."
    },
    {
        "text": "Yosemite National Park is a United States National Park, covering over 750,000 acres of land in California. A UNESCO World Heritage Site, the park is best known for its granite cliffs, waterfalls and giant sequoia trees. Yosemite hosts over four million visitors in most years, with a peak of five million visitors in 2016. The park is home to a diverse range of wildlife, including mule deer, black bears, and the endangered Sierra Nevada bighorn sheep. The park has 1,200 square miles of wilderness, and is a popular destination for rock climbers, with over 3,000 feet of vertical granite to climb. Its most famous and cliff is the El Capitan, a 3,000 feet monolith along its tallest face."
    },
    {
        "text": "Rocky Mountain National Park is one of the most popular national parks in the United States. It receives over 4.5 million visitors annually, and is known for its mountainous terrain, including Longs Peak, which is the highest peak in the park. The park is home to a variety of wildlife, including elk, mule deer, moose, and bighorn sheep. The park is also home to a variety of ecosystems, including montane, subalpine, and alpine tundra. The park is a popular destination for hiking, camping, and wildlife viewing, and is a UNESCO World Heritage Site."
    }
]
# -----------------------------
# Bulk ingest documents
# -----------------------------
ingestion_timeout=300 # Allow time for semantic ML model to load
bulk_response = helpers.bulk(
    client.options(request_timeout=ingestion_timeout),
    docs,
    index=index_name,
    refresh="wait_for" # Wait until indexed documents are visible for search before returning the response
)
print(bulk_response)
# -----------------------------
# Define semantic search query
# -----------------------------
retriever_object = {
    "standard": {
        "query": {
            "semantic": {
                "field": "text",
                "query": "Sierra Nevada"
            }
        }
    }
}
search_response = client.search(
    index=index_name,
    retriever=retriever_object,
)
print(search_response['hits']['hits'])