"""Basic ChromaDB usage: store raw text and query it by meaning.

No embedding model is passed, so Chroma embeds the text itself with its default
local model (all-MiniLM-L6-v2). No API key is needed.
"""
import chromadb

# In-memory client: data is lost when the script ends.
chromadb_client = chromadb.Client()

collection_name = "my_collection"
collection = chromadb_client.get_or_create_collection(collection_name)

documents = [
    {"id": "doc1", "content": "hello world hi hw are you?"},
    {"id": "doc2", "content": "This is the second document about how to get ai job"},
    {"id": "doc3", "content": "I want to become an android developer and a successful trader."},
]

# upsert inserts new ids and overwrites existing ones, so re-running is safe.
for doc in documents:
    collection.upsert(
        ids=[doc["id"]],
        documents=[doc["content"]],
    )

query = "what do i want to be?"

# The query text is embedded and compared against stored vectors.
results = collection.query(
    query_texts=[query],
    n_results=2
)

print(results)

