"""
Store and search embeddings using ChromaDB (local vector database).
"""

# Disable Chroma telemetry for error i was getting
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

import chromadb
from chromadb.config import Settings

COLLECTION_NAME = "chunks"

_client = None


def get_chroma_client():
    """
    Create a single Chroma client instance.
    """
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=".chroma",
            settings=Settings(anonymized_telemetry=False)
        )
        
    return _client


def get_collection():
    """Get or create the collection."""
    client = get_chroma_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)


def upsert_chunks(chunk_ids: list[str], embeddings: list[list[float]], metadatas: list[dict], documents: list[str]):
    """
    Insert or update chunk data in Chroma.
    """
    col = get_collection()
    col.upsert(
        ids=chunk_ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )


def query_chunks(query_embedding: list[float], top_k: int = 5):
    """
    Retrieve the top_k most similar chunks.
    """
    col = get_collection()
    return col.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
