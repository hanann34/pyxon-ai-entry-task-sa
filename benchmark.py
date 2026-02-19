"""
Small benchmark:
- Recall@K , basic indexing time
"""

import os
import time

from src.ingest import ingest_file
from src.chunking import intelligent_chunk
from src.embeddings import embed_texts, embed_query
from src.storage_vector import upsert_chunks, query_chunks
from src.normalize_ar import normalize_ar_for_search

TEST_DOC = "sampletest.docx"


def benchmark_recall_at_k(k: int = 5) -> None:
    print("Running Recall@K benchmark...")

    blocks = ingest_file(TEST_DOC)
    strategy, chunks = intelligent_chunk(blocks)

    chunk_texts = [c["text"] for c in chunks]
    vectors = embed_texts(chunk_texts)

    doc_id = "benchmark_doc"
    chunk_ids = [f"{doc_id}::chunk_{c['chunk_id']}" for c in chunks]
    metadatas = [
        {"doc_id": doc_id, "chunk_id": c["chunk_id"], "strategy": strategy}
        for c in chunks
    ]

    upsert_chunks(chunk_ids, vectors.tolist(), metadatas, chunk_texts)

    test_queries = ["title","question" ,"inquiry"]

    correct_hits = 0
    for q in test_queries:
        q_norm = normalize_ar_for_search(q)
        q_vec = embed_query(q_norm)

        res = query_chunks(q_vec.tolist(), top_k=k)
        metas = res.get("metadatas", [[]])[0]

        # Count as correct if we retrieved at least one chunk from our indexed doc
        if any(m.get("doc_id") == doc_id for m in metas):
            correct_hits += 1

    recall = correct_hits / len(test_queries)
    print(f"Recall@{k}: {recall:.2f}")


def benchmark_speed() -> None:
    print("Running speed benchmark...")
    start = time.time()

    blocks = ingest_file(TEST_DOC)
    _strategy, chunks = intelligent_chunk(blocks)
    chunk_texts = [c["text"] for c in chunks]
    _ = embed_texts(chunk_texts)

    end = time.time()
    print(f"Indexing time: {end - start:.2f} seconds")


if __name__ == "__main__":
    if not os.path.exists(TEST_DOC):
        print(f"Test file not found: {TEST_DOC}")
        print("Put a DOCX file with this name in the project folder, then run again.")
    else:
        benchmark_recall_at_k(k=5)
        benchmark_speed()

