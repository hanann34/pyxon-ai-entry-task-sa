import os
from typing import Any, Dict, List

from src.ingest import ingest_file
from src.chunking import intelligent_chunk
from src.normalize_ar import normalize_ar_for_search, has_diacritics
from src.embeddings import embed_texts, embed_query
from src.storage_vector import upsert_chunks, query_chunks
from src.storage_sql import upsert_document, upsert_chunk


def index_file_to_stores(filepath: str) -> Dict[str, Any]:
    """
    End-to-end indexing for RAG:
    file turn into text blocks then seperate into chunks
    embeddings then store them (Vector DB + SQL).
    """
    filename = os.path.basename(filepath)
    doc_id = filename

    blocks = ingest_file(filepath)
    raw_text = "\n".join(b.text for b in blocks)

    strategy, chunks = intelligent_chunk(blocks)
    chunk_texts = [c["text"] for c in chunks]

    vectors = embed_texts(chunk_texts)

    chunk_ids = [f"{doc_id}::chunk_{c['chunk_id']}" for c in chunks]
    metadatas = [
        {
            "doc_id": doc_id,
            "chunk_id": c["chunk_id"],
            "strategy": strategy,
            "has_diacritics": has_diacritics(c["text"]),
        }
        for c in chunks
    ]

    # Vector DB: text , embeddings , metadata for semantic retrieval
    upsert_chunks(chunk_ids, vectors.tolist(), metadatas, chunk_texts)

    # SQL DB: structured metadata which is the doc + chunk summaries
    filetype = os.path.splitext(filename)[1].lower()
    upsert_document(doc_id=doc_id, filename=filename, filetype=filetype)

    for c in chunks:
        chunk_uid = f"{doc_id}::chunk_{c['chunk_id']}"
        upsert_chunk(
            chunk_uid=chunk_uid,
            doc_id=doc_id,
            chunk_index=int(c["chunk_id"]),
            strategy=strategy,
            has_diacritics=has_diacritics(c["text"]),
            char_count=len(c["text"]),
            preview=c["text"][:300],
        )

    return {
        "doc_id": doc_id,
        "filename": filename,
        "strategy": strategy,
        "num_chunks": len(chunks),
        "has_harakat": has_diacritics(raw_text),
    }


def retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Semantic retrieval:
    normalize query then embed then top-k from vector DB 
    """
    q_norm = normalize_ar_for_search(query)
    q_vec = embed_query(q_norm)

    res = query_chunks(q_vec.tolist(), top_k=int(top_k))

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    results: List[Dict[str, Any]] = []
    for text, meta, dist in zip(docs, metas, dists):
        results.append(
            {
                "text": text,
                "meta": meta,
                "distance": float(dist),
            }
        )
    return results

