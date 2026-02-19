import os
import gradio as gr

from src.ingest import ingest_file
from src.normalize_ar import has_diacritics, normalize_ar_for_search
from src.chunking import intelligent_chunk
from src.embeddings import embed_texts, embed_query
from src.storage_vector import upsert_chunks, query_chunks
from src.storage_sql import init_db, upsert_document, upsert_chunk


def run(file_obj, do_index, query, top_k):
    """
    Gradio callback:
    - If indexing is enabled and a file is provided: parse, create chunks, embed then store (Vector , SQL)
    - If query is provided: embed then only retrieve top-k from Vector DB
    """

    # Reset outputs each run (prevents repeated text problem)
    info_lines = []
    chunk_preview = ""
    results_text = ""

    # index (usually keep it checked)
    if do_index and file_obj is not None:
        filepath = file_obj.name
        filename = os.path.basename(filepath)
        doc_id = filename

        blocks = ingest_file(filepath)
        raw_text = "\n".join(b.text for b in blocks)

        # Choose fixed vs dynamic chunking based on structure
        strategy, chunks = intelligent_chunk(blocks)
        chunk_texts = [c["text"] for c in chunks]

        # Embed chunks for semantic search
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

        # Store in Vector DB (Chroma)
        upsert_chunks(chunk_ids, vectors.tolist(), metadatas, chunk_texts)

        # Store metadata in SQL (SQLite)
        filetype = os.path.splitext(filename)[1].lower()
        upsert_document(doc_id=doc_id, filename=filename, filetype=filetype)

        for c in chunks:
            chunk_uid = f"{doc_id}::chunk_{c['chunk_id']}"
            preview = c["text"][:300]
            upsert_chunk(
                chunk_uid=chunk_uid,
                doc_id=doc_id,
                chunk_index=int(c["chunk_id"]),
                strategy=strategy,
                has_diacritics=has_diacritics(c["text"]),
                char_count=len(c["text"]),
                preview=preview,
            )

        info_lines.append(f"Indexed: {filename}")
        info_lines.append(f"Strategy: {strategy.upper()}")
        info_lines.append(f"Chunks: {len(chunks)}")
        info_lines.append(f"Has harakat: {has_diacritics(raw_text)}")

        # Show only a small preview so the UI stays readable, here chose 2 chunks
        for c in chunks[:2]:
            chunk_preview += f"\n\n--- Chunk {c['chunk_id']} ---\n{c['text']}\n"

    elif file_obj is None:
        info_lines.append("No file uploaded (index skipped).")
    else:
        info_lines.append("Index unchecked (skipping indexing).")

    # Search
    if query and query.strip():
        # Normalize Arabic for retrieval (diacritics-insensitive search)
        q_norm = normalize_ar_for_search(query)
        q_vec = embed_query(q_norm)

        res = query_chunks(q_vec.tolist(), top_k=int(top_k))

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        if not docs:
            results_text = "No results returned."
        else:
            results_text = f"Top {len(docs)} results:\n"
            for i, (text, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
                results_text += (
                    f"\n[{i}] doc={meta.get('doc_id')} "
                    f"chunk={meta.get('chunk_id')} "
                    f"strategy={meta.get('strategy')} "
                    f"dist={float(dist):.4f}\n"
                    f"{text[:600]}\n"
                    "-----------------\n"
                )
    else:
        results_text = "No query provided."

    return "\n".join(info_lines), chunk_preview.strip(), results_text.strip()


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.File(label="Upload PDF/DOCX/TXT (optional)", file_types=[".pdf", ".docx", ".doc", ".txt"]),
        gr.Checkbox(label="Index file into DB", value=True),
        gr.Textbox(label="Query (optional)", placeholder="مثال: الأمن السيبراني / cybersecurity"),
        gr.Slider(1, 10, value=5, step=1, label="Top K"),
    ],
    outputs=[
        gr.Textbox(label="Info", lines=6),
        gr.Textbox(label="Chunk preview (first 2)", lines=10),
        gr.Textbox(label="Search results", lines=18),
    ],
    title="AI Document Parser ",
)


if __name__ == "__main__":
    # Initialize SQL tables once at startup
    init_db()
    demo.launch()


 

