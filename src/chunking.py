from typing import List
from src.ingest import Block


def fixed_chunk(blocks: List[Block], max_words: int = 150):
    chunks = []
    current_chunk = []
    word_count = 0
    chunk_id = 0

    for b in blocks:
        words = b.text.split()

        if word_count + len(words) > max_words and current_chunk:
            chunks.append({
                "chunk_id": chunk_id,
                "text": "\n".join(current_chunk)
            })
            chunk_id += 1
            current_chunk = []
            word_count = 0

        current_chunk.append(b.text)
        word_count += len(words)

    if current_chunk:
        chunks.append({
            "chunk_id": chunk_id,
            "text": "\n".join(current_chunk)
        })

    return chunks


def dynamic_chunk(blocks: List[Block]):
    """
    Group text under headings.
    Each heading starts a new chunk.
    """
    chunks = []
    current_chunk = []
    chunk_id = 0

    for b in blocks:
        if b.type == "heading":
            if current_chunk:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": "\n".join(current_chunk)
                })
                chunk_id += 1
                current_chunk = []

        current_chunk.append(b.text)

    if current_chunk:
        chunks.append({
            "chunk_id": chunk_id,
            "text": "\n".join(current_chunk)
        })

    return chunks


def intelligent_chunk(blocks: List[Block]):
    heading_count = sum(1 for b in blocks if b.type == "heading")

    if heading_count > 0:
        strategy = "dynamic"
        chunks = dynamic_chunk(blocks)
    else:
        strategy = "fixed"
        chunks = fixed_chunk(blocks)

    return strategy, chunks

