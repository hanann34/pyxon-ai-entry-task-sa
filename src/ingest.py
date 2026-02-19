import os
from dataclasses import dataclass
from typing import List

from pypdf import PdfReader
from docx import Document


@dataclass
class Block:
    # "heading" or "paragraph"
    type: str
    text: str


def _looks_like_heading(text: str) -> bool:
    """Lightweight heading heuristics for Arabic + English."""
    t = text.strip()
    if not t:
        return False

    return (
        t.startswith("#")
        or t.endswith(":")
        or t.lower().startswith("heading")
        or t.startswith("العنوان")
        or t.startswith("الفصل")
        or t.startswith("المبحث")
    )


def read_txt(path: str) -> List[Block]:
    """Read TXT and return simple blocks (heading/paragraph) using heuristics."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.rstrip() for line in f]

    blocks: List[Block] = []
    for line in lines:
        t = line.strip()
        if not t:
            continue

        blocks.append(Block(type="heading" if _looks_like_heading(t) else "paragraph", text=t))

    return blocks


def read_pdf(path: str) -> List[Block]:
    """Read PDF and return paragraph blocks (per line)."""
    reader = PdfReader(path)
    blocks: List[Block] = []

    for page in reader.pages:
        text = page.extract_text() or ""
        for line in text.split("\n"):
            t = line.strip()
            if t:
                blocks.append(Block(type="paragraph", text=t))

    return blocks


def read_docx(path: str) -> List[Block]:
    """Read DOCX and return blocks using Word heading styles + heuristics."""
    doc = Document(path)
    blocks: List[Block] = []

    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if not t:
            continue

        style_name = (p.style.name or "").lower()
        is_heading = "heading" in style_name or _looks_like_heading(t)

        blocks.append(Block(type="heading" if is_heading else "paragraph", text=t))

    return blocks


def ingest_file(path: str) -> List[Block]:
    """Dispatch based on file extension and return extracted blocks."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".txt":
        return read_txt(path)
    if ext == ".pdf":
        return read_pdf(path)
    if ext in (".docx", ".doc"):
        return read_docx(path)

    raise ValueError(f"Unsupported file type: {ext}")


