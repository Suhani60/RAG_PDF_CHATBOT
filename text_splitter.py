"""
text_splitter.py
----------------
split page text into small, overlapping chunks.
"""

from dataclasses import dataclass
from typing import List

from pdf_processor import PageText

DEFAULT_CHUNK_SIZE = 800      # max characters per chunk (~150-200 words)
DEFAULT_CHUNK_OVERLAP = 150   # characters shared between neighbouring chunks


@dataclass
class Chunk:
    """A small piece of text with its origin (used for page references)."""
    chunk_id: int
    text: str
    page_number: int
    source: str


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """Split one string into overlapping pieces, breaking at spaces, not mid-word."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size.")

    text = text.strip()
    if not text:
        return []

    pieces: List[str] = []
    n = len(text)
    start = 0

    while start < n:
        end = min(start + chunk_size, n)

        # If we are not at the end, back up to the last space so we don't cut a word.
        if end < n:
            last_space = text.rfind(" ", start + chunk_size // 2, end)
            if last_space != -1:
                end = last_space

        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)

        if end >= n:
            break

        # Next chunk starts a little earlier (the overlap), but always moves forward.
        start = max(end - chunk_overlap, start + 1)

        # If we landed in the middle of a word, move to the next word start.
        if text[start - 1] != " ":
            next_space = text.find(" ", start, end)
            if next_space != -1:
                start = next_space + 1

    return pieces


def split_pages(
    pages: List[PageText],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Chunk]:
    """Split every page separately so each chunk knows its exact page number."""
    chunks: List[Chunk] = []
    next_id = 0
    for page in pages:
        for piece in split_text(page.text, chunk_size, chunk_overlap):
            chunks.append(
                Chunk(
                    chunk_id=next_id,
                    text=piece,
                    page_number=page.page_number,
                    source=page.source,
                )
            )
            next_id += 1
    return chunks