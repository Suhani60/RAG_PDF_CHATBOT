"""
pdf_processor.py
----------------
extract text from PDF files, page by page.
"""

import re
from dataclasses import dataclass
from typing import BinaryIO, List

from pypdf import PdfReader


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be read or contains no usable text."""


@dataclass
class PageText:
    """Text of a single PDF page, plus where it came from."""
    source: str        # file name, e.g. "report.pdf"
    page_number: int   # 1-based, matches what you see in a PDF viewer
    text: str


def clean_text(text: str) -> str:
    """Tidy raw extracted text so it embeds and searches better."""
    text = text.replace("\x00", " ")           # remove null characters
    text = re.sub(r"-\n(?=[a-z])", "", text)   # re-join words split by a hyphen at line end
    text = re.sub(r"\s+", " ", text)           # collapse newlines/tabs/multiple spaces
    return text.strip()


def extract_pages(file: BinaryIO, source_name: str = "document.pdf") -> List[PageText]:
    """
    Read a PDF (an open binary file or BytesIO) and return the text of every
    page that contains text.

    Raises PDFProcessingError with a friendly message if anything goes wrong.
    """
    try:
        reader = PdfReader(file)
    except Exception as exc:
        raise PDFProcessingError(
            f"'{source_name}' could not be opened. It may be corrupted or not a real PDF."
        ) from exc

    # Some PDFs are "encrypted" with an empty password. Try that; give up otherwise.
    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception:
            unlocked = 0
        if not unlocked:
            raise PDFProcessingError(
                f"'{source_name}' is password-protected. Please upload an unlocked copy."
            )

    pages: List[PageText] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            continue  # skip a damaged page instead of failing the whole file
        cleaned = clean_text(raw_text)
        if cleaned:
            pages.append(PageText(source=source_name, page_number=index, text=cleaned))

    if not pages:
        raise PDFProcessingError(
            f"No readable text found in '{source_name}'. It is probably a scanned "
            "(image-only) PDF, which needs OCR - not supported in this project."
        )
    return pages