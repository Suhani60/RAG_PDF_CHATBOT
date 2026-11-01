"""
Usage:
    python test_components.py path/to/file.pdf "your question"
    python test_components.py path/to/file.pdf "your question" --generate
"""

import sys

import numpy as np

from embeddings import EmbeddingModel
from pdf_processor import extract_pages
from rag_pipeline import RAGPipeline
from text_splitter import DEFAULT_CHUNK_SIZE, split_pages, split_text
from vector_store import VectorStore


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    pdf_path, question = sys.argv[1], sys.argv[2]
    generate = "--generate" in sys.argv

    print("\n[1] PDF extraction")
    with open(pdf_path, "rb") as f:
        pages = extract_pages(f, pdf_path)
    assert pages, "No pages extracted"
    print(f"    Pages with text: {len(pages)}")
    print(f"    Page {pages[0].page_number} preview: {pages[0].text[:150]!r}")

    print("\n[2] Text chunking")
    chunks = split_pages(pages)
    longest = max(len(c.text) for c in chunks)
    print(f"    Chunks: {len(chunks)} | longest: {longest} chars | first chunk page: {chunks[0].page_number}")
    assert longest <= DEFAULT_CHUNK_SIZE
    sample = split_text("word " * 500, 100, 20)
    assert all(len(s) <= 100 for s in sample)
    print(f"    Synthetic chunking test passed ({len(sample)} chunks)")

    print("\n[3] Embedding generation")
    embedder = EmbeddingModel()
    vectors = embedder.embed_texts([c.text for c in chunks])
    print(f"    Embedding matrix shape: {vectors.shape}")
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3), "Vectors not normalized"
    trio = embedder.embed_texts([
        "The cat sits on the sofa.",
        "A kitten is resting on the couch.",
        "Stock markets fell sharply today.",
    ])
    related, unrelated = float(trio[0] @ trio[1]), float(trio[0] @ trio[2])
    print(f"    similar sentences: {related:.2f} | unrelated sentences: {unrelated:.2f}")
    assert related > unrelated

    print("\n[4] FAISS similarity search")
    store = VectorStore()
    store.build(vectors, chunks)
    self_hit = store.search(vectors[0:1], top_k=1)[0]
    assert self_hit.chunk.chunk_id == 0 and self_hit.score > 0.99
    print("    Self-retrieval check passed")
    for rank, hit in enumerate(store.search(embedder.embed_query(question), top_k=3), start=1):
        print(f"    #{rank} score={hit.score:.2f} page={hit.chunk.page_number}: {hit.chunk.text[:100]!r}")

    print("\n[5] Final answer generation")
    if generate:
        pipeline = RAGPipeline(embedder)
        pipeline.index_documents(pages)
        result = pipeline.ask(question)
        print(f"    Found in document: {result.found}")
        print(f"    Answer: {result.answer}")
        print(f"    Pages: {sorted({s['page'] for s in result.sources})}")
    else:
        print("    Skipped (add --generate to call Gemini)")

    print("\nAll checks completed.")


if __name__ == "__main__":
    main()