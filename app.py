"""
app.py
------
Streamlit user interface. Run with:  streamlit run app.py
"""

import io

import streamlit as st

from embeddings import EmbeddingError, EmbeddingModel
from pdf_processor import PDFProcessingError, extract_pages
from rag_pipeline import RAGError, RAGPipeline, api_key_configured, get_model_name

st.set_page_config(page_title="PDF Q&A Assistant", page_icon="📚", layout="wide")

CUSTOM_CSS = """
<style>
.hero {padding: 1.5rem 2rem; border-radius: 16px; margin-bottom: 1.2rem;
       background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 55%, #0ea5e9 100%);}
.hero h1 {margin: 0; font-size: 2rem; color: #ffffff !important;}
.hero p {margin: .4rem 0 0 0; color: #ffffff; opacity: .93; font-size: 1.02rem;}
.badge {display: inline-block; padding: .15rem .65rem; margin: .7rem .35rem 0 0;
        border-radius: 999px; font-size: .78rem; color: #ffffff;
        background: rgba(255,255,255,.2);}
</style>
"""

HERO_HTML = """
<div class="hero">
  <h1>📚 AI PDF Question Answering Assistant</h1>
  <p>Upload a PDF, ask questions in plain English, and get answers grounded in the
     document, with page references.</p>
  <span class="badge">RAG</span><span class="badge">Sentence Transformers</span>
  <span class="badge">FAISS</span><span class="badge">Google Gemini</span>
</div>
"""


# ------------------------------------------------------------------ state helpers
def init_state() -> None:
    st.session_state.setdefault("messages", [])        # chat history
    st.session_state.setdefault("pipeline", None)      # RAGPipeline once a PDF is processed
    st.session_state.setdefault("doc_summary", None)   # stats shown in the sidebar
    st.session_state.setdefault("doc_signature", None) # which files were processed


def clear_chat() -> None:
    st.session_state.messages = []


@st.cache_resource(show_spinner=False)
def load_embedder() -> EmbeddingModel:
    """Load the embedding model once and reuse it (it is slow to load)."""
    return EmbeddingModel()


# ------------------------------------------------------------------ processing
def process_documents(uploaded_files) -> None:
    """Extract -> chunk -> embed -> index, with live status messages."""
    all_pages = []

    with st.status("Processing your PDF(s)...", expanded=True) as status:
        st.write("📖 Step 1/3: Extracting text from pages...")
        for f in uploaded_files:
            try:
                pages = extract_pages(io.BytesIO(f.getvalue()), f.name)
                all_pages.extend(pages)
                st.write(f"✅ {f.name}: {len(pages)} page(s) with text")
            except PDFProcessingError as exc:
                st.warning(f"⚠️ Skipped {f.name}: {exc}")

        if not all_pages:
            status.update(label="No readable text found", state="error")
            st.error("None of the uploaded files contained readable text.")
            return

        try:
            st.write("🧠 Step 2/3: Loading the embedding model (first run downloads it)...")
            embedder = load_embedder()
            st.write("🗂️ Step 3/3: Creating chunks, embeddings and the FAISS index...")
            pipeline = RAGPipeline(embedder)
            summary = pipeline.index_documents(all_pages)
        except (EmbeddingError, RAGError) as exc:
            status.update(label="Processing failed", state="error")
            st.error(str(exc))
            return
        except Exception as exc:  # last safety net
            status.update(label="Processing failed", state="error")
            st.error(f"Unexpected error: {exc}")
            return

        status.update(label="PDF processed successfully!", state="complete", expanded=False)

    st.session_state.pipeline = pipeline
    st.session_state.doc_summary = summary
    st.session_state.doc_signature = tuple((f.name, f.size) for f in uploaded_files)
    st.session_state.messages = []   # new documents -> start a fresh conversation
    st.success(
        f"✅ Ready! Indexed {summary['documents']} document(s), "
        f"{summary['pages']} pages, {summary['chunks']} chunks. Ask your questions below."
    )


# ------------------------------------------------------------------ rendering
def render_sources(sources) -> None:
    """Show page references and the retrieved passages."""
    if not sources:
        return
    pages_by_doc = {}
    for s in sources:
        pages_by_doc.setdefault(s["source"], set()).add(s["page"])
    parts = [
        f"{doc} - p. {', '.join(str(p) for p in sorted(pages))}"
        for doc, pages in pages_by_doc.items()
    ]
    st.caption("📄 **Sources:** " + "  |  ".join(parts))

    with st.expander("View retrieved passages"):
        for s in sources:
            st.markdown(f"**{s['source']} · page {s['page']}** · similarity `{s['score']:.2f}`")
            st.write(s["text"])
            st.divider()


def render_sidebar():
    with st.sidebar:
        st.header("📂 Upload documents")
        uploaded_files = st.file_uploader(
            "Choose one or more PDF files", type=["pdf"], accept_multiple_files=True
        )
        process_clicked = st.button(
            "⚙️ Process PDF", type="primary", width="stretch", disabled=not uploaded_files
        )

        if uploaded_files and st.session_state.pipeline is not None:
            signature = tuple((f.name, f.size) for f in uploaded_files)
            if signature != st.session_state.doc_signature:
                st.warning("Files changed. Click **Process PDF** to update.")

        st.subheader("📊 Status")
        summary = st.session_state.doc_summary
        if st.session_state.pipeline is not None and summary:
            st.success(
                f"Ready: {summary['documents']} doc(s) · {summary['pages']} pages · "
                f"{summary['chunks']} chunks"
            )
        else:
            st.info("No document processed yet.")

        if api_key_configured():
            st.caption(f"🔑 API key found · Model: `{get_model_name()}`")
        else:
            st.error("GEMINI_API_KEY not found. Add it to `.env`, then restart the app.")

        with st.expander("⚙️ Settings"):
            top_k = st.slider("Passages retrieved per question", 1, 8, 4, key="top_k")

        st.button(
            "🗑️ Clear chat",
            on_click=clear_chat,
            width="stretch",
            disabled=not st.session_state.messages,
        )
    return uploaded_files, process_clicked, top_k


def handle_question(pipeline: RAGPipeline, question: str) -> None:
    history = list(st.session_state.messages)   # previous turns only
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the document and asking Gemini..."):
                result = pipeline.ask(question, history=history)
            st.markdown(result.answer)
            render_sources(result.sources)
            content, sources = result.answer, result.sources
        except RAGError as exc:
            content, sources = f"⚠️ {exc}", []
            st.error(str(exc))
        except Exception as exc:
            content, sources = f"⚠️ Unexpected error: {exc}", []
            st.error(content)

    st.session_state.messages.append({"role": "assistant", "content": content, "sources": sources})


# ------------------------------------------------------------------ main
def main() -> None:
    init_state()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(HERO_HTML, unsafe_allow_html=True)

    uploaded_files, process_clicked, top_k = render_sidebar()

    if process_clicked and uploaded_files:
        process_documents(uploaded_files)

    pipeline = st.session_state.pipeline
    if pipeline is None:
        st.info("👈 Upload a PDF in the sidebar and click **Process PDF** to begin.")
    else:
        pipeline.top_k = top_k

    for msg in st.session_state.messages:      # conversation history
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            render_sources(msg.get("sources"))

    question = st.chat_input("Ask a question about your PDF...", disabled=pipeline is None)
    if question and pipeline is not None:
        handle_question(pipeline, question)


main()