The system converts the PDF into searchable meaning-vectors once, then for each question it retrieves the most similar passages with FAISS and gives them to Gemini, which writes an answer based only on that evidence. 

**Tech stack:** Python, Streamlit, PyPDF, Sentence Transformers, FAISS, Google Gemini API

**Folder Structure**: 
PDF_RAG_Chatbot/
├── app.py               # Streamlit UI (the only file you run)
├── pdf_processor.py     # PDF → text per page
├── text_splitter.py     # text → overlapping chunks
├── embeddings.py        # chunks/questions → vectors
├── vector_store.py      # FAISS index + similarity search
├── rag_pipeline.py      # connects everything + calls Gemini
├── test_components.py   # optional: tests each stage from the terminal
├── requirements.txt     # library versions
├── .env                 # your secret API key (never share/commit)
└── .gitignore           # keeps .env and venv out of Git