import streamlit as st
import tempfile

from backend import process_pdfs, rag_simple

# ================== UI ==================
st.set_page_config(page_title="PDF RAG", layout="wide")
st.title("📄 Multi-PDF RAG Q&A")

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ================== UPLOAD ==================
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# ================== PROCESS ==================
if st.button("Process PDFs"):
    if not uploaded_files:
        st.warning("Upload at least one PDF")
    else:
        file_paths = []

        with st.spinner("Processing PDFs..."):
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.read())
                    file_paths.append(tmp.name)

            try:
                retriever = process_pdfs(file_paths)
                st.session_state.retriever = retriever
                st.success("✅ Processing Complete!")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ================== QUERY ==================
query = st.text_input("Ask a question")

if st.button("Get Answer"):
    if not st.session_state.retriever:
        st.warning("Process PDFs first")
    elif not query:
        st.warning("Enter a question")
    else:
        with st.spinner("Thinking..."):
            answer = rag_simple(query, st.session_state.retriever)

        st.subheader("Answer")
        st.write(answer)

# ================== DEBUG ==================
if st.checkbox("Show Retrieved Chunks"):
    if st.session_state.retriever and query:
        results = st.session_state.retriever.retrieve(query)

        for i, r in enumerate(results):
            st.write(f"Chunk {i+1}")
            st.write(r["content"])
            st.divider()