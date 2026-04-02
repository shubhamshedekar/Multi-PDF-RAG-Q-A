import os
import uuid
import numpy as np
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# ================== SPLIT ==================
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

# ================== EMBEDDINGS ==================
class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts):
        return self.model.encode(texts)

# ================== VECTOR STORE ==================
class VectorStore:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name="pdf_docs")

    def add_documents(self, documents, embeddings):
        ids, docs, metas, embs = [], [], [], []

        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            ids.append(f"id_{i}_{uuid.uuid4().hex[:6]}")
            docs.append(doc.page_content)
            metas.append(doc.metadata)
            embs.append(emb.tolist())

        self.collection.add(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embs
        )

# ================== RETRIEVER ==================
class RAGRetriever:
    def __init__(self, vectorstore, embedding_manager):
        self.vectorstore = vectorstore
        self.embedding_manager = embedding_manager

    def retrieve(self, query, top_k=3):
        query_emb = self.embedding_manager.generate_embeddings([query])[0]

        results = self.vectorstore.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k
        )

        docs = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            docs.append({
                "content": doc,
                "metadata": meta
            })

        return docs

# ================== LLM ==================
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.1
)

# ================== RAG ==================
def rag_simple(query, retriever):
    results = retriever.retrieve(query)
    context = "\n\n".join([doc["content"] for doc in results])

    prompt = f"""
    Use the following context to answer the question.

    Context:
    {context}

    Question: {query}
    Answer:
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# ================== PIPELINE ==================
def process_pdfs(file_paths: List[str]):
    all_docs = []

    for path in file_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = os.path.basename(path)

        all_docs.extend(docs)

    if not all_docs:
        raise ValueError("No documents loaded")

    chunks = split_documents(all_docs)

    texts = [doc.page_content for doc in chunks if doc.page_content.strip()]

    if not texts:
        raise ValueError("No valid text found")

    embed_manager = EmbeddingManager()
    embeddings = embed_manager.generate_embeddings(texts)

    if len(embeddings) == 0:
        raise ValueError("Embeddings are empty")

    vectorstore = VectorStore()
    vectorstore.add_documents(chunks, embeddings)

    retriever = RAGRetriever(vectorstore, embed_manager)

    return retriever