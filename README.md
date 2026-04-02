**🧠 Step-by-Step Explanation
1. 📂 PDF Upload**

Users upload one or more PDF files through the Streamlit interface.

**2. 📖 Document Loading**
PDFs are read using PyPDFLoader
Each page is converted into structured text documents

**3. ✂️ Text Chunking**
Large documents are split into smaller chunks
Helps improve retrieval accuracy and embedding quality

**5. 🔢 Embedding Generation**
Each text chunk is converted into vector embeddings
Done using Sentence Transformers (all-MiniLM-L6-v2)

**7. 🗄️ Vector Storage**
Embeddings are stored in ChromaDB
Enables fast similarity-based search

**9. 🔍 Retrieval**
User query is converted into an embedding
Top relevant chunks are retrieved using similarity search

**11. 🤖 LLM Response Generation**
Retrieved context + user query is passed to Groq LLM (LLaMA 3)
LLM generates a contextual and accurate answer

**🛠️ Tech Stack**
Component	Technology
Frontend	Streamlit
Backend	Python
LLM	Groq (LLaMA 3)
Embeddings	Sentence Transformers
Vector DB	ChromaDB
Framework	LangChain

**🚀 Key Features**
📄 Upload multiple PDFs
🔍 Semantic search across documents
🧠 Context-aware answers using LLM
⚡ Fast retrieval with vector database
🖥️ Interactive UI using Streamlit
🧪 Debug mode to view retrieved chunks

**📊 Example Use Cases**
📚 Research paper Q&A
🏢 Enterprise document search
📑 Legal/financial document analysis
📘 Knowledge base assistant
