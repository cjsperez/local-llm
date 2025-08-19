from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

db = Chroma(
    collection_name="simple-rag",
    embedding_function=OllamaEmbeddings(
        model="nomic-embed-text:latest", 
        base_url="http://localhost:4000"
    ),
    persist_directory="./chroma_db"
)

docs = db.similarity_search("sample query", k=3)
for i, doc in enumerate(docs):
    print(f"\n--- Doc {i+1} ---\n{doc.page_content}")
