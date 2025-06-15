# type: ignore
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# === Load environment variables ===
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# === Paths ===
DATA_PATH = "data"  # Folder containing PDFs
CHROMA_PATH = "chroma_db"

# === Initialize Embeddings Model ===
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)

# === Load and split PDF documents ===
print("📄 Loading documents from:", DATA_PATH)
loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = loader.load()

print(f"✅ Loaded {len(raw_documents)} documents.")

# === Text Splitting ===
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len
)

chunks = text_splitter.split_documents(raw_documents)
print(f"✂️ Split into {len(chunks)} text chunks.")

# === Create Vector Store ===
vector_store = Chroma(
    collection_name="mental_health_self_help",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH
)

# === Add chunks to vector store with unique IDs ===
uuids = [str(uuid4()) for _ in range(len(chunks))]
vector_store.add_documents(documents=chunks, ids=uuids)
print("✅ Vectorization complete. Embeddings stored to:", CHROMA_PATH)
