from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import logging
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
)
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.node_parser import SentenceSplitter


from pinecone import Pinecone

# ---------------------------------------------------
# Logging
# ---------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# Load ENV
# ---------------------------------------------------


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# ---------------------------------------------------
# Validate ENV
# ---------------------------------------------------
required_vars = [
    GROQ_API_KEY, LLM_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    # PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME
]

if not all(required_vars):
    raise RuntimeError("❌ Missing required environment variables")

# ---------------------------------------------------
# Initialize LLM & Embeddings
# ---------------------------------------------------
# llm = Groq(
#     model=LLM_MODEL_NAME,
#     api_key=GROQ_API_KEY,
# )
llm = Groq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY
)

Settings.llm = llm

embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL_NAME
)

Settings.llm = llm
Settings.embed_model = embed_model

# ---------------------------------------------------
# Pinecone
# ---------------------------------------------------
pc = Pinecone(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV,
)

pinecone_index = pc.Index(PINECONE_INDEX_NAME)

# ---------------------------------------------------
# FastAPI
# ---------------------------------------------------
app = FastAPI(
    title="RAG API (Groq + Pinecone)",
    version="1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for demo/workshop only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------------------------------------------
# Request Schema
# ---------------------------------------------------
class QueryRequest(BaseModel):
    question: str

# ---------------------------------------------------
# Ingestion Endpoint
# ---------------------------------------------------
@app.post("/ingest")
def ingest_pdf():
    try:
        logger.info("📄 Loading PDF...")

        documents = SimpleDirectoryReader(
            input_files=["paul_graham.pdf"],
            filename_as_id=True
        ).load_data()

        logger.info("✂️ Chunking documents...")

        splitter = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50,
        )

        nodes = splitter.get_nodes_from_documents(documents)

        # ✅ Attach page-level metadata
        for node in nodes:
            node.metadata = node.metadata or {}
            node.metadata["source"] = "paul_graham.pdf"
            node.metadata["page_label"] = node.metadata.get("page_label", "N/A")

        logger.info(f"📦 Storing {len(nodes)} chunks with page metadata...")

        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
            namespace="paul_graham_pdf"
        )

        # ✅ Create LlamaIndex wrapper
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store
        )

        # ✅ ACTUAL write to Pinecone
        vector_index.insert_nodes(nodes)

        stats = pinecone_index.describe_index_stats()

        return {
            "status": "success",
            "chunks_ingested": len(nodes),
            "pages_indexed": len(set(n.metadata["page_label"] for n in nodes))
        }

    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))




# stats = index.describe_index_stats()
# print(stats)

# ---------------------------------------------------
# Query Endpoint
# ---------------------------------------------------
@app.post("/query")
def query_rag(request: QueryRequest):
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Empty question")
        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
            namespace="paul_graham_pdf"
        )
        index = VectorStoreIndex.from_vector_store(vector_store)

        query_engine = index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact",
        )

        response = query_engine.query(request.question)

        citations = []
        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                meta = node.node.metadata
                citations.append({
                    "page": meta.get("page_label", "N/A"),
                    "text": node.text[:300] + "..."
                })

        return {
            "question": request.question,
            "answer": str(response),
            "citations": citations
        }

    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))
