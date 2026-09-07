import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Pastas do pipeline
PASTA_ORIGINAIS = BASE_DIR / "data" / "raw"
PASTA_PROCESSADOS = BASE_DIR / "data" / "processed"

# Chunking hierárquico (por tokens)
CHUNK_MAX_TOKENS = 350
CHUNK_OVERLAP_TOKENS = 40

# Embeddings (modelo local, sem API)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Busca
TOP_K = 3

# LLM — Google Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-3.6-flash")

# Banco vetorial — Postgres + pgvector
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:rag@localhost:5433/rag")
USAR_PGVECTOR = os.getenv("USAR_PGVECTOR", "false").lower() in ("1", "true", "sim", "yes")
