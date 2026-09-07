import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Pastas do pipeline
PASTA_ORIGINAIS = BASE_DIR / "data" / "raw"
PASTA_PROCESSADOS = BASE_DIR / "data" / "processed"

# Chunking hierárquico (por tokens) — override via env p/ experimentos
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "500"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "75"))

# Embeddings (modelo local, sem API) — override via env p/ experimentos
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# Dimensão do vetor de cada modelo conhecido (precisa bater com a coluna no pgvector)
EMBEDDING_DIMS = {
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
    "BAAI/bge-m3": 1024,
}


def dimensao_embedding(modelo: str | None = None) -> int:
    modelo = modelo or EMBEDDING_MODEL
    try:
        return EMBEDDING_DIMS[modelo]
    except KeyError:
        raise ValueError(
            f"Modelo '{modelo}' sem dimensão mapeada. "
            f"Adicione em EMBEDDING_DIMS no config.py. Conhecidos: {sorted(EMBEDDING_DIMS)}"
        ) from None

# Busca
TOP_K = 3

# LLM — Google Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-3.1-flash-lite")

# Banco vetorial — Postgres + pgvector
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:rag@localhost:5433/rag")
USAR_PGVECTOR = os.getenv("USAR_PGVECTOR", "false").lower() in ("1", "true", "sim", "yes")
