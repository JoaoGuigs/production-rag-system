"""Índice vetorial persistente no Postgres + pgvector.

Tabelas:
  documentos — fonte da verdade (nome + hash = reindex só do que mudou)
  chunks     — um chunk por linha (texto + vetor + metadados)
"""

import hashlib
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from src.config import DATABASE_URL, dimensao_embedding
from src.models import Chunk, ChunkEmbedado, ResultadoBusca


def _schema_sql(dims: int) -> str:
    return f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documentos (
  id               SERIAL PRIMARY KEY,
  nome             TEXT NOT NULL,
  hash_sha256      TEXT NOT NULL,
  modelo_embedding TEXT NOT NULL,
  n_chunks         INT NOT NULL DEFAULT 0,
  indexado_em      TIMESTAMPTZ DEFAULT now(),
  UNIQUE (nome, hash_sha256)
);

CREATE TABLE IF NOT EXISTS chunks (
  id           SERIAL PRIMARY KEY,
  documento_id INT NOT NULL REFERENCES documentos(id) ON DELETE CASCADE,
  fonte        TEXT NOT NULL,
  indice       INT NOT NULL,
  texto        TEXT NOT NULL,
  embedding    vector({dims}) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
  ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_fonte_idx ON chunks (fonte);
"""


def conectar():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def inicializar() -> None:
    dims = dimensao_embedding()  # segue o EMBEDDING_MODEL atual (env)
    with conectar() as conn:
        conn.execute(_schema_sql(dims))
        conn.commit()
        _migrar_dimensao(conn, dims)


def _migrar_dimensao(conn, dims: int) -> None:
    """Troca de modelo = outra dimensão: limpa o índice antigo e ajusta a coluna.

    Vetor de um modelo não serve para outro, então não há o que preservar —
    a indexação incremental re-embeda tudo sozinha (o hash inclui o modelo).
    """
    linha = conn.execute(
        "SELECT atttypmod FROM pg_attribute "
        "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
    ).fetchone()
    if linha and linha["atttypmod"] != dims:  # vector(n) guarda typmod = n
        conn.execute("DELETE FROM documentos")
        conn.execute("DROP INDEX IF EXISTS chunks_embedding_hnsw")
        conn.execute(f"ALTER TABLE chunks ALTER COLUMN embedding TYPE vector({dims})")
        conn.execute(
            "CREATE INDEX chunks_embedding_hnsw "
            "ON chunks USING hnsw (embedding vector_cosine_ops)"
        )
        conn.commit()


def hash_arquivo(caminho: Path) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def doc_indexado(nome: str, hash_sha256: str, modelo: str) -> bool:
    with conectar() as conn:
        linha = conn.execute(
            "SELECT 1 FROM documentos WHERE nome = %s AND hash_sha256 = %s AND modelo_embedding = %s",
            (nome, hash_sha256, modelo),
        ).fetchone()
        return linha is not None


def salvar_documento(
    nome: str,
    hash_sha256: str,
    modelo: str,
    chunks: list[Chunk],
    vetores: list[list[float]],
) -> int:
    """Grava (ou regrava) um documento e seus chunks. Retorna nº de chunks."""
    with conectar() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documentos WHERE nome = %s", (nome,))
            cur.execute(
                "INSERT INTO documentos (nome, hash_sha256, modelo_embedding, n_chunks)"
                " VALUES (%s, %s, %s, %s) RETURNING id",
                (nome, hash_sha256, modelo, len(chunks)),
            )
            doc_id = cur.fetchone()["id"]
            cur.executemany(
                "INSERT INTO chunks (documento_id, fonte, indice, texto, embedding)"
                " VALUES (%s, %s, %s, %s, %s::vector)",
                [
                    (doc_id, chunk.fonte, chunk.indice, chunk.texto, _vetor_literal(vetor))
                    for chunk, vetor in zip(chunks, vetores)
                ],
            )
        conn.commit()
    return len(chunks)


def buscar_vetorial(
    vetor_pergunta: list[float],
    top_k: int = 3,
    fontes: list[str] | None = None,
) -> list[ResultadoBusca]:
    with conectar() as conn:
        linhas = conn.execute(
            "SELECT fonte, indice, texto, 1 - (embedding <=> %s::vector) AS similaridade"
            " FROM chunks"
            " WHERE (%s::text[] IS NULL OR fonte = ANY(%s))"
            " ORDER BY embedding <=> %s::vector"
            " LIMIT %s",
            (_vetor_literal(vetor_pergunta), fontes, fontes, _vetor_literal(vetor_pergunta), top_k),
        ).fetchall()
    return [
        ResultadoBusca(
            pergunta="",
            chunk_embedado=ChunkEmbedado(
                chunk=Chunk(indice=linha["indice"], texto=linha["texto"], fonte=linha["fonte"]),
                vetor=[],
            ),
            similaridade=float(linha["similaridade"]),
        )
        for linha in linhas
    ]


def contar() -> dict:
    with conectar() as conn:
        docs = conn.execute("SELECT COUNT(*) AS n FROM documentos").fetchone()["n"]
        chunks = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
    return {"documentos": docs, "chunks": chunks}


def _vetor_literal(vetor: list[float]) -> str:
    return "[" + ",".join(str(v) for v in vetor) + "]"
