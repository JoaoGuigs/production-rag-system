"""Gera e aplica embeddings nos chunks."""

from sentence_transformers import SentenceTransformer

from src import config as cfg
from src.models import ChunkEmbedado, DocumentoChunkado, DocumentoEmbedado

_modelo: SentenceTransformer | None = None
_modelo_nome: str | None = None


def _obter_modelo() -> SentenceTransformer:
    global _modelo, _modelo_nome
    if _modelo is None or _modelo_nome != cfg.EMBEDDING_MODEL:
        print(f"Carregando modelo de embeddings: {cfg.EMBEDDING_MODEL}")
        _modelo = SentenceTransformer(cfg.EMBEDDING_MODEL)
        _modelo_nome = cfg.EMBEDDING_MODEL
    return _modelo


def gerar_embeddings(textos: list[str]) -> list[list[float]]:
    if not textos:
        return []
    modelo = _obter_modelo()
    vetores = modelo.encode(textos, show_progress_bar=len(textos) > 5)
    return vetores.tolist()


def gerar_embedding(texto: str) -> list[float]:
    return gerar_embeddings([texto])[0]


def embedar_documento(documento: DocumentoChunkado) -> DocumentoEmbedado:
    textos = [chunk.texto for chunk in documento.chunks]
    vetores = gerar_embeddings(textos)
    chunks_embedados = [
        ChunkEmbedado(chunk=chunk, vetor=vetor)
        for chunk, vetor in zip(documento.chunks, vetores)
    ]
    return DocumentoEmbedado(
        documento_chunkado=documento,
        chunks_embedados=chunks_embedados,
    )


def embedar_todos(documentos: list[DocumentoChunkado]) -> list[DocumentoEmbedado]:
    return [embedar_documento(documento) for documento in documentos]
