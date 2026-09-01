"""Busca semântica nos chunks."""

import math

from src.config import TOP_K
from src.embedding.embedder import gerar_embedding
from src.models import ChunkEmbedado, DocumentoEmbedado, ResultadoBusca


def _similaridade_cosseno(vetor_a: list[float], vetor_b: list[float]) -> float:
    produto = sum(a * b for a, b in zip(vetor_a, vetor_b))
    norma_a = math.sqrt(sum(a * a for a in vetor_a))
    norma_b = math.sqrt(sum(b * b for b in vetor_b))
    if norma_a == 0 or norma_b == 0:
        return 0.0
    return produto / (norma_a * norma_b)


def buscar(
    pergunta: str,
    documentos: list[DocumentoEmbedado],
    top_k: int = TOP_K,
) -> list[ResultadoBusca]:
    vetor_pergunta = gerar_embedding(pergunta)
    chunks: list[ChunkEmbedado] = []
    for documento in documentos:
        chunks.extend(documento.chunks_embedados)

    resultados = []
    for chunk in chunks:
        similaridade = _similaridade_cosseno(vetor_pergunta, chunk.vetor)
        resultados.append(
            ResultadoBusca(
                pergunta=pergunta,
                chunk_embedado=chunk,
                similaridade=similaridade,
            )
        )

    resultados.sort(key=lambda item: item.similaridade, reverse=True)
    return resultados[:top_k]
