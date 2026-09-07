"""Chunking de janela fixa — corta a cada N tokens, sem respeitar estrutura.

Serve de baseline burra contra o chunking hierárquico do chunker.py.
"""

from src.config import CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS


def dividir_em_chunks_fixo(
    texto: str,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[str]:
    """Janela deslizante por caracteres (~4 chars = 1 token, mesma aproximação)."""
    if max_tokens is None:
        max_tokens = CHUNK_MAX_TOKENS
    if overlap_tokens is None:
        overlap_tokens = CHUNK_OVERLAP_TOKENS
    if not texto.strip():
        return []

    tamanho = max_tokens * 4
    passo = max(1, (max_tokens - overlap_tokens) * 4)
    chunks = []
    for inicio in range(0, len(texto), passo):
        pedaco = texto[inicio : inicio + tamanho].strip()
        if pedaco:
            chunks.append(pedaco)
    return chunks
