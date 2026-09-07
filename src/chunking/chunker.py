"""
Chunking hierárquico — preserva estrutura do documento.

Pipeline:
  documento → headings → parágrafos → frases (se preciso) → limite de tokens → overlap
"""

import re

from src.config import CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS

_HEADING_RE = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)
_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÀ-Ú0-9\"'])")


def contar_tokens(texto: str) -> int:
    """Aproximação de tokens (~4 caracteres por token em PT/EN)."""
    return max(1, len(texto) // 4)


def _tail_por_tokens(texto: str, max_tokens: int) -> str:
    if contar_tokens(texto) <= max_tokens:
        return texto
    alvo = max_tokens * 4
    return texto[-alvo:].lstrip()


def _split_por_headings(texto: str) -> list[tuple[str, str]]:
    """Divide em seções por headings markdown (# Título)."""
    if not _HEADING_RE.search(texto):
        return [("", texto.strip())]

    partes = re.split(r"(?=^#{1,6}\s)", texto, flags=re.MULTILINE)
    secoes = []

    for parte in partes:
        parte = parte.strip()
        if not parte:
            continue
        linhas = parte.split("\n", 1)
        titulo = linhas[0].strip()
        corpo = linhas[1].strip() if len(linhas) > 1 else ""
        secoes.append((titulo, corpo))

    return secoes or [("", texto.strip())]


def _split_por_paragrafos(texto: str) -> list[str]:
    if not texto.strip():
        return []
    return [p.strip() for p in re.split(r"\n{2,}", texto) if p.strip()]


def _split_por_frases(paragrafo: str) -> list[str]:
    frases = _SENTENCE_RE.split(paragrafo.strip())
    return [f.strip() for f in frases if f.strip()]


def _split_por_caracteres(texto: str, max_tokens: int) -> list[str]:
    """Fallback quando uma frase ainda é grande demais."""
    max_chars = max_tokens * 4
    pedacos = []
    inicio = 0
    while inicio < len(texto):
        pedacos.append(texto[inicio : inicio + max_chars].strip())
        inicio += max_chars
    return [p for p in pedacos if p]


def _unidades_de_texto(texto: str, max_tokens: int | None = None) -> list[str]:
    """Aplica a hierarquia: headings → parágrafos → frases."""
    if max_tokens is None:
        max_tokens = CHUNK_MAX_TOKENS  # lido na chamada: permite override por experimento
    unidades: list[str] = []

    for titulo, corpo in _split_por_headings(texto):
        blocos = _split_por_paragrafos(corpo) if corpo else []

        if not blocos and titulo:
            unidades.append(titulo)
            continue

        for bloco in blocos:
            texto_bloco = f"{titulo}\n\n{bloco}" if titulo else bloco

            if contar_tokens(texto_bloco) <= max_tokens:
                unidades.append(texto_bloco)
                continue

            frases = _split_por_frases(bloco)
            if len(frases) <= 1:
                prefixo = f"{titulo}\n\n" if titulo else ""
                unidades.extend(_split_por_caracteres(prefixo + bloco, max_tokens))
                continue

            acumulo = f"{titulo}\n\n" if titulo else ""
            for frase in frases:
                candidato = f"{acumulo}{frase}".strip()
                if contar_tokens(frase) > max_tokens:
                    if acumulo.strip():
                        unidades.append(acumulo.strip())
                        acumulo = ""
                    unidades.extend(_split_por_caracteres(frase, max_tokens))
                elif contar_tokens(candidato) > max_tokens:
                    if acumulo.strip():
                        unidades.append(acumulo.strip())
                    acumulo = frase
                else:
                    acumulo = f"{candidato}\n" if acumulo else frase

            if acumulo.strip():
                unidades.append(acumulo.strip())

    return unidades


def _agrupar_unidades(unidades: list[str], max_tokens: int) -> list[str]:
    """Junta unidades pequenas até chegar perto do limite de tokens."""
    if not unidades:
        return []

    chunks: list[str] = []
    atual: list[str] = []
    tokens_atual = 0

    for unidade in unidades:
        tokens_unidade = contar_tokens(unidade)

        if tokens_unidade > max_tokens:
            if atual:
                chunks.append("\n\n".join(atual))
                atual, tokens_atual = [], 0
            chunks.append(unidade)
            continue

        if tokens_atual + tokens_unidade > max_tokens and atual:
            chunks.append("\n\n".join(atual))
            atual, tokens_atual = [unidade], tokens_unidade
        else:
            atual.append(unidade)
            tokens_atual += tokens_unidade

    if atual:
        chunks.append("\n\n".join(atual))

    return chunks


def _aplicar_overlap(chunks: list[str], overlap_tokens: int) -> list[str]:
    if len(chunks) <= 1 or overlap_tokens <= 0:
        return chunks

    resultado = [chunks[0]]
    for chunk in chunks[1:]:
        overlap = _tail_por_tokens(resultado[-1], overlap_tokens)
        if overlap and not chunk.startswith(overlap):
            resultado.append(f"{overlap}\n\n{chunk}")
        else:
            resultado.append(chunk)

    return resultado


def dividir_em_chunks(
    texto: str,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[str]:
    """
    Divide texto respeitando estrutura semântica.

    Por que essa ordem?
    - Headings: mantém título junto do conteúdo da seção
    - Parágrafos: não corta ideias no meio de um bloco
    - Frases: só quando o parágrafo ainda é grande demais
    - Tokens: limite compatível com embedding e LLM
    - Overlap: evita perder contexto na fronteira entre chunks
    """
    if not texto.strip():
        return []

    if max_tokens is None:
        max_tokens = CHUNK_MAX_TOKENS  # lido na chamada: permite override por experimento
    if overlap_tokens is None:
        overlap_tokens = CHUNK_OVERLAP_TOKENS
    unidades = _unidades_de_texto(texto, max_tokens)
    agrupados = _agrupar_unidades(unidades, max_tokens)
    return _aplicar_overlap(agrupados, overlap_tokens)


def chunkar_documento(documento, estrategia: str = "hierarquico") -> "DocumentoChunkado":
    from src.chunking.fixed import dividir_em_chunks_fixo
    from src.models import Chunk, DocumentoChunkado

    if estrategia == "fixo":
        pedacos = dividir_em_chunks_fixo(documento.texto_limpo, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS)
    else:
        pedacos = dividir_em_chunks(documento.texto_limpo)
    chunks = [
        Chunk(indice=i, texto=texto, fonte=documento.nome)
        for i, texto in enumerate(pedacos)
    ]
    return DocumentoChunkado(documento=documento, chunks=chunks)


def chunkar_todos(documentos: list, estrategia: str = "hierarquico") -> list:
    return [chunkar_documento(documento, estrategia) for documento in documentos]
