"""Builders compartilhados pelos testes (evitam Keras/modelos reais)."""

from pathlib import Path

from src.models import (
    Chunk,
    ChunkEmbedado,
    DocumentoChunkado,
    DocumentoEmbedado,
    DocumentoProcessado,
    ResultadoBusca,
)


def make_processado(nome="doc.txt", bruto="texto bruto", limpo="texto limpo"):
    return DocumentoProcessado(
        arquivo_original=Path(nome),
        texto_bruto=bruto,
        texto_limpo=limpo,
        caminho_salvo=Path("/tmp") / f"{Path(nome).stem}.txt",
    )


def make_chunkado(nome="doc.txt", textos=("parte um", "parte dois")):
    doc = make_processado(nome)
    chunks = [Chunk(indice=i, texto=t, fonte=nome) for i, t in enumerate(textos)]
    return DocumentoChunkado(documento=doc, chunks=chunks)


def make_embedado(nome="doc.txt", textos=("parte um", "parte dois"), vetor=(1.0, 0.0)):
    chunkado = make_chunkado(nome, textos)
    embedados = [ChunkEmbedado(chunk=c, vetor=list(vetor)) for c in chunkado.chunks]
    return DocumentoEmbedado(documento_chunkado=chunkado, chunks_embedados=embedados)


def make_resultado(pergunta="pergunta?", fonte="doc.txt", texto="trecho", similaridade=0.9):
    chunk = Chunk(indice=0, texto=texto, fonte=fonte)
    return ResultadoBusca(
        pergunta=pergunta,
        chunk_embedado=ChunkEmbedado(chunk=chunk, vetor=[1.0]),
        similaridade=similaridade,
    )
