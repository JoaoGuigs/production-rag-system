"""Estado em memória dos documentos indexados (usado pela API web)."""

from src.generation.llm import gerar_resposta, llm_configurada
from src.models import DocumentoEmbedado, RespostaRAG, ResultadoBusca
from src.pipeline import executar_pipeline, listar_arquivos_originais
from src.retrieval.search import buscar

_documentos: list[DocumentoEmbedado] = []


def indexar() -> dict:
    global _documentos
    _documentos = executar_pipeline()
    total_chunks = sum(len(doc.chunks_embedados) for doc in _documentos)
    return {"documentos": len(_documentos), "chunks": total_chunks}


def perguntar(texto: str) -> list[ResultadoBusca]:
    if not _documentos:
        raise ValueError("Nenhum documento indexado. Envie arquivos e clique em indexar.")
    return buscar(texto, _documentos)


def responder(texto: str) -> RespostaRAG:
    chunks = perguntar(texto)
    resposta = gerar_resposta(texto, chunks)
    return RespostaRAG(pergunta=texto, resposta=resposta, chunks=chunks)


def obter_status() -> dict:
    total_chunks = sum(len(doc.chunks_embedados) for doc in _documentos)
    return {
        "indexado": len(_documentos) > 0,
        "documentos": len(_documentos),
        "chunks": total_chunks,
        "llm_configurada": llm_configurada(),
        "arquivos": [arquivo.name for arquivo in listar_arquivos_originais()],
    }
