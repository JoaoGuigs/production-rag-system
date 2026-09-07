"""Guarda dos documentos indexados (memória ou pgvector, via USAR_PGVECTOR)."""

from src.chunking.chunker import chunkar_documento
from src.config import EMBEDDING_MODEL, TOP_K, USAR_PGVECTOR
from src.embedding.embedder import gerar_embedding, gerar_embeddings
from src.generation.llm import gerar_resposta, llm_configurada
from src.models import DocumentoEmbedado, RespostaRAG, ResultadoBusca
from src.pipeline import executar_pipeline, listar_arquivos_originais, processar_todos
from src.retrieval.search import buscar
from src.storage import vetores as pg

_documentos: list[DocumentoEmbedado] = []


def indexar() -> dict:
    if USAR_PGVECTOR:
        return _indexar_pg()
    global _documentos
    _documentos = executar_pipeline()
    total_chunks = sum(len(doc.chunks_embedados) for doc in _documentos)
    return {"documentos": len(_documentos), "chunks": total_chunks}


def _indexar_pg() -> dict:
    """Indexação incremental: só embeda o que mudou (hash) desde a última vez."""
    pg.inicializar()
    for documento in processar_todos():
        hash_atual = pg.hash_arquivo(documento.arquivo_original)
        if pg.doc_indexado(documento.nome, hash_atual, EMBEDDING_MODEL):
            continue
        chunkado = chunkar_documento(documento)
        vetores = gerar_embeddings([c.texto for c in chunkado.chunks])
        pg.salvar_documento(documento.nome, hash_atual, EMBEDDING_MODEL, chunkado.chunks, vetores)
    return pg.contar()


def perguntar(texto: str) -> list[ResultadoBusca]:
    if USAR_PGVECTOR:
        resultados = pg.buscar_vetorial(gerar_embedding(texto), TOP_K)
        if not resultados:
            raise ValueError("Nenhum documento indexado. Envie arquivos e clique em indexar.")
        return resultados
    if not _documentos:
        raise ValueError("Nenhum documento indexado. Envie arquivos e clique em indexar.")
    return buscar(texto, _documentos)


def responder(texto: str) -> RespostaRAG:
    chunks = perguntar(texto)
    resposta = gerar_resposta(texto, chunks)
    return RespostaRAG(pergunta=texto, resposta=resposta, chunks=chunks)


def obter_status() -> dict:
    if USAR_PGVECTOR:
        try:
            contagem = pg.contar()
        except Exception:
            contagem = {"documentos": 0, "chunks": 0}
        return {
            "indexado": contagem["documentos"] > 0,
            "documentos": contagem["documentos"],
            "chunks": contagem["chunks"],
            "modo": "pgvector",
            "llm_configurada": llm_configurada(),
            "arquivos": [arquivo.name for arquivo in listar_arquivos_originais()],
        }
    total_chunks = sum(len(doc.chunks_embedados) for doc in _documentos)
    return {
        "indexado": len(_documentos) > 0,
        "documentos": len(_documentos),
        "chunks": total_chunks,
        "modo": "memoria",
        "llm_configurada": llm_configurada(),
        "arquivos": [arquivo.name for arquivo in listar_arquivos_originais()],
    }
