"""Orquestra o pipeline completo de indexação."""

from pathlib import Path

from src.chunking.chunker import chunkar_todos
from src.config import PASTA_ORIGINAIS
from src.embedding.embedder import embedar_todos
from src.ingestion.processor import processar_documento
from src.models import DocumentoChunkado, DocumentoEmbedado, DocumentoProcessado

EXTENSOES_SUPORTADAS = (".txt", ".md", ".pdf")


def listar_arquivos_originais(pasta: Path | None = None) -> list[Path]:
    pasta = pasta or PASTA_ORIGINAIS
    arquivos = [
        caminho
        for caminho in pasta.iterdir()
        if caminho.is_file() and caminho.suffix.lower() in EXTENSOES_SUPORTADAS
    ]
    return sorted(arquivos)


def processar_todos(pasta: Path | None = None) -> list[DocumentoProcessado]:
    return [processar_documento(arquivo) for arquivo in listar_arquivos_originais(pasta)]


def executar_pipeline(pasta: Path | None = None) -> list[DocumentoEmbedado]:
    documentos = processar_todos(pasta)
    documentos_chunkados = chunkar_todos(documentos)
    return embedar_todos(documentos_chunkados)
