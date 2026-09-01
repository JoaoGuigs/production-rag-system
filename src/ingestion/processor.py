"""Processa um arquivo: extrair → normalizar → salvar."""

from pathlib import Path

from src.config import PASTA_PROCESSADOS
from src.ingestion.extractors import extrair_texto
from src.ingestion.normalizer import normalizar
from src.models import DocumentoProcessado


def processar_documento(caminho_original: Path) -> DocumentoProcessado:
    texto_bruto = extrair_texto(caminho_original)
    texto_limpo = normalizar(texto_bruto)

    PASTA_PROCESSADOS.mkdir(parents=True, exist_ok=True)
    caminho_salvo = PASTA_PROCESSADOS / f"{caminho_original.stem}.txt"
    caminho_salvo.write_text(texto_limpo, encoding="utf-8")

    return DocumentoProcessado(
        arquivo_original=caminho_original,
        texto_bruto=texto_bruto,
        texto_limpo=texto_limpo,
        caminho_salvo=caminho_salvo,
    )
