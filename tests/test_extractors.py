"""Testes da extração de texto (txt / md / pdf)."""

import pytest
from pypdf import PdfWriter

from src.ingestion.extractors import extrair_texto


def test_extrair_txt_preserva_conteudo(tmp_path):
    caminho = tmp_path / "nota.txt"
    caminho.write_text("Olá, mundo com acentuação!", encoding="utf-8")
    assert extrair_texto(caminho) == "Olá, mundo com acentuação!"


def test_extrair_md_preserva_conteudo(tmp_path):
    caminho = tmp_path / "nota.md"
    caminho.write_text("# Título\n\nTexto **negrito**.", encoding="utf-8")
    assert extrair_texto(caminho) == "# Título\n\nTexto **negrito**."


def test_extrair_pdf_sem_texto_retorna_vazio(tmp_path):
    caminho = tmp_path / "vazio.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with caminho.open("wb") as f:
        writer.write(f)
    assert extrair_texto(caminho) == ""


def test_formato_nao_suportado_levanta_erro(tmp_path):
    caminho = tmp_path / "arquivo.exe"
    caminho.write_bytes(b"binario")
    with pytest.raises(ValueError, match="Formato não suportado"):
        extrair_texto(caminho)
