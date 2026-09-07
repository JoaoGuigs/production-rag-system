"""Testes do processador (extrair -> normalizar -> salvar)."""

import src.ingestion.processor as processor


def test_processar_documento_salva_texto_normalizado(tmp_path, monkeypatch):
    monkeypatch.setattr(processor, "PASTA_PROCESSADOS", tmp_path / "processed")
    origem = tmp_path / "nota.txt"
    origem.write_text("  Olá   mundo  \n\n\nfim  ", encoding="utf-8")

    doc = processor.processar_documento(origem)

    assert doc.nome == "nota.txt"
    assert doc.texto_bruto == "  Olá   mundo  \n\n\nfim  "
    assert doc.texto_limpo == "Olá mundo\n\nfim"
    assert doc.caminho_salvo == tmp_path / "processed" / "nota.txt"
    assert doc.caminho_salvo.read_text(encoding="utf-8") == "Olá mundo\n\nfim"


def test_processar_md_usa_stem_no_arquivo_salvo(tmp_path, monkeypatch):
    monkeypatch.setattr(processor, "PASTA_PROCESSADOS", tmp_path / "processed")
    origem = tmp_path / "guia.md"
    origem.write_text("# Guia\n\nConteúdo.", encoding="utf-8")

    doc = processor.processar_documento(origem)

    assert doc.caminho_salvo.name == "guia.txt"
    assert "Guia" in doc.texto_limpo
