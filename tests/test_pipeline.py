"""Testes do pipeline (listar -> processar -> chunkar -> embedar)."""

import src.pipeline as pipeline


def _criar_arquivos(pasta):
    (pasta / "b.txt").write_text("b", encoding="utf-8")
    (pasta / "a.md").write_text("a", encoding="utf-8")
    (pasta / "c.PDF").write_bytes(b"%PDF-1.4 fake")
    (pasta / "ignorar.exe").write_bytes(b"x")
    sub = pasta / "sub"
    sub.mkdir()
    (sub / "dentro.txt").write_text("não lista diretório", encoding="utf-8")


def test_listar_filtra_extensoes_e_ordena(tmp_path):
    _criar_arquivos(tmp_path)
    arquivos = pipeline.listar_arquivos_originais(tmp_path)
    assert [a.name for a in arquivos] == ["a.md", "b.txt", "c.PDF"]


def test_listar_pasta_vazia_retorna_lista_vazia(tmp_path):
    assert pipeline.listar_arquivos_originais(tmp_path) == []


def test_processar_todos_chama_processar_para_cada_arquivo(tmp_path, monkeypatch):
    _criar_arquivos(tmp_path)
    chamadas = []
    monkeypatch.setattr(pipeline, "processar_documento", lambda p: chamadas.append(p) or f"ok:{p.name}")
    resultados = pipeline.processar_todos(tmp_path)
    assert [c.name for c in chamadas] == ["a.md", "b.txt", "c.PDF"]
    assert resultados == ["ok:a.md", "ok:b.txt", "ok:c.PDF"]


def test_executar_pipeline_encadeia_etapas(monkeypatch):
    ordem = []
    monkeypatch.setattr(pipeline, "processar_todos", lambda pasta=None: ordem.append("processar") or ["docs"])
    monkeypatch.setattr(pipeline, "chunkar_todos", lambda docs: ordem.append("chunkar") or ["chunks"])
    monkeypatch.setattr(pipeline, "embedar_todos", lambda docs: ordem.append("embedar") or ["vetores"])
    assert pipeline.executar_pipeline() == ["vetores"]
    assert ordem == ["processar", "chunkar", "embedar"]
