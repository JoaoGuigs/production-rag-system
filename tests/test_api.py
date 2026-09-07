"""Testes da API HTTP (store mockado: sem pipeline nem Gemini de verdade)."""

from fastapi.testclient import TestClient

import src.web.api as api
from src.models import RespostaRAG
from tests.helpers import make_resultado

client = TestClient(api.app)


def test_pagina_inicial_retorna_html():
    resposta = client.get("/")
    assert resposta.status_code == 200
    assert "text/html" in resposta.headers["content-type"]


def test_status_repasse_do_store(monkeypatch):
    monkeypatch.setattr(api, "obter_status", lambda: {"indexado": True, "documentos": 1})
    resposta = client.get("/api/status")
    assert resposta.status_code == 200
    assert resposta.json() == {"indexado": True, "documentos": 1}


def test_upload_txt_salva_arquivo(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "PASTA_ORIGINAIS", tmp_path)
    resposta = client.post(
        "/api/upload",
        files={"arquivo": ("nota.txt", "conteúdo", "text/plain")},
    )
    assert resposta.status_code == 200
    assert resposta.json() == {"ok": True, "arquivo": "nota.txt"}
    assert (tmp_path / "nota.txt").read_text(encoding="utf-8") == "conteúdo"


def test_upload_formato_invalido_retorna_400():
    resposta = client.post(
        "/api/upload",
        files={"arquivo": ("programa.exe", "bin", "application/octet-stream")},
    )
    assert resposta.status_code == 400


def test_indexar_ok(monkeypatch):
    monkeypatch.setattr(api, "indexar", lambda: {"documentos": 2, "chunks": 5})
    resposta = client.post("/api/indexar")
    assert resposta.status_code == 200
    assert resposta.json() == {"documentos": 2, "chunks": 5}


def test_indexar_sem_arquivos_retorna_400(monkeypatch):
    def _falhar():
        raise FileNotFoundError("pasta vazia")

    monkeypatch.setattr(api, "indexar", _falhar)
    resposta = client.post("/api/indexar")
    assert resposta.status_code == 400


def test_perguntar_retorna_resposta_e_fontes(monkeypatch):
    def _responder(texto):
        return RespostaRAG(
            pergunta=texto,
            resposta="30 dias",
            chunks=[make_resultado(pergunta=texto, similaridade=0.923456)],
        )

    monkeypatch.setattr(api, "responder", _responder)
    resposta = client.post("/api/perguntar", json={"pergunta": "quantos dias?"})
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["pergunta"] == "quantos dias?"
    assert corpo["resposta"] == "30 dias"
    assert corpo["resultados"][0]["similaridade"] == 0.9235
    assert corpo["resultados"][0]["fonte"] == "doc.txt"


def test_perguntar_vazia_retorna_400(monkeypatch):
    monkeypatch.setattr(api, "responder", lambda texto: None)
    resposta = client.post("/api/perguntar", json={"pergunta": "   "})
    assert resposta.status_code == 400


def test_perguntar_sem_indice_retorna_400(monkeypatch):
    def _falhar(texto):
        raise ValueError("Nenhum documento indexado.")

    monkeypatch.setattr(api, "responder", _falhar)
    resposta = client.post("/api/perguntar", json={"pergunta": "oi?"})
    assert resposta.status_code == 400


def test_perguntar_erro_interno_retorna_500(monkeypatch):
    def _falhar(texto):
        raise RuntimeError("boom")

    monkeypatch.setattr(api, "responder", _falhar)
    resposta = client.post("/api/perguntar", json={"pergunta": "oi?"})
    assert resposta.status_code == 500
