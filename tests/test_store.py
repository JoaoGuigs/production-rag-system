"""Testes do store (memória e pgvector)."""

from pathlib import Path

import pytest

import src.web.store as store
from tests.helpers import make_embedado, make_processado, make_resultado


def _modo_memoria(monkeypatch):
    monkeypatch.setattr(store, "USAR_PGVECTOR", False)


def test_perguntar_sem_documentos_levanta_erro(monkeypatch):
    _modo_memoria(monkeypatch)
    monkeypatch.setattr(store, "_documentos", [])
    with pytest.raises(ValueError, match="Nenhum documento indexado"):
        store.perguntar("alguma pergunta?")


def test_indexar_guarda_documentos_e_conta_chunks(monkeypatch):
    _modo_memoria(monkeypatch)
    doc = make_embedado("doc.txt", ("um", "dois"))
    monkeypatch.setattr(store, "executar_pipeline", lambda: [doc])
    assert store.indexar() == {"documentos": 1, "chunks": 2}


def test_perguntar_delega_para_busca(monkeypatch):
    _modo_memoria(monkeypatch)
    doc = make_embedado()
    monkeypatch.setattr(store, "_documentos", [doc])
    monkeypatch.setattr(store, "buscar", lambda texto, docs: [f"busca:{texto}:{len(docs)}"])
    assert store.perguntar("oi?") == ["busca:oi?:1"]


def test_responder_junta_busca_e_llm(monkeypatch):
    esperado = [make_resultado()]
    monkeypatch.setattr(store, "perguntar", lambda texto: esperado)
    monkeypatch.setattr(store, "gerar_resposta", lambda texto, chunks: "resposta!")
    resposta = store.responder("pergunta?")
    assert resposta.pergunta == "pergunta?"
    assert resposta.resposta == "resposta!"
    assert resposta.chunks == esperado


def test_obter_status_resume_estado(monkeypatch):
    _modo_memoria(monkeypatch)
    doc = make_embedado("doc.txt", ("um", "dois"))
    monkeypatch.setattr(store, "_documentos", [doc])
    monkeypatch.setattr(store, "llm_configurada", lambda: True)
    monkeypatch.setattr(store, "listar_arquivos_originais", lambda: [Path("a.txt"), Path("b.pdf")])
    assert store.obter_status() == {
        "indexado": True,
        "documentos": 1,
        "chunks": 2,
        "modo": "memoria",
        "llm_configurada": True,
        "arquivos": ["a.txt", "b.pdf"],
    }


def test_obter_status_vazio_nao_indexado(monkeypatch):
    _modo_memoria(monkeypatch)
    monkeypatch.setattr(store, "_documentos", [])
    monkeypatch.setattr(store, "llm_configurada", lambda: False)
    monkeypatch.setattr(store, "listar_arquivos_originais", lambda: [])
    status = store.obter_status()
    assert status["indexado"] is False
    assert status["documentos"] == 0
    assert status["chunks"] == 0
    assert status["modo"] == "memoria"


def _modo_pg(monkeypatch):
    monkeypatch.setattr(store, "USAR_PGVECTOR", True)


def test_indexar_pg_pula_documento_ja_indexado(monkeypatch):
    _modo_pg(monkeypatch)
    doc = make_processado("a.md")
    monkeypatch.setattr(store, "processar_todos", lambda: [doc])
    monkeypatch.setattr(store.pg, "inicializar", lambda: None)
    monkeypatch.setattr(store.pg, "hash_arquivo", lambda caminho: "hash-igual")
    monkeypatch.setattr(store.pg, "doc_indexado", lambda nome, h, m: True)
    chamadas = []
    monkeypatch.setattr(store.pg, "salvar_documento", lambda *a: chamadas.append(a))
    monkeypatch.setattr(store.pg, "contar", lambda: {"documentos": 1, "chunks": 2})
    assert store.indexar() == {"documentos": 1, "chunks": 2}
    assert chamadas == []


def test_indexar_pg_embeda_e_salva_novos(monkeypatch):
    _modo_pg(monkeypatch)
    doc = make_processado("a.md")
    monkeypatch.setattr(store, "processar_todos", lambda: [doc])
    monkeypatch.setattr(store.pg, "inicializar", lambda: None)
    monkeypatch.setattr(store.pg, "hash_arquivo", lambda caminho: "hash-novo")
    monkeypatch.setattr(store.pg, "doc_indexado", lambda nome, h, m: False)
    monkeypatch.setattr(store, "gerar_embeddings", lambda textos: [[0.1] for _ in textos])
    salvos = []
    monkeypatch.setattr(store.pg, "salvar_documento", lambda *a: salvos.append(a) or 1)
    monkeypatch.setattr(store.pg, "contar", lambda: {"documentos": 1, "chunks": 1})
    assert store.indexar() == {"documentos": 1, "chunks": 1}
    assert len(salvos) == 1
    assert salvos[0][0] == "a.md"


def test_perguntar_pg_busca_no_banco(monkeypatch):
    _modo_pg(monkeypatch)
    esperado = [make_resultado()]
    monkeypatch.setattr(store, "gerar_embedding", lambda texto: [1.0])
    monkeypatch.setattr(store.pg, "buscar_vetorial", lambda vetor, top_k: esperado)
    assert store.perguntar("oi?") == esperado


def test_perguntar_pg_vazio_levanta_erro(monkeypatch):
    _modo_pg(monkeypatch)
    monkeypatch.setattr(store, "gerar_embedding", lambda texto: [1.0])
    monkeypatch.setattr(store.pg, "buscar_vetorial", lambda vetor, top_k: [])
    with pytest.raises(ValueError, match="Nenhum documento indexado"):
        store.perguntar("oi?")


def test_obter_status_pg_vem_do_banco(monkeypatch):
    _modo_pg(monkeypatch)
    monkeypatch.setattr(store.pg, "contar", lambda: {"documentos": 2, "chunks": 5})
    monkeypatch.setattr(store, "llm_configurada", lambda: True)
    monkeypatch.setattr(store, "listar_arquivos_originais", lambda: [])
    status = store.obter_status()
    assert status["modo"] == "pgvector"
    assert status["indexado"] is True
    assert status["documentos"] == 2
    assert status["chunks"] == 5
