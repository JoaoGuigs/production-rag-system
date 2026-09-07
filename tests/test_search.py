"""Testes da busca semântica (embedding mockado)."""

import pytest

import src.retrieval.search as search
from tests.helpers import make_embedado


def test_cosseno_vetores_identicos_e_1():
    assert search._similaridade_cosseno([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosseno_vetores_ortogonais_e_0():
    assert search._similaridade_cosseno([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosseno_vetor_nulo_e_0():
    assert search._similaridade_cosseno([0.0, 0.0], [1.0, 2.0]) == 0.0


def _docs_2d():
    doc_a = make_embedado("gatos.txt", ("gatos são felinos",), vetor=(1.0, 0.0))
    doc_b = make_embedado("cachorros.txt", ("cães são caninos",), vetor=(0.0, 1.0))
    return doc_a, doc_b


def test_buscar_ordena_por_similaridade(monkeypatch):
    monkeypatch.setattr(search, "gerar_embedding", lambda texto: [1.0, 0.0])
    doc_a, doc_b = _docs_2d()
    resultados = search.buscar("fale de gatos", [doc_a, doc_b], top_k=2)
    assert [r.chunk_embedado.chunk.fonte for r in resultados] == ["gatos.txt", "cachorros.txt"]
    assert resultados[0].similaridade == pytest.approx(1.0)
    assert resultados[1].similaridade == pytest.approx(0.0)


def test_buscar_respeita_top_k(monkeypatch):
    monkeypatch.setattr(search, "gerar_embedding", lambda texto: [1.0, 0.0])
    doc_a, doc_b = _docs_2d()
    resultados = search.buscar("fale de gatos", [doc_a, doc_b], top_k=1)
    assert len(resultados) == 1
    assert resultados[0].pergunta == "fale de gatos"
