"""Testes do eval end-to-end v2 sem chamar embeddings ou Gemini reais."""

import json
from argparse import Namespace

import evals.resposta as resposta_mod
from evals.resposta import avaliar_item
from evals.report import salvar_relatorio
from evals.metrics import (
    afirma_sem_recusa,
    agregar,
    avaliar_gates,
    cobertura_fatos,
    eh_recusa,
    extrair_citacoes,
    normalizar,
    tem_citacao,
)
from tests.helpers import make_resultado


def test_normalizar_remove_acentos_e_diferenca_de_caixa():
    assert normalizar("  NÃO   HÁ Informação ") == "nao ha informacao"


def test_recusa_detecta_variacoes():
    assert eh_recusa("Não encontrei essa informação nos documentos.")
    assert eh_recusa("Não há menção a isso no contexto.")
    assert not eh_recusa("Você tem 30 dias de férias.")


def test_citacao_exige_formato_entre_colchetes():
    assert extrair_citacoes("Segundo [exemplo.md], são 30 dias.") == ["exemplo.md"]
    assert extrair_citacoes("Segundo [exemplo.md, manual.md].") == ["exemplo.md", "manual.md"]
    assert extrair_citacoes("Segundo exemplo.md, são 30 dias.") == []
    assert tem_citacao("Fonte [EXEMPLO.md]", ["exemplo.md"])


def test_cobertura_fatos_exige_todos_os_grupos():
    fatos = [
        {"label": "prazo", "any_of": ["30 dias", "trinta dias"]},
        {"label": "valor", "any_of": ["R$ 80", "80 reais"]},
    ]
    cobertura, ausentes = cobertura_fatos("O prazo é de trinta dias.", fatos)
    assert cobertura == 0.5
    assert ausentes == ["valor"]


def test_mencionar_afirmacao_proibida_em_recusa_nao_e_alucinacao():
    assert not afirma_sem_recusa("Não encontrei informação sobre stock options.", ["stock options"])
    assert not afirma_sem_recusa("Não, você não pode ficar com o presente.", ["pode ficar com o presente"])
    assert afirma_sem_recusa("A empresa oferece stock options após um ano.", ["oferece stock options"])


def _item(answerable=True):
    return {
        "id": "q1", "category": "teste", "question": "Qual é o prazo?",
        "answerable": answerable,
        "expected_answer": "30 dias" if answerable else None,
        "expected_sources": ["politica.md"] if answerable else [],
        "expected_evidence": ["30 dias"] if answerable else [],
        "required_facts": ([{"label": "prazo", "any_of": ["30 dias"]}] if answerable else []),
        "forbidden_claims": ["60 dias"],
    }


def test_avaliar_item_guarda_resposta_chunks_metricas_e_latencia(monkeypatch):
    resultado = make_resultado(fonte="politica.md", texto="O prazo é 30 dias.")
    monkeypatch.setattr(resposta_mod, "buscar", lambda *args, **kwargs: [resultado])
    monkeypatch.setattr(resposta_mod, "gerar_resposta", lambda *args: "O prazo é 30 dias [politica.md].")
    registro = avaliar_item(_item(), [], top_k=3)
    assert registro["status"] == "ok"
    assert registro["retrieval_hit"] == 1.0
    assert registro["metrics"]["answer_correct"] == 1.0
    assert registro["metrics"]["expected_citation"] == 1.0
    assert registro["retrieved_chunks"][0]["texto"] == "O prazo é 30 dias."
    assert registro["latency_ms"]["total"] >= 0


def test_avaliar_item_negativo_exige_recusa(monkeypatch):
    monkeypatch.setattr(resposta_mod, "buscar", lambda *args, **kwargs: [])
    monkeypatch.setattr(resposta_mod, "gerar_resposta", lambda *args: "Não encontrei essa informação.")
    registro = avaliar_item(_item(answerable=False), [])
    assert registro["metrics"]["correct_refusal"] == 1.0
    assert registro["metrics"]["answer_correct"] == 1.0


def test_erro_de_api_entra_na_taxa_e_reprova_gate(monkeypatch):
    monkeypatch.setattr(resposta_mod, "buscar", lambda *args, **kwargs: [])
    monkeypatch.setattr(resposta_mod, "gerar_resposta", lambda *args: (_ for _ in ()).throw(ValueError("503")))
    registro = avaliar_item(_item(), [])
    resumo = agregar([registro])
    falhas = avaliar_gates(resumo, {"maximum": {"error_rate": 0.0}})
    assert registro["status"] == "error"
    assert resumo["error_rate"] == 1.0
    assert falhas


def test_gate_verifica_citacao_e_minimo_ausente():
    resumo = {"answer_accuracy": 1.0, "expected_citation_rate": 0.5}
    thresholds = {"minimum": {"answer_accuracy": 0.9, "expected_citation_rate": 0.8, "fact_completeness": 0.8}}
    falhas = avaliar_gates(resumo, thresholds)
    assert len(falhas) == 2
    assert any("expected_citation_rate" in falha for falha in falhas)
    assert any("fact_completeness" in falha for falha in falhas)


def test_agregado_calcula_p95_e_nao_da_nota_perfeita_sem_sucesso():
    erro = {"status": "error", "answerable": True, "metrics": {}, "latency_ms": {"total": 1}}
    resumo = agregar([erro])
    assert resumo["errors"] == 1
    assert resumo["answer_accuracy"] is None
    assert resumo["latency_ms"]["p95"] == 0.0


def test_salvar_relatorio_em_json(tmp_path):
    destino = salvar_relatorio({"schema_version": 2, "summary": {}}, tmp_path / "run.json")
    assert json.loads(destino.read_text(encoding="utf-8"))["schema_version"] == 2


def test_executar_persiste_configuracao_e_resultado(tmp_path, monkeypatch):
    dataset = {"_meta": {"version": 2, "name": "teste", "top_k": 3,
                         "thresholds": {"maximum": {"error_rate": 0.0}}},
               "items": [_item()]}
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    monkeypatch.setattr(resposta_mod, "executar_pipeline", lambda *args: [])
    monkeypatch.setattr(resposta_mod, "avaliar_item", lambda *args: {
        "status": "error", "answerable": True, "metrics": {}, "latency_ms": {"total": 1},
        "retrieval_hit": None, "reciprocal_rank": None,
    })
    args = Namespace(dataset=str(dataset_path), output=str(tmp_path / "out.json"),
                     top_k=None, interval=0, judge=False, replay=None)
    relatorio, falhas, destino = resposta_mod.executar(args)
    assert relatorio["schema_version"] == 2
    assert relatorio["configuration"]["top_k"] == 3
    assert falhas
    assert destino.exists()


def test_replay_recalcula_sem_chamar_pipeline(tmp_path, monkeypatch):
    dataset = {"_meta": {"version": 2, "name": "teste", "top_k": 3,
                         "thresholds": {"minimum": {"answer_accuracy": 1.0}}},
               "items": [_item()]}
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    artefato = {
        "schema_version": 2, "configuration": {"top_k": 3}, "indexing": {},
        "results": [{"id": "q1", "status": "ok", "answerable": True,
                     "answer": "São 30 dias [politica.md].",
                     "retrieved_chunks": [{"rank": 1, "fonte": "politica.md", "indice": 0,
                                           "similaridade": .9, "texto": "O prazo é 30 dias."}],
                     "latency_ms": {"total": 10}}],
    }
    origem = tmp_path / "anterior.json"
    origem.write_text(json.dumps(artefato), encoding="utf-8")
    monkeypatch.setattr(resposta_mod, "executar_pipeline", lambda *args: (_ for _ in ()).throw(AssertionError()))
    args = Namespace(dataset=str(dataset_path), output=str(tmp_path / "replay.json"), top_k=None,
                     interval=0, judge=False, replay=str(origem))
    relatorio, falhas, _ = resposta_mod.executar(args)
    assert relatorio["summary"]["answer_accuracy"] == 1.0
    assert not falhas
