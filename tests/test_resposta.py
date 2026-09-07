"""Testes das heurísticas do eval de resposta (sem chamar a API)."""

import evals.resposta as resposta_mod
from evals.resposta import afirma_sem_recusa, contem_ancora, eh_recusa, tem_citacao


def test_contem_ancora_case_insensitive():
    assert contem_ancora("Você tem 30 DIAS de férias.", ["30 dias"]) is True
    assert contem_ancora("Nada a ver com o tema.", ["30 dias"]) is False


def test_eh_recusa_detecta_variacoes():
    assert eh_recusa("Não encontrei essa informação nos documentos.") is True
    assert eh_recusa("Não há menção a isso no contexto.") is True
    assert eh_recusa("Você tem 30 dias de férias.") is False


def test_tem_citacao_procura_fonte():
    assert tem_citacao("Fontes: exemplo.md e reembolso.md", ["exemplo.md"]) is True
    assert tem_citacao("Resposta sem fonte.", ["exemplo.md"]) is False


def test_mencionar_ancora_recusando_nao_e_alucinacao():
    resposta = "Não encontrei a informação sobre o pagamento de bônus anual ou PLR nos documentos."
    assert afirma_sem_recusa(resposta, ["bônus", "PLR"]) is False


def test_afirmar_ancora_sem_recusa_e_alucinacao():
    resposta = "Sim, a empresa paga bônus anual. Não encontrei o valor exato."
    assert afirma_sem_recusa(resposta, ["bônus"]) is True


def _linha(id, tipo, acerto):
    return {"id": id, "tipo": tipo, "acerto": acerto, "detalhe": ""}


def _mock_run(monkeypatch, itens, linhas):
    monkeypatch.setattr(resposta_mod, "carregar_dataset", lambda: {"_meta": {"threshold_acerto": 0.9, "threshold_recusa": 1.0}, "itens": itens})
    monkeypatch.setattr(resposta_mod, "executar_pipeline", lambda: [])
    seq = list(linhas)
    monkeypatch.setattr(resposta_mod, "avaliar_item", lambda item, docs: seq.pop(0))
    monkeypatch.setattr(resposta_mod.time, "sleep", lambda s: None)


def test_main_reprova_abaixo_do_threshold(monkeypatch, capsys):
    itens = [{"id": f"q{i}"} for i in range(9)] + [{"id": "n1"}, {"id": "n2"}]
    linhas = [_linha(f"q{i}", "ok", 1.0 if i < 8 else 0.0) for i in range(9)]
    linhas += [_linha("n1", "recusa", 1.0), _linha("n2", "recusa", 1.0)]
    _mock_run(monkeypatch, itens, linhas)
    assert resposta_mod.main() == 1
    assert "FAIL" in capsys.readouterr().out


def test_main_passa_no_threshold(monkeypatch):
    itens = [{"id": f"q{i}"} for i in range(9)] + [{"id": "n1"}]
    linhas = [_linha(f"q{i}", "ok", 1.0) for i in range(9)] + [_linha("n1", "recusa", 1.0)]
    _mock_run(monkeypatch, itens, linhas)
    assert resposta_mod.main() == 0
