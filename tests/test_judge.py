"""Testes do juiz opcional sem chamar o Gemini."""

import pytest

import evals.judge as judge


def test_judge_exige_chave(monkeypatch):
    monkeypatch.setattr(judge.config, "GOOGLE_API_KEY", "")
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        judge.julgar("pergunta", "resposta", True, "resposta", [])


def test_judge_envia_contexto_e_normaliza_notas(monkeypatch):
    chamadas = {}

    class Models:
        def generate_content(self, **kwargs):
            chamadas.update(kwargs)
            return type("Resposta", (), {"text": '{"correctness": 1.2, "groundedness": -0.1, "citation_correctness": 0.8, "explanation": "ok"}'})()

    class Client:
        def __init__(self, api_key):
            chamadas["api_key"] = api_key
            self.models = Models()

    monkeypatch.setattr(judge.config, "GOOGLE_API_KEY", "segredo")
    monkeypatch.setattr(judge.genai, "Client", Client)
    resultado = judge.julgar(
        "Qual o prazo?", "30 dias", True, "São 30 dias [doc.md].",
        [{"fonte": "doc.md", "indice": 0, "texto": "O prazo é 30 dias."}],
    )
    assert resultado["correctness"] == 1.0
    assert resultado["groundedness"] == 0.0
    assert resultado["citation_correctness"] == 0.8
    assert "O prazo é 30 dias" in chamadas["contents"]
