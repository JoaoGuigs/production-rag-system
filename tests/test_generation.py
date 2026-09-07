"""Testes da geração com Gemini (API real é mockada)."""

from types import SimpleNamespace

import pytest

import src.generation.llm as llm
from tests.helpers import make_resultado


def test_llm_nao_configurada_sem_chave(monkeypatch):
    monkeypatch.setattr(llm, "GOOGLE_API_KEY", "")
    assert llm.llm_configurada() is False


def test_llm_configurada_com_chave(monkeypatch):
    monkeypatch.setattr(llm, "GOOGLE_API_KEY", "fake-key")
    assert llm.llm_configurada() is True


def test_gerar_resposta_sem_chave_levanta_erro(monkeypatch):
    monkeypatch.setattr(llm, "GOOGLE_API_KEY", "")
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        llm.gerar_resposta("pergunta?", [])


class _FakeModels:
    def __init__(self, caixa, resposta_ou_erro):
        self.caixa = caixa
        self.resposta_ou_erro = resposta_ou_erro

    def generate_content(self, **kwargs):
        self.caixa.update(kwargs)
        if isinstance(self.resposta_ou_erro, Exception):
            raise self.resposta_ou_erro
        return SimpleNamespace(text=self.resposta_ou_erro)


class _FakeClient:
    def __init__(self, caixa, resposta_ou_erro, api_key=None):
        self.api_key = api_key
        self.models = _FakeModels(caixa, resposta_ou_erro)


def _mock_client(monkeypatch, resposta_ou_erro="resposta final"):
    caixa = {}
    monkeypatch.setattr(llm, "GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(llm.genai, "Client", lambda api_key=None: _FakeClient(caixa, resposta_ou_erro, api_key))
    return caixa


def test_gerar_resposta_envia_contexto_e_pergunta(monkeypatch):
    caixa = _mock_client(monkeypatch)
    resultados = [make_resultado(fonte="guia.txt", texto="30 dias de férias")]
    resposta = llm.gerar_resposta("quantos dias de férias?", resultados)
    assert resposta == "resposta final"
    assert "30 dias de férias" in caixa["contents"]
    assert "guia.txt" in caixa["contents"]
    assert "quantos dias de férias?" in caixa["contents"]


def test_gerar_resposta_propaga_erro_da_google(monkeypatch):
    _mock_client(monkeypatch, RuntimeError("quota esgotada"))
    with pytest.raises(ValueError, match="Erro da Google"):
        llm.gerar_resposta("pergunta?", [make_resultado()])


def test_gerar_resposta_vazia_levanta_erro(monkeypatch):
    _mock_client(monkeypatch, "")
    with pytest.raises(ValueError, match="não retornou texto"):
        llm.gerar_resposta("pergunta?", [make_resultado()])


class _Erro503(Exception):
    code = 503


def test_gerar_resposta_tenta_de_novo_no_503_e_consegue(monkeypatch):
    from types import SimpleNamespace

    caixa = {}
    chamadas = {"n": 0}
    dormidas = []
    monkeypatch.setattr(llm, "GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(llm.time, "sleep", lambda s: dormidas.append(s))

    class _Modelos:
        def generate_content(self, **kwargs):
            caixa.update(kwargs)
            chamadas["n"] += 1
            if chamadas["n"] < 3:
                raise _Erro503("UNAVAILABLE temporário")
            return SimpleNamespace(text="resposta final")

    class _Client:
        def __init__(self, api_key=None):
            self.models = _Modelos()

    monkeypatch.setattr(llm.genai, "Client", lambda api_key=None: _Client())
    resposta = llm.gerar_resposta("pergunta?", [make_resultado()])
    assert resposta == "resposta final"
    assert chamadas["n"] == 3
    assert dormidas == [5, 10]


def test_gerar_resposta_503_persistente_da_mensagem_amigavel(monkeypatch):
    monkeypatch.setattr(llm, "GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)

    class _Modelos:
        def generate_content(self, **kwargs):
            raise _Erro503("UNAVAILABLE")

    class _Client:
        def __init__(self, api_key=None):
            self.models = _Modelos()

    monkeypatch.setattr(llm.genai, "Client", lambda api_key=None: _Client())
    with pytest.raises(ValueError, match="sobrecarregada"):
        llm.gerar_resposta("pergunta?", [make_resultado()])


def test_gerar_resposta_erro_nao_transitorio_nao_tenta_de_novo(monkeypatch):
    dormidas = []
    monkeypatch.setattr(llm, "GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(llm.time, "sleep", lambda s: dormidas.append(s))

    class _Modelos:
        def generate_content(self, **kwargs):
            raise ValueError("API key inválida (400)")

    class _Client:
        def __init__(self, api_key=None):
            self.models = _Modelos()

    monkeypatch.setattr(llm.genai, "Client", lambda api_key=None: _Client())
    with pytest.raises(ValueError, match="Erro da Google"):
        llm.gerar_resposta("pergunta?", [make_resultado()])
    assert dormidas == []
