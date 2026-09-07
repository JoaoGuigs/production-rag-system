"""Testes dos embeddings (modelo real é mockado: rápido e offline)."""

import src.embedding.embedder as embedder
from tests.helpers import make_chunkado


class _FakeArray(list):
    def tolist(self):
        return [list(v) for v in self]


class FakeModel:
    def __init__(self):
        self.chamadas = []

    def encode(self, textos, show_progress_bar=False):
        self.chamadas.append(list(textos))
        return _FakeArray([[float(len(t)), 1.0] for t in textos])


def _mock_modelo(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(embedder, "_obter_modelo", lambda: fake)
    return fake


def test_lista_vazia_nao_chama_modelo(monkeypatch):
    fake = _mock_modelo(monkeypatch)
    assert embedder.gerar_embeddings([]) == []
    assert fake.chamadas == []


def test_gerar_embedding_retorna_um_vetor(monkeypatch):
    _mock_modelo(monkeypatch)
    vetor = embedder.gerar_embedding("olá")
    assert vetor == [3.0, 1.0]


def test_embedar_documento_preserva_chunks(monkeypatch):
    _mock_modelo(monkeypatch)
    chunkado = make_chunkado(nome="doc.txt", textos=("aaa", "bbbbb"))
    resultado = embedder.embedar_documento(chunkado)
    assert resultado.nome == "doc.txt"
    assert len(resultado.chunks_embedados) == 2
    assert resultado.chunks_embedados[0].vetor == [3.0, 1.0]
    assert resultado.chunks_embedados[1].vetor == [5.0, 1.0]


def test_embedar_todos_processa_varios_documentos(monkeypatch):
    _mock_modelo(monkeypatch)
    docs = [make_chunkado("a.txt", ("x",)), make_chunkado("b.txt", ("yy", "zzz"))]
    resultados = embedder.embedar_todos(docs)
    assert [r.nome for r in resultados] == ["a.txt", "b.txt"]
    assert [len(r.chunks_embedados) for r in resultados] == [1, 2]


def test_troca_de_modelo_recarrega(monkeypatch):
    criados = []

    class FakeST:
        def __init__(self, nome):
            criados.append(nome)

    monkeypatch.setattr(embedder, "SentenceTransformer", FakeST)
    monkeypatch.setattr(embedder, "_modelo", None)
    monkeypatch.setattr(embedder, "_modelo_nome", None)
    monkeypatch.setattr(embedder.cfg, "EMBEDDING_MODEL", "modelo-a")
    embedder._obter_modelo()
    embedder._obter_modelo()
    assert criados == ["modelo-a"]
    monkeypatch.setattr(embedder.cfg, "EMBEDDING_MODEL", "modelo-b")
    embedder._obter_modelo()
    assert criados == ["modelo-a", "modelo-b"]
