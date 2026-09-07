"""Testes do storage pgvector (conexão mockada: sem banco de verdade)."""

import hashlib

import src.storage.vetores as vetores
from tests.helpers import make_chunkado


class FakeCursor:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        self._conn.calls.append(("execute", sql, params))
        return self

    def executemany(self, sql, seq):
        self._conn.calls.append(("executemany", sql, list(seq)))

    def fetchone(self):
        return self._conn.fetchone()

    def fetchall(self):
        return self._conn.fetchall()


class FakeConn:
    def __init__(self, results=()):
        self.results = list(results)
        self.calls = []
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self):
        return FakeCursor(self)

    def execute(self, sql, params=None):
        self.calls.append(("execute", sql, params))
        return self

    def fetchone(self):
        return self.results.pop(0) if self.results else None

    def fetchall(self):
        return self.results.pop(0) if self.results else []

    def commit(self):
        self.committed = True


def _mock_connect(monkeypatch, results=()):
    conns = []

    def _connect(*args, **kwargs):
        conn = FakeConn(results)
        conns.append(conn)
        return conn

    monkeypatch.setattr(vetores.psycopg, "connect", _connect)
    return conns


def test_inicializar_cria_schema_e_commita(monkeypatch):
    conns = _mock_connect(monkeypatch)
    vetores.inicializar()
    sql = conns[0].calls[0][1]
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert "CREATE TABLE IF NOT EXISTS chunks" in sql
    assert conns[0].committed is True


def test_doc_indexado_true_e_false(monkeypatch):
    _mock_connect(monkeypatch, results=[{"existe": 1}])
    assert vetores.doc_indexado("a.md", "hash", "modelo") is True

    _mock_connect(monkeypatch, results=[None])
    assert vetores.doc_indexado("a.md", "hash", "modelo") is False


def test_salvar_documento_regrava_e_retorna_n_chunks(monkeypatch):
    conns = _mock_connect(monkeypatch, results=[{"id": 7}])
    chunkado = make_chunkado("a.md", ("um", "dois"))
    n = vetores.salvar_documento("a.md", "h", "m", chunkado.chunks, [[0.1, 0.2], [0.3, 0.4]])
    assert n == 2
    tipos = [c[0] for c in conns[0].calls]
    assert tipos == ["execute", "execute", "executemany"]
    assert "DELETE FROM documentos" in conns[0].calls[0][1]
    assert "RETURNING id" in conns[0].calls[1][1]
    linhas = conns[0].calls[2][2]
    assert len(linhas) == 2
    assert linhas[0][1:4] == ("a.md", 0, "um")
    assert linhas[0][4] == "[0.1,0.2]"
    assert conns[0].committed is True


def test_buscar_vetorial_mapeia_linhas(monkeypatch):
    conns = _mock_connect(
        monkeypatch,
        results=[[{"fonte": "a.md", "indice": 2, "texto": "trecho", "similaridade": 0.9}]],
    )
    resultados = vetores.buscar_vetorial([1.0, 0.0], top_k=3)
    assert len(resultados) == 1
    assert resultados[0].chunk_embedado.chunk.fonte == "a.md"
    assert resultados[0].chunk_embedado.chunk.indice == 2
    assert resultados[0].similaridade == 0.9
    sql = conns[0].calls[0][1]
    assert "ORDER BY embedding <=> " in sql
    assert "LIMIT %s" in sql


def test_buscar_vetorial_com_filtro_de_fontes(monkeypatch):
    conns = _mock_connect(monkeypatch, results=[[]])
    vetores.buscar_vetorial([1.0], top_k=1, fontes=["a.md"])
    params = conns[0].calls[0][2]
    assert params[1] == ["a.md"]


def test_contar_retorna_docs_e_chunks(monkeypatch):
    _mock_connect(monkeypatch, results=[{"n": 2}, {"n": 5}])
    assert vetores.contar() == {"documentos": 2, "chunks": 5}


def test_hash_arquivo_e_sha256(tmp_path):
    caminho = tmp_path / "a.txt"
    caminho.write_bytes(b"conteudo")
    assert vetores.hash_arquivo(caminho) == hashlib.sha256(b"conteudo").hexdigest()


def test_vetor_literal_formata_para_pgvector():
    assert vetores._vetor_literal([1.0, 2.5]) == "[1.0,2.5]"
