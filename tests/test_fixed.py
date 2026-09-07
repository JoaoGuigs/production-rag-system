"""Testes do chunking de janela fixa (baseline do benchmark)."""

from src.chunking.chunker import chunkar_documento
from src.chunking.fixed import dividir_em_chunks_fixo
from tests.helpers import make_processado


def test_texto_vazio_nao_gera_chunks():
    assert dividir_em_chunks_fixo("   ") == []


def test_texto_curto_vira_chunk_unico():
    assert dividir_em_chunks_fixo("olá mundo", max_tokens=100, overlap_tokens=10) == ["olá mundo"]


def test_janela_respeita_tamanho_maximo():
    chunks = dividir_em_chunks_fixo("a" * 1000, max_tokens=100, overlap_tokens=0)
    assert len(chunks) == 3  # 400 + 400 + 200 chars
    assert all(len(c) <= 400 for c in chunks)


def test_overlap_repete_trecho():
    chunks = dividir_em_chunks_fixo("a" * 500, max_tokens=100, overlap_tokens=25)
    assert chunks[1].startswith("a" * 100)  # últimos 100 chars do chunk anterior
    assert len(chunks[1]) == 200  # 100 de overlap + 100 novos


def test_cobre_texto_inteiro():
    texto = "Palavra " * 300
    chunks = dividir_em_chunks_fixo(texto, max_tokens=100, overlap_tokens=20)
    assert texto[:400].strip() in chunks[0]
    assert texto.strip()[-100:] in chunks[-1]


def test_estrategia_fixo_no_pipeline():
    doc = make_processado(nome="d.txt", limpo="Conteúdo " * 500)
    resultado = chunkar_documento(doc, estrategia="fixo")
    assert len(resultado.chunks) > 1
    assert all(c.fonte == "d.txt" for c in resultado.chunks)


def test_estrategia_padrao_continua_hierarquica():
    doc = make_processado(nome="d.txt", limpo="# T\n\ncurto")
    resultado = chunkar_documento(doc)
    assert resultado.chunks[0].texto.startswith("# T")
