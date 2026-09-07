"""Testes do chunking hierárquico."""

from src.chunking.chunker import (
    _aplicar_overlap,
    chunkar_documento,
    contar_tokens,
    dividir_em_chunks,
)
from tests.helpers import make_processado


def test_contar_tokens_aproxima_4_chars_por_token():
    assert contar_tokens("") == 1
    assert contar_tokens("abcd") == 1
    assert contar_tokens("abcdefgh") == 2


def test_texto_vazio_nao_gera_chunks():
    assert dividir_em_chunks("   ") == []


def test_texto_curto_vira_chunk_unico():
    chunks = dividir_em_chunks("a\n\nb\n\nc", max_tokens=350)
    assert len(chunks) == 1
    assert "a" in chunks[0] and "c" in chunks[0]


def test_heading_fica_junto_do_conteudo():
    chunks = dividir_em_chunks("# Férias\n\nVocê tem 30 dias de férias.", max_tokens=350)
    assert len(chunks) == 1
    assert "Férias" in chunks[0]
    assert "30 dias" in chunks[0]


def test_texto_longo_gera_varios_chunks_nao_vazios():
    texto = "Palavra " * 500
    chunks = dividir_em_chunks(texto, max_tokens=100)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_frase_gigante_sem_pontuacao_cai_no_fallback():
    texto = "a" * 5000
    chunks = dividir_em_chunks(texto, max_tokens=100)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_overlap_repete_fim_do_chunk_anterior():
    chunks = _aplicar_overlap(["abcdefghij", "klmnop"], overlap_tokens=1)
    assert chunks[0] == "abcdefghij"
    assert chunks[1] == "ghij\n\nklmnop"


def test_overlap_comportamental_entre_chunks():
    frases = " ".join(f"Sentença número {i} sobre férias." for i in range(30))
    chunks = dividir_em_chunks(frases, max_tokens=30, overlap_tokens=8)
    assert len(chunks) >= 2
    ultima_palavra = chunks[0].split()[-1]
    assert ultima_palavra in chunks[1]


def test_chunkar_documento_numera_e_marca_fonte():
    doc = make_processado(nome="guia.md", limpo="# Guia\n\nPasso um.\n\nPasso dois.")
    resultado = chunkar_documento(doc)
    assert resultado.nome == "guia.md"
    assert len(resultado.chunks) >= 1
    assert [c.indice for c in resultado.chunks] == list(range(len(resultado.chunks)))
    assert all(c.fonte == "guia.md" for c in resultado.chunks)
