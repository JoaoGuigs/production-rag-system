"""Testes do normalizador de texto."""

from src.ingestion.normalizer import normalizar


def test_converte_crlf_para_lf():
    assert normalizar("a\r\nb\r\nc") == "a\nb\nc"


def test_remove_espacos_no_fim_das_linhas():
    assert normalizar("olá   \nmundo  ") == "olá\nmundo"


def test_colapsa_tres_ou_mais_quebras():
    assert normalizar("a\n\n\n\nb") == "a\n\nb"


def test_colapsa_espacos_e_tabs():
    assert normalizar("olá   \t  mundo") == "olá mundo"


def test_remove_espacos_nas_bordas():
    assert normalizar("  \n  texto  \n  ") == "texto"


def test_texto_limpo_passsa_quase_intacto():
    texto = "Título\n\nParágrafo um.\nParágrafo dois."
    assert normalizar(texto) == texto
