"""Testes dos modelos e das constantes de configuração."""

from src import config
from tests.helpers import make_chunkado, make_embedado, make_processado


def test_documento_nome_vem_do_arquivo_original():
    assert make_processado("guia.md").nome == "guia.md"


def test_chunkado_nome_delega_para_documento():
    assert make_chunkado("a.txt").nome == "a.txt"


def test_embedado_nome_delega_para_chunkado():
    assert make_embedado("b.txt").nome == "b.txt"


def test_config_chunk_tem_overlap_menor_que_maximo():
    assert 0 < config.CHUNK_OVERLAP_TOKENS < config.CHUNK_MAX_TOKENS


def test_config_busca_e_pastas_sao_validas():
    assert config.TOP_K > 0
    assert config.BASE_DIR.is_dir()
    assert config.EMBEDDING_MODEL.strip() != ""
