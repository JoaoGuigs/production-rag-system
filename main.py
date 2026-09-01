"""Ponto de entrada CLI."""

import sys

from src.cli import exibir_resposta_rag, exibir_resultados
from src.pipeline import executar_pipeline
from src.web.store import indexar, responder


def main() -> None:
    if len(sys.argv) >= 2:
        indexar()
        pergunta = " ".join(sys.argv[1:])
        exibir_resposta_rag(responder(pergunta))
        return

    exibir_resultados(executar_pipeline())
    print('\nDica: python main.py "quantos dias de férias?"')


if __name__ == "__main__":
    main()
