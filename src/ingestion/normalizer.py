"""Limpa e padroniza o texto extraído."""

import re


def normalizar(texto: str) -> str:
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    linhas = [linha.rstrip() for linha in texto.split("\n")]
    texto = "\n".join(linhas)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    texto = re.sub(r"[^\S\n]+", " ", texto)
    return texto.strip()
