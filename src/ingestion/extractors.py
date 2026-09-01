"""Extrai texto bruto de diferentes formatos de arquivo."""

from pathlib import Path

from pypdf import PdfReader


def extrair_texto(caminho: Path) -> str:
    extensao = caminho.suffix.lower()

    if extensao == ".txt":
        return caminho.read_text(encoding="utf-8")
    if extensao == ".md":
        return caminho.read_text(encoding="utf-8")
    if extensao == ".pdf":
        return _extrair_pdf(caminho)

    raise ValueError(f"Formato não suportado: {extensao}")


def _extrair_pdf(caminho: Path) -> str:
    leitor = PdfReader(str(caminho))
    paginas = []
    for pagina in leitor.pages:
        texto = pagina.extract_text()
        if texto:
            paginas.append(texto)
    return "\n\n".join(paginas)
