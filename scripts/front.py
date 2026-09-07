"""Abre o navegador no app (rode com o `make back` ligado no outro terminal)."""

import sys
import webbrowser

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

print(f"Abrindo {URL}. Deixe o make back rodando no outro terminal.")
webbrowser.open(URL)
