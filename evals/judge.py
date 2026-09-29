"""Juiz opcional por LLM para correctness, groundedness e citações."""

import json

from google import genai
from google.genai import types

from src import config

RUBRICA = """Você avalia respostas de um sistema RAG.
Use somente a pergunta, a referência e os trechos recuperados.
Dê notas entre 0 e 1 para correctness, groundedness e citation_correctness.
Para pergunta não respondível, correctness=1 somente se houver recusa clara e nenhuma invenção.
Retorne apenas JSON válido com as chaves correctness, groundedness,
citation_correctness e explanation. A explicação deve ter no máximo 30 palavras."""


def julgar(
    pergunta: str,
    resposta_esperada: str | None,
    answerable: bool,
    resposta: str,
    trechos: list[dict],
) -> dict:
    if not config.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY não configurada para o judge")

    contexto = "\n\n".join(
        f"[{t['fonte']}#chunk-{t['indice']}]\n{t['texto']}" for t in trechos
    )
    prompt = (
        f"Pergunta: {pergunta}\n"
        f"Respondível: {answerable}\n"
        f"Resposta de referência: {resposta_esperada}\n"
        f"Resposta avaliada: {resposta}\n\n"
        f"Trechos recuperados:\n{contexto}"
    )
    client = genai.Client(api_key=config.GOOGLE_API_KEY)
    response = client.models.generate_content(
        model=config.GOOGLE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=RUBRICA,
            temperature=0,
            response_mime_type="application/json",
        ),
    )
    if not response.text:
        raise ValueError("Judge não retornou conteúdo")
    resultado = json.loads(response.text)
    for campo in ("correctness", "groundedness", "citation_correctness"):
        resultado[campo] = min(1.0, max(0.0, float(resultado[campo])))
    resultado["explanation"] = str(resultado.get("explanation", ""))
    return resultado
