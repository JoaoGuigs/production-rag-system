"""Geração de resposta com Google Gemini."""

from google import genai
from google.genai import types

from src.config import GOOGLE_API_KEY, GOOGLE_MODEL
from src.models import ResultadoBusca

SYSTEM_PROMPT = """Você é um assistente que responde APENAS com base no contexto fornecido.
Se a resposta não estiver no contexto, diga claramente que não encontrou a informação.
Cite a fonte quando possível.
Responda em português, de forma clara e objetiva."""


def llm_configurada() -> bool:
    return bool(GOOGLE_API_KEY)


def gerar_resposta(pergunta: str, resultados: list[ResultadoBusca]) -> str:
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY não configurada. "
            "Gere uma chave grátis em aistudio.google.com/apikey e adicione no .env"
        )

    contexto = "\n\n".join(
        f"--- Trecho {i} (fonte: {r.chunk_embedado.chunk.fonte}) ---\n{r.chunk_embedado.chunk.texto}"
        for i, r in enumerate(resultados, start=1)
    )

    client = genai.Client(api_key=GOOGLE_API_KEY)
    try:
        response = client.models.generate_content(
            model=GOOGLE_MODEL,
            contents=f"Contexto:\n{contexto}\n\nPergunta: {pergunta}",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
            ),
        )
    except Exception as erro:
        raise ValueError(f"Erro da Google: {erro}") from erro

    if not response.text:
        raise ValueError("A Google não retornou texto na resposta.")
    return response.text
