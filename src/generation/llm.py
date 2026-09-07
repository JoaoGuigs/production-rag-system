"""Geração de resposta com Google Gemini."""

import time

from google import genai
from google.genai import types

from src.config import GOOGLE_API_KEY, GOOGLE_MODEL
from src.models import ResultadoBusca

SYSTEM_PROMPT = """Você é um assistente que responde APENAS com base no contexto fornecido.
Se a resposta não estiver no contexto, diga claramente que não encontrou a informação.
Cite a fonte quando possível.
Responda em português, de forma clara e objetiva."""

# Erros transitórios da Google que valem uma nova tentativa (com espera).
_STATUS_TRANSITORIOS = {429, 500, 502, 503, 504}
MAX_TENTATIVAS = 3
ESPERA_BASE_S = 5


def llm_configurada() -> bool:
    return bool(GOOGLE_API_KEY)


def _eh_transitorio(erro: Exception) -> bool:
    codigo = getattr(erro, "code", None)
    if codigo in _STATUS_TRANSITORIOS:
        return True
    texto = str(erro).upper()
    return any(
        chave in texto
        for chave in ("429", "500", "502", "503", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED")
    )


def _mensagem_erro(erro: Exception) -> str:
    codigo = getattr(erro, "code", None)
    if codigo == 429 or "RESOURCE_EXHAUSTED" in str(erro).upper():
        return (
            "Limite da Google atingido (cota do plano gratuito). "
            f"Aguarde um pouco e tente de novo. Detalhe: {erro}"
        )
    if codigo in (500, 502, 503, 504) or "UNAVAILABLE" in str(erro).upper():
        return (
            "Google sobrecarregada no momento. "
            f"Tente de novo em alguns minutos. Detalhe: {erro}"
        )
    return f"Erro da Google: {erro}"


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
    for tentativa in range(1, MAX_TENTATIVAS + 1):
        try:
            response = client.models.generate_content(
                model=GOOGLE_MODEL,
                contents=f"Contexto:\n{contexto}\n\nPergunta: {pergunta}",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.2,
                ),
            )
            break
        except Exception as erro:
            if _eh_transitorio(erro) and tentativa < MAX_TENTATIVAS:
                espera = ESPERA_BASE_S * tentativa
                print(f"Google ocupada (tentativa {tentativa}/{MAX_TENTATIVAS}), tentando de novo em {espera}s...")
                time.sleep(espera)
            else:
                raise ValueError(_mensagem_erro(erro)) from erro

    if not response.text:
        raise ValueError("A Google não retornou texto na resposta.")
    return response.text
