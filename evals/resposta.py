"""Eval de RESPOSTA (nível LLM): roda pipeline + Gemini nas perguntas do golden set.

Métricas v1 (heurísticas determinísticas, sem LLM-as-judge):
- respondíveis: âncora na resposta (acerto) + fonte citada (citação)
- sem resposta: frase de recusa E nenhuma âncora alucinada (recusa correta)

Uso:
    python -m evals.resposta   # ou: make eval-resposta

Custo: ~1 chamada ao Gemini por pergunta (~9% da cota diária gratuita).
Roda sequencial com pausa entre chamadas para não estourar o RPM.
"""

import re
import time

from evals.run import carregar_dataset
from src.generation.llm import gerar_resposta
from src.pipeline import executar_pipeline
from src.retrieval.search import buscar

ESPERA_ENTRE_CHAMADAS_S = 4
TOP_K = 3

FRASES_RECUSA = (
    "não encontrei",
    "nao encontrei",
    "não há informação",
    "não consta",
    "não possui",
    "não sei",
    "não tenho",
    "nao sei",
    "não é possível responder",
    "não posso responder",
    "sem informação",
    "não menciona",
    "não há menção",
)


def contem_ancora(resposta: str, ancoras: list[str]) -> bool:
    texto = resposta.lower()
    return any(a.lower() in texto for a in ancoras)


def eh_recusa(resposta: str) -> bool:
    texto = resposta.lower()
    return any(frase in texto for frase in FRASES_RECUSA)


def afirma_sem_recusa(resposta: str, ancoras: list[str]) -> bool:
    """Alucinação de verdade: frase que contém a âncora SEM frase de recusa junto.

    Evita o falso positivo de "Não encontrei informação sobre X" (menciona
    X recusando, não afirmando).
    """
    frases = [f.strip() for f in re.split(r"[.!?…\n]+", resposta) if f.strip()]
    return any(
        contem_ancora(frase, ancoras) and not eh_recusa(frase) for frase in frases
    )


def tem_citacao(resposta: str, fontes: list[str]) -> bool:
    texto = resposta.lower()
    return any(fonte.lower() in texto for fonte in fontes)


def avaliar_item(item: dict, documentos) -> dict:
    resultados = buscar(item["pergunta"], documentos, top_k=TOP_K)
    try:
        resposta = gerar_resposta(item["pergunta"], resultados)
    except Exception as erro:
        return {"id": item["id"], "tipo": "erro", "acerto": 0.0, "detalhe": f"ERRO: {erro}"}

    if item.get("modo", "recuperar") == "ausencia":
        recusa = eh_recusa(resposta)
        alucinou = afirma_sem_recusa(resposta, item["ancoras"])
        return {
            "id": item["id"],
            "tipo": "recusa",
            "acerto": 1.0 if (recusa and not alucinou) else 0.0,
            "detalhe": f"recusa={int(recusa)} alucinou={int(alucinou)} | {resposta[:80]}",
        }
    acerto = contem_ancora(resposta, item["ancoras"])
    citacao = tem_citacao(resposta, item["fontes_esperadas"])
    return {
        "id": item["id"],
        "tipo": "ok",
        "acerto": 1.0 if acerto else 0.0,
        "citacao": 1.0 if citacao else 0.0,
        "detalhe": f"ancora={int(acerto)} citacao={int(citacao)} | {resposta[:80]}",
    }


def main() -> int:
    dataset = carregar_dataset()
    meta = dataset.get("_meta", {})
    print("Indexando documentos ...")
    documentos = executar_pipeline()

    linhas = []
    itens = dataset["itens"]
    for i, item in enumerate(itens):
        print(f"[{i + 1}/{len(itens)}] {item['id']} ...")
        linhas.append(avaliar_item(item, documentos))
        if i < len(itens) - 1:
            time.sleep(ESPERA_ENTRE_CHAMADAS_S)

    print(f"\n{'id':24} {'tipo':7} {'acerto':6} detalhe")
    for linha in linhas:
        print(f"{linha['id']:24} {linha['tipo']:7} {linha['acerto']:<6.0f} {linha['detalhe']}")

    resp = [l for l in linhas if l["tipo"] == "ok"]
    recs = [l for l in linhas if l["tipo"] == "recusa"]
    taxa_acerto = sum(l["acerto"] for l in resp) / len(resp) if resp else 1.0
    taxa_citacao = sum(l.get("citacao", 0) for l in resp) / len(resp) if resp else 1.0
    taxa_recusa = sum(l["acerto"] for l in recs) / len(recs) if recs else 1.0
    print(
        f"\nRespondíveis ({len(resp)}): acerto={taxa_acerto:.0%} citacao={taxa_citacao:.0%} | "
        f"Sem resposta ({len(recs)}): recusa_correta={taxa_recusa:.0%}"
    )

    ok = True
    for nome, valor in (("threshold_acerto", taxa_acerto), ("threshold_recusa", taxa_recusa)):
        if nome in meta and valor < meta[nome]:
            print(f"FAIL: {nome}={valor:.2f} abaixo de {meta[nome]}")
            ok = False
    if ok:
        print("PASS")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
