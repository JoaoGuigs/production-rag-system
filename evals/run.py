"""Provinha do retrieval: roda o golden set e dá nota (hit@k + MRR).

Uso:
    python -m evals.run   # ou: make eval

Retorna exit 0 se hit@3 >= threshold, 1 caso contrário.
"""

import json
import sys
from pathlib import Path

from src.pipeline import executar_pipeline
from src.retrieval.search import buscar

AQUI = Path(__file__).resolve().parent


def carregar_dataset() -> dict:
    with (AQUI / "dataset.json").open(encoding="utf-8") as f:
        return json.load(f)


def chunk_relevante(chunk, item: dict) -> bool:
    """Diz se o chunk é o trecho esperado da pergunta."""
    tem_ancora = any(a.lower() in chunk.texto.lower() for a in item["ancoras"])
    if not tem_ancora:
        return False
    if item.get("modo", "recuperar") == "ausencia":
        # Achou uma afirmação que não existe nos docs: ruim.
        return True
    return chunk.fonte in item["fontes_esperadas"]


def avaliar_item(item: dict, documentos, top_k: int) -> dict:
    resultados = buscar(item["pergunta"], documentos, top_k=top_k)
    ranks = [
        i + 1
        for i, r in enumerate(resultados)
        if chunk_relevante(r.chunk_embedado.chunk, item)
    ]
    ausencia = item.get("modo", "recuperar") == "ausencia"
    if ausencia:
        # Aqui "acertar" é NÃO trazer nada: inverte a lógica.
        hit_1, hit_k, rr = (1.0 if 1 not in ranks else 0.0), (1.0 if not ranks else 0.0), (1.0 if not ranks else 0.0)
    else:
        hit_1, hit_k = (1.0 if 1 in ranks else 0.0), (1.0 if ranks else 0.0)
        rr = 1.0 / min(ranks) if ranks else 0.0

    top1 = resultados[0] if resultados else None
    return {
        "id": item["id"],
        "modo": item.get("modo", "recuperar"),
        "hit@1": hit_1,
        "hit@3": hit_k,
        "rr": rr,
        "top1_fonte": top1.chunk_embedado.chunk.fonte if top1 else "-",
        "top1_sim": round(top1.similaridade, 3) if top1 else 0.0,
    }


def main() -> int:
    dataset = carregar_dataset()
    meta = dataset.get("_meta", {})
    top_k = int(meta.get("top_k", 3))
    threshold = float(meta.get("threshold_hit_at_3", 0.8))

    print("Indexando documentos de data/raw/ ...")
    documentos = executar_pipeline()
    total_chunks = sum(len(d.chunks_embedados) for d in documentos)
    print(f"Indexados: {len(documentos)} docs, {total_chunks} chunks\n")

    linhas = [avaliar_item(item, documentos, top_k) for item in dataset["itens"]]

    print(f"{'id':22} {'modo':9} {'hit@1':6} {'hit@3':6} {'RR':5} top-1")
    for linha in linhas:
        print(
            f"{linha['id']:22} {linha['modo']:9} "
            f"{linha['hit@1']:6.1f} {linha['hit@3']:6.1f} {linha['rr']:5.2f} "
            f"{linha['top1_fonte']} ({linha['top1_sim']})"
        )

    n = len(linhas)
    medias = {m: sum(l[m] for l in linhas) / n for m in ("hit@1", "hit@3", "rr")}
    print(
        f"\nAgregado ({n} perguntas): "
        f"hit@1={medias['hit@1']:.2f}  hit@3={medias['hit@3']:.2f}  MRR={medias['rr']:.2f}"
    )

    if medias["hit@3"] >= threshold:
        print(f"PASS: hit@3 >= {threshold}")
        return 0
    print(f"FAIL: hit@3={medias['hit@3']:.2f} abaixo do threshold {threshold}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
