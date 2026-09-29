"""Comando e orquestração da avaliação end-to-end v2 do RAG."""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

from evals.dataset import DATASET_PADRAO, carregar_dataset
from evals.judge import julgar
from evals.metrics import agregar, avaliar_gates, pontuar_resposta
from evals.report import executar_replay, exibir_relatorio, git_sha, salvar_relatorio
from src import config
from src.generation.llm import gerar_resposta
from src.pipeline import executar_pipeline
from src.retrieval.search import buscar

ESPERA_ENTRE_CHAMADAS_S = 4.0


def _serializar_chunks(resultados) -> list[dict]:
    return [{
        "rank": rank, "fonte": r.chunk_embedado.chunk.fonte,
        "indice": r.chunk_embedado.chunk.indice,
        "similaridade": round(float(r.similaridade), 6),
        "texto": r.chunk_embedado.chunk.texto,
    } for rank, r in enumerate(resultados, 1)]


def avaliar_item(item: dict, documentos, top_k: int = 3, usar_judge: bool = False) -> dict:
    inicio_total = time.perf_counter()
    base = {k: item.get(k) for k in ("id", "category", "question", "answerable", "expected_answer")}
    try:
        inicio = time.perf_counter()
        resultados = buscar(item["question"], documentos, top_k=top_k)
        retrieval_ms = (time.perf_counter() - inicio) * 1000
        chunks = _serializar_chunks(resultados)
        inicio = time.perf_counter()
        resposta = gerar_resposta(item["question"], resultados)
        generation_ms = (time.perf_counter() - inicio) * 1000

        registro = {
            **base, "status": "ok", "answer": resposta, "retrieved_chunks": chunks,
            **pontuar_resposta(item, resposta, chunks),
            "latency_ms": {"retrieval": round(retrieval_ms, 2), "generation": round(generation_ms, 2)},
        }
        if usar_judge:
            inicio = time.perf_counter()
            registro["judge"] = julgar(item["question"], item.get("expected_answer"),
                                        item["answerable"], resposta, chunks)
            registro["latency_ms"]["judge"] = round((time.perf_counter() - inicio) * 1000, 2)
        registro["latency_ms"]["total"] = round((time.perf_counter() - inicio_total) * 1000, 2)
        return registro
    except Exception as erro:
        return {**base, "status": "error", "error": f"{type(erro).__name__}: {erro}",
                "answer": None, "retrieved_chunks": [], "retrieval_hit": None,
                "reciprocal_rank": None, "metrics": {},
                "latency_ms": {"total": round((time.perf_counter() - inicio_total) * 1000, 2)}}


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Avaliação end-to-end v2 do RAG")
    parser.add_argument("--dataset", default=str(DATASET_PADRAO))
    parser.add_argument("--output")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--interval", type=float, default=ESPERA_ENTRE_CHAMADAS_S)
    parser.add_argument("--judge", action="store_true", help="usa juiz LLM; dobra as chamadas")
    parser.add_argument("--replay", help="recalcula métricas de um artefato sem chamar o RAG")
    return parser


def executar(args: argparse.Namespace) -> tuple[dict, list[str], Path]:
    dataset_path, dataset = Path(args.dataset), carregar_dataset(Path(args.dataset))
    if args.replay:
        return executar_replay(args, dataset, dataset_path)
    meta = dataset["_meta"]
    top_k = args.top_k or int(meta.get("top_k", config.TOP_K))
    print("Indexando documentos ...")
    inicio = time.perf_counter()
    documentos = executar_pipeline()
    corpus_extra = config.BASE_DIR / "data" / "eval_raw"
    if corpus_extra.exists():
        documentos.extend(executar_pipeline(corpus_extra))
    indexacao_ms = (time.perf_counter() - inicio) * 1000
    registros = []
    for i, item in enumerate(dataset["items"], 1):
        print(f"[{i}/{len(dataset['items'])}] {item['id']} ...")
        registros.append(avaliar_item(item, documentos, top_k, args.judge))
        if i < len(dataset["items"]) and args.interval > 0:
            time.sleep(args.interval)
    resumo = agregar(registros, args.judge)
    falhas = avaliar_gates(resumo, meta.get("thresholds", {}), args.judge)
    relatorio = {
        "schema_version": 2, "created_at": datetime.now(timezone.utc).isoformat(), "git_sha": git_sha(),
        "dataset": {"path": str(dataset_path), "name": meta.get("name"), "version": 2},
        "configuration": {"embedding_model": config.EMBEDDING_MODEL, "llm_model": config.GOOGLE_MODEL,
                          "chunk_max_tokens": config.CHUNK_MAX_TOKENS,
                          "chunk_overlap_tokens": config.CHUNK_OVERLAP_TOKENS,
                          "top_k": top_k, "judge_enabled": args.judge},
        "indexing": {"documents": len(documentos),
                     "chunks": sum(len(d.chunks_embedados) for d in documentos),
                     "latency_ms": round(indexacao_ms, 2)},
        "summary": resumo, "gate": {"passed": not falhas, "failures": falhas}, "results": registros,
    }
    destino = salvar_relatorio(relatorio, Path(args.output) if args.output else None)
    return relatorio, falhas, destino


def main(argv: list[str] | None = None) -> int:
    args = criar_parser().parse_args(argv)
    relatorio, falhas, destino = executar(args)
    exibir_relatorio(relatorio, falhas, destino, args.judge)
    return int(bool(falhas))


if __name__ == "__main__":
    raise SystemExit(main())
