"""Persistência, replay e apresentação dos relatórios de avaliação."""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from evals.metrics import agregar, avaliar_gates, pontuar_resposta

RESULTADOS_DIR = Path(__file__).resolve().parent / "results"


def git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def salvar_relatorio(relatorio: dict, destino: Path | None = None) -> Path:
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    if destino is None:
        instante = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destino = RESULTADOS_DIR / f"answer-eval-{instante}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    return destino


def _fmt(valor: float | None) -> str:
    return "n/a" if valor is None else f"{valor:.1%}"


def executar_replay(args: argparse.Namespace, dataset: dict, dataset_path: Path) -> tuple[dict, list[str], Path]:
    if args.judge:
        raise ValueError("--judge não pode ser combinado com --replay")
    origem_path = Path(args.replay)
    with origem_path.open(encoding="utf-8") as arquivo:
        origem = json.load(arquivo)
    itens = {item["id"]: item for item in dataset["items"]}
    registros = []
    for anterior in origem["results"]:
        registro = dict(anterior)
        if registro["status"] == "ok":
            item = itens.get(registro["id"])
            if item is None:
                raise ValueError(f"caso {registro['id']} não existe no dataset atual")
            registro.update(pontuar_resposta(item, registro["answer"], registro["retrieved_chunks"]))
        registros.append(registro)
    resumo = agregar(registros)
    falhas = avaliar_gates(resumo, dataset["_meta"].get("thresholds", {}))
    relatorio = {
        **origem,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "replay_of": str(origem_path),
        "dataset": {"path": str(dataset_path), "name": dataset["_meta"].get("name"), "version": 2},
        "summary": resumo,
        "gate": {"passed": not falhas, "failures": falhas},
        "results": registros,
    }
    destino = salvar_relatorio(relatorio, Path(args.output) if args.output else None)
    return relatorio, falhas, destino


def exibir_relatorio(relatorio: dict, falhas: list[str], destino: Path, usar_judge: bool = False) -> None:
    r, k = relatorio["summary"], relatorio["configuration"]["top_k"]
    print("\nRAG EVALUATION V2")
    print(f"Retrieval  Recall@{k}: {_fmt(r['retrieval_recall_at_k'])} | MRR: {_fmt(r['retrieval_mrr'])}")
    print(f"Generation Accuracy: {_fmt(r['answer_accuracy'])} | Completeness: {_fmt(r['fact_completeness'])}")
    print(f"Citations   Expected: {_fmt(r['expected_citation_rate'])} | Precision: {_fmt(r['citation_precision'])}")
    print(f"Safety      Correct refusal: {_fmt(r['correct_refusal_rate'])} | Hallucination: {_fmt(r['hallucination_rate'])}")
    print(f"Reliability Errors: {r['errors']}/{r['total']} | Error rate: {_fmt(r['error_rate'])}")
    print(f"Latency     Average: {r['latency_ms']['average']:.0f} ms | p95: {r['latency_ms']['p95']:.0f} ms")
    if usar_judge:
        j = r["judge"]
        print(f"LLM judge   Correctness: {_fmt(j['correctness'])} | Groundedness: {_fmt(j['groundedness'])} | Citations: {_fmt(j['citation_correctness'])}")
    print(f"Artifact    {destino}")
    for falha in falhas:
        print(f"FAIL: {falha}")
    if not falhas:
        print("PASS")
