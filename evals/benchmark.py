"""Mini-benchmark: varia chunk/overlap e compara Recall@3/Recall@5.

Uso:
    python -m evals.benchmark   # ou: make bench

Reaproveita o golden set do eval (sem LLM, só retrieval).
"""

import src.chunking.chunker as chunker
from evals.run import carregar_dataset, chunk_relevante
from src import config as cfg
from src.embedding import embedder
from src.pipeline import executar_pipeline
from src.retrieval.search import buscar

EXPERIMENTOS = [
    ("001", 200, 30),
    ("002", 350, 40),
    ("003", 500, 75),
    ("004", 750, 100),
]

MODELOS = [
    ("minilm", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    ("mpnet", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
    ("bge-m3", "BAAI/bge-m3"),
]


def avaliar(documentos, itens: list, top_k: int = 5) -> tuple[float, float, float, float, float]:
    r1, r3, r5, rrs, toks = [], [], [], [], []
    for item in itens:
        resultados = buscar(item["pergunta"], documentos, top_k=top_k)
        ranks = [
            i + 1
            for i, r in enumerate(resultados)
            if chunk_relevante(r.chunk_embedado.chunk, item)
        ]
        toks.append(sum(len(r.chunk_embedado.chunk.texto) for r in resultados[:3]) // 4)
        if item.get("modo", "recuperar") == "ausencia":
            r1.append(0.0 if 1 in ranks else 1.0)
            r3.append(0.0 if any(rk <= 3 for rk in ranks) else 1.0)
            r5.append(0.0 if ranks else 1.0)
            rrs.append(0.0 if ranks else 1.0)
        else:
            r1.append(1.0 if 1 in ranks else 0.0)
            r3.append(1.0 if any(rk <= 3 for rk in ranks) else 0.0)
            r5.append(1.0 if ranks else 0.0)
            rrs.append(1.0 / min(ranks) if ranks else 0.0)
    n = len(itens)
    return sum(r1) / n, sum(r3) / n, sum(r5) / n, sum(rrs) / n, sum(toks) / n


def main() -> None:
    itens = carregar_dataset()["itens"]
    print("FASE 1 — chunk/overlap (modelo padrão)")
    print(f"{'exp':5} {'chunk':6} {'overlap':7} {'n_chunks':8} {'Recall@1':8} {'Recall@3':8} {'Recall@5':8} {'MRR':5} tok/perg")
    orig_max, orig_overlap = chunker.CHUNK_MAX_TOKENS, chunker.CHUNK_OVERLAP_TOKENS
    try:
        for nome, chunk, overlap in EXPERIMENTOS:
            chunker.CHUNK_MAX_TOKENS = chunk
            chunker.CHUNK_OVERLAP_TOKENS = overlap
            documentos = executar_pipeline()
            n_chunks = sum(len(d.chunks_embedados) for d in documentos)
            recall1, recall3, recall5, mrr, tok = avaliar(documentos, itens)
            print(f"{nome:5} {chunk:<6} {overlap:<7} {n_chunks:<8} {recall1:<8.0%} {recall3:<8.0%} {recall5:<8.0%} {mrr:<5.2f} {tok:.0f}")
    finally:
        chunker.CHUNK_MAX_TOKENS, chunker.CHUNK_OVERLAP_TOKENS = orig_max, orig_overlap

    print(f"\nFASE 2 — modelo de embedding (chunk {orig_max}/{orig_overlap})")
    print(f"{'modelo':8} {'dims':5} {'Recall@1':8} {'Recall@3':8} {'Recall@5':8} {'MRR':5} tok/perg")
    orig_modelo = cfg.EMBEDDING_MODEL
    try:
        for apelido, modelo in MODELOS:
            cfg.EMBEDDING_MODEL = modelo
            documentos = executar_pipeline()
            dims = embedder._obter_modelo().get_embedding_dimension()
            recall1, recall3, recall5, mrr, tok = avaliar(documentos, itens)
            print(f"{apelido:8} {dims:<5} {recall1:<8.0%} {recall3:<8.0%} {recall5:<8.0%} {mrr:<5.2f} {tok:.0f}")
    finally:
        cfg.EMBEDDING_MODEL = orig_modelo


if __name__ == "__main__":
    main()
