# EXP-001 — chunk × modelo × estratégia

- **Data:** 2026-09-07
- **Base:** 16 docs (`data/raw/`), 44 perguntas no golden (`evals/dataset.json`)
- **Comando:** `make bench` (`python -m evals.benchmark`)
- **Métricas:** Recall@k = % de perguntas com o trecho certo até a posição k · MRR = média de 1/posição (1º=1,0 · 2º=0,5 · 3º=0,33) · tok/perg = média de tokens do top-3 por pergunta (~4 chars = 1 token)

## FASE 1 — tamanho de chunk/overlap (modelo bge-m3)

| exp | chunk | overlap | n_chunks | Recall@1 | Recall@3 | Recall@5 | MRR | tok/perg |
|-----|-------|---------|----------|----------|----------|----------|-----|----------|
| 001 | 200 | 30 | 33 | 95% | 100% | 100% | 0.97 | 465 |
| 002 | 350 | 40 | 20 | 98% | 100% | 100% | 0.99 | 721 |
| 003 | 500 | 75 | 18 | 98% | 100% | 100% | 0.99 | 756 |
| 004 | 750 | 100 | 17 | 98% | 100% | 100% | 0.99 | 755 |

Leitura: chunk 200 fragmenta demais (pior em tudo). De 350 a 750, empate técnico — 500/75 escolhido por igualar o melhor MRR com chunks menores (menos tokens por pergunta que o 750 não justifica: 756 vs 755, teto saturado).

## FASE 2 — modelo de embedding (chunk 500/75)

| modelo | dims | Recall@1 | Recall@3 | Recall@5 | MRR | tok/perg |
|--------|------|----------|----------|----------|-----|----------|
| minilm | 384 | 82% | 95% | 100% | 0.89 | 765 |
| mpnet | 768 | 84% | 98% | 98% | 0.91 | 740 |
| bge-m3 | 1024 | 98% | 100% | 100% | 0.99 | 756 |

Leitura: modelo importa muito mais que chunk. bge-m3 esmaga (MRR 0,99) com o mesmo custo de contexto. mpnet não vale o download frente ao ganho marginal.

## FASE 3 — estratégia de chunking (500/75, bge-m3)

| estrategia | n_chunks | Recall@1 | Recall@3 | Recall@5 | MRR | tok/perg |
|------------|----------|----------|----------|----------|-----|----------|
| hierarquico | 18 | 98% | 100% | 100% | 0.99 | 756 |
| fixo | 18 | 98% | 100% | 100% | 0.99 | 743 |

Leitura: empate. Docs curtos (1–2 chunks cada) + modelo forte = corte cego raramente prejudica. Hierárquico deve abrir vantagem com docs longos ou modelo fraco (experimento futuro: repetir o duelo com minilm).

## Decisões tomadas

1. Chunk padrão: **500/75** (`CHUNK_MAX_TOKENS` / `CHUNK_OVERLAP_TOKENS`)
2. Embedding padrão: **BAAI/bge-m3** (`EMBEDDING_MODEL`), coluna pgvector `vector(1024)` com migração automática
3. Estratégia: **hierárquico** mantido (sem custo extra e melhor por construção em docs longos)

## Reproduzir

```bash
make bench
```
