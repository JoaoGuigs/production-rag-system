# EXP-003 — avaliação end-to-end v2

- **Data:** 2026-09-08
- **Base:** 20 documentos, 27 chunks
- **Dataset:** 24 perguntas independentes do golden de retrieval (16 respondíveis + 8 sem resposta)
- **Modelo:** bge-m3 + Gemini 3.1 Flash Lite
- **Configuração:** chunk 500/75, top-k 3
- **Comando:** `make eval-resposta`
- **Judge:** desativado; métricas determinísticas

## Resultado

| dimensão | métrica | valor |
|----------|---------|-------|
| Retrieval | Recall@3 | 100,0% |
| Retrieval | MRR | 1,00 |
| Generation | Answer accuracy | 93,8% |
| Generation | Fact completeness | 95,8% |
| Citações | Fonte esperada citada | 100,0% |
| Citações | Precisão das fontes citadas | 100,0% |
| Segurança | Recusa correta | 100,0% |
| Segurança | Alucinação heurística | 0,0% |
| Confiabilidade | Erros | 0/24 |
| Latência | Média | 6,5 s |
| Latência | p95 | 44,3 s |

Gate aprovado em todas as métricas.

## O que mudou em relação à v1

- Dataset próprio para resposta, sem reutilizar as 44 perguntas do golden de retrieval.
- Cada caso respondível exige vários fatos, em vez de uma única palavra-chave.
- Oito casos negativos incluem assuntos próximos aos documentos, como certificação versus bolsa universitária e combustível de viagem versus auxílio mensal.
- Quatro manuais consolidados adicionam redundância e vocabulário semelhante ao corpus de avaliação.
- Erros de retrieval, embedding, API ou judge entram na taxa de erro e reprovam o gate.
- Citações usam o formato `[arquivo.ext]`; o runner mede presença da fonte esperada e precisão contra o contexto recuperado.
- Cada execução salva perguntas, respostas, chunks, scores, configuração, commit, erros e latências em `evals/results/`.
- `--replay` permite calibrar métricas usando as mesmas respostas, sem novas chamadas ao Gemini.
- `--judge` adiciona correctness, groundedness e citation correctness por LLM quando desejado.

## Falha encontrada

`home-office-elegibilidade` respondeu corretamente que quatro meses não bastam, mas omitiu a aprovação do gestor e o limite de quatro dias remotos por semana. O retrieval trouxe a evidência correta; a falha foi de completude da geração.

## Calibração do avaliador

A primeira leitura marcou 56,2% de accuracy e 31,2% de citação. A inspeção mostrou falsos negativos: o Gemini agrupou fontes em `[a.md, b.md]`, usou flexões verbais e expressou ausência como “não oferece”. O parser e as alternativas do golden foram corrigidos, e o artefato original foi reavaliado com `--replay`. Nenhuma resposta foi regenerada durante a calibração.

## Próximo experimento

O retrieval ainda está saturado em 100%. O próximo conjunto deve incluir mais chunks, entidades com nomes e valores próximos e consultas lexicais difíceis. Com essa base, comparar busca densa, BM25 + dense por RRF e reranking passa a produzir evidência útil.
