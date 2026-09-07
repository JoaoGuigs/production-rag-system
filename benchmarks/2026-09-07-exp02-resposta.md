# EXP-002 — eval de resposta (nível LLM)

- **Data:** 2026-09-07
- **Base:** 16 docs, 44 perguntas (42 respondíveis + 2 sem resposta)
- **Comando:** `make eval-resposta` (`python -m evals.resposta`)
- **Custo:** 44 chamadas ao Gemini 3.1-flash-lite (~9% da cota diária), sequencial com 4s de intervalo
- **Método:** heurísticas determinísticas v1 — âncora na resposta (acerto), fonte citada (citação), frase de recusa sem afirmação (recusa correta). Sem LLM-as-judge (evolução futura).

## Resultado

| grupo | n | métrica | valor |
|-------|---|---------|-------|
| respondíveis | 42 | taxa de acerto (âncora na resposta) | 98% |
| respondíveis | 42 | taxa de citação (fonte mencionada) | 100% |
| sem resposta | 2 | taxa de recusa correta | 100% |

Gate calibrado no `dataset.json`: `threshold_acerto = 0.9`, `threshold_recusa = 1.0`.

## Achados

1. **Falso positivo da 1ª versão:** as recusas corretas ("Não encontrei informação sobre bônus anual ou PLR...") mencionam os termos proibidos e foram marcadas como alucinação. Correção: alucinação = frase com âncora **sem** frase de recusa junto (nível de frase, não de resposta). Recusa foi de 0% → 100% só com o fix da métrica.
2. **Único erro real (recesso-dias):** resposta listou os dias mas sem a âncora exata "26 de dezembro" — fragilidade da âncora, não necessariamente resposta errada. Candidata a revisão do golden set (aceitar "26/12" como âncora alternativa).
3. **Retry funcionou ao vivo:** 5 episódios de 503 da Google absorvidos com backoff, zero falhas.

## Limites conhecidos (v1)

- Âncora na resposta ≠ resposta 100% correta (proxy, pode dar falso positivo).
- Citação checa menção ao nome do arquivo, não se o trecho citado sustenta a frase.
- Groundedness/faithfulness de verdade exigem LLM-as-judge ou NLI — próximo passo.

## Reproduzir

```bash
make eval-resposta
```
