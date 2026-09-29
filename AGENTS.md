# RAG Lab — instruções para agentes

## Objetivo e colaboração

Este projeto é um laboratório de RAG para estudo e portfólio de engenharia de IA. Priorize código que o autor consiga entender, explicar em entrevista e avaliar com experimentos reproduzíveis.

- Converse em português brasileiro, de forma direta. Explique brevemente as decisões relevantes e o que foi validado.
- Em pedidos de implementação, conclua a mudança e os testes pertinentes. Resolva escolhas rotineiras sem pedir confirmação repetida; esclareça apenas ambiguidades que alterem materialmente o resultado.
- Mantenha o escopo solicitado. é bem vindo menciar e dar dicas sobre frameworks relacionados para melhorar a qualidade do codigo.
- Em tarefas longas, dê atualizações curtas sobre descobertas e próximos passos. Ao entregar, informe mudanças, testes executados e limitações reais.
-Use subagents para tarefas mais repetitivas e que nao exigem tanto esforço.
## Mapa do projeto

- `src/ingestion/`, `chunking/`, `embedding/`, `retrieval/` e `generation/`: etapas do RAG.
- `src/pipeline.py`: orquestra a indexação; `src/models.py`: modelos de dados compartilhados.
- `src/storage/vetores.py`: persistência e busca no Postgres com pgvector.
- `src/web/`: API FastAPI e estado da aplicação; `static/`: frontend HTML/CSS/JavaScript.
- `evals/resposta.py`: comando e execução da avaliação end-to-end.
- `evals/dataset.py`: leitura e validação do dataset de respostas.
- `evals/metrics.py`: heurísticas, agregação e critérios de aprovação.
- `evals/report.py`: persistência, apresentação e replay dos resultados.
- `evals/judge.py`: juiz LLM opcional; `evals/run.py` e `benchmark.py`: retrieval e experimentos.
- `data/raw/`: corpus da aplicação; `data/eval_raw/`: documentos adicionais da avaliação.
- `tests/`: testes offline; `benchmarks/`: relatórios de experimentos, com contexto e limitações.

## Organização do código

- Preserve a separação por responsabilidade. Comandos devem coordenar funções; métricas e validações ficam em módulos próprios.
- Prefira funções claras, nomes descritivos e tipos nas interfaces. Siga o idioma e as convenções do módulo existente.
- Se um arquivo passar a misturar execução, métricas, persistência e apresentação, separe as responsabilidades antes de ampliá-lo. Evite também fragmentar funções triviais em muitos arquivos.
- Reutilize `src/models.py` e `tests/helpers.py` quando apropriado. Evite dependências circulares e módulos genéricos de utilidades sem propósito definido.
- Refatorações devem preservar comandos, formatos de relatório e comportamento, salvo mudança explicitamente necessária à tarefa.

## Comandos e validação

Execute a partir da raiz, com o Python do ambiente que possui as dependências:

| Finalidade | Comando |
| --- | --- |
| Testes offline | `python -m pytest -q` |
| API local | `python run_web.py` |
| Retrieval | `python -m evals.run` |
| Benchmark de retrieval | `python -m evals.benchmark` |
| Avaliação com Gemini | `python -m evals.resposta` |
| Avaliação com juiz LLM | `python -m evals.resposta --judge` |
| Recalcular artefato | `python -m evals.resposta --replay CAMINHO.json` |
| Banco local | `docker compose up -d` |

- O `Makefile` oferece atalhos; não presuma que `make` esteja instalado no Windows.
- Se `python` não estiver no PATH, procure o ambiente virtual ou a instalação local antes de concluir que Python está ausente. Use a sintaxe do PowerShell e respeite as permissões do ambiente.
- Para mudanças de código, rode os testes pertinentes. Em refatorações entre módulos, rode a suíte offline completa e `git diff --check`.
- Use mocks para Gemini, embeddings e banco nos testes unitários. Não introduza chamadas externas na suíte offline.
- Prefira replay para validar mudanças nas métricas. Avaliações reais usam API e corpus; não são necessárias para uma refatoração puramente estrutural.
- Ao rodar uma avaliação real dentro do escopo autorizado, informe o custo em chamadas e respeite o intervalo configurado. O juiz adiciona chamadas; retries podem aumentar o total.
- Nunca declare um teste ou benchmark aprovado sem observar o resultado da execução.

## Integridade das avaliações

- Diferencie testes de código, avaliação heurística e julgamento semântico por LLM.
- Presença de palavras não prova correção; fonte existente no contexto não prova que ela sustenta cada afirmação. Não apresente essas proxies como groundedness real.
- Erros e casos não avaliados devem ficar visíveis e influenciar a confiabilidade do relatório; conjunto vazio não é sucesso.
- Não reduza thresholds nem adapte referências apenas para obter PASS. Correções de rótulos e heurísticas exigem justificativa e registro do efeito nas métricas.
- Se o dataset foi usado para calibrar o avaliador, trate-o como conjunto de desenvolvimento. Reserve casos independentes antes de alegar generalização.
- Ao adicionar documentos, confira evidências, duplicatas e contradições. Não coloque respostas artificiais aos casos negativos no corpus só para facilitar a avaliação.
- Registre modelo, configuração, corpus, quantidade de casos e método nos experimentos. Preserve resultados históricos e identifique replays.

## Arquivos, segurança e entrega

- Confira `git status` e preserve alterações existentes, inclusive arquivos não rastreados.
- Nunca exponha chaves de API ou conteúdo de `.env` em saída, testes, commits ou relatórios. Use `.env.example` para documentar configuração.
- `data/processed/` e `evals/results/` contêm arquivos gerados. Evite versioná-los automaticamente; relatórios selecionados pertencem a `benchmarks/`.
- Não faça commit, push, deploy ou limpeza destrutiva sem que isso esteja no escopo autorizado.
- Atualize README e comandos quando necessário. Mantenha este arquivo como instruções duráveis; resultados e histórico ficam nos relatórios.

## Referências de estrutura

Adaptado ao RAG Lab a partir do [guia oficial de AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md) e do [exemplo público do Codex](https://github.com/openai/codex/blob/main/AGENTS.md). As regras acima são específicas deste projeto; não herdam os comandos nem as políticas daquele repositório.
