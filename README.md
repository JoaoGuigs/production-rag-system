# RAG Simples

Projeto educacional de **RAG (Retrieval-Augmented Generation)** em Python. Você envia documentos, o sistema indexa o conteúdo e responde perguntas com base neles, usando busca semântica + Google Gemini.

## Diferenciais

- **Chunking recursivo de verdade** — respeita headings → parágrafos → frases (padrão da indústria, estilo `RecursiveCharacterTextSplitter`), com título da seção grudado no chunk e overlap entre chunks. Tem baseline de janela fixa em `src/chunking/fixed.py` só pra provar o valor no benchmark.
- **Nada de chute: tudo medido** — `evals/` com golden set de 44 perguntas e `make bench` comparando chunk × modelo de embedding × estratégia (Recall@k, MRR, tokens/pergunta). Decisões como bge-m3 + 500/75 saíram de números, e os relatórios ficam guardados em `benchmarks/`.
- **Dois modos de busca** — em memória (zero setup) ou Postgres + pgvector no Docker, com indexação incremental (só o que mudou) e migração automática de dimensão ao trocar de embedding.
- **Resiliente à cota grátis** — retry com backoff em 503/429 do Gemini e mensagens de erro em português.
- **88 testes pytest** que rodam offline (embeddings e Gemini mockados).

## O que foi feito

### Pipeline RAG completo

```
PDF / MD / TXT  →  extrair  →  normalizar  →  chunking  →  embeddings  →  busca  →  Gemini
```

1. **Ingestão** — extrai texto de `.pdf`, `.md` e `.txt`, normaliza e salva em `data/processed/`
2. **Chunking hierárquico** — respeita headings (`#`), parágrafos e frases antes de cortar por tokens (padrão: 500 tokens, overlap de 75; compare com `make bench`)
3. **Embeddings locais** — `BAAI/bge-m3` por padrão, trocável via `EMBEDDING_MODEL` (sem API paga)
4. **Busca semântica** — similaridade de cosseno, top-K = 3, em memória ou no Postgres + pgvector (`USAR_PGVECTOR=true`)
5. **Geração** — Google Gemini responde só com base nos trechos encontrados, com retry automático em 503/429

### Interface web

- Upload de arquivos
- Indexação da base de conhecimento (incremental no modo pgvector: só o que mudou)
- Chat com histórico, resposta do LLM e fontes com % de similaridade

### CLI

- `python main.py` — roda o pipeline e mostra cada etapa no terminal
- `python main.py "pergunta"` — indexa e responde direto no terminal

### Arquitetura

Código organizado por etapa do pipeline:

```
src/
├── config.py, models.py, pipeline.py, cli.py
├── ingestion/      # extrair + normalizar
├── chunking/       # hierárquico + janela fixa (baseline)
├── embedding/      # vetores (troca de modelo em runtime)
├── retrieval/      # busca em memória
├── storage/        # índice persistente (Postgres + pgvector)
├── generation/     # Gemini (com retry)
└── web/            # API FastAPI + store (memória ou pgvector)
```

`main.py` e `run_web.py` só fazem a ponte — a lógica fica nos módulos.

## Comandos (Makefile)

| Comando | O que faz |
|---------|-----------|
| `make back` | sobe o app em http://127.0.0.1:8000 |
| `make front` | abre o navegador no app |
| `make install` | instala dependências e cria o `.env` |
| `make cli` | roda o pipeline no terminal |
| `make test` | suíte de testes (88 testes) |
| `make eval` | provinha do retrieval (44 perguntas, com gate) |
| `make bench` | benchmark: chunk × modelo × estratégia |
| `make db` | sobe o Postgres + pgvector |

## Evals e benchmark

- `evals/dataset.json` — golden set: pergunta → trechos esperados (âncoras, robustas a mudanças de chunking)
- `make eval` — hit@3 + MRR, reprova abaixo do threshold
- `make bench` — 3 fases: tamanho de chunk, modelo de embedding e estratégia (hierárquico vs janela fixa), com Recall@k, MRR e tokens/pergunta

Último resultado (16 docs, 44 perguntas): bge-m3 com chunk 500/75 — Recall@3 = 100%, MRR = 0.99.

## Como rodar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
# ou: make install
```

### 2. Configurar a API do Gemini

Copie o exemplo e coloque sua chave (grátis em [aistudio.google.com/apikey](https://aistudio.google.com/apikey)):

```bash
cp .env.example .env
```

Edite o `.env`:

```env
GOOGLE_API_KEY=sua-chave-aqui
GOOGLE_MODEL=gemini-3.1-flash-lite
```

### 3. (Opcional) Subir o banco vetorial

```bash
make db
```

E no `.env`: `USAR_PGVECTOR=true`. Sem isso, o índice fica em memória.

### 4. Subir a interface web

```bash
python run_web.py
# ou: make back (e make front no outro terminal para abrir o navegador)
```

Abra [http://127.0.0.1:8000](http://127.0.0.1:8000), envie arquivos, clique em **Indexar** e faça perguntas.

### 5. Ou usar pelo terminal

```bash
python main.py
python main.py "quantos dias de férias?"
```

## Stack

| Parte | Tecnologia |
|-------|------------|
| API | FastAPI + Uvicorn |
| Embeddings | sentence-transformers, bge-m3 por padrão (local) |
| Banco vetorial | Postgres + pgvector (opcional) |
| LLM | Google Gemini (`google-genai`) |
| PDF | pypdf |
| Testes | pytest (88 testes, mocks — roda offline) |
| Frontend | HTML, CSS e JavaScript vanilla |

## Observações

- Sem pgvector, o índice fica **em memória** — ao reiniciar o servidor, é preciso indexar de novo. Com `USAR_PGVECTOR=true`, o índice persiste e a reindexação é incremental (hash por arquivo + modelo).
- Arquivos em `data/processed/` são gerados automaticamente; não vão pro Git.
- PDFs sem markdown (`#`, `\n\n`) caem no fallback do chunking por frases/tokens.
- Na primeira execução, cada modelo de embeddings é baixado do Hugging Face (bge-m3 tem ~2GB).
- `CHUNK_MAX_TOKENS`, `CHUNK_OVERLAP_TOKENS` e `EMBEDDING_MODEL` aceitam override via env.
- `listar_arquivos_originais` não lê subpastas — deixe os docs soltos em `data/raw/`.

## Estrutura do repositório

```
.
├── main.py              # entrada CLI
├── run_web.py           # servidor web
├── Makefile             # comandos (back/front/test/eval/bench/db)
├── docker-compose.yml   # Postgres + pgvector
├── requirements.txt
├── .env.example         # modelo de variáveis (sem chave real)
├── data/
│   ├── raw/             # documentos originais (16 docs de RH + CV)
│   └── processed/       # gerado pelo pipeline (ignorado no Git)
├── evals/               # golden set + runner + benchmark
├── tests/               # suíte pytest (88 testes)
├── scripts/             # front.py, ensure_env.py
├── static/              # frontend
└── src/                 # código do RAG
```
