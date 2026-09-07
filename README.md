# RAG Simples

Projeto educacional de **RAG (Retrieval-Augmented Generation)** em Python. Você envia documentos, o sistema indexa o conteúdo e responde perguntas com base neles, usando busca semântica + Google Gemini.

## O que foi feito

### Pipeline RAG completo

```
PDF / MD / TXT  →  extrair  →  normalizar  →  chunking  →  embeddings  →  busca  →  Gemini
```

1. **Ingestão** — extrai texto de `.pdf`, `.md` e `.txt`, normaliza e salva em `data/processed/`
2. **Chunking hierárquico** — respeita headings (`#`), parágrafos e frases antes de cortar por tokens (~350 tokens, overlap de 40)
3. **Embeddings locais** — `BAAI/bge-m3` (sem API paga; veja `make bench`)
4. **Busca semântica** — similaridade de cosseno, top-K = 3
5. **Geração** — Google Gemini responde só com base nos trechos encontrados

### Interface web

- Upload de arquivos
- Indexação da base de conhecimento
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
├── chunking/       # dividir em chunks
├── embedding/      # vetores
├── retrieval/      # busca
├── generation/     # Gemini
└── web/            # API FastAPI + estado em memória
```

`main.py` e `run_web.py` só fazem a ponte — a lógica fica nos módulos.

## Como rodar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
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

### 3. Subir a interface web

```bash
python run_web.py
```

Abra [http://127.0.0.1:8000](http://127.0.0.1:8000), envie arquivos, clique em **Indexar** e faça perguntas.

### 4. Ou usar pelo terminal

```bash
python main.py
python main.py "quantos dias de férias?"
```

Coloque seus documentos em `data/raw/` (há `exemplo.md` e `exemplo.txt` de demonstração).

## Stack

| Parte | Tecnologia |
|-------|------------|
| API | FastAPI + Uvicorn |
| Embeddings | sentence-transformers (local) |
| LLM | Google Gemini (`google-genai`) |
| PDF | pypdf |
| Frontend | HTML, CSS e JavaScript vanilla |

## Observações

- O índice fica **em memória** — ao reiniciar o servidor, é preciso indexar de novo.
- Arquivos em `data/processed/` são gerados automaticamente; não precisam ir pro Git.
- PDFs não têm markdown (`#`, `\n\n`), então o chunking cai no fallback por frases/tokens.
- Na primeira execução, o modelo de embeddings é baixado do Hugging Face (pode demorar um pouco).

## Estrutura do repositório

```
.
├── main.py              # entrada CLI
├── run_web.py           # servidor web
├── requirements.txt
├── .env.example         # modelo de variáveis (sem chave real)
├── data/
│   ├── raw/             # documentos originais
│   └── processed/       # gerado pelo pipeline (ignorado no Git)
├── static/              # frontend
└── src/                 # código do RAG
```
