PYTHON ?= python
URL ?= http://127.0.0.1:8000

.PHONY: help install back front cli test eval eval-resposta bench db

help:
	@echo Comandos disponíveis:
	@echo   make back    - sobe o app em $(URL)
	@echo   make front   - abre o navegador no app
	@echo   make install - instala dependencias e cria o .env
	@echo   make cli     - roda o pipeline no terminal
	@echo   make test    - roda a suíte de testes
	@echo   make eval    - roda a provinha do retrieval
	@echo   make eval-resposta - avalia as respostas do Gemini (44 chamadas)
	@echo   make bench   - compara chunk/overlap (Recall@3/Recall@5)
	@echo   make db      - sobe o Postgres + pgvector
	@echo Uso: terminal 1 rode make back, terminal 2 rode make front

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) scripts/ensure_env.py

back:
	$(PYTHON) run_web.py

front:
	$(PYTHON) scripts/front.py $(URL)

cli:
	$(PYTHON) main.py

test:
	$(PYTHON) -m pytest tests/ -v

eval:
	$(PYTHON) -m evals.run

eval-resposta:
	$(PYTHON) -m evals.resposta

bench:
	$(PYTHON) -m evals.benchmark

db:
	docker compose up -d
