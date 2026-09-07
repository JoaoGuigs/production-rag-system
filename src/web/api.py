"""API HTTP do RAG."""

import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import BASE_DIR, PASTA_ORIGINAIS
from src.pipeline import EXTENSOES_SUPORTADAS
from src.web.store import indexar, obter_status, responder

STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="RAG Lab")


class PerguntaRequest(BaseModel):
    pergunta: str


@app.get("/")
def pagina_inicial():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status():
    return obter_status()


@app.post("/api/upload")
async def upload(arquivo: UploadFile = File(...)):
    extensao = Path(arquivo.filename or "").suffix.lower()
    if extensao not in EXTENSOES_SUPORTADAS:
        raise HTTPException(
            status_code=400,
            detail=f"Formato não suportado. Use: {', '.join(EXTENSOES_SUPORTADAS)}",
        )

    PASTA_ORIGINAIS.mkdir(parents=True, exist_ok=True)
    destino = PASTA_ORIGINAIS / Path(arquivo.filename).name

    with destino.open("wb") as buffer:
        shutil.copyfileobj(arquivo.file, buffer)

    return {"ok": True, "arquivo": destino.name}


@app.post("/api/indexar")
def indexar_documentos():
    try:
        return indexar()
    except FileNotFoundError as erro:
        raise HTTPException(status_code=400, detail=str(erro)) from erro


@app.post("/api/perguntar")
def fazer_pergunta(body: PerguntaRequest):
    pergunta = body.pergunta.strip()
    if not pergunta:
        raise HTTPException(status_code=400, detail="Digite uma pergunta.")

    try:
        resultado = responder(pergunta)
    except ValueError as erro:
        raise HTTPException(status_code=400, detail=str(erro)) from erro
    except Exception as erro:
        raise HTTPException(status_code=500, detail=f"Erro interno: {erro}") from erro

    return {
        "pergunta": resultado.pergunta,
        "resposta": resultado.resposta,
        "resultados": [
            {
                "similaridade": round(item.similaridade, 4),
                "fonte": item.chunk_embedado.chunk.fonte,
                "indice": item.chunk_embedado.chunk.indice,
                "texto": item.chunk_embedado.chunk.texto,
            }
            for item in resultado.chunks
        ],
    }


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
