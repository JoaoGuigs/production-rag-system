"""Apresentação no terminal (prints)."""

from src.config import PASTA_ORIGINAIS, PASTA_PROCESSADOS
from src.models import DocumentoChunkado, DocumentoEmbedado, DocumentoProcessado, RespostaRAG, ResultadoBusca


def _preview(texto: str, limite: int = 250) -> str:
    return texto[:limite] + ("..." if len(texto) > limite else "")


def exibir_pasta_vazia() -> None:
    print(f"Nenhum arquivo em {PASTA_ORIGINAIS}")


def exibir_cabecalho_documentos() -> None:
    print("=== ETAPA 1: extrair + normalizar + salvar ===\n")
    print("Originais em:   ", PASTA_ORIGINAIS)
    print("Processados em: ", PASTA_PROCESSADOS)


def exibir_documento(resultado: DocumentoProcessado) -> None:
    print(f"\n{'=' * 50}")
    print(f"Arquivo: {resultado.nome}")
    print(f"{'=' * 50}")
    print(f"\n[EXTRAÍDO] {len(resultado.texto_bruto)} caracteres")
    print(_preview(resultado.texto_bruto))
    print(f"\n[NORMALIZADO] {len(resultado.texto_limpo)} caracteres")
    print(_preview(resultado.texto_limpo))
    print(f"\n[SALVO EM] {resultado.caminho_salvo}")


def exibir_cabecalho_chunks() -> None:
    print("\n\n=== ETAPA 2: dividir em chunks ===\n")


def exibir_chunks(documento: DocumentoChunkado) -> None:
    print(f"\n{'=' * 50}")
    print(f"Arquivo: {documento.nome}")
    print(f"Total de chunks: {len(documento.chunks)}")
    print(f"{'=' * 50}")

    for chunk in documento.chunks:
        print(f"\n--- Chunk {chunk.indice + 1} ({len(chunk.texto)} caracteres) ---")
        print(_preview(chunk.texto, limite=300))


def exibir_cabecalho_embeddings() -> None:
    print("\n\n=== ETAPA 3: gerar embeddings ===\n")


def exibir_embeddings(documento: DocumentoEmbedado) -> None:
    print(f"\n{'=' * 50}")
    print(f"Arquivo: {documento.nome}")
    print(f"Total de embeddings: {len(documento.chunks_embedados)}")
    print(f"{'=' * 50}")

    for item in documento.chunks_embedados:
        preview_vetor = ", ".join(f"{v:.4f}" for v in item.vetor[:5])
        print(f"\n--- Chunk {item.chunk.indice + 1} ---")
        print(f"Texto: {_preview(item.chunk.texto, limite=120)}")
        print(f"Vetor: {len(item.vetor)} dimensões | início: [{preview_vetor}, ...]")


def exibir_resultados(resultados: list[DocumentoEmbedado]) -> None:
    if not resultados:
        exibir_pasta_vazia()
        return

    exibir_cabecalho_documentos()
    for resultado in resultados:
        exibir_documento(resultado.documento_chunkado.documento)

    exibir_cabecalho_chunks()
    for resultado in resultados:
        exibir_chunks(resultado.documento_chunkado)

    exibir_cabecalho_embeddings()
    for resultado in resultados:
        exibir_embeddings(resultado)


def exibir_busca(pergunta: str, resultados: list[ResultadoBusca]) -> None:
    print("\n\n=== ETAPA 4: buscar chunks para a pergunta ===\n")
    print(f"Pergunta: {pergunta}\n")

    if not resultados:
        print("Nenhum chunk encontrado.")
        return

    for i, resultado in enumerate(resultados, start=1):
        chunk = resultado.chunk_embedado.chunk
        print(f"[{i}] similaridade={resultado.similaridade:.4f} | fonte={chunk.fonte}")
        print(_preview(chunk.texto, limite=300))
        print()


def exibir_resposta_rag(resultado: RespostaRAG) -> None:
    exibir_busca(resultado.pergunta, resultado.chunks)
    print("\n=== ETAPA 5: resposta do LLM ===\n")
    print(resultado.resposta)
