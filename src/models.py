from dataclasses import dataclass
from pathlib import Path


@dataclass
class DocumentoProcessado:
    arquivo_original: Path
    texto_bruto: str
    texto_limpo: str
    caminho_salvo: Path

    @property
    def nome(self) -> str:
        return self.arquivo_original.name


@dataclass
class Chunk:
    indice: int
    texto: str
    fonte: str


@dataclass
class DocumentoChunkado:
    documento: DocumentoProcessado
    chunks: list[Chunk]

    @property
    def nome(self) -> str:
        return self.documento.nome


@dataclass
class ChunkEmbedado:
    chunk: Chunk
    vetor: list[float]


@dataclass
class DocumentoEmbedado:
    documento_chunkado: DocumentoChunkado
    chunks_embedados: list[ChunkEmbedado]

    @property
    def nome(self) -> str:
        return self.documento_chunkado.nome


@dataclass
class ResultadoBusca:
    pergunta: str
    chunk_embedado: ChunkEmbedado
    similaridade: float


@dataclass
class RespostaRAG:
    pergunta: str
    resposta: str
    chunks: list[ResultadoBusca]
