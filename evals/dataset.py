"""Leitura e validação do dataset de respostas v2."""

import json
from pathlib import Path

DATASET_PADRAO = Path(__file__).resolve().parent / "answer_dataset.json"


def carregar_dataset(caminho: Path = DATASET_PADRAO) -> dict:
    with caminho.open(encoding="utf-8") as arquivo:
        dataset = json.load(arquivo)
    if dataset.get("_meta", {}).get("version") != 2 or not dataset.get("items"):
        raise ValueError("answer dataset inválido: esperado version=2 e items não vazio")
    ids = [item.get("id") for item in dataset["items"]]
    if len(ids) != len(set(ids)):
        raise ValueError("answer dataset possui ids duplicados")
    obrigatorios = {"id", "question", "answerable", "expected_answer", "expected_sources",
                    "expected_evidence", "required_facts", "forbidden_claims"}
    for item in dataset["items"]:
        faltando = obrigatorios.difference(item)
        if faltando:
            raise ValueError(f"caso {item.get('id', '?')} sem campos: {sorted(faltando)}")
        if item["answerable"] and (not item["expected_answer"] or not item["required_facts"]):
            raise ValueError(f"caso respondível {item['id']} precisa de resposta e fatos")
        for fato in item["required_facts"]:
            if not fato.get("label") or not fato.get("any_of"):
                raise ValueError(f"fato inválido no caso {item['id']}")
    return dataset
