"""Pontuação determinística, agregação de métricas e gates da avaliação."""

import re
import unicodedata
from statistics import mean, median

FRASES_RECUSA = (
    "não encontrei", "nao encontrei", "não há informação", "nao ha informacao",
    "não consta", "nao consta", "não sei", "nao sei", "não tenho informação",
    "nao tenho informacao", "não é possível responder", "nao e possivel responder",
    "não posso responder", "nao posso responder", "sem informação", "sem informacao",
    "não menciona", "nao menciona", "não há menção", "nao ha mencao",
    "não oferece", "nao oferece", "não paga", "nao paga",
    "não estabelece", "nao estabelece", "não define", "nao define",
)


def normalizar(texto: str) -> str:
    texto = "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", texto.lower()).strip()


def contem_ancora(resposta: str, ancoras: list[str]) -> bool:
    texto = normalizar(resposta)
    return any(normalizar(a) in texto for a in ancoras)


def eh_recusa(resposta: str) -> bool:
    texto = normalizar(resposta)
    return any(normalizar(frase) in texto for frase in FRASES_RECUSA)


def afirma_sem_recusa(resposta: str, afirmacoes: list[str]) -> bool:
    frases = [f.strip() for f in re.split(r"[.!?…\n]+", resposta) if f.strip()]
    for frase in frases:
        texto = normalizar(frase)
        if eh_recusa(frase):
            continue
        for afirmacao in afirmacoes:
            alvo = normalizar(afirmacao)
            inicio = texto.find(alvo)
            if inicio < 0:
                continue
            prefixo = texto[max(0, inicio - 45):inicio]
            negada = re.search(r"\b(?:nao|nunca|jamais)\b", prefixo) is not None
            if not negada:
                return True
    return False


def extrair_citacoes(resposta: str) -> list[str]:
    citacoes = []
    for bloco in re.findall(r"\[([^\]\n]+)\]", resposta):
        for parte in re.split(r"[,;]", bloco):
            fonte = parte.strip()
            if re.fullmatch(r"[^\]\n]+\.(?:md|txt|pdf)", fonte, re.IGNORECASE):
                citacoes.append(fonte)
    return citacoes


def tem_citacao(resposta: str, fontes: list[str]) -> bool:
    citadas = {normalizar(c) for c in extrair_citacoes(resposta)}
    return bool(citadas.intersection(normalizar(f) for f in fontes))


def cobertura_fatos(resposta: str, fatos: list[dict]) -> tuple[float, list[str]]:
    if not fatos:
        return 1.0, []
    ausentes = [f["label"] for f in fatos if not contem_ancora(resposta, f["any_of"])]
    return (len(fatos) - len(ausentes)) / len(fatos), ausentes


def _retrieval(item: dict, chunks: list[dict]) -> dict:
    if not item["answerable"]:
        return {"retrieval_hit": None, "reciprocal_rank": None}
    fontes = {normalizar(f) for f in item["expected_sources"]}
    evidencias = [normalizar(e) for e in item.get("expected_evidence", [])]
    ranks = [c["rank"] for c in chunks if normalizar(c["fonte"]) in fontes and
             (not evidencias or any(e in normalizar(c["texto"]) for e in evidencias))]
    return {"retrieval_hit": float(bool(ranks)), "reciprocal_rank": 1 / min(ranks) if ranks else 0.0}


def pontuar_resposta(item: dict, resposta: str, chunks: list[dict]) -> dict:
    cobertura, ausentes = cobertura_fatos(resposta, item.get("required_facts", []))
    proibidas = item.get("forbidden_claims", [])
    alucinou = afirma_sem_recusa(resposta, proibidas) if proibidas else False
    recusa = eh_recusa(resposta)
    citacoes = extrair_citacoes(resposta)
    fontes_contexto = {normalizar(c["fonte"]) for c in chunks}
    precisao = (sum(normalizar(c) in fontes_contexto for c in citacoes) / len(citacoes)) if citacoes else 0.0
    if item["answerable"]:
        correto, recusa_correta = cobertura == 1 and not recusa and not alucinou, None
        citacao_esperada = float(tem_citacao(resposta, item["expected_sources"]))
    else:
        correto, recusa_correta, citacao_esperada = recusa and not alucinou, float(recusa and not alucinou), None
    return {
        **_retrieval(item, chunks),
        "metrics": {"answer_correct": float(correto),
                    "fact_completeness": cobertura if item["answerable"] else None,
                    "expected_citation": citacao_esperada,
                    "citation_precision": precisao if item["answerable"] else None,
                    "correct_refusal": recusa_correta,
                    "hallucination_proxy": float(alucinou)},
        "details": {"missing_facts": ausentes, "citations": citacoes,
                    "refusal_detected": recusa, "forbidden_claim_detected": alucinou},
    }


def _media(registros: list[dict], *caminho: str) -> float | None:
    valores = []
    for registro in registros:
        valor = registro
        for chave in caminho:
            valor = valor.get(chave) if isinstance(valor, dict) else None
        if valor is not None:
            valores.append(float(valor))
    return mean(valores) if valores else None


def _percentil(valores: list[float], p: float) -> float:
    ordenados = sorted(valores)
    return ordenados[round((len(ordenados) - 1) * p)] if ordenados else 0.0


def agregar(registros: list[dict], usar_judge: bool = False) -> dict:
    sucesso = [r for r in registros if r["status"] == "ok"]
    resp = [r for r in sucesso if r["answerable"]]
    neg = [r for r in sucesso if not r["answerable"]]
    lat = [r["latency_ms"]["total"] for r in sucesso]
    resumo = {
        "total": len(registros), "successful": len(sucesso), "errors": len(registros) - len(sucesso),
        "error_rate": (len(registros) - len(sucesso)) / len(registros) if registros else 1.0,
        "retrieval_recall_at_k": _media(resp, "retrieval_hit"),
        "retrieval_mrr": _media(resp, "reciprocal_rank"),
        "answer_accuracy": _media(resp, "metrics", "answer_correct"),
        "fact_completeness": _media(resp, "metrics", "fact_completeness"),
        "expected_citation_rate": _media(resp, "metrics", "expected_citation"),
        "citation_precision": _media(resp, "metrics", "citation_precision"),
        "correct_refusal_rate": _media(neg, "metrics", "correct_refusal"),
        "hallucination_rate": _media(sucesso, "metrics", "hallucination_proxy"),
        "latency_ms": {"average": round(mean(lat), 2) if lat else 0.0,
                       "median": round(median(lat), 2) if lat else 0.0,
                       "p95": round(_percentil(lat, .95), 2)},
    }
    if usar_judge:
        resumo["judge"] = {m: _media(sucesso, "judge", m) for m in
                            ("correctness", "groundedness", "citation_correctness")}
    return resumo


def avaliar_gates(resumo: dict, thresholds: dict, usar_judge: bool = False) -> list[str]:
    falhas = []
    for metrica, limite in thresholds.get("minimum", {}).items():
        valor = resumo.get(metrica)
        if valor is None or valor < limite:
            falhas.append(f"{metrica}={valor} abaixo de {limite}")
    for metrica, limite in thresholds.get("maximum", {}).items():
        valor = resumo.get(metrica)
        if valor is None or valor > limite:
            falhas.append(f"{metrica}={valor} acima de {limite}")
    if usar_judge:
        for metrica, limite in thresholds.get("judge_minimum", {}).items():
            valor = resumo.get("judge", {}).get(metrica)
            if valor is None or valor < limite:
                falhas.append(f"judge.{metrica}={valor} abaixo de {limite}")
    return falhas
