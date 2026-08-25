"""Inspectable retrieval evaluation; metrics can be written to rag_evaluations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from rag.models import RetrievalResult


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    expected_source_url: str


def load_cases(path: str | Path) -> list[EvaluationCase]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [EvaluationCase(item["question"], item["expected_source_url"]) for item in data]


def recall_at_k(results: list[RetrievalResult], expected_source_url: str) -> float:
    return float(any(result.chunk.source_url == expected_source_url for result in results))


def evaluate(cases: list[EvaluationCase], retrieve) -> dict:
    values = [recall_at_k(retrieve(case.question), case.expected_source_url) for case in cases]
    return {"case_count": len(cases), "recall_at_k": sum(values) / len(values) if values else 0.0}

