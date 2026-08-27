"""A fixed battery of documentation questions, and how well retrieval answers it.

Each case names one page the answer must come from, so a run measures whether
retrieval surfaced that page and how high it ranked. Recall says the evidence
was there at all; reciprocal rank says whether it was near the top, which is
the difference an investigating agent is supposed to make over a single search.

The battery lives beside the code rather than under tests/ because it is an
operator tool — ``rag.cli evaluate`` runs it against whatever retrieval the
environment is configured for.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import yaml

from rag.models import RetrievalResult

CASES_PATH = Path(__file__).parent / "evaluation_cases.yaml"


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    expected_source_url: str


@dataclass(frozen=True)
class CaseOutcome:
    case: EvaluationCase
    rank: int | None
    """Where the expected page first appeared, or None if it never did."""
    evidence_count: int
    seconds: float
    error: str | None = None
    """Why the case could not be scored, for a run that failed rather than missed."""


@dataclass(frozen=True)
class Report:
    outcomes: list[CaseOutcome]

    @property
    def hits(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.rank)

    @property
    def recall(self) -> float:
        return self.hits / len(self.outcomes) if self.outcomes else 0.0

    @property
    def mean_reciprocal_rank(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(1 / o.rank for o in self.outcomes if o.rank) / len(self.outcomes)

    @property
    def errors(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.error)

    @property
    def seconds(self) -> float:
        return sum(outcome.seconds for outcome in self.outcomes)


def load_cases(path: str | Path = CASES_PATH) -> list[EvaluationCase]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [EvaluationCase(item["question"], item["expected_source_url"]) for item in data]


def _rank_of(results: list[RetrievalResult], expected_source_url: str) -> int | None:
    return next((position for position, result in enumerate(results, 1)
                 if result.chunk.source_url == expected_source_url), None)


def evaluate(cases: list[EvaluationCase], retrieve, *, on_case=None) -> Report:
    """Run every case, timing each one. ``on_case`` reports progress as it goes."""
    outcomes = []
    for case in cases:
        started = perf_counter()
        try:
            results = retrieve(case.question)
        except Exception as failure:  # noqa: BLE001 - any failure is a datum, not a crash
            outcome = CaseOutcome(case, None, 0, perf_counter() - started, str(failure))
        else:
            outcome = CaseOutcome(case, _rank_of(results, case.expected_source_url),
                                  len(results), perf_counter() - started)
        outcomes.append(outcome)
        if on_case:
            on_case(outcome)
    return Report(outcomes)


_ROW = "{marker} {index:>2}  {rank:>4}  {found:>3}  {seconds:>6}  {question}"


def format_row(index: int, outcome: CaseOutcome) -> str:
    return _ROW.format(marker=" !" if outcome.error else " +" if outcome.rank else " -", index=index,
                       rank=outcome.rank if outcome.rank else "--",
                       found=outcome.evidence_count, seconds=f"{outcome.seconds:.1f}s",
                       question=outcome.case.question[:52])


def format_header() -> str:
    header = _ROW.format(marker="  ", index="#", rank="rank", found="n",
                         seconds="time", question="question")
    return f"{header}\n{'-' * len(header)}"


def format_summary(report: Report) -> str:
    lines = [
        "-" * len(format_header().splitlines()[0]),
        f"  recall  {report.hits}/{len(report.outcomes)}  ({report.recall:.0%})",
        f"  mean reciprocal rank  {report.mean_reciprocal_rank:.3f}",
        f"  total  {report.seconds:.1f}s",
    ]
    if report.errors:
        # Distinct from a miss: retrieval never ran, so recall understates it.
        reasons = sorted({outcome.error for outcome in report.outcomes if outcome.error})
        lines.append(f"  {report.errors} failed  ({'; '.join(reasons)})")
    return "\n".join(lines)


def format_report(report: Report) -> str:
    """The whole run as a table: one row per case, then the summary."""
    rows = [format_header()]
    rows += [format_row(index, outcome) for index, outcome in enumerate(report.outcomes, 1)]
    return "\n".join([*rows, format_summary(report)])
