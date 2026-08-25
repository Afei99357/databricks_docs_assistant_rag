from pathlib import Path

from rag.evaluate import evaluate, load_cases
from rag.models import Chunk, RetrievalResult


def test_evaluation_set_has_25_representative_questions():
    cases = load_cases(Path(__file__).parent / "evaluation_cases.yaml")
    assert len(cases) == 25
    expected = cases[0].expected_source_url
    chunk = Chunk("c", "d", "v", 0, "text", (), expected, "Title")
    assert evaluate(cases[:1], lambda _: [RetrievalResult(chunk, .9, "s")])["recall_at_k"] == 1.0

