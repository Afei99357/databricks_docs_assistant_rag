from rag.evaluate import CASES_PATH, evaluate, format_report, load_cases
from rag.models import Chunk, RetrievalResult


def _result(url, score=0.9):
    return RetrievalResult(Chunk("c", "d", "v", 0, "text", (), url, "Title"), score, "snap")


def test_the_battery_holds_25_questions_with_the_page_each_should_find():
    cases = load_cases(CASES_PATH)
    assert len(cases) == 25
    assert all(case.question and case.expected_source_url.startswith("http") for case in cases)


def test_a_case_records_where_the_expected_page_ranked():
    cases = load_cases(CASES_PATH)[:1]
    expected = cases[0].expected_source_url
    report = evaluate(cases, lambda _: [_result("https://other"), _result(expected)])
    assert report.outcomes[0].rank == 2


def test_a_case_that_never_retrieves_the_expected_page_has_no_rank():
    cases = load_cases(CASES_PATH)[:1]
    report = evaluate(cases, lambda _: [_result("https://other")])
    assert report.outcomes[0].rank is None
    assert report.recall == 0.0


def test_recall_counts_cases_that_found_the_page_at_any_rank():
    cases = load_cases(CASES_PATH)[:2]
    expected = cases[0].expected_source_url
    report = evaluate(cases, lambda case: [_result(expected)] if case == cases[0].question else [])
    assert report.recall == 0.5


def test_reciprocal_rank_rewards_ranking_the_expected_page_first():
    # Recall alone cannot distinguish "answered it" from "buried it at rank 8",
    # which is the difference the agent is supposed to make.
    cases = load_cases(CASES_PATH)[:2]
    first, second = cases[0].expected_source_url, cases[1].expected_source_url
    report = evaluate(cases, lambda case: [_result(first)] if case == cases[0].question
                      else [_result("https://other"), _result("https://other"), _result(second)])
    assert report.mean_reciprocal_rank == (1.0 + 1 / 3) / 2


def test_an_empty_battery_reports_zeroes_rather_than_dividing_by_zero():
    report = evaluate([], lambda _: [])
    assert (report.recall, report.mean_reciprocal_rank) == (0.0, 0.0)


def test_the_report_tabulates_every_case_and_the_summary():
    cases = load_cases(CASES_PATH)[:1]
    text = format_report(evaluate(cases, lambda _: [_result(cases[0].expected_source_url)]))
    assert "rank" in text and "recall" in text
    assert cases[0].question[:20] in text
    assert "1/1" in text


def test_a_case_that_raises_is_recorded_and_the_battery_continues():
    # A run that aborts on the first bad question measures nothing. Recording
    # the failure is the measurement, not a fallback around it.
    cases = load_cases(CASES_PATH)[:2]
    def retrieve(question):
        if question == cases[0].question:
            raise RuntimeError("returned prose instead of a tool call")
        return [_result(cases[1].expected_source_url)]

    report = evaluate(cases, retrieve)

    assert report.outcomes[0].error == "returned prose instead of a tool call"
    assert report.outcomes[0].rank is None
    assert report.outcomes[1].rank == 1
    assert report.errors == 1


def test_the_summary_reports_failures_separately_from_misses():
    cases = load_cases(CASES_PATH)[:1]
    def raises(_):
        raise RuntimeError("boom")
    text = format_report(evaluate(cases, raises))
    assert "1 failed" in text
    assert "boom" in text
