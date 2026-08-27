from unittest.mock import patch

import requests

from rag.ingest.fetch import fetch_page


class _Response:
    def __init__(self, status_code=200, text="<html></html>", url="https://example.test/page"):
        self.status_code, self.text, self.url = status_code, text, url
        self.headers = {"content-type": "text/html"}
        self.encoding = None


def test_transient_connection_reset_is_retried_until_it_succeeds():
    calls = [requests.ConnectionError("Connection reset by peer"), _Response()]
    with patch("rag.ingest.fetch._SESSION.get", side_effect=calls) as get:
        result = fetch_page("d", "https://example.test/page", backoff=0)
    assert result.outcome == "ok"
    assert get.call_count == 2


def test_retryable_status_is_retried():
    calls = [_Response(status_code=503), _Response()]
    with patch("rag.ingest.fetch._SESSION.get", side_effect=calls) as get:
        result = fetch_page("d", "https://example.test/page", backoff=0)
    assert result.outcome == "ok"
    assert get.call_count == 2


def test_not_found_is_not_retried():
    with patch("rag.ingest.fetch._SESSION.get", return_value=_Response(status_code=404)) as get:
        result = fetch_page("d", "https://example.test/page", backoff=0)
    assert result.outcome == "not_found"
    assert get.call_count == 1


def test_exhausted_retries_report_the_last_network_error():
    with patch("rag.ingest.fetch._SESSION.get", side_effect=requests.ConnectionError("reset")) as g:
        result = fetch_page("d", "https://example.test/page", attempts=3, backoff=0)
    assert result.outcome == "network_error"
    assert "reset" in result.error_message
    assert g.call_count == 3
