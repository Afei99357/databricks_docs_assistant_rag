"""execute()'s result-chunk pagination.

The Statement Execution API splits a result set across multiple chunks once
it's large enough (row count or bytes -- big text/vector columns hit this
well before row-count limits). Reading only the first response's
result.data_array silently truncates to that one chunk, with no error:
current_chunks() and embeddings_for() both did this, so a full-corpus read
quietly returned a subset. These tests exercise the real execute() against a
fake statement_execution client instead of bypassing it, since the bug is in
execute() itself.
"""

from rag.storage.databricks import DatabricksStore


class _Name:
    def __init__(self, name):
        self.name = name


class _Status:
    def __init__(self, state_name):
        self.state = _Name(state_name)


class _Column:
    def __init__(self, name):
        self.name = name


class _Manifest:
    def __init__(self, columns):
        self.schema = type("Schema", (), {"columns": [_Column(c) for c in columns]})()


class _ResultData:
    def __init__(self, data_array, next_chunk_index=None):
        self.data_array = data_array
        self.next_chunk_index = next_chunk_index


class _StatementResponse:
    def __init__(self, statement_id, columns, first_chunk):
        self.statement_id = statement_id
        self.status = _Status("SUCCEEDED")
        self.manifest = _Manifest(columns)
        self.result = first_chunk


class FakeStatementExecution:
    """Serves a pre-chunked result set exactly like the real API would."""

    def __init__(self, columns, chunked_rows):
        self.columns = columns
        self.chunked_rows = chunked_rows
        self.get_statement_result_chunk_n_calls: list[int] = []

    def _chunk(self, index):
        next_index = index + 1 if index + 1 < len(self.chunked_rows) else None
        return _ResultData(self.chunked_rows[index], next_index)

    def execute_statement(self, **kwargs):
        return _StatementResponse("stmt-1", self.columns, self._chunk(0))

    def get_statement(self, statement_id):
        return _StatementResponse(statement_id, self.columns, self._chunk(0))

    def get_statement_result_chunk_n(self, statement_id, chunk_index):
        self.get_statement_result_chunk_n_calls.append(chunk_index)
        return self._chunk(chunk_index)


def _store(fake_statement_execution):
    store = DatabricksStore.__new__(DatabricksStore)
    store.warehouse_id = "wh-1"
    store.workspace = type(
        "Workspace", (), {"statement_execution": fake_statement_execution}
    )()
    return store


def test_execute_returns_all_rows_from_a_single_chunk_result():
    fake = FakeStatementExecution(["chunk_id"], [[["a"], ["b"]]])
    store = _store(fake)

    result = store.execute("SELECT chunk_id FROM t")

    assert result.rows == [["a"], ["b"]]
    assert fake.get_statement_result_chunk_n_calls == []


def test_execute_follows_next_chunk_index_to_read_every_row():
    fake = FakeStatementExecution(
        ["chunk_id"], [[["a"], ["b"]], [["c"], ["d"]], [["e"]]]
    )
    store = _store(fake)

    result = store.execute("SELECT chunk_id FROM t")

    assert result.rows == [["a"], ["b"], ["c"], ["d"], ["e"]]
    assert fake.get_statement_result_chunk_n_calls == [1, 2]
