from rag.store import sql_literal


def test_apostrophes_are_backslash_escaped_not_doubled():
    """Databricks SQL drops a doubled '' outright, silently deleting the quote."""
    assert sql_literal("VALUES ('low', 'high')") == r"'VALUES (\'low\', \'high\')'"


def test_backslashes_are_escaped_before_quotes():
    assert sql_literal(r"path\to" + "'x'") == r"'path\\to\'x\''"


def test_plain_values_are_unchanged():
    assert sql_literal("plain text") == "'plain text'"


def test_non_strings_keep_their_literal_form():
    assert sql_literal(None) == "NULL"
    assert sql_literal(True) == "TRUE"
    assert sql_literal(7) == "7"
