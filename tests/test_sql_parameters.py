from rag.storage.databricks import to_statement_parameters


def test_strings_bind_verbatim_without_escaping():
    params = {p.name: (p.type, p.value) for p in to_statement_parameters({"v": "it's \\ ok"})}
    assert params["v"] == ("STRING", "it's \\ ok")


def test_none_binds_as_a_null_typed_parameter():
    params = {p.name: (p.type, p.value) for p in to_statement_parameters({"v": None})}
    assert params["v"] == ("STRING", None)


def test_bool_and_numbers_get_their_own_types():
    params = {p.name: (p.type, p.value) for p in to_statement_parameters(
        {"flag": True, "count": 7, "score": 1.5}
    )}
    assert params["flag"] == ("BOOLEAN", "true")
    assert params["count"] == ("BIGINT", "7")
    assert params["score"] == ("DOUBLE", "1.5")
