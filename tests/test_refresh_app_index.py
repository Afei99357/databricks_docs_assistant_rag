from rag.jobs.refresh_app_index import _parse_bool, parse_args


def test_parse_bool_accepts_the_strings_a_job_parameter_substitutes():
    assert _parse_bool("true") is True
    assert _parse_bool("True") is True
    assert _parse_bool("1") is True
    assert _parse_bool("yes") is True
    assert _parse_bool("false") is False
    assert _parse_bool("0") is False
    assert _parse_bool("") is False


def _required_args():
    return [
        "--catalog", "c", "--schema", "s", "--warehouse-id", "w",
        "--schema-sql-path", "/p",
    ]


def test_repair_chunks_defaults_to_false_when_the_flag_is_omitted():
    args = parse_args(_required_args())
    assert args.repair_chunks is False


def test_repair_chunks_true_string_parses_to_a_real_bool():
    args = parse_args([*_required_args(), "--repair-chunks", "true"])
    assert args.repair_chunks is True


def test_repair_chunks_false_string_parses_to_a_real_bool():
    args = parse_args([*_required_args(), "--repair-chunks", "false"])
    assert args.repair_chunks is False
