from rag.conversation import resolve_follow_up
from rag.llm.providers import ToolCall

TURNS = [("id", 1, "How do agents use Volumes?", "They analyze attached files.")]


class Provider:
    name = model = "fake"

    def __init__(self, query="What Volume requirements apply to Genie Agents?"):
        self.query, self.captured = query, {}

    def call_tool(self, messages, tools):
        self.captured = {"messages": messages, "tools": tools}
        return ToolCall("standalone_query", {"query": self.query})


def test_follow_up_is_rewritten_from_bounded_prior_turns():
    assert resolve_follow_up("What are the requirements?", TURNS, Provider()) == \
        "What Volume requirements apply to Genie Agents?"


def test_the_rewrite_arrives_as_a_tool_call_rather_than_json_in_prose():
    # The model's own decoder guarantees the shape, so nothing here parses text
    # or needs a fallback for text that failed to parse.
    provider = Provider()
    resolve_follow_up("What are the requirements?", TURNS, provider)
    assert provider.captured["tools"][0]["function"]["name"] == "standalone_query"


def test_a_question_with_no_history_is_already_standalone():
    provider = Provider()
    assert resolve_follow_up("What are the requirements?", [], provider) == "What are the requirements?"
    assert provider.captured == {}


def test_only_the_most_recent_turns_are_sent():
    turns = [("id", n, f"question {n}", f"answer {n}") for n in range(1, 6)]
    provider = Provider()
    resolve_follow_up("And then?", turns, provider, limit=2)
    context = provider.captured["messages"][-1]["content"]
    assert "question 5" in context and "question 4" in context
    assert "question 3" not in context


def test_a_blank_rewrite_leaves_the_question_alone():
    assert resolve_follow_up("What are the requirements?", TURNS, Provider("  ")) == \
        "What are the requirements?"
