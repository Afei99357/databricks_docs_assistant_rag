from rag.conversation import resolve_follow_up


class Provider:
    name = "fake"
    model = "fake"

    def complete(self, prompt):
        return '{"standalone_query":"What Volume requirements apply to Genie Agents?"}'


def test_follow_up_is_rewritten_from_bounded_prior_turns():
    turns = [("id", 1, "How do agents use Volumes?", "They analyze attached files.")]
    assert resolve_follow_up("What are the requirements?", turns, Provider()) == "What Volume requirements apply to Genie Agents?"
