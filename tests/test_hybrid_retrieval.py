from rag.index.faiss_store import _lexical_score
from rag.models import Chunk


def test_lexical_signal_rewards_exact_product_phrase():
    matching = Chunk("a", "d", "v", 0, "In Chat mode, Genie works with structured data only.", (), "url", "Genie Agents concepts")
    unrelated = Chunk("b", "d", "v", 0, "Configure a Genie Agent through the API.", (), "url", "Genie API")
    query = "What is the difference between Agent mode and Chat mode?"
    assert _lexical_score(query, matching) > _lexical_score(query, unrelated)
