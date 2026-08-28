from rag.storage.databricks import DatabricksStore
from rag.storage.protocol import ConversationStore, CorpusStore, DiagnosticsStore


def test_the_databricks_adapter_satisfies_every_row_protocol():
    assert isinstance(DatabricksStore.__new__(DatabricksStore), CorpusStore)
    assert isinstance(DatabricksStore.__new__(DatabricksStore), ConversationStore)
    assert isinstance(DatabricksStore.__new__(DatabricksStore), DiagnosticsStore)
