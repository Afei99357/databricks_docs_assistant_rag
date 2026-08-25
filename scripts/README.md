# Manual operations

The first release intentionally has no scheduler. Run each expensive action explicitly.

```bash
cp .env.example .env  # then populate local values; never commit it
set -a; source .env; set +a
python -m rag.cli setup-db
python -m rag.cli discover
RAG_LOCAL_INDEX_DIR=./data python -m rag.cli serve
```

After a local Qwen model is installed, the operational sequence is: fetch/extract a small
source set, inspect chunks, build a 0.6B FAISS snapshot, run `tests/evaluation_cases.yaml`,
then repeat with 4B before selecting the active model. Snapshot activation is a separate,
atomic action: a failed build must leave the previous `active_snapshot.json` untouched.

## Databricks App preparation

The App's FAISS snapshot must be built with `databricks-qwen3-embedding-0-6b`, separately
from the local Ollama snapshot. The vector dimensions and embedding space must match at index
and query time. Publish that snapshot to the `rag_artifacts` Volume before deploying the App;
its runtime reads the active manifest from the Volume and uses `databricks-gpt-oss-20b` for
retrieval-agent reasoning and answers.

Validate the declarative bundle before deployment:

```bash
databricks bundle validate --target dev --var warehouse_id=<sql-warehouse-id>
```

The resulting App has one service principal for shared reads/writes. User OBO is used only to
look up the authenticated caller and enforce per-user conversation-history ownership.
