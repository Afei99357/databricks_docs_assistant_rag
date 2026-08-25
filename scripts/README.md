# Manual operations

The first release intentionally has no scheduler. Run each expensive action explicitly.

```bash
cp .env.example .env  # then populate local values; never commit it
set -a; source .env; set +a
python -m rag.cli setup-db
python -m rag.cli discover
RAG_LOCAL_INDEX_DIR=./data python -m rag.cli build-local-snapshot
RAG_LOCAL_INDEX_DIR=./data python -m rag.cli serve
```

`build-local-snapshot` refreshes the configured sources (including the bounded Blueprint
crawler), updates the governed document/chunk tables, embeds every refreshed chunk with the
local Ollama embedding model, and atomically activates a new snapshot under
`RAG_LOCAL_INDEX_DIR`. It does not modify the Databricks App snapshot.

After a local Qwen model is installed, the operational sequence is: fetch/extract a small
source set, inspect chunks, build a 0.6B FAISS snapshot, run `tests/evaluation_cases.yaml`,
then repeat with 4B before selecting the active model. Snapshot activation is a separate,
atomic action: a failed build must leave the previous `active_snapshot.json` untouched.

## Databricks App preparation

The App's FAISS snapshot must be built with `databricks-qwen3-embedding-0-6b`, separately
from the local Ollama snapshot. The vector dimensions and embedding space must match at index
and query time. Publish that snapshot under
`rag_artifacts/app-qwen3-embedding-0-6b/` before deploying the App; its runtime reads the
active manifest only from that App-specific directory and uses `databricks-gpt-oss-20b` for
retrieval-agent reasoning and answers. Existing local/Ollama artifacts are not touched.

Validate the declarative bundle before deployment:

```bash
databricks bundle validate --target dev --var warehouse_id=<sql-warehouse-id>
```

With the workspace profile and standard RAG environment values loaded, build and activate the
App-compatible snapshot with:

```bash
python -m rag.cli build-app-snapshot
```

The command reads `rag_chunks`, creates a fresh immutable
`app-qwen3-embedding-0-6b/snapshots/<id>/index.faiss` and `chunk_map.json` in the artifact
Volume, and uploads that directory's `active_snapshot.json` last. It does not delete or modify
the existing local/Ollama snapshots. It makes embedding calls and writes new files to the
existing artifact Volume.

The resulting App has one service principal for shared reads/writes. User OBO is used only to
look up the authenticated caller and enforce per-user conversation-history ownership.
