# Custom Databricks Documentation RAG

An internal, inspectable RAG application over official Databricks documentation. It uses
local Qwen embeddings and FAISS for retrieval; Unity Catalog stores governed metadata,
feedback, evaluation results, and immutable index artifacts. It does not use Genie Agent,
Volume Content Search, AI Search, or Vector Search.

## Current status

The local prototype includes official-source discovery/fetch/extraction, deterministic
chunking, local Qwen embedding adapters, FAISS snapshot validation/activation, retrieval
evaluation, a grounded Flask UI, and a Databricks feedback sink. No real workspace values or
credentials belong in this repository.

## Databricks App mode

The same Flask application can run as a Databricks App. It uses the App service principal
for the shared FAISS snapshot in the artifact Volume, Delta conversation history/feedback,
and model inference. It uses the forwarded user token only to derive a trusted history owner;
no browser-supplied user ID is accepted.

- embeddings: `databricks-qwen3-embedding-0-6b`
- retrieval reasoning and answer generation: `databricks-gpt-oss-20b`
- local `rag serve` remains Ollama-based and is unaffected.

Before deploying, build and publish a separate snapshot with the Databricks Qwen embedding
endpoint. A FAISS index built with the local Ollama model cannot be queried with a different
embedding model. Set the bundle's `warehouse_id` variable, then run
`databricks bundle validate` and deploy from the `databricks-app` branch. The bundle requests
least-privilege access to the artifact Volume, history/feedback tables, SQL warehouse, and the
two model endpoints.

## Local setup

1. Create and activate a virtual environment.
2. Install with `pip install -e '.[dev]'`.
3. Copy `.env.example` to `.env` and set local values.
4. Apply [`sql/001_rag_schema.sql`](sql/001_rag_schema.sql) with the configured profile:
   `python -m rag.cli setup-db`.
5. Verify live HTML extraction before spending embedding compute:
   `python -m rag.cli discover`.

## Operational boundaries

- Build Qwen 0.6B and 4B snapshots separately and select a model from the 25-case evaluation
  set; model size is not a quality decision.
- A snapshot is built in staging, validated against its chunk map, then activated atomically.
  Failed builds leave the current active snapshot unchanged.
- The UI refuses answers below the relevance threshold or without model citations. It stores
  feedback/retrieval diagnostics, not generated answer text.
- Costs remain visible: local GPU embedding is explicit, and Delta/Volume, SQL Warehouse, and
  optional Databricks endpoint usage are separately attributable.

See [`scripts/README.md`](scripts/README.md) for manual operating commands and
[`tests/evaluation_cases.yaml`](tests/evaluation_cases.yaml) for the initial benchmark.
