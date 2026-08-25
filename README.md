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
endpoint. It is stored under `rag_artifacts/app-qwen3-embedding-0-6b/`, separately from any
local/Ollama artifact. A FAISS index built with the local Ollama model cannot be queried with a
different embedding model. The bundle requests least-privilege access to the artifact Volume,
history/feedback tables, SQL warehouse, and the two model endpoints.

## Local setup

1. Create and activate a virtual environment.
2. Install with `pip install -e '.[dev]'`.
3. Copy `.env.example` to `.env` and set local values.
4. Apply [`sql/001_rag_schema.sql`](sql/001_rag_schema.sql) with the configured profile:
   `python -m rag.cli setup-db`.
5. Verify live HTML extraction before spending embedding compute:
   `python -m rag.cli discover`.

The local server uses Ollama and a local FAISS directory. It does not use the Databricks App
service principal, OBO identity, or hosted model endpoints.

## Deploy the Databricks App

Use a private `.env` only from the operator machine for the snapshot-build and deployment
commands. Never upload `.env`; it is ignored by Git. Set the profile, catalog, schema,
warehouse, and Volume values in `.env`, then run:

```bash
set -a; source .env; set +a
python -m rag.cli build-app-snapshot
databricks apps deploy --target dev \
  --profile "$RAG_DATABRICKS_PROFILE" \
  --var "warehouse_id=$RAG_WAREHOUSE_ID"
```

The deployment reads [`app.yml`](app.yml), uploads the repository source, installs the Python
dependencies, and starts `rag.app.main:app`. In App mode, Databricks injects resource values
and service-principal credentials; the deployed process does not need `.env`.

For an App that was created separately before bundle deployment, bind it once before deploying:

```bash
databricks bundle deployment bind databricks_docs_rag <app-name> \
  --target dev --profile "$RAG_DATABRICKS_PROFILE" --auto-approve
```

The App URL and deployment state are available with:

```bash
databricks apps get databricks-docs-rag-dev --profile "$RAG_DATABRICKS_PROFILE"
```

Conversation ownership comes from the forwarded Databricks OBO token. The browser cannot set
the owner ID. The App service principal stores shared history and feedback in Delta tables,
while the user token is used only to resolve the signed-in Databricks user.

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
