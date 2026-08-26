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

The local server uses Ollama for answer generation and embeddings, and a local FAISS directory
for retrieval. It still uses the configured Databricks SQL warehouse for the governed document,
history, and feedback tables.

1. Install the locked project dependencies and create a private environment file:

   ```bash
   uv sync --extra dev
   cp .env.example .env
   ```

2. Edit `.env`. At minimum, configure your Databricks profile/catalog/schema/warehouse and set:

   ```text
   RAG_LOCAL_INDEX_DIR=/absolute/path/to/local-faiss-index
   RAG_LOCAL_TEST_USER_ID=your-email@example.com
   ANSWER_PROVIDER=ollama
   RAG_EMBEDDING_MODEL=qwen3-embedding:4b
   OLLAMA_MODEL=qwen3.5:latest
   ```

   Load those settings into the current terminal before running any `rag.cli` command:

   ```bash
   set -a
   source .env
   set +a
   ```

3. Start Ollama and make sure both local models are available:

   ```bash
   ollama pull qwen3-embedding:4b
   ollama pull qwen3.5:latest
   ```

4. Create the governed tables/Volume, fetch and chunk the configured sources, and build the
   local Ollama-backed FAISS snapshot:

   ```bash
   uv run python -m rag.cli setup-db
   uv run python -m rag.cli build-local-snapshot
   ```

5. Start the local application and open `http://127.0.0.1:8000`:

   ```bash
   uv run python -m rag.cli serve
   ```

Run `uv run python -m rag.cli build-local-snapshot` again whenever you want to refresh the
configured documentation sources and rebuild the local index. In a new terminal, load `.env`
again with the three `set -a`/`source`/`set +a` commands above before using the CLI.

### Add a crawlable documentation site

For a site with many related pages, add one bounded policy to
[`rag/ingest/config/crawl_sources.yaml`](rag/ingest/config/crawl_sources.yaml). Set the
landing page, allowed section prefixes, excluded prefixes, and a maximum link depth. The
normal discovery/refresh workflow follows same-site links only within that policy, so new
pages are picked up automatically the next time discovery or refresh runs. Pages outside the
allow-list, static assets, and explicitly excluded sections are not indexed. This keeps source
selection reviewable in Git while avoiding a manually copied URL list.

The local server uses Ollama and a local FAISS directory. It does not use the Databricks App
service principal, OBO identity, or hosted model endpoints.

## Deploy to a new Databricks workspace

Deployment is deliberately two-phase: the **refresh Workflow** creates the governed data and
FAISS snapshot; the **Databricks App** only reads that snapshot and serves users. Do not deploy
the App before the first Workflow run succeeds.

### 1. Prerequisites

You need a Unity Catalog-enabled workspace, a Pro or Serverless SQL warehouse, Databricks Apps,
and serverless Jobs. The deploying identity needs permission to create the target catalog/schema,
use the warehouse, query the two model endpoints, and create Apps and Jobs. The Workflow also
needs outbound access to the configured documentation sites.

Authenticate a CLI profile for the target workspace:

```bash
databricks auth login --host https://<workspace-host> --profile <profile>
```

Choose values once and use the same values for every command below:

```bash
export RAG_PROFILE=<profile>
export RAG_CATALOG=<catalog>
export RAG_SCHEMA=<schema>
export RAG_WAREHOUSE_ID=<warehouse-id>
export RAG_VOLUME=rag_artifacts
export RAG_EMBEDDING_ENDPOINT=databricks-qwen3-embedding-0-6b
export RAG_CHAT_ENDPOINT=databricks-gpt-oss-20b
```

Confirm both endpoints exist before continuing:

```bash
databricks serving-endpoints get "$RAG_EMBEDDING_ENDPOINT" --profile "$RAG_PROFILE"
databricks serving-endpoints get "$RAG_CHAT_ENDPOINT" --profile "$RAG_PROFILE"
```

If the catalog/schema differs from `eliao.genie_kb`, update the three corresponding values in
[`app.yml`](app.yml) before creating the App: `RAG_CATALOG`, `RAG_SCHEMA`, and
`RAG_ARTIFACT_VOLUME`.

### 2. Upload the project source

From the repository root, validate and synchronize the source files to the target workspace:

```bash
(
  cd bootstrap
  databricks bundle validate --target dev --profile "$RAG_PROFILE" \
  --var "catalog=$RAG_CATALOG" --var "schema=$RAG_SCHEMA" \
  --var "artifact_volume=$RAG_VOLUME" --var "warehouse_id=$RAG_WAREHOUSE_ID" \
  --var "embedding_endpoint=$RAG_EMBEDDING_ENDPOINT"
  databricks bundle deploy --target dev --profile "$RAG_PROFILE" \
    --var "catalog=$RAG_CATALOG" --var "schema=$RAG_SCHEMA" \
    --var "artifact_volume=$RAG_VOLUME" --var "warehouse_id=$RAG_WAREHOUSE_ID" \
    --var "embedding_endpoint=$RAG_EMBEDDING_ENDPOINT"
)
```

This deploys the Job-only bootstrap Bundle and uploads all needed project files. It creates the
Workflow `databricks-docs-rag-refresh-dev`.

### 3. Create and run the bootstrap Workflow

After deployment, run the Bundle-managed Workflow from **Workflows**, or run:

```bash
(cd bootstrap && databricks bundle run refresh_databricks_docs_index --target dev \
  --profile "$RAG_PROFILE" --var "catalog=$RAG_CATALOG" \
  --var "schema=$RAG_SCHEMA" --var "artifact_volume=$RAG_VOLUME" \
  --var "warehouse_id=$RAG_WAREHOUSE_ID" \
  --var "embedding_endpoint=$RAG_EMBEDDING_ENDPOINT")
```

Wait for success. This idempotently creates the catalog, schema, all Delta
tables, and Volume; refreshes the configured sources; and writes the active App snapshot under
`rag_artifacts/app-qwen3-embedding-0-6b/`. Inspect the task output before proceeding.

For a brand-new workspace, the Job must run first: the App's resource bindings cannot be created
until its tables and Volume exist.

### 4. Create and deploy the App

Deploy the App-only Bundle after the bootstrap run succeeds:

```bash
databricks bundle deploy --target dev --profile "$RAG_PROFILE" \
  --var "catalog=$RAG_CATALOG" --var "schema=$RAG_SCHEMA" \
  --var "artifact_volume=$RAG_VOLUME" --var "warehouse_id=$RAG_WAREHOUSE_ID" \
  --var "embedding_endpoint=$RAG_EMBEDDING_ENDPOINT" \
  --var "reasoning_endpoint=$RAG_CHAT_ENDPOINT"
```

The Bundle creates the App and configures these service-principal resources:

| Resource | Permission |
| --- | --- |
| SQL warehouse | `CAN_USE` |
| `rag_artifacts` Volume | `READ_VOLUME` |
| `rag_feedback`, `rag_conversations`, `rag_conversation_turns` | `SELECT` and `MODIFY` |
| `databricks-qwen3-embedding-0-6b` | `CAN_QUERY` |
| `databricks-gpt-oss-20b` | `CAN_QUERY` |

Deploy the App. It starts `rag.app.main:app`, reads the active Volume snapshot, uses the App
service principal for shared storage/model calls, and uses OBO only to establish the signed-in
user's conversation owner. The deployed App does not read your local `.env` file.

### 5. Refresh or redeploy later

To update documentation, run the bootstrap Bundle's Workflow again from **Workflows** (or use
the `databricks bundle run` command above). It creates a new
validated snapshot and atomically marks it active; the previous active snapshot remains usable if
the refresh fails.

To deploy code/configuration changes, redeploy the App Bundle command above. If the ingestion
code, SQL schema, source configuration, or embedding endpoint changed, redeploy the bootstrap
Bundle, run its Workflow, then redeploy the App Bundle. Do not rebuild a local
Ollama snapshot for the Databricks App; the App snapshot must use the configured Databricks
embedding endpoint.

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
