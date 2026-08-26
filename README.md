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
databricks bundle validate --target dev --profile "$RAG_PROFILE" \
  --var "catalog=$RAG_CATALOG" --var "schema=$RAG_SCHEMA" \
  --var "artifact_volume=$RAG_VOLUME" --var "warehouse_id=$RAG_WAREHOUSE_ID" \
  --var "embedding_endpoint=$RAG_EMBEDDING_ENDPOINT" \
  --var "reasoning_endpoint=$RAG_CHAT_ENDPOINT"

databricks bundle sync --target dev --profile "$RAG_PROFILE"
```

The command synchronizes to
`/Workspace/Users/<your-user>/.bundle/databricks-docs-rag/dev/files`; use that exact path (shown
by the CLI for your account) as the synchronized repository folder below.

### 3. Create and run the bootstrap Workflow

In **Workflows → Create job**, create a serverless **Python script** task named
`refresh_and_publish`.

- Python file: `rag/jobs/refresh_app_index.py` from the synchronized bundle folder.
- Environment dependency: the synchronized repository folder containing `pyproject.toml`.
- Timeout: 3,600 seconds.
- Task parameters (replace the placeholders):

```text
--catalog <catalog>
--schema <schema>
--warehouse-id <warehouse-id>
--artifact-volume rag_artifacts
--embedding-endpoint databricks-qwen3-embedding-0-6b
--schema-sql-path <synchronized-repository-folder>/sql/001_rag_schema.sql
```

Click **Run now** and wait for success. This idempotently creates the catalog, schema, all Delta
tables, and Volume; refreshes the configured sources; and writes the active App snapshot under
`rag_artifacts/app-qwen3-embedding-0-6b/`. Inspect the task output before proceeding.

The bundle also contains this Job definition in [`resources/ingestion_job.yml`](resources/ingestion_job.yml).
For a brand-new workspace, create the bootstrap Job first as described above: the App's resource
bindings cannot be created until its tables and Volume exist.

### 4. Create and deploy the App

In **Apps → Create app**, use the same synchronized repository folder as the source. Configure
these resources for the App service principal:

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

To update documentation, run the existing Workflow again from **Workflows**. It creates a new
validated snapshot and atomically marks it active; the previous active snapshot remains usable if
the refresh fails.

To deploy code/configuration changes, synchronize again, then redeploy the App from its source
folder. If the ingestion code, SQL schema, source configuration, or embedding endpoint changed,
run the Workflow after synchronization and before redeploying the App. Do not rebuild a local
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
