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

Every command below is a [`just`](https://github.com/casey/just) recipe. Run `just` on its own
to list them. The recipes load `.env` themselves, so no `set -a` / `source` step is needed.

1. Install the locked project dependencies and create a private environment file:

   ```bash
   just sync
   cp .env.example .env
   ```

   Build the Preact browser application once, and again after any frontend
   change. Flask serves the compiled files, so this build is also required
   before a Databricks App Bundle deploy:

   ```bash
   just frontend
   ```

2. Edit `.env`. At minimum, configure your Databricks profile/catalog/schema/warehouse and set:

   ```text
   RAG_LOCAL_INDEX_DIR=/absolute/path/to/local-faiss-index
   RAG_LOCAL_TEST_USER_ID=your-email@example.com
   ANSWER_PROVIDER=ollama
   RAG_EMBEDDING_MODEL=qwen3-embedding:4b
   OLLAMA_MODEL=qwen3.5:latest
   ```

3. Start Ollama and make sure both local models are available:

   ```bash
   just models
   ```

4. Create the governed tables/Volume, fetch and chunk the configured sources, and build the
   local Ollama-backed FAISS snapshot:

   ```bash
   just setup-db
   just index
   ```

5. Start the local application and open `http://127.0.0.1:8000`:

   ```bash
   just serve
   ```

Run `just index` again whenever you want to refresh the configured documentation sources and
rebuild the local index.

The local server does not use the Databricks App service principal, OBO identity, or hosted
model endpoints.

## Measuring retrieval quality

`rag/evaluation_cases.yaml` holds 25 questions, each naming the one page its answer must come
from. `just eval` runs the battery and prints a row per question — where the expected page
ranked, how many chunks came back, how long it took — then recall and mean reciprocal rank.
Recall says the evidence was retrieved at all; reciprocal rank says whether it was near the top,
which is the difference an investigating agent is meant to make over a single search.

```bash
just eval plain   # one search per question
just eval agent   # the tool-using retrieval agent (default)
just eval-both    # both, to see what the agent's extra latency buys
```

A question whose retrieval raises is recorded as a failure and the run continues, so one bad
case cannot cost you the other 24 measurements.

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

Set the workspace values once, in `.env`: `RAG_DATABRICKS_PROFILE`, `RAG_CATALOG`, `RAG_SCHEMA`,
`RAG_WAREHOUSE_ID`, `RAG_ARTIFACT_VOLUME`, `RAG_DATABRICKS_EMBEDDING_ENDPOINT`,
`DATABRICKS_CHAT_ENDPOINT`, `RAG_APP_NAME`, and `RAG_APP_SOURCE_PATH`. Every deployment recipe
reads them from there, so no workspace value is repeated in a command.

Confirm both endpoints exist before continuing:

```bash
just endpoints
```

If the catalog/schema differs from `eliao.genie_kb`, update the three corresponding values in
[`app.yml`](app.yml) before creating the App: `RAG_CATALOG`, `RAG_SCHEMA`, and
`RAG_ARTIFACT_VOLUME`.

### 2. Upload the project source

From the repository root, validate and synchronize the source files to the target workspace:

```bash
just bootstrap-deploy
```

This deploys the Job-only bootstrap Bundle and uploads all needed project files. It creates the
Workflow `databricks-docs-rag-refresh-dev`.

### 3. Create and run the bootstrap Workflow

After deployment, run the Bundle-managed Workflow from **Workflows**, or run:

```bash
just bootstrap-run
```

Wait for success. This idempotently creates the catalog, schema, all Delta
tables, and Volume; refreshes the configured sources; and writes the active App snapshot under
`rag_artifacts/app-qwen3-embedding-0-6b/`. Inspect the task output before proceeding.

For a brand-new workspace, the Job must run first: the App's resource bindings cannot be created
until its tables and Volume exist.

### 4. Create and deploy the App

Deploy the App-only Bundle after the bootstrap run succeeds. This Bundle pins
the Terraform deployment engine as a compatibility workaround for a current
Databricks CLI direct-engine crash when creating Apps with resource bindings:

```bash
just app-deploy
```

The Bundle uploads the source and creates the App with its resource bindings.
For the first deployment in a workspace, start the new App and create its code
deployment from that uploaded source (replace the email address with the Bundle
deployer's workspace user):

```bash
just app-start
```

Wait for the deployment to complete, then confirm its URL and running status:

```bash
just app-status
```

The Bundle creates the App and configures these service-principal resources:

| Resource | Permission |
| --- | --- |
| SQL warehouse | `CAN_USE` |
| `rag_artifacts` Volume | `READ_VOLUME` |
| `rag_feedback`, `rag_conversations`, `rag_conversation_turns` | `SELECT` and `MODIFY` |
| `databricks-qwen3-embedding-0-6b` | `CAN_QUERY` |
| `databricks-gpt-oss-20b` | `CAN_QUERY` |

Each answer also writes a request diagnostic record to `rag_request_traces`:
the resolved query, retrieval searches and selected evidence, raw final model
output, parsed citations, and any grounding fallback reason. The bootstrap
Workflow and `rag.cli setup-db` apply the schema before ingestion, so this
table is created on a first workspace setup and on later rebuilds if missing.

The App starts `rag.app.main:app`, reads the active Volume snapshot, uses the App service
principal for shared storage/model calls, and uses OBO only to establish the signed-in user's
conversation owner. The deployed App does not read your local `.env` file.

### 5. Refresh or redeploy later

#### Deploy a code-only update to an existing App (no re-index)

Use this sequence when the Databricks App already exists and the active App
FAISS snapshot is still valid—for example, when deploying a UI, retrieval,
grounding, or request-tracing change. Your local app may remain running; these
commands operate only on the workspace copy.

One recipe does the whole sequence — build the browser assets, create any new Delta tables
introduced by the release, deploy the Bundle resource configuration, publish that uploaded
source to the existing App, and report its status:

```bash
just deploy-code
```

`RAG_APP_SOURCE_PATH` must name the workspace user who ran `bundle deploy`. `setup-db` only
applies idempotent schema creation, and none of these commands crawl sources, create embeddings, or
replace the active FAISS snapshot.

To update documentation, run the bootstrap Bundle's Workflow again from **Workflows** (or run `just bootstrap-run`). It creates a new
validated snapshot and atomically marks it active; the previous active snapshot remains usable if
the refresh fails.

To deploy code/configuration changes, run `just deploy-code`. If the ingestion
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
