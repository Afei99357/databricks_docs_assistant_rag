# Custom Databricks Documentation RAG

An internal, inspectable RAG application over official Databricks documentation. It uses
local Qwen embeddings and FAISS for retrieval. It can use Unity Catalog for governed metadata,
feedback, evaluation results, and immutable index artifacts, or SQLite for a fully local store.
It does not use Genie Agent,
Volume Content Search, AI Search, or Vector Search.

## Contents

- [Current status](#current-status)
- [Databricks App mode](#databricks-app-mode)
- [Local setup](#local-setup)
  - [Local configuration reference](#local-configuration-reference)
  - [Index and repair commands](#index-and-repair-commands)
  - [Measuring retrieval quality](#measuring-retrieval-quality)
- [Deploy to a new Databricks workspace](#deploy-to-a-new-databricks-workspace)
  - [Prerequisites](#1-prerequisites)
  - [Upload the project source](#2-upload-the-project-source)
  - [Create and run the bootstrap Workflow](#3-create-and-run-the-bootstrap-workflow)
  - [Create and deploy the App](#4-create-and-deploy-the-app)
  - [Refresh or redeploy later](#5-refresh-or-redeploy-later)
- [Chat model configuration](#chat-model-configuration)
- [Operational boundaries](#operational-boundaries)

## Current status

The local prototype includes official-source discovery/fetch/extraction, deterministic
chunking, local Qwen embedding adapters, FAISS snapshot validation/activation, retrieval
evaluation, a grounded Flask UI, and pluggable Databricks or SQLite storage. No real workspace values or
credentials belong in this repository.

## Databricks App mode

The same Flask application can run as a Databricks App. It uses the App service principal
for the shared FAISS snapshot in the artifact Volume, Delta conversation history/feedback,
and model inference. It uses the forwarded user token only to derive a trusted history owner;
no browser-supplied user ID is accepted.

- embeddings: `databricks-qwen3-embedding-0-6b`
- retrieval reasoning and answer generation: the configured `RAG_CHAT_MODEL`
  (tested with `databricks-claude-sonnet-4-5`)
- local `rag serve` can use Ollama or any OpenAI-compatible server such as
  llama.cpp hosting Muse.

Before deploying, build and publish a separate snapshot with the Databricks Qwen embedding
endpoint. It is stored under `rag_artifacts/app-qwen3-embedding-0-6b/`, separately from any
local/Ollama artifact. A FAISS index built with the local Ollama model cannot be queried with a
different embedding model. The bundle requests least-privilege access to the artifact Volume,
history/feedback tables, SQL warehouse, and the two model endpoints.

## Local setup

The local server uses Ollama for embeddings and a local FAISS directory for retrieval. Its chat
model is independently configurable: use Muse through llama.cpp's OpenAI-compatible API, native
Ollama, or a Databricks serving endpoint. The local setup below uses SQLite for documents,
history, feedback, and traces; it requires no Databricks account or SQL warehouse.

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

2. Edit `.env` for the local SQLite path. A project `.env` represents the mode you are about
   to run; for this path, use these values and do **not** set Databricks catalog, warehouse, or
   profile values:

   ```text
   RAG_STORAGE_BACKEND=sqlite
   RAG_SQLITE_PATH=./data/local.sqlite
   RAG_LOCAL_INDEX_DIR=/absolute/path/to/local-faiss-index
   RAG_LOCAL_TEST_USER_ID=your-email@example.com
   RAG_EMBEDDING_BASE_URL=http://localhost:11434
   RAG_EMBEDDING_MODEL=qwen3-embedding:4b
   RAG_AGENT_CANDIDATES_PER_SEARCH=10
   RAG_CHAT_BASE_URL=http://your-chat-host:1234/v1
   RAG_CHAT_MODEL=your-chat-model
   RAG_CHAT_API_KEY=local
   ```

   `RAG_SQLITE_PATH` is optional—the value shown is its default—but it makes the local data
   location explicit. It stores the downloaded corpus, conversation history, feedback, and
   request traces. `RAG_LOCAL_INDEX_DIR` is separate: it holds the generated FAISS artifacts.

   `RAG_CHAT_*` controls both retrieval-agent reasoning and final-answer
   generation. A model such as Muse running through llama.cpp belongs to
   the OpenAI-compatible API; the API protocol, not the model, is what the
   app requires. An Ollama chat model can use the same configuration with
   `RAG_CHAT_BASE_URL=http://localhost:11434/v1`, but should be evaluated for
   tool-call reliability before using agent mode.

   `RAG_AGENT_CANDIDATES_PER_SEARCH` is the maximum number of ranked chunks
   exposed to the agent for each search tool call. The default is 10.

3. Start Ollama and make sure the embedding model is available:

   ```bash
   just models
   ```

4. Fetch and chunk the configured sources, then build the local Ollama-backed FAISS snapshot:

   ```bash
   just index
   ```

5. Start the local application and open `http://127.0.0.1:8000`:

   ```bash
   just serve
   ```

### Local configuration reference

| Variable | Purpose |
| --- | --- |
| `RAG_STORAGE_BACKEND` | Use `sqlite` for the fully local store; use `databricks` only when deliberately sharing Unity Catalog storage. |
| `RAG_SQLITE_PATH` | SQLite file for the local corpus, conversations, feedback, and request traces. Defaults to `./data/local.sqlite`. |
| `RAG_LOCAL_INDEX_DIR` | Directory containing the local FAISS snapshot artifacts. |
| `RAG_EMBEDDING_BASE_URL` | Ollama server used to generate document and query embeddings. |
| `RAG_EMBEDDING_MODEL` | Ollama embedding model name. |
| `RAG_EMBEDDING_REVISION` | Cache revision for embedding behavior. Increment after changing model normalization, prefixes, or another vector-space behavior. |
| `RAG_CHUNKING_REVISION` | Chunking revision. Increment after changing chunking behavior; it deliberately creates new chunk IDs and embeddings. |
| `RAG_CHAT_BASE_URL` | OpenAI-compatible endpoint used for retrieval reasoning and final answers. |
| `RAG_CHAT_MODEL` | Model name served by `RAG_CHAT_BASE_URL`. |
| `RAG_CHAT_API_KEY` | Credential for the chat endpoint; use `local` if the server does not require one. |
| `RAG_AGENT_CANDIDATES_PER_SEARCH` | Maximum retrieved chunks exposed to the agent per search. Higher values increase context and latency; `10` is the balanced default. |
| `RAG_RELEVANCE_THRESHOLD` | Minimum grounding/relevance score required before the UI returns an answer. |
| `RAG_LOCAL_TEST_USER_ID` | Local identity used to scope conversation history in the browser. |

`RAG_AGENT_BASE_URL`, `RAG_AGENT_MODEL`, and `RAG_AGENT_API_KEY` are optional advanced
overrides for using a different model for retrieval reasoning. Leave all three unset to use
the regular `RAG_CHAT_*` model for both reasoning and final answers.

While developing, `just check` runs what CI runs — `ruff` followed by the test suite. Its two
halves are also available on their own, and `just test` forwards any arguments to `pytest`:

```bash
just check              # lint and tests, as CI would
just lint
just test tests/test_sources.py
```

`just discover` fetches and lists the official source URLs without ingesting them, which is the
cheap way to see what the configured roots and supplements currently resolve to.

`just check` runs the SQLite behavioural contract suite for the storage protocols
(conversation round-tripping, owner isolation, delete-twice semantics). The same suite also
runs against the real Databricks warehouse only when explicitly requested; it creates and
soft-deletes conversation rows under a clearly marked owner id
(`storage-contract-test@example.invalid`), so those rows persist. Run that check deliberately:

```bash
just test-storage-databricks
```

### Index and repair commands

#### Refresh changed sources

```bash
just index
```

Fetches configured sources, re-chunks only changed documents, embeds only missing vectors, and
builds a new local FAISS snapshot when the corpus or embedding specification changed.

#### Repair stored chunks

A refresh rewrites a document's chunks only when the fetched page differs from what was last
indexed. That is the right default, but it means stored chunks that are wrong for a reason
unrelated to the source — a write-side defect, a change to the chunker — are skipped forever,
because the page itself has not moved.

```bash
just repair-chunks
```

This clears the recorded content hash so every document re-chunks, then rebuilds the local
snapshot even though the corpus fingerprint is unchanged. It costs a full re-embed, so it is
a deliberate repair rather than part of a routine refresh. The Databricks side takes the same
repair through its Workflow: run it with the `repair_chunks` job parameter set to `true`
(the Workflow's "Run with different parameters" option, or `databricks jobs run-now <id>
--json '{"job_parameters": {"repair_chunks": "true"}}'`). It defaults to `false`, so a normal
scheduled or manual run is unaffected.

If the repair is caused by a change to chunking logic or configuration, increment
`RAG_CHUNKING_REVISION` first. That creates new chunk IDs, so the embedding cache cannot reuse
vectors produced for the old chunk text.

In Databricks-backed mode, both local and App snapshots read the same Delta tables, so one
repair fixes stored text for both; each snapshot must still be rebuilt with its own embedding
model. SQLite mode has an independent local corpus and snapshot.

#### Rebuild FAISS from cached vectors

```bash
just rebuild-index
```

Rebuilds the local FAISS snapshot from persisted vectors even when the corpus fingerprint is
unchanged. It does not fetch, re-chunk, or re-embed content; use it only when local FAISS
artifacts need replacement.

The local server does not use the Databricks App service principal, OBO identity, or hosted
model endpoints.

### Measuring retrieval quality

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

Replace the local values in `.env` with these Databricks deployment values before deploying.
This path must use `RAG_STORAGE_BACKEND=databricks`; it is distinct from the local SQLite path
above.

```text
RAG_STORAGE_BACKEND=databricks
RAG_DATABRICKS_PROFILE=<profile>
RAG_CATALOG=<catalog>
RAG_SCHEMA=<schema>
RAG_WAREHOUSE_ID=<sql-warehouse-id>
RAG_ARTIFACT_VOLUME=rag_artifacts
RAG_DATABRICKS_EMBEDDING_ENDPOINT=databricks-qwen3-embedding-0-6b
RAG_DATABRICKS_CHAT_ENDPOINT=<chat-serving-endpoint>
RAG_APP_NAME=<app-name>
RAG_APP_SOURCE_PATH=/Workspace/Users/<deployer-email>/.bundle/databricks-docs-rag/dev/files
```

`RAG_APP_SOURCE_PATH` must match the workspace path written by `databricks bundle deploy` for
the user who deploys the bundle. The `RAG_CHAT_*`, `RAG_LOCAL_INDEX_DIR`, and
`RAG_SQLITE_PATH` values used by the local server are not used by the deployed App or its
bootstrap Workflow.

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

To force a full re-chunk and App-snapshot rebuild after a storage/chunking upgrade, use:

```bash
just bootstrap-repair
```

This passes `repair_chunks=true` to the Workflow. `just` loads the project `.env`, so no
`set -a` / `source` commands are needed.

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
| `rag_feedback`, `rag_conversations`, `rag_conversation_turns`, `rag_request_traces`, `rag_retrieval_traces` | `SELECT` and `MODIFY` as required |
| `databricks-qwen3-embedding-0-6b` | `CAN_QUERY` |
| configured `RAG_DATABRICKS_CHAT_ENDPOINT` | `CAN_QUERY` |

Each answer also writes a request diagnostic record to `rag_request_traces`:
the resolved query, retrieval searches and selected evidence, raw final model
output, parsed citations, and any grounding fallback reason. The bootstrap
Workflow and `rag.cli setup-db` apply the schema before ingestion, so this
table is created on a first workspace setup and on later rebuilds if missing.

The App starts `rag.app.main:app`, reads the active Volume snapshot, uses the App service
principal for shared storage/model calls, and uses OBO only to establish the signed-in user's
conversation owner. The deployed App does not read your local `.env` file.

### 5. Refresh or redeploy later

#### Refresh changed documentation

```bash
just bootstrap-run
```

Runs the Databricks refresh Workflow against the configured source roots. It re-chunks changed
documents, embeds only missing vectors, and publishes a new App snapshot only when the corpus or
embedding specification changed.

#### Fully re-chunk and re-index the App corpus

Use this after a chunking/storage upgrade, a chunking repair, or when the existing governed
chunks cannot be trusted:

```bash
just bootstrap-deploy
just bootstrap-repair
```

`bootstrap-deploy` uploads the current job code and applies the latest schema. `bootstrap-repair`
runs the Workflow with `repair_chunks=true`: it clears indexed hashes, re-chunks all sources,
embeds the resulting vectors, builds a new Databricks-embedding FAISS snapshot, and activates it.
The existing App reads that active snapshot automatically; this data rebuild does not require an
App code deployment.

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

The normal documentation refresh above recursively checks the bounded official roots in
`rag/ingest/config/discovery_roots.yaml` and the manual supplement list. Unchanged sources do not
create embeddings or a new snapshot. A changed corpus creates a validated immutable snapshot and
atomically marks it active; the previous active snapshot remains usable if refresh or publication
fails. A direct 404, or removal from the configured source lists, must be confirmed in three runs
before a page is removed.

To deploy code/configuration changes, run `just deploy-code`. If the ingestion
code, SQL schema, source configuration, or embedding endpoint changed, redeploy the bootstrap
Bundle, run its Workflow, then redeploy the App Bundle. Do not rebuild a local
Ollama snapshot for the Databricks App; the App snapshot must use the configured Databricks
embedding endpoint.

Conversation ownership comes from the forwarded Databricks OBO token. The browser cannot set
the owner ID. The App service principal stores shared history and feedback in Delta tables,
while the user token is used only to resolve the signed-in Databricks user.

## Chat model configuration

Local and Databricks deployments configure chat models differently. In both cases, the
retrieval agent and final answer use the same model unless `RAG_AGENT_*` overrides are set.

### Local server

Changing a local model is an `.env` change followed by restarting `just serve`.

```text
RAG_CHAT_BASE_URL=http://your-muse-host:1234/v1
RAG_CHAT_MODEL=unsloth/Muse-Glimmer-30B-GGUF:UD-Q4_K_XL
RAG_CHAT_API_KEY=local
```

For an Ollama-hosted chat model, set `RAG_CHAT_BASE_URL=http://localhost:11434/v1` and use its
model name for `RAG_CHAT_MODEL`.

### Databricks App

Set `RAG_DATABRICKS_CHAT_ENDPOINT` to the serving-endpoint name before running
`just app-deploy`. The bundle binds that endpoint as the App's `RAG_CHAT_MODEL` resource;
the deployed App does not read the local `RAG_CHAT_BASE_URL`, `RAG_CHAT_MODEL`, or
`RAG_CHAT_API_KEY` values.

`OPENAI_*` and `OLLAMA_MODEL` remain accepted as compatibility aliases, but new configurations
should use `RAG_CHAT_*`. `RAG_AGENT_*` is an advanced override only—leave all three unset to
use the regular chat model for both retrieval reasoning and final answers.

## Operational boundaries

- Build Qwen 0.6B and 4B snapshots separately and select a model from the 25-case evaluation
  set; model size is not a quality decision.
- A snapshot is built in staging, validated against its chunk map, then activated atomically.
  Failed builds leave the current active snapshot unchanged.
- The UI refuses answers below the relevance threshold or without model citations.
- Conversation turns and request traces store the generated answer text, so
  `rag_conversation_turns.answer_text` and `rag_request_traces.final_answer_text` are readable
  by anyone with SELECT on those tables. The feedback sink is the exception: it records only
  request/retrieval diagnostics and the user's rating.
- Costs remain visible: local GPU embedding is explicit, and Delta/Volume, SQL Warehouse, and
  optional Databricks endpoint usage are separately attributable.

See [`scripts/README.md`](scripts/README.md) for manual operating commands and
[`rag/evaluation_cases.yaml`](rag/evaluation_cases.yaml) for the benchmark questions.
