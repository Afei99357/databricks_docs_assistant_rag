# Operator commands for the custom Databricks docs RAG.
#
# Local recipes read .env automatically, so the set -a / source / set +a dance
# is no longer needed. Deployment recipes take the workspace values as
# arguments or from the same .env, and never hardcode a workspace.

set dotenv-load := true
set positional-arguments

# List the available commands.
default:
    @just --list --unsorted

# --- setup -------------------------------------------------------------

# Install locked dependencies, including dev extras.
sync:
    uv sync --extra dev

# Build the Preact browser bundle Flask serves (required before any deploy).
frontend:
    cd frontend && npm install && npm run build

# Pull the local Ollama models.
models:
    ollama pull "$RAG_EMBEDDING_MODEL"

# --- development -------------------------------------------------------

# Run the test suite.
test *args:
    uv run pytest "$@"

# Lint.
lint:
    uv run ruff check .

# Both, as CI would.
check: lint test

# --- local index and server --------------------------------------------

# Fetch and list the official source URLs without ingesting them.
discover:
    uv run python -m rag.cli discover

# Create the governed Delta tables and artifact Volume. Idempotent.
setup-db:
    uv run python -m rag.cli setup-db

# Refresh the configured sources and rebuild the local Ollama-backed FAISS snapshot.
index:
    uv run python -m rag.cli build-local-snapshot

# Repair stored chunks: re-chunk everything, ignoring change detection.
repair-chunks:
    uv run python -m rag.cli repair-chunks

# Serve the active local snapshot on http://127.0.0.1:8000.
serve:
    uv run python -m rag.cli serve

# Score the 25-question battery. mode is "agent" (default) or "plain".
eval mode="agent":
    uv run python -m rag.cli evaluate --mode {{mode}}

# Score both, to see what the agent's extra latency buys.
eval-both:
    @just eval plain
    @just eval agent

# --- Databricks deployment ---------------------------------------------
#
# These use the profile and workspace values already in .env. The bootstrap
# Workflow must run successfully once before the App is deployed: the App's
# resource bindings cannot be created until its tables and Volume exist.

_bundle_vars := '--var "catalog=$RAG_CATALOG" --var "schema=$RAG_SCHEMA" ' + \
    '--var "artifact_volume=$RAG_ARTIFACT_VOLUME" --var "warehouse_id=$RAG_WAREHOUSE_ID" ' + \
    '--var "embedding_endpoint=$RAG_DATABRICKS_EMBEDDING_ENDPOINT"'

# Confirm both serving endpoints exist in the target workspace.
endpoints:
    databricks serving-endpoints get "$RAG_DATABRICKS_EMBEDDING_ENDPOINT" --profile "$RAG_DATABRICKS_PROFILE"
    databricks serving-endpoints get "$RAG_DATABRICKS_CHAT_ENDPOINT" --profile "$RAG_DATABRICKS_PROFILE"

# Upload the project source as the Job-only bootstrap Bundle.
bootstrap-deploy:
    cd bootstrap && databricks bundle validate --target dev --profile "$RAG_DATABRICKS_PROFILE" {{_bundle_vars}}
    cd bootstrap && databricks bundle deploy --target dev --profile "$RAG_DATABRICKS_PROFILE" {{_bundle_vars}}

# Run the bootstrap Workflow: creates the schema, refreshes sources, publishes the App snapshot.
bootstrap-run:
    cd bootstrap && databricks bundle run refresh_databricks_docs_index --target dev \
        --profile "$RAG_DATABRICKS_PROFILE" {{_bundle_vars}}

# Deploy the App Bundle's resource configuration. Run `just frontend` first.
app-deploy:
    databricks bundle deploy --target dev --profile "$RAG_DATABRICKS_PROFILE" {{_bundle_vars}} \
        --var "reasoning_endpoint=$RAG_DATABRICKS_CHAT_ENDPOINT"

# Publish the uploaded source to the App. RAG_APP_SOURCE_PATH is the Bundle deployer's workspace path.
app-publish:
    databricks apps deploy "$RAG_APP_NAME" --profile "$RAG_DATABRICKS_PROFILE" \
        --source-code-path "$RAG_APP_SOURCE_PATH" --mode SNAPSHOT

# Start an App for the first time, then publish its first code deployment.
app-start:
    databricks apps start "$RAG_APP_NAME" --profile "$RAG_DATABRICKS_PROFILE"
    @just app-publish

# Show the App's URL and running status.
app-status:
    databricks apps get "$RAG_APP_NAME" --profile "$RAG_DATABRICKS_PROFILE"

# Ship a code-only change to an existing App: no crawl, no re-embedding, no new snapshot.
deploy-code:
    @just frontend
    @just setup-db
    @just app-deploy
    @just app-publish
    @just app-status
