-- Substitute ${catalog}, ${schema}, and ${artifact_volume} in the execution layer.
CREATE CATALOG IF NOT EXISTS ${catalog};
CREATE SCHEMA IF NOT EXISTS ${catalog}.${schema};

CREATE TABLE IF NOT EXISTS ${catalog}.${schema}.rag_documents (
  doc_id STRING NOT NULL, requested_url STRING NOT NULL, canonical_url STRING NOT NULL,
  resolved_url STRING, title STRING, category STRING NOT NULL, source_last_updated STRING,
  source_content_hash STRING, document_version STRING, status STRING NOT NULL,
  http_status INT, error_message STRING, consecutive_404_count INT NOT NULL,
  retrieved_at TIMESTAMP, last_success_at TIMESTAMP, last_run_at TIMESTAMP, removed_at TIMESTAMP,
  source_origins ARRAY<STRING>, indexed_content_hash STRING, indexed_source_last_updated STRING,
  last_run_action STRING
) USING DELTA;


CREATE TABLE IF NOT EXISTS ${catalog}.${schema}.rag_chunks (
  chunk_id STRING NOT NULL, doc_id STRING NOT NULL, document_version STRING NOT NULL,
  position INT NOT NULL, chunk_text STRING NOT NULL, heading_path ARRAY<STRING>,
  source_url STRING NOT NULL, source_title STRING NOT NULL, embedding_model STRING,
  embedding_dimension INT, embedding_created_at TIMESTAMP
) USING DELTA;

CREATE TABLE IF NOT EXISTS ${catalog}.${schema}.rag_index_snapshots (
  snapshot_id STRING NOT NULL, embedding_model STRING NOT NULL, embedding_dimension INT NOT NULL,
  chunk_count BIGINT NOT NULL, artifact_path STRING NOT NULL, chunk_map_path STRING NOT NULL,
  source_snapshot_at TIMESTAMP, created_at TIMESTAMP NOT NULL, status STRING NOT NULL,
  active BOOLEAN NOT NULL, validation_error STRING, corpus_fingerprint STRING
) USING DELTA;


CREATE TABLE IF NOT EXISTS ${catalog}.${schema}.rag_evaluations (
  evaluation_id STRING NOT NULL, snapshot_id STRING NOT NULL, question STRING NOT NULL,
  expected_source_url STRING, retrieved_chunk_ids ARRAY<STRING>, recall_at_k DOUBLE,
  source_correct BOOLEAN, evaluator_notes STRING, created_at TIMESTAMP NOT NULL
) USING DELTA;

CREATE TABLE IF NOT EXISTS ${catalog}.${schema}.rag_feedback (
  feedback_id STRING NOT NULL, question STRING NOT NULL, submitted_at TIMESTAMP NOT NULL,
  provider STRING NOT NULL, model STRING, snapshot_id STRING NOT NULL,
  retrieved_chunk_ids ARRAY<STRING>, latency_ms BIGINT, rating STRING, comment STRING
) USING DELTA;

CREATE TABLE IF NOT EXISTS ${catalog}.${schema}.rag_conversations (
  conversation_id STRING NOT NULL, owner_user_id STRING NOT NULL, title STRING NOT NULL,
  status STRING NOT NULL, created_at TIMESTAMP NOT NULL, updated_at TIMESTAMP NOT NULL,
  archived_at TIMESTAMP, deleted_at TIMESTAMP
) USING DELTA;

CREATE TABLE IF NOT EXISTS ${catalog}.${schema}.rag_conversation_turns (
  turn_id STRING NOT NULL, conversation_id STRING NOT NULL, turn_number INT NOT NULL,
  user_question STRING NOT NULL, resolved_query STRING NOT NULL, answer_text STRING NOT NULL,
  supported BOOLEAN NOT NULL, provider STRING NOT NULL, model STRING, snapshot_id STRING NOT NULL,
  citation_chunk_ids ARRAY<STRING>, created_at TIMESTAMP NOT NULL, latency_ms BIGINT
) USING DELTA;

CREATE TABLE IF NOT EXISTS ${catalog}.${schema}.rag_retrieval_traces (
  trace_id STRING NOT NULL, turn_id STRING NOT NULL, search_number INT NOT NULL,
  search_query STRING NOT NULL, retrieved_chunk_ids ARRAY<STRING>, selected_chunk_ids ARRAY<STRING>,
  agent_decision STRING NOT NULL, created_at TIMESTAMP NOT NULL, latency_ms BIGINT
) USING DELTA;

CREATE TABLE IF NOT EXISTS ${catalog}.${schema}.rag_request_traces (
  trace_id STRING NOT NULL, turn_id STRING, conversation_id STRING, owner_user_id STRING,
  user_question STRING NOT NULL, resolved_query STRING NOT NULL,
  retrieval_queries ARRAY<STRING>, retrieval_status STRING,
  selected_evidence_json STRING, raw_model_output STRING,
  parsed_citation_labels ARRAY<STRING>, fallback_reason STRING,
  final_answer_text STRING NOT NULL, supported BOOLEAN NOT NULL,
  provider STRING NOT NULL, model STRING, snapshot_id STRING NOT NULL,
  latency_ms BIGINT, created_at TIMESTAMP NOT NULL
) USING DELTA;

CREATE VOLUME IF NOT EXISTS ${catalog}.${schema}.${artifact_volume};
