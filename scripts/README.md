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
