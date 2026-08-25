"""User-owned Delta conversation history and retrieval trace persistence."""
from __future__ import annotations

from uuid import uuid4

from rag.store import DatabricksStore, sql_literal


class ConversationRepository:
    def __init__(self, store: DatabricksStore, namespace: str):
        self.store, self.conversations = store, f"{namespace}.rag_conversations"
        self.turns, self.traces = f"{namespace}.rag_conversation_turns", f"{namespace}.rag_retrieval_traces"

    def create(self, owner: str, title: str) -> str:
        conversation_id = uuid4().hex
        self.store.execute(f"INSERT INTO {self.conversations} (conversation_id,owner_user_id,title,status,created_at,updated_at) VALUES ({sql_literal(conversation_id)},{sql_literal(owner)},{sql_literal(title[:160] or 'New conversation')},'active',current_timestamp(),current_timestamp())")
        return conversation_id

    def list(self, owner: str):
        return self.store.execute(f"SELECT conversation_id,title,updated_at FROM {self.conversations} WHERE owner_user_id={sql_literal(owner)} AND status='active' ORDER BY updated_at DESC").rows

    def turns_for(self, owner: str, conversation_id: str):
        return self.store.execute(f"SELECT t.turn_id,t.turn_number,t.user_question,t.answer_text,t.supported,t.snapshot_id,t.citation_chunk_ids,t.created_at FROM {self.turns} t JOIN {self.conversations} c ON t.conversation_id=c.conversation_id WHERE c.owner_user_id={sql_literal(owner)} AND c.conversation_id={sql_literal(conversation_id)} AND c.status='active' ORDER BY t.turn_number").rows

    def append_turn(self, owner: str, conversation_id: str, *, question: str, resolved_query: str, answer, citation_ids: list[str], latency_ms: int) -> None:
        owned = self.store.execute(f"SELECT count(*) FROM {self.conversations} WHERE conversation_id={sql_literal(conversation_id)} AND owner_user_id={sql_literal(owner)} AND status='active'").rows[0][0]
        if int(owned) != 1: raise PermissionError("conversation not found")
        number = int(self.store.execute(f"SELECT coalesce(max(turn_number),0)+1 FROM {self.turns} WHERE conversation_id={sql_literal(conversation_id)}").rows[0][0])
        values = ','.join(sql_literal(value) for value in citation_ids)
        self.store.execute(f"INSERT INTO {self.turns} (turn_id,conversation_id,turn_number,user_question,resolved_query,answer_text,supported,provider,model,snapshot_id,citation_chunk_ids,created_at,latency_ms) VALUES ({sql_literal(uuid4().hex)},{sql_literal(conversation_id)},{number},{sql_literal(question)},{sql_literal(resolved_query)},{sql_literal(answer.text)},{sql_literal(answer.supported)},{sql_literal(answer.provider)},NULL,{sql_literal(answer.snapshot_id)},array({values}),current_timestamp(),{latency_ms})")
        self.store.execute(f"UPDATE {self.conversations} SET updated_at=current_timestamp() WHERE conversation_id={sql_literal(conversation_id)}")
