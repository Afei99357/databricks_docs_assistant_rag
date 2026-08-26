import { render } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import DOMPurify from "dompurify";
import { marked } from "marked";
import "./styles.css";

type Citation = { label: string; title: string; url: string; excerpt: string };
type Answer = {
  question: string; answer: string; citations: Citation[]; conversation_id?: string;
  snapshot_id?: string; retrieved_chunk_ids?: string[]; latency_ms?: number;
};
type Conversation = { conversation_id: string; title: string; updated_at: string };
type Turn = { turn_id: string; question: string; answer: string };
type Message = { role: "user" | "assistant"; content: string; answer?: Answer };

const STARTERS = [
  "What is Volume Content Search and what are its limitations?",
  "How do Unity Catalog privileges affect Genie?",
  "How can I embed Genie in an application?",
  "How do Genie Agents use unstructured data in Volumes?",
];

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || "The request could not be completed.");
  return body as T;
}

function markdown(text: string, citations: Citation[]) {
  const sourceLabels = new Set(citations.map((citation) => citation.label));
  const html = marked.parse(text, { gfm: true, breaks: true }) as string;
  const withCitations = html.replace(/\[(S\d+)\]/g, (match, label: string) => (
    sourceLabels.has(label)
      ? `<a class="citation" href="#source-${label}" aria-label="Open ${label}">${label.slice(1)}</a>`
      : match
  ));
  return DOMPurify.sanitize(withCitations, {
    ADD_ATTR: ["target"],
    ADD_TAGS: ["details", "summary"],
  });
}

function References({ citations }: { citations: Citation[] }) {
  if (!citations.length) return null;
  return <details class="references">
    <summary>Official references <span>{citations.length} source{citations.length === 1 ? "" : "s"}</span></summary>
    <p class="reference-intro">Retrieved excerpts supporting this answer.</p>
    {citations.map((citation) => <article id={`source-${citation.label}`} class="reference-card" key={citation.label}>
      <a href={citation.url} target="_blank" rel="noreferrer">{citation.label}: {citation.title}</a>
      <p class="reference-url">{citation.url}</p>
      <p>{citation.excerpt}</p>
    </article>)}
  </details>;
}

function Feedback({ answer }: { answer: Answer }) {
  const [rating, setRating] = useState<"up" | "down" | "">("");
  const [comment, setComment] = useState("");
  const [saving, setSaving] = useState(false);
  const submit = async (nextRating: "up" | "down") => {
    setSaving(true);
    try {
      await json("/api/feedback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
        rating: nextRating, question: answer.question, snapshot_id: answer.snapshot_id,
        retrieved_chunk_ids: answer.retrieved_chunk_ids, latency_ms: answer.latency_ms, comment,
      }) });
      setRating(nextRating);
    } catch {
      // A feedback failure must never interrupt an otherwise useful answer.
    } finally { setSaving(false); }
  };
  return <section class="feedback" aria-label="Answer feedback">
    <span>Was this helpful?</span>
    <button class={rating === "up" ? "selected" : ""} disabled={saving} onClick={() => submit("up")} aria-label="Helpful">👍</button>
    <button class={rating === "down" ? "selected" : ""} disabled={saving} onClick={() => submit("down")} aria-label="Not helpful">👎</button>
    {rating ? <span class="feedback-thanks">Thank you for the feedback.</span> : <input value={comment} onInput={(event) => setComment(event.currentTarget.value)} maxLength={500} placeholder="Optional feedback" />}
  </section>;
}

function Transcript({ messages, busy }: { messages: Message[]; busy: boolean }) {
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => end.current?.scrollIntoView({ behavior: "smooth", block: "end" }), [messages, busy]);
  return <section class="transcript" aria-live="polite" aria-busy={busy}>
    {!messages.length && <div class="welcome"><span class="eyebrow">Grounded documentation search</span><h2>What would you like to know?</h2><p>Ask a Databricks question. Answers are based on indexed official documentation and approved supplemental guidance.</p></div>}
    {messages.map((message, index) => <article class={`message ${message.role}`} key={`${message.role}-${index}`}>
      <div class="message-label">{message.role === "user" ? "You" : "Assistant"}</div>
      {message.role === "assistant" && message.answer
        ? <><div class="answer-markdown" dangerouslySetInnerHTML={{ __html: markdown(message.content, message.answer.citations) }} /><References citations={message.answer.citations} /><Feedback answer={message.answer} /></>
        : <div class="plain-message">{message.content}</div>}
    </article>)}
    {busy && <div class="searching" role="status"><span class="spinner" />Searching indexed documentation and checking the evidence…</div>}
    <div ref={end} />
  </section>;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [conversationId, setConversationId] = useState("");
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const refreshHistory = async () => {
    const body = await json<{ conversations: Conversation[] }>("/api/conversations");
    setConversations([...body.conversations].reverse());
  };
  const openHistory = async () => {
    const next = !historyOpen; setHistoryOpen(next); setError("");
    if (next) { try { await refreshHistory(); } catch (reason) { setError((reason as Error).message); } }
  };
  const newConversation = () => { setConversationId(""); setMessages([]); setQuestion(""); setError(""); };
  const loadConversation = async (id: string) => {
    setBusy(true); setError("");
    try {
      const body = await json<{ turns: Turn[] }>(`/api/conversations/${encodeURIComponent(id)}/turns`);
      setConversationId(id);
      setMessages(body.turns.flatMap((turn) => [{ role: "user" as const, content: turn.question }, { role: "assistant" as const, content: turn.answer }]));
      setHistoryOpen(false);
    } catch (reason) { setError((reason as Error).message); } finally { setBusy(false); }
  };
  const ask = async (event?: Event, starter?: string) => {
    event?.preventDefault();
    const text = (starter || question).trim(); if (!text || busy) return;
    setBusy(true); setError(""); setQuestion("");
    setMessages((current) => [...current, { role: "user", content: text }]);
    try {
      const body = await json<Answer>("/api/answer", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: text, conversation_id: conversationId || undefined }) });
      setConversationId(body.conversation_id || conversationId);
      setMessages((current) => [...current, { role: "assistant", content: body.answer, answer: body }]);
    } catch (reason) {
      const message = (reason as Error).message; setError(message);
      setMessages((current) => [...current, { role: "assistant", content: message }]);
    } finally { setBusy(false); }
  };

  return <main class="app-shell">
    <aside class={`sidebar ${historyOpen ? "open" : ""}`} aria-label="Conversations">
      <div class="brand"><span class="brand-mark">D</span><div><strong>Databricks Docs</strong><small>Documentation Assistant</small></div></div>
      <button class="new-conversation" onClick={newConversation} disabled={busy}>＋ New conversation</button>
      <div class="history-heading"><span>History</span><button class="history-toggle" onClick={openHistory} aria-expanded={historyOpen}>{historyOpen ? "Hide" : "Show"}</button></div>
      {historyOpen && <div class="history-list">{conversations.length ? conversations.map((conversation) => <button class={conversation.conversation_id === conversationId ? "active" : ""} onClick={() => loadConversation(conversation.conversation_id)} disabled={busy} key={conversation.conversation_id}>{conversation.title}</button>) : <p>No past conversations yet.</p>}</div>}
    </aside>
    <section class="chat-pane">
      <header class="topbar"><button class="mobile-history" onClick={openHistory} aria-label="Open conversation history">☰</button><div><h1>Databricks Documentation Assistant</h1><p>Answers grounded in indexed documentation</p></div><button class="new-mobile" onClick={newConversation} disabled={busy}>＋ New</button></header>
      <Transcript messages={messages} busy={busy} />
      {error && <p class="error" role="alert">{error}</p>}
      <footer class="composer-area">
        {!messages.length && <div class="starter-list"><span>Try one:</span>{STARTERS.map((starter) => <button disabled={busy} onClick={() => ask(undefined, starter)} key={starter}>{starter}</button>)}</div>}
        <form class="composer" onSubmit={ask}>
          <textarea value={question} onInput={(event) => setQuestion(event.currentTarget.value)} disabled={busy} placeholder="Ask a Databricks question or a follow-up" rows={3} />
          <button type="submit" disabled={busy || !question.trim()}>{busy ? "Searching…" : "Ask"}</button>
        </form>
      </footer>
    </section>
  </main>;
}

render(<App />, document.getElementById("root")!);
