import { render } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import DOMPurify from "dompurify";
import { marked } from "marked";
import "./styles.css";

type Citation = { label: string; title: string; url: string; excerpt: string; chunk_id?: string };
type Answer = {
  question: string;
  answer: string;
  citations: Citation[];
  conversation_id?: string;
  snapshot_id?: string;
  retrieved_chunk_ids?: string[];
  latency_ms?: number;
};
type ResearchStep = {
  kind?: string;
  turn?: number;
  action?: string;
  status?: string;
  query?: string | null;
  count?: number;
  message?: string;
};
type Conversation = {
  conversation_id: string;
  title: string;
  updated_at: string;
};
type Turn = {
  turn_id: string;
  question: string;
  answer: string;
  citations: Citation[];
  snapshot_id?: string;
};
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

async function streamAnswer(
  payload: object,
  onProgress: (step: ResearchStep) => void,
): Promise<Answer> {
  const response = await fetch("/api/answer/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok || !response.body) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || "The request could not be completed.");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answer: Answer | undefined;
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";
    for (const entry of events) {
      const event = entry.match(/^event: (.+)$/m)?.[1];
      const data = entry.match(/^data: (.+)$/m)?.[1];
      if (!event || !data) continue;
      const body = JSON.parse(data);
      if (event === "progress") onProgress(body);
      if (event === "answer") answer = body;
      if (event === "error") throw new Error(body.error || "The request could not be completed.");
    }
    if (done) break;
  }
  if (!answer) throw new Error("The request ended before an answer was returned.");
  return answer;
}

function markdown(text: string, citations: Citation[]) {
  const sourceLabels = new Set(citations.map((citation) => citation.label));
  const html = marked.parse(text, { gfm: true, breaks: true }) as string;
  // The backend groups multiple citations sharing one sentence into a single
  // bracket, e.g. "[S3, S4]" -- render each label as its own separate link
  // instead of leaving the whole group unmatched/unlinked.
  const withCitations = html.replace(/\[((?:S\d+)(?:\s*,\s*S\d+)*)\]/g, (_match, group: string) =>
    (group.match(/S\d+/g) ?? [])
      .map((label) =>
        sourceLabels.has(label)
          ? `<a class="citation" href="#source-${label}" aria-label="Open ${label}">${label.slice(1)}</a>`
          : label,
      )
      .join(", "),
  );
  return DOMPurify.sanitize(withCitations, {
    ADD_ATTR: ["target"],
    ADD_TAGS: ["details", "summary"],
  });
}

function References({ citations }: { citations: Citation[] }) {
  if (!citations.length) return null;
  return (
    <details class="references">
      <summary>
        Official references{" "}
        <span>
          {citations.length} source{citations.length === 1 ? "" : "s"}
        </span>
      </summary>
      <p class="reference-intro">Retrieved excerpts supporting this answer.</p>
      {citations.map((citation) => (
        <article id={`source-${citation.label}`} class="reference-card" key={citation.label}>
          <a href={citation.url} target="_blank" rel="noreferrer">
            {citation.label}: {citation.title}
          </a>
          <p class="reference-url">{citation.url}</p>
          <p>{citation.excerpt}</p>
        </article>
      ))}
    </details>
  );
}

function activityText(step: ResearchStep) {
  const suffix = step.count ? ` (${step.count} section${step.count === 1 ? "" : "s"})` : "";
  const fallback =
    (
      {
        search_docs: "Searched documentation",
        read_chunks: "Read relevant sections",
        search_within_document: "Refined within a document",
        get_related_chunks: "Read related sections",
        final: "Selected supporting evidence",
      } as Record<string, string>
    )[step.action || ""] || "Updated research activity";
  return `${step.message || fallback}${suffix}`;
}

function ResearchActivity({ steps, live = false }: { steps: ResearchStep[]; live?: boolean }) {
  return (
    <section class={`research-activity ${live ? "live" : ""}`} aria-live="polite">
      <div class="research-heading">
        {live && <span class="spinner" />}
        <strong>{live ? "Researching documentation" : "Research activity"}</strong>
      </div>
      <ol>
        {steps.map((step, index) => (
          <li key={`${step.turn || "status"}-${step.action || step.kind}-${index}`}>
            <span class={`activity-mark ${step.kind === "preparing" ? "pending" : "done"}`}>
              {step.kind === "preparing" ? "•" : "✓"}
            </span>
            <span>{activityText(step)}</span>
            {step.query && <code>{step.query}</code>}
          </li>
        ))}
      </ol>
    </section>
  );
}

function Feedback({ answer }: { answer: Answer }) {
  const [rating, setRating] = useState<"up" | "down" | "">("");
  const [comment, setComment] = useState("");
  const [saving, setSaving] = useState(false);
  const submit = async (nextRating: "up" | "down") => {
    setSaving(true);
    try {
      await json("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          rating: nextRating,
          question: answer.question,
          snapshot_id: answer.snapshot_id,
          retrieved_chunk_ids: answer.retrieved_chunk_ids,
          latency_ms: answer.latency_ms,
          comment,
        }),
      });
      setRating(nextRating);
    } catch {
      // A feedback failure must never interrupt an otherwise useful answer.
    } finally {
      setSaving(false);
    }
  };
  return (
    <section class="feedback" aria-label="Answer feedback">
      <span>Was this helpful?</span>
      <button
        class={rating === "up" ? "selected" : ""}
        disabled={saving}
        onClick={() => submit("up")}
        aria-label="Helpful"
      >
        👍
      </button>
      <button
        class={rating === "down" ? "selected" : ""}
        disabled={saving}
        onClick={() => submit("down")}
        aria-label="Not helpful"
      >
        👎
      </button>
      {rating ? (
        <span class="feedback-thanks">Thank you for the feedback.</span>
      ) : (
        <input
          value={comment}
          onInput={(event) => setComment(event.currentTarget.value)}
          maxLength={500}
          placeholder="Optional feedback"
        />
      )}
    </section>
  );
}

function Transcript({
  messages,
  busy,
  activity,
}: {
  messages: Message[];
  busy: boolean;
  activity: ResearchStep[];
}) {
  const end = useRef<HTMLDivElement>(null);
  useEffect(
    () => end.current?.scrollIntoView({ behavior: "smooth", block: "end" }),
    [messages, busy],
  );
  return (
    <section class="transcript" aria-live="polite" aria-busy={busy}>
      {!messages.length && (
        <div class="welcome">
          <span class="eyebrow">Grounded documentation search</span>
          <h2>What would you like to know?</h2>
          <p>
            Ask a Databricks question. Answers are based on indexed official documentation and
            approved supplemental guidance.
          </p>
        </div>
      )}
      {messages.map((message, index) => (
        <article class={`message ${message.role}`} key={`${message.role}-${index}`}>
          <div class="message-label">{message.role === "user" ? "You" : "Assistant"}</div>
          {message.role === "assistant" && message.answer ? (
            <>
              <div
                class="answer-markdown"
                dangerouslySetInnerHTML={{
                  __html: markdown(message.content, message.answer.citations),
                }}
              />
              <References citations={message.answer.citations} />
              <Feedback answer={message.answer} />
            </>
          ) : (
            <div class="plain-message">{message.content}</div>
          )}
        </article>
      ))}
      {busy && (
        <ResearchActivity
          steps={
            activity.length
              ? activity
              : [
                  {
                    kind: "starting",
                    message: "Starting documentation research.",
                  },
                ]
          }
          live
        />
      )}
      <div ref={end} />
    </section>
  );
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [conversationId, setConversationId] = useState("");
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [activity, setActivity] = useState<ResearchStep[]>([]);
  const [deletingConversationId, setDeletingConversationId] = useState("");
  const [error, setError] = useState("");
  const [sidebarWidth, setSidebarWidth] = useState(() => {
    const saved = Number(localStorage.getItem("rag-sidebar-width"));
    return Number.isFinite(saved) && saved >= 270 && saved <= 520 ? saved : 270;
  });
  const [desktop, setDesktop] = useState(() => window.innerWidth > 760);

  const refreshHistory = async () => {
    const body = await json<{ conversations: Conversation[] }>("/api/conversations");
    const rows = [...body.conversations].reverse();
    setConversations(rows);
    return rows;
  };
  useEffect(() => {
    refreshHistory()
      .then((rows) => {
        if (rows.length) setHistoryOpen(true);
      })
      .catch(() => undefined);
  }, []);
  useEffect(() => {
    const update = () => setDesktop(window.innerWidth > 760);
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);
  const openHistory = async () => {
    const next = !historyOpen;
    setHistoryOpen(next);
    setError("");
    if (next) {
      try {
        await refreshHistory();
      } catch (reason) {
        setError((reason as Error).message);
      }
    }
  };
  const newConversation = () => {
    setConversationId("");
    setMessages([]);
    setQuestion("");
    setError("");
    setActivity([]);
  };
  const loadConversation = async (id: string) => {
    setBusy(true);
    setError("");
    try {
      const body = await json<{ turns: Turn[] }>(
        `/api/conversations/${encodeURIComponent(id)}/turns`,
      );
      setConversationId(id);
      setMessages(
        body.turns.flatMap((turn) => [
          { role: "user" as const, content: turn.question },
          {
            role: "assistant" as const,
            content: turn.answer,
            answer: {
              question: turn.question,
              answer: turn.answer,
              citations: turn.citations || [],
              snapshot_id: turn.snapshot_id,
            },
          },
        ]),
      );
      setHistoryOpen(false);
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const deleteConversation = async (id: string, event: MouseEvent) => {
    event.stopPropagation();
    if (busy || deletingConversationId) return;
    const previousConversations = conversations;
    const previousConversationId = conversationId;
    const previousMessages = messages;
    const previousQuestion = question;
    const wasActive = conversationId === id;
    setDeletingConversationId(id);
    setError("");
    setConversations((current) =>
      current.filter((conversation) => conversation.conversation_id !== id),
    );
    if (wasActive) newConversation();
    try {
      await json(`/api/conversations/${encodeURIComponent(id)}`, {
        method: "DELETE",
      });
    } catch (reason) {
      setConversations(previousConversations);
      if (wasActive) {
        setConversationId(previousConversationId);
        setMessages(previousMessages);
        setQuestion(previousQuestion);
      }
      setError((reason as Error).message);
    } finally {
      setDeletingConversationId("");
    }
  };
  const startResize = (event: PointerEvent) => {
    event.preventDefault();
    const startX = event.clientX;
    const startWidth = sidebarWidth;
    const resize = (move: PointerEvent) =>
      setSidebarWidth(Math.min(520, Math.max(270, startWidth + move.clientX - startX)));
    const finish = () => {
      window.removeEventListener("pointermove", resize);
      window.removeEventListener("pointerup", finish);
      setSidebarWidth((width) => {
        localStorage.setItem("rag-sidebar-width", String(width));
        return width;
      });
    };
    window.addEventListener("pointermove", resize);
    window.addEventListener("pointerup", finish);
  };
  const ask = async (event?: Event, starter?: string) => {
    event?.preventDefault();
    const text = (starter || question).trim();
    if (!text || busy) return;
    setBusy(true);
    setError("");
    setQuestion("");
    setActivity([]);
    setMessages((current) => [...current, { role: "user", content: text }]);
    try {
      const body = await streamAnswer(
        { question: text, conversation_id: conversationId || undefined },
        (step) => setActivity((current) => [...current, step]),
      );
      setConversationId(body.conversation_id || conversationId);
      setMessages((current) => [
        ...current,
        { role: "assistant", content: body.answer, answer: body },
      ]);
    } catch (reason) {
      const message = (reason as Error).message;
      setError(message);
      setMessages((current) => [...current, { role: "assistant", content: message }]);
    } finally {
      setBusy(false);
      setActivity([]);
    }
  };

  return (
    <main class="app-shell" style={{ gridTemplateColumns: `${sidebarWidth}px minmax(0, 1fr)` }}>
      <aside
        class={`sidebar ${historyOpen ? "open" : ""}`}
        aria-label="Conversations"
        style={{ position: "relative" }}
      >
        <div class="brand">
          <span class="brand-mark">D</span>
          <div>
            <strong>Databricks Docs</strong>
            <small>Documentation Assistant</small>
          </div>
        </div>
        <button class="new-conversation" onClick={newConversation} disabled={busy}>
          ＋ New conversation
        </button>
        <div class="history-heading">
          <span>History</span>
          <button class="history-toggle" onClick={openHistory} aria-expanded={historyOpen}>
            {historyOpen ? "Hide" : "Show"}
          </button>
        </div>
        {historyOpen && (
          <div class="history-list" style={{ overflowX: "hidden" }}>
            {conversations.length ? (
              conversations.map((conversation) => (
                <div
                  class={`history-item ${conversation.conversation_id === conversationId ? "active" : ""}`}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "minmax(0, 1fr) auto",
                    minWidth: 0,
                    width: "100%",
                  }}
                  key={conversation.conversation_id}
                >
                  <button
                    class="conversation-title"
                    onClick={() => loadConversation(conversation.conversation_id)}
                    disabled={busy || !!deletingConversationId}
                    title={conversation.title}
                    style={{ minWidth: 0, width: "100%" }}
                  >
                    {conversation.title}
                  </button>
                  <button
                    class="delete-conversation"
                    onClick={(event) => deleteConversation(conversation.conversation_id, event)}
                    disabled={busy || !!deletingConversationId}
                    aria-label={`Delete ${conversation.title}`}
                    title="Delete conversation"
                  >
                    {deletingConversationId === conversation.conversation_id ? "…" : "×"}
                  </button>
                </div>
              ))
            ) : (
              <p>No past conversations yet.</p>
            )}
          </div>
        )}
        {desktop && (
          <div
            onPointerDown={startResize}
            role="separator"
            aria-orientation="vertical"
            aria-label="Resize conversation history"
            style={{
              position: "absolute",
              top: "50%",
              right: -8,
              width: 16,
              height: 48,
              transform: "translateY(-50%)",
              display: "grid",
              placeItems: "center",
              background: "#1e4c72",
              border: "1px solid #6089ae",
              borderRadius: "0 .4rem .4rem 0",
              color: "#c9e3ff",
              cursor: "col-resize",
              userSelect: "none",
              zIndex: 2,
            }}
          >
            ⋮
          </div>
        )}
      </aside>
      <section class="chat-pane">
        <header class="topbar">
          <button
            class="mobile-history"
            onClick={openHistory}
            aria-label="Open conversation history"
          >
            ☰
          </button>
          <div>
            <h1>Databricks Documentation Assistant</h1>
            <p>Answers grounded in indexed documentation</p>
          </div>
          <button class="new-mobile" onClick={newConversation} disabled={busy}>
            ＋ New
          </button>
        </header>
        <Transcript messages={messages} busy={busy} activity={activity} />
        {error && (
          <p class="error" role="alert">
            {error}
          </p>
        )}
        <footer class="composer-area">
          {!messages.length && (
            <div class="starter-list">
              <span>Try one:</span>
              {STARTERS.map((starter) => (
                <button disabled={busy} onClick={() => ask(undefined, starter)} key={starter}>
                  {starter}
                </button>
              ))}
            </div>
          )}
          <form class="composer" onSubmit={ask}>
            <textarea
              value={question}
              onInput={(event) => setQuestion(event.currentTarget.value)}
              disabled={busy}
              placeholder="Ask a Databricks question or a follow-up"
              rows={3}
            />
            <button type="submit" disabled={busy || !question.trim()}>
              {busy ? "Searching…" : "Ask"}
            </button>
          </form>
        </footer>
      </section>
    </main>
  );
}

render(<App />, document.getElementById("root")!);
