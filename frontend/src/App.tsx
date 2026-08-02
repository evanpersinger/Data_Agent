import { useCallback, useEffect, useRef, useState } from "react";
import { streamChat } from "./api";
import type { Message } from "./types";

// One session per page load. Reloading starts a fresh conversation, and the
// backend persists each session's history to sessions.db.
const SESSION_ID = crypto.randomUUID();

export function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = useCallback(async () => {
    const prompt = input.trim();
    if (!prompt || isStreaming) return;

    const replyId = crypto.randomUUID();
    setInput("");
    setError(null);
    setIsStreaming(true);
    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: "user", content: prompt },
      { id: replyId, role: "agent", content: "" },
    ]);

    try {
      for await (const chunk of streamChat(prompt, SESSION_ID)) {
        setMessages((prev) =>
          prev.map((message) =>
            message.id === replyId
              ? { ...message, content: message.content + chunk }
              : message,
          ),
        );
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setIsStreaming(false);
    }
  }, [input, isStreaming]);

  return (
    <div className="app">
      <header className="header">
        <h1>Data Agent</h1>
      </header>

      <main className="messages">
        {messages.length === 0 && (
          <p className="empty">
            Ask about the files in <code>raw_data/</code>, or have it pull
            something down from Kaggle.
          </p>
        )}

        {messages.map((message) => (
          <article key={message.id} className={`message ${message.role}`}>
            <span className="who">{message.role === "user" ? "you" : "agent"}</span>
            <div className="bubble">
              {message.content || <span className="dots">thinking</span>}
            </div>
          </article>
        ))}

        {error && <p className="error">{error}</p>}
        <div ref={bottomRef} />
      </main>

      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault();
          void send();
        }}
      >
        <input
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="ask something about your data..."
          disabled={isStreaming}
          autoFocus
        />
        <button type="submit" disabled={isStreaming || !input.trim()}>
          {isStreaming ? "..." : "send"}
        </button>
      </form>
    </div>
  );
}
