/**
 * Streams the agent's reply. The backend sends plain text chunks, so this just
 * decodes the response body as it arrives. EventSource is not used because it
 * only supports GET and the message has to be POSTed.
 */
export async function* streamChat(
  message: string,
  sessionId: string,
  signal?: AbortSignal,
): AsyncGenerator<string> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId }),
    signal,
  });

  if (!response.ok) {
    throw new Error(`Agent request failed (${response.status})`);
  }
  if (!response.body) {
    throw new Error("Agent returned an empty response");
  }

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) yield value;
    }
  } finally {
    reader.releaseLock();
  }
}
