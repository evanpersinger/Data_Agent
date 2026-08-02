# server.py
# HTTP wrapper around the data agent so a browser can talk to it.
# The CLI in data_agent.py still works, this is just a second entry point.

from collections.abc import AsyncIterator
from pathlib import Path

from agents import Runner, SQLiteSession
from agents.stream_events import RawResponsesStreamEvent
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel, Field

# Importing the agent runs data_agent.py's module level setup (load_dotenv, tool
# registration) but not main(), which is guarded by __name__ == "__main__".
from data_agent import agent

# The CLI keeps history in memory, which is fine for one long lived REPL process.
# Over HTTP each request is separate, so history has to live on disk or the agent
# forgets the previous message.
SESSIONS_DB = Path(__file__).parent.parent / "sessions.db"

app = FastAPI(title="Data Agent")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str = Field(default="web", min_length=1)


async def stream_reply(message: str, session_id: str) -> AsyncIterator[str]:
    """Yield the agent's response text as it is generated."""
    session = SQLiteSession(session_id=session_id, db_path=SESSIONS_DB)
    result = Runner.run_streamed(agent, message, session=session)

    try:
        async for event in result.stream_events():
            if isinstance(event, RawResponsesStreamEvent) and isinstance(
                event.data, ResponseTextDeltaEvent
            ):
                yield event.data.delta
    except Exception as exc:
        # Broad on purpose: this is the outermost handler for the request. A tool
        # or API failure here would otherwise just drop the connection and leave
        # the browser showing a half finished message with no explanation.
        yield f"\n\n[agent error: {exc}]"


@app.post("/api/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_reply(request.message, request.session_id),
        media_type="text/plain; charset=utf-8",
    )
