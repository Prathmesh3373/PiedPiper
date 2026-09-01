"""LangGraph Server / Agent Chat UI adapter.

Does not reimplement tourism logic. Every turn calls run_agent()
(the same graph as python demo.py).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from src.agent.graph import run_agent

GRAPH_ID = "tourism"


def _message_text(message: BaseMessage | dict) -> str:
    if isinstance(message, dict):
        content = message.get("content") or ""
        if isinstance(content, str):
            return content
        return str(content)
    content = getattr(message, "content", "") or ""
    if not isinstance(content, str):
        content = str(content)
    return content


def _message_role(message: BaseMessage | dict) -> str:
    if isinstance(message, dict):
        role = (message.get("type") or message.get("role") or "").lower()
        if role in {"human", "user"}:
            return "traveler"
        if role in {"ai", "assistant", "ai_message"}:
            return "advisor"
        return ""
    if isinstance(message, HumanMessage):
        return "traveler"
    if isinstance(message, AIMessage):
        return "advisor"
    msg_type = (getattr(message, "type", "") or "").lower()
    if msg_type in {"human", "user"}:
        return "traveler"
    if msg_type in {"ai", "assistant"}:
        return "advisor"
    return ""


def conversation_transcript(messages: list) -> str:
    """Full back-and-forth for the brain (traveler + advisor)."""
    lines: list[str] = []
    for msg in (messages or [])[-16:]:
        text = _message_text(message=msg).strip()
        if not text:
            continue
        role = _message_role(message=msg)
        if role == "traveler":
            lines.append(f"Traveler: {text}")
        elif role == "advisor":
            lines.append(f"Advisor: {text}")
    return "\n".join(lines)


def conversation_to_user_request(messages: list) -> str:
    """Latest traveler line, with earlier turns as conversation context."""
    transcript = conversation_transcript(messages)
    humans: list[str] = []
    for msg in messages or []:
        if _message_role(message=msg) == "traveler":
            text = _message_text(message=msg).strip()
            if text:
                humans.append(text)
    latest = humans[-1] if humans else ""
    if not transcript:
        return latest
    return f"{transcript}\n\nLatest traveler message:\n{latest}"


_CHAT_SECTION_ORDER = (
    "CONCLUSION",
    "CROWD FORECAST",
    "TRAVEL WINDOW",
    "CURRENT CONTEXT",
    "ALTERNATIVES",
    "TRIP PLAN",
    "LIMITATIONS",
    "HOW I REASONED (decision log)",
)

_CHAT_TITLES = {
    "CONCLUSION": "According to our analysis",
    "CROWD FORECAST": "Crowd forecast",
    "TRAVEL WINDOW": "Travel window",
    "CURRENT CONTEXT": "Current context",
    "ALTERNATIVES": "Nearby / underrated places",
    "TRIP PLAN": "Schedule",
    "LIMITATIONS": "Limitations",
    "HOW I REASONED (decision log)": "How I reasoned",
}


def _split_agent_sections(raw: str) -> dict[str, str]:
    found: list[tuple[int, str]] = []
    for name in _CHAT_TITLES:
        idx = raw.find(name)
        if idx >= 0:
            found.append((idx, name))
    found.sort()
    sections: dict[str, str] = {}
    for i, (start, name) in enumerate(found):
        end = found[i + 1][0] if i + 1 < len(found) else len(raw)
        sections[name] = raw[start + len(name) : end].strip()
    return sections


def _trim_alt_debug(body: str) -> str:
    """Drop planner boolean dumps; keep names, modes, demand, and why."""
    keep: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("relevance=") or stripped.startswith("geo_tier="):
            continue
        keep.append(line)
    return "\n".join(keep).strip()


def format_chat_answer(raw: str) -> str:
    """Chat bubble. LLM answers are shown as written; template answers stay sectioned."""
    text = (raw or "").strip()
    if not text:
        return "(no response from tourism agent)"
    sections = _split_agent_sections(text)
    if not sections:
        return text
    parts: list[str] = []
    for key in _CHAT_SECTION_ORDER:
        body = sections.get(key)
        if body is None:
            continue
        if key == "ALTERNATIVES":
            body = _trim_alt_debug(body)
        title = _CHAT_TITLES[key]
        parts.append(f"## {title}\n\n{body}")
    return "\n\n".join(parts).strip()


def tourism_turn(state: MessagesState) -> dict:
    """One chat turn → tourism agent. Missing slots become a conversational ask."""
    messages = state.get("messages") or []
    transcript = conversation_transcript(messages)
    request = conversation_to_user_request(messages)
    if not request.strip():
        request = "(The traveler has not said anything yet.)"
        transcript = request
    try:
        result = run_agent(request, conversation=transcript)
        content = format_chat_answer(result.get("final_response") or "")
    except Exception as exc:
        content = (
            "I couldn't finish that just now. Nothing was invented as a crowd number.\n\n"
            f"{type(exc).__name__}: {exc}"
        )
    return {"messages": [AIMessage(content=content)]}


def build_chat_graph():
    g = StateGraph(MessagesState)
    g.add_node("tourism_agent", tourism_turn)
    g.add_edge(START, "tourism_agent")
    g.add_edge("tourism_agent", END)
    return g.compile()


graph = build_chat_graph()
