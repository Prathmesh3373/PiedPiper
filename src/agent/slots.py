"""Optional LLM slot filling. Does not emit visitor counts or rank scores."""

from __future__ import annotations

import json
import os
import re

from src.agent.model_provider import llm_enabled, make_chat_model

SLOT_SYSTEM = """You extract tourism trip slots from one user message.
Return JSON only with keys:
destination (string or null): the primary place (city, fort, monument, district).
Never use a question fragment such as "what should I visit instead".
start_date, end_date: YYYY-MM-DD or null.
party_type: family, friends, solo, or null.
interests: list of short tags (forts, temples, nature, heritage, beaches, ...).
must_visit: named sites the user said they must see.
needs_prediction, needs_alternatives, needs_current_context, needs_trip_plan: booleans.
Do not invent visitor numbers, occupancy, or hourly crowds.
"""


def llm_extract_slots(text: str) -> dict | None:
    if os.environ.get("TOURISM_LLM_SLOTS", "1") != "1":
        return None
    if not llm_enabled():
        return None
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
    except Exception:
        return None
    llm = make_chat_model(temperature=0, max_tokens=400)
    try:
        msg = llm.invoke(
            [
                SystemMessage(content=SLOT_SYSTEM),
                HumanMessage(content=text[:4000]),
            ]
        )
        raw = (msg.content or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None
