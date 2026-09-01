"""OpenAI brain owns tool choice and arguments. Python only executes whitelist tools."""

from __future__ import annotations

import json
import os
import re

from src.agent.dispatch import observation_text, run_named_tool

ALLOWED_TOOLS = (
    "clarify_with_user",
    "set_trip_context",
    "forecast_crowd",
    "classify_crowd",
    "get_historical_footfall",
    "get_destination_profile",
    "search_destinations",
    "web_search",
    "extract_nearby_places",
    "assess_period_pressure",
    "find_similar_destinations",
    "rank_alternatives",
    "optimize_trip",
)

MAX_BRAIN_LOOPS = 10
MAX_CALLS_PER_STEP = 4

BRAIN_SYSTEM = """You are a travel-research advisor in a live conversation. You decide every next step.
Python only executes tools (ASI + Kaggle catalogs, web search, ranking, trip sketch). You decide whether to ask, research, predict, recommend, or stop.

Conversation first:
- Read the full transcript. Destination is required before any crowd claim.
- If there is NO place yet, call clarify_with_user only. Ask where they want to go, and also dates and family/friends/solo if unknown.
- If a place IS named but dates or party are missing: still research that place. Ask for dates/party in the spoken reply (clarify_with_user after research, or leave it for the chat turn). Do not skip ASI/Kaggle lookup just because dates are missing.
- Do not re-ask what they already answered.

Research (only after you have a place; dates/party help but you may research the place and still ask for missing dates/party if needed):
1) Catalog first: search_destinations + get_destination_profile on the place, city, state, aliases.
   If found: get_historical_footfall, forecast_crowd, classify_crowd. Use whatever features the tools return (season in profile, ratings, etc.).
2) Gaps: if not in catalog, or weather/events/holidays/festivals for the window are missing, YOU write web_search queries (official name + aliases). Then extract_nearby_places if useful. assess_period_pressure for weekdays/weekends/season/holidays vs the annual figure — not a new daily count.
3) YOU decide if research is enough. If not, call more tools. If you still lack dest/dates/party, ask the user instead of guessing.

Prediction for THEIR window:
- Combine annual history (if any) with window/season/weekends/events/weather. Say if footfall pressure looks higher or not. Never invent a visitor total. found=false → no ASI number.

If pressure looks higher from that full research (not only annual HIGH):
- find_similar / extract_nearby / rank_alternatives. Same research on those names when possible.

Then stop so the reply can be written. Or clarify_with_user if you need the traveler to talk.

Never invent visitor counts. Distant same-state cities are not nearby.
"""


def canonical_tool_name(name: str) -> str:
    raw = (name or "").strip()
    if raw.endswith("_tool"):
        raw = raw[: -len("_tool")]
    return raw


def fallback_tool_plan(state: dict) -> list[str]:
    intent = state.get("intent") or {}
    dest = state.get("destination")
    start = state.get("start_date")
    tools: list[str] = []
    if dest and (intent.get("needs_prediction") or not intent.get("needs_trip_plan")):
        if intent.get("needs_prediction") or dest:
            if intent.get("needs_prediction"):
                tools += ["forecast_crowd", "classify_crowd"]
    if intent.get("needs_trip_plan") or intent.get("needs_prediction"):
        if dest and "forecast_crowd" not in tools and intent.get("needs_prediction"):
            tools += ["forecast_crowd", "classify_crowd"]
    if start or intent.get("needs_current_context"):
        tools.append("web_search")
        if start:
            tools.append("assess_period_pressure")
    from src.agent.planner import KNOWN_FORECAST_SITES

    if dest and dest not in KNOWN_FORECAST_SITES:
        who_only = (
            intent.get("needs_current_context")
            and not intent.get("needs_prediction")
            and not intent.get("needs_trip_plan")
            and not intent.get("needs_alternatives")
        )
        if not who_only:
            tools.append("web_search")
            tools += ["find_similar_destinations", "rank_alternatives"]
    if intent.get("needs_alternatives") or intent.get("needs_trip_plan"):
        tools += ["find_similar_destinations", "rank_alternatives"]
    if intent.get("needs_trip_plan"):
        tools.append("optimize_trip")
    seen = set()
    out = []
    for t in tools:
        if t in ALLOWED_TOOLS and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _parse_llm_json(raw: str) -> dict | None:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _llm_enabled() -> bool:
    if os.environ.get("TOURISM_LLM_BRAIN", "1") != "1":
        return False
    return bool((os.getenv("OPENAI_API_KEY") or "").strip())


def _state_snapshot(state: dict) -> dict:
    return {
        "user_request": (state.get("user_request") or "")[:4000],
        "conversation": (state.get("conversation") or "")[:6000],
        "awaiting_user": bool(state.get("awaiting_user")),
        "destination": state.get("destination"),
        "place_names": state.get("place_names"),
        "dest_state": state.get("dest_state"),
        "climate_zone": state.get("climate_zone"),
        "start_date": state.get("start_date"),
        "end_date": state.get("end_date"),
        "must_visit": state.get("must_visit"),
        "party_type": state.get("party_type"),
        "interests": state.get("interests"),
        "intent": state.get("intent"),
        "tool_trace": (state.get("tool_trace") or [])[-24:],
        "forecasts": {
            k: {
                "found": (v or {}).get("found"),
                "predicted_visitors": (v or {}).get("predicted_visitors"),
            }
            for k, v in (state.get("forecasts") or {}).items()
        },
        "crowd_levels": {k: (v or {}).get("level") for k, v in (state.get("crowd_levels") or {}).items()},
        "period_pressure": (state.get("period_pressure") or {}).get("footfall_direction"),
        "n_web_hits": len(state.get("web_raw_hits") or []),
        "n_candidates": len(state.get("candidate_alternatives") or []),
        "n_ranked": len(state.get("ranked_alternatives") or []),
        "n_plan_days": len(state.get("trip_plan") or []),
    }


def run_llm_tool_loop(state: dict) -> dict:
    """Model selects tools+args; Python executes them into state. Cap enforced here."""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
        from langchain_openai import ChatOpenAI

        from src.agent.registry import ALL_TOOLS
    except Exception:
        return {"brain_source": "unavailable", "brain_ran_tools": False}

    working = dict(state)
    working.setdefault("decisions", [])
    working.setdefault("tool_trace", [])
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0, max_tokens=900).bind_tools(ALL_TOOLS)
    history: list = [
        SystemMessage(content=BRAIN_SYSTEM),
        HumanMessage(content=json.dumps(_state_snapshot(working))),
    ]
    ran = False
    for step in range(MAX_BRAIN_LOOPS):
        try:
            msg = llm.invoke(history)
        except Exception as exc:
            working["decisions"] = list(working.get("decisions") or []) + [f"Brain LLM error: {exc}"]
            break
        history.append(msg)
        calls = list(getattr(msg, "tool_calls", None) or [])
        if not calls:
            working["brain_steps"] = int(state.get("brain_steps") or 0) + step + 1
            working["brain_source"] = "llm"
            working["brain_ran_tools"] = ran
            working["tool_trace"] = list(working.get("tool_trace") or []) + ["llm_brain"]
            return working
        for call in calls[:MAX_CALLS_PER_STEP]:
            name = canonical_tool_name(call.get("name") or "")
            args = call.get("args") or {}
            if name not in ALLOWED_TOOLS:
                obs = {"error": f"tool not allowed: {name}"}
            else:
                payload, patch = run_named_tool(name, args, working)
                working.update(patch)
                ran = True
                obs = payload
                if name == "clarify_with_user" or working.get("awaiting_user"):
                    working["brain_steps"] = int(state.get("brain_steps") or 0) + step + 1
                    working["brain_source"] = "llm"
                    working["brain_ran_tools"] = True
                    working["tool_trace"] = list(working.get("tool_trace") or []) + ["llm_brain"]
                    return working
            history.append(
                ToolMessage(
                    content=observation_text(name, obs),
                    tool_call_id=call.get("id") or name,
                    name=call.get("name") or name,
                )
            )
        history.append(HumanMessage(content="State now: " + json.dumps(_state_snapshot(working))[:6000]))
    working["brain_steps"] = int(state.get("brain_steps") or 0) + MAX_BRAIN_LOOPS
    working["brain_source"] = "llm_cap"
    working["brain_ran_tools"] = ran
    working["tool_trace"] = list(working.get("tool_trace") or []) + ["llm_brain"]
    working["decisions"] = list(working.get("decisions") or []) + ["Stopped at brain tool-call cap."]
    return working


def decide_next_tools(state: dict) -> dict:
    """Offline/fallback whitelist only. Live path uses run_llm_tool_loop."""
    steps = int(state.get("brain_steps") or 0)
    if steps >= MAX_BRAIN_LOOPS:
        return {"tools": [], "done": True, "source": "cap"}
    tools = fallback_tool_plan(state)
    if steps > 0:
        period = state.get("period_pressure") or {}
        ranked = state.get("ranked_alternatives") or []
        extra = []
        if state.get("start_date") and not period.get("found"):
            extra += ["web_search", "assess_period_pressure"]
        from src.agent.research import nearby_from_research

        if nearby_from_research(state).get("suggest_nearby") and not ranked:
            extra += ["find_similar_destinations", "rank_alternatives"]
        tools = extra
    return {"tools": tools, "done": not tools, "source": "fallback"}
