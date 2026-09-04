"""Gemini owns tool choice and arguments. Python only executes safe evidence tools."""

from __future__ import annotations

import json
import re
from typing import Any

from src.agent.dispatch import observation_text, run_named_tool
from src.agent.model_provider import llm_enabled as _llm_enabled, make_chat_model, make_brain_model

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
    "research_alternative",
    "optimize_trip",
)
from src.tools_period import assess_period_pressure
from src.tools_recommend import find_similar_destinations, rank_alternatives
from src.tools_web import web_search

# Keep the chat UI responsive and response time low.
MAX_BRAIN_LOOPS = 8
MAX_CALLS_PER_STEP = 4

BRAIN_SYSTEM = """You are the autonomous AI travel-crowd advisor brain for SIH 2026 — a friendly, expert guide who NEVER waits for human intervention except for privacy/security concerns.

═══════════════════════════════════════════════════════
CORE PRINCIPLE: You run the full research pipeline autonomously. Every tool-calling decision is yours. Keep it snappy — gather enough facts, then stop and let the answer be composed.
═══════════════════════════════════════════════════════

STEP 1 — SLOT COLLECTION (conversational, one question at a time)
────────────────────────────────────────────────────────────────
You need exactly THREE things before researching:
  (a) Destination  — monument, city, fort, district, or region
  (b) Travel dates — start and end date (even approximate is fine: "first week of October")
  (c) Travel party — solo | family | friends

Rules:
- Call set_trip_context as soon as destination is clear (even if dates/party are still missing).
- If ANY of the three are genuinely missing from the conversation, call clarify_with_user with ONE warm, specific question covering the most important missing item.
- NEVER ask for something the user already mentioned. NEVER ask multiple questions at once.
- Once all three are known, proceed immediately to Step 2 — do NOT ask for confirmation.

STEP 2 — DATASET-FIRST RESEARCH
────────────────────────────────
Always check local data before hitting the web. This keeps responses fast and grounded.

  A. If destination is in our dataset:
     1. search_destinations  → verify the name
     2. get_destination_profile  → type, state, zone, ratings
     3. get_historical_footfall  → official ASI annual visitor history
     4. forecast_crowd  → persistence prediction for next period
     5. classify_crowd  → LOW / MODERATE / HIGH / VERY HIGH relative level

  B. Fill in dynamic features that the dataset cannot provide:
     - Weather conditions for the travel window → web_search("{destination} weather {month} {year}")
     - Upcoming festivals, fairs, events in that period → web_search("{destination} festivals events {month} {year}")
     - National/regional holidays overlapping the dates → already handled by assess_period_pressure
     - Official timings or closures → web_search if relevant

  C. If destination is NOT in our dataset at all:
     - Run web_search to build the place profile, discover type/season/popularity
     - Then continue with assess_period_pressure and alternatives

  D. Always call assess_period_pressure once web + local data is ready — it synthesises:
     • Overall crowd density (HIGH / MODERATE / LOW)
     • Most probable peak day with reason
     • Peak rush hours to avoid
     • Best low-crowd visiting hours
     Never skip this step when dates are known.

STEP 3 — CROWD DENSITY CONCLUSION + DUAL-TIER ALTERNATIVES
─────────────────────────────────────────────────────────────
After assess_period_pressure completes:
  - Determine if crowd is HIGH, MODERATE, or LOW for the travel window.
  - ALWAYS recommend alternatives regardless of crowd level (users appreciate options).
  - Call find_similar_destinations → rank_alternatives.
    rank_alternatives automatically tags each result as:
      🌟 Popular Alternative  (well-known, lower relative footfall than anchor)
      💎 Underrated / Hidden Gem  (lesser-known, discovery mode, peaceful ambience)
  - For the top 1–2 alternatives only, call research_alternative to verify their details.
  - STOP tool calls after that. Do not loop more than 6 times total.

STEP 4 — WHEN TO STOP
──────────────────────
Stop calling tools when you have:
  ✓ Trip context set (destination, dates, party)
  ✓ Local data checked (profile + history + forecast + crowd level) OR web fallback done
  ✓ Web dynamic features fetched (weather, events, holidays)
  ✓ assess_period_pressure result
  ✓ At least 2–3 ranked alternatives
Then return — the answer composer will write the final response from your evidence.

EFFICIENCY RULES
────────────────
- MAX 6 tool-call loops total. Make parallel calls (up to 4 per step) when possible.
- Do NOT call web_search more than 3 times per request.
- Do NOT call research_alternative for more than 2 candidates.
- Never invent visitor counts, occupancy %, or hourly headcounts.
- Never fabricate URLs or event names.
"""


def canonical_tool_name(name: str) -> str:
    raw = (name or "").strip()
    if raw.endswith("_tool"):
        raw = raw[: -len("_tool")]
    return raw


def fallback_tool_plan(state: dict) -> list[str]:
    """Minimal offline fallback. Live path uses run_llm_tool_loop — Gemini decides everything.
    This only fires when TOURISM_LLM_BRAIN=0 (test/offline mode).
    We do the absolute minimum: web search for context + period pressure if dates exist.
    """
    tools: list[str] = []
    if state.get("start_date"):
        if "web_search" not in (state.get("tool_trace") or []):
            tools.append("web_search")
        if "assess_period_pressure" not in (state.get("tool_trace") or []):
            tools.append("assess_period_pressure")
    if not tools and state.get("destination"):
        tools.append("web_search")
    seen: set[str] = set()
    return [t for t in tools if t in ALLOWED_TOOLS and not seen.add(t)]  # type: ignore[func-returns-value]


def _parse_llm_json(raw: str) -> dict | None:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


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
        "crowd_density": (state.get("period_pressure") or {}).get("crowd_density"),
        "peak_day": (state.get("period_pressure") or {}).get("peak_day"),
        "peak_hours": (state.get("period_pressure") or {}).get("peak_hours"),
        "recommended_hours": (state.get("period_pressure") or {}).get("recommended_hours"),
        "n_web_hits": len(state.get("web_raw_hits") or []),
        "n_candidates": len(state.get("candidate_alternatives") or []),
        "n_ranked": len(state.get("ranked_alternatives") or []),
        "n_plan_days": len(state.get("trip_plan") or []),
    }

    # 1. Dataset lookup (ASI + Kaggle)
    profile = get_destination_profile(dest)
    patch["profile"] = profile
    has_asi = bool(profile.get("has_asi_history"))
    patch["has_asi_history"] = has_asi

def run_llm_tool_loop(state: dict) -> dict:
    """Model selects tools+args; Python executes them into state. Cap enforced here."""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    # 3. Period pressure assessment (weekends, holidays, season)
    kaggle_meta = profile.get("kaggle") or {}
    pressure = assess_period_pressure(
        dest,
        start,
        end or start,
        crowd_level=crowd_info.get("level"),
        web_context=web_res,
        dest_state=kaggle_meta.get("state"),
        climate_zone=kaggle_meta.get("zone"),
    )
    patch["period_pressure"] = pressure

    # 4. Alternatives & underrated places
    try:
        cands = find_similar_destinations(dest, interests=interests, limit=8, include_wider=True)
        ranked = rank_alternatives(dest, candidates=cands, user_preferences={"interests": interests})
    except Exception:
        ranked = []
    patch["ranked_alternatives"] = ranked

    working = dict(state)
    # Distinguish a Gemini-directed call from offline/direct compatibility
    # calls without changing the model's authority over the live workflow.
    working["brain_source"] = "llm"
    working.setdefault("decisions", [])
    working.setdefault("tool_trace", [])
    llm = make_brain_model(max_tokens=1200).bind_tools(ALL_TOOLS)
    history: list = [
        SystemMessage(content=BRAIN_SYSTEM),
        HumanMessage(content=json.dumps(_state_snapshot(working))),
    ]
    if has_asi and forecast_info.get("found"):
        summary_lines.append(
            f"- Official ASI Annual Persistence Forecast: {forecast_info.get('predicted_visitors'):,.0f} visitors ({forecast_info.get('forecast_period')})"
        )
        summary_lines.append(
            f"- Relative Crowd Level: {crowd_info.get('level', 'MODERATE')} (historical quartile)"
        )
        if forecast_info.get("trend"):
            summary_lines.append(f"- Recent Trend: {forecast_info.get('trend')}")
    else:
        summary_lines.append("- Official ASI Catalog: Destination is not in ASI annual ticketed monument panel.")

    kaggle_meta = profile.get("kaggle") or {}
    if kaggle_meta:
        summary_lines.append(
            f"- Catalog Metadata: State={kaggle_meta.get('state')}, Rating={kaggle_meta.get('google_rating')}, Best Season={kaggle_meta.get('season')}"
        )

    if pressure.get("found"):
        summary_lines.append(
            f"- Travel Window Pressure: Footfall trend is {pressure.get('footfall_direction')} "
            f"({pressure.get('weekend_days')} weekend day(s), {pressure.get('weekday_days')} weekday(s), season={pressure.get('season')})"
        )
        for r in pressure.get("reasons") or []:
            summary_lines.append(f"  • {r}")

    web_hits = web_res.get("results") or []
    if web_hits:
        summary_lines.append("- Web Context Snippets:")
        for h in web_hits[:3]:
            summary_lines.append(f"  • {h.get('title')}: {(h.get('snippet') or '')[:140]}")

    if ranked:
        summary_lines.append("- Nearby Alternatives & Underrated Spots:")
        for r in ranked[:5]:
            dist = f"{r.get('distance_km')} km" if r.get("distance_km") else "nearby"
            mode = "Underrated / Hidden Gem" if r.get("recommendation_mode") == "discovery" else "Alternative Heritage Site"
            summary_lines.append(f"  • {r.get('name')} ({mode}, ~{dist}) — {'; '.join((r.get('why') or [])[:2])}")
    else:
        # Fallback alternatives if gazetteer had no coordinates
        summary_lines.append(f"- Nearby Alternatives: Explore local spots and cultural heritage around {dest}.")

    patch["research_summary"] = "\n".join(summary_lines)
    return intent, patch


def run_llm_tool_loop(state: dict) -> dict:
    """Single-pass LLM brain: gathers all local/web research in Python, then calls Gemini
    once to produce the complete, intelligent conversational response."""
    working = dict(state)
    user_req = working.get("user_request") or working.get("conversation") or ""

    # Check destination and gather research
    intent, research = gather_trip_research(working)
    working.update(research)

    if not research.get("has_destination"):
        # No destination named yet — ask conversationally
        working["awaiting_user"] = True
        working["missing_slots"] = ["destination"]
        if _llm_enabled():
            try:
                llm = _get_llm(temperature=0.4, max_tokens=250)
                prompt = CLARIFICATION_PROMPT.format(
                    user_request=user_req,
                    conversation=working.get("conversation") or user_req,
                )
                msg = llm.invoke(prompt)
                resp = _extract_text(msg)
                working["brain_response"] = resp
                working["pending_question"] = resp
                working["final_response"] = resp
                working["answer_source"] = "llm_brain"
                working["brain_source"] = "llm"
                return working
            except Exception:
                pass
        fallback_msg = (
            "Hey! I'd love to help you plan your trip. Which destination are you thinking of visiting? "
            "Also let me know your planned travel dates and whether you'll be traveling solo, with family, or with friends!"
        )
        working["brain_response"] = fallback_msg
        working["pending_question"] = fallback_msg
        working["final_response"] = fallback_msg
        working["answer_source"] = "llm_brain_clarification"
        working["brain_source"] = "fallback"
        return working

    # Destination is present — run the LLM brain to synthesize and write conversational answer
    if _llm_enabled():
        try:
            llm = _get_llm(temperature=0.3, max_tokens=3000)
            prompt = BRAIN_CONVERSATIONAL_PROMPT.format(
                research_summary=research.get("research_summary", "")
            )
            msg = llm.invoke(prompt)
            resp = _extract_text(msg)
            if resp:
                working["brain_response"] = resp
                working["final_response"] = resp
                working["answer_source"] = "llm_brain"
                working["brain_source"] = "llm"
                working["tool_trace"] = list(working.get("tool_trace") or []) + ["llm_brain"]
                return working
        except Exception as exc:
            working["decisions"] = list(working.get("decisions") or []) + [f"Brain LLM error: {exc}"]
            working["brain_source"] = "llm_error"
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
    return {"tools": [], "done": True, "source": "single_pass"}
