"""Gemini brain: autonomous travel intelligence agent.
Consolidated single-pass reasoning for fast response times and zero rate-limit errors."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from src.agent.intent import resolve_intent
from src.tools_local import (
    classify_crowd,
    forecast_crowd,
    get_destination_profile,
    get_historical_footfall,
    search_destinations,
)
from src.tools_period import assess_period_pressure
from src.tools_recommend import find_similar_destinations, rank_alternatives
from src.tools_web import web_search

BRAIN_CONVERSATIONAL_PROMPT = """You are a knowledgeable, friendly, and expert travel companion and crowd prediction assistant.
You talk naturally with the traveler — warm, professional, engaging, and conversational.
NEVER sound like a robotic template, and NEVER output section titles like "CONCLUSION", "CROWD FORECAST", "LIMITATIONS", or "HOW I REASONED". Just talk naturally using clean markdown formatting (bullet points, bold text).

HERE IS THE RESEARCH DATA GATHERED FOR THIS TRIP:
{research_summary}

YOUR TASK:
1. Warmly acknowledge their trip: Destination, travel window (if given), and traveling style (family, friends, solo).
2. Crowd Prediction & Reasoning:
   - State whether the crowd density during their visit duration will be HIGH, MODERATE, or LOW.
   - Ground your reasoning in the actual features:
     • Official ASI monument persistence demand and historical trend (if in the dataset)
     • Weather conditions during that period
     • Any festivals, events, or local celebrations happening
     • Gazetted / school holidays
     • Weekday vs weekend distribution
   - If the place is NOT in the official ASI dataset, clearly mention that official ticketed history is uncataloged, but give your crowd prediction based on web features, seasonal trends, and weekends/holidays.
3. Specific Peak Days & Best Calm Hours:
   - Identify which specific days in their window are likely to see the heaviest rush (e.g., Saturday/Sunday, festival days).
   - Recommend the best visiting hours (e.g., early morning 6:00 AM – 8:30 AM or late afternoon) when they won't face heavy crowds.
4. Nearby Alternative & Underrated Destinations:
   - Recommend 3 to 5 nearby places to explore.
   - Clearly label which ones are mainstream alternatives and which ones are **underrated / hidden gems**.
   - Mention approximate distance from the main destination and give a 1–2 sentence reason why each is worth visiting.
5. Friendly Closing:
   - End with a welcoming note inviting any questions or itinerary refinements.

Keep your response clean, engaging, informative, and around 200–350 words.
"""

CLARIFICATION_PROMPT = """You are a warm, helpful, and friendly travel advisor.
The traveler just sent a message, but they haven't named a destination yet, or their request is unclear.

Traveler's message: "{user_request}"
Conversation history: "{conversation}"

Reply conversationally to welcome them and ask:
1. Where they are planning to visit (city, monument, or state)
2. Their approximate travel dates or month
3. Who they are traveling with (solo, family, or friends)

Keep it short, friendly, and natural.
"""


def _llm_enabled() -> bool:
    if os.environ.get("TOURISM_LLM_BRAIN", "1") != "1":
        return False
    return bool((os.getenv("GOOGLE_API_KEY") or "").strip())


def _get_llm(temperature: float = 0.3, max_tokens: int = 1000):
    from langchain_google_genai import ChatGoogleGenerativeAI

    model = os.getenv("GOOGLE_MODEL", "gemini-3.5-flash")
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=temperature,
        max_output_tokens=max_tokens,
    )


def _extract_text(msg: Any) -> str:
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    parts.append(item["text"])
                elif "text" in item:
                    parts.append(str(item["text"]))
        return "\n".join(parts).strip()
    return str(content or "").strip()


def gather_trip_research(state: dict) -> tuple[dict, dict]:
    """Execute local dataset lookups and web searches in Python in ~0.5s."""
    req = state.get("user_request") or state.get("conversation") or ""
    intent = resolve_intent(req)
    dest = intent.get("destination") or state.get("destination")
    start = intent.get("start_date") or state.get("start_date")
    end = intent.get("end_date") or state.get("end_date")
    party = intent.get("party_type") or state.get("party_type") or "not specified"
    interests = intent.get("interests") or state.get("interests") or []

    if not dest:
        return intent, {"has_destination": False}

    patch: dict[str, Any] = {
        "destination": dest,
        "start_date": start,
        "end_date": end,
        "party_type": party,
        "interests": interests,
        "has_destination": True,
    }

    # 1. Dataset lookup (ASI + Kaggle)
    profile = get_destination_profile(dest)
    patch["profile"] = profile
    has_asi = bool(profile.get("has_asi_history"))
    patch["has_asi_history"] = has_asi

    forecast_info = {}
    crowd_info = {}
    if has_asi:
        asi_name = profile.get("asi_name") or dest
        hist = get_historical_footfall(asi_name)
        patch["historical_demand"] = {asi_name: hist}
        fc = forecast_crowd(asi_name)
        patch["forecasts"] = {asi_name: fc}
        forecast_info = fc
        if fc.get("found"):
            cl = classify_crowd(asi_name, fc.get("predicted_visitors"))
            patch["crowd_levels"] = {asi_name: cl}
            crowd_info = cl

    # 2. Web search for features (weather, festivals, events, holidays)
    search_q = f"{dest} tourism weather events festivals holidays"
    web_res = web_search(search_q, max_results=4)
    patch["web_context"] = web_res

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

    # Build clean research summary for the LLM
    summary_lines = [
        f"- Destination: {dest}",
        f"- Travel Window: {start or 'Dates not specified'} to {end or start or 'Dates not specified'}",
        f"- Party Type: {party}",
        f"- Interests: {', '.join(interests) if interests else 'General sightseeing'}",
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

    # Fallback if Gemini API is temporarily unavailable
    from src.agent.compose import compose_final_response

    fallback_resp = compose_final_response(working)
    working["brain_response"] = fallback_resp
    working["final_response"] = fallback_resp
    working["answer_source"] = "offline_template"
    working["brain_source"] = "fallback"
    return working


def decide_next_tools(state: dict) -> dict:
    return {"tools": [], "done": True, "source": "single_pass"}
