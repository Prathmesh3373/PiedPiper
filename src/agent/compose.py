"""Traveler-facing answer. Live path: Gemini writes from tool facts. Offline: template."""

from __future__ import annotations

import json
import os

from src.agent.model_provider import llm_enabled, make_chat_model
from src.agent.reasoning import build_decision_reasoning
from src.agent.research import nearby_from_research
from src.agent.state import TourismState

ANSWER_SYSTEM = """You are an expert, warm travel-crowd advisor for SIH 2026 — chatting naturally with a traveler.
Write like a knowledgeable friend, not a system log. Clean Markdown, structured but never robotic.

RESEARCH_FACTS below is your ONLY ground truth. Do not invent numbers, URLs, or event names not present in RESEARCH_FACTS.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IF awaiting_user = true  (missing slots)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ask ONLY the single most important missing piece (destination, dates, or travel party).
Keep it warm and conversational — 1–2 sentences max.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IF research ran  (forecasts or period_pressure present)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Structure your answer with these sections, in order:

**1. Trip Overview**
Acknowledge destination, travel dates, and party type naturally in 1–2 sentences.
Example: "Planning a family trip to Hampi from Oct 10–13 — great choice!"

**2. Crowd Density Verdict**
Lead with a clear verdict badge: 🔴 HIGH | 🟡 MODERATE | 🟢 LOW
Then give crisp reasoning (3–5 bullet points max) covering:
- Weather conditions during that window (from web facts if available)
- Active festivals, fairs, or events overlapping the dates (from web facts only — never invent)
- Weekend / public holiday overlap in the travel window
- Season classification (peak tourist season vs off-peak vs monsoon)
- Historical footfall trend (increasing / stable / decreasing) if ASI data was found

**3. Timing Intelligence**
📅 **Most Probable Peak Day:** [Day, Date] — [1-line reason]
⏰ **Peak Rush Hours:** [hours to avoid] — [1-line why]
💡 **Best Time to Visit:** [optimal window] — [1-line why]

**4. Smart Alternatives** *(always include, even if crowd is low)*

🌟 **Popular Alternatives** (well-known, manageable crowds)
For each (2–3 places):
- **[Name]** — [distance if available] | Crowd: [level or "discovery"]
  Peak hours: [hours] | Best visit: [hours]
  [1-line why it's a good swap or complement]

💎 **Hidden Gems** (underrated, off-beat, peaceful)
For each (2–3 places):
- **[Name]** — [distance if available] | Why it's special: [1-line]
  Best visit: [hours]

> 🗺️ **Tip:** Click any recommended destination in the dashboard to open an interactive side-map showing pinned locations, distances, and suggested routes!

**5. Smart Tips for Your Trip**
Give 2–3 party-specific tips:
- Family → kid-friendly pacing, shade/rest spots, entry ticket info
- Solo → photography golden hours, quiet corners, self-guided trails
- Friends → group photo spots, nearby food/chai stops, adventure add-ons

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Never invent visitor counts, occupancy %, or hourly headcounts.
- Only mention an event/festival if it appears in the web facts section of RESEARCH_FACTS.
- Crowd level comes from crowd_density in period_pressure (HIGH/MODERATE/LOW), not from your own guess.
- Peak day and hours come from peak_day, peak_hours, recommended_hours in period_pressure.
- For alternatives: use the ranked_alternatives list. Popular = is_underrated: false. Hidden gem = is_underrated: true.
- If no ASI data exists, say so honestly; still give timing and alternatives from web/heuristic evidence.
- Keep total response under ~600 words. Structured, scannable, warm.
"""


def _fmt_visitors(value) -> str:
    if value is None:
        return "not available"
    try:
        return f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return str(value)


def _brief_conclusion(state: TourismState) -> list[str]:
    dest = state.get("destination") or "this trip"
    party = state.get("party_type")
    start, end = state.get("start_date"), state.get("end_date")
    forecasts = state.get("forecasts") or {}
    crowds = state.get("crowd_levels") or {}
    historical = state.get("historical_demand") or {}
    pressure = state.get("period_pressure") or {}
    ranked = state.get("ranked_alternatives") or []
    plan = state.get("trip_plan") or []
    found = [v for v in forecasts.values() if v.get("found")]
    lines = ["CONCLUSION"]
    who = f" ({party})" if party else ""
    if start:
        lines.append(
            f"You asked about {dest}{who} from {start} to {end or start}."
        )
    else:
        lines.append(f"You asked about {dest}{who}.")

    if found:
        fc = found[0]
        name = fc.get("destination") or dest
        crowd = crowds.get(name) or crowds.get(fc.get("destination") or "") or {}
        hist = historical.get(name) or historical.get(dest) or {}
        direction = hist.get("annual_direction") or fc.get("trend") or "unknown"
        lines.append(
            f"According to our analysis of official ASI annual history, {name} "
            f"has persistence demand {_fmt_visitors(fc.get('predicted_visitors'))} "
            f"visitors ({fc.get('forecast_period')}). Relative level: "
            f"{crowd.get('level') or 'n/a'} (not occupancy %)."
        )
        if direction and direction != "unknown":
            lines.append(
                f"The recent annual series looks {direction} — that is year-to-year ticketed "
                "footfall, not a daily prediction."
            )
        if pressure.get("found"):
            lines.append(
                f"For this travel window, pressure vs that annual baseline looks "
                f"{pressure.get('footfall_direction')} "
                f"(season={pressure.get('season')}; "
                f"{pressure.get('weekend_days')} weekend / {pressure.get('weekday_days')} weekday). "
                "This does not replace the ASI number."
            )
        research = nearby_from_research(state)
        if research.get("suggest_nearby") and ranked:
            lines.append(
                "Because the research step (window, weather/events if found, history if any) "
                "points to extra pressure, nearby/underrated places are listed below: "
                + "; ".join(research.get("reasons") or [])
                + "."
            )
        elif plan:
            lines.append("A sketch schedule from your dates is below.")
    else:
        lines.append(
            "According to our analysis, this name is not in the ASI monument visitor tables, "
            "so we do not invent a visitor total."
        )
        if pressure.get("found"):
            lines.append(
                f"Travel-window research (season/weekends/events) suggests "
                f"{pressure.get('footfall_direction')} pressure vs a typical year — heuristic, not a count."
            )
        if ranked:
            lines.append(
                "Nearby places below are research recommendations (catalog + web), not occupancy ranks."
            )
        if plan:
            lines.append("You can follow this sketch schedule; keep days flexible where we lack nearby orbits.")
    lines.append("")
    return lines


def compose_final_response(state: TourismState) -> str:
    """Conversational fallback when LLM API is unavailable."""
    dest = state.get("destination") or "your destination"
    party = f" with your {state['party_type']}" if state.get("party_type") else ""
    dates = f" from {state['start_date']} to {state.get('end_date') or state['start_date']}" if state.get("start_date") else ""
    forecasts = state.get("forecasts") or {}
    crowds = state.get("crowd_levels") or {}
    pressure = state.get("period_pressure") or {}
    ranked = state.get("ranked_alternatives") or []

    parts = [f"I've analyzed the travel and crowd conditions for **{dest}**{party}{dates}."]
    found = [v for v in forecasts.values() if v.get("found")]
    if found:
        fc = found[0]
        c = crowds.get(fc.get("destination") or dest) or {}
        parts.append(
            f"**Crowd Forecast**: Expected annual demand is approximately {fc.get('predicted_visitors'):,.0f} visitors ({fc.get('forecast_period')}), with a relative crowd density of **{c.get('level', 'MODERATE')}**."
        )
    else:
        parts.append(
            f"**Crowd Forecast**: {dest} is outside the official ASI ticketed monument catalog, so historical visitor counts are uncataloged. Based on typical seasonal patterns, expect steady tourist flow."
        )
    if pressure.get("found"):
        parts.append(
            f"**Travel Window**: Footfall pressure is **{pressure.get('footfall_direction', 'normal')}** ({pressure.get('weekend_days', 0)} weekend day(s), season: {pressure.get('season')}). Weekends and midday hours (11 AM – 4 PM) will be the most crowded."
        )
        lines.append(f"Season (heuristic): {pressure.get('season')}")
        if pressure.get("crowd_density"):
            lines.append(f"Overall Predicted Crowd Density: {pressure.get('crowd_density')}")
        if pressure.get("peak_day"):
            pd = pressure.get("peak_day") or {}
            lines.append(f"Most Probable Peak Day: {pd.get('day')} ({pd.get('date')}) — {pd.get('reason')}")
        if pressure.get("peak_hours"):
            lines.append(f"Peak Rush Hours: {pressure.get('peak_hours')}")
        if pressure.get("recommended_hours"):
            lines.append(f"Recommended Best Visiting Hours: {pressure.get('recommended_hours')}")
        if pressure.get("holiday_dates"):
            lines.append("Holidays in window: " + ", ".join(pressure["holiday_dates"]))
        lines.append(
            f"Footfall vs annual baseline: {pressure.get('footfall_direction')} "
            f"| layer: {pressure.get('layer')} (does not change the ASI annual number)"
        )
        for reason in pressure.get("reasons") or []:
            lines.append(f"- {reason}")
        lines.append(pressure.get("note") or "")
        lines.append("")

    lines += ["CURRENT CONTEXT"]
    _append_web(lines, web)
    lines.append("")

    if ranked or dests or intent.get("needs_alternatives"):
        lines += ["ALTERNATIVES"]
        if not found_any and (ranked or intent.get("needs_alternatives")):
            lines.append(
                "These are discovery recommendations because validated crowd history is unavailable."
            )
        if intent.get("insufficient_candidates"):
            lines.append(intent.get("candidate_note") or "Fewer than requested practical nearby sites.")
        if not ranked:
            lines.append(
                "No practical nearby candidates passed geography + relevance filters; not padding with distant sites."
            )
        for r in ranked[:8]:
            mode = r.get("recommendation_mode") or "discovery"
            lines.append(f"- {r.get('name')} [mode={mode}] source={r.get('source')}")
            dist = r.get("distance_km")
            lines.append(
                f"  relevance={r.get('relevant')} practical={r.get('practical')} "
                f"crowd_data_available={r.get('crowd_data_available')} "
                f"geography_verified={r.get('geography_verified')}"
            )
            lines.append(
                f"  geo_tier={r.get('geo_tier')} | distance_km={dist if dist is not None else 'null'}"
            )
            if mode == "crowd_backed" and r.get("predicted_visitors") is not None:
                lines.append(
                    f"  Expected demand (annual persistence / MODEL OUTPUT): {_fmt_visitors(r.get('predicted_visitors'))}"
                )
            else:
                lines.append("  Expected demand: not available (discovery; no invented crowd number)")
            why_bits = r.get("why") or []
            if why_bits:
                lines.append("  Why this place: " + "; ".join(why_bits[:5]))
        lines.append("")

    if plan:
        lines += ["TRIP PLAN"]
        lines.append("Anchors = requested sites. Orbits = nearby ranked alternatives, not distant same-state fillers.")
        for day in plan:
            lines.append(
                f"- {day.get('date')}: {day.get('destination')} ({day.get('type')}) — {day.get('reason')}"
            )
        lines.append("")

    lines += [
        "LIMITATIONS",
        "Crowd prediction is annual ASI persistence only, and only where a monument series exists.",
        "Travel-window up/down is heuristic + web (weekends, holidays, season, events) — not a new daily count.",
        "Web snippets are not treated as visitor counts. Evidence tiers: official / credible_secondary / general_web.",
        "Same-state is not treated as nearby; distant Maharashtra/Karnataka sites are excluded from ordinary orbits.",
    ]
    if not lookup_has_coords_note(ranked):
        lines.append("Some distance_km values are missing because monument GPS was not fabricated.")
    lines.append("")
    lines += ["HOW I REASONED (decision log)"]
    lines.append("Each step below is why the agent chose a tool, a number, a place, or a refusal.")
    lines.append("")
    lines.extend(build_decision_reasoning(state))
    return "\n".join(lines).strip()


def _answer_facts(state: TourismState) -> dict:
    web = state.get("web_context") or {}

    def _hits(key: str) -> list[dict]:
        out = []
        for hit in (web.get(key) or [])[:4]:
            out.append(
                {
                    "title": hit.get("title"),
                    "url": hit.get("url"),
                    "snippet": (hit.get("snippet") or "")[:180],
                    "evidence_tier": hit.get("evidence_tier"),
                }
            )
        return out

    ranked = []
    for r in (state.get("ranked_alternatives") or [])[:8]:
        ranked.append(
            {
                "name": r.get("name"),
                "distance_km": r.get("distance_km"),
                "city": r.get("city"),
                "state": r.get("state"),
                # Coordinates for map rendering in the side-tab
                "lat": r.get("lat"),
                "lng": r.get("lng"),
                # Dual-tier classification
                "is_underrated": bool(r.get("is_underrated")),
                "category": r.get("category") or ("underrated" if r.get("is_underrated") else "popular"),
                # Timing
                "peak_hours": r.get("peak_hours"),
                "recommended_hours": r.get("recommended_hours"),
                # Evidence
                "why": (r.get("why") or [])[:4],
                "recommendation_mode": r.get("recommendation_mode"),
                "predicted_visitors": r.get("predicted_visitors"),
                "crowd_data_available": r.get("crowd_data_available"),
                "crowd_assessment": r.get("crowd_assessment"),
                "final_score": r.get("final_score"),
            }
        )
    hist = {}
    for k, v in (state.get("historical_demand") or {}).items():
        hist[k] = {
            "found": (v or {}).get("found"),
            "annual_direction": (v or {}).get("annual_direction"),
            "n_years": len((v or {}).get("observations") or []),
        }
    forecasts = {}
    for k, v in (state.get("forecasts") or {}).items():
        forecasts[k] = {
            "found": (v or {}).get("found"),
            "destination": (v or {}).get("destination"),
            "predicted_visitors": (v or {}).get("predicted_visitors"),
            "forecast_period": (v or {}).get("forecast_period"),
            "forecast_method": (v or {}).get("forecast_method"),
            "trend": (v or {}).get("trend"),
            "layer": (v or {}).get("layer"),
        }
    return {
        "user_request": (state.get("user_request") or "")[:2000],
        "conversation": (state.get("conversation") or "")[:4000],
        "awaiting_user": bool(state.get("awaiting_user")),
        "pending_question": state.get("pending_question"),
        "missing_slots": state.get("missing_slots") or [],
        "destination": state.get("destination"),
        "place_names": state.get("place_names"),
        "dest_state": state.get("dest_state"),
        "party_type": state.get("party_type"),
        "start_date": state.get("start_date"),
        "end_date": state.get("end_date"),
        "must_visit": state.get("must_visit"),
        "interests": state.get("interests"),
        "forecasts": forecasts,
        "crowd_levels": {
            k: {"level": (v or {}).get("level"), "found": (v or {}).get("found")}
            for k, v in (state.get("crowd_levels") or {}).items()
        },
        "historical": hist,
        "period_pressure": {
            "found": (state.get("period_pressure") or {}).get("found"),
            "season": (state.get("period_pressure") or {}).get("season"),
            "weekend_days": (state.get("period_pressure") or {}).get("weekend_days"),
            "weekday_days": (state.get("period_pressure") or {}).get("weekday_days"),
            "holiday_dates": (state.get("period_pressure") or {}).get("holiday_dates"),
            "footfall_direction": (state.get("period_pressure") or {}).get("footfall_direction"),
            "crowd_density": (state.get("period_pressure") or {}).get("crowd_density"),
            "peak_day": (state.get("period_pressure") or {}).get("peak_day"),
            "peak_hours": (state.get("period_pressure") or {}).get("peak_hours"),
            "recommended_hours": (state.get("period_pressure") or {}).get("recommended_hours"),
            "reasons": (state.get("period_pressure") or {}).get("reasons"),
        },
        "web": {
            "weather": _hits("weather"),
            "events": _hits("events"),
            "holidays": _hits("holidays"),
            "closures": _hits("closures"),
            "advisories": _hits("advisories"),
        },
        "nearby_from_research": nearby_from_research(state),
        "ranked_alternatives": ranked,
        "popular_alternatives": [r for r in ranked if not r.get("is_underrated")],
        "underrated_alternatives": [r for r in ranked if r.get("is_underrated")],
        "alternative_research": {
            name: {
                "used_web_fallback": (facts or {}).get("used_web_fallback"),
                "forecast_found": ((facts or {}).get("forecast") or {}).get("found"),
                "window_pressure": ((facts or {}).get("period_pressure") or {}).get("footfall_direction"),
            }
            for name, facts in (state.get("alternative_research") or {}).items()
        },
        "trip_plan": state.get("trip_plan") or [],
    }


def llm_write_answer(state: TourismState) -> str | None:
    """Ask Gemini to write the final conversational answer from collected evidence.

    Retries once on transient server errors (503 / 429 with retry-after).
    Returns None only when the error is permanent (bad key, quota exhausted for the day)
    so compose_answer() can fall back to the deterministic template.
    """
    import time as _time

    if os.environ.get("TOURISM_LLM_BRAIN", "1") != "1":
        return None
    if not llm_enabled():
        return None
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
    except Exception:
        return None

    facts_json = json.dumps(_answer_facts(state), default=str)[:14000]
    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=facts_json),
    ]

    last_exc: Exception | None = None
    for attempt in range(3):           # up to 3 attempts
        try:
            llm = make_chat_model(temperature=0.45, max_tokens=1400)
            msg = llm.invoke(messages)
            text = (msg.content or "").strip()
            if text:
                return text
            # Empty string from Gemini — treat as transient, retry once
            if attempt < 2:
                _time.sleep(2)
                continue
            return None
        except Exception as exc:
            last_exc = exc
            err_str = str(exc)
            # 503 UNAVAILABLE or 429 with a retry-after → wait and retry
            if ("503" in err_str or "UNAVAILABLE" in err_str):
                wait = 5 * (attempt + 1)
                _time.sleep(wait)
                continue
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                # Per-minute rate limit — short wait then try again once
                if attempt == 0:
                    _time.sleep(15)
                    continue
                # Daily quota exhausted — no point retrying
                return None
            # Any other error (bad key, network etc) — fall through to template
            return None
    return None


def _conversational_fallback(state: TourismState) -> str:
    """Short traveler-facing answer when the live writer is unavailable."""
    dest = state.get("destination") or "your destination"
    party = state.get("party_type")
    start, end = state.get("start_date"), state.get("end_date")
    forecasts = state.get("forecasts") or {}
    crowds = state.get("crowd_levels") or {}
    pressure = state.get("period_pressure") or {}
    ranked = state.get("ranked_alternatives") or []
    found = next((v for v in forecasts.values() if (v or {}).get("found")), None)
    trip = f" for your {party} trip" if party else ""
    dates = f" from {start} to {end or start}" if start else ""

    if not found:
        return (
            f"I couldn't complete the live analysis for {dest}{trip}{dates} just now. "
            "I have not guessed a crowd number. Please try again in a moment."
        )

    name = found.get("destination") or dest
    level = (crowds.get(name) or crowds.get(dest) or {}).get("level") or "unknown"
    answer = (
        f"For {name}{trip}{dates}, the available ASI annual history indicates a "
        f"{level.lower()} relative crowd level."
    )
    if pressure.get("found"):
        direction = pressure.get("footfall_direction") or "similar"
        answer += f" Your travel window looks {direction.replace('_', ' ')} than its usual annual baseline."
    if ranked:
        names = ", ".join(str(item.get("name")) for item in ranked[:3] if item.get("name"))
        if names:
            answer += f" If you prefer alternatives, consider {names}."
    answer += " I can refine this further when the live research connection is available."
    return answer


def compose_answer(state: TourismState) -> str:
    """Live: Gemini conversation from tool facts. Offline: template. Asks stay questions."""
    if state.get("brain_source") == "gemini_unavailable":
        return (
            "I need the Gemini brain to continue this conversation. Please set GEMINI_API_KEY "
            "and keep TOURISM_LLM_PROVIDER=gemini, then try again."
        )
    if state.get("awaiting_user") and (state.get("pending_question") or "").strip():
        return str(state["pending_question"]).strip()
    # Only fall back to short template on hard errors — NOT on llm_cap.
    # llm_cap means the brain ran all loops and gathered full evidence; Gemini
    # should still write the final answer from that evidence.
    if state.get("brain_source") == "llm_error":
        return _conversational_fallback(state)
    live = llm_write_answer(state)
    if live:
        return live
    return compose_final_response(state)


def lookup_has_coords_note(ranked: list) -> bool:
    return all(r.get("distance_km") is not None for r in ranked) if ranked else True


def _append_web(lines: list[str], web: dict) -> None:
    if not web:
        lines.append("No current-context search was required or no usable hits remained after filtering.")
        return
    lines.append(f"Provider: {web.get('provider')} | retrieved_on: {web.get('retrieved_on')}")
    if web.get("queries"):
        lines.append("Queries: " + "; ".join(web.get("queries")[:4]))
    any_bucket = False
    for key in ("weather", "events", "holidays", "closures", "timings", "advisories"):
        hits = web.get(key) or []
        if not hits:
            continue
        any_bucket = True
        lines.append(f"{key}:")
        for hit in hits[:3]:
            tier = hit.get("evidence_tier") or "general_web"
            snippet = (hit.get("snippet") or "")[:160]
            lines.append(f"- [{tier}] {hit.get('title')} | {hit.get('url')}")
            if snippet:
                lines.append(f"  {snippet}")
    leftover = web.get("discovery") or []
    official_disc = [h for h in leftover if h.get("evidence_tier") == "official"]
    if official_disc and not any_bucket:
        lines.append("official destination pages (not classified as weather/events):")
        for hit in official_disc[:4]:
            lines.append(f"- [official] {hit.get('title')} | {hit.get('url')}")
    elif not any_bucket and leftover:
        lines.append("No weather/events/closures matched the trip dates. Unclassified hits were not used as event facts.")
    for w in web.get("warnings") or []:
        lines.append(f"Note: {w}")
    if not (any_bucket or official_disc or leftover or web.get("warnings")):
        lines.append("No current-context facts retained after junk/relevance filters.")
    lines.append("Web facts are not MODEL OUTPUT and are not crowd counts.")
