"""Traveler-facing answer. Live path: OpenAI writes from tool facts. Offline: template."""

from __future__ import annotations

import json
import os

from src.agent.reasoning import build_decision_reasoning
from src.agent.research import nearby_from_research
from src.agent.state import TourismState

ANSWER_SYSTEM = """You are chatting with the traveler. Write like a helpful person, not a report generator and not a form.

RESEARCH_FACTS is the only evidence.

If awaiting_user is true and there are no forecasts:
- Only continue the conversation. Ask for missing_slots / pending_question.
- Do not dump a crowd report. Do not invent a place, dates, or party.

If awaiting_user is true AND forecasts or ranked places exist:
- Share those findings in conversation, then ask the missing dates/party naturally.

If research ran:
- Talk naturally: acknowledge what they said (place, dates, family/friends/solo).
- Say whether pressure for THEIR dates looks higher or not, using the research bundle (catalog history if any, season, weekends, events, weather). Annual HIGH is not the only reason.
- Do not invent visitor counts. If found=false, say we have no official ASI series.
- Copy visitor numbers from facts exactly if present.
- Nearby names only from ranked_alternatives. Offer them as suggestions in conversation.
- Mention a loose schedule only if trip_plan exists.
- Keep it short enough to feel like chat. One or two follow-up questions are welcome if something is still missing.
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
    intent = state.get("intent") or {}
    forecasts = state.get("forecasts") or {}
    crowds = state.get("crowd_levels") or {}
    ranked = state.get("ranked_alternatives") or []
    dests = state.get("destinations") or []
    web = state.get("web_context") or {}
    plan = state.get("trip_plan") or []
    dest = state.get("destination")
    found_any = any(v.get("found") for v in forecasts.values())
    hourly = intent.get("wants_hourly") or state.get("wants_hourly")
    lines: list[str] = _brief_conclusion(state)

    lines += ["CROWD FORECAST"]
    if found_any:
        for name, fc in forecasts.items():
            if not fc.get("found"):
                lines.append(f"{name}: no validated ASI series (not invented).")
                continue
            crowd = crowds.get(fc.get("destination") or name) or crowds.get(name) or {}
            lines.append(f"Destination: {fc.get('destination')}")
            lines.append(
                f"Predicted demand: {_fmt_visitors(fc.get('predicted_visitors'))} "
                f"visitors ({fc.get('forecast_period')})"
            )
            lines.append(
                f"Relative crowd level: {crowd.get('level') or 'n/a'} "
                "(quartile vs this monument's own ASI history, not occupancy %)"
            )
            lines.append(f"Method: {fc.get('forecast_method')} | layer: MODEL OUTPUT")
            lines.append("")
    else:
        lines.append(
            "No validated monument-level crowd history"
            + (f" for {dest}." if dest else ".")
        )
        missing = [k for k, v in forecasts.items() if not v.get("found")]
        if missing:
            lines.append("Checked and unavailable: " + ", ".join(missing))
        lines.append("No crowd count is invented.")
        lines.append("")

    if hourly:
        lines += [
            "Daily and hourly headcount is not supported. The figure above is annual only.",
            "",
        ]

    pressure = state.get("period_pressure") or {}
    if pressure.get("found"):
        lines += ["TRAVEL WINDOW"]
        lines.append(
            f"Dates: {pressure.get('start_date')} → {pressure.get('end_date')} "
            f"({pressure.get('n_days')} day(s); "
            f"{pressure.get('weekend_days')} weekend / {pressure.get('weekday_days')} weekday)"
        )
        lines.append(f"Season (heuristic): {pressure.get('season')}")
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
                "why": (r.get("why") or [])[:4],
                "recommendation_mode": r.get("recommendation_mode"),
                "predicted_visitors": r.get("predicted_visitors"),
                "crowd_data_available": r.get("crowd_data_available"),
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
        "trip_plan": state.get("trip_plan") or [],
    }


def llm_write_answer(state: TourismState) -> str | None:
    if os.environ.get("TOURISM_LLM_BRAIN", "1") != "1":
        return None
    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        return None
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
    except Exception:
        return None
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0.45, max_tokens=900)
    try:
        msg = llm.invoke(
            [
                SystemMessage(content=ANSWER_SYSTEM),
                HumanMessage(content=json.dumps(_answer_facts(state), default=str)[:14000]),
            ]
        )
        text = (msg.content or "").strip()
        return text or None
    except Exception:
        return None


def compose_answer(state: TourismState) -> str:
    """Live: OpenAI conversation from tool facts. Offline: template. Asks stay questions."""
    live = llm_write_answer(state)
    if live:
        return live
    if state.get("awaiting_user") and (state.get("pending_question") or "").strip():
        return str(state["pending_question"]).strip()
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
