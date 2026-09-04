"""Travel-window pressure. Does not invent a new visitor headcount."""

from __future__ import annotations

from datetime import date, datetime, timedelta

# Gazetted-style dates we treat as India-wide busy (heuristic, not a visitor model).
INDIA_BUSY_DATES = {
    date(2025, 1, 1),
    date(2025, 1, 26),
    date(2025, 8, 15),
    date(2025, 10, 2),
    date(2025, 12, 25),
    date(2026, 1, 1),
    date(2026, 1, 26),
    date(2026, 8, 15),
    date(2026, 10, 2),
    date(2026, 12, 25),
    date(2022, 1, 26),
    date(2022, 8, 15),
    date(2022, 10, 2),
    date(2022, 12, 25),
}

PEAK_WINTER_MONTHS = {10, 11, 12, 1, 2, 3}
MONSOON_MONTHS = {6, 7, 8, 9}


def _parse(iso: str | None) -> date | None:
    if not iso:
        return None
    try:
        return datetime.strptime(iso[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def iter_days(start: date, end: date) -> list[date]:
    if end < start:
        start, end = end, start
    out = []
    cur = start
    while cur <= end and len(out) < 31:
        out.append(cur)
        cur += timedelta(days=1)
    return out


def season_for_months(
    months: list[int],
    dest_state: str | None = None,
    climate_zone: str | None = None,
    destination: str | None = None,
) -> str:
    if not months:
        return "unknown"
    monsoon = sum(1 for m in months if m in MONSOON_MONTHS)
    winter = sum(1 for m in months if m in PEAK_WINTER_MONTHS)
    if monsoon >= winter and monsoon > 0:
        return "monsoon"
    if winter <= 0:
        return "shoulder"
    blob = f"{destination or ''} {dest_state or ''} {climate_zone or ''}".lower()
    south = any(
        x in blob
        for x in (
            "karnataka",
            "maharashtra",
            "tamil",
            "kerala",
            "goa",
            "andhra",
            "telangana",
            "peninsular",
            "coastal",
            "belagavi",
            "belgaum",
            "belgaon",
            "hampi",
            "mumbai",
            "pune",
            "kolhapur",
            "sangli",
            "chennai",
            "bengaluru",
            "bangalore",
        )
    )
    north = any(
        x in blob
        for x in (
            "north_india",
            "delhi",
            "uttar pradesh",
            "rajasthan",
            "punjab",
            "haryana",
            "taj mahal",
            "agra",
            "qutub",
        )
    )
    if climate_zone in {"peninsular", "coastal"} or (south and not north):
        return "dry_cool_season"
    if climate_zone == "north_india" or north:
        return "peak_winter_tourism"
    return "winter"


def assess_period_pressure(
    destination: str | None,
    start_date: str | None,
    end_date: str | None,
    crowd_level: str | None = None,
    web_context: dict | None = None,
    dest_state: str | None = None,
    climate_zone: str | None = None,
) -> dict:
    """Direction of pressure vs the annual ASI figure. Never a new count."""
    start = _parse(start_date)
    end = _parse(end_date) or start
    web = web_context or {}
    if not start:
        return {
            "found": False,
            "footfall_direction": "unknown",
            "layer": "HEURISTIC",
            "note": "No travel dates, so no weekend/holiday/season window.",
            "does_not_change_annual_count": True,
        }

    days = iter_days(start, end)
    weekend_days = [d for d in days if d.weekday() >= 5]
    holiday_days = [d for d in days if d in INDIA_BUSY_DATES]
    months = [d.month for d in days]
    season = season_for_months(months, dest_state, climate_zone, destination)

    weather_hits = list(web.get("weather") or [])
    event_hits = list(web.get("events") or []) + list(web.get("holidays") or [])

    flags: list[str] = []
    if weekend_days:
        flags.append(f"{len(weekend_days)} weekend day(s) in a {len(days)}-day window")
    if holiday_days:
        flags.append("public-holiday date(s) in window: " + ", ".join(d.isoformat() for d in holiday_days))
    if season == "peak_winter_tourism":
        flags.append("window overlaps typical north-India heritage peak months (heuristic)")
    if season == "dry_cool_season":
        flags.append("window is the cooler dry season in peninsular India (heuristic)")
    if season == "winter":
        flags.append("window overlaps winter months (heuristic; region not treated as north-India peak)")
    if season == "monsoon":
        flags.append("window overlaps monsoon months (heuristic; outdoor sites often slower except festivals)")
    if event_hits:
        flags.append("web mentioned events/holidays (WEB FACT, not a visitor count)")
    if weather_hits:
        flags.append("web mentioned weather (WEB FACT, not a visitor count)")

    up = 0
    down = 0
    if len(weekend_days) >= max(1, len(days) // 3):
        up += 1
    if holiday_days:
        up += 1
    if season == "peak_winter_tourism":
        up += 1
    if season == "monsoon" and not holiday_days and not event_hits:
        down += 1
    if event_hits:
        up += 1
    if crowd_level in {"HIGH", "VERY HIGH"}:
        up += 1

    if up >= 2 and up > down:
        direction = "likely_higher"
    elif down >= 1 and down > up:
        direction = "likely_lower"
    elif up == 1 and down == 0:
        direction = "likely_higher"
    else:
        direction = "similar"

    # Compute overall crowd density level
    if direction == "likely_higher" or crowd_level in {"HIGH", "VERY HIGH"} or (weekend_days and holiday_days):
        crowd_density = "HIGH"
    elif direction == "likely_lower" or crowd_level == "LOW":
        crowd_density = "LOW"
    else:
        crowd_density = "MODERATE"

    # Compute day-by-day score to pinpoint the most probable peak day
    day_breakdown = []
    best_day = None
    best_score = -1.0
    dest_str = (destination or "").lower()

    for d in days:
        d_score = 1.0
        reasons = []
        if d.weekday() == 6:  # Sunday
            d_score += 3.0
            reasons.append("Sunday weekend peak")
        elif d.weekday() == 5:  # Saturday
            d_score += 2.8
            reasons.append("Saturday weekend rush")
        elif d.weekday() == 4:  # Friday
            d_score += 1.2
            reasons.append("Pre-weekend travel")
        else:
            reasons.append(f"{d.strftime('%A')} weekday")

        if d in INDIA_BUSY_DATES:
            d_score += 3.5
            reasons.append("National/Gazetted public holiday")

        if event_hits:
            d_score += 1.0
            reasons.append("Active festival / event window")

        level = "HIGH" if d_score >= 4.0 else ("MODERATE" if d_score >= 2.0 else "LOW")
        day_breakdown.append({
            "date": d.isoformat(),
            "day": d.strftime("%A"),
            "predicted_level": level,
            "score": round(d_score, 1),
            "reasons": reasons,
        })

        if d_score > best_score:
            best_score = d_score
            best_day = {
                "date": d.isoformat(),
                "day": d.strftime("%A"),
                "reason": ", ".join(reasons),
            }

    # Predict peak hours and recommended visiting hours based on destination type
    is_temple = any(k in dest_str for k in ("temple", "mandir", "mahalakshmi", "jyotiba", "kopeshwar", "darshan", "mathura", "vrindavan", "puri", "madurai", "shrine"))
    is_viewpoint = any(k in dest_str for k in ("fort", "gad", "gadh", "point", "lake", "falls", "panhala", "sinhagad", "pratapgad", "plateau", "beach", "sunset"))

    if is_temple:
        peak_hours = "7:30 AM – 12:30 PM & 5:30 PM – 8:30 PM (Aarti & Darshan rush)"
        recommended_hours = "5:30 AM – 7:00 AM (Early Kakad Aarti) or 2:00 PM – 4:00 PM (Midday lull)"
    elif is_viewpoint:
        peak_hours = "3:30 PM – 6:30 PM (Sunset & weekend afternoon influx)"
        recommended_hours = "6:30 AM – 9:30 AM (Cool morning weather & unhindered views)"
    else:
        # Standard monuments / heritage sites (Taj Mahal, Qutub Minar, Hampi, Sanchi, etc.)
        peak_hours = "10:30 AM – 3:30 PM (Tour buses, day trippers & ticketing queues)"
        recommended_hours = "6:00 AM – 8:30 AM (Sunrise soft light & minimal queues) or 4:30 PM – 6:00 PM"

    return {
        "found": True,
        "destination": destination,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "n_days": len(days),
        "weekend_days": len(weekend_days),
        "weekday_days": len(days) - len(weekend_days),
        "holiday_dates": [d.isoformat() for d in holiday_days],
        "season": season,
        "crowd_level_annual": crowd_level,
        "crowd_density": crowd_density,
        "footfall_direction": direction,
        "peak_day": best_day,
        "peak_hours": peak_hours,
        "recommended_hours": recommended_hours,
        "day_breakdown": day_breakdown,
        "reasons": flags,
        "layer": "HEURISTIC",
        "web_used": bool(event_hits or weather_hits),
        "does_not_change_annual_count": True,
        "note": (
            "This is pressure on the travel window vs the annual ASI persistence figure. "
            "It is not a new daily/hourly visitor total."
        ),
    }
