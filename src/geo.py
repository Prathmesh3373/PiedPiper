"""Geographic practicality for nearby (orbit) recommendations.

City/town coordinates are a small gazetteer of published centroids.
Monument-level GPS is not invented: unknown places get distance_km=None.
"""

from __future__ import annotations

import math

# Published city/town centroids (decimal degrees). Not monument GPS.
CITY_COORDS: dict[str, tuple[float, float]] = {
    # Published city/town centroids (decimal degrees).
    "kolhapur": (16.7050, 74.2433),
    "panhala": (16.8120, 74.1240),
    "jyotiba": (16.7972, 74.1772),
    "kopeshwar": (16.6920, 74.6860),
    "khidrapur": (16.6920, 74.6860),
    "rankala": (16.6890, 74.2170),
    "narsobachi wadi": (16.6920, 74.5980),
    "radhanagari": (16.4167, 74.0000),
    "dajipur": (16.3833, 73.9167),
    "satara": (17.6805, 74.0183),
    "sajjangad": (17.6333, 73.9167),
    "thoseghar": (17.5960, 73.8440),
    "kaas": (17.7200, 73.8160),
    "kaas plateau": (17.7200, 73.8160),
    "ajinkyatara": (17.6740, 74.0150),
    "pratapgad": (17.9317, 73.5786),
    "solapur": (17.6599, 75.9064),
    "sholapur": (17.6599, 75.9064),
    "sangli": (16.8524, 74.5815),
    "miraj": (16.8302, 74.6470),
    "sagareshwar": (17.1660, 74.3830),
    "ratnagiri": (16.9902, 73.3120),
    "ganpatipule": (17.1460, 73.2670),
    "jaigad": (17.3000, 73.2167),
    "sindhudurg": (16.1700, 73.7000),
    "malvan": (16.0598, 73.4700),
    "tarkarli": (16.0300, 73.4900),
    "vengurla": (15.8610, 73.6320),
    "sawantwadi": (15.9050, 73.8210),
    "amboli": (15.9600, 73.9980),
    "belgaum": (15.8497, 74.4977),
    "belagavi": (15.8497, 74.4977),
    "belgaon": (15.8497, 74.4977),
    "pune": (18.5204, 73.8567),
    "sinhagad": (18.3663, 73.7558),
    "mahabaleshwar": (17.9300, 73.6470),
    "panchgani": (17.9244, 73.8167),
    "wai": (17.9480, 73.8910),
    "lonavala": (18.7481, 73.4072),
    "khandala": (18.7564, 73.3725),
    "karla": (18.7820, 73.4700),
    "mumbai": (19.0760, 72.8777),
    "elephanta": (18.9633, 72.9315),
    "alibaug": (18.6411, 72.8722),
    "murud janjira": (18.3006, 72.9644),
    "raigad": (18.2356, 73.4447),
    "aurangabad": (19.8762, 75.3433),
    "chhatrapati sambhajinagar": (19.8762, 75.3433),
    "ellora": (20.0268, 75.1780),
    "daulatabad": (19.9430, 75.2130),
    "ajanta": (20.5519, 75.7033),
    "lonar": (19.9756, 76.5058),
    "delhi": (28.6139, 77.2090),
    "new delhi": (28.6139, 77.2090),
    "qutub minar": (28.5245, 77.1855),
    "red fort": (28.6562, 77.2410),
    "humayun tomb": (28.5933, 77.2507),
    "agra": (27.1767, 78.0081),
    "taj mahal": (27.1751, 78.0421),
    "agra fort": (27.1795, 78.0211),
    "sikandra": (27.2207, 77.9507),
    "fatehpur sikri": (27.0940, 77.6680),
    "mathura": (27.4924, 77.6737),
    "vrindavan": (27.5806, 77.7006),
    "deeg": (27.4725, 77.3256),
    "bharatpur": (27.2170, 77.4895),
    "hampi": (15.3350, 76.4600),
    "hospet": (15.2695, 76.3871),
    "hosapete": (15.2695, 76.3871),
    "bellary": (15.1394, 76.9214),
    "ballari": (15.1394, 76.9214),
    "badami": (15.9189, 75.6766),
    "aihole": (16.0200, 75.8820),
    "pattadakal": (15.9486, 75.8161),
    "bijapur": (16.8302, 75.7100),
    "vijayapura": (16.8302, 75.7100),
    "chitradurga": (14.2251, 76.3980),
    "sanchi": (23.4793, 77.7398),
    "bhopal": (23.2599, 77.4126),
    "vidisha": (23.5251, 77.8081),
    "gwalior": (26.2183, 78.1828),
    "orchha": (25.3510, 78.6416),
    "jhansi": (25.4484, 78.5685),
    "khajuraho": (24.8318, 79.9199),
    "chennai": (13.0827, 80.2707),
    "mamallapuram": (12.6208, 80.1920),
    "mahabalipuram": (12.6208, 80.1920),
    "kanchipuram": (12.8342, 79.7036),
    "thanjavur": (10.7870, 79.1378),
    "madurai": (9.9252, 78.1198),
    "bengaluru": (12.9716, 77.5946),
    "bangalore": (12.9716, 77.5946),
    "mysore": (12.2958, 76.6394),
    "mysuru": (12.2958, 76.6394),
    "srirangapatna": (12.4181, 76.6947),
    "belur": (13.1622, 75.8643),
    "halebidu": (13.2167, 75.9833),
    "shravanabelagola": (12.8570, 76.4860),
    "gokarna": (14.5479, 74.3188),
    "murudeshwar": (14.0940, 74.4899),
    "coorg": (12.4244, 75.7382),
    "madikeri": (12.4244, 75.7382),
    "chikmagalur": (13.3161, 75.7720),
    "kochi": (9.9312, 76.2673),
    "munnar": (10.0889, 77.0595),
    "alleppey": (9.4981, 76.3388),
    "alappuzha": (9.4981, 76.3388),
    "wayanad": (11.6854, 76.1320),
    "bekal": (12.3928, 75.0347),
    "panaji": (15.4909, 73.8278),
    "goa": (15.2993, 74.1240),
    "jaipur": (26.9124, 75.7873),
    "udaipur": (24.5854, 73.7125),
    "jodhpur": (26.2389, 73.0243),
    "jaisalmer": (26.9157, 70.9083),
    "pushkar": (26.4897, 74.5511),
    "chittorgarh": (24.8887, 74.6269),
    "kumbhalgarh": (25.1479, 73.5873),
    "bundi": (25.4415, 75.6440),
    "varanasi": (25.3176, 82.9739),
    "sarnath": (25.3811, 83.0214),
    "lucknow": (26.8467, 80.9462),
    "allahabad": (25.4358, 81.8463),
    "prayagraj": (25.4358, 81.8463),
    "konark": (19.8876, 86.0945),
    "puri": (19.8135, 85.8312),
    "bhubaneswar": (20.2961, 85.8245),
    "amritsar": (31.6340, 74.8723),
    "rishikesh": (30.0869, 78.2676),
    "haridwar": (29.9457, 78.1642),
    "shimla": (31.1048, 77.1734),
    "manali": (32.2432, 77.1892),
}


def get_coordinates(name_or_city: str | None) -> tuple[float, float] | None:
    """Resolve coordinates from gazetteer with fuzzy matching."""
    if not name_or_city:
        return None
    raw = str(name_or_city).strip().lower()
    if raw in CITY_COORDS:
        return CITY_COORDS[raw]
    for key, coords in CITY_COORDS.items():
        if key in raw or raw in key:
            return coords
    return None

# Ordinary alternative radius (day-trip). Same-state is not "nearby".
MAX_NEARBY_KM = 160.0
MAX_RELAXED_KM = 220.0

NEARBY_CITIES: dict[str, tuple[str, ...]] = {
    "kolhapur": (
        "kolhapur",
        "panhala",
        "satara",
        "sangli",
        "ratnagiri",
        "sindhudurg",
        "belgaum",
        "belagavi",
    ),
    "satara": ("satara", "pune", "mahabaleshwar", "kolhapur", "lonavala"),
    "pune": ("pune", "lonavala", "mahabaleshwar", "satara", "mumbai"),
    "sindhudurg": (
        "sindhudurg",
        "ratnagiri",
        "malvan",
        "vengurla",
        "sawantwadi",
        "kolhapur",
    ),
    "hampi": ("hampi", "hospet", "hosapete", "bellary", "ballari"),
    "agra": ("agra", "mathura", "fatehpur sikri", "sikandra"),
    "taj mahal": ("agra", "mathura", "fatehpur sikri", "sikandra"),
    "delhi": ("delhi", "new delhi", "noida", "gurgaon", "gurugram", "faridabad"),
    "qutub minar": ("delhi", "new delhi"),
    "chennai": ("chennai", "mamallapuram", "mahabalipuram", "kanchipuram"),
    "bengaluru": ("bengaluru", "bangalore", "mysore", "mysuru"),
    "bangalore": ("bengaluru", "bangalore", "mysore", "mysuru"),
    "sanchi": ("sanchi", "bhopal", "vidisha"),
    "belgaum": (
        "belgaum",
        "belagavi",
        "belgaon",
        "kolhapur",
        "sangli",
        "dharwad",
    ),
    "belagavi": (
        "belgaum",
        "belagavi",
        "belgaon",
        "kolhapur",
        "sangli",
        "dharwad",
    ),
    "belgaon": (
        "belgaum",
        "belagavi",
        "belgaon",
        "kolhapur",
        "sangli",
        "dharwad",
    ),
    "sangli": (
        "sangli",
        "miraj",
        "kolhapur",
        "satara",
        "kopeshwar",
        "khidrapur",
        "sagareshwar",
        "belgaum",
        "belagavi",
    ),
}

STATE_ALIASES = {
    "maharastra": "maharashtra",
    "maharashtra": "maharashtra",
    "karnataka": "karnataka",
    "delhi": "delhi",
    "tamil nadu": "tamil nadu",
    "tamilnadu": "tamil nadu",
    "uttar pradesh": "uttar pradesh",
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return round(2 * r * math.asin(math.sqrt(a)), 1)


def lookup_coords(name: str | None) -> tuple[float, float] | None:
    if not name:
        return None
    key = name.strip().lower()
    if key in CITY_COORDS:
        return CITY_COORDS[key]
    for token, coords in CITY_COORDS.items():
        if token in key or key in token:
            return coords
    return None


def normalize_state(value: str | None) -> str:
    if not value:
        return ""
    return STATE_ALIASES.get(value.strip().lower(), value.strip().lower())


def nearby_set(origin: str | None) -> set[str]:
    if not origin:
        return set()
    from src.place_aliases import spellings

    key = origin.strip().lower()
    extra = NEARBY_CITIES.get(key, ())
    out = {key, *extra}
    for token in list(out):
        out |= spellings(token)
    return {t for t in out if t}


def geo_tier(
    *,
    origin_name: str,
    origin_city: str | None,
    origin_state: str | None,
    cand_city: str | None,
    cand_state: str | None,
    cand_name: str | None = None,
    same_asi_circle: bool = False,
    distance_km: float | None = None,
) -> str:
    """Geography only. same_asi_circle is ignored for locality (similarity is separate)."""
    del same_asi_circle
    oc = (origin_city or origin_name or "").lower()
    cc = (cand_city or "").lower()
    name_l = (cand_name or "").lower()
    near = nearby_set(origin_name) | nearby_set(origin_city)
    blob = f"{cc} {name_l}".strip()
    if cc and (cc == oc or oc in cc or cc in oc):
        return "locality_city"
    if blob and any(n and n in blob for n in near if n != oc):
        return "nearby_region"
    if cc and cc in near:
        return "nearby_region"
    if distance_km is not None and distance_km <= 40:
        return "locality_district"
    if distance_km is not None and distance_km <= MAX_NEARBY_KM:
        return "nearby_region"
    os_ = normalize_state(origin_state)
    cs = normalize_state(cand_state)
    if os_ and cs and os_ == cs:
        return "same_state"
    return "unverified"


def geography_verified(tier: str, distance_km: float | None) -> bool:
    if distance_km is not None:
        return distance_km <= MAX_NEARBY_KM
    return tier in {"locality_city", "locality_district", "nearby_region"}


def distance_score(distance_km: float | None, tier: str) -> float:
    """1 = next door, 0 = too far. Missing coords: never assume nearby via ASI circle."""
    if distance_km is not None:
        if distance_km <= 25:
            return 1.0
        if distance_km <= 60:
            return 0.85
        if distance_km <= 100:
            return 0.55
        if distance_km <= MAX_NEARBY_KM:
            return 0.25
        return 0.0
    if tier in {"locality_city", "locality_district"}:
        return 0.75
    if tier == "nearby_region":
        return 0.55
    if tier == "same_state":
        return 0.1
    return 0.05


def is_practical_orbit(tier: str, distance_km: float | None, *, relax: bool = False) -> bool:
    """ASI-circle-only is not practical. Geography must be verified by place names or km."""
    cap = MAX_RELAXED_KM if relax else MAX_NEARBY_KM
    if distance_km is not None:
        return distance_km <= cap
    return tier in {"locality_city", "locality_district", "nearby_region"}


def pair_distance_km(origin_name: str, cand_name: str, cand_city: str | None = None) -> float | None:
    a = lookup_coords(origin_name)
    b = lookup_coords(cand_city) or lookup_coords(cand_name)
    if not a or not b:
        return None
    return haversine_km(a[0], a[1], b[0], b[1])
