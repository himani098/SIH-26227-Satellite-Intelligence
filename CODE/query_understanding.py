"""Deterministic offline parsing of common satellite search constraints."""

import math
import re


RANCHI_BOUNDS = {
    "lon_min": 85.15,
    "lat_min": 23.20,
    "lon_max": 85.45,
    "lat_max": 23.45,
}

OBJECT_TERMS = {
    "construction": ("new construction", "construction", "built-up", "built up", "building"),
    "road_development": ("road development", "new road", "roads", "road"),
    "water_extent": ("water extent", "water body", "waterbody", "lake", "river"),
    "clearance": ("clearance", "cleared land", "land clearing", "deforestation"),
    "vehicle_concentration": ("vehicle concentration", "vehicles", "vehicle"),
}


def parse_query(query):
    """Extract known object, location, date, and proximity constraints.

    The original query remains available for CLIP embedding. Place resolution is
    deliberately limited to the project's Ranchi AOI; no network gazetteer is used.
    """
    text = query.strip()
    lowered = text.lower()
    parsed = {
        "object": None,
        "location": None,
        "location_bounds": None,
        "date_from": None,
        "date_to": None,
        "near_feature": None,
        "radius_m": None,
    }

    for object_name, terms in OBJECT_TERMS.items():
        if any(re.search(r"\b" + re.escape(term) + r"\b", lowered) for term in terms):
            parsed["object"] = object_name
            break

    date_values = re.findall(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b", lowered)
    year_values = re.findall(r"\b(?:19|20)\d{2}\b", lowered)
    if date_values:
        parsed["date_from"] = min(date_values)
        parsed["date_to"] = max(date_values)
    elif year_values:
        years = sorted(set(year_values))
        if re.search(r"\b(before|until|through)\b", lowered) and len(years) == 1:
            parsed["date_to"] = f"{years[0]}-12-31"
        elif re.search(r"\b(since|after)\b", lowered) and len(years) == 1:
            parsed["date_from"] = f"{years[0]}-01-01"
        else:
            parsed["date_from"] = f"{years[0]}-01-01"
            parsed["date_to"] = f"{years[-1]}-12-31"

    radius_match = re.search(
        r"\b(?:within|under|up to)\s+(\d+(?:\.\d+)?)\s*(km|kilometers?|m|meters?)\s+of\s+(?:a\s+)?(river|stream|waterbody|water body)\b",
        lowered,
    )
    if radius_match:
        distance = float(radius_match.group(1))
        parsed["radius_m"] = distance * (1000 if radius_match.group(2).startswith("k") else 1)
        parsed["near_feature"] = "water"
    elif re.search(r"\b(?:near|beside|along)\s+(?:a\s+)?(river|stream|waterbody|water body)\b", lowered):
        parsed["near_feature"] = "water"
        parsed["radius_m"] = 500.0

    coordinate_match = re.search(
        r"\b(-?\d{1,2}(?:\.\d+)?)\s*[,/]\s*(-?\d{1,3}(?:\.\d+)?)\b", text
    )
    if coordinate_match:
        lat, lon = map(float, coordinate_match.groups())
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            radius = parsed["radius_m"] or 500.0
            lat_delta = radius / 111_320
            lon_delta = radius / (111_320 * max(0.1, abs(math.cos(math.radians(lat)))))
            parsed["location"] = f"{lat:.6f},{lon:.6f}"
            parsed["location_bounds"] = {
                "lon_min": lon - lon_delta,
                "lat_min": lat - lat_delta,
                "lon_max": lon + lon_delta,
                "lat_max": lat + lat_delta,
            }

    if re.search(r"\branchi\b", lowered):
        parsed["location"] = "Ranchi"
        parsed["location_bounds"] = dict(RANCHI_BOUNDS)

    return parsed