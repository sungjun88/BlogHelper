from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import ExifTags, Image


PLACE_LOOKUP_ENABLED = os.getenv("BLOGHELPER_PLACE_LOOKUP_ENABLED", "1").lower() not in {"0", "false", "no"}
PLACE_LOOKUP_TIMEOUT = float(os.getenv("BLOGHELPER_PLACE_LOOKUP_TIMEOUT", "8"))
PLACE_LOOKUP_RADIUS_METERS = float(os.getenv("BLOGHELPER_PLACE_LOOKUP_RADIUS_METERS", "120"))
PLACE_LOOKUP_USER_AGENT = os.getenv(
    "BLOGHELPER_PLACE_LOOKUP_USER_AGENT",
    "BlogHelper/1.0 (https://github.com/sungjun88/BlogHelper)",
)
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
OVERPASS_API_URL = os.getenv("BLOGHELPER_OVERPASS_API_URL", "https://overpass-api.de/api/interpreter")


GPS_TAG = next(key for key, value in ExifTags.TAGS.items() if value == "GPSInfo")
GPS_NAME_BY_KEY = ExifTags.GPSTAGS


def _extract_gps_ifd(exif: Any, gps_raw: Any) -> dict[str, Any] | None:
    if isinstance(gps_raw, dict):
        return gps_raw

    if hasattr(exif, "get_ifd") and hasattr(ExifTags, "IFD"):
        try:
            gps_ifd_key = getattr(ExifTags.IFD, "GPSInfo", None)
            if gps_ifd_key is not None:
                gps_ifd = exif.get_ifd(gps_ifd_key)
                if isinstance(gps_ifd, dict):
                    return gps_ifd
        except Exception:
            return None

    return None


def _rational_to_float(value: Any) -> float:
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        denominator = value.denominator or 1
        return float(value.numerator) / float(denominator)
    if isinstance(value, tuple) and len(value) == 2:
        denominator = value[1] or 1
        return float(value[0]) / float(denominator)
    return float(value)


def _dms_to_decimal(values: Any, ref: str) -> float | None:
    if not values or len(values) < 3:
        return None

    degrees = _rational_to_float(values[0])
    minutes = _rational_to_float(values[1])
    seconds = _rational_to_float(values[2])
    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)

    if ref in {"S", "W"}:
        decimal *= -1.0
    return decimal


def extract_gps_info(image_path: Path) -> dict[str, float] | None:
    try:
        with Image.open(image_path) as image:
            exif = image.getexif()
    except Exception:
        return None

    gps_raw = exif.get(GPS_TAG)
    if not gps_raw:
        return None

    gps_ifd = _extract_gps_ifd(exif, gps_raw)
    if not isinstance(gps_ifd, dict):
        return None

    gps_info = {
        GPS_NAME_BY_KEY.get(key, key): value
        for key, value in gps_ifd.items()
    }

    latitude = _dms_to_decimal(gps_info.get("GPSLatitude"), gps_info.get("GPSLatitudeRef", "N"))
    longitude = _dms_to_decimal(gps_info.get("GPSLongitude"), gps_info.get("GPSLongitudeRef", "E"))
    if latitude is None or longitude is None:
        return None

    return {
        "latitude": round(latitude, 7),
        "longitude": round(longitude, 7),
    }


def _fetch_json(url: str, params: dict[str, Any] | None = None, *, method: str = "GET", data: bytes | None = None) -> Any:
    target_url = url
    if params:
        target_url = f"{url}?{urlencode(params)}"

    request = Request(
        target_url,
        data=data,
        method=method,
        headers={
            "User-Agent": PLACE_LOOKUP_USER_AGENT,
            "Accept": "application/json",
        },
    )

    with urlopen(request, timeout=PLACE_LOOKUP_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def _haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_m * c


def reverse_geocode(lat: float, lon: float) -> dict[str, Any] | None:
    if not PLACE_LOOKUP_ENABLED:
        return None

    try:
        payload = _fetch_json(
            NOMINATIM_REVERSE_URL,
            {
                "lat": lat,
                "lon": lon,
                "format": "jsonv2",
                "addressdetails": 1,
                "zoom": 18,
                "namedetails": 1,
            },
        )
    except (HTTPError, URLError, TimeoutError, ValueError):
        return None

    if not isinstance(payload, dict):
        return None

    address = payload.get("address") or {}
    return {
        "display_name": payload.get("display_name"),
        "name": payload.get("name") or payload.get("namedetails", {}).get("name"),
        "category": payload.get("category"),
        "type": payload.get("type"),
        "address": address,
    }


def _build_overpass_query(lat: float, lon: float, radius_meters: float) -> str:
    return f"""
[out:json][timeout:10];
(
  node(around:{radius_meters},{lat},{lon})[name][amenity];
  way(around:{radius_meters},{lat},{lon})[name][amenity];
  rel(around:{radius_meters},{lat},{lon})[name][amenity];
  node(around:{radius_meters},{lat},{lon})[name][shop];
  way(around:{radius_meters},{lat},{lon})[name][shop];
  rel(around:{radius_meters},{lat},{lon})[name][shop];
  node(around:{radius_meters},{lat},{lon})[name][tourism];
  way(around:{radius_meters},{lat},{lon})[name][tourism];
  rel(around:{radius_meters},{lat},{lon})[name][tourism];
);
out center tags;
""".strip()


def find_nearest_places(lat: float, lon: float, limit: int = 3) -> list[dict[str, Any]]:
    if not PLACE_LOOKUP_ENABLED:
        return []

    query = _build_overpass_query(lat, lon, PLACE_LOOKUP_RADIUS_METERS)

    try:
        payload = _fetch_json(
            OVERPASS_API_URL,
            method="POST",
            data=query.encode("utf-8"),
        )
    except (HTTPError, URLError, TimeoutError, ValueError):
        return []

    elements = payload.get("elements") if isinstance(payload, dict) else None
    if not elements:
        return []

    candidates: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str | None]] = set()

    for element in elements:
        tags = element.get("tags") or {}
        name = tags.get("name")
        if not name:
            continue

        candidate_lat = element.get("lat")
        candidate_lon = element.get("lon")
        center = element.get("center") or {}
        if candidate_lat is None:
            candidate_lat = center.get("lat")
        if candidate_lon is None:
            candidate_lon = center.get("lon")
        if candidate_lat is None or candidate_lon is None:
            continue

        distance = _haversine_distance_meters(lat, lon, float(candidate_lat), float(candidate_lon))
        dedupe_key = (name, tags.get("brand"))
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)

        candidates.append(
            {
                "name": name,
                "distance_meters": round(distance, 1),
                "kind": tags.get("amenity") or tags.get("shop") or tags.get("tourism"),
                "brand": tags.get("brand"),
                "cuisine": tags.get("cuisine"),
                "website": tags.get("website"),
                "phone": tags.get("phone"),
                "osm_type": element.get("type"),
                "osm_id": element.get("id"),
            }
        )

    candidates.sort(key=lambda item: item["distance_meters"])
    return candidates[:limit]


def lookup_place_info(image_path: Path) -> dict[str, Any] | None:
    gps = extract_gps_info(image_path)
    if not gps:
        return None

    lat = gps["latitude"]
    lon = gps["longitude"]
    reverse = reverse_geocode(lat, lon)
    nearby_places = find_nearest_places(lat, lon, limit=3)
    nearest_place = nearby_places[0] if nearby_places else None

    return {
        "gps": gps,
        "reverse_geocode": reverse,
        "nearest_place": nearest_place,
        "nearby_places": nearby_places,
    }


def cluster_media_by_gps(items: list[dict[str, Any]], max_distance_meters: float = 10.0) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []

    for item in items:
        place_info = item.get("place_info") or {}
        gps = place_info.get("gps")
        if not gps:
            continue

        lat = float(gps["latitude"])
        lon = float(gps["longitude"])
        matched_group = None
        matched_distance = None

        for group in groups:
            center = group["center"]
            distance = _haversine_distance_meters(lat, lon, center["latitude"], center["longitude"])
            if distance <= max_distance_meters and (matched_distance is None or distance < matched_distance):
                matched_group = group
                matched_distance = distance

        if matched_group is None:
            nearest = place_info.get("nearest_place") or {}
            reverse = place_info.get("reverse_geocode") or {}
            groups.append(
                {
                    "group_id": f"gps-{len(groups) + 1}",
                    "center": {
                        "latitude": lat,
                        "longitude": lon,
                    },
                    "items": [item],
                    "place_name": nearest.get("name") or reverse.get("name"),
                    "address": reverse.get("display_name"),
                }
            )
            continue

        matched_group["items"].append(item)
        item_count = len(matched_group["items"])
        center = matched_group["center"]
        center["latitude"] = round(((center["latitude"] * (item_count - 1)) + lat) / item_count, 7)
        center["longitude"] = round(((center["longitude"] * (item_count - 1)) + lon) / item_count, 7)

        if not matched_group.get("place_name"):
            nearest = place_info.get("nearest_place") or {}
            reverse = place_info.get("reverse_geocode") or {}
            matched_group["place_name"] = nearest.get("name") or reverse.get("name")
        if not matched_group.get("address"):
            reverse = place_info.get("reverse_geocode") or {}
            matched_group["address"] = reverse.get("display_name")

    for group in groups:
        group["count"] = len(group["items"])

    groups.sort(key=lambda group: group["count"], reverse=True)
    return groups
