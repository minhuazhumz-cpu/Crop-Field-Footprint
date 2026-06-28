"""USDA CDL crop code resolution and common US agricultural region bounding boxes."""

from __future__ import annotations

import re
from typing import Final

from src.config_loader import BboxWgs84

# USDA NASS CDL classification codes (CONUS). Aliases are lowercase keys.
CDL_CROP_CODES: Final[dict[str, int]] = {
    "corn": 1,
    "maize": 1,
    "cotton": 2,
    "rice": 3,
    "sorghum": 4,
    "soybeans": 5,
    "soybean": 5,
    "soy": 5,
    "sunflower": 6,
    "barley": 21,
    "wheat": 23,
    "winter wheat": 24,
    "spring wheat": 23,
    "durum wheat": 22,
    "canola": 31,
    "rapeseed": 31,
    "alfalfa": 36,
    "hay": 37,
    "other hay": 37,
    "lettuce": 227,
    "broccoli": 214,
    "cabbage": 243,
    "cauliflower": 244,
    "celery": 245,
    "carrots": 206,
    "carrot": 206,
    "tomatoes": 54,
    "tomato": 54,
    "potatoes": 43,
    "potato": 43,
    "grapes": 69,
    "grape": 69,
    "wine grapes": 69,
    "vineyard": 69,
    "almonds": 75,
    "almond": 75,
    "walnuts": 76,
    "walnut": 76,
    "pistachios": 204,
    "pistachio": 204,
    "oranges": 212,
    "orange": 212,
    "citrus": 72,
    "lemons": 72,
    "lemon": 72,
    "avocados": 215,
    "avocado": 215,
    "spinach": 219,
    "watermelon": 48,
    "watermelons": 48,
    "melons": 213,
    "melon": 213,
    "cantaloupe": 209,
    "cantaloupes": 209,
    "eggplant": 248,
    "eggplants": 248,
    "pumpkins": 229,
    "pumpkin": 229,
    "peppers": 216,
    "pepper": 216,
    "onions": 49,
    "onion": 49,
    "garlic": 208,
    "beans": 42,
    "dry beans": 42,
    "peas": 53,
    "pea": 53,
    "sugar beets": 41,
    "sugarbeet": 41,
    "peanuts": 10,
    "peanut": 10,
    "tobacco": 11,
    "sugarcane": 45,
    "hops": 56,
    "mint": 14,
}

# Well-known agricultural corridors — WGS84 [xmin, ymin, xmax, ymax].
REGION_BBOX_WGS84: Final[dict[str, BboxWgs84]] = {
    "yuma": (-114.85, 32.45, -113.90, 32.80),
    "yuma_az": (-114.85, 32.45, -113.90, 32.80),
    "yuma_arizona": (-114.85, 32.45, -113.90, 32.80),
    "gila_river": (-114.85, 32.45, -113.90, 32.80),
    "imperial_valley": (-115.65, 32.55, -114.45, 33.55),
    "imperial": (-115.65, 32.55, -114.45, 33.55),
    "salinas_valley": (-121.55, 36.35, -121.05, 36.95),
    "salinas": (-121.55, 36.35, -121.05, 36.95),
    "monterey_county": (-121.55, 36.35, -121.05, 36.95),
    "central_valley_ca": (-121.00, 35.00, -118.50, 40.50),
    "san_joaquin": (-121.00, 35.00, -118.50, 40.50),
    "fresno_county": (-120.85, 36.25, -118.95, 37.00),
    "kern_county": (-119.60, 34.80, -118.40, 35.80),
    "palouse": (-117.50, 46.50, -116.50, 47.20),
    "red_river_valley": (-97.50, 47.00, -96.50, 48.50),
    "texas_high_plains": (-102.50, 33.50, -101.00, 35.50),
    "panhandle_tx": (-102.50, 33.50, -101.00, 35.50),
    "iowa_core": (-95.50, 41.50, -90.50, 43.50),
    "illinois_corn_belt": (-91.00, 39.50, -87.50, 41.50),
}


def normalize_crop_key(crop_name: str) -> str:
    return re.sub(r"\s+", " ", crop_name.strip().lower())


def resolve_cdl_crop_code(crop_name: str, cdl_crop_code: int | None = None) -> int:
    """Map a human crop label to a USDA CDL integer code."""
    if cdl_crop_code is not None:
        return int(cdl_crop_code)

    key = normalize_crop_key(crop_name)
    if key in CDL_CROP_CODES:
        return CDL_CROP_CODES[key]

    # Partial match (e.g. "winter wheat fields" → winter wheat)
    for alias, code in sorted(CDL_CROP_CODES.items(), key=lambda x: -len(x[0])):
        if alias in key or key in alias:
            return code

    known = ", ".join(sorted(set(CDL_CROP_CODES.keys()))[:20])
    raise ValueError(
        f"Unknown crop '{crop_name}'. Pass cdl_crop_code explicitly or use a known "
        f"name (e.g. broccoli, lettuce, corn). Examples: {known}, ..."
    )


def resolve_region_bbox(region_name: str) -> BboxWgs84:
    """Resolve a place nickname to a WGS84 bounding box."""
    key = re.sub(r"[\s\-]+", "_", region_name.strip().lower())
    if key in REGION_BBOX_WGS84:
        return REGION_BBOX_WGS84[key]
    raise ValueError(
        f"Unknown region preset '{region_name}'. Pass bbox_wgs84 explicitly or use "
        f"a known preset: {', '.join(sorted(REGION_BBOX_WGS84.keys()))}."
    )


def validate_bbox_wgs84(bbox: list[float]) -> BboxWgs84:
    if len(bbox) != 4:
        raise ValueError("bbox_wgs84 must be [xmin, ymin, xmax, ymax] in WGS84 degrees.")
    xmin, ymin, xmax, ymax = (float(v) for v in bbox)
    if xmin >= xmax or ymin >= ymax:
        raise ValueError(f"Invalid bbox_wgs84: {bbox}")
    if not (-180 <= xmin <= 180 and -180 <= xmax <= 180):
        raise ValueError("Longitude must be between -180 and 180.")
    if not (-90 <= ymin <= 90 and -90 <= ymax <= 90):
        raise ValueError("Latitude must be between -90 and 90.")
    return (xmin, ymin, xmax, ymax)
