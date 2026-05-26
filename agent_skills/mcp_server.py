#!/usr/bin/env python3
"""
MCP server exposing Crop Field Footprint as a single tool: analyze_crop_footprint.

Run locally:
    python agent_skills/mcp_server.py

Or from the repo root with the venv active:
    python -m agent_skills.mcp_server
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

# Ensure repo root is importable when launched as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from agent_skills.crop_registry import (  # noqa: E402
    resolve_cdl_crop_code,
    validate_bbox_wgs84,
)
from src.config_loader import (  # noqa: E402
    PreprocessConfig,
    ThresholdConfig,
    build_profile,
)
from src.data_streamer import DataAccessError  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "Crop Field Footprint",
    instructions=(
        "Geospatial crop footprint analysis over the contiguous US. "
        "Use analyze_crop_footprint to stream USDA CDL rasters and FTW field "
        "boundaries, classify parcels, and write CSV + HTML map outputs."
    ),
)

OUTPUTS_ROOT = ROOT / "outputs"


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "crop"


def _format_markdown_summary(result) -> str:
    """Render a concise markdown report for the agent/user."""
    p = result.profile
    crop_field = f"{p.crop_name} Field"
    lines = [
        f"# {p.project_name}",
        "",
        "## Run parameters",
        f"- **Crop:** {p.crop_name} (USDA CDL code `{p.cdl_crop_code}`)",
        f"- **Year:** {p.year}",
        f"- **CDL source:** `{result.cdl_source}` "
        f"(STAC for year ≤ {p.cdl.stac_max_year}, CropScape otherwise)",
        f"- **BBox WGS84:** `{list(p.bbox_wgs84)}` — "
        "`[xmin, ymin, xmax, ymax]` in degrees",
        f"- **Classification:** `{p.classification_strategy}`",
        "",
        "## Key metrics",
        f"- **FTW parcels in AOI:** {result.n_fields:,}",
        f"- **Parcels classified as {crop_field}:** {result.n_crop_fields:,}",
        f"- **CDL unique {p.crop_name.lower()} acres (AOI raster):** "
        f"{result.cdl_unique_crop_acres:,.2f} ac",
        f"- **Sum of CDL acres inside {crop_field} parcels:** "
        f"{result.crop_field_cdl_acres_sum:,.2f} ac "
        "_(may double-count overlapping parcels)_",
        "",
        "## Output files",
        f"- CSV: `{result.csv_path}`",
        f"- Map: `{result.html_path}`",
        f"- Metadata: `{result.metadata_path}`",
        "",
        "Open the HTML file in a browser for an interactive field map with "
        "sticky hover tooltips.",
    ]
    return "\n".join(lines)


@mcp.tool()
def analyze_crop_footprint(
    crop_name: str,
    bbox_wgs84: list[float],
    year: int,
    cdl_crop_code: int | None = None,
    classification_strategy: str = "hybrid_dynamic",
    min_crop_acres: float | None = None,
    min_crop_pct: float | None = None,
    small_parcel_max_acres: float | None = None,
    small_parcel_min_crop_acres: float | None = None,
    small_parcel_min_pct: float | None = None,
    min_cluster_pixels: int | None = None,
    morph_open: bool | None = None,
) -> str:
    """
    Run cloud-native CDL × FTW crop footprint analysis for any US bounding box.

    **When to call this tool**
    Use when the user asks how many acres of a crop are growing inside field
    parcels, wants a parcel-level classification map, or needs zonal overlap
    stats between USDA Cropland Data Layer (CDL) and Fields of the World (FTW).

    **Parsing user intent (critical for agents)**

    1. **Crop name → USDA CDL code**
       - Pass the human crop string in `crop_name` (e.g. `"broccoli"`, `"lettuce"`,
         `"winter wheat"`).
       - The server resolves common names to official NASS CDL integer codes.
       - If ambiguous or rare, pass `cdl_crop_code` explicitly from the
         [NASS CDL legend](https://www.nass.usda.gov/Research_and_Science/Cropland/lookup_tables.php).
       - Examples: `broccoli` → 214, `lettuce` → 209, `corn` → 1, `cotton` → 2,
         `grapes`/`vineyard` → 68, `almonds` → 75, `tomatoes` → 320.

    2. **Place name → bbox_wgs84**
       - `bbox_wgs84` must be exactly **four floats**:
         `[xmin, ymin, xmax, ymax]` in **WGS84 degrees** (lon/lat).
       - Do **not** pass place names into this parameter — geocode first.
       - Known agricultural presets (use these coordinates when users say…):
         - **"Yuma" / "Yuma AZ" / "Gila River"**:
           `[-114.85, 32.45, -113.90, 32.80]`
         - **"Imperial Valley"**:
           `[-115.65, 32.55, -114.45, 33.55]`
         - **"Salinas Valley" / "Salinas"**:
           `[-121.55, 36.35, -121.05, 36.95]`
         - **"Central Valley" (broad CA)**:
           `[-121.00, 35.00, -118.50, 40.50]`
         - **"Iowa corn belt" (example)**:
           `[-95.50, 41.50, -90.50, 43.50]`
       - For other cities/regions, web-search or geocode the centroid, then build
         a tight bbox (~0.3–1.0° per side for county-scale; wider for multi-county).

    3. **Year**
       - Integer CDL product year (e.g. `2023`).
       - Years **≤ 2021** stream from Microsoft Planetary Computer STAC COGs.
       - Years **> 2021** automatically use USDA CropScape `GetCDLFile` (in-memory).

    4. **Threshold overrides (optional)**
       - Default strategy is `hybrid_dynamic` (large fields need more acres;
         small parcels have relaxed acre rules).
       - Override only when the user states explicit cutoffs:
         - `min_crop_acres`: minimum CDL crop acres to call a large parcel positive.
         - `small_parcel_max_acres`: parcel size cap for the small-field rule.
         - `small_parcel_min_crop_acres`: minimum CDL acres for small parcels.
         - `min_crop_pct`: use with `classification_strategy="pct"`.
       - `min_cluster_pixels` / `morph_open`: raster noise cleaning before zonal stats.

    **Outputs**
    Writes to `outputs/<crop>_<year>/` under the repo:
    - `insights_report.csv` — parcel table without geometry
    - `coverage_map.html` — interactive Folium map
    - `run_metadata.json` — machine-readable run summary

    **Returns**
    Markdown text with metrics and absolute file paths.

    **Runtime**
    Typically 5–10 minutes for county-scale AOIs (FTW cloud query dominates).
  """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    try:
        bbox = validate_bbox_wgs84(bbox_wgs84)
        code = resolve_cdl_crop_code(crop_name, cdl_crop_code)

        base_thresholds = ThresholdConfig()
        thresholds = ThresholdConfig(
            min_crop_acres=(
                min_crop_acres
                if min_crop_acres is not None
                else base_thresholds.min_crop_acres
            ),
            min_crop_pct=min_crop_pct,
            small_parcel_max_acres=(
                small_parcel_max_acres
                if small_parcel_max_acres is not None
                else base_thresholds.small_parcel_max_acres
            ),
            small_parcel_min_crop_acres=(
                small_parcel_min_crop_acres
                if small_parcel_min_crop_acres is not None
                else base_thresholds.small_parcel_min_crop_acres
            ),
            small_parcel_min_pct=small_parcel_min_pct,
        )

        base_preprocess = PreprocessConfig()
        preprocessing = PreprocessConfig(
            min_cluster_pixels=(
                min_cluster_pixels
                if min_cluster_pixels is not None
                else base_preprocess.min_cluster_pixels
            ),
            morph_open=(
                morph_open if morph_open is not None else base_preprocess.morph_open
            ),
        )

        profile = build_profile(
            crop_name=crop_name.strip().title(),
            cdl_crop_code=code,
            year=int(year),
            bbox_wgs84=bbox,
            classification_strategy=classification_strategy,
            thresholds=thresholds,
            preprocessing=preprocessing,
        )

        output_dir = OUTPUTS_ROOT / f"{_slugify(crop_name)}_{year}"
        result = run_pipeline(profile, output_dir)
        return _format_markdown_summary(result)

    except (ValueError, DataAccessError) as exc:
        return (
            f"## Crop footprint analysis failed\n\n"
            f"**Error:** {exc}\n\n"
            "Check crop name / CDL code, bbox order `[xmin, ymin, xmax, ymax]`, "
            "and year. Ensure network access to Planetary Computer, CropScape, "
            "and Source Cooperative."
        )
    except Exception as exc:
        logger.exception("Pipeline error")
        return f"## Crop footprint analysis failed\n\n**Unexpected error:** {exc}"


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
