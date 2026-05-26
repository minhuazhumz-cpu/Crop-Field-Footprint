"""Zonal statistics, flexible field classification, CSV and Folium map outputs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import folium
import geopandas as gpd
import numpy as np
from rasterio.transform import Affine
from rasterstats import zonal_stats
from shapely.geometry import box

from src.config_loader import AnalysisProfile, ThresholdConfig

logger = logging.getLogger(__name__)

CELL_AREA_M2 = 30.0 * 30.0
CELL_AREA_ACRES = CELL_AREA_M2 / 4046.86
CELL_AREA_HA = CELL_AREA_M2 / 10_000.0

CATEGORY_CROP = "Crop Field"
CATEGORY_LOW = "Low/Trace"

CATEGORY_COLORS = {
    CATEGORY_CROP: "#2ca25f",
    CATEGORY_LOW: "#e31a1c",
}

MAX_INLINE_GEOJSON_MB = 6.0
DEFAULT_TILE_GRID = (4, 4)

CSV_COLUMNS = [
    "ftw_field_id",
    "area_acres",
    "crop_acres_in_field",
    "pct_of_polygon",
    "pixel_purity_pct",
    "coverage_category",
    "classification_rule",
]


def align_vectors_to_raster(
    gdf: gpd.GeoDataFrame,
    raster_crs: str,
) -> gpd.GeoDataFrame:
    if str(gdf.crs) != str(raster_crs):
        return gdf.to_crs(raster_crs)
    return gdf


def run_zonal_stats(
    gdf: gpd.GeoDataFrame,
    crop_mask_u8: np.ndarray,
    transform: Affine,
    *,
    nodata: int = 0,
) -> gpd.GeoDataFrame:
    """Append zonal sum/count and crop acreage metrics per FTW parcel."""
    stats = zonal_stats(
        gdf.geometry,
        crop_mask_u8,
        affine=transform,
        stats=["sum", "count"],
        nodata=nodata,
        all_touched=False,
    )
    out = gdf.copy()
    out["crop_sum"] = [s.get("sum") or 0 for s in stats]
    out["pixel_count"] = [s.get("count") or 0 for s in stats]
    out["pixel_purity_pct"] = np.where(
        out["pixel_count"] > 0,
        100.0 * out["crop_sum"] / out["pixel_count"],
        0.0,
    )
    out["area_m2"] = out.geometry.area
    out["area_ha"] = (out["area_m2"] / 10_000).round(3)
    out["area_acres"] = (out["area_m2"] / 4046.86).round(2)
    out["crop_ha_in_field"] = (out["crop_sum"] * CELL_AREA_HA).round(4)
    out["crop_acres_in_field"] = (out["crop_sum"] * CELL_AREA_ACRES).round(4)
    out["pct_of_polygon"] = np.where(
        out["area_acres"] > 0,
        100.0 * out["crop_acres_in_field"] / out["area_acres"],
        0.0,
    ).round(2)
    return out


def apply_classification(
    gdf: gpd.GeoDataFrame,
    profile: AnalysisProfile,
) -> gpd.GeoDataFrame:
    """Classify parcels using the strategy and thresholds from the profile."""
    strategy = profile.classification_strategy
    thresholds = profile.thresholds
    crop_label = profile.crop_name

    out = gdf.copy()
    crop_acres = out["crop_acres_in_field"]
    area_acres = out["area_acres"]
    pct = out["pct_of_polygon"]

    if strategy in ("hybrid_dynamic", "hybrid"):
        is_crop, rule = _classify_hybrid(
            crop_acres.to_numpy(),
            area_acres.to_numpy(),
            pct.to_numpy(),
            thresholds,
            crop_label,
        )
    elif strategy in ("min_acres", "acreage"):
        is_crop, rule = _classify_min_acres(
            crop_acres.to_numpy(), thresholds, crop_label
        )
    elif strategy in ("pct", "purity", "coverage_pct"):
        is_crop, rule = _classify_pct(pct.to_numpy(), thresholds, crop_label)
    else:
        raise ValueError(
            f"Unknown classification_strategy '{strategy}'. "
            "Use hybrid_dynamic, min_acres, or pct."
        )

    out["classification_rule"] = rule
    out["coverage_category"] = np.where(
        is_crop,
        f"{crop_label} Field",
        CATEGORY_LOW,
    )
    return out


def _classify_hybrid(
    crop_acres: np.ndarray,
    area_acres: np.ndarray,
    pct: np.ndarray,
    thresholds: ThresholdConfig,
    crop_label: str,
) -> tuple[np.ndarray, str]:
    large_field = crop_acres >= thresholds.min_crop_acres
    small_mask = area_acres < thresholds.small_parcel_max_acres
    small_dense = small_mask & (crop_acres >= thresholds.small_parcel_min_crop_acres)

    if thresholds.small_parcel_min_pct is not None:
        small_dense = small_dense | (
            small_mask & (pct >= thresholds.small_parcel_min_pct)
        )

    is_crop = large_field | small_dense
    rule = (
        f"{crop_label}: crop_acres >= {thresholds.min_crop_acres:g} ac OR "
        f"(parcel < {thresholds.small_parcel_max_acres:g} ac AND "
        f"crop_acres >= {thresholds.small_parcel_min_crop_acres:g} ac)"
    )
    if thresholds.small_parcel_min_pct is not None:
        rule += f", or pct >= {thresholds.small_parcel_min_pct:g}% on small parcels"
    return is_crop, rule


def _classify_min_acres(
    crop_acres: np.ndarray,
    thresholds: ThresholdConfig,
    crop_label: str,
) -> tuple[np.ndarray, str]:
    is_crop = crop_acres >= thresholds.min_crop_acres
    rule = f"{crop_label}: crop_acres_in_field >= {thresholds.min_crop_acres:g} ac"
    return is_crop, rule


def _classify_pct(
    pct: np.ndarray,
    thresholds: ThresholdConfig,
    crop_label: str,
) -> tuple[np.ndarray, str]:
    min_pct = thresholds.min_crop_pct
    if min_pct is None:
        raise ValueError(
            "classification_strategy 'pct' requires thresholds.min_crop_pct in the profile."
        )
    is_crop = pct > min_pct
    rule = f"{crop_label}: pct_of_polygon > {min_pct:g}%"
    return is_crop, rule


def export_insights_csv(gdf: gpd.GeoDataFrame, path: Path) -> None:
    """Write a lightweight tabular report without geometry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = gdf[CSV_COLUMNS].copy()
    table.to_csv(path, index=False)
    logger.info("Wrote insights report: %s (%s rows)", path, len(table))


def _style_function(feature: dict, crop_field_label: str) -> dict:
    cat = feature["properties"].get("coverage_category", CATEGORY_LOW)
    color = CATEGORY_COLORS.get(CATEGORY_CROP if crop_field_label in cat else CATEGORY_LOW)
    if crop_field_label in cat:
        color = CATEGORY_COLORS[CATEGORY_CROP]
        fill_opacity = 0.65
    else:
        color = CATEGORY_COLORS[CATEGORY_LOW]
        fill_opacity = 0.25
    return {
        "fillColor": color,
        "color": "#333333",
        "weight": 1,
        "fillOpacity": fill_opacity,
    }


def _prepare_map_geojson(gdf_wgs84: gpd.GeoDataFrame, profile: AnalysisProfile) -> gpd.GeoDataFrame:
    map_cols = [
        "ftw_field_id",
        "area_acres",
        "crop_acres_in_field",
        "pct_of_polygon",
        "pixel_purity_pct",
        "coverage_category",
        "classification_rule",
        "geometry",
    ]
    geojson = gdf_wgs84[map_cols].copy()
    for col in ("pct_of_polygon", "pixel_purity_pct", "crop_acres_in_field"):
        geojson[col] = geojson[col].round(2)
    return geojson


def _estimate_geojson_bytes(gdf_wgs84: gpd.GeoDataFrame, profile: AnalysisProfile) -> int:
    return len(_prepare_map_geojson(gdf_wgs84, profile).to_json().encode("utf-8"))


def export_geojson_tiles(
    gdf_wgs84: gpd.GeoDataFrame,
    *,
    tile_dir: Path,
    out_html: Path,
    n_rows: int,
    n_cols: int,
    profile: AnalysisProfile,
) -> dict:
    """
    Split field polygons into a fixed grid and write GeoJSON tiles + manifest.

    This keeps the Folium HTML small by loading features on-demand in the browser.
    """
    tile_dir.mkdir(parents=True, exist_ok=True)
    geojson = _prepare_map_geojson(gdf_wgs84, profile)
    xmin, ymin, xmax, ymax = geojson.total_bounds

    lon_width = xmax - xmin or 1e-9
    lat_height = ymax - ymin or 1e-9

    # Assign tiles in projected CRS to avoid skew from lon/lat degrees.
    # Choose a reasonable default UTM zone from bbox center longitude.
    center_lon = (xmin + xmax) / 2.0
    utm_zone = int((center_lon + 180) // 6) + 1
    utm_epsg = 32600 + utm_zone
    geo_utm = geojson.to_crs(f"EPSG:{utm_epsg}")
    uxmin, uymin, uxmax, uymax = geo_utm.total_bounds
    utm_width = uxmax - uxmin or 1e-9
    utm_height = uymax - uymin or 1e-9
    cent = geo_utm.geometry.centroid

    col_idx = np.clip(((cent.x - uxmin) / utm_width * n_cols).astype(int), 0, n_cols - 1)
    row_idx = np.clip(((cent.y - uymin) / utm_height * n_rows).astype(int), 0, n_rows - 1)

    tiles: list[dict] = []
    for r in range(n_rows):
        for c in range(n_cols):
            mask = (row_idx == r) & (col_idx == c)
            subset = geojson.loc[mask]
            fname = f"tile_r{r}_c{c}.geojson"
            tile_path = tile_dir / fname
            if len(subset) == 0:
                tile_path.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
            else:
                subset.to_file(tile_path, driver="GeoJSON")

            # Manifest bounds must be WGS84 [[lat, lon], [lat, lon]] for Leaflet.
            ty0 = ymin + lat_height * r / n_rows
            ty1 = ymin + lat_height * (r + 1) / n_rows
            tx0 = xmin + lon_width * c / n_cols
            tx1 = xmin + lon_width * (c + 1) / n_cols
            tiles.append(
                {
                    "id": f"r{r}_c{c}",
                    "url": (tile_dir / fname).relative_to(out_html.parent).as_posix(),
                    "bounds": [[ty0, tx0], [ty1, tx1]],
                    "n_features": int(len(subset)),
                }
            )

    manifest = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "tiles": tiles,
        "manifest_url": (tile_dir / "manifest.json").relative_to(out_html.parent).as_posix(),
    }
    (tile_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _tile_loader_js(map_var: str, field_group_var: str, manifest_url: str, crop_field_label: str) -> str:
    """Leaflet JS: load GeoJSON tiles intersecting the viewport into a layer group."""
    broccoli_color = CATEGORY_COLORS[CATEGORY_CROP]
    low_color = CATEGORY_COLORS[CATEGORY_LOW]
    return f"""
    (function() {{
        var manifestUrl = {json.dumps(manifest_url)};
        var loadedTiles = {{}};
        var ftwLayer = null;

        function boot() {{
            if (typeof {map_var} === "undefined" || typeof {field_group_var} === "undefined") {{
                setTimeout(boot, 10);
                return;
            }}
            var map = {map_var};
            var fieldGroup = {field_group_var};
            if (ftwLayer) return;

            ftwLayer = L.geoJSON(null, {{
                style: function(feature) {{
                    var cat = (feature.properties || {{}}).coverage_category || "";
                    var isCrop = cat.indexOf({json.dumps(crop_field_label)}) !== -1;
                    return {{
                        color: "#333333",
                        fillColor: isCrop ? {json.dumps(broccoli_color)} : {json.dumps(low_color)},
                        fillOpacity: isCrop ? 0.65 : 0.25,
                        weight: 1
                    }};
                }},
                onEachFeature: function(feature, layer) {{
                    fieldGroup.addLayer(layer);
                    var p = feature.properties || {{}};
                    layer.bindTooltip(
                        "<table>" +
                        "<tr><th>FTW Field ID</th><td>" + p.ftw_field_id + "</td></tr>" +
                        "<tr><th>Parcel area (acres)</th><td>" + p.area_acres + "</td></tr>" +
                        "<tr><th>CDL crop in parcel (acres)</th><td>" + p.crop_acres_in_field + "</td></tr>" +
                        "<tr><th>% of parcel area</th><td>" + p.pct_of_polygon + "</td></tr>" +
                        "<tr><th>Category</th><td>" + p.coverage_category + "</td></tr>" +
                        "</table>",
                        {{sticky: true}}
                    );
                }}
            }});

            function tileBounds(tile) {{
                return L.latLngBounds(tile.bounds[0], tile.bounds[1]);
            }}

            function loadVisibleTiles(manifest) {{
                var view = map.getBounds().pad(0.08);
                manifest.tiles.forEach(function(tile) {{
                    if (loadedTiles[tile.id]) return;
                    if (!view.intersects(tileBounds(tile))) return;
                    loadedTiles[tile.id] = "loading";
                    fetch(tile.url)
                        .then(function(resp) {{ return resp.json(); }})
                        .then(function(data) {{
                            ftwLayer.addData(data);
                            loadedTiles[tile.id] = "done";
                        }})
                        .catch(function(err) {{
                            console.error("Failed to load tile", tile.id, err);
                            delete loadedTiles[tile.id];
                        }});
                }});
            }}

            fetch(manifestUrl)
                .then(function(resp) {{ return resp.json(); }})
                .then(function(manifest) {{
                    map.whenReady(function() {{ loadVisibleTiles(manifest); }});
                    map.on("moveend zoomend", function() {{ loadVisibleTiles(manifest); }});
                }})
                .catch(function(err) {{
                    console.error("Failed to load map tile manifest", err);
                }});
        }}

        boot();
    }})();
    """.strip()


def _append_tile_loader_script(out_html: Path, js: str) -> None:
    text = out_html.read_text(encoding="utf-8")
    marker = "<!-- crop-field-footprint-tile-loader -->"
    if marker in text:
        start = text.index(marker)
        end = text.index(marker, start + len(marker)) + len(marker)
        text = text[:start] + text[end:]
    block = f"\n{marker}\n<script>\n{js}\n</script>\n{marker}\n"
    if "</html>" in text:
        text = text.replace("</html>", block + "</html>", 1)
    else:
        text = text + block
    out_html.write_text(text, encoding="utf-8")


def build_folium_map(
    gdf_wgs84: gpd.GeoDataFrame,
    profile: AnalysisProfile,
    out_html: Path,
) -> None:
    """Interactive map with sticky hover tooltips colored by coverage category."""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    lat, lon = profile.map_center
    xmin, ymin, xmax, ymax = profile.bbox_wgs84

    m = folium.Map(
        location=[lat, lon],
        zoom_start=11,
        tiles="CartoDB positron",
        attr="CartoDB",
    )
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    folium.Rectangle(
        bounds=[[ymin, xmin], [ymax, xmax]],
        color="#984ea3",
        weight=2,
        fill=False,
        dash_array="6",
        tooltip="Analysis bounding box",
    ).add_to(m)

    crop_field_label = f"{profile.crop_name} Field"

    embed_bytes = _estimate_geojson_bytes(gdf_wgs84, profile)
    use_tiles = embed_bytes > MAX_INLINE_GEOJSON_MB * 1024 * 1024

    tile_loader_js: str | None = None
    if use_tiles:
        tile_dir = out_html.parent / "map_tiles" / out_html.stem
        n_rows, n_cols = DEFAULT_TILE_GRID
        manifest = export_geojson_tiles(
            gdf_wgs84,
            tile_dir=tile_dir,
            out_html=out_html,
            n_rows=n_rows,
            n_cols=n_cols,
            profile=profile,
        )
        field_group = folium.FeatureGroup(
            name=f"FTW fields ({profile.crop_name} coverage)",
            show=True,
        )
        field_group.add_to(m)
        tile_loader_js = _tile_loader_js(
            m.get_name(),
            field_group.get_name(),
            manifest["manifest_url"],
            crop_field_label,
        )
        logger.info(
            "Externalizing GeoJSON into tiles: %s (%.1f MB inline estimate)",
            tile_dir,
            embed_bytes / 1e6,
        )
    else:
        geojson = _prepare_map_geojson(gdf_wgs84, profile)
        folium.GeoJson(
            geojson,
            name=f"FTW fields ({profile.crop_name} coverage)",
            style_function=lambda feat: _style_function(feat, crop_field_label),
            tooltip=folium.GeoJsonTooltip(
                fields=[
                    "ftw_field_id",
                    "area_acres",
                    "crop_acres_in_field",
                    "pct_of_polygon",
                    "coverage_category",
                ],
                aliases=[
                    "FTW Field ID",
                    "Parcel area (acres)",
                    f"CDL {profile.crop_name} in parcel (acres)",
                    "% of parcel area",
                    "Category",
                ],
                localize=True,
                sticky=True,
            ),
        ).add_to(m)

    rule = ""
    if "classification_rule" in gdf_wgs84.columns and len(gdf_wgs84):
        rule = str(gdf_wgs84["classification_rule"].iloc[0])
    legend_html = f"""
    <div style="position: fixed; bottom: 28px; left: 28px; z-index: 9999;
                background: white; padding: 10px 12px; border: 2px solid grey;
                border-radius: 6px; font-size: 13px;">
      <b>{profile.project_name}</b><br>
      CDL {profile.year} — {profile.crop_name} (code {profile.cdl_crop_code})<br>
      <span style="color:#2ca25f;">&#9632;</span> {crop_field_label}<br>
      <span style="color:#e31a1c;">&#9632;</span> {CATEGORY_LOW}<br>
      <small>{rule}</small>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)
    m.save(str(out_html))
    if tile_loader_js:
        _append_tile_loader_script(out_html, tile_loader_js)
    logger.info("Wrote Folium map: %s (%.2f MB)", out_html, out_html.stat().st_size / 1e6)


def write_run_metadata(
    gdf: gpd.GeoDataFrame,
    profile: AnalysisProfile,
    *,
    raster_crs: str,
    unique_crop_acres: float,
    output_dir: Path,
    cdl_source: str | None = None,
) -> Path:
    crop_field_label = f"{profile.crop_name} Field"
    is_crop = gdf["coverage_category"] == crop_field_label
    meta = {
        "project_name": profile.project_name,
        "crop_name": profile.crop_name,
        "cdl_crop_code": profile.cdl_crop_code,
        "year": profile.year,
        "cdl_source": cdl_source,
        "cdl_stac_max_year": profile.cdl.stac_max_year,
        "aoi_wgs84": profile.bbox_wgs84,
        "raster_crs": raster_crs,
        "classification_strategy": profile.classification_strategy,
        "classification_rule": gdf["classification_rule"].iloc[0],
        "thresholds": {
            "min_crop_acres": profile.thresholds.min_crop_acres,
            "min_crop_pct": profile.thresholds.min_crop_pct,
            "small_parcel_max_acres": profile.thresholds.small_parcel_max_acres,
            "small_parcel_min_crop_acres": profile.thresholds.small_parcel_min_crop_acres,
            "small_parcel_min_pct": profile.thresholds.small_parcel_min_pct,
        },
        "n_fields": int(len(gdf)),
        "category_counts": {
            crop_field_label: int(is_crop.sum()),
            CATEGORY_LOW: int((~is_crop).sum()),
        },
        "cdl_unique_crop_acres": round(unique_crop_acres, 2),
        "preprocessing": {
            "min_cluster_pixels": profile.preprocessing.min_cluster_pixels,
            "morph_open": profile.preprocessing.morph_open,
        },
    }
    meta_path = output_dir / "run_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta_path


def run_analysis(
    gdf: gpd.GeoDataFrame,
    crop_mask_u8: np.ndarray,
    transform: Affine,
    raster_crs: str,
    profile: AnalysisProfile,
    output_dir: Path,
    *,
    cdl_source: str | None = None,
) -> gpd.GeoDataFrame:
    """Full vector analysis pipeline: zonal stats → classify → export."""
    output_dir.mkdir(parents=True, exist_ok=True)

    gdf = align_vectors_to_raster(gdf, raster_crs)
    logger.info("Running zonal statistics on %s fields...", f"{len(gdf):,}")
    gdf = run_zonal_stats(gdf, crop_mask_u8, transform)
    gdf = apply_classification(gdf, profile)

    unique_crop_acres = float(crop_mask_u8.sum()) * CELL_AREA_ACRES
    csv_path = output_dir / "insights_report.csv"
    export_insights_csv(gdf, csv_path)

    gdf_wgs84 = gdf.to_crs("EPSG:4326")
    html_path = output_dir / "coverage_map.html"
    build_folium_map(gdf_wgs84, profile, html_path)
    write_run_metadata(
        gdf,
        profile,
        raster_crs=raster_crs,
        unique_crop_acres=unique_crop_acres,
        output_dir=output_dir,
        cdl_source=cdl_source,
    )
    return gdf
