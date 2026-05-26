"""Load YAML analysis profiles and resolve spatial reference metadata."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pyproj import Transformer

logger = logging.getLogger(__name__)

BboxWgs84 = tuple[float, float, float, float]
BboxProjected = tuple[float, float, float, float]

CDL_CRS = "EPSG:5070"
WGS84 = "EPSG:4326"


@dataclass(frozen=True)
class PreprocessConfig:
    min_cluster_pixels: int = 4
    morph_open: bool = False


@dataclass(frozen=True)
class ThresholdConfig:
    min_crop_acres: float = 10.0
    min_crop_pct: float | None = None
    small_parcel_max_acres: float = 10.0
    small_parcel_min_crop_acres: float = 3.0
    small_parcel_min_pct: float | None = None


@dataclass(frozen=True)
class StacConfig:
    api_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1"
    collection: str = "usda-cdl"
    item_type: str = "cropland"
    asset_key: str = "cropland"
    sign_with_planetary_computer: bool = True


@dataclass(frozen=True)
class CdlConfig:
    """CDL source routing: STAC for early years, CropScape for newer releases."""

    stac_max_year: int = 2021
    cropscape_api: str = (
        "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile"
    )
    timeout_s: int = 600
    retries: int = 3


@dataclass(frozen=True)
class FtwConfig:
    parquet_glob: str = (
        "s3://ftw/global-data/predictions/vectors/alpha/results/*.parquet"
    )
    s3_endpoint: str = "data.source.coop"
    label: str = "field"


@dataclass(frozen=True)
class AnalysisProfile:
    project_name: str
    crop_name: str
    cdl_crop_code: int
    year: int
    bbox_wgs84: BboxWgs84
    preprocessing: PreprocessConfig
    classification_strategy: str
    thresholds: ThresholdConfig
    stac: StacConfig = field(default_factory=StacConfig)
    cdl: CdlConfig = field(default_factory=CdlConfig)
    ftw: FtwConfig = field(default_factory=FtwConfig)

    @property
    def bbox_5070(self) -> BboxProjected:
        """AOI corners in CDL Albers (EPSG:5070), derived from WGS84 bbox."""
        return wgs84_bbox_to_5070(self.bbox_wgs84)

    @property
    def map_center(self) -> tuple[float, float]:
        """Folium map center as (lat, lon)."""
        xmin, ymin, xmax, ymax = self.bbox_wgs84
        return ((ymin + ymax) / 2.0, (xmin + xmax) / 2.0)


def wgs84_bbox_to_5070(bbox_wgs84: BboxWgs84) -> BboxProjected:
    transformer = Transformer.from_crs(WGS84, CDL_CRS, always_xy=True)
    xmin, ymin, xmax, ymax = bbox_wgs84
    corners = [
        transformer.transform(xmin, ymin),
        transformer.transform(xmax, ymin),
        transformer.transform(xmin, ymax),
        transformer.transform(xmax, ymax),
    ]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def _require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Profile must define a '{key}' mapping.")
    return value


def build_profile(
    *,
    crop_name: str,
    cdl_crop_code: int,
    year: int,
    bbox_wgs84: BboxWgs84,
    project_name: str | None = None,
    classification_strategy: str = "hybrid_dynamic",
    thresholds: ThresholdConfig | None = None,
    preprocessing: PreprocessConfig | None = None,
    stac: StacConfig | None = None,
    cdl: CdlConfig | None = None,
    ftw: FtwConfig | None = None,
) -> AnalysisProfile:
    """Construct an :class:`AnalysisProfile` without a YAML file."""
    return AnalysisProfile(
        project_name=project_name or f"{crop_name} Spatial Overlap",
        crop_name=crop_name,
        cdl_crop_code=cdl_crop_code,
        year=year,
        bbox_wgs84=bbox_wgs84,
        preprocessing=preprocessing or PreprocessConfig(),
        classification_strategy=classification_strategy,
        thresholds=thresholds or ThresholdConfig(),
        stac=stac or StacConfig(),
        cdl=cdl or CdlConfig(),
        ftw=ftw or FtwConfig(),
    )


def load_profile(path: str | Path) -> AnalysisProfile:
    """Parse a YAML profile into a typed :class:`AnalysisProfile`."""
    profile_path = Path(path).expanduser().resolve()
    if not profile_path.is_file():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    with profile_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    bbox = raw.get("bbox_wgs84")
    if not bbox or len(bbox) != 4:
        raise ValueError("bbox_wgs84 must be [xmin, ymin, xmax, ymax] in WGS84 degrees.")
    bbox_wgs84: BboxWgs84 = tuple(float(v) for v in bbox)
    if bbox_wgs84[0] >= bbox_wgs84[2] or bbox_wgs84[1] >= bbox_wgs84[3]:
        raise ValueError(f"Invalid bbox_wgs84: {bbox_wgs84}")

    preprocess_raw = _require_mapping(raw, "preprocessing")
    thresholds_raw = _require_mapping(raw, "thresholds")

    stac_raw = raw.get("stac") or {}
    cdl_raw = raw.get("cdl") or {}
    ftw_raw = raw.get("ftw") or {}

    profile = AnalysisProfile(
        project_name=str(raw.get("project_name", "Crop Spatial Analysis")),
        crop_name=str(raw.get("crop_name", "Crop")),
        cdl_crop_code=int(raw["cdl_crop_code"]),
        year=int(raw["year"]),
        bbox_wgs84=bbox_wgs84,
        preprocessing=PreprocessConfig(
            min_cluster_pixels=int(preprocess_raw.get("min_cluster_pixels", 4)),
            morph_open=bool(preprocess_raw.get("morph_open", False)),
        ),
        classification_strategy=str(
            raw.get("classification_strategy", "hybrid_dynamic")
        ),
        thresholds=ThresholdConfig(
            min_crop_acres=float(thresholds_raw.get("min_crop_acres", 10.0)),
            min_crop_pct=(
                float(thresholds_raw["min_crop_pct"])
                if thresholds_raw.get("min_crop_pct") is not None
                else None
            ),
            small_parcel_max_acres=float(
                thresholds_raw.get("small_parcel_max_acres", 10.0)
            ),
            small_parcel_min_crop_acres=float(
                thresholds_raw.get("small_parcel_min_crop_acres", 3.0)
            ),
            small_parcel_min_pct=(
                float(thresholds_raw["small_parcel_min_pct"])
                if thresholds_raw.get("small_parcel_min_pct") is not None
                else None
            ),
        ),
        stac=StacConfig(
            api_url=str(
                stac_raw.get(
                    "api_url",
                    "https://planetarycomputer.microsoft.com/api/stac/v1",
                )
            ),
            collection=str(stac_raw.get("collection", "usda-cdl")),
            item_type=str(stac_raw.get("item_type", "cropland")),
            asset_key=str(stac_raw.get("asset_key", "cropland")),
            sign_with_planetary_computer=bool(
                stac_raw.get("sign_with_planetary_computer", True)
            ),
        ),
        cdl=CdlConfig(
            stac_max_year=int(cdl_raw.get("stac_max_year", 2021)),
            cropscape_api=str(
                cdl_raw.get(
                    "cropscape_api",
                    "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile",
                )
            ),
            timeout_s=int(cdl_raw.get("timeout_s", 600)),
            retries=int(cdl_raw.get("retries", 3)),
        ),
        ftw=FtwConfig(
            parquet_glob=str(
                ftw_raw.get(
                    "parquet_glob",
                    "s3://ftw/global-data/predictions/vectors/alpha/results/*.parquet",
                )
            ),
            s3_endpoint=str(ftw_raw.get("s3_endpoint", "data.source.coop")),
            label=str(ftw_raw.get("label", "field")),
        ),
    )

    logger.info(
        "Loaded profile '%s' — %s (CDL %s), year %s, bbox WGS84 %s",
        profile.project_name,
        profile.crop_name,
        profile.cdl_crop_code,
        profile.year,
        profile.bbox_wgs84,
    )
    return profile
