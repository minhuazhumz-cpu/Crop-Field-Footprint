"""Programmatic entry point for the crop spatial intel analysis pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd

from src.analyzer import run_analysis
from src.config_loader import AnalysisProfile
from src.data_streamer import CDLStreamer, DataAccessError, FTWStreamer
from src.pixel_processor import build_crop_masks

logger = logging.getLogger(__name__)

CELL_AREA_ACRES = (30.0 * 30.0) / 4046.86


@dataclass(frozen=True)
class PipelineResult:
    profile: AnalysisProfile
    output_dir: Path
    cdl_source: str
    n_fields: int
    n_crop_fields: int
    cdl_unique_crop_acres: float
    crop_field_cdl_acres_sum: float
    csv_path: Path
    html_path: Path
    metadata_path: Path

    def to_summary_dict(self) -> dict:
        crop_label = f"{self.profile.crop_name} Field"
        return {
            "project_name": self.profile.project_name,
            "crop_name": self.profile.crop_name,
            "cdl_crop_code": self.profile.cdl_crop_code,
            "year": self.profile.year,
            "cdl_source": self.cdl_source,
            "bbox_wgs84": list(self.profile.bbox_wgs84),
            "classification_strategy": self.profile.classification_strategy,
            "n_fields": self.n_fields,
            "n_crop_fields": self.n_crop_fields,
            "cdl_unique_crop_acres": round(self.cdl_unique_crop_acres, 2),
            "crop_field_cdl_acres_sum": round(self.crop_field_cdl_acres_sum, 2),
            "outputs": {
                "csv": str(self.csv_path),
                "html": str(self.html_path),
                "metadata": str(self.metadata_path),
            },
        }


def run_pipeline(
    profile: AnalysisProfile,
    output_dir: Path,
) -> PipelineResult:
    """
    Execute the full CDL × FTW pipeline and write standard outputs.

    Raises :class:`DataAccessError` on network or spatial intersection failures.
    """
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cdl_streamer = CDLStreamer(profile)
    cdl_array, transform, raster_crs = cdl_streamer.fetch_cdl_array()

    _, cleaned_mask = build_crop_masks(
        cdl_array,
        profile.cdl_crop_code,
        profile.preprocessing,
    )
    crop_mask_u8 = cleaned_mask.astype("uint8")

    ftw_streamer = FTWStreamer(profile)
    fields = ftw_streamer.fetch_fields()

    gdf = run_analysis(
        fields,
        crop_mask_u8,
        transform,
        raster_crs,
        profile,
        output_dir,
        cdl_source=cdl_streamer.last_source,
    )

    crop_label = f"{profile.crop_name} Field"
    is_crop = gdf["coverage_category"] == crop_label
    cdl_unique = float(crop_mask_u8.sum()) * CELL_AREA_ACRES

    return PipelineResult(
        profile=profile,
        output_dir=output_dir,
        cdl_source=cdl_streamer.last_source or "unknown",
        n_fields=len(gdf),
        n_crop_fields=int(is_crop.sum()),
        cdl_unique_crop_acres=cdl_unique,
        crop_field_cdl_acres_sum=float(gdf.loc[is_crop, "crop_acres_in_field"].sum()),
        csv_path=output_dir / "insights_report.csv",
        html_path=output_dir / "coverage_map.html",
        metadata_path=output_dir / "run_metadata.json",
    )
