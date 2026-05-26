#!/usr/bin/env python3
"""
Crop Field Footprint — cloud-native CDL × FTW overlap analysis.

Unified CLI entry point. All source rasters and vectors are streamed from
public STAC COGs and Source Cooperative GeoParquet (no local downloads).
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from src.analyzer import run_analysis
from src.config_loader import load_profile
from src.data_streamer import CDLStreamer, DataAccessError, FTWStreamer
from src.pixel_processor import build_crop_masks

ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILE = ROOT / "config" / "profiles" / "yuma_broccoli.yaml"


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream USDA CDL and FTW data from the cloud, clean crop pixels, "
            "run zonal statistics, classify fields, and export CSV + HTML map."
        ),
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE,
        help=f"YAML profile path (default: {DEFAULT_PROFILE})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output",
        help="Directory for insights_report.csv, coverage_map.html, metadata",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Override CDL year from the profile (useful when STAC lags NASS releases)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    logger = logging.getLogger("crop_spatial_intel")

    try:
        profile = load_profile(args.profile)
        if args.year is not None:
            profile = replace(profile, year=args.year)
            logger.info("CDL year overridden to %s", args.year)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    output_dir = args.output_dir.resolve()
    logger.info("Output directory: %s", output_dir)

    try:
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

        run_analysis(
            fields,
            crop_mask_u8,
            transform,
            raster_crs,
            profile,
            output_dir,
            cdl_source=cdl_streamer.last_source,
        )
    except DataAccessError as exc:
        logger.error("Data access failed: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 1

    logger.info("Pipeline complete. Open %s/coverage_map.html", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
