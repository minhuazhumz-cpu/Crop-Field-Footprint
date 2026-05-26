"""CDL raster cleaning: crop mask, sieve, and optional morphological opening."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from rasterio.features import sieve
from scipy.ndimage import binary_opening, generate_binary_structure

if TYPE_CHECKING:
    from src.config_loader import PreprocessConfig

logger = logging.getLogger(__name__)


def build_crop_masks(
    cdl: np.ndarray,
    crop_code: int,
    preprocess: PreprocessConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build raw and cleaned binary crop masks from a CDL classification array.

    Sequence (matches legacy Yuma pipeline):
      1. Raw mask: CDL == crop_code
      2. Optional 3×3 morphological opening (--morph-open equivalent)
      3. rasterio.features.sieve (8-connected, min cluster size from profile)
    """
    raw = cdl == crop_code
    work = raw.copy()
    if preprocess.morph_open:
        structure = generate_binary_structure(2, 1)
        work = binary_opening(work, structure=structure)
    work_u8 = work.astype(np.uint8)
    cleaned_u8 = sieve(
        work_u8,
        size=preprocess.min_cluster_pixels,
        connectivity=8,
    )
    cleaned = cleaned_u8.astype(bool)

    raw_n = int(raw.sum())
    clean_n = int(cleaned.sum())
    removed_n = int((raw & ~cleaned).sum())
    logger.info(
        "Mask cleaning — raw=%s retained=%s removed=%s (%.1f%%) sieve>=%s px morph_open=%s",
        f"{raw_n:,}",
        f"{clean_n:,}",
        f"{removed_n:,}",
        100.0 * removed_n / raw_n if raw_n else 0.0,
        preprocess.min_cluster_pixels,
        preprocess.morph_open,
    )
    return raw, cleaned
