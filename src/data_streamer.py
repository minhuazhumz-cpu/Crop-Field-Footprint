"""Cloud-native data access: CDL routing (STAC vs CropScape) and DuckDB/FTW GeoParquet."""

from __future__ import annotations

import logging
import re
import time
from contextlib import ExitStack
from typing import TYPE_CHECKING, Literal

import duckdb
import geopandas as gpd
import numpy as np
import pystac_client
import rasterio
import requests
import rioxarray
from pyproj import Transformer
from rasterio.transform import Affine
from rioxarray import merge as rxr_merge
from shapely.geometry import box
from shapely.ops import transform as shp_transform

from src.config_loader import CDL_CRS, WGS84

if TYPE_CHECKING:
    from src.config_loader import AnalysisProfile, BboxWgs84

logger = logging.getLogger(__name__)

CdlSource = Literal["stac", "cropscape"]


class DataAccessError(RuntimeError):
    """Raised when cloud data cannot be retrieved or does not intersect the AOI."""


def _squeeze_to_int32_array(data: np.ndarray) -> np.ndarray:
    """Convert a 2D (or squeezed) raster band to a clean int32 numpy array."""
    if data.ndim != 2:
        raise DataAccessError(f"Expected 2D CDL array, got shape {data.shape}")

    if hasattr(data, "filled"):
        arr = data.filled(0)
    else:
        arr = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    if not np.any(arr):
        raise DataAccessError(
            "CDL array is empty — bbox may lie outside the returned extent."
        )
    return arr.astype(np.int32)


def _clip_to_wgs84_bbox(
    data_array: rioxarray.raster_array.RasterArray,
    bbox_wgs84: BboxWgs84,
) -> tuple[np.ndarray, Affine, str]:
    raster_crs = data_array.rio.crs
    if raster_crs is None:
        raise DataAccessError("CDL raster has no CRS.")

    if str(raster_crs) == WGS84:
        clipped = data_array.rio.clip_box(*bbox_wgs84)
    else:
        transformer = Transformer.from_crs(WGS84, raster_crs, always_xy=True)
        geom = shp_transform(
            lambda x, y, z=None: transformer.transform(x, y),
            box(*bbox_wgs84),
        )
        clipped = data_array.rio.clip([geom], all_touched=False, drop=False)

    arr = _squeeze_to_int32_array(clipped.values.squeeze())
    return arr, clipped.rio.transform(), str(clipped.rio.crs)


class CDLStreamer:
    """
    Intelligent CDL router.

    * ``year <= stac_max_year`` → Planetary Computer STAC + COG HTTP range reads
    * ``year > stac_max_year``  → USDA CropScape GetCDLFile (streamed in memory)
    """

    def __init__(self, profile: AnalysisProfile) -> None:
        self.profile = profile
        self._catalog: pystac_client.Client | None = None
        self.last_source: CdlSource | None = None

    @property
    def _stac_max_year(self) -> int:
        return self.profile.cdl.stac_max_year

    def fetch_cdl_array(self) -> tuple[np.ndarray, Affine, str]:
        """Return ``(cdl_array, transform, crs)`` for the profile AOI."""
        year = self.profile.year
        if year <= self._stac_max_year:
            logger.info(
                "CDL router: year %s ≤ %s → Planetary Computer STAC",
                year,
                self._stac_max_year,
            )
            self.last_source = "stac"
            return self._fetch_via_stac()

        logger.info(
            "CDL router: year %s > %s → USDA CropScape GetCDLFile",
            year,
            self._stac_max_year,
        )
        self.last_source = "cropscape"
        return self._fetch_via_cropscape()

    def _open_catalog(self) -> pystac_client.Client:
        if self._catalog is not None:
            return self._catalog

        modifier = None
        if self.profile.stac.sign_with_planetary_computer:
            try:
                import planetary_computer  # noqa: PLC0415

                modifier = planetary_computer.sign_inplace
            except ImportError as exc:
                raise DataAccessError(
                    "planetary-computer is required to sign Microsoft-hosted CDL COGs. "
                    "Install planetary-computer or set stac.sign_with_planetary_computer: false."
                ) from exc

        self._catalog = pystac_client.Client.open(
            self.profile.stac.api_url,
            modifier=modifier,
        )
        return self._catalog

    def _search_stac_items(self) -> list:
        catalog = self._open_catalog()
        year = self.profile.year
        datetime_range = f"{year}-01-01/{year}-12-31"
        bbox = list(self.profile.bbox_wgs84)

        logger.info(
            "STAC search — collection=%s year=%s bbox=%s type=%s",
            self.profile.stac.collection,
            year,
            bbox,
            self.profile.stac.item_type,
        )

        search = catalog.search(
            collections=[self.profile.stac.collection],
            datetime=datetime_range,
            bbox=bbox,
            query={"usda_cdl:type": {"eq": self.profile.stac.item_type}},
        )
        items = list(search.items())
        if items:
            return items

        raise DataAccessError(
            f"No '{self.profile.stac.item_type}' CDL STAC items for year {year} "
            f"intersecting bbox {bbox}. Verify year and bbox."
        )

    def _fetch_via_stac(self) -> tuple[np.ndarray, Affine, str]:
        """Merge STAC COG tiles and clip to the profile WGS84 bbox."""
        items = self._search_stac_items()
        asset_key = self.profile.stac.asset_key
        hrefs: list[str] = []

        for item in items:
            if asset_key not in item.assets:
                raise DataAccessError(
                    f"STAC item {item.id} missing asset '{asset_key}'. "
                    f"Available: {list(item.assets.keys())}"
                )
            hrefs.append(item.assets[asset_key].href)

        logger.info("Opening %s CDL COG tile(s) via HTTP range reads", len(hrefs))

        with ExitStack() as stack:
            arrays = [
                stack.enter_context(rioxarray.open_rasterio(href, masked=True))
                for href in hrefs
            ]
            mosaic = arrays[0] if len(arrays) == 1 else rxr_merge.merge_arrays(arrays)
            arr, transform, crs = _clip_to_wgs84_bbox(mosaic, self.profile.bbox_wgs84)
            logger.info("STAC CDL ready — shape=%s crs=%s", arr.shape, crs)
            return arr, transform, crs

    @staticmethod
    def _parse_cropscape_return_url(xml_text: str) -> str:
        match = re.search(r"<returnURL>([^<]+)</returnURL>", xml_text)
        if not match:
            raise DataAccessError(
                f"CropScape API did not return a download URL: {xml_text[:300]}"
            )
        return match.group(1)

    def _resolve_cropscape_geotiff_url(self) -> str:
        """Call GetCDLFile and return the GeoTIFF URL (no local persistence)."""
        cfg = self.profile.cdl
        xmin, ymin, xmax, ymax = self.profile.bbox_5070
        bbox_str = f"{xmin:.0f},{ymin:.0f},{xmax:.0f},{ymax:.0f}"
        params = {"year": self.profile.year, "bbox": bbox_str}

        logger.info(
            "CropScape GetCDLFile — year=%s bbox_5070=%s",
            self.profile.year,
            self.profile.bbox_5070,
        )

        last_err: Exception | None = None
        for attempt in range(1, cfg.retries + 1):
            try:
                resp = requests.get(
                    cfg.cropscape_api,
                    params=params,
                    timeout=cfg.timeout_s,
                )
                resp.raise_for_status()
                tif_url = self._parse_cropscape_return_url(resp.text)
                head = requests.head(tif_url, timeout=cfg.timeout_s, allow_redirects=True)
                if head.status_code >= 400:
                    raise DataAccessError(
                        f"CropScape GeoTIFF URL not reachable (HTTP {head.status_code})"
                    )
                logger.info("CropScape GeoTIFF URL resolved (streaming via GDAL)")
                return tif_url
            except DataAccessError:
                raise
            except requests.RequestException as exc:
                last_err = exc
                logger.warning(
                    "CropScape request attempt %s/%s failed: %s",
                    attempt,
                    cfg.retries,
                    exc,
                )
                if attempt < cfg.retries:
                    time.sleep(5 * attempt)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < cfg.retries:
                    time.sleep(5 * attempt)

        raise DataAccessError(
            f"CropScape GetCDLFile failed for year {self.profile.year}"
        ) from last_err

    def _fetch_via_cropscape(self) -> tuple[np.ndarray, Affine, str]:
        """
        Stream a server-clipped CDL GeoTIFF from CropScape directly into memory.

        GetCDLFile returns a GeoTIFF already clipped to the EPSG:5070 bbox; GDAL
        reads it over HTTP without writing to disk.
        """
        tif_url = self._resolve_cropscape_geotiff_url()
        try:
            with rasterio.Env(
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
            ):
                with rasterio.open(tif_url) as src:
                    data = src.read(1, masked=True)
                    transform = src.transform
                    crs = str(src.crs) if src.crs else CDL_CRS
                    arr = _squeeze_to_int32_array(data)
                    logger.info(
                        "CropScape CDL ready — shape=%s crs=%s",
                        arr.shape,
                        crs,
                    )
                    return arr, transform, crs
        except DataAccessError:
            raise
        except Exception as exc:
            raise DataAccessError(
                f"Failed to stream CropScape GeoTIFF: {exc}"
            ) from exc


class FTWStreamer:
    """Query global FTW GeoParquet on Source Cooperative via DuckDB (no local cache)."""

    def __init__(self, profile: AnalysisProfile) -> None:
        self.profile = profile

    @staticmethod
    def _configure_duckdb(con: duckdb.DuckDBPyConnection) -> None:
        con.execute("INSTALL spatial; LOAD spatial;")
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("SET s3_endpoint='data.source.coop';")
        con.execute("SET s3_url_style='path';")
        con.execute("SET s3_use_ssl=true;")

    def fetch_fields(self) -> gpd.GeoDataFrame:
        """Return FTW field polygons intersecting the profile WGS84 bbox."""
        xmin, ymin, xmax, ymax = self.profile.bbox_wgs84
        ftw = self.profile.ftw
        parquet = ftw.parquet_glob.replace("'", "''")

        logger.info(
            "DuckDB FTW query — label=%s bbox=%s",
            ftw.label,
            self.profile.bbox_wgs84,
        )

        sql = f"""
        SELECT
            row_number() OVER () AS ftw_field_id,
            geometry,
            time AS ftw_time
        FROM read_parquet('{parquet}')
        WHERE label = '{ftw.label}'
          AND struct_extract(bbox, 'xmax') >= {xmin}
          AND struct_extract(bbox, 'xmin') <= {xmax}
          AND struct_extract(bbox, 'ymax') >= {ymin}
          AND struct_extract(bbox, 'ymin') <= {ymax}
        """

        con = duckdb.connect()
        try:
            self._configure_duckdb(con)
            con.execute(f"SET s3_endpoint='{ftw.s3_endpoint}';")
            table = con.sql(sql).to_arrow_table()
            gdf = gpd.GeoDataFrame.from_arrow(table)
        except DataAccessError:
            raise
        except Exception as exc:
            raise DataAccessError(f"FTW DuckDB query failed: {exc}") from exc
        finally:
            con.close()

        if gdf.empty:
            raise DataAccessError(
                "FTW query returned zero field polygons for this bbox. "
                "Check bbox_wgs84 or FTW parquet path."
            )

        if gdf.crs is None:
            gdf = gdf.set_crs(WGS84)

        clip_geom = box(xmin, ymin, xmax, ymax)
        gdf = gdf[gdf.intersects(clip_geom)].copy()
        if gdf.empty:
            raise DataAccessError(
                "FTW geometries do not intersect the clip box after download."
            )

        if "ftw_field_id" not in gdf.columns:
            gdf["ftw_field_id"] = np.arange(1, len(gdf) + 1, dtype=np.int64)

        logger.info("FTW fields loaded — %s parcels in AOI", f"{len(gdf):,}")
        return gdf.reset_index(drop=True)
