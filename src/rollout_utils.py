from pathlib import Path
from typing import List, Optional, Union

import geopandas as gpd
import pandas as pd
import rasterio as rio
from geowrangler.validation import GeometryValidation
from loguru import logger
from pyproj import CRS
from rasterio import features
from shapely.geometry import shape


def polygonize_raster_mask(
    raster_mask_fpath: Union[str, Path],
    skip_labels: List[int],
    simplify_tolerance_m: Optional[float] = None,
) -> gpd.GeoDataFrame:
    raster_mask_fpath = Path(raster_mask_fpath)
    with rio.open(raster_mask_fpath) as src:
        raster_mask = src.read(indexes=1)
        raster_meta = src.meta
        raster_transform = src.transform

    # Generate a mask for skipped labels (e.g. for background)
    # entries in skip_mask that are False will not be polygonized
    skip_mask = True
    if skip_labels is not None:
        for label in skip_labels:
            skip_mask = skip_mask & (raster_mask != label)

    shape_gen = features.shapes(raster_mask, mask=skip_mask, transform=raster_transform)

    mask_polygons = ((shape(s), v) for s, v in shape_gen)
    mask_polygons = pd.DataFrame(mask_polygons, columns=["geometry", "label"])
    mask_polygons = gpd.GeoDataFrame(
        mask_polygons["label"],
        geometry=mask_polygons["geometry"],
        crs=raster_meta["crs"],
    )

    if mask_polygons.empty:
        return mask_polygons

    if simplify_tolerance_m is not None:
        crs_obj = CRS.from_user_input(raster_meta["crs"])
        if crs_obj.is_projected:
            mask_polygons["geometry"] = mask_polygons["geometry"].simplify(
                simplify_tolerance_m
            )
        else:
            raise ValueError(
                "CRS is not projected. Specify output_crs to be a projected CRS (ex. EPSG:3857)"
            )

    uid_prefix = raster_mask_fpath.stem
    mask_polygons["uid"] = [f"{uid_prefix}-{i}" for i in range(len(mask_polygons))]

    geom_validation = GeometryValidation(
        mask_polygons,
        apply_fixes=True,
        validators=("null", "self_intersecting", "orientation", "area"),
        add_validation_columns=False,
    )
    mask_polygons = geom_validation.validate_all()

    return mask_polygons
