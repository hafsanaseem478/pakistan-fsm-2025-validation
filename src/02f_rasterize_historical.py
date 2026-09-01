from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, bounds as window_bounds
from shapely.geometry import box


# PATHS
RAW = Path("data/raw")
INTERIM = Path("data/interim")

REF = RAW / "fsm_lgbm_pakistan.tif"

INTERIM.mkdir(parents=True, exist_ok=True)

BLOCK = 4096


# READ REFERENCE FSM GRID
with rasterio.open(REF) as ref:
    meta = ref.meta.copy()
    full_transform = ref.transform
    crs = ref.crs
    H = ref.height
    W = ref.width

    print("=" * 70)
    print("REFERENCE FSM")
    print("=" * 70)
    print(f"CRS: {crs}")
    print(f"Size: {W:,} x {H:,}")
    print(f"Resolution: {ref.res}")
    print(f"Bounds: {ref.bounds}")


meta.update(
    driver="GTiff",
    dtype="uint8",
    count=1,
    nodata=255,
    compress="lzw",
    tiled=True,
    blockxsize=256,
    blockysize=256,
    BIGTIFF="YES"
)


# LOAD ONE VECTOR LAYER
def load_layer(source, target_crs, layer=None):

    if layer is None:
        gdf = gpd.read_file(source)
    else:
        gdf = gpd.read_file(source, layer=layer)

    if gdf.crs is None:
        raise ValueError(f"No CRS found: {source} / {layer}")

    # Remove null/empty features
    gdf = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
    ].copy()

    # Reproject onto FSM CRS
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    return gdf


# RASTERIZE USING SPATIAL INDEX
def rasterize_windowed(gdf, outpath):

    print(f"Features: {len(gdf):,}")

    if len(gdf) == 0:
        raise ValueError(f"No valid geometries for {outpath}")

    # Spatial index
    sindex = gdf.sindex

    total_ones = 0
    blocks_done = 0

    total_blocks = (
        ((H + BLOCK - 1) // BLOCK)
        *
        ((W + BLOCK - 1) // BLOCK)
    )

    with rasterio.open(outpath, "w", **meta) as dst:

        for row_start in range(0, H, BLOCK):

            row_size = min(
                BLOCK,
                H - row_start
            )

            for col_start in range(0, W, BLOCK):

                col_size = min(
                    BLOCK,
                    W - col_start
                )

                window = Window(
                    col_off=col_start,
                    row_off=row_start,
                    width=col_size,
                    height=row_size
                )

                win_transform = rasterio.windows.transform(
                    window,
                    full_transform
                )

                # Bounds of current raster block
                left, bottom, right, top = window_bounds(
                    window,
                    full_transform
                )

                block_geom = box(
                    left,
                    bottom,
                    right,
                    top
                )

                # Find only polygons whose bounding boxes
                # intersect the current block
                candidate_ids = list(
                    sindex.intersection(
                        (left, bottom, right, top)
                    )
                )

                if candidate_ids:

                    candidates = gdf.geometry.iloc[
                        candidate_ids
                    ]

                    # Exact intersection check
                    candidates = [
                        geom
                        for geom in candidates
                        if geom.intersects(block_geom)
                    ]

                else:
                    candidates = []

                if candidates:

                    block = rasterize(
                        [
                            (geom, 1)
                            for geom in candidates
                        ],
                        out_shape=(
                            int(row_size),
                            int(col_size)
                        ),
                        transform=win_transform,
                        fill=0,
                        default_value=1,
                        dtype="uint8",
                        all_touched=False
                    )

                else:

                    block = np.zeros(
                        (
                            int(row_size),
                            int(col_size)
                        ),
                        dtype="uint8"
                    )

                dst.write(
                    block,
                    1,
                    window=window
                )

                total_ones += int(
                    np.count_nonzero(block == 1)
                )

                blocks_done += 1

                if (
                    blocks_done % 20 == 0
                    or blocks_done == total_blocks
                ):
                    print(
                        f"  {blocks_done:,}/"
                        f"{total_blocks:,} blocks completed"
                    )

    print(f"Written: {outpath}")
    print(f"Pixels = 1: {total_ones:,}")

    return total_ones


# 2010
print("\n" + "=" * 70)
print("2010 CUMULATIVE FLOOD")
print("=" * 70)

gdf_2010 = load_layer(
    RAW / "FL20100802PAK.gdb",
    crs,
    layer="Cumulative_FloodExtent"
)

rasterize_windowed(
    gdf_2010,
    INTERIM / "flood_2010.tif"
)


# 2014
print("\n" + "=" * 70)
print("2014 FLOOD")
print("=" * 70)

flood_layers_2014 = [
    "TX_20140915_Flood",
    "TX_20140916_Flood",
    "SN1_20140916_Flood",
    "TSX_20140916_Flood",
    "LS_20140910_Flood",
]

gdfs_2014 = []

for layer in flood_layers_2014:

    print(f"Loading: {layer}")

    temp = load_layer(
        RAW / "FL20140910PAK.gdb",
        crs,
        layer=layer
    )

    print(f"  Features: {len(temp):,}")

    gdfs_2014.append(temp[["geometry"]])


# Combine all 2014 flood sensors
gdf_2014 = gpd.GeoDataFrame(
    pd.concat(
        gdfs_2014,
        ignore_index=True
    ),
    geometry="geometry",
    crs=crs
)

print(
    f"Total 2014 flood features: "
    f"{len(gdf_2014):,}"
)

rasterize_windowed(
    gdf_2014,
    INTERIM / "flood_2014.tif"
)


# 2022
print("\n" + "=" * 70)
print("2022 CUMULATIVE FLOOD")
print("=" * 70)

gdf_2022 = load_layer(
    RAW / "VIIRS_20220701_20220831_FloodExtent_PAK.shp",
    crs
)

rasterize_windowed(
    gdf_2022,
    INTERIM / "flood_2022.tif"
)


# CREATE TRAINING INVENTORY MASK
# 2010 OR 2014 OR 2022
print("\n" + "=" * 70)
print("CREATING TRAINING INVENTORY MASK")
print("=" * 70)

paths = [
    INTERIM / "flood_2010.tif",
    INTERIM / "flood_2014.tif",
    INTERIM / "flood_2022.tif",
]

with (
    rasterio.open(paths[0]) as src2010,
    rasterio.open(paths[1]) as src2014,
    rasterio.open(paths[2]) as src2022,
    rasterio.open(
        INTERIM / "training_inventory_mask.tif",
        "w",
        **meta
    ) as dst
):

    total_training = 0
    blocks_done = 0

    total_blocks = (
        ((H + BLOCK - 1) // BLOCK)
        *
        ((W + BLOCK - 1) // BLOCK)
    )

    for row_start in range(0, H, BLOCK):

        row_size = min(
            BLOCK,
            H - row_start
        )

        for col_start in range(0, W, BLOCK):

            col_size = min(
                BLOCK,
                W - col_start
            )

            window = Window(
                col_off=col_start,
                row_off=row_start,
                width=col_size,
                height=row_size
            )

            a = src2010.read(
                1,
                window=window
            )

            b = src2014.read(
                1,
                window=window
            )

            c = src2022.read(
                1,
                window=window
            )

            # Union
            combined = (
                (a == 1)
                | (b == 1)
                | (c == 1)
            ).astype("uint8")

            dst.write(
                combined,
                1,
                window=window
            )

            total_training += int(
                np.count_nonzero(
                    combined == 1
                )
            )

            blocks_done += 1

            if (
                blocks_done % 20 == 0
                or blocks_done == total_blocks
            ):
                print(
                    f"  {blocks_done:,}/"
                    f"{total_blocks:,} blocks completed"
                )


print("\n" + "=" * 70)
print("HISTORICAL TRAINING INVENTORY COMPLETE")
print("=" * 70)

print(
    f"Training inventory pixels: "
    f"{total_training:,}"
)

print("\nCreated:")
print("  data/interim/flood_2010.tif")
print("  data/interim/flood_2014.tif")
print("  data/interim/flood_2022.tif")
print("  data/interim/training_inventory_mask.tif")

print("\nDone.")