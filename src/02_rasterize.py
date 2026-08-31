from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
RAW = Path("data/raw/FL20250818PAK_SHP")
INTERIM = Path("data/interim")
REF = Path("data/raw/fsm_lgbm_pakistan.tif")

INTERIM.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Processing settings
# ---------------------------------------------------------
BLOCK = 4096


# ---------------------------------------------------------
# Read reference FSM grid
# ---------------------------------------------------------
with rasterio.open(REF) as ref:
    meta = ref.meta.copy()
    full_transform = ref.transform
    crs = ref.crs
    H = ref.height
    W = ref.width

    print("Reference FSM")
    print(f"CRS: {crs}")
    print(f"Size: {W:,} x {H:,}")
    print(f"Resolution: {ref.res}")
    print(f"Bounds: {ref.bounds}")


# Output raster settings
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


# ---------------------------------------------------------
# Vector layers to rasterize
# ---------------------------------------------------------
layers = {
    "flood_2025.tif":
        RAW / "VIIRS_20250826_20250907_FloodWaterExtent_PAK.shp",

    "analysis_mask.tif":
        RAW / "VIIRS_20250826_20250907_AnalysisExtent_PAK.shp",

    "cloud_mask.tif":
        RAW / "VIIRS_20250826_20250907_CloudObstruction_PAK.shp",
}


# ---------------------------------------------------------
# Rasterize each layer block-by-block
# ---------------------------------------------------------
for outname, shppath in layers.items():

    print("\n" + "=" * 60)
    print(f"Processing: {shppath.name}")

    # Load shapefile
    gdf = gpd.read_file(shppath)

    if gdf.crs is None:
        raise ValueError(f"{shppath.name} has no CRS information.")

    # Reproject to exact FSM CRS
    if gdf.crs != crs:
        print(f"Reprojecting: {gdf.crs} -> {crs}")
        gdf = gdf.to_crs(crs)

    # Keep only valid, non-empty geometries
    geoms = [
        (geom, 1)
        for geom in gdf.geometry
        if geom is not None and not geom.is_empty
    ]

    if not geoms:
        raise ValueError(f"No valid geometry found in {shppath.name}")

    print(f"Valid source features: {len(geoms):,}")

    outpath = INTERIM / outname

    total_ones = 0
    blocks_done = 0

    total_blocks = (
        ((H + BLOCK - 1) // BLOCK)
        *
        ((W + BLOCK - 1) // BLOCK)
    )

    # -----------------------------------------------------
    # Create output raster
    # -----------------------------------------------------
    with rasterio.open(outpath, "w", **meta) as dst:

        for row_start in range(0, H, BLOCK):

            for col_start in range(0, W, BLOCK):

                row_size = min(BLOCK, H - row_start)
                col_size = min(BLOCK, W - col_start)

                window = Window(
                    col_off=col_start,
                    row_off=row_start,
                    width=col_size,
                    height=row_size
                )

                # Transform corresponding to this block
                win_transform = rasterio.windows.transform(
                    window,
                    full_transform
                )

                # Rasterize only this part of the national grid
                block = rasterize(
                    geoms,
                    out_shape=(row_size, col_size),
                    transform=win_transform,
                    fill=0,
                    default_value=1,
                    dtype="uint8",
                    all_touched=False
                )

                # Write block directly to disk
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
                        f"  {blocks_done:,}/{total_blocks:,} "
                        f"blocks completed"
                    )

    print(f"Written: {outpath}")
    print(f"Pixels = 1: {total_ones:,}")


print("\n" + "=" * 60)
print("Rasterization complete.")