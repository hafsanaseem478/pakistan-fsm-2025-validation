from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window


# Paths
REF = Path("data/raw/fsm_lgbm_pakistan.tif")
JRC_RAW = Path("data/raw/jrc_permanent_water_pakistan.tif")
OUT = Path("data/interim/permanent_water.tif")

OUT.parent.mkdir(parents=True, exist_ok=True)


# Processing settings
BLOCK = 4096


# Read exact FSM reference grid
with rasterio.open(REF) as ref:

    H = ref.height
    W = ref.width
    dst_crs = ref.crs
    dst_transform = ref.transform

    meta = ref.meta.copy()

    print("=" * 65)
    print("REFERENCE FSM")
    print("=" * 65)

    print(f"CRS:        {dst_crs}")
    print(f"Size:       {W:,} x {H:,}")
    print(f"Resolution: {ref.res}")
    print(f"Bounds:     {ref.bounds}")


# Output settings
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


total_blocks = (
    ((H + BLOCK - 1) // BLOCK)
    *
    ((W + BLOCK - 1) // BLOCK)
)


# Open raw JRC raster
with rasterio.open(JRC_RAW) as src:

    print("\n" + "=" * 65)
    print("RAW JRC WATER MASK")
    print("=" * 65)

    print(f"CRS:        {src.crs}")
    print(f"Size:       {src.width:,} x {src.height:,}")
    print(f"Resolution: {src.res}")
    print(f"NoData:     {src.nodata}")


    # Virtual raster aligned exactly to the FSM
    with WarpedVRT(
        src,
        crs=dst_crs,
        transform=dst_transform,
        width=W,
        height=H,
        src_nodata=src.nodata,
        nodata=255,
        resampling=Resampling.nearest
    ) as vrt:

        total_ones = 0
        total_zero = 0
        total_nodata = 0

        blocks_done = 0

        with rasterio.open(OUT, "w", **meta) as dst:

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

                    # Read only this block,
                    block = vrt.read(
                        1,
                        window=window
                    )

                    # Ensure expected classes:
                    # 1   = permanent water
                    # 0   = not permanent water
                    # 255 = NoData
                    block = block.astype("uint8")

                    dst.write(
                        block,
                        1,
                        window=window
                    )

                    total_ones += int(
                        np.count_nonzero(block == 1)
                    )

                    total_zero += int(
                        np.count_nonzero(block == 0)
                    )

                    total_nodata += int(
                        np.count_nonzero(block == 255)
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


# Summary
print("\n" + "=" * 65)
print("PERMANENT WATER ALIGNMENT COMPLETE")
print("=" * 65)

print(f"Written: {OUT}")
print(f"Permanent-water pixels: {total_ones:,}")
print(f"Non-water pixels:       {total_zero:,}")
print(f"NoData pixels:          {total_nodata:,}")
print("\nDone.")