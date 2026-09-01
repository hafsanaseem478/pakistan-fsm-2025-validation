from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


# ============================================================
# PATHS
# ============================================================

RAW = Path("data/raw")
INTERIM = Path("data/interim")
PROCESSED = Path("data/processed")

PROCESSED.mkdir(parents=True, exist_ok=True)

FSM = RAW / "fsm_lgbm_pakistan.tif"
FLOOD = INTERIM / "flood_2025.tif"

OUT = PROCESSED / "fsm_2025_validation_map.tif"

BLOCK = 4096


# ============================================================
# OPEN INPUTS
# ============================================================

with rasterio.open(FSM) as fsm_src, rasterio.open(FLOOD) as flood_src:

    # Verify exact alignment
    assert fsm_src.crs == flood_src.crs, "CRS mismatch"
    assert fsm_src.transform == flood_src.transform, "Transform mismatch"
    assert fsm_src.width == flood_src.width, "Width mismatch"
    assert fsm_src.height == flood_src.height, "Height mismatch"

    print("Raster grids aligned correctly.")
    print(f"Size: {fsm_src.width:,} x {fsm_src.height:,}")

    profile = fsm_src.profile.copy()

    profile.update(
        dtype="uint8",
        count=1,
        nodata=0,
        compress="DEFLATE",
        tiled=True,
        BIGTIFF="YES"
    )

    H = fsm_src.height
    W = fsm_src.width

    total_blocks = (
        ((H + BLOCK - 1) // BLOCK)
        *
        ((W + BLOCK - 1) // BLOCK)
    )

    block_number = 0


    # ========================================================
    # CREATE OUTPUT
    # ========================================================

    with rasterio.open(OUT, "w", **profile) as dst:

        for row_start in range(0, H, BLOCK):

            row_size = min(BLOCK, H - row_start)

            for col_start in range(0, W, BLOCK):

                col_size = min(BLOCK, W - col_start)

                window = Window(
                    col_start,
                    row_start,
                    col_size,
                    row_size
                )

                fsm = fsm_src.read(
                    1,
                    window=window
                )

                flood = flood_src.read(
                    1,
                    window=window
                )


                # ------------------------------------------------
                # OUTPUT CLASSES
                #
                # 0 = not flooded / transparent
                # 1 = Flooded in Very Low or Low susceptibility
                # 2 = Flooded in Moderate susceptibility
                # 3 = Flooded in High or Very High susceptibility
                # ------------------------------------------------

                out = np.zeros(
                    fsm.shape,
                    dtype=np.uint8
                )


                flooded = flood == 1


                # Very Low + Low = model miss
                out[
                    flooded
                    &
                    (
                        (fsm == 1)
                        |
                        (fsm == 2)
                    )
                ] = 1


                # Moderate
                out[
                    flooded
                    &
                    (fsm == 3)
                ] = 2


                # High + Very High = captured
                out[
                    flooded
                    &
                    (
                        (fsm == 4)
                        |
                        (fsm == 5)
                    )
                ] = 3


                dst.write(
                    out,
                    1,
                    window=window
                )


                block_number += 1

                if (
                    block_number % 20 == 0
                    or block_number == total_blocks
                ):
                    print(
                        f"  {block_number}/{total_blocks} blocks completed"
                    )


print("\nDone.")
print(f"Saved: {OUT}")