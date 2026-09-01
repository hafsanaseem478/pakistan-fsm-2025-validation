from pathlib import Path
import rasterio
import numpy as np

RAW = Path("data/raw")
INTERIM = Path("data/interim")

REF = RAW / "fsm_lgbm_pakistan.tif"

rasters = {
    "2010": INTERIM / "flood_2010.tif",
    "2014": INTERIM / "flood_2014.tif",
    "2022": INTERIM / "flood_2022.tif",
    "training": INTERIM / "training_inventory_mask.tif",
}

print("=" * 70)
print("HISTORICAL MASK VERIFICATION")
print("=" * 70)

with rasterio.open(REF) as ref:

    for name, path in rasters.items():

        with rasterio.open(path) as src:

            checks = {
                "CRS": src.crs == ref.crs,
                "Width": src.width == ref.width,
                "Height": src.height == ref.height,
                "Transform": src.transform == ref.transform,
                "Bounds": src.bounds == ref.bounds,
            }

            print(f"\n{name}:")
            for key, value in checks.items():
                print(f"  {key}: {'OK' if value else 'FAIL'}")


# ---------------------------------------------------------
# Verify union logic block by block
# ---------------------------------------------------------
with rasterio.open(rasters["2010"]) as a, \
     rasterio.open(rasters["2014"]) as b, \
     rasterio.open(rasters["2022"]) as c, \
     rasterio.open(rasters["training"]) as t:

    mismatch = 0

    overlap_2010_2014 = 0
    overlap_2010_2022 = 0
    overlap_2014_2022 = 0
    overlap_all_three = 0

    for _, window in t.block_windows(1):

        x2010 = a.read(1, window=window) == 1
        x2014 = b.read(1, window=window) == 1
        x2022 = c.read(1, window=window) == 1

        training = t.read(1, window=window) == 1

        expected = x2010 | x2014 | x2022

        mismatch += int(
            np.count_nonzero(training != expected)
        )

        overlap_2010_2014 += int(
            np.count_nonzero(x2010 & x2014)
        )

        overlap_2010_2022 += int(
            np.count_nonzero(x2010 & x2022)
        )

        overlap_2014_2022 += int(
            np.count_nonzero(x2014 & x2022)
        )

        overlap_all_three += int(
            np.count_nonzero(
                x2010 & x2014 & x2022
            )
        )


print("\n" + "=" * 70)
print("UNION CHECK")
print("=" * 70)

print(f"Union mismatches: {mismatch:,}")

print("\nHistorical overlap:")
print(f"2010 ∩ 2014: {overlap_2010_2014:,}")
print(f"2010 ∩ 2022: {overlap_2010_2022:,}")
print(f"2014 ∩ 2022: {overlap_2014_2022:,}")
print(f"All three:   {overlap_all_three:,}")

if mismatch == 0:
    print("\nSUCCESS: training_inventory_mask.tif is correct.")
else:
    print("\nWARNING: union mask contains mismatches.")