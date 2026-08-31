from pathlib import Path
import rasterio

RAW = Path("data/raw")
INTERIM = Path("data/interim")

# Adjust the FSM filename if yours is different
rasters = {
    "FSM": RAW / "fsm_lgbm_pakistan.tif",
    "Flood 2025": INTERIM / "flood_2025.tif",
    "Analysis mask": INTERIM / "analysis_mask.tif",
    "Cloud mask": INTERIM / "cloud_mask.tif",
}

info = {}

print("=" * 70)
print("RASTER ALIGNMENT CHECK")
print("=" * 70)

for name, path in rasters.items():
    with rasterio.open(path) as src:
        info[name] = {
            "crs": src.crs,
            "width": src.width,
            "height": src.height,
            "res": src.res,
            "transform": src.transform,
            "bounds": src.bounds,
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
        }

        print(f"\n{name}")
        print(f"Path:       {path}")
        print(f"CRS:        {src.crs}")
        print(f"Size:       {src.width:,} x {src.height:,}")
        print(f"Resolution: {src.res}")
        print(f"Bounds:     {src.bounds}")
        print(f"dtype:      {src.dtypes[0]}")
        print(f"NoData:     {src.nodata}")


# Compare every raster against FSM
reference = info["FSM"]

print("\n" + "=" * 70)
print("COMPARISON WITH FSM")
print("=" * 70)

all_ok = True

for name, values in info.items():

    if name == "FSM":
        continue

    print(f"\n{name}")

    checks = {
        "CRS": values["crs"] == reference["crs"],
        "Width": values["width"] == reference["width"],
        "Height": values["height"] == reference["height"],
        "Resolution": values["res"] == reference["res"],
        "Transform": values["transform"] == reference["transform"],
        "Bounds": values["bounds"] == reference["bounds"],
    }

    for check_name, result in checks.items():
        symbol = "OK" if result else "FAIL"
        print(f"{check_name:12}: {symbol}")

        if not result:
            all_ok = False


print("\n" + "=" * 70)

if all_ok:
    print("SUCCESS: All rasters are perfectly aligned with the FSM.")
else:
    print("WARNING: One or more rasters are not aligned with the FSM.")

print("=" * 70)
