from pathlib import Path

import numpy as np
import rasterio


INTERIM = Path("data/interim")

FLOOD_2025 = INTERIM / "flood_2025.tif"
HISTORICAL = INTERIM / "training_inventory_mask.tif"


inside = 0
outside = 0
total_flood = 0


print("=" * 65)
print("2025 FLOOD vs HISTORICAL TRAINING INVENTORY")
print("=" * 65)


with rasterio.open(FLOOD_2025) as flood, \
     rasterio.open(HISTORICAL) as hist:

    # Safety checks
    assert flood.crs == hist.crs, "CRS mismatch"
    assert flood.transform == hist.transform, "Transform mismatch"
    assert flood.width == hist.width, "Width mismatch"
    assert flood.height == hist.height, "Height mismatch"

    for _, window in flood.block_windows(1):

        f = flood.read(1, window=window)
        h = hist.read(1, window=window)

        flooded = f == 1

        flood_inside = flooded & (h == 1)
        flood_outside = flooded & (h == 0)

        inside += int(np.count_nonzero(flood_inside))
        outside += int(np.count_nonzero(flood_outside))
        total_flood += int(np.count_nonzero(flooded))


print(f"\nTotal 2025 flood pixels:       {total_flood:,}")
print(f"Inside historical inventory:  {inside:,}")
print(f"Outside historical inventory: {outside:,}")


if total_flood > 0:

    inside_pct = inside / total_flood * 100
    outside_pct = outside / total_flood * 100

    print("\n" + "-" * 65)

    print(
        f"Inside historical inventory:  "
        f"{inside_pct:.2f}%"
    )

    print(
        f"Outside historical inventory: "
        f"{outside_pct:.2f}%"
    )


# Check accounting
if inside + outside == total_flood:
    print("\nSUCCESS: all flooded pixels accounted for.")
else:
    missing = total_flood - (inside + outside)

    print(
        f"\nWARNING: {missing:,} flood pixels "
        "were not classified."
    )