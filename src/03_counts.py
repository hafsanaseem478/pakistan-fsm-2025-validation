from pathlib import Path

import numpy as np
import rasterio


INTERIM = Path("data/interim")
REF = Path("data/raw/fsm_lgbm_pakistan.tif")

files = {
    "fsm": REF,
    "flood": INTERIM / "flood_2025.tif",
    "analysis": INTERIM / "analysis_mask.tif",
    "cloud": INTERIM / "cloud_mask.tif",
    "perm_water": INTERIM / "permanent_water.tif",
    "historical": INTERIM / "training_inventory_mask.tif",
}


# counts[class, category]
#
# 0 = valid area, all
# 1 = flooded area, all
#
# 2 = valid area inside historical inventory
# 3 = flooded area inside historical inventory
#
# 4 = valid area outside historical inventory
# 5 = flooded area outside historical inventory

counts = np.zeros((256, 6), dtype=np.int64)


print("=" * 70)
print("FSM FREQUENCY RATIO ANALYSIS")
print("=" * 70)


readers = {
    key: rasterio.open(path)
    for key, path in files.items()
}


try:

    # Verify grids

    ref = readers["fsm"]

    for name, src in readers.items():

        assert src.crs == ref.crs, \
            f"CRS mismatch: {name}"

        assert src.width == ref.width, \
            f"Width mismatch: {name}"

        assert src.height == ref.height, \
            f"Height mismatch: {name}"

        assert src.transform == ref.transform, \
            f"Transform mismatch: {name}"


    # Process block-wise

    block_num = 0

    for _, window in ref.block_windows(1):

        fsm = readers["fsm"].read(
            1,
            window=window
        )

        flood = readers["flood"].read(
            1,
            window=window
        )

        analysis = readers["analysis"].read(
            1,
            window=window
        )

        cloud = readers["cloud"].read(
            1,
            window=window
        )

        permanent = readers["perm_water"].read(
            1,
            window=window
        )

        historical = readers["historical"].read(
            1,
            window=window
        )


        # Valid analysis domain

        valid = (
            (analysis == 1)
            &
            (cloud == 0)
            &
            (permanent == 0)
            &
            (fsm != 255)
        )


        flooded = (
            valid
            &
            (flood == 1)
        )


        # Historical subsets

        valid_inside = (
            valid
            &
            (historical == 1)
        )

        flooded_inside = (
            flooded
            &
            (historical == 1)
        )


        valid_outside = (
            valid
            &
            (historical == 0)
        )

        flooded_outside = (
            flooded
            &
            (historical == 0)
        )


        # Faster than looping through all 256 classes

        masks = [
            valid,
            flooded,
            valid_inside,
            flooded_inside,
            valid_outside,
            flooded_outside,
        ]


        for category, mask in enumerate(masks):

            values = fsm[mask]

            if values.size > 0:

                bc = np.bincount(
                    values,
                    minlength=256
                )

                counts[:, category] += bc


        block_num += 1

        if block_num % 500 == 0:
            print(
                f"{block_num:,} blocks processed"
            )


finally:

    for src in readers.values():
        src.close()


# Active FSM values

active = np.where(
    counts[:, 0] > 0
)[0]

active = active[
    active != 255
]


print(
    f"\nFSM values with valid data: "
    f"{active}"
)


# Totals

total_valid = counts[active, 0].sum()
total_flood = counts[active, 1].sum()

total_valid_inside = counts[active, 2].sum()
total_flood_inside = counts[active, 3].sum()

total_valid_outside = counts[active, 4].sum()
total_flood_outside = counts[active, 5].sum()


print("\n" + "=" * 70)
print("TOTALS")
print("=" * 70)

print(
    f"All valid pixels:                "
    f"{total_valid:,}"
)

print(
    f"All valid flooded pixels:        "
    f"{total_flood:,}"
)

print(
    f"\nValid pixels INSIDE history:     "
    f"{total_valid_inside:,}"
)

print(
    f"Flood pixels INSIDE history:     "
    f"{total_flood_inside:,}"
)

print(
    f"\nValid pixels OUTSIDE history:    "
    f"{total_valid_outside:,}"
)

print(
    f"Flood pixels OUTSIDE history:    "
    f"{total_flood_outside:,}"
)


# Helper function

def print_fr_table(
    title,
    valid_col,
    flood_col
):

    total_area = counts[
        active,
        valid_col
    ].sum()

    total_flooded = counts[
        active,
        flood_col
    ].sum()


    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print(
        f"{'Class':>7} "
        f"{'Valid':>15} "
        f"{'Flooded':>15} "
        f"{'%Area':>9} "
        f"{'%Flood':>9} "
        f"{'FR':>9}"
    )

    print("-" * 80)


    for c in active:

        n_valid = counts[c, valid_col]
        n_flood = counts[c, flood_col]


        pct_area = (
            n_valid
            /
            total_area
            *
            100
        ) if total_area else 0


        pct_flood = (
            n_flood
            /
            total_flooded
            *
            100
        ) if total_flooded else 0


        fr = (
            pct_flood
            /
            pct_area
        ) if pct_area else np.nan


        print(
            f"{c:>7} "
            f"{n_valid:>15,} "
            f"{n_flood:>15,} "
            f"{pct_area:>8.2f}% "
            f"{pct_flood:>8.2f}% "
            f"{fr:>9.3f}"
        )


# Overall FR

print_fr_table(
    "ALL 2025 FLOOD",
    valid_col=0,
    flood_col=1
)


# Historical-overlap FR

print_fr_table(
    "2025 FLOOD — HISTORICAL-INVENTORY OVERLAP",
    valid_col=2,
    flood_col=3
)


# Historical-non-overlap FR

print_fr_table(
    "2025 FLOOD — HISTORICAL-INVENTORY NON-OVERLAP",
    valid_col=4,
    flood_col=5
)


# QA

print("\n" + "=" * 70)
print("QA")
print("=" * 70)

print(
    "Valid inside + outside = total: ",
    total_valid_inside
    +
    total_valid_outside
    ==
    total_valid
)

print(
    "Flood inside + outside = total: ",
    total_flood_inside
    +
    total_flood_outside
    ==
    total_flood
)


# Save

np.save(
    INTERIM / "fr_counts.npy",
    counts
)

print(
    "\nRaw counts saved to "
    "data/interim/fr_counts.npy"
)

print("\nDone.")
