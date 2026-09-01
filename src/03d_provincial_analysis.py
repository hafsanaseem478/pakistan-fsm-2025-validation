from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, bounds as window_bounds



RAW = Path("data/raw")
INTERIM = Path("data/interim")
OUTPUT = Path("outputs/tables")

OUTPUT.mkdir(parents=True, exist_ok=True)

REF = RAW / "fsm_lgbm_pakistan.tif"

FLOOD = INTERIM / "flood_2025.tif"
ANALYSIS = INTERIM / "analysis_mask.tif"
CLOUD = INTERIM / "cloud_mask.tif"
PERM_WATER = INTERIM / "permanent_water.tif"
HISTORICAL = INTERIM / "training_inventory_mask.tif"

# GADM ADM1 boundary
ADM_FILE = RAW / "gadm41_PAK_1.shp"
# Window processing size
BLOCK = 4096



required_files = {
    "FSM": REF,
    "2025 flood": FLOOD,
    "analysis mask": ANALYSIS,
    "cloud mask": CLOUD,
    "permanent water": PERM_WATER,
    "historical inventory": HISTORICAL,
    "GADM ADM1": ADM_FILE,
}

for name, path in required_files.items():
    if not path.exists():
        raise FileNotFoundError(
            f"{name} file not found:\n{path}"
        )


print("=" * 75)
print("PROVINCIAL / ADM1 VALIDATION")
print("=" * 75)

print(f"\nBoundary file: {ADM_FILE}")


# READ FSM REFERENCE GRID

with rasterio.open(REF) as ref:

    CRS = ref.crs
    TRANSFORM = ref.transform
    H = ref.height
    W = ref.width

    print("\nFSM reference")
    print(f"CRS:        {CRS}")
    print(f"Size:       {W:,} x {H:,}")
    print(f"Resolution: {ref.res}")
    print(f"Bounds:     {ref.bounds}")


# READ GADM ADM1

adm = gpd.read_file(ADM_FILE)

print("\nADM1 columns:")
print(adm.columns.tolist())


# GADM ADM1 province/region name
if "NAME_1" not in adm.columns:
    raise ValueError(
        "NAME_1 field not found in GADM ADM1 shapefile."
    )

name_field = "NAME_1"


# Remove null / empty geometries
adm = adm[
    adm.geometry.notna()
    & ~adm.geometry.is_empty
].copy()


# Fix invalid geometries if necessary
if (~adm.geometry.is_valid).any():

    print("\nFixing invalid ADM1 geometries...")

    adm.geometry = adm.geometry.make_valid()


# Reproject boundaries to FSM CRS
adm = adm.to_crs(CRS)


# Give each ADM1 unit an integer ID
adm["adm_id"] = np.arange(
    1,
    len(adm) + 1
)


id_to_name = dict(
    zip(
        adm["adm_id"],
        adm[name_field]
    )
)


print("\nADM1 units found:")

for adm_id, name in id_to_name.items():
    print(f"  {adm_id}: {name}")


# COUNTS
#
# dimensions:
# ADM1 × FSM class × category
#
# category:
#
# 0 = valid area
# 1 = flooded area
#
# 2 = valid area inside historical inventory
# 3 = flooded area inside historical inventory
#
# 4 = valid area outside historical inventory
# 5 = flooded area outside historical inventory

counts = np.zeros(
    (
        len(adm) + 1,
        256,
        6
    ),
    dtype=np.int64
)


# Spatial index makes province lookup faster
sindex = adm.sindex


# OPEN RASTERS

paths = {
    "fsm": REF,
    "flood": FLOOD,
    "analysis": ANALYSIS,
    "cloud": CLOUD,
    "pw": PERM_WATER,
    "hist": HISTORICAL,
}


readers = {
    name: rasterio.open(path)
    for name, path in paths.items()
}


try:

    # VERIFY EXACT GRID ALIGNMENT

    reference = readers["fsm"]

    for name, src in readers.items():

        assert src.crs == reference.crs, \
            f"CRS mismatch: {name}"

        assert src.transform == reference.transform, \
            f"Transform mismatch: {name}"

        assert src.width == reference.width, \
            f"Width mismatch: {name}"

        assert src.height == reference.height, \
            f"Height mismatch: {name}"


    print("\nAll raster grids aligned correctly.")


    # WINDOW PROCESSING

    total_blocks = (
        ((H + BLOCK - 1) // BLOCK)
        *
        ((W + BLOCK - 1) // BLOCK)
    )

    block_number = 0


    print("\nProcessing blocks...")


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
                TRANSFORM
            )


            # READ RASTER BLOCKS

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


            pw = readers["pw"].read(
                1,
                window=window
            )


            hist = readers["hist"].read(
                1,
                window=window
            )


            # VALID ANALYSIS DOMAIN

            valid = (
                (analysis == 1)
                &
                (cloud == 0)
                &
                (pw == 0)
                &
                (fsm != 255)
            )


            flooded = (
                valid
                &
                (flood == 1)
            )


            # HISTORICAL INVENTORY SUBSETS

            valid_inside = (
                valid
                &
                (hist == 1)
            )


            flooded_inside = (
                flooded
                &
                (hist == 1)
            )


            valid_outside = (
                valid
                &
                (hist == 0)
            )


            flooded_outside = (
                flooded
                &
                (hist == 0)
            )


            # CREATE ADM1 RASTER FOR CURRENT BLOCK

            left, bottom, right, top = window_bounds(
                window,
                TRANSFORM
            )


            candidate_ids = list(
                sindex.intersection(
                    (
                        left,
                        bottom,
                        right,
                        top
                    )
                )
            )


            if candidate_ids:

                subset = adm.iloc[candidate_ids]


                shapes = [
                    (
                        geom,
                        int(adm_id)
                    )
                    for geom, adm_id
                    in zip(
                        subset.geometry,
                        subset["adm_id"]
                    )
                    if geom is not None
                    and not geom.is_empty
                ]


                adm_block = rasterize(
                    shapes,
                    out_shape=(
                        int(row_size),
                        int(col_size)
                    ),
                    transform=win_transform,
                    fill=0,
                    dtype="uint8",
                    all_touched=False
                )


            else:

                adm_block = np.zeros(
                    (
                        int(row_size),
                        int(col_size)
                    ),
                    dtype="uint8"
                )


            # COUNT EACH ADM1 UNIT

            present_ids = np.unique(
                adm_block
            )


            present_ids = present_ids[
                present_ids > 0
            ]


            masks = [
                valid,
                flooded,
                valid_inside,
                flooded_inside,
                valid_outside,
                flooded_outside,
            ]


            for adm_id in present_ids:

                admin_pixels = (
                    adm_block == adm_id
                )


                for category, mask in enumerate(masks):

                    combined_mask = (
                        admin_pixels
                        &
                        mask
                    )


                    values = fsm[
                        combined_mask
                    ]


                    if values.size > 0:

                        bc = np.bincount(
                            values,
                            minlength=256
                        )


                        counts[
                            int(adm_id),
                            :,
                            category
                        ] += bc


            block_number += 1


            if (
                block_number % 20 == 0
                or block_number == total_blocks
            ):

                print(
                    f"  {block_number:,}/"
                    f"{total_blocks:,} blocks completed"
                )


finally:

    for reader in readers.values():
        reader.close()


# BUILD RESULTS TABLE

rows = []

fsm_classes = [
    1,
    2,
    3,
    4,
    5
]


for adm_id, name in id_to_name.items():

    c = counts[
        adm_id
    ]


    # ========================================================
    # TOTALS
    # ========================================================

    valid_total = c[
        fsm_classes,
        0
    ].sum()


    flood_total = c[
        fsm_classes,
        1
    ].sum()


    valid_inside = c[
        fsm_classes,
        2
    ].sum()


    flood_inside = c[
        fsm_classes,
        3
    ].sum()


    valid_outside = c[
        fsm_classes,
        4
    ].sum()


    flood_outside = c[
        fsm_classes,
        5
    ].sum()


    # Skip ADM1 units with no usable 2025 flood
    if flood_total == 0:
        continue


    # ========================================================
    # FLOOD SHARE BY FSM CLASS
    # ========================================================

    flood_class_pct = {}


    for cls in fsm_classes:

        flood_class_pct[cls] = (
            c[cls, 1]
            /
            flood_total
            *
            100
        )


    # ========================================================
    # HIGH + VERY HIGH — OVERALL
    # ========================================================

    high_vhigh_flood = (
        c[4, 1]
        +
        c[5, 1]
    )


    high_vhigh_capture = (
        high_vhigh_flood
        /
        flood_total
        *
        100
    )


    high_vhigh_valid = (
        c[4, 0]
        +
        c[5, 0]
    )


    high_vhigh_area = (
        high_vhigh_valid
        /
        valid_total
        *
        100
    ) if valid_total > 0 else np.nan


    # ========================================================
    # HISTORICAL OVERLAP
    # ========================================================

    inside_history_flood_pct = (
        flood_inside
        /
        flood_total
        *
        100
    )


    # HISTORICAL NON-OVERLAP

    outside_history_flood_pct = (
        flood_outside
        /
        flood_total
        *
        100
    )


    if flood_outside > 0:

        outside_high_vhigh_flood = (
            c[4, 5]
            +
            c[5, 5]
        )


        outside_high_vhigh_capture = (
            outside_high_vhigh_flood
            /
            flood_outside
            *
            100
        )


        outside_high_vhigh_valid = (
            c[4, 4]
            +
            c[5, 4]
        )


        outside_high_vhigh_area = (
            outside_high_vhigh_valid
            /
            valid_outside
            *
            100
        ) if valid_outside > 0 else np.nan


    else:

        outside_high_vhigh_capture = np.nan
        outside_high_vhigh_area = np.nan


    # ========================================================
    # HIGH + VERY HIGH FREQUENCY RATIO
    # ========================================================

    if high_vhigh_area > 0:

        high_vhigh_fr = (
            high_vhigh_capture
            /
            high_vhigh_area
        )

    else:

        high_vhigh_fr = np.nan


    if (
        outside_high_vhigh_area is not np.nan
        and outside_high_vhigh_area > 0
    ):

        outside_high_vhigh_fr = (
            outside_high_vhigh_capture
            /
            outside_high_vhigh_area
        )

    else:

        outside_high_vhigh_fr = np.nan


    # ========================================================
    # RESULT ROW
    # ========================================================

    rows.append({

        "ADM1":
            name,

        "valid_pixels":
            int(valid_total),

        "flood_pixels":
            int(flood_total),

        "class1_flood_pct":
            flood_class_pct[1],

        "class2_flood_pct":
            flood_class_pct[2],

        "class3_flood_pct":
            flood_class_pct[3],

        "class4_flood_pct":
            flood_class_pct[4],

        "class5_flood_pct":
            flood_class_pct[5],

        "high_very_high_area_pct":
            high_vhigh_area,

        "high_very_high_capture_pct":
            high_vhigh_capture,

        "high_very_high_FR":
            high_vhigh_fr,

        "inside_history_flood_pct":
            inside_history_flood_pct,

        "outside_history_flood_pct":
            outside_history_flood_pct,

        "outside_high_very_high_area_pct":
            outside_high_vhigh_area,

        "outside_high_very_high_capture_pct":
            outside_high_vhigh_capture,

        "outside_high_very_high_FR":
            outside_high_vhigh_fr,
    })


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    rows
)


if df.empty:

    raise RuntimeError(
        "No provincial flood results were generated."
    )


# ============================================================
# SORT BY 2025 FLOOD
# ============================================================

df = df.sort_values(
    "flood_pixels",
    ascending=False
).reset_index(
    drop=True
)


# NATIONAL SHARE OF OBSERVED 2025 FLOOD

national_flood = df[
    "flood_pixels"
].sum()


df[
    "national_flood_share_pct"
] = (
    df["flood_pixels"]
    /
    national_flood
    *
    100
)


# REORDER COLUMNS

column_order = [

    "ADM1",

    "valid_pixels",
    "flood_pixels",
    "national_flood_share_pct",

    "class1_flood_pct",
    "class2_flood_pct",
    "class3_flood_pct",
    "class4_flood_pct",
    "class5_flood_pct",

    "high_very_high_area_pct",
    "high_very_high_capture_pct",
    "high_very_high_FR",

    "inside_history_flood_pct",
    "outside_history_flood_pct",

    "outside_high_very_high_area_pct",
    "outside_high_very_high_capture_pct",
    "outside_high_very_high_FR",
]


df = df[
    column_order
]


# SAVE FULL TABLE

OUT = OUTPUT / "provincial_validation.csv"


df.to_csv(
    OUT,
    index=False
)


# PRINT IMPORTANT RESULTS

print("\n" + "=" * 110)
print("PROVINCIAL / ADM1 RESULTS")
print("=" * 110)


display_cols = [

    "ADM1",

    "national_flood_share_pct",

    "high_very_high_capture_pct",

    "outside_history_flood_pct",

    "outside_high_very_high_capture_pct",

    "outside_high_very_high_FR",
]


print(
    df[
        display_cols
    ]
    .round(2)
    .to_string(
        index=False
    )
)


# QA SUMMARY

print("\n" + "=" * 75)
print("QA SUMMARY")
print("=" * 75)


print(
    f"ADM1 units with valid flooding: "
    f"{len(df)}"
)


print(
    f"2025 flooded pixels assigned to ADM1 units: "
    f"{national_flood:,}"
)


print(
    f"National flood share sum: "
    f"{df['national_flood_share_pct'].sum():.2f}%"
)


print("\nSaved:")
print(f"  {OUT}")

print("\nDone.")