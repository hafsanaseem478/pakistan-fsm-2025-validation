from pathlib import Path
import geopandas as gpd


gdbs = [
    Path("data/raw/FL20100802PAK.gdb"),
    Path("data/raw/FL20140910PAK.gdb"),
    Path("data/raw/FL20220808PAK.gdb"),
]


keywords = [
    "flood",
    "cumulative",
    "analysis",
    "cloud",
]


for gdb in gdbs:

    print("\n" + "=" * 70)
    print(gdb)
    print("=" * 70)

    layers_df = gpd.list_layers(gdb)

    # Keep only potentially relevant spatial layers
    relevant = layers_df[
        layers_df["name"].str.lower().apply(
            lambda x: any(k in x for k in keywords)
        )
    ]

    print("\nRelevant layers:")

    for _, row in relevant.iterrows():
        print(f"  {row['name']} | geometry: {row['geometry_type']}")

    print("\nDETAILS")

    for layer in relevant["name"]:

        print("\n" + "-" * 60)
        print(f"Layer: {layer}")

        obj = gpd.read_file(gdb, layer=layer)

        # Some GDB entries are ordinary tables rather than spatial layers
        if not isinstance(obj, gpd.GeoDataFrame):
            print("Non-spatial table — skipped.")
            continue

        print(f"Rows: {len(obj):,}")
        print(f"CRS: {obj.crs}")
        print(f"Geometry: {obj.geom_type.unique() if len(obj) else 'EMPTY'}")

        # Useful flood-related fields
        fields = [
            "Water_Class",
            "Water_Clas",
            "Water_StatusID",
            "Water_Stat",
            "Confidence_ID",
            "Confidence",
            "Sensor_Date",
            "SensorDate",
            "Area_m2",
            "Area_ha",
            "Notes",
        ]

        for col in fields:

            if col not in obj.columns:
                continue

            if col in ["Area_m2", "Area_ha"]:

                total = obj[col].fillna(0).sum()

                print(f"{col} total: {total:,.2f}")

            else:

                values = obj[col].dropna().unique()

                # Avoid dumping hundreds of values
                if len(values) <= 20:
                    print(f"{col}: {values}")
                else:
                    print(
                        f"{col}: {len(values)} unique values "
                        f"(first 10: {values[:10]})"
                    )


# ------------------------------------------------------------
# Find and inspect the national 2022 SHP automatically
# ------------------------------------------------------------
matches = list(
    Path("data/raw").rglob(
        "VIIRS_20220701_20220831_FloodExtent_PAK.shp"
    )
)

print("\n" + "=" * 70)
print("2022 NATIONAL SHAPEFILE")
print("=" * 70)

if matches:

    shp = matches[0]

    print(f"Found: {shp}")

    gdf = gpd.read_file(shp)

    print(f"Rows: {len(gdf):,}")
    print(f"CRS: {gdf.crs}")
    print(f"Columns: {gdf.columns.tolist()}")

    for col in [
        "Water_Clas",
        "Water_Class",
        "Water_Stat",
        "Water_StatusID",
        "Confidence",
        "Confidence_ID",
        "Area_m2",
        "Area_ha",
    ]:

        if col not in gdf.columns:
            continue

        if col in ["Area_m2", "Area_ha"]:
            print(
                f"{col} total: "
                f"{gdf[col].fillna(0).sum():,.2f}"
            )
        else:
            print(
                f"{col}: "
                f"{gdf[col].dropna().unique()}"
            )

else:
    print("2022 national flood shapefile not found.")