import geopandas as gpd

RAW = "data/raw/FL20250818PAK_SHP"

files = [
    f"{RAW}/VIIRS_20250826_20250907_FloodWaterExtent_PAK.shp",
    f"{RAW}/VIIRS_20250826_20250907_CloudObstruction_PAK.shp",
    f"{RAW}/VIIRS_20250826_20250907_AnalysisExtent_PAK.shp",
    f"{RAW}/S1_20250827_20250828_FloodExtent_Punjab.shp",
]

for f in files:
    print(f"\n{'='*60}")
    print(f.split('/')[-1])
    gdf = gpd.read_file(f)
    print(f"Rows: {gdf.shape[0]}, Columns: {gdf.shape[1]}")
    print(f"CRS: {gdf.crs}")
    print(f"Columns: {gdf.columns.tolist()}")
    for col in gdf.columns:
        if gdf[col].dtype == 'object' and col != 'geometry':
            print(f"  {col}: {gdf[col].unique()}")
    print(f"Total area (km²): {gdf.to_crs(epsg=32642).area.sum() / 1e6:.1f}")
    # inspect key fields in flood layer
RAW = "data/raw/FL20250818PAK_SHP"
gdf = gpd.read_file(f"{RAW}/VIIRS_20250826_20250907_FloodWaterExtent_PAK.shp")
print("\nWater_Clas:", gdf['Water_Clas'].unique())
print("Confidence:", gdf['Confidence'].unique())
print("Water_Stat:", gdf['Water_Stat'].unique())