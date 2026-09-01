from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


# ============================================================
# PATHS
# ============================================================

RAW = Path("data/raw")
INTERIM = Path("data/interim")
OUT = Path("outputs/figures")

OUT.mkdir(parents=True, exist_ok=True)

FSM_PATH = RAW / "fsm_lgbm_pakistan.tif"

FLOOD_PATH = INTERIM / "flood_2025.tif"
ANALYSIS_PATH = INTERIM / "analysis_mask.tif"
CLOUD_PATH = INTERIM / "cloud_mask.tif"
PERM_PATH = INTERIM / "permanent_water.tif"


# ============================================================
# DISPLAY RESOLUTION
# ============================================================

# 30 m × 20 = approximately 600 m display resolution
# Analysis itself is NOT being recomputed here.
DS = 20


def read_downsampled(path, ds=DS, resampling=Resampling.nearest):

    with rasterio.open(path) as src:

        h = int(np.ceil(src.height / ds))
        w = int(np.ceil(src.width / ds))

        data = src.read(
            1,
            out_shape=(h, w),
            resampling=resampling
        )

        bounds = src.bounds

    return data, bounds


# ============================================================
# READ DATA
# ============================================================

print("Reading downsampled rasters...")


# FSM is categorical → nearest neighbour
fsm, bounds = read_downsampled(
    FSM_PATH,
    resampling=Resampling.nearest
)


# Flood/masks
flood, _ = read_downsampled(
    FLOOD_PATH,
    resampling=Resampling.nearest
)

analysis, _ = read_downsampled(
    ANALYSIS_PATH,
    resampling=Resampling.nearest
)

cloud, _ = read_downsampled(
    CLOUD_PATH,
    resampling=Resampling.nearest
)

perm, _ = read_downsampled(
    PERM_PATH,
    resampling=Resampling.nearest
)


# ============================================================
# FINAL VALID DOMAIN
# ============================================================

valid = (
    (analysis == 1)
    &
    (cloud == 0)
    &
    (perm == 0)
    &
    (fsm != 255)
)


flooded = (
    valid
    &
    (flood == 1)
)


# ============================================================
# PLOT EXTENT
# ============================================================

extent = [
    bounds.left,
    bounds.right,
    bounds.bottom,
    bounds.top
]


# ============================================================
# PANEL A — FSM
# ============================================================

fsm_display = np.ma.masked_where(
    ~valid,
    fsm
)


# Strong publication-style susceptibility colors
fsm_colors = [
    "#FFFFB2",  # Very Low
    "#FECC5C",  # Low
    "#FD8D3C",  # Moderate
    "#F03B20",  # High
    "#BD0026",  # Very High
]

fsm_cmap = ListedColormap(fsm_colors)

fsm_norm = BoundaryNorm(
    [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
    fsm_cmap.N
)


# ============================================================
# PANEL B — OBSERVED 2025 FLOOD
# ============================================================

flood_display = np.zeros_like(flood, dtype=np.uint8)

flood_display[flooded] = 1

flood_display = np.ma.masked_where(
    flood_display == 0,
    flood_display
)


# ============================================================
# PANEL C — VALIDATION
#
# 1 = Flood in Very Low / Low → Miss
# 2 = Flood in Moderate
# 3 = Flood in High / Very High → Captured
# ============================================================

validation = np.zeros_like(
    flood,
    dtype=np.uint8
)


validation[
    flooded
    &
    (
        (fsm == 1)
        |
        (fsm == 2)
    )
] = 1


validation[
    flooded
    &
    (fsm == 3)
] = 2


validation[
    flooded
    &
    (
        (fsm == 4)
        |
        (fsm == 5)
    )
] = 3


validation_display = np.ma.masked_where(
    validation == 0,
    validation
)


validation_colors = [
    "#D73027",  # Miss
    "#FDAE61",  # Moderate
    "#1F78B4",  # Captured
]

validation_cmap = ListedColormap(
    validation_colors
)

validation_norm = BoundaryNorm(
    [0.5, 1.5, 2.5, 3.5],
    validation_cmap.N
)


# ============================================================
# BACKGROUND
# ============================================================

background = np.ma.masked_where(
    ~valid,
    np.ones_like(valid, dtype=np.uint8)
)


# ============================================================
# CREATE FIGURE
# ============================================================

print("Plotting...")


fig, axes = plt.subplots(
    1,
    3,
    figsize=(17, 7)
)


# ------------------------------------------------------------
# PANEL A
# ------------------------------------------------------------

axes[0].imshow(
    fsm_display,
    cmap=fsm_cmap,
    norm=fsm_norm,
    extent=extent,
    aspect="equal"
)

axes[0].set_title(
    "(a) Published Flood Susceptibility Map",
    fontsize=12,
    fontweight="bold",
    pad=12
)


# ------------------------------------------------------------
# PANEL B
# ------------------------------------------------------------

axes[1].imshow(
    background,
    cmap=ListedColormap(["#F2F2F2"]),
    extent=extent,
    aspect="equal"
)

axes[1].imshow(
    flood_display,
    cmap=ListedColormap(["#0077B6"]),
    extent=extent,
    aspect="equal",
    interpolation="nearest"
)

axes[1].set_title(
    "(b) Observed 2025 Flood Extent",
    fontsize=12,
    fontweight="bold",
    pad=12
)


# ------------------------------------------------------------
# PANEL C
# ------------------------------------------------------------

axes[2].imshow(
    background,
    cmap=ListedColormap(["#F2F2F2"]),
    extent=extent,
    aspect="equal"
)

axes[2].imshow(
    validation_display,
    cmap=validation_cmap,
    norm=validation_norm,
    extent=extent,
    aspect="equal",
    interpolation="nearest"
)

axes[2].set_title(
    "(c) Validation Against Observed Flooding",
    fontsize=12,
    fontweight="bold",
    pad=12
)


# ============================================================
# CLEAN AXES
# ============================================================

for ax in axes:

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)


# ============================================================
# LEGENDS
# ============================================================

fsm_legend = [

    Patch(
        facecolor="#FFFFB2",
        label="Very Low"
    ),

    Patch(
        facecolor="#FECC5C",
        label="Low"
    ),

    Patch(
        facecolor="#FD8D3C",
        label="Moderate"
    ),

    Patch(
        facecolor="#F03B20",
        label="High"
    ),

    Patch(
        facecolor="#BD0026",
        label="Very High"
    ),
]


axes[0].legend(
    handles=fsm_legend,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.10),
    ncol=3,
    fontsize=8,
    frameon=False
)


flood_legend = [

    Patch(
        facecolor="#0077B6",
        label="Observed 2025 flood"
    )
]


axes[1].legend(
    handles=flood_legend,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.07),
    fontsize=8,
    frameon=False
)


validation_legend = [

    Patch(
        facecolor="#D73027",
        label="Flood in Very Low / Low — Miss"
    ),

    Patch(
        facecolor="#FDAE61",
        label="Flood in Moderate"
    ),

    Patch(
        facecolor="#1F78B4",
        label="Flood in High / Very High — Captured"
    ),
]


axes[2].legend(
    handles=validation_legend,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.12),
    fontsize=8,
    frameon=False
)


# ============================================================
# MAIN TITLE
# ============================================================

fig.suptitle(
    "Independent Validation of Pakistan's Published Flood Susceptibility Map",
    fontsize=16,
    fontweight="bold",
    y=0.98
)


# ============================================================
# SMALL SUBTITLE
# ============================================================

fig.text(
    0.5,
    0.925,
    "Published susceptibility compared with satellite-observed flooding during the 2025 event",
    ha="center",
    fontsize=10
)


# ============================================================
# LAYOUT
# ============================================================

plt.subplots_adjust(
    left=0.02,
    right=0.98,
    top=0.88,
    bottom=0.16,
    wspace=0.04
)


# ============================================================
# SAVE
# ============================================================

output_file = (
    OUT
    / "figure1_three_panel_validation.png"
)


plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)


plt.close()


print(f"\nSaved: {output_file}")
print("Done.")