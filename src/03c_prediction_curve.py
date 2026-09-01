from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INTERIM = Path("data/interim")
OUTPUTS = Path("outputs")

FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"

FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)


# Load counts from 03b_frequency_ratio.py
#
# columns:
# 0 = all valid
# 1 = all flooded
# 2 = valid inside historical inventory
# 3 = flooded inside
# 4 = valid outside historical inventory
# 5 = flooded outside

counts = np.load(
    INTERIM / "fr_counts.npy"
)

classes = [5, 4, 3, 2, 1]


def make_curve(valid_col, flood_col, name):

    total_valid = counts[classes, valid_col].sum()
    total_flood = counts[classes, flood_col].sum()

    rows = []

    cumulative_area = 0
    cumulative_flood = 0

    # Start curve at origin
    x = [0]
    y = [0]

    for c in classes:

        area_share = (
            counts[c, valid_col]
            / total_valid
            * 100
        )

        flood_share = (
            counts[c, flood_col]
            / total_flood
            * 100
        )

        cumulative_area += area_share
        cumulative_flood += flood_share

        x.append(cumulative_area)
        y.append(cumulative_flood)

        rows.append({
            "FSM_class": c,
            "area_share_pct": area_share,
            "flood_share_pct": flood_share,
            "cumulative_area_pct": cumulative_area,
            "cumulative_flood_pct": cumulative_flood,
        })

    df = pd.DataFrame(rows)

    # Area under cumulative capture curve
    # NOTE: this is NOT ROC-AUC.
    auc = np.trapezoid(
        np.array(y) / 100,
        np.array(x) / 100
    )

    df.to_csv(
        TABLES / f"{name}_capture_curve.csv",
        index=False
    )

    return np.array(x), np.array(y), auc, df


# Overall

x_all, y_all, auc_all, df_all = make_curve(
    valid_col=0,
    flood_col=1,
    name="overall"
)


# Outside historical inventory

x_out, y_out, auc_out, df_out = make_curve(
    valid_col=4,
    flood_col=5,
    name="historical_nonoverlap"
)


# Plot

plt.figure(figsize=(7, 6))

plt.plot(
    x_all,
    y_all,
    marker="o",
    label="All valid 2025 flood"
)

plt.plot(
    x_out,
    y_out,
    marker="o",
    label="Historical-inventory non-overlap"
)

# Random / proportional reference
plt.plot(
    [0, 100],
    [0, 100],
    linestyle="--",
    label="Proportional reference"
)

plt.xlabel("Cumulative share of valid analysis area (%)")
plt.ylabel("Cumulative share of 2025 flood captured (%)")

plt.title(
    "Cumulative Flood Capture by FSM Susceptibility"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES / "cumulative_flood_capture.png",
    dpi=300
)

plt.close()


print("=" * 65)
print("CUMULATIVE FLOOD CAPTURE")
print("=" * 65)

print("\nOVERALL")
print(df_all.to_string(index=False))
print(f"\nCapture-curve AUC: {auc_all:.3f}")

print("\nHISTORICAL-INVENTORY NON-OVERLAP")
print(df_out.to_string(index=False))
print(f"\nCapture-curve AUC: {auc_out:.3f}")

print("\nSaved:")
print("outputs/figures/cumulative_flood_capture.png")
print("outputs/tables/overall_capture_curve.csv")
print("outputs/tables/historical_nonoverlap_capture_curve.csv")