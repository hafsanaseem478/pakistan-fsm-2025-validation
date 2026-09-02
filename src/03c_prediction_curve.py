import matplotlib.pyplot as plt

cum_area_all =  [0, 12.84, 21.15, 30.81, 43.26, 100]
cum_flood_all = [0, 56.60, 76.13, 86.63, 93.30, 100]

cum_area_out =  [0, 7.90, 14.77, 23.85, 36.75, 100]
cum_flood_out = [0, 36.14, 59.00, 75.52, 86.27, 100]

fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=200)

ax.plot(
    cum_area_all, cum_flood_all,
    'o-', color='#4393c3',
    label='All 2025 flooding (AUC = 0.830)',
    linewidth=1.8, markersize=5
)

ax.plot(
    cum_area_out, cum_flood_out,
    's-', color='#d73027',
    label='Outside historical inventory (AUC = 0.801)',
    linewidth=1.8, markersize=5
)

ax.plot(
    [0, 100], [0, 100],
    '--', color='grey',
    linewidth=1,
    label='Random (AUC = 0.500)'
)

ax.set_title('Cumulative Capture of the 2025 Flood Footprint', fontsize=13, fontweight='bold')
ax.set_xlabel('Cumulative area (%)')
ax.set_ylabel('Cumulative flood captured (%)')

ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_aspect('equal')

ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.7)
ax.set_axisbelow(True)

# annotate key points
ax.annotate(
    '21.15% area\ncaptures 76.13% flood',
    xy=(21.15, 76.13),
    xytext=(28, 68),
    fontsize=8,
    arrowprops=dict(arrowstyle='->', lw=0.8, color='black')
)

ax.annotate(
    '14.77% area\ncaptures 59.00% flood',
    xy=(14.77, 59.00),
    xytext=(38, 45),
    fontsize=8,
    arrowprops=dict(arrowstyle='->', lw=0.8, color='black')
)

ax.legend(fontsize=8, loc='lower right', frameon=False)

plt.tight_layout()
plt.savefig('outputs/figures/figure3_prediction_curve.png', dpi=300, bbox_inches='tight')
plt.show()

print("Done.")