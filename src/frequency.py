import matplotlib.pyplot as plt
import numpy as np

classes = ['Very\nlow', 'Low', 'Moderate', 'High', 'Very\nhigh']
fr_all =     [0.118, 0.536, 1.087, 2.350, 4.407]
fr_inside =  [0.683, 0.585, 0.596, 0.942, 1.244]
fr_outside = [0.217, 0.833, 1.818, 3.328, 4.575]

x = np.arange(5)
w = 0.25

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
ax.bar(x - w, fr_all, w, label='All 2025 flooding', color='#4393c3')
ax.bar(x, fr_inside, w, label='Inside historical inventory', color='#2166ac')
ax.bar(x, fr_outside, w, label='Outside historical inventory', color='#d73027')
ax.axhline(y=1.0, color='grey', linestyle='--', linewidth=0.8, label='FR = 1 (random)')
ax.set_ylabel('Frequency ratio')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_xlabel('Susceptibility class')
ax.legend(fontsize=8)
ax.set_ylim(0, 5.2)
plt.tight_layout()
plt.savefig('outputs/figures/figure2_frequency_ratio.png', dpi=300, bbox_inches='tight')
print("Done.")