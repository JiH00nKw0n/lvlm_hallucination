"""
Temporary script to replot results with adjusted y-axis ranges.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path

output_dir = "analysis_color_noise"

# Load JSON files
with open(f"{output_dir}/exp1_hue_detailed.json", "r") as f:
    hue_detailed = json.load(f)

with open(f"{output_dir}/exp2_blend_detailed.json", "r") as f:
    blend_detailed = json.load(f)


# Process hue results (skip 0, start from 30)
hue_results = {}
for key, values in hue_detailed.items():
    hue_val = int(key)
    if hue_val >= 30:  # Skip 0
        logprobs = [v["logprob"] for v in values]
        hue_results[hue_val] = logprobs

hue_avg = {k: np.mean(v) for k, v in hue_results.items()}
hue_std = {k: np.std(v) for k, v in hue_results.items()}

# Process blend results
blend_results = {}
for key, values in blend_detailed.items():
    blend_val = float(key)
    logprobs = [v["logprob"] for v in values]
    blend_results[blend_val] = logprobs

blend_avg = {k: np.mean(v) for k, v in blend_results.items()}
blend_std = {k: np.std(v) for k, v in blend_results.items()}

# Process noise results

# Helper function to get y-axis range
def get_ylim(avg_dict, std_dict):
    """Get y-axis limits: floor to nearest 0.25 multiple for min, ceil to nearest 0.25 multiple for max"""
    all_avgs = list(avg_dict.values())

    y_min = math.floor(min(all_avgs) / 0.25) * 0.25
    y_max = math.ceil(max(all_avgs) / 0.25) * 0.25

    return y_min, y_max

# Plot 1: Hue range effect
fig, ax = plt.subplots(figsize=(10, 6))
hues = sorted(hue_avg.keys())
avgs = [hue_avg[h] for h in hues]
stds = [hue_std[h] for h in hues]

ax.plot(hues, avgs, marker='o', linewidth=2, markersize=8, label='Mean logprob')
ax.fill_between(hues, [a - s for a, s in zip(avgs, stds)], [a + s for a, s in zip(avgs, stds)], alpha=0.3)

# Set y-axis limits
y_min, y_max = get_ylim(hue_avg, hue_std)
ax.set_ylim(y_min, y_max)

ax.set_xlabel('Hue Range Value', fontsize=12)
ax.set_ylabel('Log Probability', fontsize=12)
ax.set_title('Effect of Hue Range on Object Recognition\n(blend_strength=1.0)', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(f"{output_dir}/hue_range_effect.png", dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir}/hue_range_effect.png (y range: [{y_min}, {y_max}])")
plt.close()

# Plot 2: Blend strength effect
fig, ax = plt.subplots(figsize=(10, 6))
blends = sorted(blend_avg.keys())
avgs = [blend_avg[b] for b in blends]
stds = [blend_std[b] for b in blends]

ax.plot(blends, avgs, marker='o', linewidth=2, markersize=8, label='Mean logprob', color='coral')
ax.fill_between(blends, [a - s for a, s in zip(avgs, stds)], [a + s for a, s in zip(avgs, stds)], alpha=0.3, color='coral')

# Set y-axis limits
y_min, y_max = get_ylim(blend_avg, blend_std)
ax.set_ylim(y_min, y_max)

ax.set_xlabel('Blend Strength', fontsize=12)
ax.set_ylabel('Log Probability', fontsize=12)
ax.set_title('Effect of Blend Strength on Object Recognition\n(hue_range=(180, 180))', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(f"{output_dir}/blend_strength_effect.png", dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir}/blend_strength_effect.png (y range: [{y_min}, {y_max}])")
plt.close()

print("\nDone! All plots updated with adjusted y-axis ranges.")