import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import metmhn.Utilityfunctions as utils
import os

# set working directory to project root
os.chdir("/Users/maurin/Documents/ETH/MSc/Rotation Beerenwinkel/metMHN/")

print("Current working directory:", os.getcwd())

# ---- file path ----
file_path = "results/brca/brca_g19_10muts.csv"

# ---- load data ----
df = pd.read_csv(file_path, index_col=0)

events_plot = df.columns.tolist()
th_plot = df.values

# ---- optional sanity check ----
print("Theta shape:", th_plot.shape)
print("Number of events:", len(events_plot))

# ---- dynamic figure sizing ----
n_events = len(events_plot)

fig_width = max(7, 0.6 * n_events + 1.5)
fig_height = max(6, 0.6 * n_events)

# ---- create figure ----
fig, axes = plt.subplots(
    1, 2,
    sharey=True,
    figsize=(fig_width, fig_height),
    gridspec_kw={
        "width_ratios": [n_events, 1],
        "wspace": 0.05
    }
)

ax1, ax2 = axes

# ---- plot theta matrix ----
utils.plot_theta(
    ax1,
    ax2,
    th_plot,
    events_plot,
    alpha=0.15,      # slightly lower threshold for visibility
    font_size=10     # easier to read for 10 mutations
)

# ---- cosmetics ----
ax1.set_title("BRCA metMHN (10 mutations)", fontsize=12)

plt.tight_layout()

# ---- save high-resolution figure ----
plt.savefig(
    "results/brca/brca_g19_10muts.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()