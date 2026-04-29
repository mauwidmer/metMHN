import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import metmhn.Utilityfunctions as utils

# ---- file path (single file only) ----
file_path = "../results/luad/luad_g14_10muts_41.csv"

# ---- load data ----
df = pd.read_csv(file_path, index_col=0)

events_plot = df.columns.tolist()
th_plot = df.values

# ---- create figure for one plot only ----
fig, axes = plt.subplots(
    1, 2,
    sharey=True,
    figsize=(6, 6),
    gridspec_kw={
        "width_ratios": [len(events_plot), 1],
        "wspace": 0
    }
)

ax1, ax2 = axes

# ---- plot ----
utils.plot_theta(
    ax1,
    ax2,
    th_plot,
    events_plot,
    alpha=0.2,
    font_size=8
)

ax1.set_title("Penalty 8")

plt.tight_layout()
plt.show()