import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import metmhn.Utilityfunctions as utils

# ---- file paths ----
base_path = "../results/luad/luad_g14_10muts_"
files = [f"{base_path}{i}.csv" for i in [1, 2, 3, 11, 21, 31]]

# ---- load data ----
dfs = [pd.read_csv(f, index_col=0) for f in files]

# assume same structure for all
events_plot = dfs[0].columns.tolist()
n_tot = len(events_plot)

# ---- create figure: 2 row, 6 columns (2 per run) ----
fig, axes = plt.subplots(
    2, 6,
    sharey=True,
    figsize=(16, 7),
    gridspec_kw={
        "width_ratios": [n_tot, 0.8] * 3,  # make colorbar columns a bit smaller
        "wspace": 0.0,  # small spacing between plots
        "hspace": 0.3   # slightly reduced vertical spacing
    }
)

# ---- plot each dataset ----
for i, df in enumerate(dfs):
    th_plot = df.values

    row = i // 3          # 0 or 1
    col_pair = (i % 3) * 2  # 0, 2, 4

    ax1 = axes[row, col_pair]
    ax2 = axes[row, col_pair + 1]

    utils.plot_theta(
        ax1, ax2,
        th_plot,
        events_plot,
        alpha=0.2,
        font_size=5
    )

    ax1.set_title(f"Penalty {i+1}")

# ---- manually force tight outer margins ----
fig.subplots_adjust(
    left=0.01,    # almost no left margin
    right=0.95,
    bottom=0.03,  # almost no bottom margin
    top=0.85
)

plt.show()