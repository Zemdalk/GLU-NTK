import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches

df = pd.read_csv("ntk_condition_numbers_mixer.csv")

non_glu_map = {
    "ReLU": "relu",
    "GELU": "gelu",
    "SiLU": "silu",
}

glu_map = {
    "ReLU": "reglu",
    "GELU": "geglu",
    "SiLU": "swiglu",
}

groups = list(non_glu_map.keys())

seed_cols = [c for c in df.columns if c.startswith("seed_")]

non_glu_means = [
    df.loc[df["activation"] == non_glu_map[g], "mean"].values[0]
    for g in groups
]
glu_means = [
    df.loc[df["activation"] == glu_map[g], "mean"].values[0]
    for g in groups
]

non_glu_seeds = [
    df.loc[df["activation"] == non_glu_map[g], seed_cols].values.flatten()
    for g in groups
]
glu_seeds = [
    df.loc[df["activation"] == glu_map[g], seed_cols].values.flatten()
    for g in groups
]

x = np.arange(len(groups))
width = 0.32
gap = 0.03

plt.rcParams.update({
    "font.size": 10,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
})

color_non_glu = "#d4cee4"
color_glu     = "#cee3d5"
edge_color_non_glu = "#8E7FB8"
edge_color_glu = "#7db58e"

fig, ax = plt.subplots(figsize=(6, 4.8))
        
bars1 = ax.bar(
    x - width/2 - gap, non_glu_means, width,
    label="Non-GLU",
    color=color_non_glu,
    edgecolor=edge_color_non_glu,
    linewidth=3.0,
    joinstyle='round',
    zorder=2
)

bars2 = ax.bar(
    x + width/2 + gap, glu_means, width,
    label="GLU",
    color=color_glu,
    edgecolor=edge_color_glu,
    linewidth=3.0,
    joinstyle='round',
    zorder=2
)

rng = np.random.default_rng(42)
jitter_scale = 0.04

for i, vals in enumerate(non_glu_seeds):
    jitter = rng.normal(0, jitter_scale, size=len(vals))
    ax.scatter(
        np.full(len(vals), x[i] - width/2 - gap) + jitter,
        vals,
        s=120, alpha=0.60,
        edgecolors=edge_color_non_glu,
        linewidths=2.2,
        color=color_non_glu,
        zorder=5
    )

for i, vals in enumerate(glu_seeds):
    jitter = rng.normal(0, jitter_scale, size=len(vals))
    ax.scatter(
        np.full(len(vals), x[i] + width/2 + gap) + jitter,
        vals,
        s=120, alpha=0.60,
        edgecolors=edge_color_glu,
        linewidths=2.2,
        color=color_glu,
        zorder=5
    )

ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=18)
ax.set_xlabel("FFN Structure", fontsize=18)
ax.set_ylabel("Condition Number", fontsize=18)
ax.set_ylim(0, 14000)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    loc="upper right",
    bbox_to_anchor=(1.0, 1.0),
    frameon=True,
    fontsize=18
)

plt.tight_layout()
plt.savefig("condition_number_mixer+scatter.pdf", bbox_inches="tight")
plt.show()
