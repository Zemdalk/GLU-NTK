import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path
from scipy.stats import energy_distance

mpl.rcParams.update({
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "lines.linewidth": 3.5,
})

colors = {
    "1": {"dark": "#FF912F"},
    "2": {"dark": "#6A8EC9"},
}

DATASET, MODEL = "tiny", "mixer"
ACT_GLU, ACT_NONGLU = "ReGLU", "ReLU"
LR = {"adamw": "0.001", "sgd": "0.005"}
EP = 100
E_START, E_END = 0, 25 

def get_permutation_p(data1, data2, n_perm=1000):
    n1 = len(data1)
    combined = np.vstack([data1, data2])
    obs_dist = energy_distance(data1.flatten(), data2.flatten())
    
    hits = 0
    for _ in range(n_perm):
        idx = np.random.permutation(len(combined))
        if energy_distance(combined[idx[:n1]].flatten(), combined[idx[n1:]].flatten()) >= obs_dist:
            hits += 1
    return obs_dist, hits / n_perm

fig, ax = plt.subplots(figsize=(7, 5))
group_data = {}

combo_color = {
    (ACT_NONGLU.lower(), "sgd"): colors["2"]["dark"],
    (ACT_GLU.lower(), "sgd"): colors["1"]["dark"],
}

opt='sgd'
for act_name in [ACT_GLU, ACT_NONGLU]:
    f_path = f"./logs_gengap_aug/{DATASET}_{MODEL}_{act_name.lower()}_{opt}_lr{LR[opt]}_ep{EP}.csv"
    if not Path(f_path).exists(): continue
    
    df = pd.read_csv(f_path)
    mask = (df["epoch"] >= E_START) & (df["epoch"] <= (E_END or 1e9))
    df = df[mask]
    
    x = df["train"].values
    y = (df["test"] - df["train"]).values
    points = np.column_stack((x, y))
    group_data[(act_name, opt)] = points 
    
    ax.scatter(x, y, s=30, alpha=0.85, 
                color=combo_color.get((act_name.lower(), opt)), 
                label=f"{act_name}")

ax.set_xlabel(r"$L_{train}$", fontsize=22)
ax.set_ylabel(r"$L_{test}-L_{train}$", fontsize=22)
ax.set_title("MLP Mixer on Tiny ImageNet", fontsize=20)

leg = ax.legend(fontsize=18, loc='upper right', frameon=True)

if (ACT_GLU, "sgd") in group_data and (ACT_NONGLU, "sgd") in group_data:
    dist_sgd, p_val_sgd = get_permutation_p(group_data[(ACT_GLU, "sgd")], 
                                            group_data[(ACT_NONGLU, "sgd")])
    
    plt.draw() 
    bbox = leg.get_window_extent()
    inv = ax.transAxes.inverted()
    bbox_ax = inv.transform(bbox)
    
    stat_text = f"p-value: {p_val_sgd:.4f}"
    ax.text(bbox_ax[1, 0], bbox_ax[0, 1] - 0.02, stat_text, 
            transform=ax.transAxes,
            fontsize=18, fontweight='bold', 
            verticalalignment='top', horizontalalignment='right')
    
    print(f"SGD Test: Dist={dist_sgd:.4f}, P={p_val_sgd:.4f}")

plt.tight_layout()
plt.savefig(f"train_vs_gap_{MODEL}_{DATASET}_{ACT_NONGLU}+test.pdf", bbox_inches="tight")
plt.show()