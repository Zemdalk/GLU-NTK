import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

mpl.rcParams.update({
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "lines.linewidth": 3.5,
})

colors = {
    "1": {"dark": "#42a3d7", "light": "#91CAE8"},
    "2": {"dark": "#ee4455", "light": "#F48892"},
    "3": {"dark": "#8e7fb8", "light": "#c5bddb"},
}

color_map = {
    "relu":  colors["1"]["light"],
    "reglu": colors["1"]["dark"],
    "gelu":  colors["2"]["light"],
    "geglu": colors["2"]["dark"],
    "silu":  colors["3"]["light"],
    "swiglu": colors["3"]["dark"],
}

legend_name = {
    "relu":  "ReLU",
    "reglu": "ReGLU",
    "gelu":  "GELU",
    "geglu": "GEGLU",
    "silu":  "SiLU",
    "swiglu": "SwiGLU",
}

glu_group  = ["reglu", "geglu", "swiglu"]
nonglu_group = ["relu", "gelu", "silu"]


log_dir = "logs"
activations = {
    "relu": "mnist_relu_lr1e-05_ep150.csv",
    "reglu": "mnist_reglu_lr1e-05_ep150.csv",
    "gelu": "mnist_gelu_lr1e-05_ep150.csv",
    "geglu": "mnist_geglu_lr1e-05_ep150.csv",
    "silu": "mnist_silu_lr1e-05_ep150.csv",
    "swiglu": "mnist_swiglu_lr1e-05_ep150.csv",
}

plt.figure(figsize=(7, 5))
lw = 3.5
min_epoch = 0
max_epoch = 80

for key in nonglu_group:
    file = activations[key]
    path = os.path.join(log_dir, file)
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[df["epoch"] >= min_epoch]
        df = df[df["epoch"] <= max_epoch]
        plt.plot(df["epoch"], df["train_loss"],
                 linestyle="--",
                 lw=lw,
                 color=color_map[key],
                 label=legend_name[key])
    else:
        print(f"Missing: {path}")

for key in glu_group:
    file = activations[key]
    path = os.path.join(log_dir, file)
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[df["epoch"] >= min_epoch]
        df = df[df["epoch"] <= max_epoch]
        plt.plot(df["epoch"], df["train_loss"],
                 linestyle="-",
                 lw=lw,
                 color=color_map[key],
                 label=legend_name[key])
    else:
        print(f"Missing: {path}")

plt.yscale("log")
plt.xlabel("Epoch", fontsize=25)
plt.ylabel("Train Loss (log)", fontsize=25)

order = ["ReLU", "ReGLU", "GELU", "GEGLU", "SiLU", "SwiGLU"]
handles, labels = plt.gca().get_legend_handles_labels()
ordered_handles = [handles[labels.index(name)] for name in order]
plt.legend(ordered_handles, order, fontsize=17)

plt.tight_layout()

out = "imgs/loss_compare_mlp_mnist_lr1e-05.pdf"
plt.savefig(out, bbox_inches="tight")
print(f"Saved: {out}")

plt.show()
