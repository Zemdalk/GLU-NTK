import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "lines.linewidth": 3.5,
})

colors = {
    "ReLU": {
        "dark": "#1f77b4",
        "middle": "#91CAE8",
        "shallow": "#c0e0f2",
    },
    "ReGLU": {
        "dark": "#d62728",
        "middle": "#F48892",
        "shallow": "#f8b9bf",
    },
}

csv_file = "condition_number.csv"
df = pd.read_csv(csv_file)
d = df["d"]


plt.figure(figsize=(9, 6))
lw = 5.5
plt.plot(
    d, df["relu_cond_theory"],
    "--", lw=lw, color=colors["ReLU"]["middle"],
    label="ReLU (theory)"
)
plt.plot(
    d, df["relu_cond_numeric"],
    "-", lw=lw, color=colors["ReLU"]["dark"],
    label="ReLU (numeric)"
)

plt.plot(
    d, df["reglu_cond_theory"],
    "--", lw=lw, color=colors["ReGLU"]["middle"],
    label="ReGLU (theory)"
)
plt.plot(
    d, df["reglu_cond_numeric"],
    "-", lw=lw, color=colors["ReGLU"]["dark"],
    label="ReGLU (numeric)"
)

plt.yscale("log")
plt.xlabel("Input dimension", fontsize=20)
plt.ylabel("Condition number (log)", fontsize=20)

plt.grid(True, which="both", alpha=0.3)
plt.legend(fontsize=20)
plt.tight_layout()

plt.savefig("imgs/condition_number.pdf", bbox_inches='tight')
plt.show()
