import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from ntk.models.mlp import MLP, ZeroOutput
from ntk.utils import compute_ntk

seed = 42
torch.set_default_dtype(torch.float64)
torch.manual_seed(seed)
np.random.seed(seed)

n_samples = 2
input_dim = 75

xdata = torch.randn(n_samples, input_dim)
ydata = torch.tensor([0.03, 0.008]).unsqueeze(1)      

print(xdata.norm(dim=1)**2)
center = ydata.numpy().reshape(-1)

hidden_dim = 300
num_classes = 1

lr = 1e-4
steps = 1000
record_every = 1

acts = ["ReLU", "ReGLU"]
colors = {
    "ReLU": {
        "dark": "#1f77b4",
        "middle": "#91CAE8",
        "light": "#c0e0f2",
    },
    "ReGLU": {
        "dark": "#d62728",
        "middle": "#F48892",
        "light": "#f8b9bf",
    },
    "default": {
        "dark": "#8e7fb8",
        "middle": "#c5bddb",
    }
}

def kernel_to_ellipse(K):
    eigvals, eigvecs = np.linalg.eigh(K)
    v = eigvecs[:, 0]
    angle = np.degrees(np.arctan2(v[1], v[0]))
    width = 1.0 / eigvals[0]
    height = 1.0 / eigvals[1]
    return width, height, angle


def print_kernel_info(act, step, K):
    eigvals = np.linalg.eigvalsh(K)
    print(
        f"[{act.upper():5s}] step={step:4d}\n"
        f"NTK =\n{K}\n"
        f"eigvals = {eigvals}\n"
    )

fig, ax = plt.subplots(figsize=(7, 5))
final_losses = {}

for act in acts:
    print(f"\n===== {act.upper()} =====")

    model = MLP(
        in_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        hidden_type=act,
        init_type="lecun",
    )
    model = ZeroOutput(model)

    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    ys = []
    kernels = []

    for t in range(steps):
        out = model(xdata)
        loss = loss_fn(out, ydata)

        opt.zero_grad()
        loss.backward()
        opt.step()

        ys.append(out.detach().numpy().reshape(-1))
        
        if t == steps - 1: 
            print("loss: ", loss)
            final_losses[act] = loss.item()
            print("pred labels: ", out.detach().numpy().reshape(-1))

        if t % record_every == 0:
            K = compute_ntk(model, xdata).detach().numpy()
            kernels.append(K)

            if t == record_every:
                print_kernel_info(act, t, K)

    ys = np.array(ys)

    ax.plot(
        ys[:, 0],
        ys[:, 1],
        lw=7,
        color=colors[act]["middle"],
        label=f"{act}",
    )

    ax.scatter(ys[0, 0], ys[0, 1], color=colors["default"]["middle"], s=250, linewidths=4, edgecolor=colors["default"]["dark"], marker="o", zorder=5)
    if steps != 1000:
        ax.scatter(ys[-1, 0], ys[-1, 1], color=colors[act]["middle"], s=250, linewidths=4, edgecolor=colors[act]["dark"], marker="o", zorder=6)

    for K in kernels:
        w, h, ang = kernel_to_ellipse(K)
        e = Ellipse(
            xy=center,
            width=w,
            height=h,
            angle=ang,
            fill=False,
            alpha=0.5,
            lw=5,
            linestyle="--",
            color=colors[act]["light"],
        )
        e.set_alpha(0.25)
        ax.add_patch(e)

ax.scatter(ydata[0], ydata[1], color=colors["default"]["middle"], s=600, linewidths=3.5, edgecolor=colors["default"]["dark"], marker="*", zorder=7)


text_lines = [
    f"{act}:  ℓ = {final_losses[act]:.2e}" if steps != 1000 else f"{act}:  ℓ = 0"
    for act in acts
]
text = "\n".join(text_lines)
ax.text(
    0.98, 0.03,               
    text,
    transform=ax.transAxes,   
    fontsize=25,
    fontweight="bold",
    verticalalignment="bottom",
    horizontalalignment="right",
    bbox=dict(
        boxstyle="round,pad=0.3",
        facecolor="white",
        edgecolor="gray",
        alpha=0.85,
    ),
)


ax.set_title(f"{steps} STEPS", fontsize=25, fontweight="bold")
ax.grid(alpha=0.3)

ax.legend(prop={"size": 25, "weight": "bold"})
fig.tight_layout()

plt.savefig(f"imgs/training_dynamics_{steps}.pdf")
plt.show()
