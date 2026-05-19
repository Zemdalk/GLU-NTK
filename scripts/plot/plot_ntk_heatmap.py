import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

from ntk.models.mlp import MLP
from ntk.utils import compute_ntk

mpl.rcParams.update({
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})

seed = 0
n = 50
d = 100
hidden_dim = 400
num_classes = 1

torch.set_default_dtype(torch.float64)
torch.manual_seed(seed)
np.random.seed(seed)

X = torch.randn(n, d)

cases = {
    "ReLU NTK":  "ReLU",
    "ReGLU NTK": "ReGLU",
}

ntk_mats = {}

for name, act in cases.items():
    model = MLP(
        in_dim=d,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        hidden_type=act,
        init_type="lecun",
    )
    model.eval()

    K = compute_ntk(model, X).cpu().numpy()
    ntk_mats[name] = K

vmin = min(K.min() for K in ntk_mats.values())
vmax = max(K.max() for K in ntk_mats.values())

fig = plt.figure(figsize=(14, 6))
gs = GridSpec(
    1, 3,
    width_ratios=[1, 1, 0.05],
    wspace=0.25,
)

ax_relu  = fig.add_subplot(gs[0, 0])
ax_reglu = fig.add_subplot(gs[0, 1])
cax      = fig.add_subplot(gs[0, 2])

axes = [ax_relu, ax_reglu]

for ax, (name, K) in zip(axes, ntk_mats.items()):
    eigs = np.linalg.eigvalsh(K)
    lam_min, lam_max = eigs[0], eigs[-1]
    cond = lam_max / lam_min

    im = ax.imshow(
        K,
        vmin=vmin,
        vmax=vmax,
        cmap="magma",
        origin="upper",   
    )
    
    ax.set_xticks([])
    ax.set_yticks([])


cbar = fig.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=20)

plt.savefig("heatmap_teaser.svg", bbox_inches='tight')
plt.show()
