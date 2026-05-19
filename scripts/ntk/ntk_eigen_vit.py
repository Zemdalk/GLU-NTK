import torch
from torch.func import functional_call, jacrev, vmap

from ntk.models.vit import ViT


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

N = 64
image_size = 32
patch = 4
channels = 3
dim = 4096
depth = 4
heads = 4
mlp_dim = 256
num_classes = 1

activations = ["relu", "reglu", "gelu", "geglu", "silu", "swiglu"]

X = torch.randn(N, channels, image_size, image_size, device=device)


def compute_ntk(model, x):
    model.eval().to(device)

    params = {k: v.detach().clone() for k, v in model.named_parameters()}

    def f(current_params, sample):
        out = functional_call(model, current_params, (sample.unsqueeze(0),))
        return out.squeeze()

    dfdtheta = jacrev(f, argnums=0)
    jac_fn = vmap(dfdtheta, (None, 0))
    jacs = jac_fn(params, x)

    grads = []
    for g in jacs.values():
        if g is None:
            continue
        grads.append(g.reshape(N, -1))

    flat_j = torch.cat(grads, dim=1)
    k = flat_j @ flat_j.t()
    eigs = torch.linalg.eigvalsh(k)

    return eigs.cpu()


def effective_rank(eigs, tol=1e-6):
    return (eigs > tol * eigs[-1]).sum().item()


for act in activations:
    model = ViT(
        image_size=image_size,
        patch_size=patch,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        hidden_type=act,
        channels=channels,
    )

    eigs = compute_ntk(model, X)
    cond = float(eigs[-1] / max(eigs[0], 1e-12))
    rank = effective_rank(eigs)

    print(f"\n==== ViT NTK | Act: {act} ====")
    print("Top 5 eigenvalues:", eigs[-5:].flip(dims=[0]))
    print("Bottom 5 eigenvalues:", eigs[:5])
    print(f"Condition number: {cond:.2e}")
