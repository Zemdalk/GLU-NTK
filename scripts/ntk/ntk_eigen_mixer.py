import torch
from torch.func import functional_call, jacrev, vmap

from ntk.models.mlp_mixer import MLPMixer


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

N = 32
image_size = 32
channels = 3
patch_size = 4
dim = 256
num_classes = 1
depth = 4

init_type = "xavier"
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

    eigvals = torch.linalg.eigvalsh(k)
    return eigvals.cpu()


def effective_rank(eigs, tol=1e-6):
    return (eigs > tol * eigs[-1]).sum().item()


for act in activations:
    model = MLPMixer(
        image_size=(image_size, image_size),
        channels=channels,
        patch_size=patch_size,
        dim=dim,
        depth=depth,
        num_classes=num_classes,
        token_hidden_type=act,
        channel_hidden_type=act,
        token_init_type=init_type,
        channel_init_type=init_type,
    )

    eigs = compute_ntk(model, X)
    cond = float(eigs[-1] / max(eigs[0], 1e-12))
    rank = effective_rank(eigs)

    print(f"\n==== Mixer NTK | Activation: {act} ====")
    print("Top 5 eigenvalues:", eigs[-5:])
    print("Bottom 5 eigenvalues:", eigs[:5])
    print(f"Condition number: {cond:.2e}")
    print(f"Effective Rank: {rank}/{len(eigs)}")
