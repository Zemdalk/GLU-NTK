import torch
from torch.func import functional_call, jacrev, vmap

from ntk.models.mlp import MLP


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(160)

N = 50
in_dim = 200
hidden_dim = 800
num_classes = 1
init_type = "lecun"
activations = ["relu", "reglu"]

X = torch.randn(N, in_dim, device=device)


def compute_ntk(model, x):
    model.eval().to(device)

    params = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def f(current_params, sample):
        out = functional_call(model, current_params, (sample.unsqueeze(0),))
        return out.squeeze(0).squeeze(-1)

    jac_fn = vmap(jacrev(f), (None, 0))
    jacs = jac_fn(params, x)

    flat_j = torch.cat([g.reshape(N, -1) for g in jacs.values()], dim=1)
    k = flat_j @ flat_j.t()

    eigvals = torch.linalg.eigvalsh(k.cpu())
    return eigvals


for act in activations:
    model = MLP(
        in_dim,
        hidden_dim,
        num_classes,
        hidden_type=act,
        init_type=init_type,
    )

    eigvals = compute_ntk(model, X)

    print(f"\n==== Activation: {act}, Init: {init_type} ====")
    print("Top 5 largest eigenvalues:", eigvals[-5:].flip(dims=[0]))
    print("Top 5 smallest eigenvalues:", eigvals[:5])
    cond = (eigvals[-1] / eigvals[0]).item()
    print(f"Condition number: {cond:.2f}")
