import torch
import numpy as np

def flatten_grads(grads):
    return torch.cat([g.reshape(-1) for g in grads if g is not None])

def compute_ntk(model, x):
    """
    Empirical NTK matrix on a minibatch x.

    For scalar-output models:
      K_ij = <∇θ f(x_i), ∇θ f(x_j)>

    For vector-output models:
      K_ij = Σ_c <∇θ f_c(x_i), ∇θ f_c(x_j)>
    """
    params = [p for p in model.parameters() if p.requires_grad]
    grads = []

    for i in range(x.shape[0]):
        out_i = model(x[i:i+1]).reshape(-1)
        per_out_grads = []
        for k in range(out_i.numel()):
            model.zero_grad(set_to_none=True)
            retain = k < out_i.numel() - 1
            g = torch.autograd.grad(out_i[k], params, retain_graph=retain)
            per_out_grads.append(flatten_grads(g))
        grads.append(torch.cat(per_out_grads))

    G = torch.stack(grads)
    return G @ G.T


def ntk_condition_number(model, x, eps=1e-12):
    K = compute_ntk(model, x).to(torch.float64)
    eigvals = torch.linalg.eigvalsh(K)
    lam_min = torch.clamp(eigvals[0], min=eps)
    lam_max = eigvals[-1]
    return (lam_max / lam_min).item()
