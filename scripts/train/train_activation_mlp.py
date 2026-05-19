import os
import argparse
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from ntk.models.mlp import MLP
from ntk.utils import compute_ntk


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def gaussian_data(N=2000, D=75, batch_size=128):
    x = torch.randn(N, D)
    w = torch.randn(D, 1)
    y = x @ w + 0.1 * torch.randn(N, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    return loader, None, D, 1


def mnist_data(batch_size):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(root="./data", train=True, download=True, transform=tf)
    test  = datasets.MNIST(root="./data", train=False, download=True, transform=tf)

    def to_reg(ds):
        xs, ys = [], []
        for x, y in ds:
            xs.append(x.view(-1))
            ys.append(torch.tensor([float(y)]))
        return torch.stack(xs), torch.stack(ys)

    xtr, ytr = to_reg(train)
    xte, yte = to_reg(test)
    train_loader = DataLoader(TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(xte, yte), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, 28*28, 1

def cifar10_data(batch_size):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train = datasets.CIFAR10(root="./data", train=True, download=True, transform=tf)
    test  = datasets.CIFAR10(root="./data", train=False, download=True, transform=tf)

    def to_reg(ds):
        xs, ys = [], []
        for x, y in ds:
            xs.append(x.view(-1))
            ys.append(torch.tensor([float(y)]))
        return torch.stack(xs), torch.stack(ys)

    xtr, ytr = to_reg(train)
    xte, yte = to_reg(test)
    tr_loader = DataLoader(TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True)
    te_loader = DataLoader(TensorDataset(xte, yte), batch_size=batch_size, shuffle=False)
    return tr_loader, te_loader, 3*32*32, 1



def compute_init_ntk_stats(model, xb):
    K = compute_ntk(model, xb).cpu().numpy()
    eigvals = np.linalg.eigvalsh(K)
    lam_min, lam_max = eigvals[0], eigvals[-1]
    cond = lam_max / max(lam_min, 1e-12)
    return lam_max, lam_min, cond


def train_one(model, train_loader, test_loader, steps, lr, device, activation_name, dataset_name):
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    xb_probe, yb_probe = next(iter(train_loader))
    xb_probe = xb_probe[:50].to(device)

    lam_max, lam_min, cond = compute_init_ntk_stats(model, xb_probe)
    print(f"[{activation_name}] init NTK: λmax={lam_max:.2e}, λmin={lam_min:.2e}, cond={cond:.2e}")

    train_curve, test_curve = [], []

    for step in range(steps):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad():
                pred = model(xb)
                train_losses.append(loss_fn(pred, yb).item())
        train_curve.append(np.mean(train_losses))

        if test_loader is not None:
            test_losses = []
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                with torch.no_grad():
                    pred = model(xb)
                    test_losses.append(loss_fn(pred, yb).item())
            test_curve.append(np.mean(test_losses))

        if step % 10 == 0 or step == steps - 1:
            print(f"[{activation_name}] step {step:4d} | train={train_curve[-1]:.4f}")

    os.makedirs("./logs", exist_ok=True)
    csv_name = f"logs/{dataset_name}_{activation_name}_lr{lr}_ep{steps}.csv"
    df = pd.DataFrame({
        "epoch": np.arange(steps),
        "train_loss": train_curve,
        "test_loss": test_curve if (test_loader is not None) else np.nan
    })
    df.to_csv(csv_name, index=False)
    print(f"Saved CSV → {csv_name}")

    return train_curve, test_curve


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="gauss", choices=["gauss", "mnist", "cifar10"])
    parser.add_argument("--activations", nargs=2, default=["relu", "reglu"])
    parser.add_argument("--gauss_num", type=int, default=2000)
    parser.add_argument("--gauss_dim", type=int, default=75)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = f"cuda:{args.gpu}"
    print(f"Using device: cuda:{args.gpu}")

    if args.dataset == "gauss":
        tr, te, Din, Cout = gaussian_data(args.gauss_num, args.gauss_dim, args.batch_size)
    elif args.dataset == "mnist":
        tr, te, Din, Cout = mnist_data(args.batch_size)
    else:
        tr, te, Din, Cout = cifar10_data(args.batch_size)


    curves = {}
    for act in args.activations:
        model = MLP(
            in_dim=Din,
            hidden_dim=args.hidden_dim,
            num_classes=Cout,
            hidden_type=act,
            init_type="lecun",
        )
        train_curve, test_curve = train_one(model, tr, te, args.steps, args.lr, device, act, args.dataset)
        curves[act] = train_curve

