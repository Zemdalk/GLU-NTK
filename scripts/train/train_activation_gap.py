#!/usr/bin/env python3
import os, argparse, random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import SGD, Adam, AdamW

from ntk.models.mlp_mixer import MLPMixer
from ntk.models.vit import ViT



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_optimizer(name, params, lr, wd):
    name = name.lower()
    if name == "adam":
        return Adam(params, lr=lr, weight_decay=wd)
    elif name == "adamw":
        return AdamW(params, lr=lr, weight_decay=wd)
    return SGD(params, lr=lr, momentum=0.9, weight_decay=wd)



def load_gaussian(N, D, batch_size):
    x = torch.randn(N, D)
    y = torch.randint(low=0, high=2, size=(N,))
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    return loader, None, (D, 1, 1), 2

def load_mnist(batch_size):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tr = datasets.MNIST("./data", train=True, download=True, transform=tf)
    te = datasets.MNIST("./data", train=False, download=True, transform=tf)
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True),
        DataLoader(te, batch_size=batch_size, shuffle=False),
        (1, 28, 28),
        10
    )

def load_cifar10(batch_size):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.1)
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    tr = datasets.CIFAR10("./data", train=True, download=True, transform=train_tf)
    te = datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True),
        DataLoader(te, batch_size=batch_size, shuffle=False),
        (3, 32, 32),
        10
    )

def load_tiny(batch_size):
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.2)
    ])

    test_tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    tr = datasets.ImageFolder("./data/tiny-imagenet-200/train", transform=train_tf)
    te = datasets.ImageFolder("./data/tiny-imagenet-200/val",   transform=test_tf)
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True),
        DataLoader(te, batch_size=batch_size, shuffle=False),
        (3, 64, 64),
        200
    )



def train_one(model, tr, te, steps, lr, wd, device,
              dataset, optimizer, num_classes, exp_tag):
    """exp_tag distinguishes activation/model variants in output filenames."""
    model = model.to(device)
    opt = select_optimizer(optimizer, model.parameters(), lr, wd)
    loss_fn = nn.CrossEntropyLoss()

    train_curve, test_curve = [], []

    for step in range(steps):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device).long()
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        train_losses = []
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device).long()
            with torch.no_grad():
                train_losses.append(loss_fn(model(xb), yb).item())
        train_curve.append(np.mean(train_losses))

        test_losses = []
        if te is not None:
            for xb, yb in te:
                xb, yb = xb.to(device), yb.to(device).long()
                with torch.no_grad():
                    test_losses.append(loss_fn(model(xb), yb).item())
            test_curve.append(np.mean(test_losses))

        if step % 1 == 0:
            print(f"[{dataset}][{exp_tag}] step={step} train={train_curve[-1]:.4f}")

    os.makedirs("logs_gengap_aug", exist_ok=True)
    csv = f"logs_gengap_aug/{dataset}_{exp_tag}_{optimizer}_lr{lr}_ep{steps}.csv"
    pd.DataFrame(
        {"epoch": np.arange(steps), "train": train_curve, "test": test_curve}
    ).to_csv(csv, index=False)
    print("Saved CSV:", csv)
    return train_curve, test_curve



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mixer",
                        choices=["mixer", "vit"])
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["gauss", "mnist", "cifar10", "tiny"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["sgd", "adam", "adamw"])
    parser.add_argument("--patch", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--activations", nargs=2,
                        default=["relu", "reglu"])
    args = parser.parse_args()

    set_seed(args.seed)
    device = f"cuda:{args.gpu}"
    print("Using device:", device)

    if args.dataset == "gauss":
        tr, te, (C, H, W), num_classes = load_gaussian(2000, 75, args.batch_size)
    elif args.dataset == "mnist":
        tr, te, (C, H, W), num_classes = load_mnist(args.batch_size)
    elif args.dataset == "cifar10":
        tr, te, (C, H, W), num_classes = load_cifar10(args.batch_size)
    else:
        tr, te, (C, H, W), num_classes = load_tiny(args.batch_size)

    curves = {}

    for act in args.activations:
        if args.model == "mixer":
            model = MLPMixer(
                image_size=(H, W),
                channels=C,
                patch_size=args.patch,
                dim=args.dim,
                depth=args.depth,
                num_classes=num_classes,
                token_hidden_type=act,
                channel_hidden_type=act,
            )
        else:
            model = ViT(
                image_size=(H, W),
                patch_size=args.patch,
                num_classes=num_classes,
                dim=args.dim,
                depth=args.depth,
                mlp_dim=args.dim * 4,
                heads=8,
                channels=C,
                pool="cls",
                hidden_type=act,
            )

        exp_tag = f"{args.model}_{act}"
        curve, _ = train_one(
            model, tr, te,
            args.steps, args.lr, args.wd,
            device, args.dataset,
            args.optimizer, num_classes,
            exp_tag
        )
        curves[act] = curve

    os.makedirs("logs_gengap_aug_fig", exist_ok=True)
    plt.figure(figsize=(10, 6))
    for act, curve in curves.items():
        plt.plot(curve, label=act)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("train loss")
    plt.grid(alpha=0.4)
    plt.legend()
    fig = (
        f"logs_gengap_aug_fig/"
        f"{args.dataset}_{args.model}_{args.optimizer}_"
        f"{args.activations}_lr{args.lr}_ep{args.steps}.pdf"
    )
    plt.savefig(fig)
    print("Saved figure:", fig)
