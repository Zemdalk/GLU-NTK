import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder


def gaussian_data(N=2000, D=75, batch_size=128, loss_type="mse"):
    x = torch.randn(N, D)
    w = torch.randn(D, 1)
    y = x @ w + 0.1 * torch.randn(N, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    return loader, None, D, 1


def mnist_data(batch_size, loss_type="mse"):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST("./data", train=True, download=True, transform=tf)
    test  = datasets.MNIST("./data", train=False, download=True, transform=tf)

    if loss_type == "mse":
        def to_reg(ds):
            xs, ys = [], []
            for x, y in ds:
                xs.append(x.view(-1))
                ys.append(torch.tensor([float(y)]))
            return torch.stack(xs), torch.stack(ys)

        xtr, ytr = to_reg(train)
        xte, yte = to_reg(test)
        return (DataLoader(TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True),
                DataLoader(TensorDataset(xte, yte), batch_size=batch_size, shuffle=False),
                28 * 28, 1)

    else:
        return (
            DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(test, batch_size=batch_size, shuffle=False),
            (1, 28, 28), 10
        )


def cifar10_data(batch_size, loss_type="mse"):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.247, 0.243, 0.261)
        )
    ])
    train = datasets.CIFAR10("./data", train=True, download=True, transform=tf)
    test  = datasets.CIFAR10("./data", train=False, download=True, transform=tf)

    if loss_type == "mse":
        def to_reg(ds):
            xs, ys = [], []
            for x, y in ds:
                xs.append(x.view(-1))
                ys.append(torch.tensor([float(y)]))
            return torch.stack(xs), torch.stack(ys)

        xtr, ytr = to_reg(train)
        xte, yte = to_reg(test)
        return (
            DataLoader(TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True),
            DataLoader(TensorDataset(xte, yte), batch_size=batch_size, shuffle=False),
            3 * 32 * 32,
            1
        )
    else:
        return (
            DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(test, batch_size=batch_size, shuffle=False),
            (3, 32, 32),
            10
        )


def tiny_imagenet_data(batch_size, loss_type="mse"):
    data_dir = "./data/tiny-imagenet-200"
    tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])
    train = ImageFolder(os.path.join(data_dir, "train"), transform=tf)
    val   = ImageFolder(os.path.join(data_dir, "val"),   transform=tf)

    if loss_type == "mse":
        def to_reg(ds):
            xs, ys = [], []
            for x, y in ds:
                xs.append(x.view(-1))
                ys.append(torch.tensor([float(y)]))
            return torch.stack(xs), torch.stack(ys)

        xtr, ytr = to_reg(train)
        xte, yte = to_reg(val)
        return (
            DataLoader(TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True),
            DataLoader(TensorDataset(xte, yte), batch_size=batch_size, shuffle=False),
            3 * 64 * 64,
            1
        )
    else:
        return (
            DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(val, batch_size=batch_size, shuffle=False),
            (3, 64, 64),
            200
        )
