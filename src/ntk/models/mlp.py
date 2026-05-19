import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def xavier_(layer):
    nn.init.xavier_normal_(layer.weight)
    nn.init.zeros_(layer.bias)

def lecun_(layer):
    nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")
    nn.init.zeros_(layer.bias)

def kaiming_(layer):
    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    nn.init.zeros_(layer.bias)

def get_init(init="xavier"):
    if init == "xavier":
        return xavier_
    elif init == "lecun":
        return lecun_
    elif init == "kaiming":
        return kaiming_
    else:
        raise ValueError(f"Unknown initialization type: {init}")

class MLP(nn.Module):
    """
    Single hidden layer MLP with selectable hidden transform.
    Output is linear logits to num_classes.
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, hidden_type: str = "relu", init_type: str = "lecun", in_flatten: bool = False):
        super().__init__()
        self.hidden_type = hidden_type.lower()
        self.num_classes = num_classes
        self.in_flatten = in_flatten
        self.init_fn = get_init(init_type)

        if self.hidden_type in ("relu", "gelu", "sigmoid", "silu"):
            self.fc1 = nn.Linear(in_dim, hidden_dim)
        elif self.hidden_type in ("bilinear", "glu", "reglu", "geglu", "swiglu"):
            hidden_dim = int(hidden_dim * (in_dim + num_classes) / (2*in_dim + num_classes) )
            self.fc1_a = nn.Linear(in_dim, hidden_dim)
            self.fc1_b = nn.Linear(in_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown hidden_type: {hidden_type}")

        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        if hasattr(self, "fc1"):
            self.init_fn(self.fc1)
        else:
            self.init_fn(self.fc1_a)
            self.init_fn(self.fc1_b)

        self.init_fn(self.fc2)

    def forward(self, x):
        if self.in_flatten == True:
            x = x.view(x.size(0), -1)

        if self.hidden_type == "relu":
            h = F.relu(self.fc1(x))
        elif self.hidden_type == "gelu":
            h = F.gelu(self.fc1(x))
        elif self.hidden_type == "sigmoid":
            h = torch.sigmoid(self.fc1(x))
        elif self.hidden_type == "silu":
            h = F.silu(self.fc1(x))
        elif self.hidden_type == "glu":
            h = self.fc1_a(x) * torch.sigmoid(self.fc1_b(x))
        elif self.hidden_type == "reglu":
            h = self.fc1_a(x) * F.relu(self.fc1_b(x))
        elif self.hidden_type == "geglu":
            h = self.fc1_a(x) * F.gelu(self.fc1_b(x))
        elif self.hidden_type == "swiglu":
            h = self.fc1_a(x) * F.silu(self.fc1_b(x))
        elif self.hidden_type == "bilinear":
            h = self.fc1_a(x) * self.fc1_b(x)
        else:
            raise RuntimeError("invalid hidden_type")

        logits = self.fc2(h)
        return logits

        
class ZeroOutput(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.init_model = [copy.deepcopy(model).eval()]
        self.model = model

    def forward(self, x):
        return self.model(x) - self.init_model[0](x)