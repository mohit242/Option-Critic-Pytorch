import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from torch.distributions import Normal
# Adapted from ShantongZhang/DeepRL

DEVICE = torch.device('cpu')
MIN_LOG_STD = -20
MAX_LOG_STD = 2
EPSILON = 1e-6


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_device(device):
    global DEVICE
    DEVICE = device


def layer_init(layer, w_scale=1.0):
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        nn.init.orthogonal_(layer.weight.data)
        layer.weight.data.mul_(w_scale)
        nn.init.constant_(layer.bias.data, 0)
    return layer


def tensor(x):
    """Returns a tensor created from x.


    Args:
        x: array-like object.

    Returns: tensor x

    """
    if x is None or isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float)
    x = torch.tensor(x, device=DEVICE, dtype=torch.float32)
    return x


class FCBody(nn.Module):
    def __init__(self, input_dim, hidden_units=(64, 64), gate=nn.ReLU()):
        super().__init__()
        dims = [input_dim,] + hidden_units
        fc_func = lambda dim_in, dim_out: nn.Linear(dim_in, dim_out)
        gate_func = lambda dim_in, dim_out: gate
        self.net = nn.Sequential(
            *[func(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:]) for func in [fc_func, gate_func]]
        )
        self.feature_dim = dims[-1]
        self.net.apply(layer_init)

    def forward(self, x, action=None):
        if action is not None:
            x = torch.cat([x, action], dim=1)
        return self.net(x)


class VanillaNet(nn.Module):
    def __init__(self, output_dim, body):
        super().__init__()
        self.fc_head = layer_init(nn.Linear(body.feature_dim, output_dim))
        self.body = body
        self.to(DEVICE)

    def forward(self, x, action=None):
        phi = self.body(tensor(x), tensor(action))
        y = self.fc_head(phi)
        return y


class PolicyNet(nn.Module):
    def __init__(self, num_options, num_actions, body):
        super().__init__()
        self.fc_pi = layer_init(nn.Linear(body.feature_dim, num_actions*num_options))
        self.fc_term = layer_init(nn.Linear(body.feature_dim, num_options))
        self.body = body

    def forward(self, x):
        phi = self.body(tensor(x))
        action_probs = F.softmax(self.fc_pi(phi), dim=-1)
        termination_probs = F.sigmoid(self.fc_term(phi))
        log_probs = torch.log(action_probs)

        return action_probs, log_probs, termination_probs



if __name__ == "__main__":
    net = FCBody(12)
    print(net)
    print(net(torch.Tensor([list(range(12))])))
