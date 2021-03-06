import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
    if x.shape == ():
        x = x.reshape((1,))
    x = torch.tensor(x, device=DEVICE, dtype=torch.float32)
    return x


class FCBody(nn.Module):
    def __init__(self, input_dim, hidden_units=[64, 64], gate=nn.ReLU()):
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
        # print(self.net[0].state_dict())
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
        self.fc_qu = layer_init(nn.Linear(body.feature_dim, num_actions*num_options))
        self.body = body
        self.num_options = num_options
        self.num_actions = num_actions

    def forward(self, x):
        phi = self.body(tensor(x))
        action_probs = F.softmax(self.fc_pi(phi).view(-1, self.num_options, self.num_actions), dim=-1)
        termination_probs = F.sigmoid(self.fc_term(phi)).view(-1, self.num_options)
        q_u = self.fc_qu(phi).view(-1, self.num_options, self.num_actions)
        q_omega = (q_u * action_probs).sum(dim=-1)
        log_probs = torch.log(action_probs)
        # print(action_probs.size(), log_probs.size(), termination_probs.size(), q_u.size(), q_omega.size())
        return action_probs, log_probs, termination_probs, q_u, q_omega
