"""
networks and helpers
"""
import torch
import torch.nn as nn
from torch.distributions import Normal


def soft_update(target, source, tau=0.01):
    """
    performs soft update of network parameters with coefficient tau
    tau = 1 => copy parameters
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def init_wb(net):
    """
    initialize weights and biases
    """
    if type(net) == nn.Linear:
        nn.init.xavier_uniform_(net.weight)
        torch.nn.init.constant_(net.bias, 0)


class QNet(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNet, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        self.apply(init_wb)

    def forward(self, state, action):
        x = torch.relu(self.l1(torch.cat((state, action), dim=1)))
        x = torch.relu(self.l2(x))
        return self.l3(x)


class ValueNet(nn.Module):

    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        self.apply(init_wb)

    def forward(self, state):
        x = torch.relu(self.l1(state))
        x = torch.relu(self.l2(x))
        return self.l3(x)


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNet, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3m = nn.Linear(hidden_dim, action_dim)
        self.l3lsd = nn.Linear(hidden_dim, action_dim)
        self.apply(init_wb)

    def forward(self, state):
        x = torch.relu(self.l1(state))
        x = torch.relu(self.l2(x))
        return self.l3m(x), torch.clamp(self.l3lsd(x), min=-20, max=2)

    def get_action_logprob(self, state):
        mean, log_std = self.forward(state)
        # reparameterization trick, resulting in a lower variance estimator
        normal = Normal(mean, log_std.exp())
        sample = normal.rsample()
        action = torch.tanh(sample)
        # Enforcing Action Bounds
        log_prob = normal.log_prob(sample) - torch.log((1 - action.pow(2)) + 1e-7)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

    def get_action(self, state):
        mean, log_std = self.forward(state)
        normal = Normal(mean,  log_std.exp())
        return torch.tanh(normal.rsample())


