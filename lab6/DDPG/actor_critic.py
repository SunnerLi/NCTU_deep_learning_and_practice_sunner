import torch.nn as nn
import torch

class Actor(nn.Module):
    def __init__(self, n_state, n_action, a_limit, trainable = True):
        super(Actor, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(n_state, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, n_action),
            nn.Tanh()
        )
        self.limit = float(a_limit)
        if not trainable:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.limit * self.main(x)

class Critic(nn.Module):
    def __init__(self, n_state, n_action, trainable = True):
        super(Critic, self).__init__()
        self.front = nn.Sequential(
            nn.Linear(n_state, 400),
            nn.ReLU()
        )
        self.end = nn.Sequential(
            nn.Linear(400 + n_action, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
            nn.ReLU(),
            nn.Linear(1, 1)
        )
        if not trainable:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x, act):
        x = self.front(x)
        x = torch.cat([x, act], 1)
        return self.end(x)