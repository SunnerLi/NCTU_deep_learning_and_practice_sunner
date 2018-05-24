from torch.autograd import Variable
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import gym
import time


#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = True
ENV_NAME = 'Pendulum-v0'

###############################  DDPG  ####################################
class Actor(nn.Module):
    def __init__(self, a_dim, s_dim, a_bound, trainable = True):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, 30),
            nn.ReLU(),
            nn.Linear(30, a_dim),
            nn.Tanh()
        )
        self.a_bound = float(a_bound)
        if not trainable:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.a_bound * self.net(x)

class Critic(nn.Module):
    def __init__(self, a_dim, s_dim, trainable = True):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(s_dim, 30)
        self.l2 = nn.Linear(a_dim, 30)
        self.l3 = nn.Linear(30, 1)
        if not trainable:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, s, a):
        net = F.relu(self.l1(s) + self.l2(a))
        return self.l3(net)

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound

        self.eval_actor = Actor(a_dim, s_dim, a_bound, trainable = True)
        self.targ_actor = Actor(a_dim, s_dim, a_bound, trainable = False)
        self.eval_critic = Critic(a_dim, s_dim, trainable = True)
        self.targ_critic = Critic(a_dim, s_dim, trainable = False)

        self.ctrain = Adam(self.eval_critic.parameters(), lr = LR_C)
        self.atrain = Adam(self.eval_actor.parameters(), lr = LR_A)

    def soft_replace(self):
        for ta, ea in zip(self.targ_actor.parameters(), self.eval_actor.parameters()):
            ta.data.copy_((1 - TAU) * ta.data + TAU * ea.data)
        for tc, ec in zip(self.targ_critic.parameters(), self.eval_critic.parameters()):
            tc.data.copy_((1 - TAU) * tc.data + TAU * ec.data)

    def choose_action(self, s):
        s = Variable(torch.from_numpy(s).float())
        a = self.eval_actor(s)
        return a.cpu().data.numpy()

    def learn(self):
        # soft target replacement
        self.soft_replace()

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt  = Variable(torch.from_numpy(self.memory[indices, :]))
        bs  = bt[:, :self.s_dim]
        ba  = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br  = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.atrain.zero_grad()
        a_loss = -self.eval_critic(bs, self.eval_actor(bs)).mean()
        a_loss.backward()
        self.atrain.step()

        self.ctrain.zero_grad()
        q_target = br + GAMMA * self.targ_critic(bs_, self.targ_actor(bs_))
        # q = self.eval_critic(bs, self.eval_actor(bs))
        q = self.eval_critic(bs, ba)
        td_loss = F.mse_loss(input = q, target = q_target)
        td_loss.backward()
        self.ctrain.step()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

###############################  training  ####################################

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ddpg = DDPG(a_dim, s_dim, a_bound)

var = 3  # control exploration
t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # Add exploration noise
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, a, r / 10, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            ddpg.learn()

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, 'Index: %i' % ddpg.pointer)
            # if ep_reward > -300:RENDER = True
            break
print('Running time: ', time.time() - t1)