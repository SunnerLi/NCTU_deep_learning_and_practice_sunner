from random_process import OrnsteinUhlenbeckProcess
from torch.autograd import Variable
from torch.optim import Adam
from util import to_var
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

class Actor(nn.Module):
    def __init__(self, n_state, limit, trainable = True):
        super(Actor, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(n_state, 40),
            nn.ReLU(),
            nn.Linear(40, 30),
            nn.ReLU(),
            nn.Linear(30, 1),
            nn.Tanh()
        )
        if not trainable:
            for param in self.parameters():
                param.requires_grad = False
        self.limit = float(limit)

    def forward(self, x):
        return self.main(x) * self.limit

class Critic(nn.Module):
    def __init__(self, n_state, trainable = True):
        super(Critic, self).__init__()
        self.front = nn.Sequential(
            nn.Linear(n_state, 40),
            nn.ReLU()
        )
        self.end = nn.Sequential(
            nn.Linear(41, 30),
            nn.ReLU(),
            nn.Linear(30, 1),
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

class DDPG:
    def __init__(self, n_state, n_action, a_limit, model_folder, memory_size = 10000, 
                    batch_size = 32, tau = 0.01, gamma = 0.9, 
                    epsilon = 50000, is_training = True):
        # Record the parameters
        self.n_state = n_state
        self.n_action = n_action
        self.a_limit = a_limit
        self.memory_size = memory_size
        self.model_folder = model_folder
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.epsilon = epsilon
        self.is_training = is_training
        self.depsilon = 1.0 / self.epsilon

        # Create the network and related objects
        self.memory = np.zeros([self.memory_size, 2 * self.n_state + self.n_action + 1], dtype = np.float32)
        self.memory_counter = 0
        self.eval_actor = Actor(self.n_state, a_limit)
        self.eval_critic = Critic(self.n_state)
        self.target_actor = Actor(self.n_state, a_limit, trainable = False)
        self.target_critic = Critic(self.n_state, trainable = False)
        self.hardCopy(self.target_actor, self.eval_actor)
        self.hardCopy(self.target_critic, self.eval_critic)
        self.actor_optimizer = Adam(self.eval_actor.parameters(), lr = 0.001)
        self.critic_optimizer = Adam(self.eval_critic.parameters(), lr = 0.002)
        self.criterion = nn.MSELoss()
        self.random_process = OrnsteinUhlenbeckProcess(size = n_action,
            theta = 0.15, mu = 0.0, sigma = 0.2)

    def load(self):
        pass

    # def chooseAction(self, s, should_random = False, decay_epsilon = True):
    #     """
    #         給定輸入state，透過evaluate actor輸出[-1, 1]之間的實數動作值
    #     """
    #     # if not should_random:
    #     #     s = to_var(s)
    #     #     a = self.eval_actor(s)
    #     #     a = a.cpu().data.numpy()
    #     #     a += self.is_training * max(self.epsilon, 0) * self.random_process.sample()
    #     #     a = np.clip(a, -1.0, 1.0)
    #     #     if decay_epsilon:
    #     #         self.epsilon -= self.depsilon
    #     # else:
    #     #     a = np.random.uniform(-1.0, 1.0, size = self.n_action)
    #     # return a
    #     s = to_var(s)
    #     a = self.eval_actor(s)
    #     a = a.cpu().data.numpy()
    #     a = np.clip(np.random.normal(a, self.var), -2, 2)
    #     if self.memory_counter > self.memory_size:
    #         self.var *= 0.9995
    #     return a
    def chooseAction(self, s):
        s = to_var(s)
        a = self.eval_actor(s)
        a = a.cpu().data.numpy()
        return a

    def store_path(self, s, a, r, s_):
        """
            儲存state transition相關資訊
        """
        transition = np.hstack((s, a, [r], s_))
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = transition
        self.memory_counter += 1

    def softCopy(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                (1.0 - self.tau) * target_param.data + self.tau * source_param.data
            )

    def hardCopy(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def update(self):       
        # 將evaluate network的參數複製進入target network中
        self.softCopy(self.target_actor, self.eval_actor)
        self.softCopy(self.target_critic, self.eval_critic)

        # 決定輸入的batch data
        sample_idx = np.random.choice(self.memory_size, size = self.batch_size)           
        batch_data = self.memory[sample_idx, :]
        batch_s = batch_data[:, :self.n_state]
        batch_a = batch_data[:, self.n_state:self.n_state+self.n_action]
        batch_r = batch_data[:, -self.n_state-1:-self.n_state]
        batch_s_ = batch_data[:, -self.n_state:]

        # 送入Pytorch中
        batch_s = to_var(batch_s)
        batch_a = to_var(batch_a)
        batch_r = to_var(batch_r)
        batch_s_ = to_var(batch_s_)


        self.actor_optimizer.zero_grad()
        q = self.eval_critic(batch_s, self.eval_actor(batch_s))
        a_loss = q.mean()
        a_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        q = self.eval_critic(batch_s, self.eval_actor(batch_s))        
        q_target = batch_r + 0.9 * self.target_critic(batch_s_, self.target_actor(batch_s_))
        td_error = self.criterion(q, q_target)
        td_error.backward()
        self.critic_optimizer.step()


        