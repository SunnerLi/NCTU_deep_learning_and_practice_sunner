from torch.autograd import Variable
from torch.optim import Adam
from util import to_var
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

class Actor(nn.Module):
    def __init__(self, n_state):
        super(Actor, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(n_state, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class Critic(nn.Module):
    def __init__(self, n_state):
        super(Critic, self).__init__()
        self.front = nn.Sequential(
            nn.Linear(n_state, 400),
            nn.ReLU()
        )
        self.end = nn.Sequential(
            nn.Linear(401, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
            nn.ReLU(),
            nn.Linear(1, 1)
        )

    def forward(self, x, act):
        x = self.front(x)
        x = torch.cat([x, act], 1)
        return self.end(x)

class DDPG:
    def __init__(self, n_state, n_action, a_limit, model_folder, memory_size = 1000, batch_size = 32, tau = 0.01, epsilon = 0.1, gamma = 0.8):
        # Record the parameters
        self.n_state = n_state
        self.n_action = n_action
        self.a_limit = a_limit
        self.memory_size = memory_size
        self.model_folder = model_folder
        self.batch_size = batch_size
        self.tau = tau
        self.epsilon = epsilon
        self.gamma = gamma

        # Create the network and related objects
        self.memory = np.zeros([self.memory_size, 2 * self.n_state + self.n_action + 1], dtype = np.float32)
        self.memory_counter = 0
        self.eval_actor = Actor(self.n_state)
        self.eval_critic = Critic(self.n_state)
        self.target_actor = Actor(self.n_state)
        self.target_critic = Critic(self.n_state)
        self.hardCopy(self.target_actor, self.eval_actor)
        self.hardCopy(self.target_critic, self.eval_critic)
        self.actor_optimizer = Adam(self.eval_actor.parameters(), lr = 0.001)
        self.critic_optimizer = Adam(self.eval_critic.parameters(), lr = 0.002)
        self.criterion = nn.MSELoss()

    def load(self):
        pass

    def chooseAction(self, s):
        """
            給定輸入state，透過evaluate actor輸出[-1, 1]之間的實數動作值
        """
        if self.memory_counter > 100:
            s = to_var(s)
            a = self.eval_actor(s)
            return a.cpu().data.numpy()
        else:
            a = np.random.normal()
            return [a]

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
        # 如果儲存的資訊太少就不更新
        if self.memory_counter <= self.batch_size:
            return
        
        # 決定輸入的batch data
        if self.memory_counter > self.memory_size:
            sample_idx = np.random.choice(self.memory_size, size = self.batch_size)
        else:
            sample_idx = np.random.choice(self.memory_counter, size = self.batch_size)                
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

        # 用target network計算target Q值
        next_q_target = self.target_critic(batch_s_, self.target_actor(batch_s_))
        next_q_target = Variable(next_q_target.data, volatile = False)
        q_target = batch_r + self.gamma * next_q_target

        # 更新critic
        self.critic_optimizer.zero_grad()
        q_batch = self.eval_critic(batch_s, batch_a)
        value_loss = self.criterion(q_batch, q_target)
        value_loss.backward()
        self.critic_optimizer.step()

        # 更新actor
        self.actor_optimizer.zero_grad()
        policy_loss = -self.eval_critic(batch_s, self.eval_actor(batch_s)).mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        # 將evaluate network的參數複製進入target network中
        self.softCopy(self.target_actor, self.eval_actor)
        self.softCopy(self.target_critic, self.eval_critic)