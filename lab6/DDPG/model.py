from actor_critic import Actor, Critic
from torch.autograd import Variable
from torch.optim import Adam
from util import to_var
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import os

class DDPG:
    def __init__(self, n_state, n_action, a_limit, model_folder = None, memory_size = 10000, batch_size = 32, tau = 0.01, gamma = 0.99, var = 3.0):
        # Record the parameters
        self.n_state = n_state
        self.n_action = n_action
        self.a_limit = a_limit
        self.memory_size = memory_size
        self.model_folder = model_folder
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.var = var

        # Create the network and related objects
        self.memory = np.zeros([self.memory_size, 2 * self.n_state + self.n_action + 1], dtype = np.float32)
        self.memory_counter = 0
        self.eval_actor = Actor(self.n_state, self.n_action, self.a_limit)
        self.eval_critic = Critic(self.n_state, self.n_action)
        self.target_actor = Actor(self.n_state, self.n_action, self.a_limit, trainable = False)
        self.target_critic = Critic(self.n_state, self.n_action, trainable = False)
        
        self.actor_optimizer = Adam(self.eval_actor.parameters(), lr = 0.001)
        self.critic_optimizer = Adam(self.eval_critic.parameters(), lr = 0.002)
        self.criterion = nn.MSELoss()

        # Make sure the parameter of target network is the same as evaluate network
        self.hardCopy()

    def load(self):
        if os.path.exists(self.model_folder):
            self.eval_actor.load_state_dict(torch.load(os.path.join(self.model_folder, 'actor.pth')))
            self.eval_critic.load_state_dict(torch.load(os.path.join(self.model_folder, 'critic.pth')))
        self.hardCopy()

    def save(self):
        if not os.path.exists(self.model_folder):
            os.mkdir(self.model_folder)
        torch.save(self.eval_actor.state_dict(), os.path.join(self.model_folder, 'actor.pth'))
        torch.save(self.eval_critic.state_dict(), os.path.join(self.model_folder, 'critic.pth'))

    def chooseAction(self, s):
        """
            給定輸入state，透過evaluate actor輸出[-1, 1]之間的實數動作值
        """
        s = to_var(s)
        a = self.eval_actor(s)
        a = a.cpu().data.numpy()
        if self.var > 0:
            a = np.clip(np.random.normal(a, self.var), -2, 2)
        return a

    def store_path(self, s, a, r, s_):
        """
            儲存state transition相關資訊
        """
        transition = np.hstack((s, a, [r], s_))
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = transition
        self.memory_counter += 1

    def softCopy(self):
        for ta, ea in zip(self.target_actor.parameters(), self.eval_actor.parameters()):
            ta.data.copy_((1.0 - self.tau) * ta.data + self.tau * ea.data)
        for tc, ec in zip(self.target_critic.parameters(), self.eval_critic.parameters()):
            tc.data.copy_((1.0 - self.tau) * tc.data + self.tau * ec.data)

    def hardCopy(self):
        for ta, ea in zip(self.target_actor.parameters(), self.eval_actor.parameters()):
            ta.data.copy_(ea.data)
        for tc, ec in zip(self.target_critic.parameters(), self.eval_critic.parameters()):
            tc.data.copy_(ec.data)

    def update(self):
        # 如果儲存的資訊太少就不更新
        if self.memory_counter <= 5000:
            return

        # 將evaluate network的參數複製進入target network中
        self.softCopy()
        
        # 決定輸入的batch data
        if self.memory_counter > self.memory_size:
            sample_idx = np.random.choice(self.memory_size, size = self.batch_size)
        else:
            sample_idx = np.random.choice(self.memory_counter, size = self.batch_size)                

        # 從記憶庫中擷取要訓練的資料
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
        q_target = batch_r + self.gamma * next_q_target            

        # 更新critic
        self.critic_optimizer.zero_grad()
        q_batch = self.eval_critic(batch_s, batch_a)
        value_loss = F.mse_loss(input = q_batch, target = q_target)
        value_loss.backward()
        self.critic_optimizer.step()

        # 更新actor
        self.actor_optimizer.zero_grad()
        policy_loss = -self.eval_critic(batch_s, self.eval_actor(batch_s)).mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        # 降低action隨機搜索廣度
        self.var *= .9995                            