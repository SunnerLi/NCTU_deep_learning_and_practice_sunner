from torch.optim import Adam
from util import to_var
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import os

# 定義 DQN 架構
class Net(nn.Module):
    def __init__(self, n_state, n_action):
        super(Net, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(n_state, 32),
            nn.LeakyReLU(),
            nn.Linear(32, n_action)
        )

    def forward(self, x):
        return self.main(x)

class DQN:
    def __init__(self, n_state, n_action, memory_size = 500, lr = 0.0002, epsilon = 1, 
                    epsilon_decay = 0.99, update_iter = 100, batch_size = 32, gamma = 0.8, model_path = 'dqn.pth'):
        # Record the parameter
        self.n_state = n_state
        self.n_action = n_action
        self.memory_size = memory_size
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.update_iter = update_iter
        self.batch_size = batch_size
        self.model_path = model_path
        self.gamma = gamma

        # Initialize the other object
        self.eval_net, self.target_net = Net(n_state, n_action), Net(n_state, n_action)
        self.learn_step_counter = 0     # 紀錄次數，用來判別是否要更新target network
        self.memory_counter = 0         # 紀錄目前memory記了多少筆資訊
        self.had_fill_memory = False    # 紀錄是否已經充滿了記憶庫
        self.memory = np.zeros([memory_size, n_state * 2 + 2])
        self.optimizer = Adam(self.eval_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def load(self):
        if os.path.exists(self.model_path):
            self.eval_net.load_state_dict(torch.load(self.model_path))
            self.target_net.load_state_dict(self.eval_net.state_dict())

    def save(self):
        torch.save(self.eval_net.state_dict(), self.model_path)

    def episodeDecay(self):
        self.epsilon *= self.epsilon_decay

    def chooseAction(self, x):
        """
            透過network決定action，
            極少數用隨機來決定
        """
        x = to_var(x)
        if np.random.uniform() > self.epsilon:  # 用DQN決定動作
            action_value = self.eval_net(x)
            _, action = torch.max(action_value, 0)
            action = action[0].data.numpy()[0]
        else:                                   # 用隨機指定動作
            action = np.random.randint(0, self.n_action)
        return action

    def store_path(self, s, a, r, s_):
        """
            儲存玩過的資訊
        """
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index:] = transition
        self.memory_counter += 1
        if self.memory_counter >= self.memory_size and not self.had_fill_memory:
            self.had_fill_memory = True

    def update(self):
        # 如果儲存的transition資訊太少，則不更新
        if self.memory_counter < self.batch_size:
            return

        # 將evaluate network中的所有參數複製到target network中
        if self.learn_step_counter % self.update_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 從記憶庫中隨機挑選batch個transition來更新
        if self.had_fill_memory:
            sample_idx = np.random.choice(self.memory_size, self.batch_size)
        else:
            sample_idx = np.random.choice(self.memory_counter, self.batch_size)

        # 隨機從記憶庫中選取transition (Game re-play)            
        batch_memory = self.memory[sample_idx, :]
        batch_s = to_var(batch_memory[:, :self.n_state])
        batch_a = to_var(batch_memory[:, self.n_state:self.n_state+1].astype(int), to_float = False)
        batch_r = to_var(batch_memory[:, self.n_state+1:self.n_state+2])
        batch_s_ = to_var(batch_memory[:, -self.n_state:])

        # ------------------------------------------------
        #                       更新
        # 1. 用eval_net做預測後，取出t+1時間點的V值
        # 2. 用target_net做預測後，得到t+2時間點的V值
        # 3. 假設t+2時間點是正確答案，套入DQN更新公式
        # 4. 更新參數
        # ------------------------------------------------       
        q_eval = self.eval_net(batch_s).gather(1, batch_a)                          # 1 
        q_next = self.target_net(batch_s_).detach()                                 # 2
        q_target = batch_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1) # 3
        loss = self.criterion(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()                                                             # 4
        self.optimizer.step()