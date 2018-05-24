from matplotlib import pyplot as plt
from model import DDPG
import numpy as np
import torch
import gym

"""
    ref: https://github.com/ghliu/pytorch-ddpg/blob/master/ddpg.py
"""

# 定義hyper-parameters
episode = 1000
max_iter = 200
model_folder = './model'
memory_size = 5000          # 記憶庫大小
batch_size = 32
tau = 0.01                  # 用多少比例的evaluate network參數來更新target network
gamma = 0.99                # 用多少比例的 critic value來當作target q value
var = 3.0                   # 動作搜索變異性

if __name__ == '__main__':
    # Create environment
    env = gym.make('Pendulum-v0').unwrapped
    n_state = env.observation_space.shape[0]        # 提取state的維度
    n_action = env.action_space.shape[0]            # 提取action的維度
    a_limit = env.action_space.high[0]              # 提取action連續動作中，最大的可能數值

    # Create network
    net = DDPG(
        n_state = n_state,
        n_action = n_action,
        a_limit = a_limit,
        model_folder = model_folder,
        memory_size = memory_size,
        batch_size = batch_size,
        tau = tau,
        gamma = gamma,
        var = var
    )
    net.load()

    # Train
    reward_list = []
    for i in range(episode):
        s = env.reset()
        total_reward = 0
        for j in range(max_iter):
            # env.render()
            a = net.chooseAction(s)
            s_, r, finish, info = env.step(a)

            # 將資料存到記憶庫並更新參數
            net.store_path(s, a, r / 10, s_)
            net.update()

            # 更新total reward和s_t資訊
            s = s_
            total_reward += r
            if j == max_iter - 1:
                print("Episode: %d \tReward: %i \t Explore: %.2f \t Pointer: %d" % (i, total_reward, net.var, net.memory_counter))
                reward_list.append(total_reward)
                break
    net.save()
    env.close()
    plt.plot(range(len(reward_list)), reward_list, '-o')
    plt.title('The reward curve of DDPG')
    plt.show()