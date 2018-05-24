from model import DDPG
import numpy as np
import torch
import gym

"""
    ref: https://github.com/ghliu/pytorch-ddpg/blob/master/ddpg.py
"""

# 定義hyper-parameters
episode = 5
max_iter = 200
model_folder = './model'
var = 0.0                   # 動作搜索變異性

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
        var = var
    )
    net.load()

    # Train
    reward_list = []
    for i in range(episode):
        s = env.reset()
        total_reward = 0
        for j in range(max_iter):
            env.render()
            a = net.chooseAction(s)
            s_, r, finish, info = env.step(a)

            # 將資料存到記憶庫並更新參數
            net.store_path(s, a, r / 10, s_)
            net.update()

            # 更新total reward和s_t資訊
            s = s_
            total_reward += r
            if j == max_iter - 1:
                print("Episode: %d \tReward: %i \t Explore: %.2f " % (i, total_reward, net.var))
                reward_list.append(total_reward)
                break
    env.close()
    print("Average total reward: ", np.mean(reward_list))