from model2 import DDPG
import numpy as np
import torch
import gym

"""
    ref: https://github.com/ghliu/pytorch-ddpg/blob/master/ddpg.py
"""

# 定義hyper-parameters
episode = 500
max_iter = 200
model_folder = './model'
warmup = 20

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
        model_folder = model_folder
    )
    var = 3
    for i in range(episode):
        s = env.reset()
        total_reward = 0
        for j in range(max_iter):
            # env.render()
           
            a = net.chooseAction(s)
            a = np.clip(np.random.normal(a, var), -2, 2)
            s_, r, finish, info = env.step(a)

            net.store_path(s, a, r / 10, s_)

            if net.memory_counter > net.memory_size:
                var *= .9995
                net.update()

            s = s_
            total_reward += r
            if j == max_iter - 1:
                print("Episode: %d \tReward: %i \t Explore: %.2f \t Pointer: %d" % (i, total_reward, var, net.memory_counter))
                break
    env.close()