"""
    Lab6 - 用DQN來玩立桿子的遊戲
    這個程式是測試模型的程式，
    你必須先訓練模型
"""
from torch.autograd import Variable
from model import DQN
import numpy as np
import torch
import gym
import os

# 定義 hyper parameters
batch_size = 128
epsilon = 0.00          # 以多少機率隨機決定動作
episode = 2             # 總共玩幾次
model_path = './result/dqn2.pth'

def main():
    # Create carpole environment and network
    env = gym.make('CartPole-v0').unwrapped
    if not os.path.exists(model_path):
        raise Exception("You should train the DQN first!")
    net = DQN(
        n_state = env.observation_space.shape[0], 
        n_action = env.action_space.n,
        epsilon = epsilon,
        batch_size = batch_size,
        model_path = model_path
    )
    net.load()
    net.cuda()
    reward_list = []
    for i in range(episode):
        s = env.reset()
        total_reward = 0
        while True:
            # env.render()

            # Select action and obtain the reward
            a = net.chooseAction(s)
            s_, r, finish, _ = env.step(a)

            total_reward += r
            if finish:
                print("Episode: %d \t Total reward: %d \t Eps: %f" % (i, total_reward, net.epsilon))  
                reward_list.append(total_reward)
                break
            s = s_
    env.close()
    print("Testing average reward: ", np.mean(reward_list))

if __name__ == '__main__':
    main()