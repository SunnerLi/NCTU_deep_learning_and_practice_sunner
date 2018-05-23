"""
    Lab6 - 用DQN來玩立桿子的遊戲
    這個程式是訓練模型的程式
"""
from matplotlib import pyplot as plt
from torch.autograd import Variable
from model import DQN
import torch
import gym

# 定義 hyper parameters
batch_size = 128
lr = 0.0005
gamma = 0.8             # discount factor
epsilon_decay = 0.99    # 每次遊戲結束epsilon衰減的比例
epsilon = 1             # 以多少機率隨機決定動作
update_iter = 50        # 多少iteration更新一次target network的參數
memory_size = 5000      # 記憶庫的空間
episode = 1000          # 總共玩幾次
model_path = 'dqn3.pth'

if __name__ == '__main__':
    # Create carpole environment and network
    env = gym.make('CartPole-v0').unwrapped
    net = DQN(
        n_state = env.observation_space.shape[0], 
        n_action = env.action_space.n,
        memory_size = memory_size,
        lr = lr,
        epsilon = epsilon,
        epsilon_decay = epsilon_decay,
        update_iter = update_iter,
        batch_size = batch_size,
        gamma = gamma,
        model_path = model_path
    )
    net.cuda()
    net.load()
    reward_list = []
    for i in range(episode):
        s = env.reset()
        total_reward = 0
        while True:
            # env.render()
            # Select action and obtain the reward
            a = net.chooseAction(s)
            s_, r, finish, info = env.step(a)
            
            # Record the total reward
            total_reward += r

            # Revised the reward
            if finish:
                # 如果遊戲已結束，則將reward設為0以讓網路收斂
                r = 0
            else:
                # ----------------------------------------------------
                #   拆解reward，更精準的給予環境需要的訊息
                # 1.  r1得到的是對於距離的資訊，
                #     -abs term代表鼓勵agent不要去移動車子，
                #     一直維持在中間才能獲得很高的獎賞！
                # 2.  r2得到的是角度的資訊，
                #     儘量讓棒子跟垂直線的角度愈小愈好（就是讓棒子立正），
                #     角度愈小獎賞愈高
                # 最後扣0.5是讓獎勵區間分布於[-1~1]之間
                # ----------------------------------------------------
                x, x_dot, theta, theta_dot = s_
                r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
                r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
                r = r1 + r2

            # Sotre the transition and update the network
            net.store_path(s, a, r, s_)
            net.update()
            
            # Judge if finish an episode (and decay the epsilon)
            if finish:
                print("Episode: %d \t Total reward: %d \t Eps: %f" % (i, total_reward, net.epsilon))  
                reward_list.append(total_reward)
                if net.epsilon > 0.01:
                    net.episodeDecay()
                break

            # Update the current state as the future state of previous state
            s = s_
    net.save()
    plt.plot(range(len(reward_list)), reward_list, '-')
    plt.show()