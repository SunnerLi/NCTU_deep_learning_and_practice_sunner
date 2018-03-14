import numpy as np
import nn as nn

class Network(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(2, 3)
        self.act1 = nn.Sigmoid()
        self.fc2 = nn.Linear(3, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        x = x.T
        self.graph = [self.fc1, self.act1, self.fc2, self.act2]
        for op in self.graph:
            x = op(x)
        return x

    def backward(self, v):
        for op in reversed(self.graph):
            v = op.backward(v)

    def step(self, lr = 0.001):
        for op in self.graph:
            op.step(lr)

if __name__ == '__main__':
    net = Network()
    x = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.asarray([[0], [1], [1], [0]])

    for i in range(200):
        loss_sum = 0.
        for j in range(len(x)):
            batch_x = x[j][np.newaxis, :]
            batch_y = y[j][np.newaxis, :]
            batch_y_ = net(batch_x)
            net.backward(batch_y_ - batch_y)
            net.step(1.0)
            loss_sum += np.sum((batch_y_ - batch_y) ** 2)
        print('iter: ', i, '\tloss: ', loss_sum)