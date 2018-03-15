import numpy as np
import nn as nn

"""
    (Deep learning and practice Lab-0)

    The bonus part
"""

# Training parameters
epochs = 100000

class Network(nn.Module):
    """
        Define network structure (2 hidden layers)
    """
    def __init__(self):
        self.fc1 = nn.Linear(2, 3)
        self.act1 = nn.Sigmoid()
        self.fc2 = nn.Linear(3, 10)
        self.act2 = nn.Sigmoid()
        self.fc3 = nn.Linear(10, 1)
        self.act3 = nn.Sigmoid()

    def forward(self, x):
        x = x.T
        self.graph = [self.fc1, self.act1, self.fc2, self.act2, self.fc3, self.act3]
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
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 0])

    # Train
    for i in range(epochs):
        loss_sum = 0.
        for j in range(len(X)):
            batch_x = X[j][np.newaxis, :]
            batch_y = np.array([[Y[j]]])
            batch_y_ = net(batch_x)
            net.backward(batch_y_ - batch_y)
            net.step(1.0)
            loss_sum += np.sum((batch_y_ - batch_y) ** 2)
        if i % (epochs // 10) == 0:
            print('epochs: ', i)
    
    # Test
    for i in range(len(X)):
        batch_x = X[i][np.newaxis, :]
        batch_y_ = net(batch_x)
        print((np.reshape(batch_x, [-1]), np.reshape(batch_y_, [-1])))