# ref: https://gist.github.com/annanay25/b6a94ab7e399f75411b2

import numpy as np
import nn as nn

class Linear(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Linear, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        # self.W = np.random.uniform(low = -1.0, high = 1.0, size=[in_channel, out_channel])
        # self.B = np.random.uniform(low = -1.0, high = 1.0, size=[out_channel])
        if out_channel == 3:
            self.W = np.asarray([[1., 1., 1.], [-1., -1., 1.], [1., 1., 1.]])
            self.B = np.asarray([1., 1., 1.])
        else:
            self.W = np.asarray([[1.], [1.], [1.]])
            self.B = np.asarray([[1.]])

    def forward(self, x):
        self.input_tensor = x
        print(self.W)
        return np.dot(self.input_tensor, self.W) + self.B

    def backward(self, x):
        self.W_grad = np.dot(self.input_tensor.T, x)
        print()
        print('grad: ', self.W_grad)
        self.B_grad = np.sum(self.W_grad, axis=0)
        return np.dot(x, self.W.T)

    def step(self, lr):
        self.W += lr * self.W_grad
        self.B += lr * self.B_grad

class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def work(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, x):
        return self.work(x)

    def backward(self, x):
        return self.work(x) * (1 - self.work(x))

    def step(self, lr):
        pass

class Network(nn.Module):
    def __init__(self):
        self.fc1 = Linear(3, 3)
        # self.act1 = Sigmoid()
        self.fc2 = Linear(3, 1)
        self.act2 = Sigmoid()

    def forward(self, x):
        # Form the graph
        self.graph = [self.fc1,  self.fc2, self.act2]
        for op in self.graph:
            x = op(x)
        return x

    def backward(self, x):
        for op in reversed(self.graph):
            x = op.backward(x)

    def step(self, lr = 0.001):
        for op in self.graph:
            op.step(lr)

net = Network()
x = np.asarray([[0.1, 0.1, 1], [0.1, 1, 1]])
y = np.asarray([[1], [0]])

for i in range(2):
    loss_sum = 0
    for j in range(len(x)):
        batch_x = x[j][np.newaxis, :]
        batch_y = y[j][np.newaxis, :]
        y_ = net.forward(batch_x)
        loss = np.sum((y_ - batch_y) ** 2)
        net.backward(batch_y - y_)
        net.step(lr = 0.1)
        print('<< abs: ', y_ - batch_y, '\t sqr: ', loss)
        loss_sum += loss
        print('ans: ', y_, batch_y)
    print('iter: ', i, '\tloss: ', loss_sum)