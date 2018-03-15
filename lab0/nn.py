import numpy as np

"""
    (Deep learning and practice Lab-0)

    The script to stimulate the function of nn in pytorch (without pytorch library)
"""

class Module(object):
    def __init__(self):
        pass

    def __call__(self, x):
        return self.forward(x)

    def forward(self, *args):
        pass

    def step(self, *args):
        pass

class Linear(Module):
    def __init__(self, in_channel, out_channel):
        self.out_channel = out_channel
        self.W = np.random.uniform(low = -1.0, high = 1.0, size = [in_channel, out_channel])
        self.B = np.random.uniform(low = -1.0, high = 1.0, size = [out_channel])

    def forward(self, x):
        self.input_tensor = x
        return self.W.T.dot(x)

    def backward(self, x):
        self.W_grad = self.input_tensor.dot(x.T)
        self.B_grad = x.T
        return x.T.dot(self.W.T)

    def step(self, lr):
        self.W -= (lr * self.W_grad)
        self.B_grad = np.reshape(self.B_grad, [self.out_channel])
        self.B -= (lr * self.B_grad)

class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        self.input_tensor = x
        return 1.0 / (1.0 + np.exp(-x))

    def backward(self, x):
        theta_tensor = self.forward(self.input_tensor)
        return x.T * theta_tensor * (1 - theta_tensor)