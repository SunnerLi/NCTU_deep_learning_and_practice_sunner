import numpy as np

class Module(object):
    def __init__(self):
        pass

    def __call__(self, x):
        return self.forward(x)

    def forward(self, *args):
        pass