import torch
import torch.nn as nn


class Snake(nn.Module):
    def __init__(self,
                 a: float = 1):
        super(Snake, self).__init__()
        self.a = a

    def forward(self, x):
        return (x + (torch.sin(self.a * x) ** 2) / self.a)
