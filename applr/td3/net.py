import numpy as np
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim, num_layers=2, hidden_layer_size=512):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = hidden_layer_size

        layers = []
        for i in range(num_layers):
            input_dim = hidden_layer_size if i > 0 else self.input_dim
            layers.append(nn.Linear(input_dim, hidden_layer_size))
            layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class CNN(nn.Module):

    def __init__(self, in_channels=1):
        super().__init__()
        self.feature_dim = 512
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1568, 512),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.fc(x)
