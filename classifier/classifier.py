import torch

import torch.nn as nn
import torch.nn.functional as F


class WormClassifier(nn.Module):
    def __init__(self, dim=64):
        super(WormClassifier, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 12, 5, 1, padding=2)
        self.conv1_2 = nn.Conv2d(12, 12, 5, 1, padding=2)

        self.conv2_1 = nn.Conv2d(12, 24, 5, 1, padding=2)
        self.conv2_2 = nn.Conv2d(24, 24, 5, 1, padding=2)

        self.conv3_1 = nn.Conv2d(24, 36, 5, 1, padding=2)
        self.conv3_2 = nn.Conv2d(36, 36, 5, 1, padding=2)

        self.conv4_1 = nn.Conv2d(36, 48, 5, 1, padding=2)
        self.conv4_2 = nn.Conv2d(48, 48, 5, 1, padding=2)

        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x, features=False):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.max_pool2d(x, 2)

        xf = x
        x = torch.flatten(xf, 1)
        x_features = self.fc1(x)
        x_features = F.relu(x_features)
        x = torch.sigmoid(self.fc2(x_features))

        if features:
            return x, x_features

        return x
