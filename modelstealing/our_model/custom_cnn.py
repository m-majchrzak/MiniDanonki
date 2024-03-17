from modelstealing.our_model.model import Model

import torch
import torch.nn as nn

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.features = nn.Sequential(
            # Wejście: 3 x 32 x 32
            nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Rozmiar: 48 x 16 x 16
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Rozmiar: 96 x 8 x 8
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Rozmiar: 32 x 4 x 4
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        # Po tej warstwie, tensor będzie miał rozmiar: 32 x 4 x 4, co daje 512 po spłaszczeniu.

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class CustomCNN(Model):

    def initialize_model(self) -> None:
        self.model = CustomNet()
        self.model.to(self.device)

    @property
    def name(self) -> str:
        return "CustomCNN"
