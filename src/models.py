import torch
import torch.nn as nn


class CompressionModel(nn.Module) {
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, dim=1)
        return self.conv(x)
}