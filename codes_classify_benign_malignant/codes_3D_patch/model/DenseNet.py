import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_convs):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_convs):
            self.layers.append(self._make_conv_layer(in_channels))
            in_channels += growth_rate

    def _make_conv_layer(self, in_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat((x, out), dim=1)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return F.avg_pool3d(x, kernel_size=2, stride=2)  # Use only one pooling layer here

class DenseNet3D(nn.Module):
    def __init__(self, growth_rate=32, num_blocks=(4, 4, 4), num_classes=2):
        super(DenseNet3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        self.dense_blocks = nn.ModuleList()
        in_channels = 64
        for i, num_convs in enumerate(num_blocks):
            block = DenseBlock(in_channels, growth_rate, num_convs)
            self.dense_blocks.append(block)
            in_channels += num_convs * growth_rate
            
            # Only add a transition layer if there is a subsequent block
            if i != len(num_blocks) - 1:  # No transition layer after the last block
                transition = TransitionLayer(in_channels, in_channels // 2)
                self.dense_blocks.append(transition)
                in_channels //= 2

        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)  # Initial convolution
        x = self.bn1(x)
        x = self.relu(x)

        for layer in self.dense_blocks:
            x = layer(x)

        # Ensure that x is large enough before global pooling
        x = F.avg_pool3d(x, kernel_size=x.size()[2:])  # Global Average Pooling
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        
        return x
