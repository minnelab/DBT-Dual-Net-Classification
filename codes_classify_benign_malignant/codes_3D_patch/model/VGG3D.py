import torch
import torch.nn as nn
import math


__all__ = [
    'VGG3D', 'vgg11_3d', 'vgg11_bn_3d', 'vgg13_3d', 'vgg13_bn_3d', 'vgg16_3d', 'vgg16_bn_3d',
    'vgg19_bn_3d', 'vgg19_3d',
]

def make_layers_3d(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool3d(kernel_size=2, stride=2)]
        else:
            conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv3d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG3D(nn.Module):
    def __init__(self, features, num_classes=2):
        super(VGG3D, self).__init__()
        self.features = features
        # Calculate the size of the flattened feature vector
        self.num_features = 512 * 7 * 7 * 7  # Adjust based on your input dimensions and architecture
        self.classifier = nn.Linear(self.num_features, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg11_3d(**kwargs):
    """VGG 11-layer 3D model (configuration "A")"""
    model = VGG3D(make_layers_3d(cfg['A']), **kwargs)
    return model

def vgg11_bn_3d(**kwargs):
    """VGG 11-layer 3D model (configuration "A") with batch normalization"""
    model = VGG3D(make_layers_3d(cfg['A'], batch_norm=True), **kwargs)
    return model

def vgg13_3d(**kwargs):
    """VGG 13-layer 3D model (configuration "B")"""
    model = VGG3D(make_layers_3d(cfg['B']), **kwargs)
    return model

def vgg13_bn_3d(**kwargs):
    """VGG 13-layer 3D model (configuration "B") with batch normalization"""
    model = VGG3D(make_layers_3d(cfg['B'], batch_norm=True), **kwargs)
    return model

def vgg16_3d(**kwargs):
    """VGG 16-layer 3D model (configuration "D")"""
    model = VGG3D(make_layers_3d(cfg['D']), **kwargs)
    return model

def vgg16_bn_3d(**kwargs):
    """VGG 16-layer 3D model (configuration "D") with batch normalization"""
    model = VGG3D(make_layers_3d(cfg['D'], batch_norm=True), **kwargs)
    return model

def vgg19_3d(**kwargs):
    """VGG 19-layer 3D model (configuration "E")"""
    model = VGG3D(make_layers_3d(cfg['E']), **kwargs)
    return model

def vgg19_bn_3d(**kwargs):
    """VGG 19-layer 3D model (configuration "E") with batch normalization"""
    model = VGG3D(make_layers_3d(cfg['E'], batch_norm=True), **kwargs)
    return model
