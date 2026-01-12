import torch
import torch.nn as nn

__all__ = ['alexnet3d']

class AlexNet3D(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet3D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=11, stride=4, padding=5),  # Adjust padding to keep feature size reasonable
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),  # Adjust padding to keep feature size reasonable
            nn.Conv3d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            nn.Conv3d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        # Calculate the number of features in the final fully connected layer
        self.num_features = self._calculate_feature_size()

        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, num_classes),
            nn.Softmax(dim=1)
        )

    def _calculate_feature_size(self):
        # Create a dummy input to calculate the output size after feature extraction
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 10, 224, 224)  # 1x10x224x224 (batch size, depth, height, width)
            output = self.features(dummy_input)
            return output.numel() // dummy_input.size(0)  # Number of features after flattening

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def alexnet3d(**kwargs):
    r"""AlexNet model architecture for 3D inputs.
    """
    model = AlexNet3D(**kwargs)
    return model
