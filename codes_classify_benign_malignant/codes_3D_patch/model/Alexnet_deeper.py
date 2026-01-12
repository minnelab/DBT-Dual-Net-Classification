import torch
import torch.nn as nn

class AlexNet3DAdjusted(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet3DAdjusted, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3),  # Adjust padding
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),  # Pooling layer
            nn.Conv3d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        # Calculate the number of features in the final fully connected layer
        self.num_features = self._calculate_feature_size()

        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
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

def alexnet3d_deeper(**kwargs):
    r"""Adjusted AlexNet model architecture for 3D inputs.
    """
    model = AlexNet3DAdjusted(**kwargs)
    return model
