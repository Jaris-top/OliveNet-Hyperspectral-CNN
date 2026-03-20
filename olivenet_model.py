import torch
import torch.nn as nn

class OliveNet(nn.Module):
    # OliveNet: A Lightweight Hybrid 3D-2D CNN for Olivine Authenticity Identification
    def __init__(self, in_channels=12, num_classes=3):
        super(OliveNet, self).__init__()
        
        # Spatial Size Compression (Reduces spatial size to 256x256)
        self.spatial_compress = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolution Block 1: 3D Convolution for spectral-spatial features
        self.conv_block1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(in_channels, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        
        # Convolution Block 2: 2D Convolution for abstraction
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Convolution Block 3: 2D Convolution
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Global Average Pooling (GAP) Layer (Reduces parameters by 78%)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification Layer (Natural, Synthetic, Dyed)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.spatial_compress(x)
        x = x.unsqueeze(1)
        x = self.conv_block1(x)
        x = x.squeeze(2)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        
        return output
