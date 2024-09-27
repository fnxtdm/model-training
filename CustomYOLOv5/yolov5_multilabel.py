import torch
import torch.nn as nn

# Define the CSPBlock
class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CSPBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1)
        self.concat_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.leaky_relu(self.conv3(x2))
        x2 = self.leaky_relu(self.conv4(x2))
        x = torch.cat((x1, x2), dim=1)
        return self.concat_conv(x)

# Define the Detection Head
class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DetectionHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 2, num_classes + 5, kernel_size=1)  # 4 for bbox + 1 for objectness
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        return self.conv2(x)

# Define the PANet
class PANet(nn.Module):
    def __init__(self, num_classes):
        super(PANet, self).__init__()
        self.conv1 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1)
        self.det_head = DetectionHead(128, num_classes)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        return self.det_head(x)

# Define the YOLOv5-like model
class YOLOv5(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)
        )

        self.layer2 = CSPBlock(32, 64)
        self.layer3 = CSPBlock(64, 128)
        self.layer4 = CSPBlock(128, 256)
        self.layer5 = CSPBlock(256, 512)

        self.pan = PANet(num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return self.pan(x)
