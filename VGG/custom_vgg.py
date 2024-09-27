# 自定义VGG模型
class CustomVGG(nn.Module):
    def __init__(self, num_classes):
        super(CustomVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 1st conv layer
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 2nd conv layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pooling layer
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 3rd conv layer
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 4th conv layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pooling layer
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 5th conv layer
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 6th conv layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pooling layer
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 7th conv layer
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 8th conv layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pooling layer
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 9th conv layer
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 10th conv layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling layer
        )
        # 计算全连接层输入的特征图大小
        self.fc_input_size = 512 * 16 * 16
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, 4096),  # Flatten and 1st FC layer
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),  # 2nd FC layer
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes)  # Output layer
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

