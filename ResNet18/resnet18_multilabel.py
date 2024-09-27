import torch.nn as nn
import torchvision.models as models

# 定义ResNet18模型，用于多标签分类
class ResNet18MultiLabel(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(ResNet18MultiLabel, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        # 修改ResNet18的最后一层为多标签分类输出
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = self.sigmoid(x)
        return x