import torch
from torchvision.models import vgg16

class VGGForMultiLabel(torch.nn.Module):
    def __init__(self, num_classes):
        super(VGGForMultiLabel, self).__init__()
        # 加载预训练的 VGG 模型
        self.vgg = vgg16(pretrained=True)
        # 修改最后的分类层，适应多标签分类任务
        self.vgg.classifier[6] = torch.nn.Linear(4096, num_classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.vgg(x)
        return self.sigmoid(x)