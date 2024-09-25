import os

import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import vgg16  # 直接导入 vgg16


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


def predict(model, image_path, all_labels, transform, device, threshold=0.5):
    """
    使用训练好的模型对单张图像进行预测。

    参数:
        model: 训练好的模型
        image_path: 待预测图像的路径
        all_labels: 标签列表
        transform: 数据预处理步骤
        device: 使用的设备（CPU 或 GPU）
        threshold: 用于判断标签是否为正的阈值

    返回:
        predicted_labels: 预测的标签列表
    """
    # 加载图像并进行预处理
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 增加 batch 维度
    image = image.to(device)

    # 模型设置为评估模式
    model.eval()

    with torch.no_grad():
        # 进行前向传播
        output = model(image)
        output = output.cpu().numpy()  # 将输出转换为 NumPy 格式

        # 应用阈值，将输出转换为标签
        predicted_labels_indices = (output > threshold).astype(int)[0]
        predicted_labels = [all_labels[i] for i, val in enumerate(predicted_labels_indices) if val == 1]

    return predicted_labels


if __name__ == "__main__":
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # 输入图像的大小
        transforms.ToTensor(),
    ])

    # 加载标签
    with open('all_labels.txt', 'r') as f:
        ALL_LABELS = [line.strip() for line in f.readlines()]

    NUM_CLASSES = len(ALL_LABELS)  # 获取标签数量

    # 加载模型
    model = VGGForMultiLabel(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load('vgg_model.pth'))  # 加载训练好的模型权重
    model.to(device)

    # 预测标签
    labelme_dir = 'tmp'
    png_files = [f for f in os.listdir(labelme_dir) if os.path.isfile(os.path.join(labelme_dir, f)) and f.endswith('.png')]

    for png_file in png_files:
        image_path = os.path.join(labelme_dir, png_file)

        predicted_labels = predict(model, image_path, ALL_LABELS, transform, device, threshold=0.5)
        print(f"Predicted labels for {png_file}: {predicted_labels}")


