import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from models.ResNet18.labelme_dataset import LabelMeDataset

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

# 数据增强和预处理
transform = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
])

# 超参数设置
json_dir = '../../data/labelme'  # JSON文件夹路径
image_dir = '../../data/labelme'  # 图像文件夹路径
batch_size = 8
num_epochs = 500
learning_rate = 0.001

# CUDA可用性检查
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("CUDA is available! Training on GPU.")
else:
    print("CUDA is not available. Training on CPU.")

# 加载数据集和数据加载器
dataset = LabelMeDataset(json_dir=json_dir, image_dir=image_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 模型初始化
num_classes = len(dataset.all_labels)  # 根据数据集的标签数量动态调整输出层
model = ResNet18MultiLabel(num_classes=num_classes)
criterion = nn.BCEWithLogitsLoss()  # 二分类的多标签损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.to(device)

# 训练循环
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        # Move data to the appropriate device
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

torch.save(model.state_dict(), 'resnet_model.pth')

# 评估模型
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f'Validation Loss: {avg_loss:.4f}')

# 在训练结束后进行评估
evaluate(model, dataloader)
