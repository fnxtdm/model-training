import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
from resnet18_multilabel import ResNet18MultiLabel
from labelme_dataset import LabelMeDataset
from utils.config import device, model_path, json_dir, image_dir

# 数据增强和预处理
transform = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
])

# 超参数设置
batch_size = 8
num_epochs = 50
learning_rate = 0.001

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

torch.save(model.state_dict(), f'{model_path}')

# 评估模型
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f'Validation Loss: {avg_loss:.4f}')

# 在训练结束后进行评估
evaluate(model, dataloader)
