
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from models.YOLOv5.yolov5_multilabel import YOLOv5
from config import device, model_path, json_dir, image_dir, labels_file
from labelme_dataset import LabelMeDataset
from compute_loss import YoloLoss

# Parameters
epochs = 1
batch_size = 16
learning_rate = 0.001

# 数据增强和预处理
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# Dataset and DataLoader
dataset = LabelMeDataset(json_dir=json_dir, image_dir=image_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, Loss, Optimizer
num_classes = len(dataset.all_labels)  # Adjust output layer based on dataset labels
model = YOLOv5(num_classes)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model.to(device)

# 定义损失函数
criterion = YoloLoss(num_classes=len(dataset.all_labels))

# 训练和验证循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images = images.cuda()
        targets = [target.cuda() for target in targets]

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, targets)

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss/len(train_loader)}")

    # 验证阶段
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.cuda()
            targets = [target.cuda() for target in targets]

            outputs = model(images)
            loss = YoloLoss(num_classes=len(dataset.all_labels))

            val_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss/len(val_loader)}")

torch.save(model.state_dict(), model_path)

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

# Evaluate the model after training
evaluate(model, val_loader)



