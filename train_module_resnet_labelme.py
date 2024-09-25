import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

# 定义LabelMe数据集类
class LabelMeDataset(Dataset):
    def __init__(self, json_dir, image_dir, transform=None):
        self.json_dir = json_dir
        self.image_dir = image_dir
        self.transform = transform
        self.all_labels = self._find_all_labels()  # 动态计算所有标签
        self.data = self._load_json_files()

    def _find_all_labels(self):
        all_labels = set()
        json_files = [f for f in os.listdir(self.json_dir) if f.endswith('.json')]

        for json_file in json_files:
            json_path = os.path.join(self.json_dir, json_file)
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            labels = [shape['label'] for shape in json_data['shapes']]
            all_labels.update(labels)

        return sorted(list(all_labels))  # 返回排序后的标签列表

    def _load_json_files(self):
        data = []
        json_files = [f for f in os.listdir(self.json_dir) if f.endswith('.json')]

        for json_file in json_files:
            json_path = os.path.join(self.json_dir, json_file)
            with open(json_path, 'r') as f:
                json_data = json.load(f)

            image_path = os.path.join(self.image_dir, json_data['imagePath'])
            labels = [shape['label'] for shape in json_data['shapes']]

            data.append({
                'image_path': image_path,
                'labels': labels
            })
        return data

    def __len__(self):
        return len(self.data)

    def _process_labels(self, labels):
        # 将标签转为多标签 one-hot 编码
        label_vector = torch.zeros(len(self.all_labels))

        for label in labels:
            if label in self.all_labels:
                label_index = self.all_labels.index(label)
                label_vector[label_index] = 1

        return label_vector

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image_path']
        labels = item['labels']

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # 处理标签为多标签格式
        label_vector = self._process_labels(labels)

        return image, label_vector

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
json_dir = 'labelme'  # JSON文件夹路径
image_dir = 'labelme'  # 图像文件夹路径
batch_size = 8
num_epochs = 500
learning_rate = 0.001

# 加载数据集和数据加载器
dataset = LabelMeDataset(json_dir=json_dir, image_dir=image_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 模型初始化
num_classes = len(dataset.all_labels)  # 根据数据集的标签数量动态调整输出层
model = ResNet18MultiLabel(num_classes=num_classes)
criterion = nn.BCEWithLogitsLoss()  # 二分类的多标签损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
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
