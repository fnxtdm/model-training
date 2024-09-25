import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision import transforms, models

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


# Function to calculate feature size of VGG16 dynamically
def get_feature_size(model, input_shape):
    with torch.no_grad():
        dummy_input = torch.rand(1, *input_shape)  # Create a dummy input with the given shape
        features = model.features(dummy_input)     # Pass through the feature extractor of VGG16
        return features.view(1, -1).size(1)        # Return the flattened feature size

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.priorities = []

        for app in os.listdir(root_dir):
            app_path = os.path.join(root_dir, app)
            for theme in os.listdir(app_path):
                theme_path = os.path.join(app_path, theme)
                for img_name in os.listdir(theme_path):
                    img_path = os.path.join(theme_path, img_name)
                    self.image_paths.append(img_path)
                    if f"{app}/{theme}" in self.labels:
                        continue
                    self.labels.append(f"{app}/{theme}")
                    self.priorities.append(1.0)  # 默认权重为1

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]

    def set_priority(self, img_path, weight):
        """设置特定图像的优先级"""
        if img_path in self.image_paths:
            index = self.image_paths.index(img_path)
            self.priorities[index] = weight
        else:
            print(f"Image path {img_path} not found in dataset.")


# 数据加载和转换
data_dir = 'dataset'  # 修改后的数据目录
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224, standard for VGG16
    transforms.ToTensor(),
])

# 创建数据集
custom_dataset = CustomDataset(data_dir, transform)

# 设置特定图像的优先级
# custom_dataset.set_priority('data/application1/theme1/image1.png', 10)

# 创建加权采样器
weights = custom_dataset.priorities
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
data_loader = DataLoader(custom_dataset, batch_size=32, sampler=sampler)

# 计算类别数量
num_classes = len(set(custom_dataset.labels))

# CUDA可用性检查
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("CUDA is available! Training on GPU.")
else:
    print("CUDA is not available. Training on CPU.")

# 初始化模型
# model = CustomVGG(num_classes).to(device)
# 使用VGG16模型
model = models.vgg16(pretrained=True)

# Freeze the feature layers if not training them
for param in model.features.parameters():
    param.requires_grad = False

input_shape = (3, 224, 224)  # (Channels, Height, Width) for standard VGG16
feature_size = get_feature_size(model, input_shape)

# Update the first fully connected layer to match the dynamically computed feature size
model.classifier[0] = nn.Linear(feature_size, 4096)
# Update the last layer to match the number of classes in the dataset
model.classifier[6] = nn.Linear(4096, num_classes)

# Move model to the selected device
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in data_loader:
        images = images.to(device)
        label_indices = [custom_dataset.labels.index(label) for label in labels]
        labels_tensor = torch.tensor(label_indices).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels_tensor)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(data_loader):.4f}')

# 评估模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in data_loader:
        images = images.to(device)
        label_indices = [custom_dataset.labels.index(label) for label in labels]
        labels_tensor = torch.tensor(label_indices).to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels_tensor.size(0)
        correct += (predicted == labels_tensor).sum().item()

        # 打印实际标签和预测标签
        for i in range(len(labels_tensor)):
            print(i)
            actual_label = labels_tensor[i].item()
            predicted_label = predicted[i].item()
            print(f'Actual: {custom_dataset.labels[actual_label]}, Predicted: {custom_dataset.labels[predicted_label]}')

print(f'Accuracy: {100 * correct / total:.2f}%')

# 保存模型
torch.save(model.state_dict(), 'vgg_model.pth')
