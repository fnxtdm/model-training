import os
import shutil
import time

from PIL import Image
import torch
from torchvision import transforms, models
from utils.config import model_path, device

# 自定义数据集类
class CustomDataset:
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for app in os.listdir(root_dir):
            app_path = os.path.join(root_dir, app)
            for theme in os.listdir(app_path):
                theme_path = os.path.join(app_path, theme)
                for img_name in os.listdir(theme_path):
                    img_path = os.path.join(theme_path, img_name)
                    if f"{app}/{theme}" in self.labels:
                        continue
                    self.image_paths.append(img_path)
                    self.labels.append(f"{app}/{theme}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]

# 数据加载和转换
data_dir = 'dataset'
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224, standard for VGG16
    transforms.ToTensor(),
])

# 创建数据集
custom_dataset = CustomDataset(data_dir, transform)

# 计算类别数量
num_classes = len(set(custom_dataset.labels))

# 初始化模型
model = models.vgg16(pretrained=True)

# Freeze the feature layers if not training them
for param in model.features.parameters():
    param.requires_grad = False

# Modify the classifier to match the feature map size of 512x512 input images
input_shape = (3, 224, 224)  # (Channels, Height, Width) for standard VGG16
feature_size = 512 * 7 * 7  # 计算特征图大小

# Update the first fully connected layer to match the dynamically computed feature size
model.classifier[0] = torch.nn.Linear(feature_size, 4096)
# Update the last layer to match the number of classes in the dataset
model.classifier[6] = torch.nn.Linear(4096, num_classes)

# Move model to the selected device
model = model.to(device)

# 加载训练好的模型权重
model.load_state_dict(torch.load(model_path))
model.eval()

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

    return custom_dataset.labels[predicted.item()]

# 示例预测

tmp_dir = 'tmp'
dataset_dir = 'dataset_preprocessed'
png_files = [f for f in os.listdir(tmp_dir) if os.path.isfile(os.path.join(tmp_dir, f)) and f.endswith('.png')]

while True:
    png_files = [f for f in os.listdir(tmp_dir) if os.path.isfile(os.path.join(tmp_dir, f)) and f.endswith('.png')]

    if len(png_files) <= 1:
        time.sleep(10)
        continue

    image_path = os.path.join(tmp_dir, png_files[0])
    predicted_label = predict(image_path)
    print(f'Predicted label for {png_files[0]}: {predicted_label}')

    # move to dataset dir
    new_name = f'{dataset_dir}/{predicted_label}/{png_files[0]}'
    shutil.move(image_path, new_name)

print('Done.')