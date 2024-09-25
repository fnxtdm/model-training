import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(3, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)
        self.decoder4 = self.upconv_block(1024, 512)
        self.decoder3 = self.upconv_block(512, 256)
        self.decoder2 = self.upconv_block(256, 128)
        self.decoder1 = self.upconv_block(128, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(nn.MaxPool2d(2)(enc1))
        enc3 = self.encoder3(nn.MaxPool2d(2)(enc2))
        enc4 = self.encoder4(nn.MaxPool2d(2)(enc3))
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(enc4))

        dec4 = self.decoder4(bottleneck)
        dec4 = self._crop_and_concat(enc4, dec4)

        dec3 = self.decoder3(dec4)
        dec3 = self._crop_and_concat(enc3, dec3)

        dec2 = self.decoder2(dec3)
        dec2 = self._crop_and_concat(enc2, dec2)

        dec1 = self.decoder1(dec2)
        dec1 = self._crop_and_concat(enc1, dec1)

        return self.final_conv(dec1)

    def _crop_and_concat(self, enc, dec):
        # Crop the encoder output to match the decoder output size
        enc_size = enc.size()[2:]  # (height, width)
        dec_size = dec.size()[2:]  # (height, width)

        # Calculate the cropping dimensions
        delta_height = enc_size[0] - dec_size[0]
        delta_width = enc_size[1] - dec_size[1]

        # Ensure the cropping dimensions are even
        delta_height = delta_height if delta_height % 2 == 0 else delta_height + 1
        delta_width = delta_width if delta_width % 2 == 0 else delta_width + 1

        # Crop the encoder output
        enc_cropped = enc[:, :, delta_height // 2: enc_size[0] - delta_height // 2,
                      delta_width // 2: enc_size[1] - delta_width // 2]

        return torch.cat((enc_cropped, dec), dim=1)


# 自定义 LabelMe 数据集
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


# Load dataset
json_dir = 'labelme'
image_dir = 'labelme'

transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Change to 512x512
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create dataset object
dataset = LabelMeDataset(json_dir=json_dir, image_dir=image_dir, transform=transform)

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Instantiate model
model = UNet(num_classes=len(dataset.all_labels))  # Dynamic number of labels
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)  # Shape: [batch_size, num_classes, height, width]

            # Ensure labels are in the same shape as outputs
            labels = labels.unsqueeze(2).unsqueeze(3)  # Reshape to [batch_size, num_classes, 1, 1]
            labels = labels.expand(-1, -1, outputs.shape[2], outputs.shape[3])  # Match height and width

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')

        # Validate model
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                # Ensure labels are in the same shape as outputs
                labels = labels.unsqueeze(2).unsqueeze(3)  # Reshape to [batch_size, num_classes, 1, 1]
                labels = labels.expand(-1, -1, outputs.shape[2], outputs.shape[3])  # Match height and width

                # Calculate validation loss
                val_loss += criterion(outputs, labels).item()

                # Calculate accuracy
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.numel()

        print(f'Validation Loss: {val_loss / len(val_loader)}, Accuracy: {correct / total * 100:.2f}%')

# Start training
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100)

# Save model
torch.save(model.state_dict(), 'unet_model.pth')

# Save all labels and class count to file
with open('all_labels.txt', 'w') as f:
    for label in dataset.all_labels:
        f.write(f"{label}\n")
