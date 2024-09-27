import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

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
            labels = [(shape['label'], shape['points']) for shape in json_data['shapes']]

            data.append({
                'image_path': image_path,
                'labels': labels  # labels 和 边界框信息
            })
        return data

    def __len__(self):
        return len(self.data)

    def _process_labels(self, labels, img_width, img_height):
        targets = []

        for label, points in labels:
            if label in self.all_labels:
                class_id = self.all_labels.index(label)
                x_min = min(points[0][0], points[1][0])
                y_min = min(points[0][1], points[1][1])
                x_max = max(points[0][0], points[1][0])
                y_max = max(points[0][1], points[1][1])

                # 转换为 YOLO 格式的 (class, x_center, y_center, width, height)
                x_center = (x_min + x_max) / 2 / img_width
                y_center = (y_min + y_max) / 2 / img_height
                box_width = (x_max - x_min) / img_width
                box_height = (y_max - y_min) / img_height

                targets.append([class_id, x_center, y_center, box_width, box_height])

        return torch.tensor(targets)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image_path']
        labels = item['labels']

        image = Image.open(image_path).convert('RGB')
        img_width, img_height = image.size

        if self.transform:
            image = self.transform(image)

        # 将标签处理为 YOLO 格式 (class, x_center, y_center, width, height)
        targets = self._process_labels(labels, img_width, img_height)

        return image, targets
