# labelme_dataset.py

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