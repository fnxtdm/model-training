# config.py
import torch

json_dir = '../labelme'  # labelme JSON文件夹路径
image_dir = '../datasets/thinos9/images'  # 要预测的PNG图像目录
tmp_dir = '../datasets/raw'  # 要预测的PNG图像目录

# Define input/output paths
output_img_dir = '../datasets/thinos9/images'  # Output image directory
output_label_dir = '../datasets/thinos9/labels'  # Output label directory

labels_file = 'all_labels.txt'  # 标签文件
model_path = 'model.pth'  # 训练好的模型权重路径

# CUDA可用性检查
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("CUDA is available! Training on GPU.")
else:
    print("CUDA is not available. Training on CPU.")
