import os
import torch
from PIL import Image
from torchvision import transforms
from models.YOLOv5.yolov5_multilabel import YOLOv5
from config import device, model_path, json_dir, tmp_dir, labels_file
from tools.load_labels import load_labels

# 从文件读取所有标签

# 加载模型并设置为评估模式
def load_model(model_path, num_classes):
    model = YOLOv5(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# Predict labels for a single image
def predict_image(model, image_path, transform, labels, threshold=0.5):
    try:
        image = Image.open(image_path).convert('RGB')  # Open image and convert to RGB
        image = transform(image).unsqueeze(0)  # Preprocess image and add batch dimension

        with torch.no_grad():  # Disable gradient calculation
            outputs = model(image)  # Forward pass

            # Print output shape for debugging
            print(f"Model output shape for {image_path}: {outputs.shape}")

            # Apply thresholding
            predicted = (outputs > threshold).int()  # This will give you a binary mask for each class

        # Convert predictions to label names
        predicted_labels = [labels[i] for i in range(len(labels)) if predicted[0][i] == 1]
        return predicted_labels
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return []


# 对目录中的所有PNG文件进行批量预测并直接打印
def predict_directory(model, image_dir, transform, labels):
    for image_file in os.listdir(image_dir):
        if image_file.endswith('.png'):
            image_path = os.path.join(image_dir, image_file)
            predicted_labels = predict_image(model, image_path, transform, labels)
            print(f"Image: {image_file}, Predicted Labels: {predicted_labels}")

# 主程序逻辑，不使用函数定义
if __name__ == "__main__":
    # 1. 加载所有标签
    labels = load_labels(labels_file)

    # 2. 加载预训练模型
    model = load_model(model_path, num_classes=len(labels))

    # 3. 定义图像转换操作，直接在代码中进行，而非函数调用
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # 调整图像大小
        transforms.ToTensor(),  # 转为张量
    ])

    # 4. 对图像目录进行预测并打印结果
    predict_directory(model, tmp_dir, transform, labels)
