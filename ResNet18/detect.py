import os
import torch
from PIL import Image
from torchvision import transforms
from resnet18_multilabel import ResNet18MultiLabel
from utils.config import device, model_path, tmp_dir, labels_file
from tools.load_labels import load_labels

# 加载模型并设置为评估模式
def load_model(model_path, num_classes):
    model = ResNet18MultiLabel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# 对单张图片进行预测
def predict_image(model, image_path, transform, labels):
    image = Image.open(image_path).convert('RGB')  # 打开图像并转换为RGB
    image = transform(image).unsqueeze(0)  # 预处理图像并添加batch维度

    with torch.no_grad():  # 关闭梯度计算
        outputs = model(image)  # 模型前向传播
        predicted = (outputs > 0.5).int()  # 阈值设为0.5，二值化

    # 转换预测结果为标签名
    predicted_labels = [labels[i] for i in range(len(labels)) if predicted[0][i] == 1]
    return predicted_labels

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
        transforms.Resize((244, 244)),  # 调整图像大小
        transforms.ToTensor(),  # 转为张量
    ])

    # 4. 对图像目录进行预测并打印结果
    predict_directory(model, tmp_dir, transform, labels)
