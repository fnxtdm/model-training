import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self, num_classes):
        super(YoloLoss, self).__init__()
        self.num_classes = num_classes

        # 用于回归边界框的损失
        self.mse_loss = nn.MSELoss()  # 对于边界框回归
        # 用于置信度损失的 BCE
        self.bce_loss = nn.BCELoss()  # 对于置信度
        # 用于分类的交叉熵
        self.ce_loss = nn.CrossEntropyLoss()  # 对于类别分类

    def forward(self, predictions, targets):
        # predictions 包含:
        # - predictions[..., 0]：置信度 (confidence)
        # - predictions[..., 1:5]：边界框 (x_center, y_center, width, height)
        # - predictions[..., 5:]：类别分布

        # 置信度损失
        pred_conf = predictions[..., 0]
        target_conf = targets[..., 0]
        conf_loss = self.bce_loss(pred_conf, target_conf)

        # 边界框损失 (x_center, y_center, width, height)
        pred_box = predictions[..., 1:5]
        target_box = targets[..., 1:5]
        box_loss = self.mse_loss(pred_box, target_box)

        # 分类损失
        pred_class = predictions[..., 5:]  # 预测的类别
        target_class = targets[..., 5:]  # 实际的类别 (one-hot)
        class_loss = self.bce_loss(pred_class, target_class)

        # 将各部分损失相加
        total_loss = conf_loss + box_loss + class_loss

        return total_loss
