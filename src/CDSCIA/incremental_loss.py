import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherIncrementalCrossEntropyLoss(nn.Module):
    def __init__(self, threshold=0.5, lambda_fuzzy=1.0):
        """
        Incremental Cross-Entropy Loss with threshold-based constraints.

        Args:
            threshold (float): Classification threshold for identifying incremental classes.
            lambda_fuzzy (float): Weight for fuzzy guidance loss.
        """
        super(TeacherIncrementalCrossEntropyLoss, self).__init__()
        self.threshold = threshold
        self.lambda_fuzzy = lambda_fuzzy

    def forward(self, logits, target, is_incremental):
        """
        Forward method to compute the loss.

        Args:
            logits (Tensor): Model predictions of shape (batch_size, num_classes).
            target (Tensor): Ground truth labels of shape (batch_size,).
            is_incremental (bool): Whether the current task involves incremental learning.

        Returns:
            Tuple[Tensor, Tensor]: Total loss and predicted labels.
        """
        if not is_incremental:
            return self.cross_entropy_loss(logits, target)

        return self.incremental_loss(logits, target)

    def cross_entropy_loss(self, logits, target):
        """
        Standard cross-entropy loss.

        Args:
            logits (Tensor): Predictions of shape (batch_size, num_classes).
            target (Tensor): Ground truth labels of shape (batch_size,).

        Returns:
            Tuple[Tensor, Tensor]: Loss and predicted labels.
        """
        loss = F.cross_entropy(logits, target)
        pred_labels = torch.argmax(logits, dim=1)
        return loss, pred_labels

    def incremental_loss(self, logits, target):
        """
        Compute loss for incremental learning tasks.

        Args:
            logits (Tensor): Predictions of shape (batch_size, num_classes).
            target (Tensor): Ground truth labels of shape (batch_size,).

        Returns:
            Tuple[Tensor, Tensor]: Total loss and predicted labels.
        """
        # 获取类别数
        num_classes = logits.shape[1]

        # 创建掩码
        mask_old = target < num_classes  # 老类别掩码
        mask_new = target >= num_classes  # 新类别掩码

        # 1. 对老类别计算交叉熵损失
        if mask_old.sum() > 0:
            loss_old = F.cross_entropy(logits[mask_old], target[mask_old])
        else:
            loss_old = torch.tensor(0.0, device=logits.device)

        # 2. 对新类别计算基于概率阈值的约束损失
        if mask_new.sum() > 0:
            logits_new = logits[mask_new]  # 新类别的 logits
            probs_new = F.softmax(logits_new, dim=1)  # 新类别的概率
            max_probs_new = probs_new.max(dim=1).values  # 每个样本的最大概率

            # 对最大概率添加约束损失（使其小于阈值）
            loss_new = torch.mean(torch.relu(max_probs_new - self.threshold))
        else:
            loss_new = torch.tensor(0.0, device=logits.device)

        # 总损失
        total_loss = loss_old + self.lambda_fuzzy * loss_new

        # 预测标签
        pred_labels = torch.argmax(logits, dim=1)

        return total_loss, pred_labels


# # 示例用法
# if __name__ == "__main__":
#     # 输入数据
#     logits = torch.randn(32, 5)  # 32个样本，5个老类别
#     target = torch.randint(0, 8, (32,))  # 样本标签，假设 [5, 7] 是新类别
#     is_incremental = True  # 增量模式
#     threshold = 0.5  # 分类阈值
#     lambda_fuzzy = 1.0  # 模糊损失权重
#
#     # 实例化损失类
#     loss_fn = TeacherIncrementalCrossEntropyLoss(threshold=threshold, lambda_fuzzy=lambda_fuzzy)
#
#     # 计算损失
#     loss, pred_labels = loss_fn(logits, target, is_incremental)
#     print(f"Total loss: {loss.item()}, Predicted labels: {pred_labels}")
