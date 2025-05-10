import torch
import torch.nn.functional as F


# Modified UAIO loss calculation with external loss functions passed as parameters
def calculate_loss(
        outputs: torch.Tensor,
        target: torch.Tensor,
        incremental_mask: torch.Tensor,
        non_incremental_loss_fn,
        uncertainty_loss_fn,
        fuzzy_loss_fn,
        use_incremental: bool = False,
        threshold: float = 0.5,
        alpha: float = 1.0,
        beta: float = 1.0,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    Calculate loss using external loss functions for flexibility and to avoid graph breaking.

    Args:
        outputs (torch.Tensor): Model logits (batch_size, num_classes).
        target (torch.Tensor): Ground truth labels (batch_size,).
        incremental_mask (torch.Tensor): Boolean mask indicating incremental samples (batch_size,).
        non_incremental_loss_fn: Loss function for non-incremental samples.
        uncertainty_loss_fn: Loss function for uncertainty optimization.
        fuzzy_loss_fn: Loss function for fuzzy optimization.
        use_incremental (bool): Whether to apply incremental logic. Default is False.
        threshold (float): Probability threshold for incremental loss optimization. Default is 0.5.
        alpha (float): Weight for the uncertainty loss. Default is 1.0.
        beta (float): Weight for the fuzzy loss. Default is 1.0.

    Returns:
        dict: A dictionary containing 'logits', 'predicted_labels', and 'total_loss'.
    """
    batch_size = outputs.size(0)
    predicted_probs = F.softmax(outputs, dim=-1)
    predicted_labels = torch.argmax(predicted_probs, dim=-1)

    if not use_incremental:
        # Compute standard cross-entropy loss for all samples
        loss = non_incremental_loss_fn(outputs, target)
        normalized_loss = loss / batch_size  # 按总样本数量归一化
        return {"logits": outputs, "predicted_labels": predicted_labels, "total_loss": normalized_loss}

    # Split samples into incremental and non-incremental groups
    non_incremental_mask = ~incremental_mask
    non_incremental_indices = torch.where(non_incremental_mask)[0]
    incremental_indices = torch.where(incremental_mask)[0]

    # Non-incremental loss
    non_incremental_loss = (
        non_incremental_loss_fn(
            outputs[non_incremental_indices],
            target[non_incremental_indices]
        ) if non_incremental_indices.numel() > 0 else outputs.new_tensor(0.0)
    )
    non_incremental_loss /= max(1, non_incremental_indices.numel())  # 按非增量样本数量归一化

    # Incremental loss components
    if incremental_indices.numel() > 0:
        incremental_outputs = predicted_probs.index_select(0, incremental_indices)  # 不进行 in-place 操作

        print(incremental_outputs)
        if torch.isnan(incremental_outputs).any() or torch.isinf(incremental_outputs).any():
            print("Invalid values (NaN or Inf) found in incremental_outputs")

        max_probs, _ = incremental_outputs.max(dim=-1)

        # Uncertainty loss: Penalize predictions above threshold
        uncertainty_loss = uncertainty_loss_fn(max_probs, threshold)
        uncertainty_loss /= max(1, incremental_indices.numel())  # 按增量样本数量归一化

        # Fuzzy loss: Penalize all predictions regardless of correctness
        fuzzy_loss = fuzzy_loss_fn(max_probs, threshold)
        fuzzy_loss /= max(1, incremental_indices.numel())  # 按增量样本数量归一化

        # Combine incremental losses
        incremental_loss = alpha * uncertainty_loss + beta * fuzzy_loss
    else:
        incremental_loss = outputs.new_tensor(0.0)

    # Total loss
    total_loss = non_incremental_loss + incremental_loss

    return {"logits": outputs, "predicted_labels": predicted_labels, "total_loss": total_loss}


def cross_entropy_loss_fn(outputs, targets):
    return F.cross_entropy(outputs, targets, reduction="sum")  # 计算均值而非总和


def uncertainty_loss_fn(max_probs, threshold):
    return ((max_probs - threshold).clamp(min=0) ** 2).sum()  # 改为求均值


def fuzzy_loss_fn(max_probs, threshold):
    return ((max_probs - threshold) ** 2).sum()  # 改为求均值


# 测试用例
def test_calculate_loss():
    # 模拟输出 logits 和 ground truth
    outputs = torch.tensor([
        [2.0, 0.5, 0.1],  # 样本 1: 非增量，正确分类（0）
        [0.2, 1.8, 0.1],  # 样本 2: 非增量，正确分类（1）
        [3.1, 0.1, 2.5],  # 样本 3: 增量，错误分类（2），但预测概率高
        [0.1, 0.1, 0.12],  # 样本 4: 增量，错误分类（2），预测概率低
        [1.2, 0.9, 0.3],  # 样本 5: 非增量，错误分类（1）
    ], dtype=torch.float32)

    target = torch.tensor([0, 1, 2, 2, 0], dtype=torch.long)  # Ground truth
    incremental_mask = torch.tensor([False, False, True, True, False])  # 样本 3、4 是增量样本

    # 非增量模式测试
    result_no_incremental = calculate_loss(
        outputs=outputs,
        target=target,
        incremental_mask=incremental_mask,
        non_incremental_loss_fn=cross_entropy_loss_fn,
        uncertainty_loss_fn=uncertainty_loss_fn,
        fuzzy_loss_fn=fuzzy_loss_fn,
        use_incremental=False
    )
    print("非增量模式:")
    print("预测标签:", result_no_incremental['predicted_labels'].tolist())
    print("总损失:", result_no_incremental['total_loss'].item())

    # 增量模式测试
    result_incremental = calculate_loss(
        outputs=outputs,
        target=target,
        incremental_mask=incremental_mask,
        non_incremental_loss_fn=cross_entropy_loss_fn,
        uncertainty_loss_fn=uncertainty_loss_fn,
        fuzzy_loss_fn=fuzzy_loss_fn,
        use_incremental=True,
        threshold=0.5,
        device=torch.device('cpu')
    )
    print("\n增量模式:")
    print("预测标签:")
    for i, label in enumerate(result_incremental['predicted_labels'].tolist()):
        if incremental_mask[i]:
            if label == -1:
                print(f"样本 {i + 1}: 正确识别增量样本 (标签 -1)")
            else:
                print(f"样本 {i + 1}: 错误分类为 {label}")
        else:
            # 对非增量样本，输出是否正确
            is_correct = (label == target[i]).item()
            correctness = "正确" if is_correct else "错误"
            print(f"样本 {i + 1}: 非增量样本，预测为 {label} ({correctness})")

    print("总损失:", result_incremental['total_loss'].item())


# 运行测试用例
test_calculate_loss()
