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
        normalized_loss = loss / batch_size  # Normalize by total number of samples
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
    non_incremental_loss /= max(1, non_incremental_indices.numel())  # Normalize by number of non-incremental samples

    # Incremental loss components
    if incremental_indices.numel() > 0:
        incremental_outputs = predicted_probs.index_select(0, incremental_indices)  # Avoid in-place operations

        print(incremental_outputs)
        if torch.isnan(incremental_outputs).any() or torch.isinf(incremental_outputs).any():
            print("Invalid values (NaN or Inf) found in incremental_outputs")

        max_probs, _ = incremental_outputs.max(dim=-1)

        # Uncertainty loss: Penalize predictions above threshold
        uncertainty_loss = uncertainty_loss_fn(max_probs, threshold)
        uncertainty_loss /= max(1, incremental_indices.numel())  # Normalize by number of incremental samples

        # Fuzzy loss: Penalize all predictions regardless of correctness
        fuzzy_loss = fuzzy_loss_fn(max_probs, threshold)
        fuzzy_loss /= max(1, incremental_indices.numel())  # Normalize by number of incremental samples

        # Combine incremental losses
        incremental_loss = alpha * uncertainty_loss + beta * fuzzy_loss
    else:
        incremental_loss = outputs.new_tensor(0.0)

    # Total loss
    total_loss = non_incremental_loss + incremental_loss

    return {"logits": outputs, "predicted_labels": predicted_labels, "total_loss": total_loss}


def cross_entropy_loss_fn(outputs, targets):
    return F.cross_entropy(outputs, targets, reduction="sum")  # Calculate sum instead of mean


def uncertainty_loss_fn(max_probs, threshold):
    return ((max_probs - threshold).clamp(min=0) ** 2).sum()  # Change to calculate mean


def fuzzy_loss_fn(max_probs, threshold):
    return ((max_probs - threshold) ** 2).sum()  # Change to calculate mean


# Test cases
def test_calculate_loss():
    # Simulate output logits and ground truth
    outputs = torch.tensor([
        [2.0, 0.5, 0.1],  # Sample 1: Non-incremental, correctly classified (0)
        [0.2, 1.8, 0.1],  # Sample 2: Non-incremental, correctly classified (1)
        [3.1, 0.1, 2.5],  # Sample 3: Incremental, misclassified (2), but high prediction probability
        [0.1, 0.1, 0.12],  # Sample 4: Incremental, misclassified (2), low prediction probability
        [1.2, 0.9, 0.3],  # Sample 5: Non-incremental, misclassified (1)
    ], dtype=torch.float32)

    target = torch.tensor([0, 1, 2, 2, 0], dtype=torch.long)  # Ground truth
    incremental_mask = torch.tensor([False, False, True, True, False])  # Samples 3 and 4 are incremental samples

    # Test in non-incremental mode
    result_no_incremental = calculate_loss(
        outputs=outputs,
        target=target,
        incremental_mask=incremental_mask,
        non_incremental_loss_fn=cross_entropy_loss_fn,
        uncertainty_loss_fn=uncertainty_loss_fn,
        fuzzy_loss_fn=fuzzy_loss_fn,
        use_incremental=False
    )
    print("Non-incremental mode:")
    print("Predicted labels:", result_no_incremental['predicted_labels'].tolist())
    print("Total loss:", result_no_incremental['total_loss'].item())

    # Test in incremental mode
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
    print("\nIncremental mode:")
    print("Predicted labels:")
    for i, label in enumerate(result_incremental['predicted_labels'].tolist()):
        if incremental_mask[i]:
            if label == -1:
                print(f"Sample {i + 1}: Correctly identified as incremental sample (label -1)")
            else:
                print(f"Sample {i + 1}: Misclassified as {label}")
        else:
            # For non-incremental samples, output correctness
            is_correct = (label == target[i]).item()
            correctness = "correct" if is_correct else "incorrect"
            print(f"Sample {i + 1}: Non-incremental sample, predicted as {label} ({correctness})")

    print("Total loss:", result_incremental['total_loss'].item())


# Run test cases
test_calculate_loss()
