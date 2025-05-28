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
        # Get the number of classes
        num_classes = logits.shape[1]

        # Create masks
        mask_old = target < num_classes  # Mask for old classes
        mask_new = target >= num_classes  # Mask for new classes

        # 1. Compute cross-entropy loss for old classes
        if mask_old.sum() > 0:
            loss_old = F.cross_entropy(logits[mask_old], target[mask_old])
        else:
            loss_old = torch.tensor(0.0, device=logits.device)

        # 2. Compute threshold-based constraint loss for new classes
        if mask_new.sum() > 0:
            logits_new = logits[mask_new]  # Logits for new classes
            probs_new = F.softmax(logits_new, dim=1)  # Probabilities for new classes
            max_probs_new = probs_new.max(dim=1).values  # Maximum probability for each sample

            # Apply constraint loss to keep maximum probabilities below the threshold
            loss_new = torch.mean(torch.relu(max_probs_new - self.threshold))
        else:
            loss_new = torch.tensor(0.0, device=logits.device)

        # Total loss
        total_loss = loss_old + self.lambda_fuzzy * loss_new

        # Predicted labels
        pred_labels = torch.argmax(logits, dim=1)

        return total_loss, pred_labels


# # Example usage
# if __name__ == "__main__":
#     # Input data
#     logits = torch.randn(32, 5)  # 32 samples, 5 old classes
#     target = torch.randint(0, 8, (32,))  # Sample labels, assume [5, 7] are new classes
#     is_incremental = True  # Incremental mode
#     threshold = 0.5  # Classification threshold
#     lambda_fuzzy = 1.0  # Fuzzy loss weight
#
#     # Instantiate the loss class
#     loss_fn = TeacherIncrementalCrossEntropyLoss(threshold=threshold, lambda_fuzzy=lambda_fuzzy)
#
#     # Compute loss
#     loss, pred_labels = loss_fn(logits, target, is_incremental)
#     print(f"Total loss: {loss.item()}, Predicted labels: {pred_labels}")
