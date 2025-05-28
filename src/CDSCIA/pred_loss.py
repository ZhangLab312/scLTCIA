import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class StudentIncrementalCrossEntropyLoss(nn.Module):
    def __init__(self, threshold=0.5, lambda_fuzzy=1.0, gamma=2.0, alpha=1.0, beta=1.0):
        """
        Incremental Cross-Entropy Loss with fuzzy guidance and entropy regularization.

        Args:
            threshold (float): Classification threshold for identifying incremental classes.
            lambda_fuzzy (float): Weight for fuzzy guidance loss.
            gamma (float): Weight for entropy regularization.
            alpha (float): Scaling factor for uncertainty loss.
            beta (float): Scaling factor for fuzzy loss.
        """
        super(StudentIncrementalCrossEntropyLoss, self).__init__()
        self.threshold = threshold
        self.lambda_fuzzy = lambda_fuzzy
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.inf_target = -1  # Value used for ignoring new class in loss computation.

    def forward(self, logits, target, incremental_mask, is_incremental):
        """
        Forward method to compute the loss.

        Args:
            logits (Tensor): Model predictions of shape (batch_size, num_classes).
            target (Tensor): Ground truth labels of shape (batch_size,).
            incremental_mask (Tensor): Boolean mask for incremental classes (shape: batch_size).
            is_incremental (bool): Whether the current task involves incremental learning.

        Returns:
            Tuple[Tensor, Tensor]: Total loss and predicted labels.
        """
        if not is_incremental:
            return self.cross_entropy_loss(logits, target)

        return self.incremental_loss(logits, target, incremental_mask)

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

    def incremental_loss(self, logits, target, incremental_mask):
        """
        Compute loss for incremental learning tasks.

        Args:
            logits (Tensor): Predictions of shape (batch_size, num_classes).
            target (Tensor): Ground truth labels of shape (batch_size,).
            incremental_mask (Tensor): Boolean mask for incremental classes (shape: batch_size).

        Returns:
            Tuple[Tensor, Tensor]: Total loss and predicted labels.
        """
        # Separate known and incremental samples
        known_mask = ~incremental_mask
        known_logits = logits[known_mask]
        known_target = target[known_mask]

        incremental_logits = logits[incremental_mask]
        incremental_target = target[incremental_mask]

        # Compute cross-entropy loss for known classes
        if known_logits.size(0) > 0:
            loss_known = F.cross_entropy(known_logits, known_target)
        else:
            loss_known = 0.0

        # Compute fuzzy guidance loss for incremental samples
        if incremental_logits.size(0) > 0:
            # fuzzy_loss = self.fuzzy_guide_loss(incremental_logits)
            fuzzy_loss = 0.0
        else:
            fuzzy_loss = 0.0

        # Combine losses
        total_loss = loss_known + self.lambda_fuzzy * fuzzy_loss

        # Predict labels for all samples
        pred_labels = torch.argmax(logits, dim=1)
        # pred_labels[incremental_mask] = self.inf_target  # Assign new class label to incremental samples

        return total_loss, pred_labels

    def fuzzy_guide_loss(self, logits):
        """
        Compute fuzzy guidance loss including dynamic distance penalty and entropy regularization.

        Args:
            logits (Tensor): Predictions of shape (num_incremental_samples, num_classes).

        Returns:
            Tensor: Average fuzzy guidance loss.
        """
        # Check for empty logits
        if logits.size(0) == 0:
            return torch.tensor(0.0, device=logits.device)

        # Compute probabilities
        pred_probs = torch.softmax(logits, dim=1)

        # Debug: Check if logits contain invalid values
        if torch.isnan(pred_probs).any():
            raise ValueError("Softmax produced NaN values. Check logits for invalid values.")

        # Compute max probabilities for each sample
        max_probs = pred_probs.max(dim=1).values

        # Uncertainty loss
        uncertainty_loss = ((max_probs - self.threshold).clamp(min=0) ** 2).mean()

        # Fuzzy loss
        fuzzy_loss = ((max_probs - self.threshold) ** 2).mean()

        # Combine losses
        total_fuzzy_loss = self.alpha * uncertainty_loss + self.beta * fuzzy_loss

        return total_fuzzy_loss
