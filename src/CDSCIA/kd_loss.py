import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, alpha_attn=0.5, beta_attn=0.5, lambda_repr=1.0, attention_weights=None):
        """
        Initialize the knowledge distillation loss class.
        :param alpha_attn: Weight controlling the impact of MSE loss on attention distillation.
        :param beta_attn: Weight controlling the impact of KL divergence on attention distillation.
        :param lambda_repr: Weight controlling the impact of representation distillation loss.
        :param attention_weights: Dictionary specifying loss weights for different attention maps, 
                                  formatted as {"att1": weight1, "att2": weight2, ...}.
        """
        super(KnowledgeDistillationLoss, self).__init__()
        self.alpha_attn = alpha_attn
        self.beta_attn = beta_attn
        self.lambda_repr = lambda_repr
        self.attention_weights = attention_weights if attention_weights is not None else {}

    def forward(self, teacher_outputs, student_outputs):
        """
        Compute the knowledge distillation loss.
        :param teacher_outputs: Outputs from the teacher model, including attentions dictionary and representation.
                                Format: {"attentions": Dict[str, Tensor], "repr": Tensor}
                                Attentions: Dict[str, Tensor(batch, seq_len, seq_len)]
                                Representation: Tensor(batch, num_genes)
        :param student_outputs: Outputs from the student model, formatted identically to the teacher's outputs.
        :return: Total distillation loss.
        """
        # Parse teacher and student outputs
        teacher_attentions = teacher_outputs["attentions"]  # Dictionary format {att1, att2, ...}
        teacher_repr = teacher_outputs["repr"]  # Tensor(batch, num_genes)

        student_attentions = student_outputs["attentions"]  # Dictionary format {att1, att2, ...}
        student_repr = student_outputs["repr"]  # Tensor(batch, num_genes)

        # Initialize loss
        total_loss = 0.0

        # Attention weight distillation
        for key in teacher_attentions.keys():
            t_attn = teacher_attentions[key]  # (batch, seq_len, seq_len)
            s_attn = student_attentions[key]  # (batch, seq_len, seq_len)

            # Ensure input attention weights are normalized
            t_attn = F.softmax(t_attn, dim=-1)
            s_attn = F.softmax(s_attn, dim=-1)

            # KL divergence loss (maintain batch dimension mean)
            kl_loss = F.kl_div(s_attn.log(), t_attn, reduction="batchmean")
            # MSE loss
            mse_loss = F.mse_loss(s_attn, t_attn)

            # Combined attention loss
            attention_loss = self.alpha_attn * mse_loss + self.beta_attn * kl_loss
            weight = self.attention_weights.get(key, 1.0)  # Get weight for each attention map, default to 1.0
            total_loss += weight * attention_loss

        # Representation distillation
        # Compute cosine similarity loss (1 - cosine similarity)
        cosine_loss = 1 - F.cosine_similarity(teacher_repr, student_repr, dim=-1).mean()
        # Optionally, use MSE representation loss instead
        # cosine_loss = F.mse_loss(teacher_repr, student_repr)

        total_loss += self.lambda_repr * cosine_loss

        return total_loss


# # Example teacher outputs
# teacher_outputs = {
#     "attentions": {
#         "att1": torch.randn(256, 1024, 1024),  # First attention map
#         "att2": torch.randn(256, 512, 512),
#         "att3": torch.randn(256, 256, 256),
#         "att4": torch.randn(256, 128, 128)
#     },
#     "repr": torch.randn(256, 1024)  # Teacher's linear layer output representation
# }
#
# # Example student outputs
# student_outputs = {
#     "attentions": {
#         "att1": torch.randn(256, 1024, 1024),  # Student's first attention map
#         "att2": torch.randn(256, 512, 512),
#         "att3": torch.randn(256, 256, 256),
#         "att4": torch.randn(256, 128, 128)
#     },
#     "repr": torch.randn(256, 1024)  # Student's linear layer output representation
# }
#
# # Initialize distillation loss class with specified attention map weights
# attention_weights = {
#     "att1": 0.6,  # Weight for first attention map
#     "att2": 0.4,
#     "att3": 0.3,
#     "att4": 0.2
# }
# kd_loss_fn = KnowledgeDistillationLoss(alpha_attn=0.5, beta_attn=0.5, lambda_repr=1.0,
#                                        attention_weights=attention_weights)
#
# # Compute distillation loss
# loss = kd_loss_fn(teacher_outputs, student_outputs)
# print("Distillation Loss:", loss.item())
