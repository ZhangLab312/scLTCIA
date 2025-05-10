import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, alpha_attn=0.5, beta_attn=0.5, lambda_repr=1.0, attention_weights=None):
        """
        初始化蒸馏损失类
        :param alpha_attn: 权重，控制 MSE 对注意力蒸馏的影响
        :param beta_attn: 权重，控制 KL 散度对注意力蒸馏的影响
        :param lambda_repr: 权重，控制表示蒸馏损失的影响
        :param attention_weights: 字典，用于指定不同注意力图的损失加权，格式为 {"att1": weight1, "att2": weight2, ...}
        """
        super(KnowledgeDistillationLoss, self).__init__()
        self.alpha_attn = alpha_attn
        self.beta_attn = beta_attn
        self.lambda_repr = lambda_repr
        self.attention_weights = attention_weights if attention_weights is not None else {}

    def forward(self, teacher_outputs, student_outputs):
        """
        计算知识蒸馏的损失
        :param teacher_outputs: 教师模型的输出，包括 attentions 字典和表示
                                格式: {"attentions": Dict[str, Tensor], "repr": Tensor}
                                注意力: Dict[str, Tensor(batch, seq_len, seq_len)]
                                表示: Tensor(batch, num_genes)
        :param student_outputs: 学生模型的输出，格式与教师模型一致
        :return: 总的蒸馏损失
        """
        # 解析教师与学生的输出
        teacher_attentions = teacher_outputs["attentions"]  # 字典格式 {att1, att2, ...}
        teacher_repr = teacher_outputs["repr"]  # Tensor(batch, num_genes)

        student_attentions = student_outputs["attentions"]  # 字典格式 {att1, att2, ...}
        student_repr = student_outputs["repr"]  # Tensor(batch, num_genes)

        # 初始化损失
        total_loss = 0.0

        # 注意力权重蒸馏
        for key in teacher_attentions.keys():
            t_attn = teacher_attentions[key]  # (batch, seq_len, seq_len)
            s_attn = student_attentions[key]  # (batch, seq_len, seq_len)

            # 确保输入的注意力权重已经归一化
            t_attn = F.softmax(t_attn, dim=-1)
            s_attn = F.softmax(s_attn, dim=-1)

            # KL散度损失（注意保持 batch 维度的均值）
            kl_loss = F.kl_div(s_attn.log(), t_attn, reduction="batchmean")
            # MSE损失
            mse_loss = F.mse_loss(s_attn, t_attn)

            # 综合注意力损失
            attention_loss = self.alpha_attn * mse_loss + self.beta_attn * kl_loss
            weight = self.attention_weights.get(key, 1.0)  # 获取每个注意力图的权重，默认为 1.0
            total_loss += weight * attention_loss

        # 表示蒸馏
        # 计算余弦相似度损失（1 - 余弦相似度）
        cosine_loss = 1 - F.cosine_similarity(teacher_repr, student_repr, dim=-1).mean()
        # 如果需要，可以改为 MSE 表示损失
        # cosine_loss = F.mse_loss(teacher_repr, student_repr)

        total_loss += self.lambda_repr * cosine_loss

        return total_loss


# # 教师输出示例
# teacher_outputs = {
#     "attentions": {
#         "att1": torch.randn(256, 1024, 1024),  # 第一个注意力图
#         "att2": torch.randn(256, 512, 512),
#         "att3": torch.randn(256, 256, 256),
#         "att4": torch.randn(256, 128, 128)
#     },
#     "repr": torch.randn(256, 1024)  # 教师的线性层输出表示
# }
#
# # 学生输出示例
# student_outputs = {
#     "attentions": {
#         "att1": torch.randn(256, 1024, 1024),  # 学生的第一个注意力图
#         "att2": torch.randn(256, 512, 512),
#         "att3": torch.randn(256, 256, 256),
#         "att4": torch.randn(256, 128, 128)
#     },
#     "repr": torch.randn(256, 1024)  # 学生的线性层输出表示
# }
#
# # 初始化蒸馏损失类，并指定注意力图权重
# attention_weights = {
#     "att1": 0.6,  # 第一个注意力图的权重
#     "att2": 0.4,
#     "att3": 0.3,
#     "att4": 0.2
# }
# kd_loss_fn = KnowledgeDistillationLoss(alpha_attn=0.5, beta_attn=0.5, lambda_repr=1.0,
#                                        attention_weights=attention_weights)
#
# # 计算蒸馏损失
# loss = kd_loss_fn(teacher_outputs, student_outputs)
# print("Distillation Loss:", loss.item())
