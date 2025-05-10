import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """
    Self-Attention mechanism for gene-gene relationships with a linear layer on q, k, v.
    """

    def __init__(self):
        super(AttentionBlock, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))  # Trainable scalar weight

    def forward(self, x):
        # Ensure all tensors are on the same device as the input `x`
        device = x.device

        # Get the sequence length (number of genes)
        seq_len = x.size(1)

        # Initialize the Linear layers dynamically based on x size, and move them to the device
        query_linear = nn.Linear(seq_len, seq_len).to(device)
        key_linear = nn.Linear(seq_len, seq_len).to(device)
        value_linear = nn.Linear(seq_len, seq_len).to(device)

        q = query_linear(x)  # Query
        k = key_linear(x)  # Key
        v = value_linear(x)  # Value

        # Compute attention matrix (genes, genes) for each batch
        attention = torch.bmm(q.unsqueeze(2), k.unsqueeze(1)) / (x.size(1) ** 0.5)  # (batch, genes, genes)
        attention = F.softmax(attention, dim=-1)

        # Weighted sum of values
        out = torch.bmm(attention, v.unsqueeze(2)).squeeze(2)  # (batch, genes)

        # Apply the residual connection
        out = self.gamma * out + x  # Residual connection

        return out, attention


class EncoderBlock(nn.Module):
    """
    Encoder block with Attention and Downsampling.
    """

    def __init__(self, pool_size=2):
        super(EncoderBlock, self).__init__()
        self.attention = AttentionBlock()
        self.pool = nn.AvgPool1d(pool_size, stride=pool_size)

    def forward(self, x):
        # x: (batch, genes)
        x, attention = self.attention(x)
        pooled = F.avg_pool1d(x.unsqueeze(1), kernel_size=2).squeeze(1)  # Downsampling
        return pooled, x, attention


class DecoderBlock(nn.Module):
    """
    Decoder block with Upsampling and Skip Connection.
    """

    def __init__(self):
        super(DecoderBlock, self).__init__()

    def forward(self, x, skip):
        # Upsample and concatenate skip connection
        upsampled = F.interpolate(x.unsqueeze(1), scale_factor=2, mode='linear', align_corners=True).squeeze(1)
        x = torch.cat([upsampled, skip], dim=1)  # Concatenate skip connection
        return x


class UNet1D(nn.Module):
    """
    U-Net structure for gene expression data with no explicit channels.
    """

    def __init__(self, num_genes):
        super(UNet1D, self).__init__()
        self.encoder1 = EncoderBlock()
        self.encoder2 = EncoderBlock()
        self.encoder3 = EncoderBlock()
        self.encoder4 = EncoderBlock()

        self.decoder1 = DecoderBlock()
        self.decoder2 = DecoderBlock()
        self.decoder3 = DecoderBlock()

        # Temporary layer to calculate correct number of features after encoding
        self.temp_restore = nn.Linear(0, num_genes)

        # Add a linear layer to restore the final gene dimension
        self.fc = nn.Linear(3584, num_genes)  # 3584 is calculated dynamically after flattening

    def forward(self, x):
        batch_size = x.size(0)

        # Encoder
        x1, skip1, attn1 = self.encoder1(x)
        x2, skip2, attn2 = self.encoder2(x1)
        x3, skip3, attn3 = self.encoder3(x2)
        x4, _, attn4 = self.encoder4(x3)

        # Decoder
        x = self.decoder1(x4, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder3(x, skip1)

        # Flatten the output to calculate the correct number of features
        x = x.view(batch_size, -1)  # Flatten the output for the linear layer

        # Apply the fully connected layer to get the correct gene dimension
        x = self.fc(x)  # (batch, num_genes)

        attentions = {"att1": attn1, "att2": attn2, "att3": attn3, "att4": attn4}
        return x, attentions
