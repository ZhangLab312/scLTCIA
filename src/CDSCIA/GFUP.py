import torch
import torch.nn as nn
from src.CDSCIA.utils import *
from src.CDSCIA.UNet1D import *


class GeneFlowUNetPredictor(nn.Module):
    """
    An integrated model for gene expression analysis.
    Combines a dynamic input receiver, UNet1D, and a predictor layer.
    """

    def __init__(self, num_genes, num_cell_types, hidden_dims=None, dropout=0.5, activation='relu'):
        """
        Initializes the GeneFlowUNetPredictor.

        Args:
            num_genes (int): Number of genes (intermediate representation dimension).
            num_cell_types (int): Number of cell type classes for prediction.
            hidden_dims (list, optional): Hidden layer dimensions for the predictor layer. Default is None.
            dropout (float, optional): Dropout rate for the predictor layer. Default is 0.5.
            activation (str, optional): Activation function for the predictor layer. Default is 'relu'.
        """
        super(GeneFlowUNetPredictor, self).__init__()
        self.input_receiver = DynamicInputReceiver(output_dim=1024)  # Adjusts input to 1024-dimensional space
        self.unet = UNet1D(num_genes=num_genes)  # UNet1D for processing gene expression data
        self.predictor = PredictorLayer(
            num_genes=num_genes,
            num_cell_types=num_cell_types,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation
        )  # Predicts cell types

    def forward(self, x):
        """
        Forward pass of the GeneFlowUNetPredictor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            dict: A dictionary containing:
                - 'logits': Prediction logits of shape (batch_size, num_cell_types).
                - 'gene_features': Processed gene features of shape (batch_size, num_genes).
                - 'attentions': Attention maps from the UNet1D encoder.
        """
        # Step 1: Adjust input dimensions
        adjusted_input = self.input_receiver(x)  # (batch_size, 1024)

        # Step 2: Process data through the UNet1D
        gene_features, attentions = self.unet(adjusted_input)  # gene_features: (batch_size, num_genes)

        # Step 3: Predict cell types
        logits = self.predictor(gene_features)  # (batch_size, num_cell_types)

        # Return a comprehensive output
        return  logits, gene_features, attentions

#
# # Simulate input data
# batch_size = 16
# input_dim = 2000  # Initial input data dimension
# num_genes = 1024  # Gene feature dimension output by UNet1D
# num_cell_types = 10  # Number of cell type categories
#
# # Initialize the model
# model = GeneFlowUNetPredictor(
#     num_genes=num_genes,
#     num_cell_types=num_cell_types,
#     hidden_dims=[512, 256],  # Hidden layer dimensions of the predictor layer
#     dropout=0.3,
#     activation='leaky_relu'
# )
#
# # Print the model structure
# print(model)
#
# # Create random input data (batch_size, input_dim)
# input_data = torch.randn(batch_size, input_dim)
#
# # Test the forward pass
# logits, gene_features, attentions = model(input_data)
#
# # Verify the output results
# print("\n==== Test Results ====")
# print(f"Input data shape: {input_data.shape}")
# print(f"Classification logits shape: {logits.shape}")  # (batch_size, num_cell_types)
# print(f"Gene features shape: {gene_features.shape}")  # (batch_size, num_genes)
# for attn_key, attn_map in attentions.items():
#     print(f"{attn_key} shape: {attn_map.shape}")  # Attention maps from UNet1D encoder
