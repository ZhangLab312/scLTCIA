import torch
import torch.nn as nn


# Dynamic Input Receiver (Matches dimensions between base model output and Unet1D model input)
class DynamicInputReceiver(nn.Module):
    """
    A module that dynamically adjusts to any input dimension and outputs a fixed 1024-dimensional tensor.
    """

    def __init__(self, output_dim=1024):
        """
        Initializes the receiver with a fixed output dimension.

        Args:
            output_dim (int): The fixed output dimension. Default is 1024.
        """
        super(DynamicInputReceiver, self).__init__()
        self.output_dim = output_dim
        self.linear = None  # Placeholder for the dynamically created linear layer

    def forward(self, x):
        """
        Forward pass for the dynamic receiver.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        input_dim = x.size(-1)  # Infer the input dimension
        if self.linear is None or self.linear.in_features != input_dim:
            # Create a new linear layer dynamically if input dimension changes
            self.linear = nn.Linear(input_dim, self.output_dim).to(x.device)

        return self.linear(x)


# Predictor (Implements cell annotation prediction and returns logits)
class PredictorLayer(nn.Module):
    """
    A predictor layer for mapping gene features to cell type predictions with enhanced functionality.
    """

    def __init__(self, num_genes, num_cell_types, hidden_dims=None, dropout=0.5, activation='relu'):
        """
        Initializes the PredictorLayer.

        Args:
            num_genes (int): Number of input features (genes).
            num_cell_types (int): Number of output classes (cell types).
            hidden_dims (list, optional): List of hidden layer dimensions. Default is None (single-layer linear mapping).
            dropout (float, optional): Dropout rate for regularization. Default is 0.5.
            activation (str, optional): Activation function to use ('relu', 'gelu', etc.). Default is 'relu'.
        """
        super(PredictorLayer, self).__init__()
        self.num_genes = num_genes
        self.num_cell_types = num_cell_types
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.activation = activation

        # Build the network layers
        layers = []
        input_dim = num_genes
        if hidden_dims:
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(self._get_activation_layer(activation))
                layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, num_cell_types))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass for the predictor layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_genes).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_cell_types).
        """
        return self.network(x)

    def _get_activation_layer(self, activation):
        """
        Returns the activation layer based on the activation type.

        Args:
            activation (str): The activation function name.

        Returns:
            nn.Module: The corresponding activation layer.
        """
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU()
        }
        return activations.get(activation, nn.ReLU())  # Default to ReLU


# # Create dynamic input receiver
# receiver = DynamicInputReceiver(output_dim=1024)
#
# # Input data with different dimensions
# input1 = torch.randn(32, 128)  # batch_size=32, input_dim=128
# output1 = receiver(input1)
# print(f"Output shape: {output1.shape}")  # Output: (32, 1024)
#
# input2 = torch.randn(64, 256)  # batch_size=64, input_dim=256
# output2 = receiver(input2)
# print(f"Output shape: {output2.shape}")  # Output: (64, 1024)
#
#
# num_genes = 1024
# num_cell_types = 10
# predictor = PredictorLayer(
#     num_genes=num_genes, num_cell_types=num_cell_types, hidden_dims=[128], dropout=0.5, activation='leaky_relu'
# )
# print(predictor)
#
# # Simulate input
# x = torch.randn(32, num_genes)  # batch_size=32
# output = predictor(x)
# print(f"Output shape: {output.shape}")  # Output: (32, 10)
