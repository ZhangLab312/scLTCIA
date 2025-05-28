import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# from ipykernel.pickleutil import cell_type
from sklearn.metrics import accuracy_score, f1_score, silhouette_score, homogeneity_score, normalized_mutual_info_score, \
    adjusted_rand_score
from sklearn.manifold import TSNE
from tqdm import tqdm
import scanpy as sc
from sklearn.model_selection import train_test_split
from CDSCIA.src.CDSCIA.GFUP import *
from CDSCIA.src.CDSCIA.compute_loss import *
from CDSCIA.src.CDSCIA.kd_loss import *
from CDSCIA.src.CDSCIA.utils import *
from CDSCIA.src.test_mlp.model_mlp import *


def load_and_preprocess_data(
        file_path,
        cell_type_column,
        test_size=0.2,
        random_state=20020130,
        batch_size=512,
        device=None,
        end=6
):
    """
    Load and preprocess h5ad data file, and generate DataLoader.

    Args:
        file_path (str): Path to the h5ad file.
        cell_type_column (str): Column name where cell types are located.
        test_size (float): Proportion of the test set.
        random_state (int): Random seed.
        batch_size (int): Batch size for DataLoader.
        device (torch.device, optional): Device to load data to, defaults to auto-detect.
        end (int): Cursor for initial stage categories.

    Returns:
        dict: Data dictionary containing the following key-value pairs:
            - 'train_loader': DataLoader for training data.
            - 'test_loader': DataLoader for test data.
            - 'encoder': OneHotEncoder instance used during training.
            - 'device': Device used.
            - 'train_incremental_mask': Boolean mask for incremental classes in the training set.
            - 'test_incremental_mask': Boolean mask for incremental classes in the test set.
    """
    # Check device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read h5ad file
    adata = sc.read(file_path)

    # Get cell type column
    # cell_types = adata.obs[cell_type_column]

    # Get cell type names and their corresponding counts
    cell_type_counts = adata.obs['cell_type1'].value_counts()
    # Sort cell types by count (from most to least)
    sorted_cell_types = cell_type_counts.index.tolist()
    # Divide the required cell types
    first_cell_types = sorted_cell_types[:end]
    # Filtered cell type data
    adata = adata[adata.obs['cell_type1'].isin(first_cell_types), :]

    # Get cell type column
    cell_types = adata.obs[cell_type_column]
    num_cell_types = len(set(cell_types))

    # Split by cell type
    train_indices, test_indices = train_test_split(
        adata.obs.index, stratify=cell_types, test_size=test_size, random_state=random_state
    )

    # Get training and test sets
    X_train = adata[train_indices, :].X.todense()
    X_test = adata[test_indices, :].X.todense()
    y_train = adata[train_indices, :].obs[cell_type_column]
    y_test = adata[test_indices, :].obs[cell_type_column]

    # Print dataset sizes
    print(f"Training set X shape: {X_train.shape}")
    print(f"Test set X shape: {X_test.shape}")
    print(f"Training set labels shape: {y_train.shape}")
    print(f"Test set labels shape: {y_test.shape}")

    # Incremental class boolean labels, all set to False
    train_incremental_mask = np.zeros(y_train.shape, dtype=bool)
    test_incremental_mask = np.zeros(y_test.shape, dtype=bool)

    # Label encoding
    encoder = OneHotEncoder(sparse_output=True)
    y_train_encoded = encoder.fit_transform(y_train.to_numpy().reshape(-1, 1)).toarray()
    y_test_encoded = encoder.transform(y_test.to_numpy().reshape(-1, 1)).toarray()

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_train_tensor = torch.FloatTensor(y_train_encoded).to(device)
    y_test_tensor = torch.FloatTensor(y_test_encoded).to(device)
    train_incremental_tensor = torch.BoolTensor(train_incremental_mask).to(device)
    test_incremental_tensor = torch.BoolTensor(test_incremental_mask).to(device)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, train_incremental_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor, test_incremental_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "encoder": encoder,
        "device": device,
        "train_incremental_mask": train_incremental_mask,
        "test_incremental_mask": test_incremental_mask,
        "num_cell_types": num_cell_types,
    }


def cross_entropy_loss_fn(outputs, targets):
    return F.cross_entropy(outputs, targets, reduction="sum")


def uncertainty_loss_fn(max_probs, threshold):
    return ((max_probs - threshold).clamp(min=0) ** 2).sum()


def fuzzy_loss_fn(max_probs, threshold):
    return ((max_probs - threshold) ** 2).sum()


# General training function
def train(model, train_loader, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for data, target, incremental_mask in tqdm(train_loader, desc="Training", leave=False):
        data, target, incremental_mask = data.to(device), target.to(device), incremental_mask.to(device)
        optimizer.zero_grad()

        logits, gene_features, attentions = model(data)

        if target.dim() > 1:
            target = target.argmax(dim=1)

        outputs_dict = calculate_loss(
            logits,
            target,
            incremental_mask=incremental_mask,
            non_incremental_loss_fn=cross_entropy_loss_fn,
            uncertainty_loss_fn=uncertainty_loss_fn,
            fuzzy_loss_fn=fuzzy_loss_fn,
            use_incremental=False,
            threshold=0.5,
            alpha=1.0,
            beta=1.0
        )
        loss = outputs_dict['total_loss']
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        predicted = outputs_dict['predicted_labels']

        if predicted.dim() > 1:
            predicted = predicted.argmax(dim=1)

        if target.dim() > 1:
            target = target.argmax(dim=1)

        total_train += target.size(0)
        correct_train += (((predicted == target) & (incremental_mask == False)) | (predicted == -1)).sum().item()

    train_loss = running_loss / total_train
    train_accuracy = correct_train / total_train
    return train_loss, train_accuracy


# General validation function
def validate(model, test_loader, device):
    model.eval()
    running_test_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for data, target, incremental_mask in tqdm(test_loader, desc="Validation", leave=False):
            data, target, incremental_mask = data.to(device), target.to(device), incremental_mask.to(device)

            logits, gene_features, attentions = model(data)

            if target.dim() > 1:
                target = target.argmax(dim=1)

            outputs_dict = calculate_loss(
                logits,
                target,
                incremental_mask=incremental_mask,
                non_incremental_loss_fn=cross_entropy_loss_fn,
                uncertainty_loss_fn=uncertainty_loss_fn,
                fuzzy_loss_fn=fuzzy_loss_fn,
                use_incremental=False,
                threshold=0.5,
                alpha=1.0,
                beta=1.0
            )
            loss = outputs_dict['total_loss']
            running_test_loss += loss.item() * data.size(0)
            predicted = outputs_dict['predicted_labels']

            if predicted.dim() > 1:
                predicted = predicted.argmax(dim=1)

            if target.dim() > 1:
                target = target.argmax(dim=1)

            total_test += target.size(0)
            correct_test += (((predicted == target) & (incremental_mask == False)) | (predicted == -1)).sum().item()

    test_loss = running_test_loss / total_test
    test_accuracy = correct_test / total_test
    return test_loss, test_accuracy


# Early stopping mechanism
class EarlyStopping:
    def __init__(self, patience=5, verbose=True, delta=0, save_path=None):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_loss = None
        self.epochs_without_improvement = 0
        self.best_model_wts = None
        self.save_path = save_path
        self.best_epoch = None
        self.current_epoch = 0

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_wts = model.state_dict()
            self.best_epoch = self.current_epoch
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.best_model_wts = model.state_dict()
            self.best_epoch = self.current_epoch
            self.epochs_without_improvement = 0
            self.save_model(model)
        else:
            self.epochs_without_improvement += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.epochs_without_improvement} out of {self.patience}')
            if self.epochs_without_improvement >= self.patience:
                print("Early stopping triggered!")
                return True
        self.current_epoch += 1
        return False

    def save_model(self, model):
        # Save model weights to the specified path
        torch.save(model.state_dict(), self.save_path)
        print(f"Model saved to {self.save_path}")

    def load_best_model(self, model):
        # Load the best model
        model.load_state_dict(self.best_model_wts)


# General plotting function
def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies):
    # Plot loss graph
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.title('Loss vs Epoch')
    plt.show()

    # Plot accuracy graph
    plt.figure()
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.legend()
    plt.title('Accuracy vs Epoch')
    plt.show()


# Create neural network model
class Model(nn.Module):
    def __init__(self, input_size, num_genes, num_cell_types):
        super(Model, self).__init__()
        # self.mlp = MLP(input_size=input_size)
        self.GFUP = GeneFlowUNetPredictor(
            num_genes=num_genes,
            num_cell_types=num_cell_types,
            hidden_dims=[512, 256],
            dropout=0.3,
            activation='leaky_relu'
        )

    def forward(self, x):
        # x = self.mlp(x)
        logits, gene_features, attentions = self.GFUP(x)
        return logits, gene_features, attentions


if __name__ == "__main__":
    # Parameters
    file_path = r"He_Long_Bone_1024.h5ad"
    cell_type_column = 'cell_type1'
    batch_size = 256
    save_path = './pth/stage_01_model.pth'

    # Call the module
    data = load_and_preprocess_data(
        file_path=file_path,
        cell_type_column=cell_type_column,
        batch_size=batch_size,
        end=6
    )

    # Get DataLoader
    train_loader = data["train_loader"]
    test_loader = data["test_loader"]
    device = data["device"]
    encoder = data["encoder"]
    num_cell_types = data["num_cell_types"]

    # Print device information
    print(f"Using device: {device}")

    # Initialize model, loss function, and optimizer
    model = Model(input_size=1024, num_genes=1024, num_cell_types=num_cell_types).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # Initialize early stopping mechanism
    early_stopping = EarlyStopping(patience=10, verbose=True, save_path=save_path)

    # Training and validation
    num_epochs = 50
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_loader, optimizer, device)
        test_loss, test_accuracy = validate(model, test_loader, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

        # Early stopping check
        if early_stopping(test_loss, model):
            break

    # Load the best model
    early_stopping.load_best_model(model)

    # Plot the graphs
    plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies)

    # Record the index of the best epoch saved by early stopping (0-based)
    # This value is typically returned or recorded in the early stopping implementation
    best_epoch = early_stopping.best_epoch  # Best epoch saved by early stopping (0-based)

    # Final evaluation
    final_train_accuracy = train_accuracies[best_epoch]
    final_test_accuracy = test_accuracies[best_epoch]
    print(f'Final Training Accuracy: {final_train_accuracy:.4f} (Epoch {best_epoch + 1})')
    print(f'Final Testing Accuracy: {final_test_accuracy:.4f} (Epoch {best_epoch + 1})')
