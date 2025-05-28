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
# from src.CDSCIA.compute_loss import *
from CDSCIA.src.CDSCIA.pred_loss import *
from CDSCIA.src.CDSCIA.incremental_loss import *
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
        end_1=6,
        end_2=8
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
        end_1 (int): Cursor for previous stage categories.
        end_2 (int): Cursor for new stage categories.

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

    # Get cell type names and their corresponding counts
    cell_type_counts = adata.obs[cell_type_column].value_counts()
    # Sort cell types by count (from most to least)
    sorted_cell_types = cell_type_counts.index.tolist()
    # Divide the required cell types
    old_cell_types = sorted_cell_types[:end_1]
    new_cell_types = sorted_cell_types[end_1:end_2]

    # Add "is_incremental" column (if it's in old_cell_types, is_incremental is False, otherwise True)
    adata.obs['is_incremental'] = ~adata.obs[cell_type_column].isin(old_cell_types)

    # Filtered cell type data
    selected_cell_types = old_cell_types + new_cell_types
    adata = adata[adata.obs[cell_type_column].isin(selected_cell_types), :]

    # Get cell type column
    cell_types = adata.obs[cell_type_column]

    # Split by cell type
    train_indices, test_indices = train_test_split(
        adata.obs.index, stratify=cell_types, test_size=test_size, random_state=random_state
    )

    # Get data from the training set
    X_train = adata[train_indices, :].X.todense()
    y_train = adata[train_indices, :].obs[cell_type_column]
    z_train = adata[train_indices, :].obs['is_incremental']

    # Get data from the test set
    X_test = adata[test_indices, :].X.todense()
    y_test = adata[test_indices, :].obs[cell_type_column]
    z_test = adata[test_indices, :].obs['is_incremental']

    # Split old data (old_cell_types) from the test set
    X_test_old = X_test[z_test == False]
    y_test_old = y_test[z_test == False]
    z_test_old = z_test[z_test == False]

    # Split new data (new_cell_types) from the test set
    X_test_new = X_test[z_test == True]
    y_test_new = y_test[z_test == True]
    z_test_new = z_test[z_test == True]

    # Print dataset sizes
    print(f"Training set X shape: {X_train.shape}")
    print(f"Training set labels shape: {y_train.shape}")
    print(f"Training set is incremental shape: {z_train.shape}")
    print(f"Test set X shape: {X_test.shape}")
    print(f"Test set labels shape: {y_test.shape}")
    print(f"Test set is incremental shape: {z_test.shape}")
    print(f"Test set X of old shape: {X_test_old.shape}")
    print(f"Test set X of new shape: {X_test_new.shape}")

    # Label encoding
    encoder = OneHotEncoder(sparse_output=True)
    y_train_encoded = encoder.fit_transform(y_train.to_numpy().reshape(-1, 1)).toarray()
    y_test_encoded = encoder.transform(y_test.to_numpy().reshape(-1, 1)).toarray()
    y_test_old_encoded = encoder.transform(y_test_old.to_numpy().reshape(-1, 1)).toarray()
    y_test_new_encoded = encoder.transform(y_test_new.to_numpy().reshape(-1, 1)).toarray()

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train_encoded).to(device)
    z_train_tensor = torch.BoolTensor(z_train).to(device)

    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test_encoded).to(device)
    z_test_tensor = torch.BoolTensor(z_test).to(device)

    X_test_old_tensor = torch.FloatTensor(X_test_old).to(device)
    y_test_old_tensor = torch.FloatTensor(y_test_old_encoded).to(device)
    z_test_old_tensor = torch.BoolTensor(z_test_old).to(device)

    X_test_new_tensor = torch.FloatTensor(X_test_new).to(device)
    y_test_new_tensor = torch.FloatTensor(y_test_new_encoded).to(device)
    z_test_new_tensor = torch.BoolTensor(z_test_new).to(device)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, z_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor, z_test_tensor)
    test_old_dataset = TensorDataset(X_test_old_tensor, y_test_old_tensor, z_test_old_tensor)
    test_new_dataset = TensorDataset(X_test_new_tensor, y_test_new_tensor, z_test_new_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    test_old_loader = DataLoader(test_old_dataset, batch_size=batch_size, shuffle=True)
    test_new_loader = DataLoader(test_new_dataset, batch_size=batch_size, shuffle=True)

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "test_old_loader": test_old_loader,
        "test_new_loader": test_new_loader,
        "encoder": encoder,
        "device": device,
        "old_cell_types": end_1,
        "new_cell_types": end_2 - end_1,
        "num_cell_types": end_2,
    }


def cross_entropy_loss_fn(outputs, targets):
    return F.cross_entropy(outputs, targets, reduction="sum")


def uncertainty_loss_fn(max_probs, threshold):
    return ((max_probs - threshold).clamp(min=0) ** 2).sum()


def fuzzy_loss_fn(max_probs, threshold):
    return ((max_probs - threshold) ** 2).sum()


# Define distillation loss function
def distillation_loss(teacher_outputs, student_outputs, alpha, beta, gamma, temperature):
    """
    Compute distillation loss, including gene features, classification logits, and attention feature maps.

    teacher_outputs: Outputs from the teacher model (logits, gene_features, attentions)
    student_outputs: Outputs from the student model (logits, gene_features, attentions)
    alpha, beta, gamma: Loss weights
    temperature: Temperature for logits distillation
    """
    # Unpack teacher and student model outputs
    teacher_logits, teacher_gene_features, teacher_attentions = teacher_outputs
    student_logits, student_gene_features, student_attentions = student_outputs

    # 1. Gene feature distillation loss
    gene_loss = F.mse_loss(student_gene_features, teacher_gene_features)

    # 2. Classification logits distillation loss (KL divergence)
    # Only distill for shared classes
    shared_student_logits = student_logits[:, :teacher_logits.shape[1]]  # Take the shared classes part
    student_logits_soft = F.log_softmax(shared_student_logits / temperature, dim=-1)
    teacher_logits_soft = F.softmax(teacher_logits / temperature, dim=-1)
    logits_loss = F.kl_div(student_logits_soft, teacher_logits_soft, reduction='batchmean') * (temperature ** 2)

    # 3. Attention feature map distillation loss (sum over multiple scales)
    attention_loss = 0.0
    for key in teacher_attentions:
        teacher_map = teacher_attentions[key]
        student_map = student_attentions[key]
        attention_loss += F.mse_loss(student_map, teacher_map)

    # Total loss
    kd_loss = alpha * gene_loss + beta * logits_loss + gamma * attention_loss
    return kd_loss, gene_loss, logits_loss, attention_loss


def wrap_outputs(logits, gene_features, attentions):
    """
    Wrap the model outputs into a unified structure.

    Parameters:
        logits: Classification logits from the model
        gene_features: Gene features
        attentions: Attention feature maps (dict or other format)

    Returns:
        Wrapped outputs (logits, gene_features, attentions)
    """
    return logits, gene_features, attentions


# General training function
def train(model_1, model_2, train_loader, optimizer_1, optimizer_2, device):
    model_1.train()
    model_2.train()

    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for data, target, incremental_mask in tqdm(train_loader, desc="Training", leave=False):
        data, target, incremental_mask = data.to(device), target.to(device), incremental_mask.to(device)
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()

        # Teacher model outputs
        logits_1, gene_features_1, attentions_1 = model_1(data)
        # Student model outputs
        logits_2, gene_features_2, attentions_2 = model_2(data)

        # Teacher model outputs
        teacher_outputs = wrap_outputs(logits_1, gene_features_1, attentions_1)
        # Student model outputs
        student_outputs = wrap_outputs(logits_2, gene_features_2, attentions_2)

        # Compute distillation loss
        kd_loss, _, _, _ = distillation_loss(
            teacher_outputs, student_outputs,
            alpha=0.5, beta=1.0, gamma=0.2, temperature=2.0
        )

        if target.dim() > 1:
            target = target.argmax(dim=1)

        # Define fuzzy-guided incremental loss calculation function
        loss_t_fn = TeacherIncrementalCrossEntropyLoss(
            threshold=0.5, lambda_fuzzy=2.0
        )

        loss_s_fn = StudentIncrementalCrossEntropyLoss(
            threshold=0.5, lambda_fuzzy=2.0, gamma=1.5, alpha=2.0
        )

        # Use fuzzy reference incremental loss (teacher)
        loss_1, pred_labels_1 = loss_t_fn(logits=logits_1, target=target, is_incremental=True)
        # Use cross-entropy incremental loss (student)
        loss_2, pred_labels_2 = loss_s_fn(logits_2, target, incremental_mask, is_incremental=False)
        # Add distillation loss
        loss_2 = 0.3 * loss_2 + 0.5 * kd_loss + 0.2 * loss_1

        loss_2.backward()
        optimizer_2.step()

        running_loss += (loss_2.item()) * data.size(0)
        predicted = pred_labels_2

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

            # Define fuzzy-guided incremental loss calculation function
            loss_s_fn = StudentIncrementalCrossEntropyLoss(
                threshold=0.5, lambda_fuzzy=2.0, gamma=1.5, alpha=2.0
            )

            loss, pred_labels = loss_s_fn(logits, target, incremental_mask, is_incremental=False)
            running_test_loss += loss.item() * data.size(0)
            predicted = pred_labels

            if predicted.dim() > 1:
                predicted = predicted.argmax(dim=1)

            if target.dim() > 1:
                target = target.argmax(dim=1)

            total_test += target.size(0)
            correct_test += (predicted == target).sum().item()

    test_loss = running_test_loss / total_test
    test_accuracy = correct_test / total_test
    return test_loss, test_accuracy


# Early stopping mechanism
class EarlyStopping:
    def __init__(self, patience=5, verbose=True, delta=0, save_path=None, weights=None):
        """
        Initialize the EarlyStopping class.

        Parameters:
        patience (int, default 5): Number of epochs with no improvement after which training will be stopped.
        verbose (bool, default True): If True, prints a message for each validation loss improvement.
        delta (float, default 0): Minimum change in the monitored quantity to qualify as an improvement.
        save_path (str, default None): Path for saving the best model. If provided, the best model will be saved when early stopping is triggered.
        weights (list, default None): Weights for the losses, used to balance contributions from different losses.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_loss = None
        self.epochs_without_improvement = 0
        self.best_model_wts = None
        self.save_path = save_path
        self.best_epoch = None
        self.current_epoch = 0
        self.weights = weights if weights is not None else [1.0, 1.0, 1.0]  # Default weight 1 for each loss

    def __call__(self, val_losses, model):
        """
        Determine whether early stopping should be triggered.

        Parameters:
        val_losses (list): List containing multiple losses [val_loss, val_old_loss, val_new_loss].
        model (torch.nn.Module): The currently trained model, used to save the best weights.

        Returns:
        bool: True if early stopping criteria are met, False otherwise.
        """
        # Normalize losses
        normalized_losses = [(loss / (loss + 1e-8)) for loss in val_losses]  # Prevent division by zero
        # Weighted sum of losses
        total_loss = sum(w * loss for w, loss in zip(self.weights, normalized_losses))

        if self.best_loss is None:
            # Initialize best loss
            self.best_loss = total_loss
            self.best_model_wts = model.state_dict()
            self.best_epoch = self.current_epoch
        elif total_loss < self.best_loss - self.delta:
            # If loss improves
            self.best_loss = total_loss
            self.best_model_wts = model.state_dict()
            self.best_epoch = self.current_epoch
            self.epochs_without_improvement = 0
            self.save_model(model)
        else:
            # If loss does not improve
            self.epochs_without_improvement += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.epochs_without_improvement} out of {self.patience}')
            if self.epochs_without_improvement >= self.patience:
                print("Early stopping triggered!")
                return True
        self.current_epoch += 1
        return False

    def save_model(self, model):
        """
        Save the current model weights to the specified path.

        Parameters:
        model (torch.nn.Module): The model whose weights need to be saved.
        """
        if self.save_path:
            torch.save(model.state_dict(), self.save_path)
            print(f"Model saved to {self.save_path}")


# General plotting function
def plot_metrics(
        train_losses, test_losses, test_old_losses, test_new_losses,
        train_accuracies, test_accuracies, test_old_accuracies, test_new_accuracies
):
    # Plot loss graph
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.plot(test_old_losses, label='Test Old Loss')
    plt.plot(test_new_losses, label='Test New Loss')
    plt.legend()
    plt.title('Loss vs Epoch')
    plt.show()

    # Plot accuracy graph
    plt.figure()
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.plot(test_old_accuracies, label='Test Old Accuracy')
    plt.plot(test_new_accuracies, label='Test New Accuracy')
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
    file_path = 'He_Long_Bone_1024.h5ad'
    cell_type_column = 'cell_type1'
    batch_size = 256
    checkpoint_path = './pth/stage_01_model.pth'
    save_path = './pth/stage_02_model.pth'

    # Call the module
    data = load_and_preprocess_data(
        file_path=file_path,
        cell_type_column=cell_type_column,
        batch_size=batch_size,
        end_1=6,
        end_2=8
    )

    # Get DataLoader
    train_loader = data["train_loader"]
    test_loader = data["test_loader"]
    test_old_loader = data["test_old_loader"]
    test_new_loader = data["test_new_loader"]
    device = data["device"]
    encoder = data["encoder"]
    old_cell_types = data["old_cell_types"]
    new_cell_types = data["new_cell_types"]
    num_cell_types = data["num_cell_types"]

    # Print device information
    print(f"Using device: {device}")

    # Initialize models, loss functions, and optimizers
    model_1 = Model(input_size=1024, num_genes=1024, num_cell_types=old_cell_types).to(device)
    model_2 = Model(input_size=1024, num_genes=1024, num_cell_types=num_cell_types).to(device)
    optimizer_1 = optim.Adam(model_1.parameters(), lr=0.00001)
    optimizer_2 = optim.Adam(model_2.parameters(), lr=0.00001)

    # Load weights for model_1
    try:
        model_1.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True), strict=False)
        print(f"Successfully loaded weights from {checkpoint_path}")
    except Exception as e:
        print(f"Failed to load weights: {e}")

    # Initialize early stopping mechanism
    early_stopping = EarlyStopping(patience=10, verbose=True, save_path=save_path)

    # Training and validation
    num_epochs = 50
    train_losses, test_losses, test_old_losses, test_new_losses = [], [], [], []
    train_accuracies, test_accuracies, test_old_accuracies, test_new_accuracies = [], [], [], []

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model_1, model_2, train_loader, optimizer_1, optimizer_2, device)
        test_loss, test_accuracy = validate(model_2, test_loader, device)
        test_old_loss, test_old_accuracy = validate(model_2, test_old_loader, device)
        test_new_loss, test_new_accuracy = validate(model_2, test_new_loader, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_old_losses.append(test_old_loss)
        test_new_losses.append(test_new_loss)

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        test_old_accuracies.append(test_old_accuracy)
        test_new_accuracies.append(test_new_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, '
              f'Test Old Loss: {test_old_loss:.4f}, Test Old Accuracy: {test_old_accuracy:.4f}, '
              f'Test New Loss: {test_new_loss:.4f}, Test New Accuracy: {test_new_accuracy:.4f},')

        # Early stopping check
        if early_stopping([test_loss, test_old_loss, test_new_loss], model_2):
            break

    # Plot the graphs
    plot_metrics(
        train_losses, test_losses, test_old_losses, test_new_losses,
        train_accuracies, test_accuracies, test_old_accuracies, test_new_accuracies
    )

    # Record the index of the best epoch saved by early stopping (0-based)
    # This value is typically returned or recorded in the early stopping implementation
    best_epoch = early_stopping.best_epoch  # Best epoch saved by early stopping (0-based)

    # Get the accuracies corresponding to the best model saved by early stopping
    best_train_accuracy = train_accuracies[best_epoch]
    best_test_accuracy = test_accuracies[best_epoch]
    best_test_old_accuracy = test_old_accuracies[best_epoch]
    best_test_new_accuracy = test_new_accuracies[best_epoch]

    # Output results
    print(f'Best Model Training Accuracy: {best_train_accuracy:.4f} (Epoch {best_epoch + 1})')
    print(f'Best Model Testing Accuracy: {best_test_accuracy:.4f} (Epoch {best_epoch + 1})')
    print(f'Best Model Testing Old Accuracy: {best_test_old_accuracy:.4f} (Epoch {best_epoch + 1})')
    print(f'Best Model Testing New Accuracy: {best_test_new_accuracy:.4f} (Epoch {best_epoch + 1})')
