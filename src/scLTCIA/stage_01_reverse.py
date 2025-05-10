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
import random
import scanpy as sc
from sklearn.model_selection import train_test_split
from CDSCIA.src.CDSCIA.GFUP import *
from CDSCIA.src.CDSCIA.compute_loss import *
from CDSCIA.src.CDSCIA.kd_loss import *
from CDSCIA.src.CDSCIA.utils import *
from CDSCIA.src.test_mlp.model_mlp import *


def set_global_seed(seed):
    """
    Set a global random seed for reproducibility across Python, NumPy, and PyTorch (including CUDA).

    Parameters:
        seed (int): The seed value to set.
    """
    # Set Python's built-in random seed
    random.seed(seed)

    # Set NumPy's random seed
    np.random.seed(seed)

    # Set PyTorch's random seed
    torch.manual_seed(seed)

    # If using GPUs, ensure reproducibility for CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If multi-GPU

        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Example usage
random_state = 20020130
set_global_seed(random_state)


def shuffle_list(input_list, seed=20020130):
    """
    Shuffle a list with a fixed random seed to ensure reproducibility.

    Parameters:
        input_list (list): The list to shuffle.
        seed (int): The random seed to use for shuffling.

    Returns:
        list: A new list that is a shuffled version of the input list.
    """
    random.seed(seed)  # Set the seed for reproducibility
    shuffled = input_list[:]  # Create a copy of the input list
    random.shuffle(shuffled)  # Shuffle the copy
    return shuffled


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
    加载和预处理 h5ad 数据文件，并生成 DataLoader。

    Args:
        file_path (str): h5ad 文件的路径。
        cell_type_column (str): 细胞类型所在列名。
        test_size (float): 测试集比例。
        random_state (int): 随机种子。
        batch_size (int): DataLoader 的批次大小。
        device (torch.device, optional): 数据加载的设备，默认为自动检测。
        end (int): 初始阶段类别游标。

    Returns:
        dict: 包含以下键值对的数据字典：
            - 'train_loader': 训练数据的 DataLoader。
            - 'test_loader': 测试数据的 DataLoader。
            - 'encoder': 训练时使用的 OneHotEncoder 实例。
            - 'device': 使用的设备。
            - 'train_incremental_mask': 训练集的增量类布尔掩码。
            - 'test_incremental_mask': 测试集的增量类布尔掩码。
    """
    # 检查设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读取 h5ad 文件
    adata = sc.read(file_path)

    # 获取细胞类型列
    # cell_types = adata.obs[cell_type_column]

    # 获取细胞类型名称及其对应的数量
    cell_type_counts = adata.obs['cell_type1'].value_counts()
    # 按照细胞类型数量排序（从多到少）
    sorted_cell_types = cell_type_counts.index.tolist()
    sorted_cell_types = shuffle_list(sorted_cell_types)
    # 划分需要的细胞类型
    first_cell_types = sorted_cell_types[:end]
    # 筛选出后的细胞类型数据
    adata = adata[adata.obs['cell_type1'].isin(first_cell_types), :]

    # 获取细胞类型列
    cell_types = adata.obs[cell_type_column]
    num_cell_types = len(set(cell_types))

    # 按细胞类型进行划分
    train_indices, test_indices = train_test_split(
        adata.obs.index, stratify=cell_types, test_size=test_size, random_state=random_state
    )

    # 获取训练集和测试集
    X_train = adata[train_indices, :].X.todense()
    X_test = adata[test_indices, :].X.todense()
    y_train = adata[train_indices, :].obs[cell_type_column]
    y_test = adata[test_indices, :].obs[cell_type_column]

    # 打印数据集大小
    print(f"Training set X shape: {X_train.shape}")
    print(f"Test set X shape: {X_test.shape}")
    print(f"Training set labels shape: {y_train.shape}")
    print(f"Test set labels shape: {y_test.shape}")

    # 增量类布尔标签，全部设置为 False
    train_incremental_mask = np.zeros(y_train.shape, dtype=bool)
    test_incremental_mask = np.zeros(y_test.shape, dtype=bool)

    # 标签编码
    encoder = OneHotEncoder(sparse_output=True)
    y_train_encoded = encoder.fit_transform(y_train.to_numpy().reshape(-1, 1)).toarray()
    y_test_encoded = encoder.transform(y_test.to_numpy().reshape(-1, 1)).toarray()

    # 转换为 PyTorch 张量
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_train_tensor = torch.FloatTensor(y_train_encoded).to(device)
    y_test_tensor = torch.FloatTensor(y_test_encoded).to(device)
    train_incremental_tensor = torch.BoolTensor(train_incremental_mask).to(device)
    test_incremental_tensor = torch.BoolTensor(test_incremental_mask).to(device)

    # 创建 DataLoader
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


# 通用训练函数
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


# 通用验证函数
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


# 早停机制
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
        # 保存模型权重到指定路径
        torch.save(model.state_dict(), self.save_path)
        print(f"Model saved to {self.save_path}")

    def load_best_model(self, model):
        # 加载最佳模型
        model.load_state_dict(self.best_model_wts)


# 通用绘图函数
def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies):
    # 绘制损失图
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.title('Loss vs Epoch')
    plt.show()

    # 绘制准确率图
    plt.figure()
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.legend()
    plt.title('Accuracy vs Epoch')
    plt.show()


# 创建神经网络模型
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
    # 参数
    file_path = r'He_Long_Bone_1024.h5ad'
    cell_type_column = 'cell_type1'
    batch_size = 256
    save_path = './pth/stage_01_model_reverse.pth'

    # 调用模块
    data = load_and_preprocess_data(
        file_path=file_path,
        cell_type_column=cell_type_column,
        batch_size=batch_size,
        end=6
    )

    # 获取 DataLoader
    train_loader = data["train_loader"]
    test_loader = data["test_loader"]
    device = data["device"]
    encoder = data["encoder"]
    num_cell_types = data["num_cell_types"]

    # 打印设备信息
    print(f"Using device: {device}")

    # 初始化模型、损失函数和优化器
    model = Model(input_size=1024, num_genes=1024, num_cell_types=num_cell_types).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # 初始化早停机制
    early_stopping = EarlyStopping(patience=15, verbose=True, save_path=save_path)

    # 训练和验证
    num_epochs = 100
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

        # 早停检查
        if early_stopping(test_loss, model):
            break

    # 加载最好的模型
    early_stopping.load_best_model(model)

    # 绘制图表
    plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies)

    # 记录早停保存的最佳轮次的索引（0-based）
    # 通常在早停实现中会返回或记录这个值
    best_epoch = early_stopping.best_epoch  # 早停保存的最佳轮次（0-based）

    # 最终评估
    final_train_accuracy = train_accuracies[best_epoch]
    final_test_accuracy = test_accuracies[best_epoch]
    print(f'Final Training Accuracy: {final_train_accuracy:.4f} (Epoch {best_epoch + 1})')
    print(f'Final Testing Accuracy: {final_test_accuracy:.4f} (Epoch {best_epoch + 1})')
