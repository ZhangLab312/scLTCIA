import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from ipykernel.pickleutil import cell_type
from sklearn.metrics import accuracy_score, f1_score, silhouette_score, homogeneity_score, normalized_mutual_info_score, \
    adjusted_rand_score
from sklearn.manifold import TSNE
from tqdm import tqdm
import random
import scanpy as sc
from sklearn.model_selection import train_test_split
from src.CDSCIA.GFUP import *
# from src.CDSCIA.compute_loss import *
from src.CDSCIA.pred_loss import *
from src.CDSCIA.incremental_loss import *
from src.CDSCIA.kd_loss import *
from src.CDSCIA.utils import *
from src.test_ciform.model_ciform import *


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
        end_1=6,
        end_2=8
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
        end_1 (int): 上阶段类别游标。
        end_2 (int): 新阶段类别游标。

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
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 读取 h5ad 文件
    adata = sc.read(file_path)

    # 获取细胞类型名称及其对应的数量
    cell_type_counts = adata.obs[cell_type_column].value_counts()
    # 按照细胞类型数量排序（从多到少）
    sorted_cell_types = cell_type_counts.index.tolist()
    sorted_cell_types = shuffle_list(sorted_cell_types)
    # 划分需要的细胞类型
    old_cell_types = sorted_cell_types[:end_1]
    new_cell_types = sorted_cell_types[end_1:end_2]

    # 添加 "is_incremental" 列（如果是 old_cell_types，is_incremental 为 False，反之亦然）
    adata.obs['is_incremental'] = ~adata.obs[cell_type_column].isin(old_cell_types)

    # 筛选出后的细胞类型数据
    selected_cell_types = old_cell_types + new_cell_types
    adata = adata[adata.obs[cell_type_column].isin(selected_cell_types), :]

    # 获取细胞类型列
    cell_types = adata.obs[cell_type_column]

    # 按细胞类型进行划分
    train_indices, test_indices = train_test_split(
        adata.obs.index, stratify=cell_types, test_size=test_size, random_state=random_state
    )

    # 获取训练集中的数据
    X_train = adata[train_indices, :].X.todense()
    y_train = adata[train_indices, :].obs[cell_type_column]
    z_train = adata[train_indices, :].obs['is_incremental']

    # 获取测试集中的数据
    X_test = adata[test_indices, :].X.todense()
    y_test = adata[test_indices, :].obs[cell_type_column]
    z_test = adata[test_indices, :].obs['is_incremental']

    # 数据reshape适应CIForm
    X_train = split(X_train, gap=256)
    X_test = split(X_test, gap=256)

    # 从测试集拆分老数据 (old_cell_types)
    X_test_old = X_test[z_test == False]
    y_test_old = y_test[z_test == False]
    z_test_old = z_test[z_test == False]

    # 从测试集拆分新数据 (new_cell_types)
    X_test_new = X_test[z_test == True]
    y_test_new = y_test[z_test == True]
    z_test_new = z_test[z_test == True]

    # 打印数据集大小
    print(f"Training set X shape: {X_train.shape}")
    print(f"Training set labels shape: {y_train.shape}")
    print(f"Training set is incremental shape: {z_train.shape}")
    print(f"Test set X shape: {X_test.shape}")
    print(f"Test set labels shape: {y_test.shape}")
    print(f"Test set is incremental shape: {z_test.shape}")
    print(f"Test set X of old shape: {X_test_old.shape}")
    print(f"Test set X of new shape: {X_test_new.shape}")

    # 标签编码
    encoder = OneHotEncoder(sparse_output=True)
    y_train_encoded = encoder.fit_transform(y_train.to_numpy().reshape(-1, 1)).toarray()
    y_test_encoded = encoder.transform(y_test.to_numpy().reshape(-1, 1)).toarray()
    y_test_old_encoded = encoder.transform(y_test_old.to_numpy().reshape(-1, 1)).toarray()
    y_test_new_encoded = encoder.transform(y_test_new.to_numpy().reshape(-1, 1)).toarray()

    # 转换为 PyTorch 张量
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

    # 创建 DataLoader
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


# 定义蒸馏损失函数
def distillation_loss(teacher_outputs, student_outputs, alpha, beta, gamma, temperature):
    """
    计算蒸馏损失，包括基因特征、分类logits和注意力特征图。

    teacher_outputs: 教师模型输出 (logits, gene_features, attentions)
    student_outputs: 学生模型输出 (logits, gene_features, attentions)
    alpha, beta, gamma: 损失权重
    temperature: logits 蒸馏温度
    """
    # 解包教师和学生模型输出
    teacher_logits, teacher_gene_features, teacher_attentions = teacher_outputs
    student_logits, student_gene_features, student_attentions = student_outputs

    # 1. 基因特征蒸馏损失
    gene_loss = F.mse_loss(student_gene_features, teacher_gene_features)

    # 2. 分类logits蒸馏损失 (KL散度)
    # 只对共有类别进行蒸馏
    shared_student_logits = student_logits[:, :teacher_logits.shape[1]]  # 取共有类别部分
    student_logits_soft = F.log_softmax(shared_student_logits / temperature, dim=-1)
    teacher_logits_soft = F.softmax(teacher_logits / temperature, dim=-1)
    logits_loss = F.kl_div(student_logits_soft, teacher_logits_soft, reduction='batchmean') * (temperature ** 2)

    # 3. 注意力特征图蒸馏损失 (多尺度求和)
    attention_loss = 0.0
    for key in teacher_attentions:
        teacher_map = teacher_attentions[key]
        student_map = student_attentions[key]
        attention_loss += F.mse_loss(student_map, teacher_map)

    # 总损失
    kd_loss = alpha * gene_loss + beta * logits_loss + gamma * attention_loss
    return kd_loss, gene_loss, logits_loss, attention_loss


def wrap_outputs(logits, gene_features, attentions):
    """
    将模型的输出封装成统一结构。

    Parameters:
        logits: 模型的分类 logits
        gene_features: 基因特征
        attentions: 注意力特征图 (dict 或其他格式)

    Returns:
        包装后的输出 (logits, gene_features, attentions)
    """
    return logits, gene_features, attentions


# 通用训练函数
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

        # 教师模型输出
        logits_1, gene_features_1, attentions_1 = model_1(data)
        # 学生模型输出
        logits_2, gene_features_2, attentions_2 = model_2(data)

        # 教师模型输出
        teacher_outputs = wrap_outputs(logits_1, gene_features_1, attentions_1)
        # 学生模型输出
        student_outputs = wrap_outputs(logits_2, gene_features_2, attentions_2)

        # 计算蒸馏损失
        kd_loss, _, _, _ = distillation_loss(
            teacher_outputs, student_outputs,
            alpha=0.5, beta=1.0, gamma=0.2, temperature=2.0
        )

        if target.dim() > 1:
            target = target.argmax(dim=1)

        # 定义模糊引导的增量损失计算函数
        loss_t_fn = TeacherIncrementalCrossEntropyLoss(
            threshold=0.5, lambda_fuzzy=2.0
        )

        loss_s_fn = StudentIncrementalCrossEntropyLoss(
            threshold=0.5, lambda_fuzzy=2.0, gamma=1.5, alpha=2.0
        )

        # 使用模糊引用增量损失（教师）
        loss_1, pred_labels_1 = loss_t_fn(logits=logits_1, target=target, is_incremental=True)
        # 使用交叉熵增量损失（学生）
        loss_2, pred_labels_2 = loss_s_fn(logits_2, target, incremental_mask, is_incremental=False)
        # 增加蒸馏损失
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

            # 定义模糊引导的增量损失计算函数
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


# 早停机制
class EarlyStopping:
    def __init__(self, patience=5, verbose=True, delta=0, save_path=None, weights=None):
        """
        初始化 EarlyStopping 类。

        参数:
        patience (int, 默认值 5): 如果验证集损失在连续多少轮内没有改善，则触发早停机制。
        verbose (bool, 默认值 True): 如果为 True，则在每次验证集损失没有改善时打印信息。
        delta (float, 默认值 0): 用于控制“改善”的定义。如果当前验证集损失比最佳损失小于 delta，则认为有改善。
        save_path (str, 默认值 None): 模型保存路径。若提供，则在早停触发时保存最佳模型。
        weights (list, 默认值 None): 损失的权重，用于平衡不同损失的贡献。
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
        self.weights = weights if weights is not None else [1.0, 1.0, 1.0]  # 默认每个损失权重为 1

    def __call__(self, val_losses, model):
        """
        判断是否触发早停。

        参数:
        val_losses (list): 包含多个损失的列表 [val_loss, val_old_loss, val_new_loss]。
        model (torch.nn.Module): 当前训练的模型，用于保存最佳权重。

        返回:
        bool: 如果满足早停条件，则返回 True，否则返回 False。
        """
        # 归一化损失
        normalized_losses = [(loss / (loss + 1e-8)) for loss in val_losses]  # 防止除以 0
        # 加权合并损失
        total_loss = sum(w * loss for w, loss in zip(self.weights, normalized_losses))

        if self.best_loss is None:
            # 初始化最佳损失
            self.best_loss = total_loss
            self.best_model_wts = model.state_dict()
            self.best_epoch = self.current_epoch
        elif total_loss < self.best_loss - self.delta:
            # 如果损失改善
            self.best_loss = total_loss
            self.best_model_wts = model.state_dict()
            self.best_epoch = self.current_epoch
            self.epochs_without_improvement = 0
            self.save_model(model)
        else:
            # 如果损失没有改善
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
        保存当前模型的权重到指定路径。

        参数:
        model (torch.nn.Module): 需要保存权重的模型。
        """
        if self.save_path:
            torch.save(model.state_dict(), self.save_path)
            print(f"Model saved to {self.save_path}")


# 通用绘图函数
def plot_metrics(
        train_losses, test_losses, test_old_losses, test_new_losses,
        train_accuracies, test_accuracies, test_old_accuracies, test_new_accuracies
):
    # 绘制损失图
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.plot(test_old_losses, label='Test Old Loss')
    plt.plot(test_new_losses, label='Test New Loss')
    plt.legend()
    plt.title('Loss vs Epoch')
    plt.show()

    # 绘制准确率图
    plt.figure()
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.plot(test_old_accuracies, label='Test Old Accuracy')
    plt.plot(test_new_accuracies, label='Test New Accuracy')
    plt.legend()
    plt.title('Accuracy vs Epoch')
    plt.show()


# 创建神经网络模型
class Model(nn.Module):
    def __init__(self, num_genes, num_cell_types, nhead, d_model, dropout):
        super(Model, self).__init__()
        self.ciform = CIForm(nhead=nhead, d_model=d_model, dropout=dropout)
        self.GFUP = GeneFlowUNetPredictor(
            num_genes=num_genes,
            num_cell_types=num_cell_types,
            hidden_dims=[512, 256],
            dropout=0.3,
            activation='leaky_relu'
        )

    def forward(self, x):
        x = self.ciform(x)
        logits, gene_features, attentions = self.GFUP(x)
        return logits, gene_features, attentions


if __name__ == "__main__":
    # 参数
    file_path = '../../../data/scrna_data/Inter/Vento-Tormo_10x_1024.h5ad'
    cell_type_column = 'cell_type1'
    batch_size = 256
    checkpoint_path = './pth/stage_03_model_reverse.pth'
    save_path = './pth/stage_04_model_reverse.pth'

    # 调用模块
    data = load_and_preprocess_data(
        file_path=file_path,
        cell_type_column=cell_type_column,
        batch_size=batch_size,
        end_1=18,
        end_2=25
    )

    # 获取 DataLoader
    train_loader = data["train_loader"]
    test_loader = data["test_loader"]
    test_old_loader = data["test_old_loader"]
    test_new_loader = data["test_new_loader"]
    device = data["device"]
    encoder = data["encoder"]
    old_cell_types = data["old_cell_types"]
    new_cell_types = data["new_cell_types"]
    num_cell_types = data["num_cell_types"]

    # 打印设备信息
    print(f"Using device: {device}")

    # 初始化模型、损失函数和优化器
    model_1 = Model(num_genes=1024, num_cell_types=num_cell_types, nhead=2, d_model=256, dropout=0.3).to(device)
    model_2 = Model(num_genes=1024, num_cell_types=num_cell_types, nhead=2, d_model=256, dropout=0.3).to(device)
    optimizer_1 = optim.Adam(model_1.parameters(), lr=0.00005)
    optimizer_2 = optim.Adam(model_2.parameters(), lr=0.00005)

    # 加载 model_1 的权重
    try:
        model_1.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True), strict=False)
        print(f"Successfully loaded weights from {checkpoint_path}")
    except Exception as e:
        print(f"Failed to load weights: {e}")

    # 初始化早停机制
    early_stopping = EarlyStopping(patience=15, verbose=True, save_path=save_path)

    # 训练和验证
    num_epochs = 100
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

        # 早停检查
        if early_stopping([test_loss, test_old_loss, test_new_loss], model_2):
            break

    # 绘制图表
    plot_metrics(
        train_losses, test_losses, test_old_losses, test_new_losses,
        train_accuracies, test_accuracies, test_old_accuracies, test_new_accuracies
    )

    # 记录早停保存的最佳轮次的索引（0-based）
    # 通常在早停实现中会返回或记录这个值
    best_epoch = early_stopping.best_epoch  # 早停保存的最佳轮次（0-based）

    # 获取早停保存的最佳模型对应的准确率
    best_train_accuracy = train_accuracies[best_epoch]
    best_test_accuracy = test_accuracies[best_epoch]
    best_test_old_accuracy = test_old_accuracies[best_epoch]
    best_test_new_accuracy = test_new_accuracies[best_epoch]

    # 输出结果
    print(f'Best Model Training Accuracy: {best_train_accuracy:.4f} (Epoch {best_epoch + 1})')
    print(f'Best Model Testing Accuracy: {best_test_accuracy:.4f} (Epoch {best_epoch + 1})')
    print(f'Best Model Testing Old Accuracy: {best_test_old_accuracy:.4f} (Epoch {best_epoch + 1})')
    print(f'Best Model Testing New Accuracy: {best_test_new_accuracy:.4f} (Epoch {best_epoch + 1})')
