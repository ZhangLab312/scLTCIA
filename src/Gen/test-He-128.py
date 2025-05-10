import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
import os
import torchmetrics
import scanpy as sc
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def set_global_seed(seed: int) -> None:
    """
    Sets the global random seed for reproducibility across multiple libraries.

    Args:
        seed (int): The seed value to set for random number generators.
    """
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy's random generator
    torch.manual_seed(seed)  # PyTorch's CPU random seed
    torch.cuda.manual_seed(seed)  # PyTorch's GPU random seed (if CUDA is available)
    torch.cuda.manual_seed_all(seed)  # Seed all GPUs (if using multi-GPU)

    # For deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Ensures reproducibility for os-related randomness (if any)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_global_seed(20020130)

# 配置参数
config = {
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'image_size': (128, 128),
    'batch_size': 256,
    'noise_dim': 16,
    'diffusion_steps': 1000,
    'learning_rate': 1e-3,
    'num_epochs': 100,
    'validation_interval': 5,
    'selected_indices': [1, 244, 550, 655, 999],
    'early_stopping_patience': 10,
    'min_delta': 1e-4
}


# Function to reshape (cells, 1024) to (cells, 32, 32)
def reshape_to_image(hvg_matrix):
    if hvg_matrix.shape[1] != 16384:
        raise ValueError("Input matrix must have 16384 features to reshape into 128x128.")
    return hvg_matrix.reshape(-1, 128, 128)


# Function to reshape (cells, 32, 32) back to (cells, 1024)
def reshape_to_vector(image_matrix):
    if image_matrix.shape[1:] != (128, 128):
        raise ValueError("Input matrix must have shape (cells, 128, 128) to reshape back to 16384 features.")
    return image_matrix.reshape(-1, 16384)


# 读入后顺便数据预处理
def preprocess_h5ad(adata, target_genes=16384):
    # Step 1: Load the data from the h5ad file
    # adata = sc.read_h5ad(file_path)

    # Step 2: Perform basic preprocessing (optional)
    sc.pp.filter_cells(adata, min_genes=200)  # filter out cells with fewer than 200 genes
    sc.pp.filter_genes(adata, min_cells=3)  # filter out genes that are detected in fewer than 3 cells

    # Step 3: Normalize the data (optional, depending on your downstream task)
    sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize to counts per cell
    sc.pp.log1p(adata)  # Log-transform the data

    # Step 4: Store the raw data for later use in HVG selection
    adata.raw = adata  # Store the unnormalized counts (before normalization) in the raw attribute

    # Step 5: High variance gene selection (HVGs)
    sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=None, subset=True)

    # Step 6: Ensure we have 1024 genes, select the top ones if necessary
    if adata.raw.X.shape[1] > target_genes:
        # If more than 1024 HVGs, select top 1024
        adata = adata[:, adata.var['highly_variable']].copy()
        adata = adata[:, adata.var['highly_variable']].copy()  # Keep the AnnData structure intact
        adata = adata[:, adata.var['highly_variable'].head(target_genes).index]  # Select top 1024 genes
    else:
        # If there are fewer than 1024 HVGs, pad with other genes (optional)
        adata = adata[:, adata.var['highly_variable']].copy()
        extra_genes = adata.var[~adata.var['highly_variable']].head(target_genes - adata.raw.X.shape[1])
        adata = adata[:, list(adata.var['highly_variable'].index) + list(extra_genes.index)]

    # Step 7: Print the shape of the preprocessed data
    print(f"Preprocessed data shape: {adata.shape}")

    # Step 8: Return the preprocessed AnnData object (keep it in AnnData format)
    return adata


# 数据加载与处理
class SingleCellDataset(Dataset):
    def __init__(self, data_path, labels_path):
        # 读取数据
        adata = sc.read(data_path)  # 读取 h5ad 格式的单细胞数据

        # adata = preprocess_h5ad(adata, target_genes=4096)

        # 检查是否是稀疏矩阵，如果是则转为numpy数组
        if isinstance(adata.X, np.ndarray):
            self.data = adata.X[:10000]  # 如果已经是 numpy 数组，直接使用
        else:
            self.data = adata.X[:10000].toarray()  # 如果是稀疏矩阵，转换为 numpy 数组

        # 获取标签，并转换为 numpy 数组
        self.labels = adata.obs[labels_path][:10000].values

        # 如果需要转换成特定形状（如图像），可以在这里进行
        self.data = reshape_to_image(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0), self.labels[idx]


# 加载所有类型的单细胞数据
def get_single_cell_loader(data_path, labels_path):
    dataset = SingleCellDataset(data_path, labels_path)
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)


# 替换为加载所有类别的数据
data_loader = get_single_cell_loader(
    data_path='../../data/scrna_data/Inter/He_Long_Bone_16384.h5ad',
    labels_path='cell_type1'
)


# U-Net 实现
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU()
            )

        def upconv_block(in_ch, out_ch):
            return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(256, 512)

        self.upconv3 = upconv_block(512, 256)
        self.decoder3 = conv_block(512, 256)

        self.upconv2 = upconv_block(256, 128)
        self.decoder2 = conv_block(256, 128)

        self.upconv1 = upconv_block(128, 64)
        self.decoder1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))

        bottleneck = self.bottleneck(self.pool(enc3))

        dec3 = self.decoder3(torch.cat((self.upconv3(bottleneck), enc3), dim=1))
        dec2 = self.decoder2(torch.cat((self.upconv2(dec3), enc2), dim=1))
        dec1 = self.decoder1(torch.cat((self.upconv1(dec2), enc1), dim=1))

        return self.final(dec1)


# 噪声生成器
class NoiseGenerator(nn.Module):
    def __init__(self, input_dim, image_size, time_embed_dim=128):
        """
        input_dim: 噪声的随机输入维度
        image_size: 图像尺寸 (H, W)
        time_embed_dim: 时间嵌入维度
        """
        super(NoiseGenerator, self).__init__()
        self.image_size = image_size
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),  # 时间步 t 的嵌入
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.ReLU(),
        )
        self.fc = nn.Linear(input_dim, time_embed_dim)  # 初始噪声的全连接映射
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),  # 时间步 & 噪声通道融合
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),  # 输出为单通道噪声
        )

    def forward(self, z, t):
        """
        z: 随机输入 (batch_size, input_dim)
        t: 时间步 (batch_size, 1)
        """
        batch_size = z.size(0)
        # 嵌入时间步
        time_embedding = self.time_embed(t).view(batch_size, -1, 1, 1)  # (batch, time_embed_dim, 1, 1)

        # 初始噪声生成
        z_mapped = self.fc(z).view(batch_size, -1, self.image_size[0], self.image_size[1])  # (batch, embed_dim, H, W)

        # 融合时间步和噪声
        x = torch.cat((z_mapped, time_embedding.expand_as(z_mapped)), dim=1)  # (batch, 2, H, W)

        # 卷积生成最终噪声
        return self.conv(x)


# 时间嵌入
class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super(TimeEmbedding, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, t):
        """
        使用正弦和余弦函数进行时间嵌入
        t: 时间步 (batch_size, 1)
        """
        half_dim = self.embed_dim // 2
        emb = torch.log(10000) / (half_dim - 1)  # 控制频率
        emb = torch.exp(emb * torch.arange(half_dim, dtype=torch.float32, device=t.device))  # 频率
        emb = t * emb  # 时间步与频率的乘积

        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # 拼接正弦和余弦部分

        return emb


# 扩散模型
class DiffusionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_embed_dim=128):
        super(DiffusionUNet, self).__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, in_channels * 128 * 128),  # 生成与输入图像匹配的嵌入形状
        )
        self.unet = UNet(in_channels=2, out_channels=out_channels)  # 输入通道数 + 时间通道数

    def forward(self, x, t):
        batch_size, _, h, w = x.size()

        # 时间嵌入
        time_embedding = self.time_embed(t).view(batch_size, 1, h, w)  # 形状调整为 (batch, 1, H, W)

        # 拼接时间嵌入与输入
        x_with_time = torch.cat((x, time_embedding), dim=1)  # 拼接通道维度

        # 通过 U-Net 处理
        return self.unet(x_with_time)


class Text2Vec(nn.Module):
    def __init__(self, max_batches=1000, embedding_dim=64, target_size=(128, 128)):
        """
        max_batches: 批次编号的最大值，用于确定嵌入字典大小。
        embedding_dim: 嵌入空间的维度。
        target_size: 输出嵌入的目标尺寸 (H, W)。
        """
        super(Text2Vec, self).__init__()
        self.embedding = nn.Embedding(max_batches, embedding_dim)  # 嵌入层
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, target_size[0] * target_size[1]),
            nn.ReLU()
        )
        self.target_size = target_size

    def forward(self, batch_indices):
        """
        batch_indices: 批次编号，形状为 (batch_size,) 的一维张量。
        """
        # 通过嵌入层
        embeddings = self.embedding(batch_indices)  # 形状 (batch_size, embedding_dim)

        # 通过全连接层调整到目标尺寸
        flat_embeddings = self.fc(embeddings)  # 形状 (batch_size, target_size[0] * target_size[1])

        # 调整形状为 (batch_size, target_size[0], target_size[1])
        embeddings_reshaped = flat_embeddings.view(-1, self.target_size[0], self.target_size[1])

        return embeddings_reshaped


# 实例化模型
# noise_gen = NoiseGenerator(config['noise_dim'], config['image_size'][0] * config['image_size'][1]).to(config['device'])
noise_gen = NoiseGenerator(input_dim=config['noise_dim'], image_size=config['image_size']).to(config['device'])
diffusion_model = DiffusionUNet().to(config['device'])  # 替换为 DiffusionUNet

# 定义 text2vec 模型
text2vec = Text2Vec(max_batches=len(data_loader), embedding_dim=64, target_size=(128, 128)).to(config['device'])

# 训练逻辑和可视化保持不变，直接复用原有代码
criterion = nn.MSELoss()
optimizer = optim.Adam(list(diffusion_model.parameters()) + list(noise_gen.parameters()), lr=config['learning_rate'])


def compute_metrics(pred_images, target_images):
    """
    计算 MAE、PSNR 和 SSIM
    """
    pred_images_np = pred_images.detach().cpu().numpy()
    target_images_np = target_images.detach().cpu().numpy()

    mae = torchmetrics.functional.mean_absolute_error(
        pred_images.view(-1), target_images.view(-1)
    ).item()

    psnr = 0
    ssim = 0
    for pred, target in zip(pred_images_np, target_images_np):
        pred = pred.squeeze()
        target = target.squeeze()

        # 计算每对图像的最大值和最小值
        data_range = target.max() - target.min()

        # 使用动态计算的 data_range 计算 PSNR 和 SSIM
        psnr += compare_psnr(target, pred, data_range=data_range)
        ssim += compare_ssim(target, pred, data_range=data_range)

    psnr /= len(pred_images_np)
    ssim /= len(pred_images_np)

    return mae, psnr, ssim


def train_diffusion_model_with_metrics():
    best_loss = float('inf')
    patience_counter = 0
    best_model_path = "./pth/checkpoint.pth"  # 保存最佳模型的路径
    save_dir = './gen/16384/'

    for epoch in range(config['num_epochs']):
        diffusion_model.train()
        noise_gen.train()

        epoch_loss = 0.0
        for batch_idx, (images, _) in enumerate(tqdm(data_loader)):
            images = images.to(config['device'])

            batch_size = images.size(0)
            timesteps = torch.randint(0, config['diffusion_steps'], (batch_size, 1), device=config['device']).float()
            timesteps = timesteps.view(batch_size, 1, 1, 1)

            # 动态生成嵌入
            batch_indices = torch.tensor([batch_idx] * images.size(0), device=config['device'])
            embeddings = text2vec(batch_indices).unsqueeze(1)  # 调整为 (B, 1, H, W)

            noise = torch.randn_like(embeddings)
            noisy_images = images + noise * (0.1 * timesteps / config['diffusion_steps'])

            pred_images = diffusion_model(noisy_images, timesteps)

            loss = criterion(pred_images, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(data_loader)
        print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Avg Loss: {avg_loss:.4f}")

        # 计算并打印指标
        diffusion_model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for images, _ in data_loader:
                images = images.to(config['device'])
                timesteps = torch.zeros((images.size(0), 1, 1, 1), device=config['device'])
                pred_images = diffusion_model(images, timesteps)
                all_preds.append(pred_images)
                all_targets.append(images)

            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

        mae, psnr, ssim = compute_metrics(all_preds, all_targets)
        print(f"Metrics at Epoch {epoch + 1}: MAE: {mae:.4f}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")

        # 检查是否为最佳模型
        if avg_loss < best_loss - config['min_delta']:
            best_loss = avg_loss
            patience_counter = 0

            # 保存最佳模型
            torch.save({
                'model_state_dict': diffusion_model.state_dict(),
                'text2vec_state_dict': text2vec.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_loss': best_loss
            }, best_model_path)

            '''
            如果需要加载权重请用下面代码
            checkpoint = torch.load('checkpoint.pth')
            diffusion_model.load_state_dict(checkpoint['model_state_dict'])
            text2vec.load_state_dict(checkpoint['text2vec_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['best_loss']
            '''

            print(f"Best model saved at epoch {epoch + 1} with loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print("Early stopping triggered.")
                break

        if (epoch + 1) % config['validation_interval'] == 0:
            visualize_results(epoch, data_loader, diffusion_model, noise_gen, save_dir)


def visualize_results(epoch, data_loader, diffusion_model, noise_gen, save_dir="results"):
    diffusion_model.eval()
    noise_gen.eval()

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    images, labels = next(iter(data_loader))
    images = images.to(config['device'])

    batch_size = images.size(0)
    num_indices = min(len(config['selected_indices']), batch_size)
    selected_indices = np.random.choice(batch_size, num_indices, replace=False)

    fig, axes = plt.subplots(2, num_indices, figsize=(15, 6))

    for i, idx in enumerate(selected_indices):
        real_image = images[idx].cpu().squeeze().numpy()  # 转换为 numpy 数组
        real_label = labels[idx]  # 直接使用标签，不需要 .item() 方法

        t = torch.tensor([config['diffusion_steps']], dtype=torch.float32, device=config['device']).view(1, 1, 1, 1)
        noise = torch.randn_like(images[idx:idx + 1])
        noisy_image = images[idx:idx + 1] + noise * (0.1 * t / config['diffusion_steps'])

        pred_image = diffusion_model(noisy_image, t.view(1, 1)).detach().cpu().squeeze().numpy()  # 转换为 numpy 数组

        # 保存真实图像和生成图像的数值为 .npy 文件
        real_image_filename = os.path.join(save_dir, f"epoch_{epoch + 1}_real_image_{i}.npy")
        pred_image_filename = os.path.join(save_dir, f"epoch_{epoch + 1}_pred_image_{i}.npy")

        np.save(real_image_filename, real_image)
        np.save(pred_image_filename, pred_image)

        axes[0, i].imshow(real_image, cmap="gray")
        axes[0, i].set_title(f"Real (Label {real_label})")
        axes[0, i].axis("off")

        axes[1, i].imshow(pred_image, cmap="gray")
        axes[1, i].set_title(f"Generated (Label {real_label})")
        axes[1, i].axis("off")

    plt.suptitle(f"Validation Results at Epoch {epoch + 1}", fontsize=16)
    plt.tight_layout()
    plt.show()


train_diffusion_model_with_metrics()
