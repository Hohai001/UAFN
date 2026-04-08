import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import pandas as pd
from pathlib import Path

# ==========================================
# 1. 基础模块 (之前写好的组件)
# ==========================================
class EvidentialModule(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(EvidentialModule, self).__init__()
        self.K = num_classes
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, h_m):
        o_m = self.fc(h_m)
        e_m = F.softplus(o_m)
        alpha_m = e_m + 1.0
        S_m = torch.sum(alpha_m, dim=1, keepdim=True)
        u_m = self.K / S_m
        return alpha_m, u_m

class UncertaintyAwareFusion(nn.Module):
    def __init__(self, dim_t, dim_a, dim_v, num_classes):
        super(UncertaintyAwareFusion, self).__init__()
        self.fused_dim = dim_t + dim_a + dim_v
        self.global_classifier = nn.Linear(self.fused_dim, num_classes)
        self.K = num_classes

    def forward(self, h_t, h_a, h_v, u_t, u_a, u_v):
        c_t, c_a, c_v = 1.0 - u_t, 1.0 - u_a, 1.0 - u_v
        c_sum = c_t + c_a + c_v + 1e-8
        
        w_t, w_a, w_v = c_t / c_sum, c_a / c_sum, c_v / c_sum
        
        H_fused = torch.cat([w_t * h_t, w_a * h_a, w_v * h_v], dim=1)
        
        o_final = self.global_classifier(H_fused)
        alpha_final = F.softplus(o_final) + 1.0
        p_final = alpha_final / torch.sum(alpha_final, dim=1, keepdim=True)
        
        return alpha_final, p_final, (w_t, w_a, w_v)

class BayesRiskEDLLoss(nn.Module):
    def __init__(self, num_classes, annealing_step=10):
        super(BayesRiskEDLLoss, self).__init__()
        self.K = num_classes
        self.annealing_step = annealing_step

    def forward(self, alpha, y_indices, epoch_num):
        y = F.one_hot(y_indices, num_classes=self.K).float()
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        l_risk = torch.sum(y * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
        
        alpha_tilde = y + (1 - y) * alpha 
        first_term = torch.lgamma(torch.sum(alpha_tilde, dim=1, keepdim=True)) \
                     - torch.lgamma(torch.tensor(float(self.K)).to(alpha.device)) \
                     - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True)
        second_term = torch.sum((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - \
                      torch.digamma(torch.sum(alpha_tilde, dim=1, keepdim=True))), dim=1, keepdim=True)
        l_kl = first_term + second_term
        
        lambda_t = min(1.0, epoch_num / self.annealing_step)
        return torch.mean(l_risk + lambda_t * l_kl)

# ==========================================
# 2. 整体网络架构 (MultiModal Evidential Network)
# ==========================================
class MultiModalEvidentialNetwork(nn.Module):
    def __init__(self, dim_t=768, dim_a=768, dim_v=512, num_classes=5):
        super(MultiModalEvidentialNetwork, self).__init__()
        # 三个模态独立的证据估计模块
        self.em_t = EvidentialModule(dim_t, num_classes)
        self.em_a = EvidentialModule(dim_a, num_classes)
        self.em_v = EvidentialModule(dim_v, num_classes)
        
        # 动态自适应融合模块
        self.fusion = UncertaintyAwareFusion(dim_t, dim_a, dim_v, num_classes)

    def forward(self, h_t, h_a, h_v):
        # 1. 估计各个模态的不确定性
        alpha_t, u_t = self.em_t(h_t)
        alpha_a, u_a = self.em_a(h_a)
        alpha_v, u_v = self.em_v(h_v)
        
        # 2. 传入融合模块
        alpha_final, p_final, weights = self.fusion(h_t, h_a, h_v, u_t, u_a, u_v)
        
        return alpha_final, p_final, weights

# ==========================================
# 3. 数据集读取逻辑 (这里用 Dummy 数据演示，方便你直接跑通)
# ==========================================
class CHSIMSDataset(Dataset):
    def __init__(self, h5_t_path, h5_a_path, h5_v_path, meta_path, split='train'):
        """
        Args:
            h5_t_path: 文本特征h5文件路径
            h5_a_path: 音频特征h5文件路径
            h5_v_path: 视频特征h5文件路径
            meta_path: 元数据csv文件路径
            split: 'train', 'valid', or 'test'
        """
        self.h5_t_path = h5_t_path
        self.h5_a_path = h5_a_path
        self.h5_v_path = h5_v_path
        
        self.h5_t = h5py.File(h5_t_path, 'r')
        self.h5_a = h5py.File(h5_a_path, 'r')
        self.h5_v = h5py.File(h5_v_path, 'r')
        
        df = pd.read_csv(meta_path)
        df['mode'] = df['mode'].str.lower()
        df = df[df['mode'] == split].reset_index(drop=True)
        
        self.sample_keys = [f"{vid.replace('/', '_')}_{i}" for i, vid in enumerate(df['video_id'].astype(str))]
        self.labels = df['annotation'].values
        self.num_samples = len(self.sample_keys)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        key = self.sample_keys[idx]
        
        feat_t = torch.from_numpy(np.asarray(self.h5_t[key])).float()
        feat_a = torch.from_numpy(np.asarray(self.h5_a[key])).float()
        feat_v = torch.from_numpy(np.asarray(self.h5_v[key])).float()
        label = int(self.labels[idx])
        
        return feat_t, feat_a, feat_v, label

# ==========================================
# 4. 主训练循环 (Training Loop)
# ==========================================
def train():
    # 超参数设置
    EPOCHS = 20
    BATCH_SIZE = 16
    LR = 1e-4
    NUM_CLASSES = 3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 特征文件路径
    feature_dir = Path("feature")
    h5_t_path = feature_dir / "text_features.h5"
    h5_a_path = feature_dir / "audio_features.h5"
    h5_v_path = feature_dir / "video_features.h5"
    meta_path = "/mnt/f/dataset/CH-SIMS v2.0/ch-simsv2s/meta.csv"

    # 初始化模型、数据、损失函数、优化器
    model = MultiModalEvidentialNetwork(num_classes=NUM_CLASSES).to(DEVICE)
    train_dataset = CHSIMSDataset(h5_t_path, h5_a_path, h5_v_path, meta_path, split='train')
    valid_dataset = CHSIMSDataset(h5_t_path, h5_a_path, h5_v_path, meta_path, split='valid')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}")
    
    criterion = BayesRiskEDLLoss(num_classes=NUM_CLASSES, annealing_step=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 开始训练
    best_valid_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        # 训练阶段
        model.train()
        total_loss = 0.0
        correct_preds = 0
        total_samples = 0

        for batch_idx, (h_t, h_a, h_v, labels) in enumerate(train_loader):
            # 将数据推送到 GPU/CPU
            h_t, h_a, h_v, labels = h_t.to(DEVICE), h_a.to(DEVICE), h_v.to(DEVICE), labels.to(DEVICE)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            alpha_final, p_final, weights = model(h_t, h_a, h_v)

            # 计算贝叶斯风险损失
            loss = criterion(alpha_final, labels, epoch_num=epoch)

            # 反向传播 & 更新权重
            loss.backward()
            optimizer.step()

            # 统计指标
            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(p_final, dim=1)
            correct_preds += (preds == labels).sum().item()
            total_samples += labels.size(0)

        # 打印 Epoch 级别的训练日志
        epoch_loss = total_loss / total_samples
        epoch_acc = correct_preds / total_samples * 100
        
        # 验证阶段
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        
        with torch.no_grad():
            for h_t, h_a, h_v, labels in valid_loader:
                h_t, h_a, h_v, labels = h_t.to(DEVICE), h_a.to(DEVICE), h_v.to(DEVICE), labels.to(DEVICE)
                alpha_final, p_final, _ = model(h_t, h_a, h_v)
                loss = criterion(alpha_final, labels, epoch_num=epoch)
                
                valid_loss += loss.item() * labels.size(0)
                preds = torch.argmax(p_final, dim=1)
                valid_correct += (preds == labels).sum().item()
                valid_total += labels.size(0)
        
        valid_epoch_loss = valid_loss / valid_total
        valid_epoch_acc = valid_correct / valid_total * 100
        
        # 保存最佳模型
        if valid_epoch_acc > best_valid_acc:
            best_valid_acc = valid_epoch_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        # 打印第一批数据的权重情况 (直观展示动态融合效果)
        w_t, w_a, w_v = weights
        print(f"Epoch [{epoch}/{EPOCHS}] | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | "
              f"Valid Loss: {valid_epoch_loss:.4f} | Valid Acc: {valid_epoch_acc:.2f}% | "
              f"Mean Weights - T:{w_t.mean().item():.2f}, A:{w_a.mean().item():.2f}, V:{w_v.mean().item():.2f}")
    
    print(f"\nTraining completed! Best validation accuracy: {best_valid_acc:.2f}%")

if __name__ == "__main__":
    train()