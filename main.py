import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import h5py
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score

# ==========================================
# 1. 基础组件模块 (Evidential & Fusion)
# ==========================================
class EvidentialModule(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.5, hidden_dim=128):
        super(EvidentialModule, self).__init__()
        self.K = num_classes

        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, h_m):
        o_m = self.encoder(h_m)

        # 计算 EDL 证据和不确定性
        e_m = F.softplus(o_m)
        alpha_m = e_m + 1.0
        S_m = torch.sum(alpha_m, dim=1, keepdim=True)
        u_m = self.K / S_m

        return alpha_m, u_m

class UncertaintyAwareFusion(nn.Module):
    def __init__(self, dim_t, dim_a, dim_v, num_classes, dropout=0.5, min_weight=0.01, temperature=0.1):
        super(UncertaintyAwareFusion, self).__init__()
        
        # ====== 核心瘦身：模态独立降维层 (Bottleneck) ======
        # 在拼接前，强制把 768 维的高维噪音挤压掉，只保留最核心的 64 维情感语义
        self.proj_t = nn.Sequential(nn.Linear(dim_t, 128), nn.ReLU(), nn.Dropout(dropout))
        self.proj_a = nn.Sequential(nn.Linear(dim_a, 128), nn.ReLU(), nn.Dropout(dropout))
        self.proj_v = nn.Sequential(nn.Linear(dim_v, 128), nn.ReLU(), nn.Dropout(dropout))

        self.fused_dim = 128 * 3
        self.dropout = nn.Dropout(dropout)

        self.global_classifier = nn.Sequential(
            nn.Linear(self.fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        self.K = num_classes
        self.min_weight = min_weight
        self.temperature = temperature

    def forward(self, h_t, h_a, h_v, u_t, u_a, u_v):
        # 1. 权重计算 (保持 Softmax 锐化不变)
        c_t, c_a, c_v = 1.0 - u_t, 1.0 - u_a, 1.0 - u_v
        c_concat = torch.cat([c_t, c_a, c_v], dim=1)
        weights = F.softmax(c_concat / self.temperature, dim=1)
        
        w_t, w_a, w_v = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        
        w_t = torch.clamp(w_t, self.min_weight, 1.0 - 2 * self.min_weight)
        w_a = torch.clamp(w_a, self.min_weight, 1.0 - 2 * self.min_weight)
        w_v = torch.clamp(w_v, self.min_weight, 1.0 - 2 * self.min_weight)
        
        w_sum = w_t + w_a + w_v
        w_t, w_a, w_v = w_t / w_sum, w_a / w_sum, w_v / w_sum
        
        # ====== 2. 先降维，再加权拼接 ======
        proj_t = self.proj_t(h_t)
        proj_a = self.proj_a(h_a)
        proj_v = self.proj_v(h_v)
        
        # 现在 H_fused 为 384 维，兼顾表达能力与过拟合风险
        H_fused = torch.cat([w_t * proj_t, w_a * proj_a, w_v * proj_v], dim=1)
        H_fused = self.dropout(H_fused)
        
        o_final = self.global_classifier(H_fused)
        alpha_final = F.softplus(o_final) + 1.0
        p_final = alpha_final / torch.sum(alpha_final, dim=1, keepdim=True)
        
        return alpha_final, p_final, (w_t, w_a, w_v)
class BayesRiskEDLLoss(nn.Module):
    def __init__(self, num_classes, annealing_step=20):
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
    def __init__(self, dim_t=768, dim_a=768, dim_v=768, num_classes=3, dropout=0.5, min_weight=0.01, temperature=0.5):
        super(MultiModalEvidentialNetwork, self).__init__()
        self.em_t = EvidentialModule(dim_t, num_classes, dropout)
        self.em_a = EvidentialModule(dim_a, num_classes, dropout)
        self.em_v = EvidentialModule(dim_v, num_classes, dropout)
        
        self.fusion = UncertaintyAwareFusion(dim_t, dim_a, dim_v, num_classes, dropout, min_weight, temperature)

    def forward(self, h_t, h_a, h_v):
        alpha_t, u_t = self.em_t(h_t)
        alpha_a, u_a = self.em_a(h_a)
        alpha_v, u_v = self.em_v(h_v)
        
        alpha_final, p_final, weights = self.fusion(h_t, h_a, h_v, u_t, u_a, u_v)
        return alpha_final, p_final, weights, (alpha_t, alpha_a, alpha_v)

# ==========================================
# 3. 数据集读取逻辑 (绝对干净版，防崩溃，无重复方法)
# ==========================================
class CHSIMSDataset(Dataset):
    def __init__(self, h5_t_path, h5_a_path, h5_v_path, meta_path, split='train', augment=True, label_map=None):
        self.h5_t_path = h5_t_path
        self.h5_a_path = h5_a_path
        self.h5_v_path = h5_v_path
        self.split = split
        self.augment = augment and (split == 'train')

        self.h5_t = h5py.File(h5_t_path, 'r')
        self.h5_a = h5py.File(h5_a_path, 'r')
        self.h5_v = h5py.File(h5_v_path, 'r')

        df = pd.read_csv(meta_path)
        df['mode'] = df['mode'].str.lower()
        # 清洗标签
        df['annotation'] = df['annotation'].astype(str).str.strip().str.capitalize()
        df['original_idx'] = range(len(df))

        # 修复：label_map 必须基于全量数据构建，保证 train/valid/test 标签一致
        if label_map is not None:
            self.label_map = label_map
        else:
            unique_labels = sorted(df['annotation'].unique())  # 全量数据
            self.label_map = {label: idx for idx, label in enumerate(unique_labels)}

        df_split = df[df['mode'] == split].reset_index(drop=True)
        
        valid_keys = []
        valid_labels = []
        missing_count = 0
        
        for vid, idx, ann in zip(df_split['video_id'].astype(str), df_split['original_idx'], df_split['annotation']):
            key = f"{vid.replace('/', '_')}_{idx}"
            # 严格对齐检查
            if key in self.h5_t and key in self.h5_a and key in self.h5_v:
                valid_keys.append(key)
                valid_labels.append(self.label_map[ann])
            else:
                missing_count += 1
                
        self.sample_keys = valid_keys
        self.labels = np.array(valid_labels)
        self.num_samples = len(self.sample_keys)
        
        print(f"[{split.upper()}] Dataset Loaded! Valid samples: {self.num_samples} (Skipped {missing_count} missing samples). Label map: {self.label_map}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        key = self.sample_keys[idx]
        
        feat_t = torch.from_numpy(np.asarray(self.h5_t[key])).float()
        feat_a = torch.from_numpy(np.asarray(self.h5_a[key])).float()
        feat_v = torch.from_numpy(np.asarray(self.h5_v[key])).float()
        label = self.labels[idx]
        
        if self.augment:
            feat_t = self._augment_feature(feat_t)
            feat_a = self._augment_feature(feat_a)
            feat_v = self._augment_feature(feat_v)
            
        return feat_t, feat_a, feat_v, label
    
    def _augment_feature(self, feat, noise_level=0.02):
        noise = torch.randn_like(feat) * noise_level
        return feat + noise

# ==========================================
# 4. 主训练循环 (Training Loop)
# ==========================================
def train():
    # 超参数设置
    EPOCHS = 50
    BATCH_SIZE = 16
    LR = 1e-4
    NUM_CLASSES = 3
    AUX_WEIGHT = 0.3  # 辅助损失权重
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {DEVICE}")

    # 特征文件路径
    feature_dir = Path("feature")
    h5_t_path = feature_dir / "text_features_v2_macbert_large.h5"
    h5_a_path = feature_dir / "audio_features_v4_last4layers.h5"
    h5_v_path = feature_dir / "video_features_v3.h5"
    meta_path = "/mnt/f/dataset/CH-SIMS v2.0/ch-simsv2s/meta.csv"

    # ==========================================
    # 🌟 终极参数设置区域
    # ==========================================
    # 注意：如果你的 video_features_v3.h5 维度是 512，请务必把 dim_v 改为 512
    model = MultiModalEvidentialNetwork(
        dim_t=1024,     # macbert-large 输出 1024 维
        dim_a=768,
        dim_v=768,
        num_classes=NUM_CLASSES,
        dropout=0.45,       # 适中 dropout：既防过拟合，又保留足够学习能力
        min_weight=0.05,
        temperature=0.5
    ).to(DEVICE)
    
    # 先构建全局 label_map，保证 train/valid 标签索引完全一致
    _df = pd.read_csv(meta_path)
    _df['annotation'] = _df['annotation'].astype(str).str.strip().str.capitalize()
    global_label_map = {label: idx for idx, label in enumerate(sorted(_df['annotation'].unique()))}
    print(f"Global label map: {global_label_map}")

    train_dataset = CHSIMSDataset(h5_t_path, h5_a_path, h5_v_path, meta_path, split='train', label_map=global_label_map)
    valid_dataset = CHSIMSDataset(h5_t_path, h5_a_path, h5_v_path, meta_path, split='valid', label_map=global_label_map)
    
    # 类别均衡采样：解决数据不均衡问题
    class_counts = np.bincount(train_dataset.labels)
    sample_weights = 1.0 / class_counts[train_dataset.labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    criterion = BayesRiskEDLLoss(num_classes=NUM_CLASSES, annealing_step=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_valid_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        # --- 训练阶段 ---
        model.train()
        total_loss = 0.0
        correct_preds = 0
        total_samples = 0
        running_loss_fusion = 0.0
        running_loss_aux = 0.0

        for batch_idx, (h_t, h_a, h_v, labels) in enumerate(train_loader):
            h_t, h_a, h_v, labels = h_t.to(DEVICE), h_a.to(DEVICE), h_v.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            alpha_final, p_final, weights, (alpha_t, alpha_a, alpha_v) = model(h_t, h_a, h_v)

            loss_fusion = criterion(alpha_final, labels, epoch_num=epoch)
            loss_t = criterion(alpha_t, labels, epoch_num=epoch)
            loss_a = criterion(alpha_a, labels, epoch_num=epoch)
            loss_v = criterion(alpha_v, labels, epoch_num=epoch)

            loss_aux = loss_t + loss_a + loss_v
            loss = loss_fusion + AUX_WEIGHT * loss_aux

            loss.backward()
            optimizer.step()
            # CosineAnnealingWarmRestarts 需要每个 batch 更新才能正确触发热重启
            scheduler.step(epoch - 1 + batch_idx / len(train_loader))

            total_loss += loss.item() * labels.size(0)
            running_loss_fusion += loss_fusion.item() * labels.size(0)
            running_loss_aux += loss_aux.item() * labels.size(0)
            
            preds = torch.argmax(p_final, dim=1)
            correct_preds += (preds == labels).sum().item()
            total_samples += labels.size(0)

        epoch_loss = total_loss / total_samples
        epoch_acc = correct_preds / total_samples * 100
        
        # --- 验证阶段 ---
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        
        val_w_t_sum, val_w_a_sum, val_w_v_sum = 0.0, 0.0, 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for h_t, h_a, h_v, labels in valid_loader:
                h_t, h_a, h_v, labels = h_t.to(DEVICE), h_a.to(DEVICE), h_v.to(DEVICE), labels.to(DEVICE)

                alpha_final, p_final, weights, _ = model(h_t, h_a, h_v)
                loss = criterion(alpha_final, labels, epoch_num=epoch)

                valid_loss += loss.item() * labels.size(0)
                preds = torch.argmax(p_final, dim=1)
                valid_correct += (preds == labels).sum().item()
                valid_total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                val_w_t_sum += weights[0].mean().item()
                val_w_a_sum += weights[1].mean().item()
                val_w_v_sum += weights[2].mean().item()

        valid_epoch_loss = valid_loss / valid_total
        valid_epoch_acc = valid_correct / valid_total * 100
        valid_f1 = f1_score(all_labels, all_preds, average='macro')

        if valid_epoch_acc > best_valid_acc:
            best_valid_acc = valid_epoch_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f" 🌟 -> New best model saved! Valid Acc: {best_valid_acc:.2f}%")

        num_val_batches = len(valid_loader)
        print(f"Epoch [{epoch:02d}/{EPOCHS}] | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | "
              f"Valid Acc: {valid_epoch_acc:.2f}% | Valid F1: {valid_f1:.4f} | "
              f"Val Weights - T:{val_w_t_sum/num_val_batches:.2f}, A:{val_w_a_sum/num_val_batches:.2f}, V:{val_w_v_sum/num_val_batches:.2f}")

    print(f"\n🎉 Training completed! Best validation accuracy: {best_valid_acc:.2f}%")

if __name__ == "__main__":
    train()