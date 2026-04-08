import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import h5py
import pandas as pd
from pathlib import Path

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
        e_m = F.softplus(o_m)
        alpha_m = e_m + 1.0
        S_m = torch.sum(alpha_m, dim=1, keepdim=True)
        u_m = self.K / S_m
        return alpha_m, u_m


class UncertaintyAwareFusion(nn.Module):
    def __init__(self, dim_t, dim_a, dim_v, num_classes, dropout=0.5, min_weight=0.01, temperature=0.1):
        super(UncertaintyAwareFusion, self).__init__()
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
        c_t, c_a, c_v = 1.0 - u_t, 1.0 - u_a, 1.0 - u_v
        c_concat = torch.cat([c_t, c_a, c_v], dim=1)
        weights = F.softmax(c_concat / self.temperature, dim=1)
        w_t, w_a, w_v = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        w_t = torch.clamp(w_t, self.min_weight, 1.0 - 2 * self.min_weight)
        w_a = torch.clamp(w_a, self.min_weight, 1.0 - 2 * self.min_weight)
        w_v = torch.clamp(w_v, self.min_weight, 1.0 - 2 * self.min_weight)
        w_sum = w_t + w_a + w_v
        w_t, w_a, w_v = w_t / w_sum, w_a / w_sum, w_v / w_sum

        proj_t = self.proj_t(h_t)
        proj_a = self.proj_a(h_a)
        proj_v = self.proj_v(h_v)
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
        second_term = torch.sum((alpha_tilde - 1) * (torch.digamma(alpha_tilde) -
                      torch.digamma(torch.sum(alpha_tilde, dim=1, keepdim=True))), dim=1, keepdim=True)
        l_kl = first_term + second_term
        lambda_t = min(1.0, epoch_num / self.annealing_step)
        return torch.mean(l_risk + lambda_t * l_kl)


# ==========================================
# 2. 整体网络架构
# ==========================================
class MultiModalEvidentialNetwork(nn.Module):
    def __init__(self, dim_t=768, dim_a=768, dim_v=768, num_classes=2, dropout=0.5, min_weight=0.01, temperature=0.5):
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
# 3. CMU-MOSI 数据集（二分类，丢弃 Neutral）
# ==========================================
BINARY_LABEL_MAP = {'Negative': 0, 'Positive': 1}

class MOSIDataset(Dataset):
    def __init__(self, h5_t_path, h5_a_path, h5_v_path, meta_path, split='train', augment=True):
        self.h5_t_path = h5_t_path
        self.h5_a_path = h5_a_path
        self.h5_v_path = h5_v_path
        self.split = split
        self.augment = augment and (split == 'train')
        self.label_map = BINARY_LABEL_MAP

        self.h5_t = h5py.File(h5_t_path, 'r')
        self.h5_a = h5py.File(h5_a_path, 'r')
        self.h5_v = h5py.File(h5_v_path, 'r')

        df = pd.read_csv(meta_path)
        df['mode'] = df['mode'].str.lower()
        df['annotation'] = df['annotation'].astype(str).str.strip().str.capitalize()

        # 丢弃 Neutral，只保留 Negative 和 Positive
        df = df[df['annotation'].isin(BINARY_LABEL_MAP.keys())].reset_index(drop=True)
        df_split = df[df['mode'] == split].reset_index(drop=True)

        valid_keys = []
        valid_labels = []
        missing_count = 0

        for _, row in df_split.iterrows():
            key = f"{row['video_id']}_{row['clip_id']}"
            if key in self.h5_t and key in self.h5_a and key in self.h5_v:
                valid_keys.append(key)
                valid_labels.append(self.label_map[row['annotation']])
            else:
                missing_count += 1

        self.sample_keys = valid_keys
        self.labels = np.array(valid_labels)
        self.num_samples = len(self.sample_keys)

        neg = np.sum(self.labels == 0)
        pos = np.sum(self.labels == 1)
        print(f"[{split.upper()}] MOSI Dataset Loaded! Valid: {self.num_samples} "
              f"(Skipped {missing_count} missing/neutral). "
              f"Negative: {neg}, Positive: {pos}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        key = self.sample_keys[idx]
        feat_t = torch.from_numpy(np.asarray(self.h5_t[key])).float()
        feat_a = torch.from_numpy(np.asarray(self.h5_a[key])).float()
        feat_v = torch.from_numpy(np.asarray(self.h5_v[key])).float()
        label = self.labels[idx]

        if self.augment:
            feat_t = feat_t + torch.randn_like(feat_t) * 0.02
            feat_a = feat_a + torch.randn_like(feat_a) * 0.02
            feat_v = feat_v + torch.randn_like(feat_v) * 0.02

        return feat_t, feat_a, feat_v, label


# ==========================================
# 4. 主训练循环
# ==========================================
def train():
    EPOCHS      = 50
    BATCH_SIZE  = 16
    LR          = 1e-4
    NUM_CLASSES = 2
    AUX_WEIGHT  = 0.3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    mosi_dir  = Path("mosifeature")
    h5_t_path = mosi_dir / "text_features_mosi.h5"
    h5_a_path = mosi_dir / "audio_features_mosi.h5"
    h5_v_path = mosi_dir / "video_features_mosi.h5"
    meta_path = mosi_dir / "label.csv"

    model = MultiModalEvidentialNetwork(
        dim_t=768,
        dim_a=768,
        dim_v=768,
        num_classes=NUM_CLASSES,
        dropout=0.45,
        min_weight=0.05,
        temperature=0.5
    ).to(DEVICE)

    train_dataset = MOSIDataset(h5_t_path, h5_a_path, h5_v_path, meta_path, split='train')
    valid_dataset = MOSIDataset(h5_t_path, h5_a_path, h5_v_path, meta_path, split='valid')

    # 类别均衡采样
    class_counts  = np.bincount(train_dataset.labels)
    sample_weights = 1.0 / class_counts[train_dataset.labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    criterion = BayesRiskEDLLoss(num_classes=NUM_CLASSES, annealing_step=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_valid_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        # --- 训练阶段 ---
        model.train()
        total_loss, correct_preds, total_samples = 0.0, 0, 0

        for batch_idx, (h_t, h_a, h_v, labels) in enumerate(train_loader):
            h_t, h_a, h_v, labels = h_t.to(DEVICE), h_a.to(DEVICE), h_v.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            alpha_final, p_final, weights, (alpha_t, alpha_a, alpha_v) = model(h_t, h_a, h_v)

            loss_fusion = criterion(alpha_final, labels, epoch_num=epoch)
            loss_aux = criterion(alpha_t, labels, epoch_num=epoch) + \
                       criterion(alpha_a, labels, epoch_num=epoch) + \
                       criterion(alpha_v, labels, epoch_num=epoch)
            loss = loss_fusion + AUX_WEIGHT * loss_aux

            loss.backward()
            optimizer.step()
            scheduler.step(epoch - 1 + batch_idx / len(train_loader))

            total_loss    += loss.item() * labels.size(0)
            preds          = torch.argmax(p_final, dim=1)
            correct_preds += (preds == labels).sum().item()
            total_samples += labels.size(0)

        epoch_loss = total_loss / total_samples
        epoch_acc  = correct_preds / total_samples * 100

        # --- 验证阶段 ---
        model.eval()
        valid_correct, valid_total = 0, 0
        val_w_t_sum, val_w_a_sum, val_w_v_sum = 0.0, 0.0, 0.0

        with torch.no_grad():
            for h_t, h_a, h_v, labels in valid_loader:
                h_t, h_a, h_v, labels = h_t.to(DEVICE), h_a.to(DEVICE), h_v.to(DEVICE), labels.to(DEVICE)
                alpha_final, p_final, weights, _ = model(h_t, h_a, h_v)

                preds          = torch.argmax(p_final, dim=1)
                valid_correct += (preds == labels).sum().item()
                valid_total   += labels.size(0)
                val_w_t_sum   += weights[0].mean().item()
                val_w_a_sum   += weights[1].mean().item()
                val_w_v_sum   += weights[2].mean().item()

        valid_acc = valid_correct / valid_total * 100

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), 'best_model_mosi.pth')
            print(f" -> New best model saved! Valid Acc: {best_valid_acc:.2f}%")

        n_val = len(valid_loader)
        print(f"Epoch [{epoch:02d}/{EPOCHS}] | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | "
              f"Valid Acc: {valid_acc:.2f}% | "
              f"Val Weights - T:{val_w_t_sum/n_val:.2f}, A:{val_w_a_sum/n_val:.2f}, V:{val_w_v_sum/n_val:.2f}")

    print(f"\nTraining completed! Best validation accuracy: {best_valid_acc:.2f}%")


if __name__ == "__main__":
    train()
