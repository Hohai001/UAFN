import argparse
import random
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ablation_config import ABLATION_CHOICES, TEXT_BACKBONE_CHOICES, exp_name_from_ablation, resolve_ablation_config


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
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, h_m):
        o_m = self.encoder(h_m)
        e_m = F.softplus(o_m)
        alpha_m = e_m + 1.0
        s_m = torch.sum(alpha_m, dim=1, keepdim=True)
        u_m = self.K / s_m
        return alpha_m, u_m


class UncertaintyAwareFusion(nn.Module):
    def __init__(
        self,
        dim_t,
        dim_a,
        dim_v,
        num_classes,
        dropout=0.5,
        min_weight=0.01,
        temperature=0.1,
        use_gating=True,
        use_edl=True,
    ):
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
            nn.Linear(128, num_classes),
        )

        self.min_weight = min_weight
        self.temperature = temperature
        self.use_gating = use_gating
        self.use_edl = use_edl

    def forward(self, h_t, h_a, h_v, u_t=None, u_a=None, u_v=None):
        batch_size = h_t.size(0)
        if self.use_gating and u_t is not None and u_a is not None and u_v is not None:
            c_t, c_a, c_v = 1.0 - u_t, 1.0 - u_a, 1.0 - u_v
            c_concat = torch.cat([c_t, c_a, c_v], dim=1)
            weights = F.softmax(c_concat / self.temperature, dim=1)

            w_t, w_a, w_v = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
            w_t = torch.clamp(w_t, self.min_weight, 1.0 - 2 * self.min_weight)
            w_a = torch.clamp(w_a, self.min_weight, 1.0 - 2 * self.min_weight)
            w_v = torch.clamp(w_v, self.min_weight, 1.0 - 2 * self.min_weight)
            w_sum = w_t + w_a + w_v
            w_t, w_a, w_v = w_t / w_sum, w_a / w_sum, w_v / w_sum
        else:
            # `wo_gating` / `wo_edl` 分支：固定等权融合
            uniform = h_t.new_full((batch_size, 1), 1.0 / 3.0)
            w_t, w_a, w_v = uniform, uniform, uniform

        proj_t = self.proj_t(h_t)
        proj_a = self.proj_a(h_a)
        proj_v = self.proj_v(h_v)

        h_fused = torch.cat([w_t * proj_t, w_a * proj_a, w_v * proj_v], dim=1)
        h_fused = self.dropout(h_fused)
        logits_final = self.global_classifier(h_fused)

        if self.use_edl:
            alpha_final = F.softplus(logits_final) + 1.0
            probs_final = alpha_final / torch.sum(alpha_final, dim=1, keepdim=True)
        else:
            alpha_final = None
            probs_final = F.softmax(logits_final, dim=1)

        return alpha_final, logits_final, probs_final, (w_t, w_a, w_v)


class BayesRiskEDLLoss(nn.Module):
    def __init__(self, num_classes, annealing_step=20, use_kl=True):
        super(BayesRiskEDLLoss, self).__init__()
        self.K = num_classes
        self.annealing_step = annealing_step
        self.use_kl = use_kl

    def forward(self, alpha, y_indices, epoch_num, return_details=False):
        y = F.one_hot(y_indices, num_classes=self.K).float()
        s = torch.sum(alpha, dim=1, keepdim=True)

        l_risk = torch.sum(y * (torch.digamma(s) - torch.digamma(alpha)), dim=1, keepdim=True)

        alpha_tilde = y + (1 - y) * alpha
        first_term = (
            torch.lgamma(torch.sum(alpha_tilde, dim=1, keepdim=True))
            - torch.lgamma(torch.tensor(float(self.K), device=alpha.device))
            - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True)
        )
        second_term = torch.sum(
            (alpha_tilde - 1)
            * (torch.digamma(alpha_tilde) - torch.digamma(torch.sum(alpha_tilde, dim=1, keepdim=True))),
            dim=1,
            keepdim=True,
        )
        l_kl = first_term + second_term

        risk_loss = torch.mean(l_risk)
        kl_loss = torch.mean(l_kl)
        lambda_t = min(1.0, epoch_num / self.annealing_step)
        weighted_kl = lambda_t * kl_loss if self.use_kl else torch.zeros((), device=alpha.device)
        total_loss = risk_loss + weighted_kl

        if return_details:
            return total_loss, risk_loss, weighted_kl
        return total_loss


# ==========================================
# 2. 整体网络架构 (MultiModal Evidential Network)
# ==========================================
class MultiModalEvidentialNetwork(nn.Module):
    def __init__(
        self,
        dim_t=768,
        dim_a=768,
        dim_v=768,
        num_classes=2,
        dropout=0.5,
        min_weight=0.01,
        temperature=0.5,
        use_edl=True,
        use_gating=True,
    ):
        super(MultiModalEvidentialNetwork, self).__init__()
        self.use_edl = use_edl

        if self.use_edl:
            self.em_t = EvidentialModule(dim_t, num_classes, dropout)
            self.em_a = EvidentialModule(dim_a, num_classes, dropout)
            self.em_v = EvidentialModule(dim_v, num_classes, dropout)
        else:
            self.em_t = None
            self.em_a = None
            self.em_v = None

        self.fusion = UncertaintyAwareFusion(
            dim_t,
            dim_a,
            dim_v,
            num_classes,
            dropout,
            min_weight,
            temperature,
            use_gating=use_gating,
            use_edl=use_edl,
        )

    def forward(self, h_t, h_a, h_v):
        if self.use_edl:
            alpha_t, u_t = self.em_t(h_t)
            alpha_a, u_a = self.em_a(h_a)
            alpha_v, u_v = self.em_v(h_v)
            alpha_aux = (alpha_t, alpha_a, alpha_v)
            uncertainty = (u_t, u_a, u_v)
        else:
            alpha_aux = (None, None, None)
            uncertainty = (None, None, None)

        alpha_final, logits_final, probs_final, weights = self.fusion(
            h_t,
            h_a,
            h_v,
            uncertainty[0],
            uncertainty[1],
            uncertainty[2],
        )

        return {
            "alpha_final": alpha_final,
            "logits_final": logits_final,
            "probs_final": probs_final,
            "weights": weights,
            "alpha_aux": alpha_aux,
            "uncertainty": uncertainty,
        }


# ==========================================
# 3. 数据集读取逻辑 (二分类版：过滤 Neutral)
# ==========================================
class CHSIMSDataset(Dataset):
    def __init__(self, h5_t_path, h5_a_path, h5_v_path, meta_path, split="train", augment=True, label_map=None):
        self.h5_t_path = h5_t_path
        self.h5_a_path = h5_a_path
        self.h5_v_path = h5_v_path
        self.split = split
        self.augment = augment and (split == "train")

        self.h5_t = h5py.File(h5_t_path, "r")
        self.h5_a = h5py.File(h5_a_path, "r")
        self.h5_v = h5py.File(h5_v_path, "r")

        df = pd.read_csv(meta_path)
        df["mode"] = df["mode"].str.lower()
        df["annotation"] = df["annotation"].astype(str).str.strip().str.capitalize()
        df["original_idx"] = range(len(df))
        df = df[df["annotation"] != "Neutral"].reset_index(drop=True)

        if label_map is not None:
            self.label_map = label_map
        else:
            unique_labels = sorted(df["annotation"].unique())
            self.label_map = {label: idx for idx, label in enumerate(unique_labels)}

        df_split = df[df["mode"] == split].reset_index(drop=True)
        valid_keys, valid_labels = [], []
        missing_count = 0

        for vid, idx, ann in zip(
            df_split["video_id"].astype(str),
            df_split["original_idx"],
            df_split["annotation"],
        ):
            key = f"{vid.replace('/', '_')}_{idx}"
            if key in self.h5_t and key in self.h5_a and key in self.h5_v:
                valid_keys.append(key)
                valid_labels.append(self.label_map[ann])
            else:
                missing_count += 1

        self.sample_keys = valid_keys
        self.labels = np.array(valid_labels)
        self.num_samples = len(self.sample_keys)

        print(
            f"[{split.upper()}] Dataset Loaded! Valid samples: {self.num_samples} "
            f"(Skipped {missing_count} missing samples). Label map: {self.label_map}"
        )

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


def str2bool(v):
    if isinstance(v, bool):
        return v
    value = str(v).strip().lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_h5_feature_dim(h5_path, fallback_dim):
    with h5py.File(h5_path, "r") as h5f:
        keys = list(h5f.keys())
        if not keys:
            return fallback_dim
        sample = np.asarray(h5f[keys[0]])
        if sample.ndim == 0:
            return fallback_dim
        return int(sample.shape[-1])


def resolve_text_feature_path(args, ablation_cfg):
    if args.text_feature_path:
        return Path(args.text_feature_path)
    if ablation_cfg.text_backbone == "bert":
        return Path(args.bert_feature_path)
    return Path(args.macbert_feature_path)


def build_parser():
    parser = argparse.ArgumentParser(description="CH-SIMS Binary EDL Fusion Training with Ablation Switches")
    parser.add_argument("--ablation_mode", type=str, default="none", choices=ABLATION_CHOICES)
    parser.add_argument("--text_backbone", type=str, default="macbert", choices=TEXT_BACKBONE_CHOICES)
    parser.add_argument("--use_kl_loss", type=str2bool, default=None, help="Optional override for KL loss switch")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--aux_weight", type=float, default=0.3)
    parser.add_argument("--annealing_step", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.45)
    parser.add_argument("--min_weight", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--meta_path", type=str, default="/mnt/f/dataset/CH-SIMS v2.0/ch-simsv2s/meta.csv")
    parser.add_argument("--macbert_feature_path", type=str, default="feature/text_features_v2_macbert_large.h5")
    parser.add_argument("--bert_feature_path", type=str, default="feature/text_features_bert_base_chinese.h5")
    parser.add_argument("--text_feature_path", type=str, default=None)
    parser.add_argument("--audio_feature_path", type=str, default="feature/audio_features_v4_last4layers.h5")
    parser.add_argument("--video_feature_path", type=str, default="feature/video_features_v3.h5")

    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--exp_name", type=str, default=None)
    return parser


# ==========================================
# 4. 主训练循环 (Training Loop)
# ==========================================
def train(args):
    set_seed(args.seed)

    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ablation_cfg = resolve_ablation_config(
        ablation_mode=args.ablation_mode,
        text_backbone=args.text_backbone,
        use_kl_loss_override=args.use_kl_loss,
    )
    exp_name = args.exp_name or exp_name_from_ablation(ablation_cfg.ablation_mode)

    text_path = resolve_text_feature_path(args, ablation_cfg)
    audio_path = Path(args.audio_feature_path)
    video_path = Path(args.video_feature_path)
    meta_path = Path(args.meta_path)

    if not text_path.exists():
        raise FileNotFoundError(
            f"Required text feature file not found: {text_path}. "
            f"You can extract it with: python feature/sims_v2_text_to_h5.py --text_backbone {ablation_cfg.text_backbone}"
        )
    for required_path in (audio_path, video_path, meta_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Required file not found: {required_path}")

    text_fallback_dim = 1024 if ablation_cfg.text_backbone == "macbert" else 768
    dim_t = infer_h5_feature_dim(text_path, text_fallback_dim)
    dim_a = infer_h5_feature_dim(audio_path, 768)
    dim_v = infer_h5_feature_dim(video_path, 768)

    print(f"🚀 Using device: {device} (Binary Mode)")
    print(
        f"[Config] exp={exp_name} | ablation={ablation_cfg.ablation_mode} | "
        f"use_edl={ablation_cfg.use_edl} | use_gating={ablation_cfg.use_gating} | "
        f"text_backbone={ablation_cfg.text_backbone} | use_kl_loss={ablation_cfg.use_kl_loss}"
    )
    print(f"[Feature] text={text_path} (dim={dim_t}) | audio={audio_path} (dim={dim_a}) | video={video_path} (dim={dim_v})")

    model = MultiModalEvidentialNetwork(
        dim_t=dim_t,
        dim_a=dim_a,
        dim_v=dim_v,
        num_classes=num_classes,
        dropout=args.dropout,
        min_weight=args.min_weight,
        temperature=args.temperature,
        use_edl=ablation_cfg.use_edl,
        use_gating=ablation_cfg.use_gating,
    ).to(device)

    _df = pd.read_csv(meta_path)
    _df["annotation"] = _df["annotation"].astype(str).str.strip().str.capitalize()
    _df = _df[_df["annotation"] != "Neutral"]
    global_label_map = {label: idx for idx, label in enumerate(sorted(_df["annotation"].unique()))}
    print(f"Global label map: {global_label_map}")

    train_dataset = CHSIMSDataset(text_path, audio_path, video_path, meta_path, split="train", label_map=global_label_map)
    valid_dataset = CHSIMSDataset(text_path, audio_path, video_path, meta_path, split="valid", label_map=global_label_map)

    class_counts = np.bincount(train_dataset.labels, minlength=num_classes)
    class_weights = np.zeros_like(class_counts, dtype=np.float64)
    non_zero = class_counts > 0
    class_weights[non_zero] = 1.0 / class_counts[non_zero]
    sample_weights = class_weights[train_dataset.labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    criterion_edl = BayesRiskEDLLoss(
        num_classes=num_classes,
        annealing_step=args.annealing_step,
        use_kl=ablation_cfg.use_kl_loss,
    )
    criterion_ce = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    save_root = Path(args.save_dir) / exp_name
    save_root.mkdir(parents=True, exist_ok=True)
    best_valid_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_main_loss, total_kl_loss, total_aux_loss = 0.0, 0.0, 0.0, 0.0
        correct_preds, total_samples = 0, 0
        train_w_t_sum, train_w_a_sum, train_w_v_sum = 0.0, 0.0, 0.0
        train_u_t_sum, train_u_a_sum, train_u_v_sum = 0.0, 0.0, 0.0
        train_u_count = 0

        for batch_idx, (h_t, h_a, h_v, labels) in enumerate(train_loader):
            h_t, h_a, h_v = h_t.to(device), h_a.to(device), h_v.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(h_t, h_a, h_v)

            if ablation_cfg.use_edl:
                loss_fusion, main_fusion, kl_fusion = criterion_edl(
                    outputs["alpha_final"], labels, epoch_num=epoch, return_details=True
                )
                aux_loss = torch.zeros((), device=device)
                aux_main = torch.zeros((), device=device)
                aux_kl = torch.zeros((), device=device)
                for alpha_m in outputs["alpha_aux"]:
                    branch_loss, branch_main, branch_kl = criterion_edl(
                        alpha_m, labels, epoch_num=epoch, return_details=True
                    )
                    aux_loss = aux_loss + branch_loss
                    aux_main = aux_main + branch_main
                    aux_kl = aux_kl + branch_kl

                loss = loss_fusion + args.aux_weight * aux_loss
                main_loss = main_fusion + args.aux_weight * aux_main
                kl_loss = kl_fusion + args.aux_weight * aux_kl
            else:
                loss_fusion = criterion_ce(outputs["logits_final"], labels)
                aux_loss = torch.zeros((), device=device)
                loss = loss_fusion
                main_loss = loss_fusion
                kl_loss = torch.zeros((), device=device)

            loss.backward()
            optimizer.step()
            scheduler.step(epoch - 1 + batch_idx / len(train_loader))

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_main_loss += main_loss.item() * batch_size
            total_kl_loss += kl_loss.item() * batch_size
            total_aux_loss += aux_loss.item() * batch_size

            preds = torch.argmax(outputs["probs_final"], dim=1)
            correct_preds += (preds == labels).sum().item()
            total_samples += batch_size

            w_t, w_a, w_v = outputs["weights"]
            train_w_t_sum += w_t.mean().item()
            train_w_a_sum += w_a.mean().item()
            train_w_v_sum += w_v.mean().item()

            u_t, u_a, u_v = outputs["uncertainty"]
            if u_t is not None:
                train_u_t_sum += u_t.mean().item()
                train_u_a_sum += u_a.mean().item()
                train_u_v_sum += u_v.mean().item()
                train_u_count += 1

        epoch_total_loss = total_loss / total_samples
        epoch_main_loss = total_main_loss / total_samples
        epoch_kl_loss = total_kl_loss / total_samples
        epoch_aux_loss = total_aux_loss / total_samples
        epoch_acc = correct_preds / total_samples * 100

        model.eval()
        valid_total_loss, valid_main_loss, valid_kl_loss = 0.0, 0.0, 0.0
        valid_correct, valid_total = 0, 0
        val_w_t_sum, val_w_a_sum, val_w_v_sum = 0.0, 0.0, 0.0
        val_u_t_sum, val_u_a_sum, val_u_v_sum = 0.0, 0.0, 0.0
        val_u_count = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for h_t, h_a, h_v, labels in valid_loader:
                h_t, h_a, h_v = h_t.to(device), h_a.to(device), h_v.to(device)
                labels = labels.to(device)
                outputs = model(h_t, h_a, h_v)

                if ablation_cfg.use_edl:
                    loss, main_loss, kl_loss = criterion_edl(
                        outputs["alpha_final"], labels, epoch_num=epoch, return_details=True
                    )
                else:
                    loss = criterion_ce(outputs["logits_final"], labels)
                    main_loss = loss
                    kl_loss = torch.zeros((), device=device)

                batch_size = labels.size(0)
                valid_total_loss += loss.item() * batch_size
                valid_main_loss += main_loss.item() * batch_size
                valid_kl_loss += kl_loss.item() * batch_size

                preds = torch.argmax(outputs["probs_final"], dim=1)
                valid_correct += (preds == labels).sum().item()
                valid_total += batch_size
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                w_t, w_a, w_v = outputs["weights"]
                val_w_t_sum += w_t.mean().item()
                val_w_a_sum += w_a.mean().item()
                val_w_v_sum += w_v.mean().item()

                u_t, u_a, u_v = outputs["uncertainty"]
                if u_t is not None:
                    val_u_t_sum += u_t.mean().item()
                    val_u_a_sum += u_a.mean().item()
                    val_u_v_sum += u_v.mean().item()
                    val_u_count += 1

        valid_epoch_total_loss = valid_total_loss / valid_total
        valid_epoch_main_loss = valid_main_loss / valid_total
        valid_epoch_kl_loss = valid_kl_loss / valid_total
        valid_epoch_acc = valid_correct / valid_total * 100
        valid_f1 = f1_score(all_labels, all_preds, average="macro")

        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = save_root / f"best_model_{exp_name}_{timestamp}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"🌟 New best model saved! Valid F1: {valid_f1:.4f} | Path: {save_path}")

        train_batch_count = len(train_loader)
        val_batch_count = len(valid_loader)
        train_uncertainty_msg = ""
        val_uncertainty_msg = ""
        if train_u_count > 0:
            train_uncertainty_msg = (
                f" | Train U - T:{train_u_t_sum / train_u_count:.4f}, "
                f"A:{train_u_a_sum / train_u_count:.4f}, V:{train_u_v_sum / train_u_count:.4f}"
            )
        if val_u_count > 0:
            val_uncertainty_msg = (
                f" | Val U - T:{val_u_t_sum / val_u_count:.4f}, "
                f"A:{val_u_a_sum / val_u_count:.4f}, V:{val_u_v_sum / val_u_count:.4f}"
            )

        print(
            f"[{exp_name}] Epoch [{epoch:02d}/{args.epochs}] | "
            f"Train Total/Main/KL/Aux: {epoch_total_loss:.4f}/{epoch_main_loss:.4f}/{epoch_kl_loss:.4f}/{epoch_aux_loss:.4f} | "
            f"Train Acc: {epoch_acc:.2f}% | "
            f"Train Weights - T:{train_w_t_sum / train_batch_count:.4f}, "
            f"A:{train_w_a_sum / train_batch_count:.4f}, V:{train_w_v_sum / train_batch_count:.4f}"
            f"{train_uncertainty_msg}"
        )
        print(
            f"[{exp_name}] Valid Total/Main/KL: {valid_epoch_total_loss:.4f}/{valid_epoch_main_loss:.4f}/{valid_epoch_kl_loss:.4f} | "
            f"Valid Acc: {valid_epoch_acc:.2f}% | Valid F1: {valid_f1:.4f} | "
            f"Val Weights - T:{val_w_t_sum / val_batch_count:.4f}, "
            f"A:{val_w_a_sum / val_batch_count:.4f}, V:{val_w_v_sum / val_batch_count:.4f}"
            f"{val_uncertainty_msg}"
        )

    print(f"\n🎉 Binary training completed! Best validation F1: {best_valid_f1:.4f} | Exp: {exp_name}")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)
