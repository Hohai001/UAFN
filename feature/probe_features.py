"""
特征质量诊断脚本 —— Linear Probe + t-SNE
用法: python probe_features.py
"""
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 无头模式，保存为图片

# ==========================================
# 配置
# ==========================================
META_PATH  = "/mnt/f/dataset/CH-SIMS v2.0/ch-simsv2s/meta.csv"
FEATURE_DIR = Path(__file__).parent

PROBES = {
    "audio_v3_middle": FEATURE_DIR / "audio_features_v3_middle_layer.h5",
    "audio_v4_last4":  FEATURE_DIR / "audio_features_v4_last4layers.h5",
    "text":            FEATURE_DIR / "text_features_v2_macbert_large.h5",
    "video":           FEATURE_DIR / "video_features_v3.h5",
}

# ==========================================
# 工具函数
# ==========================================
def load_split(h5_path, meta_path, split):
    """从 h5 文件中按 split 读取特征和标签"""
    df = pd.read_csv(meta_path)
    df['mode'] = df['mode'].str.lower()
    df['annotation'] = df['annotation'].astype(str).str.strip().str.capitalize()
    df['original_idx'] = range(len(df))

    # 全局 label_map，保证 train/valid 一致
    label_map = {l: i for i, l in enumerate(sorted(df['annotation'].unique()))}

    df_split = df[df['mode'] == split].reset_index(drop=True)

    feats, labels = [], []
    with h5py.File(h5_path, 'r') as f:
        for vid, idx, ann in zip(df_split['video_id'].astype(str),
                                  df_split['original_idx'],
                                  df_split['annotation']):
            key = f"{vid.replace('/', '_')}_{idx}"
            if key in f:
                feats.append(np.array(f[key]))
                labels.append(label_map[ann])

    return np.array(feats), np.array(labels), label_map


def linear_probe(name, h5_path):
    """训练逻辑回归，返回验证集准确率"""
    print(f"\n{'='*50}")
    print(f"  Linear Probe: {name}")
    print(f"{'='*50}")

    X_train, y_train, label_map = load_split(h5_path, META_PATH, 'train')
    X_valid, y_valid, _         = load_split(h5_path, META_PATH, 'valid')

    print(f"  Train: {len(X_train)} samples | Valid: {len(X_valid)} samples")
    print(f"  Feature dim: {X_train.shape[1]} | Labels: {label_map}")

    # 检查特征是否有 NaN/Inf（特征提取 bug 的常见症状）
    bad_train = np.isnan(X_train).any() or np.isinf(X_train).any()
    bad_valid = np.isnan(X_valid).any() or np.isinf(X_valid).any()
    if bad_train or bad_valid:
        print(f"  ⚠️  警告：特征中含有 NaN 或 Inf！train={bad_train}, valid={bad_valid}")

    # 特征统计（健康特征应接近均值0、标准差1）
    print(f"  Train 特征统计: mean={X_train.mean():.4f}, std={X_train.std():.4f}, "
          f"min={X_train.min():.4f}, max={X_train.max():.4f}")

    # 标准化 + 逻辑回归
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_valid_s = scaler.transform(X_valid)

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train_s, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train_s)) * 100
    valid_acc = accuracy_score(y_valid, clf.predict(X_valid_s)) * 100

    print(f"\n  Train Acc: {train_acc:.2f}%  |  Valid Acc: {valid_acc:.2f}%")
    print(f"\n  分类报告 (Valid):")
    label_names = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]
    print(classification_report(y_valid, clf.predict(X_valid_s),
                                 target_names=label_names, digits=3))
    return valid_acc


def tsne_plot(name, h5_path, save_path):
    """t-SNE 可视化，保存为 PNG"""
    X, y, label_map = load_split(h5_path, META_PATH, 'valid')

    # t-SNE 降到 2 维
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    X_2d = tsne.fit_transform(X)

    colors = ['#e74c3c', '#3498db', '#2ecc71']
    label_names = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]

    plt.figure(figsize=(7, 6))
    for i, lname in enumerate(label_names):
        mask = y == i
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                    c=colors[i], label=lname, alpha=0.6, s=20)
    plt.title(f"t-SNE: {name} (valid set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  t-SNE 图已保存: {save_path}")


# ==========================================
# 主流程
# ==========================================
if __name__ == "__main__":
    results = {}

    # 1. Linear Probe 对比
    for name, path in PROBES.items():
        if path.exists():
            results[name] = linear_probe(name, path)
        else:
            print(f"\n  跳过 {name}（文件不存在: {path}）")

    # 2. 汇总对比表
    print(f"\n{'='*50}")
    print("  汇总：各模态线性探针验证集准确率")
    print(f"{'='*50}")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        bar = '█' * int(acc / 2)
        print(f"  {name:<25} {acc:5.2f}%  {bar}")

    # 3. t-SNE（对 audio v3 vs v4 可视化）
    print("\n正在生成 t-SNE 图（这可能需要 1-2 分钟）...")
    for name in ["audio_v3_middle", "audio_v4_last4"]:
        path = PROBES.get(name)
        if path and path.exists():
            tsne_plot(name, path, FEATURE_DIR / f"tsne_{name}.png")
