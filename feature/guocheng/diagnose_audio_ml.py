import h5py
import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ==========================================
# 参数配置
# ==========================================
META_PATH = "/mnt/f/dataset/CH-SIMS v2.0/ch-simsv2s/meta.csv"
H5_PATH = "audio_features_v3_middle_layer.h5"
LABEL_COL = "annotation"
SPLIT_COL = "mode"

def load_data():
    df = pd.read_csv(META_PATH)
    
    # 构建 keys
    expected_keys = [f"{str(vid).replace('/', '_')}_{i}" for i, vid in enumerate(df['video_id'])]
    
    vectors = []
    labels = []
    splits = []
    
    with h5py.File(H5_PATH, "r") as h5f:
        for i, key in enumerate(expected_keys):
            if key in h5f:
                vectors.append(np.array(h5f[key]))
                
                # 【修复核心】：直接将表格里的单词读作标签，去掉空格并首字母大写统一格式
                cls = str(df.iloc[i][LABEL_COL]).strip().capitalize()
                labels.append(cls)
                
                splits.append(str(df.iloc[i][SPLIT_COL]).lower())
                
    return np.array(vectors), np.array(labels), np.array(splits)

def main():
    print(f"Loading data from {H5_PATH}...")
    X, y, splits = load_data()
    
    if len(X) == 0:
        print("Error: No data loaded.")
        return
        
    print(f"Total samples: {len(X)}")
    
    # 划分训练集和测试集
    train_mask = splits == "train"
    test_mask = splits == "test"
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # 打印类别分布，看看是不是某一类特别多
    train_counts = pd.Series(y_train).value_counts()
    print("\n[Train Label Distribution]")
    print(train_counts)
    
    # ==========================================
    # 核心测试：使用带权重的机器学习模型
    # ==========================================
    # 必须做标准化，否则大数值维度会吃掉小数值维度
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    print("\n" + "="*40)
    print("🚀 诊断开始：机器学习探测")
    print("="*40)
    
    # 1. 逻辑回归 (带L2正则化，抗噪性强)
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.1)
    lr.fit(X_train_s, y_train)
    y_pred_lr = lr.predict(X_test_s)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr, average='macro')
    print(f"📊 Logistic Regression -> Acc: {acc_lr:.4f}, Macro F1: {f1_lr:.4f}")
    
    # 2. 随机森林 (非线性映射，自带特征选择)
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train) # 树模型不用标准化
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf, average='macro')
    print(f"🌲 Random Forest       -> Acc: {acc_rf:.4f}, Macro F1: {f1_rf:.4f}")

if __name__ == "__main__":
    main()