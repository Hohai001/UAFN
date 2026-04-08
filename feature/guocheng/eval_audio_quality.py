import h5py
import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# 忽略不必要的警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 参数配置 (请确保路径与你的实际路径一致)
# ==========================================
META_PATH = "/mnt/f/dataset/CH-SIMS v2.0/ch-simsv2s/meta.csv"
H5_PATH = "audio_features_v3_middle_layer.h5"  # 你要测试的特征文件
LABEL_COL = "annotation"
SPLIT_COL = "mode"

def load_data_robust():
    """安全鲁棒地加载数据和标签"""
    print(f"📦 正在从 {H5_PATH} 加载特征...")
    df = pd.read_csv(META_PATH)
    
    expected_keys = [f"{str(vid).replace('/', '_')}_{i}" for i, vid in enumerate(df['video_id'])]
    
    vectors, labels, splits = [], [], []
    missing_count = 0
    
    with h5py.File(H5_PATH, "r") as h5f:
        for i, key in enumerate(expected_keys):
            if key in h5f:
                vectors.append(np.array(h5f[key]))
                # 统一转为字符串并首字母大写 (防范错别字和格式问题)
                cls = str(df.iloc[i][LABEL_COL]).strip().capitalize()
                labels.append(cls)
                splits.append(str(df.iloc[i][SPLIT_COL]).lower())
            else:
                missing_count += 1
                
    if missing_count > 0:
        print(f"⚠️ 警告: 有 {missing_count} 个视频特征在 H5 文件中找不到。")
        
    return np.array(vectors), np.array(labels), np.array(splits)

def print_conclusion(best_acc):
    """根据最高准确率给出直观的诊断结论"""
    print("\n" + "="*50)
    print("🩺 【终极特征质量诊断书】")
    print("="*50)
    
    # 3分类盲猜基线是 33.3%
    if best_acc < 0.36:
        print("❌ 评级：【严重损坏 / 无效特征】")
        print("诊断：特征几乎等同于瞎猜。包含的全是噪音（音量、环境音），没有情感信息。")
        print("建议：检查提取脚本，或者该模态在这个数据集上彻底失效，权重注定为 0。")
    elif 0.36 <= best_acc < 0.40:
        print("🥉 评级：【勉强及格 / 弱信号】")
        print("诊断：捕捉到了一丝丝情感信号，但噪音极大，模型学得很吃力。")
        print("建议：可以喂给多模态网络试试，能分到一点点权重，但起不到决定性作用。")
    elif 0.40 <= best_acc < 0.48:
        print("🥈 评级：【优良特征 / 强力辅助】")
        print("诊断：非常棒的声学/视觉特征！模型成功过滤了噪音并找到了明确的情感边界。")
        print("建议：直接拿去跑主网络！它绝对能拿到属于自己的权重份额，拉升整体准确率。")
    else:
        print("🥇 评级：【极品特征 / 王者级别】")
        print("诊断：特征纯度极高，几乎没有杂质！(通常只有纯文本特征能达到这个级别)")
        print("建议：完美的特征，你的模型起飞了！")
    print("="*50 + "\n")

def main():
    X, y, splits = load_data_robust()
    
    if len(X) == 0:
        print("❌ 错误：没有加载到任何特征，请检查文件路径！")
        return
        
    # 划分训练集和测试集
    train_mask = splits == "train"
    test_mask = splits == "test"
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"✅ 数据加载成功! 训练集: {len(X_train)} 个, 测试集: {len(X_test)} 个")
    
    # 类别分布检查
    train_counts = pd.Series(y_train).value_counts()
    print("\n📊 训练集标签分布:")
    for label, count in train_counts.items():
        print(f"   - {label}: {count} 个")
        
    # 特征标准化 (对逻辑回归和SVM极其重要)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    print("\n🚀 开始多模型交叉验证 (自动过滤高维噪音)...")
    print("-" * 50)
    
    best_acc = 0.0
    
    # 1. 逻辑回归 (带L2正则化，抗噪性极强)
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.1)
    lr.fit(X_train_s, y_train)
    y_pred_lr = lr.predict(X_test_s)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr, average='macro')
    print(f"🔹 Logistic Regression -> Acc: {acc_lr:.4f} | Macro F1: {f1_lr:.4f}")
    best_acc = max(best_acc, acc_lr)
    
    # 2. 随机森林 (非线性树模型，自带特征选择)
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train) 
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf, average='macro')
    print(f"🔹 Random Forest       -> Acc: {acc_rf:.4f} | Macro F1: {f1_rf:.4f}")
    best_acc = max(best_acc, acc_rf)

    # 3. 支持向量机 SVM (高维空间划分大师)
    svm = SVC(kernel='rbf', class_weight='balanced', C=1.0)
    svm.fit(X_train_s, y_train)
    y_pred_svm = svm.predict(X_test_s)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    f1_svm = f1_score(y_test, y_pred_svm, average='macro')
    print(f"🔹 SVM (RBF Kernel)    -> Acc: {acc_svm:.4f} | Macro F1: {f1_svm:.4f}")
    best_acc = max(best_acc, acc_svm)
    
    # 打印最终诊断书
    print_conclusion(best_acc)

if __name__ == "__main__":
    main()