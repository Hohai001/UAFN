import os
import cv2
import torch
import torch.nn as nn
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision.models.video import swin3d_s, Swin3D_S_Weights

# ==========================================
# 1. 配置参数
# ==========================================
META_PATH = "/mnt/f/dataset/CH-SIMS v2.0/ch-simsv2s/meta.csv"  # 标签文件
RAW_VIDEO_DIR = "/mnt/f/dataset/CH-SIMS v2.0/ch-simsv2s/Raw"   # 原始视频文件夹路径
OUTPUT_PATH = "video_features.h5"     # 输出的特征文件
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_FRAMES = 16                       # 提取的帧数 (T)
IMAGE_SIZE = 224                      # 模型需要的输入分辨率 (H, W)

def get_video_frames(video_path, num_frames=16, size=224):
    """从视频中均匀抽取 num_frames 帧，并处理成 Video Swin 需要的格式"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        frames = [np.zeros((size, size, 3), dtype=np.uint8) for _ in range(num_frames)]
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            frames = [np.zeros((size, size, 3), dtype=np.uint8) for _ in range(num_frames)]
        else:
            # 计算均匀采样的索引
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    # 如果读取失败（视频末尾偶尔会发生），用全黑帧代替
                    frame = np.zeros((size, size, 3), dtype=np.uint8)
                else:
                    # BGR 转 RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # 缩放至 224x224
                    frame = cv2.resize(frame, (size, size))
                frames.append(frame)
    cap.release()
    
    # 转换为 numpy array: 形状为 (T, H, W, C)
    frames = np.array(frames)
    
    # 转换为 PyTorch tensor 并归一化到 [0, 1]
    tensor_frames = torch.from_numpy(frames).float() / 255.0
    
    # ImageNet 标准归一化参数
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3)
    tensor_frames = (tensor_frames - mean) / std
    
    # 调整维度顺序为 Video Swin 要求的: (C, T, H, W)
    tensor_frames = tensor_frames.permute(3, 0, 1, 2)
    return tensor_frames

def extract_visual_features():
    df = pd.read_csv(META_PATH, dtype={"video_id": str, "clip_id": str})
    video_ids = df['video_id'].astype(str).tolist()
    clip_ids = df['clip_id'].astype(str).tolist()
    sample_keys = [f"{vid.replace('/', '_')}_{i}" for i, vid in enumerate(video_ids)]

    # 3. 初始化 Video Swin Transformer (这里使用 Small 版本平衡性能与显存)
    print(f"Loading Video Swin3D model on {DEVICE}...")
    # 加载预训练权重 (Kinetics-400 动作识别数据集)
    weights = Swin3D_S_Weights.DEFAULT
    model = swin3d_s(weights=weights).to(DEVICE)
    
    # 使用 Identity 替换分类头，保留原始 forward，输出 768 维特征
    model.head = nn.Identity()
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print("开始提取视觉时空特征 (非常消耗 GPU 算力，请耐心等待)...")
    
    error_logs = []

    with h5py.File(OUTPUT_PATH, 'w') as h5f:
        for i, vid in enumerate(tqdm(video_ids)):
            # CH-SIMS v2.0: Raw/video_id/clip_id.mp4
            video_path = os.path.join(RAW_VIDEO_DIR, vid, f"{clip_ids[i]}.mp4")
            
            if not os.path.exists(video_path):
                error_logs.append(f"{vid}/{clip_ids[i]}: File not found")
                continue
                
            try:
                # 1. 抽帧并预处理 -> shape: (C, T, H, W)
                video_tensor = get_video_frames(video_path, NUM_FRAMES, IMAGE_SIZE)
                # 增加 Batch 维度 -> shape: (1, C, T, H, W)
                video_tensor = video_tensor.unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    # 2. 前向传播提取特征
                    # 输出形状: (1, 768)
                    features = model(video_tensor)
                    
                    # 3. 取出一维向量 h_v (768 维)
                    h_v = features.squeeze(0).cpu().numpy()
                
                # 保存到 h5 文件，使用唯一 key（video_id + 行号）
                h5f.create_dataset(sample_keys[i], data=h_v)
                
            except Exception as e:
                error_logs.append(f"{vid}/{clip_ids[i]}: {str(e)}")

    print(f"提取完成！特征已保存至: {OUTPUT_PATH}")
    if error_logs:
        print(f"警告: 有 {len(error_logs)} 个文件处理失败，详情见下:")
        for log in error_logs[:5]:
            print(log)

if __name__ == "__main__":
    extract_visual_features()
