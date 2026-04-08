import os
import torch
import librosa
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# ==========================================
# 1. 配置参数
# ==========================================
# 如果你想强调中文语音的韵律，推荐使用腾讯开源的中文 Wav2Vec2，或者直接用 facebook 原版
MODEL_NAME = "facebook/wav2vec2-base" # 你也可以换成 "TencentGameMate/chinese-wav2vec2-base"
META_PATH = "/mnt/f/dataset/CH-SIMS v2.0/ch-simsv2s/meta.csv"  # 标签文件
RAW_VIDEO_DIR = "/mnt/f/dataset/CH-SIMS v2.0/ch-simsv2s/Raw"   # 原始视频文件夹路径
OUTPUT_PATH = "audio_features.h5"     # 输出文件
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR = 16000                     # Wav2Vec 2.0 强制要求的采样率

def extract_audio_features():
    # 2. 加载数据索引
    df = pd.read_csv(META_PATH, dtype={"video_id": str, "clip_id": str})
    video_ids = df['video_id'].astype(str).tolist()
    clip_ids = df['clip_id'].astype(str).tolist()
    sample_keys = [f"{vid.replace('/', '_')}_{i}" for i, vid in enumerate(video_ids)]

    # 3. 初始化处理器和模型
    print(f"Loading Wav2Vec 2.0 model on {DEVICE}...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(DEVICE)
    
    # 按照论文设计：冻结参数
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # 4. 开始提取
    print("开始提取音频特征 (这可能会花费一些时间，因为涉及视频解码)...")
    
    # 记录未能成功提取的视频（比如视频损坏或没声音）
    error_logs = []

    with h5py.File(OUTPUT_PATH, 'w') as h5f:
        for i, vid in enumerate(tqdm(video_ids)):
            # CH-SIMS v2.0: Raw/video_id/clip_id.mp4
            video_path = os.path.join(RAW_VIDEO_DIR, vid, f"{clip_ids[i]}.mp4")
            
            if not os.path.exists(video_path):
                error_logs.append(f"{vid}/{clip_ids[i]}: File not found")
                continue
                
            try:
                # [关键步骤] 1. 使用 librosa 从 mp4 中提取音频并重采样为 16kHz
                # sr=TARGET_SR 会自动进行重采样，mono=True 会自动转为单声道
                speech_array, sr = librosa.load(video_path, sr=TARGET_SR, mono=True)
                
                # [关键步骤] 2. 数据预处理，转为模型需要的 tensor
                inputs = processor(speech_array, sampling_rate=TARGET_SR, return_tensors="pt")
                input_values = inputs.input_values.to(DEVICE)
                
                with torch.no_grad():
                    # [关键步骤] 3. 模型前向传播
                    outputs = model(input_values)
                    
                    # 提取最后一层的隐藏状态: [batch_size=1, seq_len, hidden_dim]
                    # base 模型的 hidden_dim 通常是 768
                    last_hidden_state = outputs.last_hidden_state
                    
                    # 按照论文设计：Global Average Pooling 得到 h_a
                    # h_a ∈ R^{d_a}
                    h_a = torch.mean(last_hidden_state, dim=1).squeeze().cpu().numpy()
                
                # 保存到 h5 文件，使用唯一 key（video_id + 行号）
                h5f.create_dataset(sample_keys[i], data=h_a)
                
            except Exception as e:
                error_logs.append(f"{vid}: {str(e)}")

    print(f"提取完成！特征已保存至: {OUTPUT_PATH}")
    if error_logs:
        print(f"警告: 有 {len(error_logs)} 个文件处理失败，详情见下:")
        for log in error_logs[:5]: # 打印前几个错误
            print(log)

if __name__ == "__main__":
    extract_audio_features()
