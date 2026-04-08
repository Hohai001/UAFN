import os
import torch
import librosa
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from moviepy import VideoFileClip
import warnings

# 忽略 moviepy 和 transformers 的冗余警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 配置参数
# ==========================================
MODEL_NAME = "TencentGameMate/chinese-wav2vec2-base" 
META_PATH = "/mnt/f/dataset/CH-SIMS v2.0/ch-simsv2s/meta.csv"
RAW_VIDEO_DIR = "/mnt/f/dataset/CH-SIMS v2.0/ch-simsv2s/Raw"
OUTPUT_PATH = "audio_features_v4_last4layers.h5"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR = 16000

def extract_audio_features_final():
    df = pd.read_csv(META_PATH, dtype={"video_id": str, "clip_id": str})
    
    print(f"🚀 正在加载模型 ({MODEL_NAME}) 到 {DEVICE}...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    
    # 【核心优化 1】: output_hidden_states=True，强迫模型吐出所有 12 层的中间特征
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME, output_hidden_states=True).to(DEVICE)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print("🔥 开始音频特征提取 (最后4层均值 + mean pooling)...")
    error_logs = []
    success_count = 0

    with h5py.File(OUTPUT_PATH, 'w') as h5f:
        for i, row in tqdm(df.iterrows(), total=len(df)):
            vid = str(row['video_id'])
            cid = str(row['clip_id'])
            key = f"{vid.replace('/', '_')}_{i}"
            video_path = os.path.join(RAW_VIDEO_DIR, vid, f"{cid}.mp4")
            
            if not os.path.exists(video_path):
                error_logs.append(f"{vid}/{cid}: 视频文件不存在")
                continue
                
            try:
                # ------------------------------------------------
                # 步骤 A：安全分离音频 (moviepy)
                # ------------------------------------------------
                video_clip = VideoFileClip(video_path)
                if video_clip.audio is None:
                    error_logs.append(f"{vid}/{cid}: 视频无音轨")
                    video_clip.close()
                    continue
                
                # 提取音频矩阵
                audio_array = video_clip.audio.to_soundarray(fps=TARGET_SR)
                video_clip.close()
                
                # 转为单声道
                if len(audio_array.shape) > 1:
                    audio_array = audio_array.mean(axis=1)
                audio_array = audio_array.astype(np.float32)

                # ------------------------------------------------
                # 步骤 B：VAD 静音切除 (librosa)
                # ------------------------------------------------
                # 切除开头和结尾低于 25 分贝的无意义静音环境噪音
                audio_array_trimmed, _ = librosa.effects.trim(audio_array, top_db=25)
                
                # 安全兜底：如果切完发现音频长度小于 0.5 秒，说明全是低音量，恢复原状
                if len(audio_array_trimmed) < TARGET_SR * 0.5: 
                    audio_array_trimmed = audio_array

                # ------------------------------------------------
                # 步骤 C：模型前向传播
                # ------------------------------------------------
                inputs = processor(audio_array_trimmed, sampling_rate=TARGET_SR, return_tensors="pt")
                input_values = inputs.input_values.to(DEVICE)
                
                with torch.no_grad():
                    outputs = model(input_values)

                    # 取最后 4 层的平均，比单独一层包含更丰富的情感语义
                    # hidden_states[0] 是 CNN 输出，[1]~[12] 是 transformer 层
                    last_4_layers = torch.stack(outputs.hidden_states[-4:], dim=0)  # [4, 1, seq_len, 768]
                    avg_hidden = last_4_layers.mean(dim=0)  # [1, seq_len, 768]

                    # 对时间维度做 mean pooling，得到整段音频的全局表示
                    h_a = avg_hidden.mean(dim=1).squeeze().cpu().numpy()  # [768]
                
                # 写入 H5 文件
                h5f.create_dataset(key, data=h_a)
                success_count += 1
                
            except Exception as e:
                error_logs.append(f"{vid}/{cid}: {str(e)}")

    print(f"\n🎉 提取完成！成功处理了 {success_count} 个文件。")
    print(f"📦 终极特征已保存至: {OUTPUT_PATH}")
    
    if error_logs:
        print(f"\n⚠️ 警告: 有 {len(error_logs)} 个文件处理失败（静音或文件损坏），前 5 个报错如下:")
        for log in error_logs[:5]:
            print(log)

if __name__ == "__main__":
    extract_audio_features_final()