import os
import torch
import pandas as pd
import numpy as np
import traceback
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from moviepy import VideoFileClip

# ==========================================
# 仅测试第一条数据，排查集体报错的真凶
# ==========================================
MODEL_NAME = "TencentGameMate/chinese-wav2vec2-base" 
META_PATH = "/mnt/f/dataset/CH-SIMS v2.0/ch-simsv2s/meta.csv"
RAW_VIDEO_DIR = "/mnt/f/dataset/CH-SIMS v2.0/ch-simsv2s/Raw"
TARGET_SR = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def debug_single_audio():
    df = pd.read_csv(META_PATH, dtype={"video_id": str, "clip_id": str})
    
    # 直接拿第一行数据开刀
    row = df.iloc[0]
    vid = str(row['video_id'])
    cid = str(row['clip_id'])
    video_path = os.path.join(RAW_VIDEO_DIR, vid, f"{cid}.mp4")
    
    print(f"🔍 正在解剖测试视频: {video_path}")
    
    if not os.path.exists(video_path):
        print("❌ 错误：视频文件根本不存在，请检查路径！")
        return

    try:
        print("⏳ 1. 测试 moviepy 音频分离...")
        video_clip = VideoFileClip(video_path)
        if video_clip.audio is None:
            print("❌ 错误：视频无音轨")
            return
        audio_array = video_clip.audio.to_soundarray(fps=TARGET_SR)
        video_clip.close()
        
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        audio_array = audio_array.astype(np.float32)
        print(f"✅ moviepy 提取成功！音频形状: {audio_array.shape}")

        print("⏳ 2. 加载模型与预处理器...")
        processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
        model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(DEVICE)
        
        print("⏳ 3. 测试模型前向传播...")
        inputs = processor(audio_array, sampling_rate=TARGET_SR, return_tensors="pt")
        input_values = inputs.input_values.to(DEVICE)
        
        with torch.no_grad():
            outputs = model(input_values)
            last_hidden_state = outputs.last_hidden_state
            print(f"✅ 模型输出成功！隐藏层形状: {last_hidden_state.shape}")
            
            mean_feat = torch.mean(last_hidden_state, dim=1).squeeze()
            max_feat = torch.max(last_hidden_state, dim=1).values.squeeze()
            h_a = (mean_feat + max_feat).cpu().numpy()
            print(f"✅ 池化合并成功！最终特征形状: {h_a.shape}")
            
        print("\n🎉 没有任何报错，完美通过测试！")

    except Exception as e:
        print("\n🚨 逮到真凶了！详细报错信息如下：")
        print("-" * 50)
        traceback.print_exc()
        print("-" * 50)

if __name__ == "__main__":
    debug_single_audio()