"""
CMU-MOSI 音频特征提取
使用 facebook/wav2vec2-base-960h（英文），最后4层均值 + mean pooling -> 768 维
直接读取 Audio/WAV_16000/Segmented/ 下的 .wav 文件，无需从视频中分离

前置条件：先运行 make_mosi_meta.py 生成 mosi_meta.csv

输出：audio_features_mosi.h5
  key 格式：{video_id}_{clip_id}  (e.g. "03bSnISJMiM_1")
"""

import os
import torch
import librosa
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 配置参数
# ==========================================
MODEL_NAME   = "facebook/wav2vec2-base-960h"
META_PATH    = Path(__file__).parent / "label.csv"
WAV_DIR      = Path("/mnt/f/dataset/CMU/CMU-MOSI/Audio/WAV_16000/Segmented")
OUTPUT_PATH  = Path(__file__).parent / "audio_features_mosi.h5"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR    = 16000


def extract():
    df = pd.read_csv(META_PATH, dtype={"video_id": str, "clip_id": int})
    print(f"共 {len(df)} 个样本，设备: {DEVICE}")

    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME, output_hidden_states=True).to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    error_logs = []
    success = 0

    with h5py.File(OUTPUT_PATH, "w") as h5f:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            vid = str(row["video_id"])
            cid = int(row["clip_id"])
            key = f"{vid}_{cid}"
            wav_path = WAV_DIR / f"{vid}_{cid}.wav"

            if not wav_path.exists():
                error_logs.append(f"{key}: wav 文件不存在 ({wav_path})")
                continue

            try:
                audio, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)

                # VAD 静音切除
                audio_trimmed, _ = librosa.effects.trim(audio, top_db=25)
                if len(audio_trimmed) < TARGET_SR * 0.5:
                    audio_trimmed = audio

                inputs = processor(
                    audio_trimmed,
                    sampling_rate=TARGET_SR,
                    return_tensors="pt",
                )
                input_values = inputs.input_values.to(DEVICE)

                with torch.no_grad():
                    outputs = model(input_values)
                    # 最后4层均值
                    last_4 = torch.stack(outputs.hidden_states[-4:], dim=0)  # [4, 1, T, 768]
                    avg_hidden = last_4.mean(dim=0)                           # [1, T, 768]
                    h_a = avg_hidden.mean(dim=1).squeeze().cpu().numpy()      # [768]

                h5f.create_dataset(key, data=h_a)
                success += 1

            except Exception as e:
                error_logs.append(f"{key}: {e}")

    print(f"\n完成！成功: {success}/{len(df)}，输出: {OUTPUT_PATH}")
    if error_logs:
        print(f"失败 {len(error_logs)} 个，前5条:")
        for log in error_logs[:5]:
            print(log)


if __name__ == "__main__":
    extract()
