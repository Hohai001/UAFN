"""
CMU-MOSI 文本特征提取
使用 bert-base-uncased（英文），最后4层均值 + mean pooling -> 768 维

前置条件：先运行 make_mosi_meta.py 生成 mosi_meta.csv

输出：text_features_mosi.h5
  key 格式：{video_id}_{clip_id}  (e.g. "03bSnISJMiM_1")
"""

import torch
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ==========================================
# 配置参数
# ==========================================
MODEL_NAME  = "bert-base-uncased"
META_PATH   = Path(__file__).parent / "label.csv"
OUTPUT_PATH = Path(__file__).parent / "text_features_mosi.h5"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract():
    df = pd.read_csv(META_PATH, dtype={"video_id": str, "clip_id": int})
    print(f"共 {len(df)} 个样本，设备: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True).to(DEVICE)
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
            text = str(row["text"]) if pd.notna(row["text"]) else ""

            if not text.strip():
                error_logs.append(f"{key}: 空文本，跳过")
                continue

            try:
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                ).to(DEVICE)

                with torch.no_grad():
                    outputs = model(**inputs)
                    # 取最后4层均值
                    last_4 = torch.stack(outputs.hidden_states[-4:], dim=0)  # [4, 1, seq, 768]
                    avg_hidden = last_4.mean(dim=0)                          # [1, seq, 768]
                    # mean pooling
                    mask = inputs["attention_mask"].unsqueeze(-1).float()    # [1, seq, 1]
                    h_t = (avg_hidden * mask).sum(dim=1) / mask.sum(dim=1)  # [1, 768]
                    h_t = h_t.squeeze().cpu().numpy()

                h5f.create_dataset(key, data=h_t)
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
