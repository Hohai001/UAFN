"""
从 CMU-MOSI 的 .csd 标签文件生成 meta.csv
运行一次即可，其他三个提取脚本都依赖这个 meta.csv

CMU-MOSI 标签是连续情感分值 [-3, +3]，二分类映射：
  score >= 0  -> Positive (1)
  score <  0  -> Negative (0)
  (Neutral 不存在，MOSI 没有中性标签)

输出列：
  video_id   : YouTube 视频 ID（字符串）
  clip_id    : 片段编号（整数，从 1 开始）
  text       : 转录文本
  label      : 连续情感分值
  annotation : 二分类标签字符串 Negative / Positive
  mode       : train / valid / test（使用标准 MOSI 划分）
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================
# 配置路径
# ==========================================
CSD_LABEL_PATH  = "/mnt/f/dataset/CMU/CMU-MOSI/labels/CMU_MOSI_Opinion_Labels.csd"  # 需先下载
TRANSCRIPT_DIR  = Path("/mnt/f/dataset/CMU/CMU-MOSI/Transcript/Segmented")
VIDEO_SEG_DIR   = Path("/mnt/f/dataset/CMU/CMU-MOSI/Video/Segmented")
OUTPUT_META     = Path(__file__).parent / "mosi_meta.csv"

# 标准 CMU-MOSI train/valid/test 划分（视频 ID 级别）
# 来源：CMU-MultimodalSDK 官方划分
TRAIN_IDS = {
    "2iD-tVS8NPw", "8d-gEyoeBzc", "Af8D0E4ZXaw", "AoTD39YUXMY", "BioHAh1qJAQ",
    "BvYR0L6f2Ig", "BXuRRbG0Ugk", "c5xsKMxpXnc", "c7UH_rxdZv4", "cMLPaz0YUVY",
    "D4IOjFN_Scc", "d6hH302o4v8", "dq3Nf_lMPnE", "DX6B_EHkbCQ", "EhMJ0DuOfPM",
    "F7yD7PYp_8o", "fvVMiF_7Bkk", "h9MQ2mu92MY", "hfPQA6SWGAQ", "HOy8f0Zd_pM",
    "IumbAb8q2dM", "Jkswaaud0Ip", "jUzDDGyPkXU", "k5Y_838nuGo", "kBXNQjPU2vI",
    "Kmk59DzHxCE", "kR50nphFMv0", "KX7qBM5a4EA", "l1rrFpDYqBM", "lXPQBPVc5Cw",
    "M5_Pbzv-AdA", "muG7jZRRHNI", "n7p1C8_Gmy4", "Njok_3wLg3U", "O0FJoVKTShU",
    "oDgXfBxXjek", "oJdxMkFjTfc", "oqNFEkIBrFI", "oZSAO_iEP1A", "PZ4vGkm3xBM",
    "q_eMDfcv7gA", "Q9qhyFwUFKk", "RPdBD_VXI8c", "S8pLSGHhKCE", "Ssqu4CQFYGA",
    "SuMSuP3F0mA", "SUvHx9xMnPM", "TDMO2VqAFHo", "TXUNoIOE9bI", "tYCxUCVrM8M",
    "UCjGzMKHvOA", "ug0hQ3D-HK8", "uSBd1uHOcco", "UtpSGfcXOJ8", "VWTajMJKkw4",
    "W2KBImq5M98", "wMbj6ajWbic", "wqfO4jFM37E", "X5mkWrRTNIE", "xlKFkm_bPeY",
    "xTKTpZUSuVk", "YmQ_tDVTJb8", "Z_KMlXGPBHo", "zXu3iJx0ThY", "zyts4JeB4Bk",
}
VALID_IDS = {
    "03bSnISJMiM", "0h-zjBukYpk", "1DmNV9C1hbY", "1iG0909rllw", "2WGyTLYerpo",
    "5W7Z1C_fDaE", "6Egk_28TtTM", "6_0THN4chvY", "73jzhE8R1TQ", "7JsX8y1ysxY",
}
TEST_IDS = {
    "8OtFthrtaJM", "8qrpnFRGt2A", "9c67fiY0wGQ", "9J25DZhivz8", "9qR7uwkblbs",
    "9T9Hf74oK10", "aiEXnCPZubE", "atnd_PF-Lbs", "Bfr499ggo-0", "BI97DNYfe5I",
    "bOL9jKpeJRs", "bvLlb-M3UXU", "c-xujHNnH8w", "CB8MA3qdT0k",
}


def read_transcript(video_id: str) -> dict:
    """读取 .annotprocessed 文件，返回 {clip_id: text}"""
    path = TRANSCRIPT_DIR / f"{video_id}.annotprocessed"
    result = {}
    if not path.exists():
        return result
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if "_DELIM_" in line:
                parts = line.split("_DELIM_", 1)
                clip_id = int(parts[0])
                text = parts[1].strip()
                result[clip_id] = text
    return result


def get_mode(video_id: str) -> str:
    if video_id in TRAIN_IDS:
        return "train"
    elif video_id in VALID_IDS:
        return "valid"
    elif video_id in TEST_IDS:
        return "test"
    return "train"  # 默认归 train


def main():
    # 从 Segmented 视频目录枚举所有片段
    all_segments = sorted(VIDEO_SEG_DIR.glob("*.mp4"))
    if not all_segments:
        raise FileNotFoundError(f"找不到视频片段: {VIDEO_SEG_DIR}")

    # 解析 video_id 和 clip_id
    records = []
    for seg in all_segments:
        name = seg.stem  # e.g. 03bSnISJMiM_1
        # 最后一个 _ 后面是 clip_id
        parts = name.rsplit("_", 1)
        if len(parts) != 2:
            continue
        video_id, clip_id_str = parts[0], parts[1]
        try:
            clip_id = int(clip_id_str)
        except ValueError:
            continue
        records.append((video_id, clip_id))

    print(f"共找到 {len(records)} 个视频片段")

    # 尝试从 .csd 文件读标签（如果存在）
    label_map = {}  # (video_id, clip_id) -> score
    csd_path = Path(CSD_LABEL_PATH)
    if csd_path.exists():
        print(f"读取标签文件: {csd_path}")
        with h5py.File(str(csd_path), "r") as f:
            for vid in f.keys():
                data = f[vid]["data"][()]       # shape: (N, 1) 情感分值
                intervals = f[vid]["intervals"][()]  # shape: (N, 2) 时间戳
                for i in range(len(data)):
                    score = float(data[i][0])
                    clip_num = i + 1
                    label_map[(vid, clip_num)] = score
        print(f"标签读取完成，共 {len(label_map)} 条")
    else:
        print(f"警告: 标签文件不存在 ({CSD_LABEL_PATH})")
        print("将以 None 填充标签列，特征提取仍可进行")

    # 读取所有转录文本
    print("读取转录文本...")
    transcripts = {}  # video_id -> {clip_id: text}
    unique_vids = set(r[0] for r in records)
    for vid in unique_vids:
        transcripts[vid] = read_transcript(vid)

    # 组装 DataFrame
    rows = []
    for video_id, clip_id in records:
        score = label_map.get((video_id, clip_id), None)
        if score is None:
            annotation = None
        else:
            annotation = "Positive" if score >= 0 else "Negative"
        text = transcripts.get(video_id, {}).get(clip_id, "")
        mode = get_mode(video_id)
        rows.append({
            "video_id": video_id,
            "clip_id": clip_id,
            "text": text,
            "label": score,
            "annotation": annotation,
            "mode": mode,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["video_id", "clip_id"]).reset_index(drop=True)
    df.to_csv(OUTPUT_META, index=False)

    print(f"\n生成完成: {OUTPUT_META}")
    print(f"总样本: {len(df)}")
    print(f"分布:\n{df['mode'].value_counts().to_string()}")
    if df['annotation'].notna().any():
        print(f"标签分布:\n{df['annotation'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
