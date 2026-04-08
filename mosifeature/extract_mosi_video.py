"""
CMU-MOSI 视频特征提取
使用 Video Swin Transformer (swin3d_s)，输出 768 维
直接读取 Video/Segmented/ 下的分段 .mp4 文件

前置条件：先运行 make_mosi_meta.py 生成 mosi_meta.csv

输出：video_features_mosi.h5
  key 格式：{video_id}_{clip_id}  (e.g. "03bSnISJMiM_1")

用法：
  python extract_mosi_video.py
  python extract_mosi_video.py --use-fp16          # GPU fp16 加速
  python extract_mosi_video.py --max-samples 10    # 调试模式
"""

import argparse
import os
from typing import List, Tuple

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torchvision.models.video import Swin3D_S_Weights, swin3d_s

# ==========================================
# 配置路径
# ==========================================
META_PATH      = Path(__file__).parent / "label.csv"
VIDEO_SEG_DIR  = Path("/mnt/f/dataset/CMU/CMU-MOSI/Video/Segmented")
OUTPUT_PATH    = Path(__file__).parent / "video_features_mosi.h5"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-frames",            type=int,   default=16)
    parser.add_argument("--num-clips",             type=int,   default=3)
    parser.add_argument("--decode-size",           type=int,   default=256)
    parser.add_argument("--black-frame-threshold", type=float, default=0.7)
    parser.add_argument("--use-fp16",              action="store_true")
    parser.add_argument("--allow-cpu",             action="store_true")
    parser.add_argument("--max-samples",           type=int,   default=0)
    return parser.parse_args()


def sample_indices(total: int, num_frames: int, clip_idx: int, num_clips: int) -> np.ndarray:
    start = int(round(total * clip_idx / num_clips))
    end   = int(round(total * (clip_idx + 1) / num_clips))
    end   = max(end, start + 1)
    return np.linspace(start, end - 1, num_frames, dtype=int)


def load_clip_frames(
    video_path: str,
    num_frames: int,
    clip_idx: int,
    num_clips: int,
    decode_size: int,
) -> Tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None, 1.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None, 1.0

    indices = sample_indices(total, num_frames, clip_idx, num_clips)
    frames, fail = [], 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            fail += 1
            frame = np.zeros((decode_size, decode_size, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (decode_size, decode_size), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    cap.release()
    return np.stack(frames, axis=0), fail / float(num_frames)


def extract_feature(frames, video_tf, model, device, use_fp16) -> np.ndarray:
    video_tensor = torch.from_numpy(frames).float() / 255.0
    video_tensor = video_tensor.permute(0, 3, 1, 2)    # (T, C, H, W)
    video_tensor = video_tf(video_tensor)               # (C, T, H, W)
    video_tensor = video_tensor.unsqueeze(0).to(device, non_blocking=True)

    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(device.type == "cuda" and use_fp16),
        ):
            feat = model(video_tensor)  # (1, 768)
    return feat.squeeze(0).float().cpu().numpy()


def main():
    args = parse_args()

    has_cuda = torch.cuda.is_available()
    if not has_cuda and not args.allow_cpu:
        raise RuntimeError("CUDA 不可用，请在 GPU 环境运行，或加 --allow-cpu")
    device = torch.device("cuda:0" if has_cuda else "cpu")
    if has_cuda:
        torch.backends.cudnn.benchmark = True
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用 CPU")

    df = pd.read_csv(META_PATH, dtype={"video_id": str, "clip_id": int})
    if args.max_samples > 0:
        df = df.head(args.max_samples).copy()
    print(f"共 {len(df)} 个样本")

    weights  = Swin3D_S_Weights.DEFAULT
    video_tf = weights.transforms()
    model    = swin3d_s(weights=weights).to(device)
    model.head = nn.Identity()  # 输出 768 维
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    error_logs = []
    written    = 0
    fallback   = 0

    with h5py.File(OUTPUT_PATH, "w") as h5f:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            vid = str(row["video_id"])
            cid = int(row["clip_id"])
            key = f"{vid}_{cid}"
            video_path = str(VIDEO_SEG_DIR / f"{vid}_{cid}.mp4")

            if not os.path.exists(video_path):
                error_logs.append(f"{key}: 文件不存在")
                continue

            clip_features = []
            best_frames, best_ratio = None, 2.0

            try:
                for ci in range(args.num_clips):
                    frames, ratio = load_clip_frames(
                        video_path, args.num_frames, ci, args.num_clips, args.decode_size
                    )
                    if frames is not None and ratio < best_ratio:
                        best_frames, best_ratio = frames, ratio
                    if frames is None or ratio > args.black_frame_threshold:
                        continue
                    clip_features.append(extract_feature(frames, video_tf, model, device, args.use_fp16))

                if not clip_features:
                    if best_frames is None:
                        error_logs.append(f"{key}: 所有分段均跳过")
                        continue
                    clip_features.append(extract_feature(best_frames, video_tf, model, device, args.use_fp16))
                    fallback += 1

                h_v = np.mean(np.stack(clip_features, axis=0), axis=0)
                h5f.create_dataset(key, data=h_v)
                written += 1

            except Exception as e:
                error_logs.append(f"{key}: {e}")

    print(f"\n完成！写入: {written}/{len(df)}，fallback: {fallback}，输出: {OUTPUT_PATH}")
    if error_logs:
        print(f"失败 {len(error_logs)} 个，前5条:")
        for msg in error_logs[:5]:
            print(msg)


if __name__ == "__main__":
    main()
