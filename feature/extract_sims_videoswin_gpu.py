import argparse
import os
from typing import List, Tuple

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.models.video import Swin3D_S_Weights, swin3d_s


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract CH-SIMS v2.0 visual features with Swin3D (GPU required by default)."
    )
    parser.add_argument(
        "--meta-path",
        type=str,
        default="/mnt/f/dataset/CH-SIMS v2.0/ch-simsv2s/meta.csv",
        help="Path to meta.csv.",
    )
    parser.add_argument(
        "--raw-video-dir",
        type=str,
        default="/mnt/f/dataset/CH-SIMS v2.0/ch-simsv2s/Raw",
        help="Root directory for raw clips. Expected layout: Raw/video_id/clip_id.mp4",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="video_features_v2.h5",
        help="Output HDF5 file path.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames sampled per clip segment.",
    )
    parser.add_argument(
        "--num-clips",
        type=int,
        default=3,
        help="Number of temporal segments per sample; final feature is averaged over segments.",
    )
    parser.add_argument(
        "--decode-size",
        type=int,
        default=256,
        help="Frame resize size before model transforms.",
    )
    parser.add_argument(
        "--black-frame-threshold",
        type=float,
        default=0.7,
        help="Skip a segment if failed-frame ratio is above this value.",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="video_id",
        help="video id column in meta.csv.",
    )
    parser.add_argument(
        "--clip-col",
        type=str,
        default="clip_id",
        help="clip id column in meta.csv.",
    )
    parser.add_argument(
        "--use-fp16",
        action="store_true",
        help="Enable fp16 autocast during inference (GPU only).",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU fallback. By default, script exits when CUDA is unavailable.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Debug option: process only first N rows (0 means all).",
    )
    return parser.parse_args()


def build_sample_keys(video_ids: List[str]) -> List[str]:
    return [f"{vid.replace('/', '_')}_{i}" for i, vid in enumerate(video_ids)]


def sample_indices(total_frames: int, num_frames: int, clip_idx: int, num_clips: int) -> np.ndarray:
    start = int(round(total_frames * clip_idx / num_clips))
    end = int(round(total_frames * (clip_idx + 1) / num_clips))
    end = max(end, start + 1)
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

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None, 1.0

    indices = sample_indices(total_frames, num_frames, clip_idx, num_clips)
    frames = []
    fail_count = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            fail_count += 1
            frame = np.zeros((decode_size, decode_size, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (decode_size, decode_size), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)

    cap.release()
    return np.stack(frames, axis=0), fail_count / float(num_frames)


def extract_feature_from_frames(
    frames: np.ndarray,
    video_tf,
    model: nn.Module,
    device: torch.device,
    use_fp16: bool,
) -> Tuple[np.ndarray, str]:
    # (T, H, W, C) -> (T, C, H, W)
    video_tensor = torch.from_numpy(frames).float() / 255.0
    video_tensor = video_tensor.permute(0, 3, 1, 2)
    video_tensor = video_tf(video_tensor)  # -> (C, T, H, W)
    video_tensor = video_tensor.unsqueeze(0).to(device, non_blocking=True)

    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(device.type == "cuda" and use_fp16),
        ):
            feat = model(video_tensor)  # (1, 768)

    return feat.squeeze(0).float().cpu().numpy(), str(video_tensor.device)


def main():
    args = parse_args()

    if not os.path.exists(args.meta_path):
        raise FileNotFoundError(f"meta file not found: {args.meta_path}")
    if not os.path.isdir(args.raw_video_dir):
        raise FileNotFoundError(f"raw video directory not found: {args.raw_video_dir}")

    has_cuda = torch.cuda.is_available()
    if not has_cuda and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable. Use a CUDA-ready terminal or pass --allow-cpu.")

    device = torch.device("cuda:0" if has_cuda else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU fallback")

    df = pd.read_csv(args.meta_path, dtype={args.id_col: str, args.clip_col: str})
    if args.id_col not in df.columns or args.clip_col not in df.columns:
        raise ValueError(f"meta.csv must contain '{args.id_col}' and '{args.clip_col}' columns.")

    if args.max_samples > 0:
        df = df.head(args.max_samples).copy()

    video_ids = df[args.id_col].astype(str).tolist()
    clip_ids = df[args.clip_col].astype(str).tolist()
    sample_keys = build_sample_keys(video_ids)

    weights = Swin3D_S_Weights.DEFAULT
    video_tf = weights.transforms()  # Expects (T, C, H, W), returns (C, T, H, W)

    model = swin3d_s(weights=weights).to(device)
    model.head = nn.Identity()  # Output: (B, 768)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print(
        f"Start extracting visual features: samples={len(video_ids)}, "
        f"num_frames={args.num_frames}, num_clips={args.num_clips}"
    )

    error_logs = []
    written = 0
    fallback_used = 0

    with h5py.File(args.output_path, "w") as h5f:
        for i, (vid, cid, key) in enumerate(tqdm(zip(video_ids, clip_ids, sample_keys), total=len(video_ids))):
            video_path = os.path.join(args.raw_video_dir, vid, f"{cid}.mp4")
            if not os.path.exists(video_path):
                error_logs.append(f"{vid}/{cid}: File not found")
                continue

            clip_features = []
            segment_skipped = 0
            best_frames = None
            best_black_ratio = 2.0

            try:
                for clip_idx in range(args.num_clips):
                    frames, black_ratio = load_clip_frames(
                        video_path=video_path,
                        num_frames=args.num_frames,
                        clip_idx=clip_idx,
                        num_clips=args.num_clips,
                        decode_size=args.decode_size,
                    )

                    if frames is None or black_ratio > args.black_frame_threshold:
                        segment_skipped += 1
                        if frames is not None and black_ratio < best_black_ratio:
                            best_frames = frames
                            best_black_ratio = black_ratio
                        continue

                    if black_ratio < best_black_ratio:
                        best_frames = frames
                        best_black_ratio = black_ratio

                    feat_np, input_device = extract_feature_from_frames(
                        frames=frames,
                        video_tf=video_tf,
                        model=model,
                        device=device,
                        use_fp16=args.use_fp16,
                    )
                    clip_features.append(feat_np)

                if not clip_features:
                    if best_frames is None:
                        error_logs.append(f"{vid}/{cid}: all segments skipped ({segment_skipped}/{args.num_clips})")
                        continue
                    feat_np, input_device = extract_feature_from_frames(
                        frames=best_frames,
                        video_tf=video_tf,
                        model=model,
                        device=device,
                        use_fp16=args.use_fp16,
                    )
                    clip_features.append(feat_np)
                    fallback_used += 1

                h_v = np.mean(np.stack(clip_features, axis=0), axis=0)
                h5f.create_dataset(key, data=h_v)
                written += 1

                if i == 0:
                    print(
                        f"Sanity check: model_device={next(model.parameters()).device}, "
                        f"input_device={input_device}, feat_dim={h_v.shape[0]}"
                    )

            except Exception as exc:
                error_logs.append(f"{vid}/{cid}: {exc}")

    print(f"Done. Written features: {written}/{len(video_ids)}")
    print(f"Fallback-used samples: {fallback_used}")
    print(f"Output: {args.output_path}")
    if error_logs:
        print(f"Warnings: {len(error_logs)} failures. First 10:")
        for msg in error_logs[:10]:
            print(msg)


if __name__ == "__main__":
    main()
