import os
import torch
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import warnings

# 过滤一些不必要的警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 配置参数
# ==========================================
META_PATH = "/mnt/f/dataset/CH-SIMS v2.0/ch-simsv2s/meta.csv"
RAW_VIDEO_DIR = "/mnt/f/dataset/CH-SIMS v2.0/ch-simsv2s/Raw"
OUTPUT_PATH = "video_features_faces_v3.h5"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 【重要参数】抽帧间隔：每隔几帧取一张。
# 减小此值会增加计算量但捕捉更细致，增大此值速度更快。建议 5-10。
FRAME_INTERVAL = 5 

# 批量处理大小，防止显存爆炸
BATCH_SIZE = 32

def init_models(device):
    """初始化人脸检测器和特征提取器"""
    print(f"Initializing models on {device}...")
    
    # 1. MTCNN 人脸检测器
    # keep_all=False: 只保留最有可能是人脸的那一个
    # select_largest=True: 选择最大的人脸
    # device=device: 在 GPU 上运行检测
    mtcnn = MTCNN(keep_all=False, select_largest=True, device=device)
    
    # 2. InceptionResnetV1 特征提取器 (预训练于 VGGFace2)
    # classify=False: 我们只需要最后全连接层之前的特征 (512维)
    resnet = InceptionResnetV1(pretrained='vggface2', classify=False).to(device)
    resnet.eval()
    
    return mtcnn, resnet

def process_video(video_path, mtcnn, resnet, device):
    """处理单个视频：抽帧 -> 检测人脸 -> 提取特征"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames_to_process = []
    frame_count = 0
    
    # --- 第一步：稀疏抽帧 ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 每隔 FRAME_INTERVAL 帧取一帧
        if frame_count % FRAME_INTERVAL == 0:
            # OpenCV 读入是 BGR，转为 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_to_process.append(frame_rgb)
        frame_count += 1
    cap.release()

    if not frames_to_process:
        return None
        
    # --- 第二步：批量人脸检测与裁切 (MTCNN) ---
    # MTCNN 可以直接接收一批图片，返回裁切好并标准化的张量 (Batch, 3, 160, 160)
    # 如果某帧没检测到人脸，对应位置会是 None
    
    # 将帧列表切分为小批次以节省显存
    face_tensors = []
    for i in range(0, len(frames_to_process), BATCH_SIZE):
        batch_frames = frames_to_process[i:i+BATCH_SIZE]
        # MTCNN 返回检测到的人脸张量列表，没检测到的为 None
        batch_faces = mtcnn(batch_frames)
        
        # 过滤掉没检测到人脸的帧 (None)
        valid_faces = [f for f in batch_faces if f is not None]
        if valid_faces:
            # 堆叠成一个张量
            face_tensors.append(torch.stack(valid_faces).to(device))
            
    if not face_tensors:
        # 整个视频都没检测到人脸
        return None
        
    # 合并所有批次的人脸张量: (Total_Valid_Frames, 3, 160, 160)
    all_faces_tensor = torch.cat(face_tensors, dim=0)

    # --- 第三步：特征提取 (ResNet) ---
    with torch.no_grad():
        # 提取特征: (Total_Valid_Frames, 512)
        features = resnet(all_faces_tensor)
    
    # --- 第四步：时序聚合 (Max Pooling) ---
    # 沿着时间维度 (dim=0) 取最大值，捕捉最显著的情感特征
    # 最终形状: (512,)
    final_feature = torch.max(features, dim=0).values.cpu().numpy()
    
    return final_feature

def main():
    # 加载数据索引
    df = pd.read_csv(META_PATH, dtype={"video_id": str, "clip_id": str})
    
    # 初始化模型
    mtcnn, resnet = init_models(DEVICE)

    print(f"Starting facial feature extraction (Scheme A)...")
    print(f"Output shape will be (512,). Frame interval: {FRAME_INTERVAL}")
    
    error_logs = []
    success_count = 0

    with h5py.File(OUTPUT_PATH, 'w') as h5f:
        for i, row in tqdm(df.iterrows(), total=len(df)):
            vid = str(row['video_id'])
            cid = str(row['clip_id'])
            key = f"{vid.replace('/', '_')}_{i}"
            video_path = os.path.join(RAW_VIDEO_DIR, vid, f"{cid}.mp4")
            
            if not os.path.exists(video_path):
                error_logs.append(f"{vid}/{cid}: File not found")
                continue
                
            try:
                # 核心处理函数
                feature_vector = process_video(video_path, mtcnn, resnet, DEVICE)
                
                if feature_vector is None:
                    error_logs.append(f"{vid}/{cid}: No faces detected in video")
                    continue
                
                # 写入 H5
                h5f.create_dataset(key, data=feature_vector)
                success_count += 1
                
            except Exception as e:
                error_logs.append(f"{vid}/{cid}: Error - {str(e)}")
                # 显存清理，防止 OOM 影响下一个视频
                torch.cuda.empty_cache()

    print(f"\n提取完成！")
    print(f"成功提取: {success_count}/{len(df)}")
    print(f"特征已保存至: {OUTPUT_PATH}")
    
    if error_logs:
        print(f"\n警告: 有 {len(error_logs)} 个视频处理失败或未检测到人脸。")
        print("失败样本示例 (前 10 个):")
        for log in error_logs[:10]:
            print(log)
        print("(这些失败的样本将在训练 Dataloader 中被自动过滤)")

if __name__ == "__main__":
    # Windows 下多进程可能需要这一行，Linux 不需要但加上无妨
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()