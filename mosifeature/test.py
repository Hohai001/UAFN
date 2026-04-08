import os
import requests
from mmsdk import mmdatasdk

# 1. 强行指定下载目标和文件名
target_dir = './mosi_data_force'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 标注文件的官方直链 (这是 MOSI 最核心的文件)
label_url = "http://immortal.multicomp.cs.cmu.edu/CMU-MOSI/labels/CMU_MOSI_Opinion_Labels.csd"
label_file = os.path.join(target_dir, "CMU_MOSI_Opinion_Labels.csd")

# 2. 强行手动下载 (避开 SDK 的 Bug)
print(f"正在强行下载标注文件至: {os.path.abspath(label_file)}")
try:
    # 如果你依然有网络问题，这里会报错，但至少我们知道是网络原因
    response = requests.get(label_url, stream=True, timeout=30)
    if response.status_code == 200:
        with open(label_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ 标注文件下载成功！")
    else:
        print(f"❌ 下载失败，服务器返回代码: {response.status_code}")
except Exception as e:
    print(f"❌ 强行下载时崩溃: {e}")

# 3. 让 SDK 加载这个已经下好的文件
print("\n正在尝试让 SDK 加载本地文件...")
try:
    # 当文件夹里已经有文件时，mmdataset 就不再尝试联网下载
    dataset = mmdatasdk.mmdataset(target_dir)
    
    print("-" * 30)
    print("SDK 内存中的内容:", dataset.computational_sequences.keys())
    
    # 验证数据是否真的进去了
    if 'CMU_MOSI_Opinion_Labels' in dataset.computational_sequences:
        print("🎉 恭喜！数据已成功加载进内存，你可以开始对齐了。")
    else:
        print("⚠️ SDK 依然没认出这个文件，请检查文件名是否准确。")
except Exception as e:
    print(f"❌ SDK 加载失败: {e}")