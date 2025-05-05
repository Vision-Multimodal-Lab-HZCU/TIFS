import os
import torch
from transformers import VideoMAEImageProcessor, VideoMAEModel
import numpy as np
from PIL import Image

# 设置CUDA设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 加载预训练模型和处理器
image_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").cuda().eval()

# 数据集路径配置
frames_directory = f'../raw_data/touch_and_go/video_frame/'
save_file_path = '../data/touch_pretrained_feature_dict.npy'

# 检查并处理文件列表
video_frames = []
frame_count = 0
batch_size = 16
feature_dict = {}

def process_batch(first_frame_name, frames, features_dict):
    """处理一批次的帧"""
    inputs = image_processor(frames, return_tensors='pt')['pixel_values'].cuda()
    with torch.no_grad():
        features = model(inputs).last_hidden_state
    features_np = features.squeeze(dim=0).detach().cpu().numpy()
    features_dict[first_frame_name] = features_np
    print("当前批次提取完成")

with open('genfea.txt', 'r') as file:
    for line in file:
        frame_name = line.strip()
        full_frame_path = os.path.join(frames_directory, frame_name)
        if not os.path.exists(full_frame_path):
            frames_directory = f'../raw_data/visgel/images/touch/'
            full_frame_path = os.path.join(frames_directory, frame_name)
        # 第一帧处理
        if len(video_frames) == 0:
            print(f"开始提取 {frame_name}")
            initial_frame_name = frame_name

        video_frames.append(Image.open(full_frame_path))
        frame_count += 1
        print(f"{frame_count}: {frame_name}")

        # 批量处理帧
        if len(video_frames) == batch_size:
            process_batch(initial_frame_name, video_frames, feature_dict)
            video_frames.clear()

# 保存特征字典
np.save(save_file_path, feature_dict)
print("所有帧提取完成")
