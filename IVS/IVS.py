from collections import defaultdict
import os
from PIL import Image, ImageEnhance
import random

# 统计标签数量
filename = "list.txt"
label_count = defaultdict(int)

with open(filename, "r") as file:
    for line in file:
        label = int(line.strip().split(",")[-1])
        label_count[label] += 1

# 找出不同条件下的标签
labels_to_color_jitter_only = {label for label, count in label_count.items() if 800 <= count < 1600}
labels_to_color_jitter_and_rotate_crop = {label for label, count in label_count.items() if count < 800}

# 准备输出文件
output_file = "IVS.txt"
if not os.path.exists(output_file):
    with open(output_file, 'w') as f:
        pass  # 创建空文件

# 定义视频帧文件夹路径
video_frame_folder = "../raw_data/touch_and_go/video_frame"


# 颜色抖动函数
def apply_color_jitter(img):
    brightness_factor = random.uniform(0.8, 1.2)
    contrast_factor = random.uniform(0.8, 1.2)
    saturation_factor = random.uniform(0.8, 1.2)

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)

    return img


# 随机旋转和裁剪函数
def apply_random_rotate_crop(img, original_size):
    angle = random.randint(-30, 30)
    rotated_img = img.rotate(angle, expand=True)

    # 随机裁剪
    width, height = original_size
    left = random.randint(0, rotated_img.width - width)
    top = random.randint(0, rotated_img.height - height)
    right = left + width
    bottom = top + height

    cropped_img = rotated_img.crop((left, top, right, bottom))
    return cropped_img


# 处理图像并更新路径
with open(filename, "r") as infile, open(output_file, "a") as outfile:
    for line in infile:
        path, label_str = line.strip().split(',')
        label = int(label_str)

        # 添加video_frame/前缀到路径
        full_path = os.path.join(video_frame_folder, path)

        # 初始化一个列表来存储需要写入的新路径
        new_paths = []

        if label in labels_to_color_jitter_only or label in labels_to_color_jitter_and_rotate_crop:
            # 修改路径的第一个数字为7（颜色抖动）
            color_jitter_path = '7' + path[1:]
            color_jitter_full_path = os.path.join(video_frame_folder, color_jitter_path)
            color_jitter_dir = os.path.dirname(color_jitter_full_path)

            # 确保新目录存在
            os.makedirs(color_jitter_dir, exist_ok=True)

            # 打开原图
            img = Image.open(full_path)

            # 进行颜色抖动并保存图像
            color_jitter_img = apply_color_jitter(img)
            color_jitter_img.save(color_jitter_full_path)

            # 添加颜色抖动后的路径到列表
            new_paths.append(f"{color_jitter_path},{label}")

            if label in labels_to_color_jitter_and_rotate_crop:
                # 修改路径的第一个数字为9（旋转和裁剪）
                rotate_crop_path = '9' + path[1:]
                rotate_crop_full_path = os.path.join(video_frame_folder, rotate_crop_path)
                rotate_crop_dir = os.path.dirname(rotate_crop_full_path)

                # 确保新目录存在
                os.makedirs(rotate_crop_dir, exist_ok=True)

                # 获取原始图像尺寸
                original_size = img.size

                # 进行随机旋转和裁剪并保存图像
                rotate_crop_img = apply_random_rotate_crop(img, original_size)
                rotate_crop_img.save(rotate_crop_full_path)

                # 添加旋转和裁剪后的路径到列表
                new_paths.append(f"{rotate_crop_path},{label}")

        # 只写入处理过的文件路径
        if new_paths:  # 如果有处理过的文件，则写入
            for new_path in new_paths:
                outfile.write(new_path + '\n')
