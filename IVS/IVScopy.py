import os
import shutil

# 读取文件路径和目标文件夹
base_path = '../raw_data/touch_and_go/gelsight_frame'

with open('IVS.txt', 'r') as file:
    for line in file:
        # 去除换行符并分割字符串
        parts = line.strip().split(',')
        if len(parts) != 2:
            print(f"Line format is incorrect: {line}")
            continue

        image_path, label = parts
        # 分离出文件夹名和文件名
        folder_name, file_name = image_path.split('/')

        # 创建目标文件夹（如果不存在）
        target_folder = os.path.join(base_path, folder_name)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
            print(f"Created directory: {target_folder}")

        # 构建源文件路径和目标文件路径
        source_path = '2' + image_path[1:]
        source_file = os.path.join(base_path,source_path)
        target_file = os.path.join(target_folder, file_name)

        # 检查源文件是否存在，然后复制
        if os.path.exists(source_file):
            shutil.copy2(source_file, target_file)  # copy2 会保留元数据
            print(f"Copied {source_file} to {target_file}")
        else:
            print(f"Source file does not exist: {source_file}")

print("Operation completed.")
