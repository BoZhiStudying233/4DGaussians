import shutil
import os

# 源文件夹路径
source_folder = "/home/dzb/4DGaussians_old/data/my_data/composite1/camera_ori"
# 目标文件夹路径
target_folder = "/home/dzb/4DGaussians_old/data/my_data/composite1/camera"

# 确保目标文件夹存在
os.makedirs(target_folder, exist_ok=True)

# 获取源文件夹中的第一个 JSON 文件
source_files = [f for f in os.listdir(source_folder) if f.endswith('.json')]
if source_files:
    first_json_file = os.path.join(source_folder, source_files[0])
    # 复制 100 个文件
    for i in range(100):
        target_file = os.path.join(target_folder, f"{i:06d}.json")
        shutil.copy2(first_json_file, target_file)
    print("文件复制完成")
else:
    print("源文件夹中没有 JSON 文件")