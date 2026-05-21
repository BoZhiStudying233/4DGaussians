import json
import numpy as np
from scipy.spatial.transform import Rotation
import random
import os
import shutil

# 读取 JSON 文件
def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


# 将旋转矩阵转换为欧拉角
def rotation_matrix_to_euler(rotation_matrix):
    r = Rotation.from_matrix(rotation_matrix)
    euler_angles = r.as_euler('zyx', degrees=True)
    return euler_angles


# 计算两个视角之间的距离（这里简单使用欧拉角的欧氏距离作为示例，你可以根据实际需求调整）
def calculate_view_distance(view1, view2):
    euler1 = rotation_matrix_to_euler(np.array(view1["orientation"]))
    euler2 = rotation_matrix_to_euler(np.array(view2["orientation"]))
    position1 = np.array(view1["position"])
    position2 = np.array(view2["position"])
    angle_distance = np.linalg.norm(euler1 - euler2)
    position_distance = np.linalg.norm(position1 - position2)
    return np.sqrt(angle_distance ** 2 + position_distance ** 2)


# 读取多个 JSON 文件并分析角度和位置范围
def analyze_views(json_files):
    views = []
    azimuths = []
    elevations = []
    positions_x = []
    positions_y = []
    positions_z = []

    for i, file in enumerate(json_files):
        if i % 100 == 0:  # 每处理 100 个文件打印一次进度
            print(f"Processing file {i}/{len(json_files)}")
        data = read_json(file)
        views.append(data)

        rotation_matrix = np.array(data["orientation"])
        euler_angles = rotation_matrix_to_euler(rotation_matrix)
        azimuths.append(euler_angles[0])
        elevations.append(euler_angles[1])

        position = np.array(data["position"])
        positions_x.append(position[0])
        positions_y.append(position[1])
        positions_z.append(position[2])

    azimuth_mean = np.mean(azimuths)
    azimuth_std = np.std(azimuths)
    elevation_mean = np.mean(elevations)
    elevation_std = np.std(elevations)
    position_x_mean = np.mean(positions_x)
    position_x_std = np.std(positions_x)
    position_y_mean = np.mean(positions_y)
    position_y_std = np.std(positions_y)
    position_z_mean = np.mean(positions_z)
    position_z_std = np.std(positions_z)

    return views, {
        "azimuth_mean": azimuth_mean,
        "azimuth_std": azimuth_std,
        "elevation_mean": elevation_mean,
        "elevation_std": elevation_std,
        "position_x_mean": position_x_mean,
        "position_x_std": position_x_std,
        "position_y_mean": position_y_mean,
        "position_y_std": position_y_std,
        "position_z_mean": position_z_mean,
        "position_z_std": position_z_std
    }


# 生成新视角，使其与数据集中最近视角的距离不大于某个值
def generate_new_view(views, stats, max_distance=1.0, max_attempts=1000):
    attempt = 0
    while attempt < max_attempts:
        # 根据已有视角的统计信息调整随机生成范围
        azimuth = random.gauss(stats["azimuth_mean"], stats["azimuth_std"])
        elevation = random.gauss(stats["elevation_mean"], stats["elevation_std"])
        x = random.gauss(stats["position_x_mean"], stats["position_x_std"])
        y = random.gauss(stats["position_y_mean"], stats["position_y_std"])
        z = random.gauss(stats["position_z_mean"], stats["position_z_std"])

        # 确保角度在合理范围内
        azimuth = np.clip(azimuth, 0, 360)
        elevation = np.clip(elevation, -90, 90)

        # 将新的角度转换为旋转矩阵
        r_new = Rotation.from_euler('zyx', [azimuth, elevation, 0], degrees=True)
        R_new = r_new.as_matrix()

        new_view = {
            "orientation": R_new.tolist(),
            "position": [x, y, z],
            # 这里假设其他参数保持不变
            "focal_length": 1246.3636064553416,
            "principal_point": [960.0, 540.0],
            "skew": 0,
            "pixel_aspect_ratio": 1,
            "radial_distortion": [-0.03755372322307607, 0.0, 0.0],
            "tangential_distortion": [0.0, 0.0],
            "image_size": [1920, 1080]
        }

        # 计算新视角与所有已有视角的距离
        distances = [calculate_view_distance(new_view, view) for view in views]
        min_distance = min(distances)

        if min_distance <= max_distance:
            return new_view

        attempt += 1
        print(f"Attempt {attempt}: Generated view with min distance {min_distance:.2f}")
    print(f"Warning: Reached maximum number of attempts ({max_attempts}) without finding a valid view.")
    return None


# 保存新视角为 JSON 文件
def save_new_view_to_json(view, output_folder, index):
    file_name = os.path.join(output_folder, f"{index:06d}.json")
    with open(file_name, 'w') as file:
        json.dump(view, file, indent=4)

def save_new_view_to_json(view, output_folder, index):
    file_name = os.path.join(output_folder, f"{index:06d}.json")
    with open(file_name, 'w') as file:
        json.dump(view, file, indent=4)

# 比对两个文件夹并移动文件
def move_missing_files(source_folder, dest_folder):
    source_files = set(os.listdir(source_folder))
    dest_files = set(os.listdir(dest_folder))
    missing_files = source_files - dest_files

    for file in missing_files:
        if file.endswith('.json'):
            source_path = os.path.join(source_folder, file)
            dest_path = os.path.join(dest_folder, file)
            shutil.move(source_path, dest_path)
            print(f"Moved {file} from {source_folder} to {dest_folder}")
def get_max_index(output_folder):
    if not os.path.exists(output_folder):
        return -1
    max_index = -1
    for file in os.listdir(output_folder):
        if file.endswith('.json'):
            try:
                index = int(os.path.splitext(file)[0])
                max_index = max(max_index, index)
            except ValueError:
                continue
    return max_index



# JSON 文件所在文件夹路径
json_folder_path = '/home/dzb/4DGaussians_old1/data/my_data/A11/camera_ori'

# 输出新视角 JSON 文件的文件夹路径
output_folder = '/home/dzb/4DGaussians_old1/data/my_data/A11/camera'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)



# 获取文件夹内所有 JSON 文件路径
json_files = [os.path.join(json_folder_path, f) for f in os.listdir(json_folder_path) if f.endswith('.json')]


# 计算每个部分的长度
n = len(json_files)
part_size = n // 3
remainder = n % 3

# 划分列表
parts = [json_files[:part_size + (1 if remainder > 0 else 0)], json_files[part_size + (1 if remainder > 0 else 0):part_size * 2 + (1 if remainder > 1 else 0)], json_files[part_size * 2 + (1 if remainder > 1 else 0):]]
 

for (i, part) in enumerate(parts):
    print(f"Part {i + 1}: {len(part)} files")
    # 分析视角范围
    views, stats = analyze_views(json_files)

    # 生成并保存多个新视角
    num_new_views = 50  # 你可以修改这个数字来控制生成的新视角数量
    for i in range(num_new_views):
        new_view = generate_new_view(views, stats, max_distance=10.0)  # 这里的 max_distance 可以根据需要调整
        if new_view is not None:
            save_new_view_to_json(new_view, output_folder, get_max_index(output_folder) + 1)

    print(f"{num_new_views} new views have been saved to {output_folder}.")

print("Moving missing files...")
move_missing_files(json_folder_path, output_folder)
