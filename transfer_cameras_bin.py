import numpy as np
import json
import pycolmap
import json



def transfer_npy_to_json():
    # 定义文件路径
    npy_file_path = '/home/bo70s/Downloads/underwater_dataset/turtle/poses_bounds.npy'
    json_file_path = 'poses_bounds.json'

    # 读取.npy 文件
    data = np.load(npy_file_path)

    # 将 NumPy 数组转换为 Python 列表
    data_list = data.tolist()

    # 创建一个字典来存储数据
    data_dict = {
        "poses_bounds": data_list
    }

    # 将数据保存为 JSON 文件
    with open(json_file_path, 'w') as json_file:
        json.dump(data_dict, json_file, indent=4)

    print(f"数据已保存到 {json_file_path}")



def convert_camera_bin_to_json():
    """
    将 camera.bin 文件转换为 JSON 格式
    :param bin_file_path: camera.bin 文件的路径
    :param json_file_path: 要保存的 JSON 文件的路径
    """

    bin_file_path = "/home/bo70s/Downloads/underwater_dataset/turtle/sparse/0/camera.bin"
    json_file_path = "camera.json"
    try:
        # 读取 camera.bin 文件
        cameras = pycolmap.read_cameras_binary(bin_file_path)
        camera_list = []

        # 遍历每个相机
        for camera_id, camera in cameras.items():
            camera_info = {
                "camera_id": camera_id,
                "model": camera.model_name(),
                "width": camera.width,
                "height": camera.height,
                "parameters": camera.params.tolist()
            }
            camera_list.append(camera_info)

        # 创建包含所有相机信息的字典
        data = {
            "cameras": camera_list
        }

        # 将字典保存为 JSON 文件
        with open(json_file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

        print(f"成功将 {bin_file_path} 转换为 {json_file_path}")
    except FileNotFoundError:
        print(f"未找到文件: {bin_file_path}")
    except Exception as e:
        print(f"转换过程中出现错误: {e}")

# 示例使用
if __name__ == "__main__":
    convert_camera_bin_to_json()