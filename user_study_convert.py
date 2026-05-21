import os
import json
import numpy as np


def invert_rotation_matrix(rotation_matrix):
    """
    对旋转矩阵求逆
    :param rotation_matrix: 3x3的旋转矩阵
    :return: 求逆后的旋转矩阵
    """
    # 不将结果转换为列表，保持为 numpy 数组
    return np.linalg.inv(rotation_matrix)


def invert_translation_vector(rotation_matrix, translation_vector):
    """
    对平移向量求逆
    :param rotation_matrix: 3x3的旋转矩阵
    :param translation_vector: 平移向量
    :return: 求逆后的平移向量
    """
    inverted_rotation_matrix = np.linalg.inv(rotation_matrix)
    inverted_translation_vector = -np.dot(inverted_rotation_matrix, translation_vector)
    return inverted_translation_vector


def convert_json_file(file_path):
    """
    该函数用于将单个JSON文件中的数据转换为目标格式，并对R和T求逆
    :param file_path: 单个JSON文件的路径
    :return: 转换后的数据
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    orientation = np.array(data["orientation"])
    position = np.array(data["position"])

    # 对旋转矩阵求逆
    inverted_orientation = invert_rotation_matrix(orientation)

    # 对平移向量求逆
    inverted_position = invert_translation_vector(orientation, position)

    # 构建4x4的camera_to_world矩阵
    camera_to_world = [
        inverted_orientation[0][0], inverted_orientation[0][1], inverted_orientation[0][2], inverted_position[0],
        inverted_orientation[1][0], inverted_orientation[1][1], inverted_orientation[1][2], inverted_position[1],
        inverted_orientation[2][0], inverted_orientation[2][1], inverted_orientation[2][2], inverted_position[2],
        0.0, 0.0, 0.0, 1.0
    ]

    fov = 75.0
    aspect = data["image_size"][0] / data["image_size"][1]

    return {
        "camera_to_world": camera_to_world,
        "fov": fov,
        "aspect": aspect
    }


def convert_all_json_files(folder_path, output_file_path):
    """
    该函数用于遍历指定文件夹中的所有JSON文件，并将转换后的数据拼接成一个新的JSON文件
    :param folder_path: 包含JSON文件的文件夹路径
    :param output_file_path: 输出的JSON文件路径
    """
    all_converted_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            converted_data = convert_json_file(file_path)
            all_converted_data.append(converted_data)

    with open(output_file_path, 'w') as output_file:
        json.dump(all_converted_data, output_file, indent=4)


# 请将 'your_folder_path' 替换为实际包含JSON文件的文件夹路径
folder_path = '/home/dzb/4DGaussians_old1/data/my_data/DRUVA_1_294/camera'
# 请将 'output.json' 替换为你希望保存的输出文件的路径和文件名
output_file_path = '/home/dzb/water-splatting/outputs/unnamed/water-splatting/A1/trac.json'
convert_all_json_files(folder_path, output_file_path)