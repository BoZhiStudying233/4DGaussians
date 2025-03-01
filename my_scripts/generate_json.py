import json
import numpy as np
import os
import argparse



def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """
    将四元数转换为旋转矩阵
    """
    # qw2 = qw * qw
    qx2 = qx * qx
    qy2 = qy * qy
    qz2 = qz * qz
    qxy = qx * qy
    qxz = qx * qz
    qyz = qy * qz
    qwx = qw * qx
    qwy = qw * qy
    qwz = qw * qz

    R = np.array([
        [1 - 2 * (qy2 + qz2), 2 * (qxy - qwz), 2 * (qxz + qwy)],
        [2 * (qxy + qwz), 1 - 2 * (qx2 + qz2), 2 * (qyz - qwx)],
        [2 * (qxz - qwy), 2 * (qyz + qwx), 1 - 2 * (qx2 + qy2)]
    ])
    print(R)
    # R = np.array([
    #     [qw2 + qx2 - qy2 - qz2, 2 * (qxy - qwz), 2 * (qxz + qwy)],
    #     [2 * (qxy + qwz), qw2 - qx2 + qy2 - qz2, 2 * (qyz - qwx)],
    #     [2 * (qxz - qwy), 2 * (qyz + qwx), qw2 - qx2 - qy2 + qz2]
    # ])
    return R.tolist()

def process_txt_file(txt_file_path, first_order_radial_distortion, output_dir,     focal_length, principal_point, image_size, pixel_aspect_ratio, skew):
    # 定义定值
    # focal_length = 3521.6785829663572
    # principal_point = [1200, 750]
    # image_size = [2400, 1500]
    # pixel_aspect_ratio = 1
    # skew = 0

    # 根据一阶径向畸变系数计算径向畸变和切向畸变
    radial_distortion = [first_order_radial_distortion, 0.0, 0.0]
    tangential_distortion = [0.0, 0.0]

    with open(txt_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            image_id = parts[0]
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            image_name = parts[-1]
            image_num = image_name.split('.')[0]

            # 计算旋转矩阵
            orientation = quaternion_to_rotation_matrix(qw, qx, qy, qz)
            position = [tx, ty, tz]

            # 构建 JSON 数据
            json_data = {
                "orientation": orientation,
                "position": position,
                "focal_length": focal_length,
                "principal_point": principal_point,
                "skew": skew,
                "pixel_aspect_ratio": pixel_aspect_ratio,
                "radial_distortion": radial_distortion,
                "tangential_distortion": tangential_distortion,
                "image_size": image_size
            }

            image_num = image_num.split('/')[-1]

            # 生成 JSON 文件名称
            json_file_name = f"{int(image_num):06d}.json"

            json_file_path = os.path.join(output_dir, json_file_name)
            print(f"正在生成 {json_file_path}")
            # 保存为 JSON 文件
            with open(json_file_path, 'w') as json_file:
                json.dump(json_data, json_file, indent=2)
            # print(f"已保存 {json_file_name}")


def read_camera_parameters(file_path):
    """
    从camera.txt文件中读取相机参数。

    参数:
        file_path (str): camera.txt文件的路径。

    返回:
        dict: 包含相机参数的字典。
    """
    with open(file_path, 'r') as file:
        # 读取第一行
        line = file.readline().strip()
        
        # 按空格分割参数
        params = line.split()
        
        # 解析参数
        camera_id = int(params[0])
        camera_model = params[1]
        width = int(params[2])
        height = int(params[3])
        focal_length = float(params[4])
        cx = float(params[5])
        cy = float(params[6])
        distortion = float(params[7])
        
        # 返回解析后的参数
        return {
            "camera_id": camera_id,
            "camera_model": camera_model,
            "width": width,
            "height": height,
            "focal_length": focal_length,
            "cx": cx,
            "cy": cy,
            "distortion": distortion
        }


if __name__ == "__main__":
    # txt_file_path = 'extracted_camera_info.txt'  # 替换为实际的 txt 文件路径
    # output_dir = 'data/my_data/camera/'
    # first_order_radial_distortion = -0.011731776198732008



    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="从camera.txt文件中读取相机参数")
    parser.add_argument("--images_file_path", type=str, help="images.txt文件的路径")
    parser.add_argument("--cameras_file_path", type=str, help="camera.txt文件夹")
    # 解析命令行参数
    args = parser.parse_args()
    
    # 读取相机参数
    camera_params = read_camera_parameters(args.cameras_file_path)
    
    # 打印结果
    print("相机参数:")
    for key, value in camera_params.items():
        print(f"{key}: {value}")

    first_order_radial_distortion = camera_params['distortion']
    focal_length = camera_params['focal_length']
    principal_point = [camera_params['cx'], camera_params['cy']]
    image_size = [camera_params['width'], camera_params['height']]
    pixel_aspect_ratio = 1
    skew = 0
    
    output_dir = args.cameras_file_path.replace('cameras.txt', 'camera')
    output_dir = output_dir.replace('/Sparse/0', '')
    print(f"输出路径: {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    process_txt_file(args.images_file_path, first_order_radial_distortion, output_dir,    focal_length, principal_point, image_size, pixel_aspect_ratio, skew)