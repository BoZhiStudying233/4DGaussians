# def extract_camera_info(file_path):
#     camera_info_lines = []
#     try:
#         with open(file_path, 'r') as file:
#             lines = file.readlines()
#             for i in range(0, len(lines), 2):  # 每两行一组，取第一行
#                 camera_info_lines.append(lines[i].strip())
#     except FileNotFoundError:
#         print(f"文件 {file_path} 未找到。")
#     return camera_info_lines

# # 示例使用
# file_path = '/home/bo70s/Desktop/underwater_dataset/turtle/Sparse/txt/images.txt'  # 替换为实际的文件路径
# camera_info = extract_camera_info(file_path)

# # 打印提取的相机信息
# for line in camera_info:
#     print(line)

# 如果你想将提取的信息保存到新文件
import argparse



def extract_camera_info(file_path):
    camera_info_lines = []
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for i in range(0, len(lines), 2):  # 每两行一组，取第一行
                camera_info_lines.append(lines[i].strip())
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    return camera_info_lines


# 创建ArgumentParser对象
parser = argparse.ArgumentParser(description='提取相机信息')
# 添加file_path参数
parser.add_argument('--file_path', type=str, help='输入文件路径')
# 添加output_file_path参数
parser.add_argument('--output_file_path', type=str, help='输出文件路径')

# 解析命令行参数
args = parser.parse_args()

# 使用解析后的参数
camera_info = extract_camera_info(args.file_path)

# 打印提取的相机信息
for line in camera_info:
    print(line)

# 如果你想将提取的信息保存到新文件
with open(args.output_file_path, 'w') as output_file:
    for line in camera_info:
        output_file.write(line + '\n')
    print(f"提取的相机信息已保存到 {args.output_file_path}") 
