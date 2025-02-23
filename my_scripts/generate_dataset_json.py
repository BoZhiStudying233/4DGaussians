import json
import argparse

def generate_second_json(num_images, output_path):
    # 生成第二个JSON的数据
    json_data = {}
    for i in range(num_images):
        image_id = f"{i:06d}"  # 生成6位数的ID
        json_data[image_id] = {
            "time_id": i,
            "warp_id": i,
            "appearance_id": i,
            "camera_id": 0
        }

    # 将JSON数据写入输出文件
    with open(output_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=2)

    print(f"第二个JSON文件已生成，包含 {num_images} 张图片，保存到 {output_path}。")

def generate_first_json(input_path, output_path):
    # 读取输入文件并计算行数
    with open(input_path, 'r') as file:
        lines = file.readlines()
        num_images = len(lines)

    # 生成ids列表，格式为6位数的字符串
    ids = [f"{i:06d}" for i in range(num_images)]

    # 构建JSON数据
    json_data = {
        "count": num_images,
        "num_exemplars": num_images,
        "ids": ids,
        "train_ids": ids,
        "val_ids": []
    }

    # 将JSON数据写入输出文件
    with open(output_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=2)

    print(f"JSON文件已生成，包含 {num_images} 张图片，保存到 {output_path}。")
    return num_images
if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="根据输入文件的行数生成JSON文件。")
    parser.add_argument("--input", help="输入文件路径（例如：new_images.txt）")
    parser.add_argument("--output", help="输出文件路径（例如：output.json）")
    args = parser.parse_args()

    # 调用主函数
    num_images = generate_first_json(args.input, args.output)
    output_path = args.output.replace("dataset.json", "metadata.json")
    generate_second_json(num_images, output_path)