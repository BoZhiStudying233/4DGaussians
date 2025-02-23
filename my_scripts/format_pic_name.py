import os
import re
import argparse



def pad_numbers_in_filenames(folder_path):
    # 支持的图片文件扩展名
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_extension = os.path.splitext(filename)[1].lower()
        # 检查文件是否为图片文件
        if file_extension in image_extensions:
            # 使用正则表达式查找文件名中的数字部分
            match = re.search(r'\d+', filename)
            if match:
                number_part = match.group()
                # 将数字部分补零到六位
                padded_number = number_part.zfill(6)
                # 替换文件名中的数字部分
                new_filename = re.sub(r'\d+', padded_number, filename)
                old_file_path = os.path.join(folder_path, filename)
                new_file_path = os.path.join(folder_path, new_filename)
                # 重命名文件
                os.rename(old_file_path, new_file_path)
                print(f"Renamed {filename} to {new_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="从camera.txt文件中读取相机参数")
    parser.add_argument("--images_file_path", type=str, help="images.txt文件的路径")
    args = parser.parse_args()

# 指定文件夹路径
folder_path =  args.images_file_path # 请替换为实际的文件夹路径
pad_numbers_in_filenames(folder_path)