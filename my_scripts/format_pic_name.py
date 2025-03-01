import os
import re
import argparse

def rename_images(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    
    for file_name in files:
        # 检查文件是否为图片（可以根据需要扩展支持的格式）
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            # 使用正则表达式提取数字部分
            numbers = re.findall(r'\d+', file_name)
            if numbers:
                # 将提取的数字部分拼接成新的文件名
                new_name = ''.join(numbers) + os.path.splitext(file_name)[1]
                # 重命名文件
                os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_name))
                print(f'Renamed: {file_name} -> {new_name}')
            else:
                print(f'No numbers found in: {file_name}')
        else:
            print(f'Skipping non-image file: {file_name}')



def rename_images_from_zero(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    
    # 过滤出图片文件（可以根据需要扩展支持的格式）
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]
    


    def extract_number(file_name):
        # 提取文件名中的数字部分
        numbers = re.findall(r'\d+', file_name)
        return int(numbers[0]) if numbers else -1  # 如果没有数字，返回 -1
    # 按文件名排序（如果需要按特定顺序重命名）
    image_files.sort(key=extract_number)
    
    # 从 0 开始重命名
    for index, file_name in enumerate(image_files):
        # 获取文件扩展名
        file_extension = os.path.splitext(file_name)[1]
        # 生成新的文件名
        new_name = f"{index}{file_extension}"
        # 重命名文件
        os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_name))
        print(f'Renamed: {file_name} -> {new_name}')



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

# rename_images(folder_path)
rename_images_from_zero(folder_path)
pad_numbers_in_filenames(folder_path)

