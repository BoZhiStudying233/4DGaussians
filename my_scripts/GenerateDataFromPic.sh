#!/bin/bash
#参考指令
#bash my_scripts/GenerateDataFromPic.sh /home/bo70s/Desktop/underwater_dataset/coral
#路径中的图片文件夹应为images，且不可有其他图片文件夹。
#注意images图片的格式是png还是jpg，要与py文件中的格式对应。


# 检查输入参数是否正确
if [ "$#" -ne 1 ]; then
    echo "用法: $0 <输入图像路径> "
    exit 1
fi



# 获取输入参数,base_path为images文件夹的上一层的路径,output_path为输出路径
INPUT_IMAGE_PATH=$1
FOLDER_NAME=$(basename $INPUT_IMAGE_PATH)
OUTPUT_PATH="data/my_data/$FOLDER_NAME"

# 创建输出目录
mkdir -p $OUTPUT_PATH


#格式化图片名称
python my_scripts/format_pic_name.py --images_file_path $INPUT_IMAGE_PATH/images

colmap feature_extractor \
   --database_path $OUTPUT_PATH/database.db \
   --image_path $INPUT_IMAGE_PATH \
   --ImageReader.camera_model SIMPLE_RADIAL \
   --ImageReader.single_camera 1 

colmap exhaustive_matcher \
   --database_path $OUTPUT_PATH/database.db

mkdir $OUTPUT_PATH/Sparse

colmap mapper \
    --database_path $OUTPUT_PATH/database.db \
    --image_path $INPUT_IMAGE_PATH \
    --output_path $OUTPUT_PATH/Sparse

mkdir $OUTPUT_PATH/dense

colmap image_undistorter \
    --image_path $INPUT_IMAGE_PATH \
    --input_path $OUTPUT_PATH/Sparse/0 \
    --output_path $OUTPUT_PATH/dense \
    --output_type COLMAP \
    --max_image_size 2000

colmap patch_match_stereo \
    --workspace_path $OUTPUT_PATH/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

colmap stereo_fusion \
    --workspace_path $OUTPUT_PATH/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $OUTPUT_PATH/dense/fused.ply

colmap poisson_mesher \
    --input_path $OUTPUT_PATH/dense/fused.ply \
    --output_path $OUTPUT_PATH/dense/meshed-poisson.ply

colmap delaunay_mesher \
    --input_path $OUTPUT_PATH/dense \
    --output_path $OUTPUT_PATH/dense/meshed-delaunay.ply


echo "密集重建完成。"

echo "重建结果保存在: $OUTPUT_PATH"

# echo "开始进行下采样"

# python scripts/downsample_point.py $OUTPUT_PATH/dense/fused.ply $OUTPUT_PATH/points3D_downsample2.ply

# echo "下采样完成。进行数据集格式转换"

# colmap model_converter --input_path $OUTPUT_PATH/Sparse/0/ --output_path $OUTPUT_PATH/Sparse/0/ --output_type TXT
# sed -i '1,4d' $OUTPUT_PATH/Sparse/0/images.txt
# sed -i '1,3d' $OUTPUT_PATH/Sparse/0/cameras.txt

# echo "数据集格式转换完成。开始提取相机内外参并生成json"

# python my_scripts/extract_camera_info.py --file_path $OUTPUT_PATH/Sparse/0/images.txt --output_file_path $OUTPUT_PATH/Sparse/0/new_images.txt
# echo "提取相机内外参并生成json完成。"
# python my_scripts/generate_json.py --cameras_file_path $OUTPUT_PATH/Sparse/0/cameras.txt --images_file_path $OUTPUT_PATH/Sparse/0/new_images.txt
# echo "生成json完成。"

# mkdir $OUTPUT_PATH/rgb
# mkdir $OUTPUT_PATH/rgb/2x
# cp $INPUT_IMAGE_PATH/images/* $OUTPUT_PATH/rgb/2x/

# python my_scripts/generate_dataset_json.py --input $OUTPUT_PATH/Sparse/0/new_images.txt --output $OUTPUT_PATH/dataset.json
# cp data/my_data/turtle/scene.json $OUTPUT_PATH/
# echo "数据集格式转换完成。开始训练"

# python train.py -s data/my_data/$FOLDER_NAME --port 6017 --expname my_data/$FOLDER_NAME --configs arguments/hypernerf/default.py

# echo "训练完成。开始渲染"

# python render.py --model_path output/my_data/$FOLDER_NAME --configs arguments/hypernerf/default.py
# 0echo "渲染完成。"
