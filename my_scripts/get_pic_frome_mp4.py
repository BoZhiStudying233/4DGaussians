import cv2
import os
import numpy as np

def extract_frames(video_path, output_dir, num_frames=280):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频总帧数: {total_frames}")
    # 计算采样间隔
    interval = total_frames / num_frames
    
    print(f"每隔 {interval:.2f} 帧保存一次")
    # 当前帧计数
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 每隔interval帧保存一次
        if frame_count % int(interval) == 0 and saved_count < num_frames:
            # 保存图片,按照时间顺序命名
            frame_name = f"frame_{saved_count:04d}.png"
            save_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(save_path, frame)
            saved_count += 1
            print(f"保存 {frame_name} 到 {save_path}")
        frame_count += 1
    
    cap.release()
    print(f"成功提取 {saved_count} 帧图像到 {output_dir}")

if __name__ == "__main__":
    
    video_path = "/home/dzb/4DGaussians_old/data/DRUVA_mp4/DRUVA_MP4/A13.mp4"  # 替换为你的视频路径
    
    output_dir = "/data3/dzb/colmap_data/source/DRUVA/"     # 替换为你想要保存图片的文件夹路径
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_dir, video_name)

    extract_frames(video_path, output_dir)

    video_path = "/home/dzb/4DGaussians_old/data/DRUVA_mp4/DRUVA_MP4/A12.mp4"  # 替换为你的视频路径
    
    output_dir = "/data3/dzb/colmap_data/source/DRUVA/"     # 替换为你想要保存图片的文件夹路径
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_dir, video_name)

    extract_frames(video_path, output_dir)
    