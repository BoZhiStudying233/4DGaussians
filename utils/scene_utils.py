import torch
import os
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']

import numpy as np
import wandb


last_iter = -501

import copy
@torch.no_grad()


def render_training_image(scene, gaussians, viewpoints, render_func, pipe, background, stage, iteration, time_now, dataset_type):
    def render(gaussians, viewpoint, path, scaling, cam_type):
        global last_iter
        
        # scaling_copy = gaussians._scaling
        render_pkg = render_func(viewpoint, gaussians, pipe, background, stage=stage, cam_type=cam_type)
        label1 = f"stage:{stage},iter:{iteration}"
        times =  time_now/60
        if times < 1:
            end = "min"
        else:
            end = "mins"
        label2 = "time:%.2f" % times + end
        image = render_pkg["render"] + render_pkg["rgb_medium"]
        depth = render_pkg["depth"]
        if dataset_type == "PanopticSports":
            gt_np = viewpoint['image'].permute(1,2,0).cpu().numpy()
        else:
            gt_np = viewpoint.original_image.permute(1,2,0).cpu().numpy()
        image_np = image.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)

        render_np = render_pkg["render"].permute(1,2,0).cpu().numpy()
        rgb_medium_np = render_pkg["rgb_medium"].permute(1,2,0).cpu().numpy()


        depth_np = depth.permute(1, 2, 0).cpu().numpy()
        depth_np /= depth_np.max()
        depth_np = np.repeat(depth_np, 3, axis=2)
        image_np = np.concatenate((gt_np, image_np, depth_np, render_np, rgb_medium_np), axis=1)
        image_with_labels = Image.fromarray((np.clip(image_np,0,1) * 255).astype('uint8'))  
        draw1 = ImageDraw.Draw(image_with_labels)
        font = ImageFont.truetype('./utils/TIMES.TTF', size=40) 
        text_color = (255, 0, 0)  
        label1_position = (10, 10)
        label2_position = (image_with_labels.width - 100 - len(label2) * 10, 10) 
        draw1.text(label1_position, label1, fill=text_color, font=font)
        draw1.text(label2_position, label2, fill=text_color, font=font)
        
        image_with_labels.save(path)
        
        if wandb.run is not None:
            if wandb and (abs(iteration - last_iter) > 500):  # 每500次迭代记录一次
                if "train" in stage:
                    last_iter = iteration#因为渲染图片是先渲染test的，再渲染train，所以要渲染train的时候再更新last_iter
                print("save image to wandb")
                wandb.log({
                    f"render/{stage}_render": wandb.Image(image_with_labels),
                    "iteration": iteration
                })
    render_base_path = os.path.join(scene.model_path, f"{stage}_render")
    point_cloud_path = os.path.join(render_base_path,"pointclouds")
    image_path = os.path.join(render_base_path,"images")
    if not os.path.exists(os.path.join(scene.model_path, f"{stage}_render")):
        os.makedirs(render_base_path)
    if not os.path.exists(point_cloud_path):
        os.makedirs(point_cloud_path)
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    
    for idx in range(len(viewpoints)):
        image_save_path = os.path.join(image_path,f"{iteration}_{idx}.jpg")
        render(gaussians,viewpoints[idx],image_save_path,scaling = 1,cam_type=dataset_type)
    pc_mask = gaussians.get_opacity
    pc_mask = pc_mask > 0.1

def visualize_and_save_point_cloud(point_cloud, R, T, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    R = R.T
    T = -R.dot(T)
    transformed_point_cloud = np.dot(R, point_cloud) + T.reshape(-1, 1)
    ax.scatter(transformed_point_cloud[0], transformed_point_cloud[1], transformed_point_cloud[2], c='g', marker='o')
    ax.axis("off")
    plt.savefig(filename)

