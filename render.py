#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams

import matplotlib.pyplot as plt

from gaussian_renderer import GaussianModel
from time import time
import threading
import concurrent.futures
def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, cam_type):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")  # 新增深度图保存路径
    cos_angles_path = os.path.join(model_path, name, "ours_{}".format(iteration), "cos_angles")  # 新增 cos_angles 保存路径
    normals_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normals")
    light_uncertainty_path = os.path.join(model_path, name, "ours_{}".format(iteration), "light_uncertainty")  # 新增光照不确定性保存路径
    medium_image_path = os.path.join(model_path, name, "ours_{}".format(iteration), "medium_image")  # 新增介质输出保存路径

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)  # 创建深度图保存目录
    makedirs(cos_angles_path, exist_ok=True)  # 创建 cos_angles 保存目录
    makedirs(normals_path, exist_ok=True)
    makedirs(light_uncertainty_path, exist_ok=True)  # 创建光照不确定性保存目录
    makedirs(medium_image_path, exist_ok=True)  # 创建介质输出保存目录

    render_images = []
    medium_images = []
    medium_list = []
    gt_list = []
    render_list = []
    print("point nums:",gaussians._xyz.shape[0])
    # print("views:",len(views))
    # print("views:",views)
    all_time = []
    for idx, view in enumerate(tqdm(views, desc="Process Rendering")):
        if idx == 0:time1 = time()
        time_1 = time()
        # print("idx:",idx,"views:",len(views))
        result = render(view, gaussians, pipeline, background,cam_type=cam_type, opt=None)#需要读入这个opt里面的参数来表示对介质的输出的削减，之后加吧
        if result["cos_angles"] is not None:
            depth_image = result["depth_image"]#黑白
        # print("depth_image.shape:",depth_image.shape)
        
        
            cos_angles = result["cos_angles"]
            # 归一化 cos_angles 到 [0, 1] 范围
            cos_angles_normalized = (cos_angles - cos_angles.min()) / (cos_angles.max() - cos_angles.min())
            # 将张量转换为 numpy 数组并转换为 uint8 类型
            cos_angles_np = (cos_angles_normalized.cpu().numpy() * 255).astype(np.uint8)
            # 保存 cos_angles 图像


            cos_angles_image_path = os.path.join(cos_angles_path, '{0:05d}'.format(idx) + ".png")
            cv2.imwrite(cos_angles_image_path, cos_angles_np)

       
            normals = result["normals"]
            # 归一化法线图像到 [0, 1] 范围
            
            normals_normalized = (normals - normals.min()) / (normals.max() - normals.min())
            # 将张量转换为 numpy 数组并转换为 uint8 类型

            normals_np = (normals_normalized.cpu().numpy() * 255).astype(np.uint8)
            # 保存法线图
            normals_image_path = os.path.join(normals_path, '{0:05d}'.format(idx) + ".png")
            cv2.imwrite(normals_image_path, normals_np)




            # 归一化深度图并转换为 uint8 类型
            depth_image_np = depth_image.cpu().numpy()
            depth_image_normalized = (depth_image_np - depth_image_np.min()) / (depth_image_np.max() - depth_image_np.min()) * 255
            depth_image_uint8 = depth_image_normalized.astype(np.uint8)
            # 保存深度图
            depth_image_path = os.path.join(depth_path, '{0:05d}'.format(idx) + ".png")
            cv2.imwrite(depth_image_path, depth_image_uint8)

            light_uncertainty = result["light_uncertainty"]
            # 归一化光照不确定性图像到 [0, 1] 范围
            light_uncertainty_normalized = (light_uncertainty - light_uncertainty.min()) / (light_uncertainty.max() - light_uncertainty.min())
            # 将张量转换为 numpy 数组并转换为 uint8 类型
            light_uncertainty_np = (light_uncertainty_normalized.cpu().numpy()* 255).astype(np.uint8)

            # colored_image = (colored_image * 255).astype(np.uint8)




            # 保存光照不确定性图像
            light_uncertainty_image_path = os.path.join(light_uncertainty_path, '{0:05d}'.format(idx) + "_light_uncertainty.png")
            cv2.imwrite(light_uncertainty_image_path, light_uncertainty_np)#黑白


        
        rendering = result["render_image"] + result["rgb_medium"]
        rendering = result["render_image"]
        medium_images.append(to8b(result["rgb_medium"]).transpose(1,2,0))
        medium_list.append(result["rgb_medium"])

        render_images.append(to8b(rendering).transpose(1,2,0))
        render_list.append(rendering)
        time_2 = time()
        if name in ["train", "test"]:
            if cam_type != "PanopticSports":
                gt = view.original_image[0:3, :, :]
            else:
                gt  = view['image'].cuda()
            gt_list.append(gt)
        # print("循环+1")

    all_time.append(time_2-time_1)
    if all_time:
        print("average FPS:",1/(sum(all_time)/len(all_time)))
        # assert False
    time2=time()
    # print("FPS:",(len(views)-1)/(time2-time1))#此处的帧数代表每秒能渲染出几张图片
    multithread_write(medium_list, medium_image_path)

    multithread_write(gt_list, gts_path)

    multithread_write(render_list, render_path)
    Fps_num = len(render_list)/3
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_medium.mp4'), medium_images, fps=Fps_num)#每秒显示的图像帧数为 30 帧

    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_images, fps=Fps_num)#每秒显示的图像帧数为 30 帧
    print("Done rendering")

def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool):
    with torch.no_grad():
        # print("hyperparam:", hyperparam)
        # print("hyperparam:")
        # hyperparam_dict = vars(hyperparam)
        # for key, value in hyperparam_dict.items():
        #     print(f"{key}: {value}")
        gaussians = GaussianModel(dataset.sh_degree, hyperparam, dataset.model_path)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cam_type=scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,cam_type)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,cam_type)
        if not skip_video:
            render_set(dataset.model_path,"video", scene.loaded_iter, scene.getVideoCameras(),gaussians, pipeline, background,cam_type)
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser)#我找了好久source_path是怎么被程序读入的，原来在这里，它从model_path的cfg文件里读到的。
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video)
