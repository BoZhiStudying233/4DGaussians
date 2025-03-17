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
from typing import Tuple
import matplotlib.pyplot as plt
import torch

from gaussian_renderer.project_gaussians import project_gaussians
from gaussian_renderer.sh import spherical_harmonics
from gaussian_renderer.rasterize import rasterize_gaussians

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

import math

from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, iterations = None, scaling_modifier = 1.0, override_color = None, stage="fine", cam_type=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0  #dtype表示数据类型    +0是为了触发张量的重新计算，确保张量位于正确的设备（GPU）上
    #此变量的目的是为后续的高斯点渲染做准备，确保所有的高斯点在屏幕空间中的位置都能被正确地计算和跟踪其梯度变化。
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    means3D = pc.get_xyz#猜测pc是point cloud的意思
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)#重复时间张量。时间张量会被扩展成一个形状为 (Gaussian点的数量, 1) 的矩阵，这样每个 Gaussian 点都拥有相应的时间信息
    else:
        # raster_settings = viewpoint_camera['camera']#viewpoint_camera是一个复杂的量，好像与scene这个类有关。里面包含了batch_size个camera，每个camera都有自己的属性，
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        
    #好好检查R、T和watersplatting的对应!!!

    R = torch.from_numpy(viewpoint_camera.R).float().cuda()    #旋转矩阵，从世界坐标系转到相机坐标系
    T = torch.from_numpy(viewpoint_camera.T).float().cuda()    #平移矩阵，从世界坐标系转到相机坐标系

    # S = torch.diag(torch.tensor([1, -1, -1], device=R.device, dtype=R.dtype))  # 3x3 缩放矩阵    
    # R = S @ R
    # T = S @ T
    # R = R.T
    print("R:",R)   
    print("viewmatrix:",viewpoint_camera.world_view_transform)
    H = viewpoint_camera.image_height
    W = viewpoint_camera.image_width
    y = torch.linspace(0., H, H, device="cuda")
    x = torch.linspace(0., W, W, device="cuda")
    cy,cx = H/2, W/2
    yy, xx = torch.meshgrid(y, x)
    yy = (yy - cy) / viewpoint_camera.FoVy
    xx = (xx - cx) / viewpoint_camera.FoVx
    directions = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)#(x,y,1)
    norms = torch.linalg.norm(directions, dim=-1, keepdim=True)
    directions = directions / norms
    directions = directions @ R.T       #也就是将方向向量转换到世界坐标系下
    #colmap出来的R是world2camera，因此需要将其转换为camera2world。但我认为此处应当右乘。

    directions_flat = directions.view(-1, 3)
    directions_encoded = pc.direction_encoding(directions_flat)
    outputs_shape = directions.shape[:-1]

    medium_base_out = pc.medium_mlp(directions_encoded)
 
    # different activations for different outputs
    medium_rgb = (
        pc.colour_activation(medium_base_out[..., :3])
        .view(*outputs_shape, -1)
        .to(directions)
    )
    medium_bs = (
        pc.sigma_activation(medium_base_out[..., 3:6] + pc.medium_density_bias)
        .view(*outputs_shape, -1)
        .to(directions)
    )
    medium_attn = (
        pc.sigma_activation(medium_base_out[..., 6:] + pc.medium_density_bias)
        .view(*outputs_shape, -1)
        .to(directions)
    )

    # #当渲染清澈介质（无介质）时候，用此三句话
    medium_rgb = torch.zeros_like(medium_rgb)
    medium_bs = torch.zeros_like(medium_bs)
    medium_attn = torch.zeros_like(medium_attn)
    
   
        
    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation

    
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table
    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
    elif "fine" in stage:
        # time0 = get_time()
        # means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point], 
        #                                                                  rotations[deformation_point], opacity[deformation_point],
        #                                                                  time[deformation_point])
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, 
                                                                 rotations, opacity, shs,
                                                                 time)
    else:
        raise NotImplementedError




    

    # time2 = get_time()
    # print("asset value:",time2-time1)
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)
    # print(opacity.max())
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
            # shs = 
    else:
        colors_precomp = override_color    


    viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
    viewmat[:3, :3] = R  # 旋转部分
    viewmat[:3, 3] = T   # 平移部分

    BLOCK_WIDTH = 16

    print("viewmat:"    ,viewmat)

    # print("begin")
    # quats_crop = rotation_matrix_to_quaternion(R)
    xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
            means3D_final,
            torch.exp(scales_final),
            1,
            rotations_final,
            viewmat.squeeze()[:3, :],
            viewpoint_camera.FoVx,
            viewpoint_camera.FoVy,
            cx,
            cy,
            H,
            W,
            BLOCK_WIDTH,
            clip_thresh= 0.01,
        )  # type: ignore

    # print("end")


    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    xys_grad_abs = torch.zeros_like(xys)
    # print("shs_final.shape:",shs_final.shape)

    colors_crop = shs_final
 
    # print("colors_crop.shape:",colors_crop.shape)

    viewdirs = means3D_final.detach() - viewmat.detach()[..., :3, 3]  # (N, 3) 从相机坐标系指向高斯的向量
    viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
    n = pc.active_sh_degree
    rgbs = spherical_harmonics(n, viewdirs, colors_crop)
    rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore

    if xys.requires_grad !=False:
        xys.retain_grad()

    rendered_image, rgb_clear, rgb_medium, depth_im, alpha = rasterize_gaussians(  # type: ignore
        xys,
        xys_grad_abs,
        depths,
        radii,
        conics,
        num_tiles_hit,  # type: ignore
        rgbs,
        opacity,
        medium_rgb,
        medium_bs,
        medium_attn,
        H,
        W,
        BLOCK_WIDTH,
        background=medium_rgb,
        return_alpha=True,
        step = 0,#这个量在后面没有用到，所以随便给了值
    )  # type: ignore


     
    

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # time3 = get_time()
    rendered_image4D, _, depth = rasterizer(#在PyTorch中，当一个nn.Module类的实例被当作函数调用时，实际上是在调用它的forward方法。
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)
    # time4 = get_time()
    # print("rasterization:",time4-time3)
    # breakpoint()
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.


    image_vis1 = rendered_image.detach().cpu().numpy()
    image_vis2 = rendered_image4D.detach().cpu().permute(1, 2, 0).numpy()
    image_vis3 = rgb_medium.detach().cpu().numpy()
    plt.figure(figsize=(15, 5))

    # 创建一个1行3列的子图布局
    plt.subplot(1, 3, 1)
    plt.imshow(image_vis1)
    plt.axis('off')  # 关闭坐标轴

    plt.subplot(1, 3, 2)
    plt.imshow(image_vis2)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(image_vis3)
    plt.axis('off')

    plt.show()


    rendered_image = rendered_image.permute(2,0,1)
    rgb_medium = rgb_medium.permute(2,0,1)
    return {"render_image": rendered_image,
            "rgb_medium": rgb_medium,
            "viewspace_points": xys,
            "visibility_filter" : radii > 0,#visibility_filter用于过滤掉被视锥体裁剪掉的高斯点。看不见的点就不会参与后续的梯度更新。
            "radii": radii,#radii用于进行高斯密度的更新。
            "depth":depth_im}



