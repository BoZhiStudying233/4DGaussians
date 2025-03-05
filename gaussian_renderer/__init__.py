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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine", cam_type=None):
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
        raster_settings = viewpoint_camera['camera']#viewpoint_camera是一个复杂的量，好像与scene这个类有关。里面包含了batch_size个camera，每个camera都有自己的属性，
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        


    R = torch.from_numpy(viewpoint_camera.R).float().cuda()    #旋转矩阵，从世界坐标系转到相机坐标系
    T = torch.from_numpy(viewpoint_camera.T).float().cuda()    #平移矩阵，从世界坐标系转到相机坐标系
    H = viewpoint_camera.image_height
    W = viewpoint_camera.image_width
    y = torch.linspace(0., H, H, device="cuda")
    x = torch.linspace(0., W, W, device="cuda")
    cy,cx = H/2, W/2
    yy, xx = torch.meshgrid(y, x)
    yy = (yy - cy) / viewpoint_camera.FoVy
    xx = (xx - cx) / viewpoint_camera.FoVx
    directions = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
    norms = torch.linalg.norm(directions, dim=-1, keepdim=True)
    directions = directions / norms
    directions = directions @ R.T       #也就是将方向向量转换到世界坐标系下
    #colmap出来的R是world2camera，因此需要将其转换为camera2world

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
    # medium_rgb = torch.zeros_like(medium_rgb)
    # medium_bs = torch.zeros_like(medium_bs)
    # medium_attn = torch.zeros_like(medium_attn)


    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

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


    viewmat = torch.zeros((4, 4), device=R.device)
    viewmat[:3, :3] = R  # 旋转部分
    viewmat[:3, 3] = T   # 平移部分
    viewmat[3, 3] = 1    # 齐次坐标系数

    xys, depths, conics = project_gaussians_manual(
        means3d=means3D_final,
        viewmat=viewmat,
        fx=viewpoint_camera.FoVx,
        fy=viewpoint_camera.FoVy,
        cx=cx,
        cy=cy,
        cov_flatten=pc.get_covariance(scaling_modifier),
        clip_thresh=0.01
        )


    rgb_medium = integrate_medium_contributions(
        xys=xys,
        depths=depths,
        opacities=opacity_final,
        conics= conics,
        medium_rgb=medium_rgb,
        medium_bs=medium_bs,
        medium_attn=medium_attn,
        H=H,
        W=W
    )


    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # time3 = get_time()
    rendered_image, radii, depth = rasterizer(#在PyTorch中，当一个nn.Module类的实例被当作函数调用时，实际上是在调用它的forward方法。
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

    rgb_medium_reshaped = rgb_medium.permute(2, 0, 1)

    return {"render_image": rendered_image,
            "rgb_medium": rgb_medium_reshaped,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth":depth}



def project_gaussians_manual(
    means3d: torch.Tensor,   # [N, 3] 高斯的3D坐标（世界坐标系）
    viewmat: torch.Tensor,   # [4, 4] 视图矩阵（世界到相机坐标系的变换）
    fx: float,               # 焦距x
    fy: float,               # 焦距y
    cx: float,               # 主点x
    cy: float,               # 主点y
    
    cov_flatten,#协方差矩阵[N,6]
    clip_thresh: float = 0.01

) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算3D高斯到屏幕空间的投影坐标和深度
    返回:
        xys: Tensor [M, 2] 有效的屏幕坐标 (u, v)
        depths: Tensor [M, 1] 对应的深度值（相机空间z坐标）
    """
    # 转换为齐次坐标
    ones = torch.ones(means3d.shape[0], 1, device=means3d.device)
    points_homo = torch.cat([means3d, ones], dim=1)  # [N, 4]
    
    # 应用视图变换（世界坐标->相机坐标）
    camera_coords = (viewmat @ points_homo.T).T  # [N, 4]
    
    # 提取相机坐标系下的XYZ
    camera_xyz = camera_coords[:, :3]  # [N, 3]
    z = camera_xyz[:, 2:3]  # [N, 1]
    
    # 剔除近裁剪面后的点
    valid_mask = (z > clip_thresh).squeeze()  # [N]
    camera_xyz = camera_xyz[valid_mask]
    z = z[valid_mask]
    
    # 透视投影
    x_proj = camera_xyz[:, 0] / z.squeeze() * fx + cx  # [M]
    y_proj = camera_xyz[:, 1] / z.squeeze() * fy + cy  # [M]
    
    # 组装屏幕坐标
    xys = torch.stack([x_proj, y_proj], dim=1)  # [M, 2]
    

    R = viewmat[:3, :3]  # 取前三行前三列
    T = viewmat[:3, 3]   # 取前三行第四列

    K = build_intrinsic_matrix(fx, fy, cx, cy)
    conics = compute_all_conics(cov_flatten, means3d, R, T,K)



    return xys, z, conics

# 使用示例
# 假设你已经有以下参数：
# means3d: [N,3] 高斯的世界坐标
# viewmat: [4,4] 从世界到相机坐标的变换矩阵
# fx, fy, cx, cy: 相机内参



def build_intrinsic_matrix(fx: float, fy: float, cx: float, cy: float) -> torch.Tensor:
    """
    构建相机内参矩阵
    Args:
        fx: x方向焦距（像素单位）
        fy: y方向焦距（像素单位）
        cx: 主点x坐标（像素单位）
        cy: 主点y坐标（像素单位）
    Returns:
        K: 相机内参矩阵 [3x3]
    """
    K = torch.tensor([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ], dtype=torch.float32)
    return K


def integrate_medium_contributions(
    xys: torch.Tensor,       # [N, 2] 高斯中心屏幕坐标
    depths: torch.Tensor,    # [N, 1]
    opacities: torch.Tensor, # [N, 1] 原始不透明度o_i
    conics: torch.Tensor,    # [N, 3] 协方差逆矩阵的上三角元素 (a, b, c)
    medium_rgb: torch.Tensor,  # [H, W, 3]
    medium_bs: torch.Tensor,    # [H, W, 3]
    medium_attn: torch.Tensor, # [H, W, 3]
    H: int, W: int
) -> torch.Tensor:
    device = xys.device
    # 初始化输出（每个像素独立记录）
    rgb_medium = torch.zeros(H, W, 3, device=device)
    T_obj = torch.ones(H, W, 3, device=device)
    prev_depth = torch.zeros(H, W, 1, device=device)  # 每个像素独立记录前向深度
     
    # 同步排序所有参数
    sorted_indices = torch.argsort(depths.squeeze(), descending=True)
    xys = xys[sorted_indices]
    depths = depths[sorted_indices]
    opacities = opacities[sorted_indices]
    conics = conics[sorted_indices]
    
    # 批量处理所有高斯
    for i in range(len(depths)):
        # 提取当前高斯参数
        a, b, c = conics[i]          # Σ^{-1} = [[a, b], [b, c]]
        x_center, y_center = xys[i]  # 高斯的中心坐标
        
        # 步骤1: 计算椭圆参数
        trace = a + c
        # det = a * c - b ​**​ 2
        sqrt_term = torch.sqrt((a - c)**2 + 4 * b**2)
        eigen1 = (trace + sqrt_term) / 2
        eigen2 = (trace - sqrt_term) / 2
        major_axis = 3 * (1 / eigen2).sqrt()  # 3σ原则覆盖99.7%能量
        minor_axis = 3 * (1 / eigen1).sqrt()
        angle = 0.5 * torch.atan2(2 * b, a - c)
        
        # 步骤2: 生成椭圆覆盖的像素网格
        u_min = torch.clamp((x_center - major_axis).long(), 0, W-1)
        u_max = torch.clamp((x_center + major_axis).long() + 1, 0, W-1)
        v_min = torch.clamp((y_center - minor_axis).long(), 0, H-1)
        v_max = torch.clamp((y_center + minor_axis).long() + 1, 0, H-1)
        
        # 生成网格坐标
        u = torch.arange(u_min, u_max, device=device)
        v = torch.arange(v_min, v_max, device=device)
        grid_u, grid_v = torch.meshgrid(u, v, indexing='xy')  # [U, V]
        
        # 步骤3: 计算所有网格点的高斯权重
        dx = grid_u.float() + 0.5 - x_center  # [U, V]
        dy = grid_v.float() + 0.5 - y_center
        
        # 应用椭圆旋转（坐标变换）
        cos_theta = torch.cos(angle)
        sin_theta = torch.sin(angle)
        rot_dx = dx * cos_theta - dy * sin_theta
        rot_dy = dx * sin_theta + dy * cos_theta
        
        # 计算二次型指数项
        power = -0.5 * (a * rot_dx**2 + 2 * b * rot_dx * rot_dy + c * rot_dy**2)
        G = torch.exp(power.clamp(max=50, min=-50))  # [U, V]
        

        # 步骤4: 计算alpha并过滤低贡献像素
        alpha_i = torch.sigmoid(opacities[i]) * G
        mask = alpha_i > 1e-4  # 过滤贡献过小的像素
        grid_u = grid_u[mask]
        grid_v = grid_v[mask]
        alpha_i = alpha_i[mask]
        
        # 步骤5: 介质积分（向量化操作）
        if len(grid_u) > 0:
            σ_attn = medium_attn[grid_v, grid_u]  # [K, 3]
            σ_bs = medium_bs[grid_v, grid_u]      # [K, 3]
            c_med = medium_rgb[grid_v, grid_u]    # [K, 3]
            s_i = depths[i]                       # [1]
            
            # 计算介质贡献
            prev_depth_pixel = prev_depth[grid_v, grid_u]  # [K, 1]
            exp_prev = torch.exp(-σ_bs * prev_depth_pixel)
            exp_curr = torch.exp(-σ_bs * s_i)
            delta_exp = exp_prev - exp_curr
            rgb_medium[grid_v, grid_u] += T_obj[grid_v, grid_u] * c_med * delta_exp

            T_obj[grid_v, grid_u] *= (1 - alpha_i.unsqueeze(-1))
            
            # 更新深度记录
            prev_depth[grid_v, grid_u] = s_i
    

    rgb_medium += T_obj*medium_rgb*torch.exp(-medium_bs*prev_depth)


    
    return rgb_medium







def flatten_to_cov3d(σ_flatten: torch.Tensor) -> torch.Tensor:
    """
    将 [N,6] 的扁平化协方差参数转换为 [N,3,3] 对称矩阵
    Input: σ_flatten [N,6]
    Output: Σ_3d [N,3,3]
    """
    N = σ_flatten.shape[0]
    Σ_3d = torch.zeros(N, 3, 3, device=σ_flatten.device)
    Σ_3d[:, 0, 0] = σ_flatten[:, 0]  # σ_xx
    Σ_3d[:, 0, 1] = σ_flatten[:, 1]  # σ_xy
    Σ_3d[:, 0, 2] = σ_flatten[:, 2]  # σ_xz
    Σ_3d[:, 1, 1] = σ_flatten[:, 3]  # σ_yy
    Σ_3d[:, 1, 2] = σ_flatten[:, 4]  # σ_yz
    Σ_3d[:, 2, 2] = σ_flatten[:, 5]  # σ_zz
    Σ_3d = Σ_3d + Σ_3d.transpose(1,2) - torch.diag_embed(torch.diagonal(Σ_3d, dim1=1, dim2=2))
    return Σ_3d




def transform_to_camera(Σ_3d: torch.Tensor, R: torch.Tensor, t: torch.Tensor, centers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Input:
        Σ_3d: [N,3,3] 世界坐标系下的协方差矩阵
        R: [3,3] 相机旋转矩阵
        t: [3] 相机平移向量
        centers: [N,3] 高斯中心的世界坐标
    Output:
        Σ_camera: [N,3,3] 相机坐标系下的协方差矩阵
        μ_camera: [N,3] 相机坐标系下的高斯中心坐标
    """
    μ_camera = (centers @ R.T) + t.unsqueeze(0)  # [N,3]
    Σ_camera = R @ Σ_3d @ R.T                   # [N,3,3]
    return Σ_camera, μ_camera



def compute_jacobian(μ_camera: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Input:
        μ_camera: [N,3] 相机坐标系下的高斯中心坐标 (X, Y, Z)
        K: [3,3] 相机内参矩阵 [[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]]
    Output:
        J: [N,2,3] 雅可比矩阵
    """
    f_x = K[0,0]
    f_y = K[1,1]
    X = μ_camera[:, 0]
    Y = μ_camera[:, 1]
    Z = μ_camera[:, 2].clamp(min=1e-3)  # 避免除零
    
    J = torch.zeros(μ_camera.shape[0], 2, 3, device=μ_camera.device)
    J[:, 0, 0] = f_x / Z    # ∂u/∂X
    J[:, 0, 2] = -f_x * X / Z**2  # ∂u/∂Z
    J[:, 1, 1] = f_y / Z    # ∂v/∂Y
    J[:, 1, 2] = -f_y * Y / Z**2  # ∂v/∂Z
    return J




def project_to_2d(Σ_camera: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """
    Input:
        Σ_camera: [N,3,3] 相机坐标系下的协方差矩阵
        J: [N,2,3] 雅可比矩阵
    Output:
        Σ_2d: [N,2,2] 图像平面上的 2D 协方差矩阵
    """
    Σ_2d = torch.einsum("nij,njk->nik", J, Σ_camera)  # J @ Σ_camera
    Σ_2d = torch.einsum("nij,njk->nik", Σ_2d, J.transpose(1,2))  # (J @ Σ_camera) @ J.T
    return Σ_2d


def compute_conics(Σ_2d: torch.Tensor) -> torch.Tensor:
    """
    Input:
        Σ_2d: [N,2,2] 2D 协方差矩阵
    Output:
        conics: [N,3] 二维逆协方差矩阵的上三角元素 [a, b, c]
    """
    # 添加正则化项避免奇异矩阵
    Σ_2d_reg = Σ_2d + 1e-6 * torch.eye(2, device=Σ_2d.device).unsqueeze(0)
    Σ_2d_inv = torch.linalg.inv(Σ_2d_reg)
    
    # 提取上三角元素
    a = Σ_2d_inv[:, 0, 0]
    b = Σ_2d_inv[:, 0, 1]
    c = Σ_2d_inv[:, 1, 1]
    return torch.stack([a, b, c], dim=-1)
def compute_all_conics(
    cov_flatten: torch.Tensor,  # [N,6] 扁平化的协方差参数
    centers: torch.Tensor,    # [N,3] 高斯中心的世界坐标
    R: torch.Tensor,         # [3,3] 相机旋转矩阵
    t: torch.Tensor,         # [3] 相机平移向量
    K: torch.Tensor          # [3,3] 相机内参矩阵
) -> torch.Tensor:
    # Step 1: 组装 3D 协方差矩阵
    Σ_3d = flatten_to_cov3d(cov_flatten)  # [N,3,3]
    
    # Step 2: 转换到相机坐标系
    Σ_camera, μ_camera = transform_to_camera(Σ_3d, R, t, centers)  # [N,3,3], [N,3]
    
    # Step 3: 计算雅可比矩阵
    J = compute_jacobian(μ_camera, K)  # [N,2,3]
    
    # Step 4: 投影到 2D 图像平面
    Σ_2d = project_to_2d(Σ_camera, J)  # [N,2,2]
    
    # Step 5: 计算 conics 参数
    conics = compute_conics(Σ_2d)      # [N,3]
    return conics