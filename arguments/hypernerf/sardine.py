_base_="default.py"
ModelParams=dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 150]
    },
)
OptimizationParams=dict(
    position_lr_init = 0.00016,#0.00016
    position_lr_final = 5e-5,
    position_lr_delay_mult = 0.01,#0.01
    position_lr_max_steps = 20_000,#20_000
    deformation_lr_init = 0.00016,#0.00016
    deformation_lr_final = 0.000016,#0.000016
    deformation_lr_delay_mult = 0.01,
    grid_lr_init = 0.0016,#0.0016
    grid_lr_final = 0.00016,#0.00016

    medium_MLP_lr_init = 1e-3,
    medium_MLP_lr_final = 1.5e-4,
    medium_lr_delay_mult = 0.01,#0.01


    feature_lr_init = 0.0025,#0.0025
    feature_lr_final = 0.0025,#0.00025
    feature_lr_delay_mult = 0.0,

    opacity_lr_init = 0.05,#0.05
    opacity_lr_final = 0.05,#0.005
    opacity_lr_delay_mult = 0.0,

    scaling_lr_init = 0.005,#0.005
    scaling_lr_final = 0.005,
    scaling_lr_delay_mult = 0.0,

    rotation_lr_init = 0.001,#0.001
    rotation_lr_final = 0.001,
    rotation_lr_delay_mult = 0.0,

    percent_dense = 0.01,
    lambda_dssim = 0,
    lambda_lpips = 0,
    weight_constraint_init= 1,
    weight_constraint_after = 0.2,
    weight_decay_iteration = 5000,
    opacity_reset_interval = 30_0000,
    densification_interval = 100,
    densify_from_iter = 500,#500
    densify_until_iter = 10_000,
    densify_grad_threshold_coarse = 0.00001,#0.0002
    densify_grad_threshold_fine_init = 0.00001,
    densify_grad_threshold_after = 0.0002,#0.0002
    pruning_from_iter = 500,#500
    pruning_interval = 100,
    opacity_threshold_coarse = 0.0005,#0.005
    opacity_threshold_fine_init = 0.0005,#0.005
    opacity_threshold_fine_after = 0.0005,#0.005
    batch_size=2,
    add_point=False,
    coarse_iterations = 8000,

    attn_scale = 50,#介质输出的削弱系数
    uncertainty_weight = 0.0,
    prune_threshold = 40000,
    grow_threshold = 180_000,
    coarse_densify_threshold = 180_000,
    fine_densify_threshold = 180_000,

    #梯度裁剪
    max_norms = {
            "xyz": 5.0,           # 位置参数较大梯度 5
            "deformation":1,   # 形变MLP参数较小梯度1
            "grid": 2,          # 网格参数中等梯度2
            "f_dc": 0.5,          # 颜色参数严格限制0.5
            "f_rest": 0.5,          #0.5
            "opacity": 0.1,       # 透明度参数需谨慎更新0.1
            "scaling": 1.0,         #1
            "rotation": 1.0,        #1
            "medium_mlp": 0.001     # MLP参数单独控制0.001
        },
)